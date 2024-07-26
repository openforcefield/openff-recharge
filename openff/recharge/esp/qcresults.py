"""Reconstruct ESP and electric field data from existing QC records"""

import json
import logging
import re
from typing import TYPE_CHECKING

import numpy
from openff.units import unit, Quantity
from openff.utilities import requires_package
from openff.recharge._pydantic import ValidationError

from openff.recharge.esp import ESPSettings, PCMSettings
from openff.recharge.esp.storage import MoleculeESPRecord
from openff.recharge.grids import GridGenerator, GridSettingsType
from openff.recharge.utilities.exceptions import RechargeException
from openff.recharge.utilities.molecule import extract_conformers

if TYPE_CHECKING:
    import qcelemental.models
    import qcelemental.models.results
    import qcportal.models

_logger = logging.getLogger(__name__)


class MissingQCWaveFunctionError(RechargeException):
    """An exception raised when a result does not store the required information about
    a computed QM wavefunction."""

    def __init__(self, result_id: str):
        super().__init__(
            f"The result with id={result_id} does not store the required wavefunction."
            f"Make sure to use at minimum the 'orbitals_and_eigenvalues' wavefunction "
            f"protocol when computing the data set."
        )
        self.result_id = result_id


class InvalidPCMKeywordError(RechargeException):
    """An exception raised when the PCM settings found in the 'pcm__input' entry of
    an entries keywords cannot be safely parsed."""

    def __init__(self, input_string: str):
        super().__init__(f"The PCM settings could not be safely parsed: {input_string}")


def _parse_pcm_input(input_string: str) -> PCMSettings:
    """Attempts to parse a set of PCM settings from a PSI4 keyword string."""

    # Convert the string to a JSON like string.
    value = input_string.replace(" ", "").replace("=", ":").replace("{", ":{")
    value = re.sub(r"(\d*[a-z][a-z\d]*)", r'"\1"', value)
    value = re.sub(r'(["\d}])"', r'\1,"', value.replace("\n", ""))
    value = value.replace('"true"', "true")
    value = value.replace('"false"', "false")

    solvent_map = {"H2O": "Water"}
    radii_map = {"BONDI": "Bondi", "UFF": "UFF", "ALLINGER": "Allinger"}

    try:
        # Load the string into a dictionary.
        pcm_dict = json.loads(f"{{{value}}}")

        # Validate some of the settings which we do not store in the settings
        # object yet.
        assert pcm_dict["cavity"]["type"].upper() == "GEPOL"
        assert pcm_dict["cavity"]["mode"].upper() == "IMPLICIT"
        assert numpy.isclose(pcm_dict["cavity"]["minradius"], 52.917721067)
        assert pcm_dict["units"].upper() == "ANGSTROM"
        assert pcm_dict["codata"] == 2010
        assert pcm_dict["medium"]["nonequilibrium"] is False
        assert pcm_dict["medium"]["matrixsymm"] is True
        assert numpy.isclose(pcm_dict["medium"]["diagonalscaling"], 1.07)
        assert numpy.isclose(pcm_dict["medium"]["proberadius"], 0.52917721067)
        assert numpy.isclose(pcm_dict["medium"]["correction"], 0.0)

        # noinspection PyTypeChecker
        pcm_settings = PCMSettings(
            solver=pcm_dict["medium"]["solvertype"].upper(),
            solvent=solvent_map[pcm_dict["medium"]["solvent"].upper()],
            radii_model=radii_map[pcm_dict["cavity"]["radiiset"].upper()],
            radii_scaling=pcm_dict["cavity"]["scaling"],
            cavity_area=pcm_dict["cavity"]["area"],
        )

    except (AssertionError, ValidationError) as error:
        raise InvalidPCMKeywordError(input_string) from error
    except Exception as e:
        raise e from None

    return pcm_settings


def reconstruct_density(
    wavefunction: "qcelemental.models.results.WavefunctionProperties", n_alpha: int
) -> numpy.ndarray:
    """Reconstructs a density matrix from a QCFractal wavefunction, making sure to
    order the entries in the ordering that psi4 expects (e.g. spherical, cartesian).

    Parameters
    ----------
    wavefunction
        The wavefunction return by QCFractal.
    n_alpha
        The number of alpha electrons in the computation.

    Returns
    -------
        The reconstructed density.
    """

    # Reconstruct the density in CCA order
    orbitals = wavefunction.scf_orbitals_a
    density = numpy.dot(orbitals[:, :n_alpha], orbitals[:, :n_alpha].T)

    # Re-order the density matrix to match the ordering expected by psi4.
    angular_momenta = {
        angular_momentum
        for atom in wavefunction.basis.atom_map
        for shell in wavefunction.basis.center_data[atom].electron_shells
        for angular_momentum in shell.angular_momentum
    }

    spherical_maps = {
        L: numpy.array(
            list(range(L * 2 - 1, 0, -2)) + [0] + list(range(2, L * 2 + 1, 2))
        )
        for L in angular_momenta
    }

    # Build a flat index that we can transform the AO quantities
    ao_map = []
    counter = 0

    for atom in wavefunction.basis.atom_map:
        center = wavefunction.basis.center_data[atom]
        for shell in center.electron_shells:
            if shell.harmonic_type == "cartesian":
                ao_map.append(numpy.arange(counter, counter + shell.nfunctions()))

            else:
                smap = spherical_maps[shell.angular_momentum[0]]
                ao_map.append(smap + counter)

            counter += shell.nfunctions()

    ao_map = numpy.hstack(ao_map)

    reverse_ao_map = {map_index: i for i, map_index in enumerate(ao_map)}
    reverse_ao_map = numpy.array([reverse_ao_map[i] for i in range(len(ao_map))])

    reordered_density = density[reverse_ao_map[:, None], reverse_ao_map]
    return reordered_density


@requires_package("psi4")
def compute_esp(
    qc_molecule: "qcelemental.models.Molecule",
    density: numpy.ndarray,
    esp_settings: ESPSettings,
    grid: Quantity,
    compute_field: bool = True,
) -> tuple[Quantity, Quantity | None]:
    """Computes the ESP and electric field for a particular molecule on
    a specified grid and using the specified settings.

    Parameters
    ----------
    qc_molecule
        The molecule to compute the ESP / electric field of.
    density
        The electron density of the molecule.
    esp_settings
        The settings to use when computing the ESP / electric field.
    grid
        The grid to evaluate the ESP and electric field on.
    compute_field
        Whether to compute the electric field in addition to the ESP.

    Returns
    -------
        A tuple of the evaluated ESP with shape=(n_grid_points, 1) and the electric
        field with shape=(n_grid_points, 3)
    """
    import psi4

    psi4.core.be_quiet()

    psi4_molecule = psi4.geometry(qc_molecule.to_string("psi4", "angstrom"))
    psi4_molecule.reset_point_group("c1")

    psi4_wavefunction = psi4.core.RHF(
        psi4.core.Wavefunction.build(psi4_molecule, esp_settings.basis),
        psi4.core.SuperFunctional(),
    )
    psi4_wavefunction.Da().copy(psi4.core.Matrix.from_array(density))

    psi4_calculator = psi4.core.ESPPropCalc(psi4_wavefunction)
    psi4_grid = psi4.core.Matrix.from_array(grid.to(unit.angstrom).m)

    esp = numpy.array(
        psi4_calculator.compute_esp_over_grid_in_memory(psi4_grid)
    ).reshape(-1, 1)

    field = None

    if compute_field:
        field = (
            numpy.array(psi4_calculator.compute_field_over_grid_in_memory(psi4_grid))
            * unit.hartree
            / (unit.bohr * unit.e)
        )

    return esp * unit.hartree / unit.e, field


@requires_package("qcportal")
def from_qcportal_results(
    qc_result: "qcportal.record_models.BaseRecord",
    qc_molecule: "qcelemental.models.Molecule",
    qc_keyword_set: dict,
    grid_settings: GridSettingsType,
    compute_field: bool = True,
) -> MoleculeESPRecord:
    """A function which will re-construct the ESP and optionally the electric field from
    a set of wavefunctions that have been computed by a QCFractal instance using the Psi4
    package.

    Parameters
    ----------
    qc_result
        The QCFractal result record which encodes the wavefunction
    qc_molecule
        The QC molecule corresponding to the result record.
    qc_keyword_set
        The keyword set used when computing the result record.
    grid_settings
        The settings which define the grid to evaluate the electronic properties on.
    compute_field
        Whether to compute the electric field in addition to the ESP.

    Returns
    -------
        The values of the ESP and (optionally) the electric field stored in a storable
        record object.
    """

    from openff.toolkit import Molecule

    # Compute and store the ESP and electric field for each result.
    if qc_result.wavefunction is None:
        raise MissingQCWaveFunctionError(qc_result.id)

    # Retrieve the wavefunction and use it to reconstruct the electron density.
    density = reconstruct_density(
        wavefunction=qc_result.wavefunction,
        n_alpha=qc_result.properties["calcinfo_nalpha"],
    )

    # Convert the OE molecule to a QC molecule and extract the conformer of
    # interest.
    molecule = Molecule.from_qcschema(
        qc_molecule.dict(encoding="json"), allow_undefined_stereo=True
    )

    conformers = extract_conformers(molecule)
    assert len(conformers) == 1

    conformer = conformers[0]

    # Construct the grid to evaluate the ESP / electric field on.
    grid = GridGenerator.generate(molecule, conformer, grid_settings)

    # Retrieve the ESP settings from the record.
    enable_pcm = bool(qc_keyword_set.get("pcm"))

    esp_settings = ESPSettings(
        basis=qc_result.specification.basis,
        method=qc_result.specification.method,
        grid_settings=grid_settings,
        pcm_settings=(
            None if not enable_pcm else _parse_pcm_input(qc_keyword_set["pcm__input"])
        ),
    )

    # Reconstruct the ESP and field from the density.
    esp, electric_field = compute_esp(
        qc_molecule, density, esp_settings, grid, compute_field
    )

    dipole = numpy.array(qc_result.properties["scf_dipole_moment"]).reshape((3, 1))

    return MoleculeESPRecord.from_molecule(
        molecule,
        conformer=conformer,
        grid_coordinates=grid,
        esp=esp,
        electric_field=electric_field,
        dipole=dipole,
        esp_settings=esp_settings,
    )
