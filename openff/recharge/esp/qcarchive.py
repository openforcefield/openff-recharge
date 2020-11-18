from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple

import numpy

from openff.recharge.esp import DFTGridSettings, ESPSettings
from openff.recharge.esp.storage import MoleculeESPRecord
from openff.recharge.grids import GridGenerator
from openff.recharge.utilities import requires_package
from openff.recharge.utilities.exceptions import RechargeException
from openff.recharge.utilities.openeye import molecule_to_conformers

if TYPE_CHECKING:
    import qcelemental.models
    import qcelemental.models.results
    import qcportal.models


class MissingQCMoleculesError(RechargeException):
    """An exception raised when an expected set of molecules are not present
    in a QC data set."""

    def __init__(self, data_set_name: str, missing_smiles: Iterable[str]):

        smiles_string = "\n".join(missing_smiles)

        super(MissingQCMoleculesError, self).__init__(
            f"The {smiles_string} SMILES patterns were not found in the "
            f"{data_set_name} data set."
        )

        self.data_set_name = data_set_name
        self.missing_smiles = missing_smiles


class MissingQCResultsError(RechargeException):
    """An exception raised when an expected set of results are not present
    in a QC data set."""

    def __init__(self, data_set_name: str, missing_ids: Iterable[str]):

        id_string = "\n".join(missing_ids)

        super(MissingQCResultsError, self).__init__(
            f"The result records associated with the following molecule ids from the "
            f"{data_set_name} data set could not be retrieved from QCA: {id_string}"
        )

        self.data_set_name = data_set_name
        self.missing_ids = missing_ids


class MissingQCWaveFunctionError(RechargeException):
    """An exception raised when a result does not store the required information about
    a computed QM wavefunction."""

    def __init__(self, result_id: str):

        super(MissingQCWaveFunctionError, self).__init__(
            f"The result with id={result_id} does not store the required wavefunction."
            f"Make sure to use at minimum the 'orbitals_and_eigenvalues' wavefunction "
            f"protocol when computing the data set."
        )
        self.result_id = result_id


def _retrieve_qca_results(
    data_set_name: str,
    subset: Optional[Iterable[str]],
    esp_settings: ESPSettings,
    qcfractal_address: Optional[str] = None,
) -> List[Tuple["qcelemental.models.Molecule", "qcportal.models.ResultRecord"]]:
    """Attempt to retrieve the results for the requested data set from a QCFractal
    server.

    Parameters
    ----------
    data_set_name
        The name of the data set to retrive the results from.
    subset
        The SMILES representations of the subset of molecules to retrieve from the data
        set.
    esp_settings
        The settings which define the method and basis which the results should have
        been computed using.
    qcfractal_address
        An optional address to the QCFractal server instance which stores the data set.

    Returns
    -------
        A list of the retrieved results alongside their corresponding molecule records.
    """

    import cmiles
    import qcportal
    from qcelemental.models import Molecule as QCMolecule

    # Map the input smiles to uniform isomeric and explicit hydrogen smiles.
    subset = (
        None
        if subset is None
        else [
            cmiles.get_molecule_ids(smiles, "openeye", strict=False)[
                "canonical_isomeric_explicit_hydrogen_smiles"
            ]
            for smiles in subset
        ]
    )

    # Connect to the default QCA server and retrieve the data set of interest.
    if qcfractal_address is None:
        client = qcportal.FractalClient()
    else:
        client = qcportal.FractalClient(address=qcfractal_address)

    # noinspection PyTypeChecker
    collection: qcportal.collections.Dataset = client.get_collection(
        "Dataset", data_set_name
    )

    # Retrieve the ids of the molecules of interest.
    molecules = {}
    found_smiles = set()

    for _, molecule_row in collection.get_molecules().iterrows():

        qc_molecule: QCMolecule = molecule_row["molecule"]

        # Manually map the molecule to a dictionary as CMILES expects a flat geometry
        # array.
        qc_molecule_dict = {
            "symbols": qc_molecule.symbols,
            "connectivity": qc_molecule.connectivity,
            "geometry": qc_molecule.geometry.flatten(),
            "molecular_charge": qc_molecule.molecular_charge,
            "molecular_multiplicity": qc_molecule.molecular_multiplicity,
        }

        cmiles_ids = cmiles.get_molecule_ids(qc_molecule_dict, toolkit="openeye")
        molecule_smiles = cmiles_ids["canonical_isomeric_explicit_hydrogen_smiles"]

        if subset is not None and molecule_smiles not in subset:
            continue

        molecules[qc_molecule.id] = qc_molecule
        found_smiles.add(molecule_smiles)

    molecule_ids = sorted(molecules)

    # Make sure the data set contains the requested subset.
    missing_smiles = (set() if subset is None else {*subset}) - found_smiles

    if len(missing_smiles) > 0:
        raise MissingQCMoleculesError(data_set_name, missing_smiles)

    # Retrieve the data sets results records
    all_results = []

    paginating = True
    page_index = 0

    while paginating:

        results = client.query_results(
            molecule=molecule_ids,
            method=esp_settings.method,
            basis=esp_settings.basis,
            limit=client.server_info["query_limit"],
            skip=page_index,
        )

        all_results.extend(results)

        paginating = len(results) > 0
        page_index += client.server_info["query_limit"]

    # Make sure none of the records are missing.
    result_ids = {result.molecule for result in all_results}

    missing_result_ids = {*molecule_ids} - {*result_ids}

    if len(missing_result_ids) > 0:
        raise MissingQCResultsError(data_set_name, missing_result_ids)

    return [(molecules[result.molecule], result) for result in all_results]


def _reconstruct_density(
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
    orbitals = getattr(wavefunction, wavefunction.orbitals_a)
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
def _compute_esp(
    qc_molecule, density, esp_settings, grid
) -> Tuple[numpy.ndarray, numpy.ndarray]:
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

    Returns
    -------
        A tuple of the evaluated ESP with shape=(n_grid_points, 1) and the electric
        field with shape=(n_grid_points, 3)
    """
    import psi4

    psi4_molecule = psi4.geometry(qc_molecule.to_string("psi4", "angstrom"))
    psi4_molecule.reset_point_group("c1")

    psi4_wavefunction = psi4.core.RHF(
        psi4.core.Wavefunction.build(psi4_molecule, esp_settings.basis),
        psi4.core.SuperFunctional(),
    )
    psi4_wavefunction.Da().copy(psi4.core.Matrix.from_array(density))

    psi4_calculator = psi4.core.ESPPropCalc(psi4_wavefunction)
    psi4_grid = psi4.core.Matrix.from_array(grid)

    esp = numpy.array(
        psi4_calculator.compute_esp_over_grid_in_memory(psi4_grid)
    ).reshape(-1, 1)

    # TODO: enable this once an updated psi4 is released.
    # field = numpy.array(psi4_calculator.compute_field_over_grid_in_memory(psi4_grid))

    field = numpy.zeros((len(esp), 3))

    return esp, field


@requires_package("qcportal")
@requires_package("cmiles")
def from_qcarchive(
    data_set_name: str,
    subset: Optional[Iterable[str]],
    esp_settings: ESPSettings,
    qcfractal_address: Optional[str] = None,
) -> List[MoleculeESPRecord]:
    """A function to compute the ESP and electric field from a set of precomputed
    wavefunctions stored in a particular ``Dataset`` in the MOLSSI `QCArchive
    <https://qcarchive.molssi.org/>`_ repository.

    The ESP and electric fields are currently constructed from the stored wavefunctions
    using the Psi4 package.

    Parameters
    ----------
    data_set_name
        The name of the data set to compute the ESP and field for.
    subset
        The SMILES representations of the subset of molecules to compute the
        ESP and field for.
    esp_settings
        The settings which define how to compute the ESP and field.
    qcfractal_address
        An optional address to the QCFractal server instance which stores the data set.

    Returns
    -------
        The computed records.
    """

    import cmiles.utils
    from qcelemental.models.results import WavefunctionProperties

    if esp_settings.psi4_dft_grid_settings != DFTGridSettings.Default:

        raise NotImplementedError(
            "The `psi4_dft_grid_settings` are `openff-recharge` specific and hence "
            "cannot be used in combination with QCFractal computations."
        )

    # Pull down the appropriate results from the QCA.
    qca_results = _retrieve_qca_results(
        data_set_name, subset, esp_settings, qcfractal_address
    )

    # Compute and store the ESP and electric field for each result.
    records = []

    for (qc_molecule, result) in qca_results:

        if result.wavefunction is None:
            raise MissingQCWaveFunctionError(result.id)

        # Retrieve the wavefunction and use it to reconstruct the electron density.
        wavefunction = WavefunctionProperties(
            **result.get_wavefunction(
                ["scf_eigenvalues_a", "scf_orbitals_a", "basis", "restricted"]
            ),
            **result.wavefunction["return_map"],
        )

        density = _reconstruct_density(wavefunction, result.properties.calcinfo_nalpha)

        # Convert the OE molecule to a QC molecule and extract the conformer of
        # interest.
        oe_molecule = cmiles.utils.load_molecule(
            {
                "symbols": qc_molecule.symbols,
                "connectivity": qc_molecule.connectivity,
                "geometry": qc_molecule.geometry.flatten(),
                "molecular_charge": qc_molecule.molecular_charge,
                "molecular_multiplicity": qc_molecule.molecular_multiplicity,
            },
            toolkit="openeye",
        )

        conformers = molecule_to_conformers(oe_molecule)
        assert len(conformers) == 1

        conformer = conformers[0]

        # Contruct the grid to evaluate the ESP / electric field on.
        grid = GridGenerator.generate(
            oe_molecule, conformer, esp_settings.grid_settings
        )

        # Reconstruct the ESP and field from the density.
        esp, electric_field = _compute_esp(qc_molecule, density, esp_settings, grid)

        records.append(
            MoleculeESPRecord.from_oe_molecule(
                oe_molecule,
                conformer=conformer,
                grid_coordinates=grid,
                esp=esp,
                electric_field=electric_field,
                esp_settings=esp_settings,
            )
        )

    return records
