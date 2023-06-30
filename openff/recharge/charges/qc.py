"""Generate partial for molecules from a QC calculation."""
import copy
from typing import TYPE_CHECKING, List, Literal, cast

import numpy
from openff.units import unit
from openff.units.elements import SYMBOLS
from pydantic import BaseModel, Field

from openff.recharge.charges.exceptions import ChargeAssignmentError
from openff.recharge.utilities.toolkits import get_atom_symmetries
from openff.utilities.utilities import requires_oe_module

if TYPE_CHECKING:
    from openff.toolkit import Molecule

QCChargeTheory = Literal["am1", "am1bcc", "GFN1-xTB", "GFN2-xTB"]


class QCChargeSettings(BaseModel):
    """The settings to use when assigning partial charges from
    quantum chemical calculations.
    """

    theory: QCChargeTheory = Field(
        "am1", description="The level of theory to use when computing the charges."
    )

    symmetrize: bool = Field(
        True,
        description="Whether the partial charges should be made equal for bond-"
        "topology equivalent atoms.",
    )
    optimize: bool = Field(
        True,
        description="Whether to optimize the input conformer during the charge"
        "calculation.",
    )


class QCChargeGenerator:
    """A class which will compute the partial charges of a molecule
    from a quantum chemical calculation."""

    @classmethod
    def _symmetrize_charges(
        cls, molecule: "Molecule", charges: numpy.ndarray
    ) -> numpy.ndarray:
        """Sets the charge on each atom to be the average value computed across all
        charges on atoms with the same topological symmetry group.
        """

        symmetry_groups = get_atom_symmetries(molecule)

        charges_by_group = {group: [] for group in symmetry_groups}

        for group, charge in zip(symmetry_groups, charges):
            charges_by_group[group].append(charge)

        average_charges = {
            group: float(numpy.mean(charges_by_group[group]))
            for group in charges_by_group
        }

        return numpy.array([[average_charges[group]] for group in symmetry_groups])

    @classmethod
    def _check_connectivity(cls, molecule: "Molecule", conformer: unit.Quantity):
        from qcelemental.molutil import guess_connectivity

        expected_connectivity = {
            tuple(sorted([bond.atom1_index, bond.atom2_index]))
            for bond in molecule.bonds
        }

        symbols = numpy.array([SYMBOLS[atom.atomic_number] for atom in molecule.atoms])

        actual_connectivity = {
            tuple(sorted(connection))
            for connection in guess_connectivity(
                symbols, conformer.m_as(unit.bohr), threshold=1.2
            )
        }

        if actual_connectivity == expected_connectivity:
            return

        raise ChargeAssignmentError(
            "The connectivity of the molecule changed when optimizing the initial "
            "conformer and so the output charges will be incorrect."
        )

    @classmethod
    def _generate_xtb_charges(
        cls,
        molecule: "Molecule",
        conformer: unit.Quantity,
        settings: QCChargeSettings,
    ):
        from qcelemental.models.common_models import DriverEnum, Model
        from qcelemental.models.procedures import (
            OptimizationInput,
            OptimizationProtocols,
            OptimizationResult,
            QCInputSpecification,
            TrajectoryProtocolEnum,
        )
        from qcelemental.models.results import AtomicInput, AtomicResult
        from qcengine import compute, compute_procedure

        molecule = copy.deepcopy(molecule)
        molecule._conformers = [conformer]

        qc_molecule = molecule.to_qcschema()

        if settings.optimize:
            optimization_schema = OptimizationInput(
                initial_molecule=qc_molecule,
                input_specification=QCInputSpecification(
                    model=Model(method=settings.theory),
                ),
                protocols=OptimizationProtocols(
                    trajectory=TrajectoryProtocolEnum.final
                ),
                keywords={"program": "xtb"},
            )
            optimization_results: OptimizationResult = cast(
                OptimizationResult,
                compute_procedure(optimization_schema, "geometric", raise_error=True),
            )

            cls._check_connectivity(
                molecule, optimization_results.final_molecule.geometry * unit.bohr
            )

            results = optimization_results.trajectory[-1]
        else:
            input_schema = AtomicInput(
                molecule=qc_molecule,
                driver=DriverEnum.energy,
                model=Model(method=settings.theory),
            )

            results = cast(AtomicResult, compute(input_schema, "xtb", raise_error=True))

        charges = numpy.array(results.extras["xtb"]["mulliken_charges"]).reshape(
            (-1, 1)
        )

        if settings.symmetrize:
            charges = cls._symmetrize_charges(molecule, charges)

        return charges

    @requires_oe_module("oechem")
    @classmethod
    def _generate_omega_charges(
        cls,
        molecule: "Molecule",
        conformer: numpy.ndarray,
        settings: QCChargeSettings,
    ) -> numpy.ndarray:
        oe_molecule = molecule.to_openeye()

        from openeye import oechem, oequacpac

        oe_molecule.DeleteConfs()
        oe_molecule.NewConf(oechem.OEFloatArray(conformer.flatten()))

        if settings.theory == "am1":
            assert oequacpac.OEAssignCharges(
                oe_molecule,
                oequacpac.OEAM1Charges(
                    optimize=settings.optimize, symmetrize=settings.symmetrize
                ),
            ), f"QUACPAC failed to generate {settings.theory} charges"
        elif settings.theory == "am1bcc":
            oequacpac.OEAssignCharges(
                oe_molecule,
                oequacpac.OEAM1BCCCharges(
                    optimize=settings.optimize, symmetrize=settings.symmetrize
                ),
            ), f"QUACPAC failed to generate {settings.theory} charges"
        else:
            raise NotImplementedError()

        atoms = {atom.GetIdx(): atom for atom in oe_molecule.GetAtoms()}
        return numpy.array(
            [
                [atoms[index].GetPartialCharge()]
                for index in range(oe_molecule.NumAtoms())
            ]
        )

    @classmethod
    def _generate_am1_charges(
        cls,
        molecule: "Molecule",
        conformer: numpy.ndarray,
        settings: QCChargeSettings,
    ):
        if settings.theory == "am1" and settings.optimize and settings.symmetrize:
            charge_method = "am1-mulliken"
        elif settings.theory == "am1bcc" and settings.optimize and settings.symmetrize:
            charge_method = "am1bcc"
        elif (
            settings.theory == "am1bcc"
            and not settings.optimize
            and not settings.symmetrize
        ):
            charge_method = "am1bccnosymspt"
        else:
            charge_method = None

        if charge_method:
            molecule.assign_partial_charges(
                charge_method, use_conformers=[conformer * unit.angstrom]
            )
            return molecule.partial_charges.m_as(unit.elementary_charge)

        return cls._generate_omega_charges(molecule, conformer, settings)

    @classmethod
    def generate(
        cls,
        molecule: "Molecule",
        conformers: List[unit.Quantity],
        settings: QCChargeSettings,
    ) -> numpy.ndarray:
        """Generates the averaged partial charges from multiple conformers of a
        specified molecule.

        Notes
        -----
        * Virtual sites will be assigned a partial charge of 0.0 e.

        Parameters
        ----------
        molecule
            The molecule to compute the partial charges for.
        conformers
            The conformers to use in the partial charge calculations
            where each conformer should have a shape=(n_atoms + n_vsites, 3).
        settings
            The settings to use in the charge generation.

        Returns
        -------
            The computed partial charges.
        """

        # Make a copy of the molecule so as not to perturb the original.
        molecule = copy.deepcopy(molecule)

        conformer_charges = []

        for conformer in conformers:
            conformer = conformer[: molecule.n_atoms]

            if settings.theory in {"am1", "am1bcc"}:
                conformer_charges.append(
                    cls._generate_am1_charges(
                        molecule, conformer.m_as(unit.angstrom), settings
                    )
                )
            elif settings.theory.lower().endswith("xtb"):
                conformer_charges.append(
                    cls._generate_xtb_charges(molecule, conformer, settings)
                )
            else:
                raise NotImplementedError()

        charges = numpy.mean(conformer_charges, axis=0).reshape(-1, 1)
        n_vsites = len(conformers[0]) - molecule.n_atoms

        if n_vsites:
            charges = numpy.vstack((charges, numpy.zeros((n_vsites, 1))))

        return charges
