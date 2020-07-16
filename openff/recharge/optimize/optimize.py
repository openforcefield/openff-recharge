from typing import List

import numpy
from openeye import oechem

from openff.recharge.charges.bcc import BCCGenerator, BCCSettings
from openff.recharge.charges.charges import ChargeGenerator, ChargeSettings
from openff.recharge.esp.storage import MoleculeESPRecord, MoleculeESPStore
from openff.recharge.utilities.geometry import (
    INVERSE_ANGSTROM_TO_BOHR,
    compute_inverse_distance_matrix,
    reorder_conformer,
)


class PrecalulatedObjective:
    """A class which stores the precalculated portion of the
    objective function. This includes the grid point - atom
    inverse distance matrix, the BCC parameter assignment
    matrix, and the difference between the full QM and
    uncorrected charge ESP."""

    def __init__(self, inverse_distance_matrix, assignment_matrix, v_difference):

        self.inverse_distance_matrix = inverse_distance_matrix
        self.assignment_matrix = assignment_matrix
        self.v_difference = v_difference


class ESPOptimization:
    """A utility class which contains helper functions for computing the
    contributions to a least squares objective function which captures the
    deviation of the ESP computed using molecular partial charges and the ESP
    computed by a QM calculation."""

    @classmethod
    def inverse_distance_matrix(cls, esp_record: MoleculeESPRecord) -> numpy.ndarray:
        """A convenience function for computing the matrix of inverse distances
        between each atom in a molecule and the grid points which the electrostatic
        potential was calculated on for a specific stored record.

        Parameters
        ----------
        esp_record
            The record which contains both the molecule definition and the
            electrostatic potential grid points.

        Returns
        -------
            The inverse distance matrix with shape=(n_grid_points, n_atoms)
        """

        return compute_inverse_distance_matrix(
            esp_record.grid_coordinates, esp_record.conformer
        )

    @classmethod
    def compute_v_difference(
        cls,
        inverse_distance_matrix: numpy.ndarray,
        uncorrected_charges: numpy.ndarray,
        electrostatic_potentials: numpy.ndarray,
    ) -> numpy.ndarray:
        """Computes the difference between a QM calculated ESP and an ESP calculated
        using a set of uncorrected partial charges [Hartree / e].

        Parameters
        ----------
        inverse_distance_matrix
            The inverse distances between all atoms in a molecule and the set of
            grid points which the ``electrostatic_potentials`` were calculated on.
        uncorrected_charges
            The partial charges on a molecule which haven't been corrected by a
            set of bond charge corrections.
        electrostatic_potentials
            The electrostatic potentials generated by a QM calculation.
        """
        return electrostatic_potentials - inverse_distance_matrix @ uncorrected_charges

    @classmethod
    def compute_v_correction(
        cls,
        inverse_distance_matrix: numpy.ndarray,
        assignment_matrix: numpy.ndarray,
        bond_charge_corrections: numpy.ndarray,
    ) -> numpy.ndarray:
        """Computes the contribution of a set of bond charge corrections to
        the ESP of a molecule [Hartree / e].

        Parameters
        ----------
        inverse_distance_matrix
            The inverse distances between all atoms in a molecule and the set of
            grid points which the ``electrostatic_potentials`` were calculated on.
        assignment_matrix
            The matrix which maps the bond charge correction parameters onto atoms
            in the molecule.
        bond_charge_corrections
            The values of the bond charge correction parameters..
        """
        return inverse_distance_matrix @ (assignment_matrix @ bond_charge_corrections)

    @classmethod
    def compute_objective_function(cls, v_difference, v_correction) -> numpy.ndarray:
        """Computes the least squares objective function.

        Parameters
        ----------
        v_difference
            The difference between a QM calculated ESP and an ESP calculated
            using a set of uncorrected partial charges [Hartree / e].
        v_correction
            The contribution of a set of bond charge corrections to
            the ESP of a molecule [Hartree / e].
        """
        delta = v_difference - v_correction
        return (delta * delta).sum()

    @classmethod
    def precalculate(
        cls,
        smiles: List[str],
        esp_store: MoleculeESPStore,
        bcc_settings: BCCSettings,
        fixed_parameter_indices: List[int],
        charge_settings: ChargeSettings,
    ) -> List[PrecalulatedObjective]:
        """Pre-calculates those terms which appear in the objective function,
        particularly the difference between the ESP calculated using QM methods
        and using the uncorrected atoms partial charges, the matrix of which
        bond charge correction parameters should be applied to which atoms, and
        the inverse distance matrix between atoms and grid points.

        Parameters
        ----------
        smiles
            The SMILES patterns of the molecules being optimised against.
        esp_store
            The store which contains the calculated ESP records.
        bcc_settings
            The settings which describe the parameters which are to be trained,
            and how they should be applied to molecules.
        fixed_parameter_indices
            The indices of the bond charge parameters specified by ``bcc_settings``
            which should be kept fixed while training.
        charge_settings
            The settings which define how to calculate the uncorrected partial charges
            on each molecule.
        """

        precalculated_components = []

        trainable_parameter_indices = numpy.array(
            [
                i
                for i in range(len(bcc_settings.bond_charge_corrections))
                if i not in fixed_parameter_indices
            ]
        )

        for smiles_pattern in smiles:

            esp_records = esp_store.retrieve(smiles=smiles_pattern)

            for esp_record in esp_records:

                oe_molecule = oechem.OEMol()
                oechem.OESmilesToMol(oe_molecule, esp_record.tagged_smiles)

                ordered_conformer = reorder_conformer(oe_molecule, esp_record.conformer)

                # Clear the records index map.
                for atom in oe_molecule.GetAtoms():
                    atom.SetMapIdx(0)

                assignment_matrix = BCCGenerator.build_assignment_matrix(
                    oe_molecule, bcc_settings
                )
                assignment_matrix = assignment_matrix[:, trainable_parameter_indices]

                # Pre-compute the inverse distance between each atom in the molecule
                # and each grid point. Care must be taken to ensure that length units
                # are converted from [Angstrom] to [Bohr].
                inverse_distance_matrix = (
                    compute_inverse_distance_matrix(
                        esp_record.grid_coordinates, ordered_conformer
                    )
                    * INVERSE_ANGSTROM_TO_BOHR
                )

                # Pre-compute the difference between the QM and the uncorrected ESP.
                am1_charges = ChargeGenerator.generate(
                    oe_molecule, [ordered_conformer], charge_settings
                )

                v_difference = ESPOptimization.compute_v_difference(
                    inverse_distance_matrix, am1_charges, esp_record.esp
                )

                precalculated_components.append(
                    PrecalulatedObjective(
                        inverse_distance_matrix, assignment_matrix, v_difference
                    )
                )

        return precalculated_components
