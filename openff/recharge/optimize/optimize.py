from typing import List

import numpy
from openeye import oechem

from openff.recharge.charges.bcc import BCCCollection, BCCGenerator
from openff.recharge.charges.charges import ChargeGenerator, ChargeSettings
from openff.recharge.esp.storage import MoleculeESPStore
from openff.recharge.utilities.geometry import (
    INVERSE_ANGSTROM_TO_BOHR,
    compute_inverse_distance_matrix,
    reorder_conformer,
)


class ObjectiveTerm:
    """A class which stores precalculated values used to compute a single term of
    the objective function. In particular the `design_matrix` (a combination
    of the grid point distance matrix and the assigment matrix) and the
    target ESP residuals which the BCC values are being trained to reproduce
    are stored in this object.
    """

    def __init__(self, design_matrix: numpy.ndarray, target_residuals: numpy.ndarray):
        """

        Parameters
        ----------
        design_matrix
            A combination of the inverse distances between the grid points
            and atoms and the BCC assignment matrix which when left multiplied
            by the current BCC parameters yields the BCC contribution to the
            ESP. This should have shape=(n_grid_points, n_bcc_parameters) and
            units of [1 / Bohr].
        target_residuals
            The difference between the QM ESP values and the ESP values
            calculated using the uncorrected partial charges with
            shape=(n_grid_points, 1).
        """

        self.design_matrix = design_matrix
        self.target_residuals = target_residuals


class ESPOptimization:
    """A utility class which contains helper functions for computing the
    contributions to a least squares objective function which captures the
    deviation of the ESP computed using molecular partial charges and the ESP
    computed by a QM calculation."""

    @classmethod
    def compute_esp_residuals(
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
            grid points which the ``electrostatic_potentials`` were calculated on
            with shape=(n_grid_points, n_atoms) in units of [1 / Bohr].
        uncorrected_charges
            The partial charges on a molecule which haven't been corrected by a
            set of bond charge corrections with shape=(n_atoms, 1) in units of [e].
        electrostatic_potentials
            The electrostatic potentials generated by a QM calculation with
            shape=(n_grid_points, 1) in units of [Hartree / e].

        Returns
        -------
            The difference between the `electrostatic_potentials` and the ESP
            calculated using the uncorrected partial charges with
            shape=(n_grid_points, 1) in units of [Hartree / e].
        """
        return electrostatic_potentials - inverse_distance_matrix @ uncorrected_charges

    @classmethod
    def compute_bcc_esp(
        cls,
        inverse_distance_matrix: numpy.ndarray,
        assignment_matrix: numpy.ndarray,
        bond_charge_corrections: numpy.ndarray,
    ) -> numpy.ndarray:
        """Computes the contribution of a set of bond charge corrections to
        the ESP [Hartree / e].

        Parameters
        ----------
        inverse_distance_matrix
            The inverse distances between all atoms in a molecule and the set of
            grid points which the ``electrostatic_potentials`` were calculated on
            in units of [1 / Bohr].
        assignment_matrix
            The matrix which maps the bond charge correction parameters onto atoms
            in the molecule.
        bond_charge_corrections
            The values of the bond charge correction parameters in units of [e].

        Returns
        -------
            The contribution of a set of bond charge corrections to
            the ESP with shape=(n_grid_points, 1) in units of [Hartree / e].
        """
        return inverse_distance_matrix @ (assignment_matrix @ bond_charge_corrections)

    @classmethod
    def compute_objective_terms(
        cls,
        smiles: List[str],
        esp_store: MoleculeESPStore,
        bcc_collection: BCCCollection,
        fixed_parameter_indices: List[int],
        charge_settings: ChargeSettings,
    ) -> List[ObjectiveTerm]:
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
        bcc_collection
            The collection of bond charge correction parameter to be trained.
        fixed_parameter_indices
            The indices of the bond charge parameters specified by ``bcc_collection``
            which should be kept fixed while training. The contribution of these
            fixed terms will be included in the `target_residuals` along with the
            contribution of the uncorrected partial charges.
        charge_settings
            The settings which define how to calculate the uncorrected partial charges
            on each molecule.

        Returns
        -------
            The precalculated terms which may be used to compute the full
            contribution to the objective function.
        """

        objective_terms = []

        trainable_parameter_indices = numpy.array(
            [
                i
                for i in range(len(bcc_collection.parameters))
                if i not in fixed_parameter_indices
            ]
        )
        fixed_parameter_indices = numpy.array(fixed_parameter_indices)

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
                    oe_molecule, bcc_collection
                )
                trainable_assignment_matrix = assignment_matrix[
                    :, trainable_parameter_indices
                ]

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
                uncorrected_charges = ChargeGenerator.generate(
                    oe_molecule, [ordered_conformer], charge_settings
                )

                target_residuals = ESPOptimization.compute_esp_residuals(
                    inverse_distance_matrix, uncorrected_charges, esp_record.esp
                )

                if len(fixed_parameter_indices) > 0:

                    # Compute the contribution of the fixed BCC parameters.
                    fixed_assignment_matrix = assignment_matrix[
                        :, fixed_parameter_indices
                    ]
                    fixed_bcc_values = numpy.array(
                        [
                            [bcc_collection.parameters[index].value]
                            for index in fixed_parameter_indices
                        ]
                    )

                    fixed_bcc_esp = cls.compute_bcc_esp(
                        inverse_distance_matrix,
                        fixed_assignment_matrix,
                        fixed_bcc_values,
                    )

                    target_residuals -= fixed_bcc_esp

                objective_terms.append(
                    ObjectiveTerm(
                        inverse_distance_matrix @ trainable_assignment_matrix,
                        target_residuals,
                    )
                )

        return objective_terms
