import abc
from typing import Generator, List, Union, Optional

import numpy
from openff.toolkit.topology import Molecule, TopologyMolecule

from openff.recharge.charges.bcc import (
    BCCCollection,
    BCCGenerator,
    VSiteSMIRNOFFCollection,
    VSiteSMIRNOFFGenerator,
)
from openff.recharge.charges.charges import ChargeGenerator, ChargeSettings
from openff.recharge.esp.storage import MoleculeESPRecord, MoleculeESPStore
from openff.recharge.utilities.geometry import (
    ANGSTROM_TO_BOHR,
    INVERSE_ANGSTROM_TO_BOHR,
    combine_assignments,
    compute_inverse_distance_matrix,
    compute_vector_field,
    reorder_conformer,
)
from openff.recharge.utilities.openeye import import_oechem
from openff.recharge.utilities.utilities import requires_package


class ObjectiveTerm(abc.ABC):
    """A class which stores precalculated values used to compute the terms of
    the objective function.

    In particular, this object contains the `design_matrix` and the target
    residuals which the BCC values are being trained to reproduce.
    """

    def __init__(self, design_matrix: numpy.ndarray, target_residuals: numpy.ndarray):
        self.design_matrix = design_matrix
        self.target_residuals = target_residuals


class _Optimization(abc.ABC):
    """A utility class which contains helper functions for computing the
    contributions to a least squares objective function which captures the
    deviation of the ESP computed using molecular partial charges and the ESP
    computed by a QM calculation."""

    @classmethod
    @abc.abstractmethod
    def _flatten_charges(cls) -> bool:
        """Whether to operate on a flattened array of charges rather than the
        2D array. This is mainly to be used when the design matrix is a tensor."""

    @classmethod
    def compute_residuals(
        cls,
        design_matrix_precursor: numpy.ndarray,
        uncorrected_charges: numpy.ndarray,
        target_values: numpy.ndarray,
    ) -> numpy.ndarray:
        """Computes the difference between a set of QM values and a corresponding set
        of values computed using a set of uncorrected partial charges.

        Parameters
        ----------
        design_matrix_precursor
            The precursor to the design matrix.
        uncorrected_charges
            The partial charges on a molecule which haven't been corrected by a
            set of bond charge corrections with shape=(n_atoms, 1) in units of [e].
        target_values
            The target values of the electronic properties being optimized against.

        Returns
        -------
            The difference between the target values and the values calculated using
            the uncorrected partial charges with.
        """

        if cls._flatten_charges():
            uncorrected_charges = uncorrected_charges.flatten()

        return target_values - design_matrix_precursor @ uncorrected_charges

    @classmethod
    def compute_bcc_contribution(
        cls,
        design_matrix_precursor: numpy.ndarray,
        assignment_matrix: numpy.ndarray,
        bcc_values: numpy.ndarray,
    ) -> numpy.ndarray:
        """Computes the contribution of a set of bond charge correction parameters to
        the electronic properties being optimized against.

        Parameters
        ----------
        design_matrix_precursor
            The precursor to the design matrix.
        assignment_matrix
            The matrix which maps the bond charge correction parameters onto atoms
            in the molecule.
        bcc_values
            The values of the bond charge correction parameters in units of [e].

        Returns
        -------
            The contribution of the bond charge correction parameters to the electronic
            properties being optimized against.
        """

        if cls._flatten_charges():
            bcc_values = bcc_values.flatten()

        return design_matrix_precursor @ (assignment_matrix @ bcc_values)

    @classmethod
    @abc.abstractmethod
    def _compute_design_matrix_precursor(
        cls, grid_coordinates: numpy, conformer: numpy
    ):
        """Computes the design matrix precursor which, when combined with the BCC
        assignment matrix, yields the full design matrix.

        Parameters
        ----------
        grid_coordinates
            The grid coordinates which the electronic property was computed on.
        conformer
            The coordinates of the molecule conformer the property was computed for.
        """

    @classmethod
    @abc.abstractmethod
    def _electronic_property(cls, record: MoleculeESPRecord) -> numpy.ndarray:
        """Returns the value of the electronic property being refit from an ESP
        record.

        Parameters
        ----------
        record
            The record which stores the electronic property being optimized against.
        """

    @classmethod
    @abc.abstractmethod
    def _create_term_object(
        cls, design_matrix: numpy.ndarray, target_residuals: numpy.ndarray
    ):
        """Returns the correct term object which contains the target residuals
        and the design matrix."""

    @classmethod
    def compute_objective_terms(
        cls,
        smiles: List[str],
        esp_store: MoleculeESPStore,
        bcc_collection: BCCCollection,
        fixed_parameter_indices: Union[numpy.ndarray, List[int]],
        charge_settings: ChargeSettings,
        vsite_collection: Optional[VSiteSMIRNOFFCollection],
    ) -> Generator[ObjectiveTerm, None, None]:
        """Pre-calculates those terms which appear in the objective function.

        These include:

            * the difference between the target values calculated using QM methods
              and using the uncorrected atoms partial charges

            * the design matrix which may be left multiplied with the bond charge
            correction values to yield the residuals due to the bond charge correction
            parameters.

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

        oechem = import_oechem()

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

                if vsite_collection:

                    vsite_assignments = VSiteSMIRNOFFGenerator.build_assignment_matrix(
                        oe_molecule, vsite_collection
                    )

                    vsite_positions = (
                        VSiteSMIRNOFFGenerator.compute_virtual_site_positions(
                            oe_molecule, vsite_collection.parameter_handler, ordered_conformer
                        )
                    )

                    vsite_positions /= vsite_positions.unit

                    ordered_conformer = numpy.vstack(
                        (ordered_conformer, vsite_positions)
                    )

                    trainable_assignment_matrix = combine_assignments(
                        trainable_assignment_matrix, vsite_assignments
                    )

                # Pre-compute the design matrix for this molecule.
                design_matrix_precursor = cls._compute_design_matrix_precursor(
                    esp_record.grid_coordinates, ordered_conformer
                )

                # Pre-compute the difference between the QM and the uncorrected
                # electronic property.
                uncorrected_charges = ChargeGenerator.generate(
                    oe_molecule, [ordered_conformer], charge_settings
                )


                target_residuals = cls.compute_residuals(
                    design_matrix_precursor,
                    uncorrected_charges,
                    cls._electronic_property(esp_record),
                )

                if len(fixed_parameter_indices) > 0:

                    if vsite_collection:
                        assignment_matrix = combine_assignments(
                            assignment_matrix, vsite_assignments
                        )
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

                    fixed_bcc_contribution = cls.compute_bcc_contribution(
                        design_matrix_precursor,
                        fixed_assignment_matrix,
                        fixed_bcc_values,
                    )

                    target_residuals -= fixed_bcc_contribution

                yield cls._create_term_object(
                    design_matrix_precursor @ trainable_assignment_matrix,
                    target_residuals,
                )


class ESPObjectiveTerm(ObjectiveTerm):
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

        super(ESPObjectiveTerm, self).__init__(design_matrix, target_residuals)


class ESPOptimization(_Optimization):
    """A utility class which contains helper functions for computing the
    contributions to a least squares objective function which captures the
    deviation of the ESP computed using molecular partial charges and the ESP
    computed by a QM calculation."""

    @classmethod
    def _flatten_charges(cls) -> bool:
        return False

    @classmethod
    def compute_residuals(
        cls,
        design_matrix_precursor: numpy.ndarray,
        uncorrected_charges: numpy.ndarray,
        target_values: numpy.ndarray,
    ) -> numpy.ndarray:
        """Computes the difference between a QM calculated ESP and an ESP calculated
        using a set of uncorrected partial charges [Hartree / e].

        Parameters
        ----------
        design_matrix_precursor
            The inverse distances between all atoms in a molecule and the set of
            grid points which the ``electrostatic_potentials`` were calculated on
            with shape=(n_grid_points, n_atoms) in units of [1 / Bohr].
        uncorrected_charges
            The partial charges on a molecule which haven't been corrected by a
            set of bond charge corrections with shape=(n_atoms, 1) in units of [e].
        target_values
            The electrostatic potentials generated by a QM calculation with
            shape=(n_grid_points, 1) in units of [Hartree / e].

        Returns
        -------
            The difference between the `electrostatic_potentials` and the ESP
            calculated using the uncorrected partial charges with
            shape=(n_grid_points, 1) in units of [Hartree / e].
        """
        return super(ESPOptimization, cls).compute_residuals(
            design_matrix_precursor, uncorrected_charges, target_values
        )

    @classmethod
    def compute_bcc_contribution(
        cls,
        design_matrix_precursor: numpy.ndarray,
        assignment_matrix: numpy.ndarray,
        bcc_values: numpy.ndarray,
    ) -> numpy.ndarray:
        """Computes the contribution of a set of bond charge correction parameters to
        the total ESP [Hartree / e].

        Parameters
        ----------
        design_matrix_precursor
            The inverse distances between all atoms in a molecule and the set of
            grid points which the ESP values were calculated on in units of [1 / Bohr].
        assignment_matrix
            The matrix which maps the bond charge correction parameters onto atoms
            in the molecule.
        bcc_values
            The values of the bond charge correction parameters in units of [e].

        Returns
        -------
            The contribution of the bond charge correction parameters to the total ESP
            with shape=(n_grid_points, 1) in units of [Hartree / e].
        """
        return super(ESPOptimization, cls).compute_bcc_contribution(
            design_matrix_precursor, assignment_matrix, bcc_values
        )

    @classmethod
    def _compute_design_matrix_precursor(
        cls, grid_coordinates: numpy, conformer: numpy
    ):
        # Pre-compute the inverse distance between each atom in the molecule
        # and each grid point. Care must be taken to ensure that length units
        # are converted from [Angstrom] to [Bohr].
        inverse_distance_matrix = (
            compute_inverse_distance_matrix(grid_coordinates, conformer)
            * INVERSE_ANGSTROM_TO_BOHR
        )

        return inverse_distance_matrix

    @classmethod
    def _electronic_property(cls, record: MoleculeESPRecord) -> numpy.ndarray:
        return record.esp

    @classmethod
    def _create_term_object(
        cls, design_matrix: numpy.ndarray, target_residuals: numpy.ndarray
    ):
        return ESPObjectiveTerm(design_matrix, target_residuals)


class ElectricFieldObjectiveTerm(ObjectiveTerm):
    def __init__(self, design_matrix: numpy.ndarray, target_residuals: numpy.ndarray):
        """

        Parameters
        ----------
        design_matrix
            A combination of the vector field point from the atoms to the grid points
            and the BCC assignment matrix which when left multiplied
            by the current BCC parameters yields the BCC contribution to the
            ESP. This should have shape=(n_grid_points, 3, n_bcc_parameters) and
            units of [1 / Bohr ^ 2].
        target_residuals
            The difference between the QM electric field values and the electric field
            values calculated using the uncorrected partial charges with
            shape=(n_grid_points, 3).
        """

        super(ElectricFieldObjectiveTerm, self).__init__(
            design_matrix, target_residuals
        )


class ElectricFieldOptimization(_Optimization):
    """A utility class which contains helper functions for computing the
    contributions to a least squares objective function which captures the
    deviation of the ESP computed using molecular partial charges and the ESP
    computed by a QM calculation."""

    @classmethod
    def _flatten_charges(cls) -> bool:
        return True

    @classmethod
    def compute_residuals(
        cls,
        design_matrix_precursor: numpy.ndarray,
        uncorrected_charges: numpy.ndarray,
        target_values: numpy.ndarray,
    ) -> numpy.ndarray:
        """Computes the difference between a QM calculated ESP and an ESP calculated
        using a set of uncorrected partial charges [Hartree / e].

        Parameters
        ----------
        design_matrix_precursor
            The vector field which points from the atoms in a molecule to the grid
            points which the electronic property was calculated on with
            shape=(n_grid_points, 3, n_bcc_parameters) and units of [1 / Bohr ^ 2].
        uncorrected_charges
            The partial charges on a molecule which haven't been corrected by a
            set of bond charge corrections with shape=(n_atoms, 1) in units of [e].
        target_values
            The electric field generated by a QM calculation with
            shape=(n_grid_points, 3) in units of [Hartree / (e . a0)].

        Returns
        -------
            The difference between the QM electric field and the electric field
            calculated using the uncorrected partial charges with
            shape=(n_grid_points, 3) in units of [Hartree / (e . a0)].
        """
        return super(ElectricFieldOptimization, cls).compute_residuals(
            design_matrix_precursor, uncorrected_charges, target_values
        )

    @classmethod
    def compute_bcc_contribution(
        cls,
        design_matrix_precursor: numpy.ndarray,
        assignment_matrix: numpy.ndarray,
        bcc_values: numpy.ndarray,
    ) -> numpy.ndarray:
        """Computes the contribution of a set of bond charge correction parameters to
        the total electric field [Hartree / (e . a0)].

        Parameters
        ----------
        design_matrix_precursor
            The vector field which points from the atoms in a molecule to the grid
            points which the electronic property was calculated on with
            shape=(n_grid_points, 3, n_bcc_parameters) and units of [1 / Bohr ^ 2].
        assignment_matrix
            The matrix which maps the bond charge correction parameters onto atoms
            in the molecule.
        bcc_values
            The values of the bond charge correction parameters in units of [e].

        Returns
        -------
            The contribution of the bond charge correction parameters to the total
            electric field with shape=(n_grid_points, 3) in units of
            [Hartree / (e . a0)].
        """
        return super(ElectricFieldOptimization, cls).compute_bcc_contribution(
            design_matrix_precursor, assignment_matrix, bcc_values
        )

    @classmethod
    def _compute_design_matrix_precursor(
        cls, grid_coordinates: numpy, conformer: numpy
    ):
        vector_field = compute_vector_field(
            conformer * ANGSTROM_TO_BOHR, grid_coordinates * ANGSTROM_TO_BOHR
        )
        return vector_field

    @classmethod
    def _electronic_property(cls, record: MoleculeESPRecord) -> numpy.ndarray:
        return record.electric_field

    @classmethod
    def _create_term_object(
        cls, design_matrix: numpy.ndarray, target_residuals: numpy.ndarray
    ):
        return ElectricFieldObjectiveTerm(design_matrix, target_residuals)
