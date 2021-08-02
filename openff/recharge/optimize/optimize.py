import abc
from typing import TYPE_CHECKING, Generator, List, Optional, Tuple

import numpy

from openff.recharge.charges.bcc import BCCCollection, BCCGenerator
from openff.recharge.charges.charges import ChargeGenerator, ChargeSettings
from openff.recharge.charges.vsite import (
    VirtualSiteChargeKey,
    VirtualSiteCollection,
    VirtualSiteGenerator,
)
from openff.recharge.esp.storage import MoleculeESPRecord
from openff.recharge.utilities.geometry import (
    ANGSTROM_TO_BOHR,
    INVERSE_ANGSTROM_TO_BOHR,
    compute_inverse_distance_matrix,
    compute_vector_field,
    reorder_conformer,
)
from openff.recharge.utilities.openeye import import_oechem

if TYPE_CHECKING:
    from openeye.oechem import OEMol


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
            set of charge increments with shape=(n_atoms, 1) in units of [e].
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
    def compute_charge_increment_contribution(
        cls,
        design_matrix_precursor: numpy.ndarray,
        assignment_matrix: numpy.ndarray,
        charge_increments: numpy.ndarray,
    ) -> numpy.ndarray:
        """Computes the contribution of a set of charge increments parameters to
        the electronic properties being optimized against.

        Parameters
        ----------
        design_matrix_precursor
            The precursor to the design matrix.
        assignment_matrix
            The matrix which maps the charge increment parameters onto atoms / virtual
            sites in the molecule.
        charge_increments
            The values of the charge increments in units of [e].

        Returns
        -------
            The contribution of the charge increment parameters to the electronic
            properties being optimized against.
        """

        if cls._flatten_charges():
            charge_increments = charge_increments.flatten()

        return design_matrix_precursor @ (assignment_matrix @ charge_increments)

    @classmethod
    @abc.abstractmethod
    def _compute_design_matrix_precursor(
        cls, grid_coordinates: numpy.ndarray, conformer: numpy.ndarray
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
    def _compute_bcc_terms(
        cls,
        oe_molecule: "OEMol",
        bcc_collection: BCCCollection,
        trainable_bcc_parameters: List[str],
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:

        trainable_parameter_indices = [
            i
            for i, parameter in enumerate(bcc_collection.parameters)
            if parameter.smirks in trainable_bcc_parameters
        ]
        fixed_parameter_indices = [
            i
            for i in range(len(bcc_collection.parameters))
            if i not in trainable_parameter_indices
        ]

        assignment_matrix = BCCGenerator.build_assignment_matrix(
            oe_molecule, bcc_collection
        )

        fixed_assignment_matrix = assignment_matrix[:, fixed_parameter_indices]
        fixed_bcc_values = numpy.array(
            [
                [bcc_collection.parameters[index].value]
                for index in fixed_parameter_indices
            ]
        )

        fixed_charges = fixed_assignment_matrix @ fixed_bcc_values

        trainable_assignment_matrix = assignment_matrix[:, trainable_parameter_indices]

        return trainable_assignment_matrix, fixed_charges.reshape(-1, 1)

    @classmethod
    def _compute_vsite_terms(
        cls,
        oe_molecule: "OEMol",
        vsite_collection: VirtualSiteCollection,
        trainable_vsite_parameters: List[VirtualSiteChargeKey],
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:

        trainable_parameter_indices = []

        fixed_parameter_indices = []
        fixed_parameter_values = []

        array_counter = -1

        for parameter, parameter_index, charge_index in [
            (parameter, i, j)
            for i, parameter in enumerate(vsite_collection.parameters)
            for j in range(len(parameter.charge_increments))
        ]:

            array_counter += 1

            if (
                parameter.smirks,
                parameter.type,
                parameter.name,
                charge_index,
            ) in trainable_vsite_parameters:

                trainable_parameter_indices.append(array_counter)

            else:

                fixed_parameter_indices.append(array_counter)
                fixed_parameter_values.append(parameter.charge_increments[charge_index])

        assignment_matrix = VirtualSiteGenerator.build_charge_assignment_matrix(
            oe_molecule, vsite_collection
        )

        fixed_assignment_matrix = assignment_matrix[:, fixed_parameter_indices]
        fixed_parameter_values = numpy.array(fixed_parameter_values)

        fixed_charges = fixed_assignment_matrix @ fixed_parameter_values

        trainable_assignment_matrix = assignment_matrix[:, trainable_parameter_indices]

        return trainable_assignment_matrix, fixed_charges.reshape(-1, 1)

    @classmethod
    def compute_objective_terms(
        cls,
        esp_records: List[MoleculeESPRecord],
        charge_settings: Optional[ChargeSettings],
        bcc_collection: Optional[BCCCollection] = None,
        trainable_bcc_parameters: Optional[List[str]] = None,
        vsite_collection: Optional[VirtualSiteCollection] = None,
        trainable_vsite_parameters: Optional[List[VirtualSiteChargeKey]] = None,
    ) -> Generator[ObjectiveTerm, None, None]:
        """Pre-calculates the terms that contribute to the objective function in the
        form ``chi_i = y_i - X_i @ beta`` where ``y_i`` and ``X_i`` are the target
        residual and design matrix of term ``i`` and ``beta`` is an array of the
        current parameters.

        Parameters
        ----------
        esp_records
            The calculated records that contain the reference ESP and electric field
            data to train against.
        charge_settings
            The (optional) settings that define how to calculate the base set of charges
            that will be perturbed by trainable charge increment parameters. If no value
            is provided all base charges will be set to zero.
        bcc_collection
            A collection of bond charge correction parameters that should perturb the
            base set of charges for each molecule in the ``esp_records`` list.
        trainable_bcc_parameters
            A list of SMIRKS patterns referencing those parameters in the
            ``bcc_collection`` that should be trained.
        vsite_collection
            A collection of virtual site parameters that should create virtual sites
            for each molecule in the ``esp_records`` list and perturb the base charges
            on each atom.
        trainable_vsite_parameters
            A list of virtual site keys (tuples of the form ``(smirks, type, name)``)
            referencing those parameters in the ``vsite_collection`` that should be
            trained.

        Returns
        -------
            The precalculated terms which may be used to compute the full
            contribution to the objective function.
        """

        oechem = import_oechem()

        for esp_record in esp_records:

            oe_molecule = oechem.OEMol()
            oechem.OESmilesToMol(oe_molecule, esp_record.tagged_smiles)

            ordered_conformer = reorder_conformer(oe_molecule, esp_record.conformer)

            # Clear the records index map as these may throw off future steps.
            for atom in oe_molecule.GetAtoms():
                atom.SetMapIdx(0)

            if charge_settings is not None:

                fixed_charges = ChargeGenerator.generate(
                    oe_molecule, [ordered_conformer], charge_settings
                )

            else:

                fixed_charges = numpy.zeros((oe_molecule.NumAtoms(), 1))

            trainable_assignment_matrices = []

            n_vsites = 0

            if vsite_collection is not None:

                vsite_assignment_matrix, vsite_fixed_charges = cls._compute_vsite_terms(
                    oe_molecule, vsite_collection, trainable_vsite_parameters
                )

                n_vsites = len(vsite_assignment_matrix) - oe_molecule.NumAtoms()
                fixed_charges = numpy.vstack(
                    [fixed_charges, numpy.zeros((n_vsites, 1))]
                )

                fixed_charges += vsite_fixed_charges
                trainable_assignment_matrices.append(vsite_assignment_matrix)

                ordered_conformer = numpy.vstack(
                    [
                        ordered_conformer,
                        VirtualSiteGenerator.generate_positions(
                            oe_molecule, vsite_collection, ordered_conformer
                        ),
                    ]
                )

            if bcc_collection is not None:

                bcc_assignment_matrix, bcc_fixed_charges = cls._compute_bcc_terms(
                    oe_molecule, bcc_collection, trainable_bcc_parameters
                )
                bcc_fixed_charges = numpy.vstack(
                    [bcc_fixed_charges, numpy.zeros((n_vsites, 1))]
                )

                fixed_charges += bcc_fixed_charges

                bcc_assignment_matrix = numpy.vstack(
                    [
                        bcc_assignment_matrix,
                        numpy.zeros((n_vsites, bcc_assignment_matrix.shape[1])),
                    ]
                )
                trainable_assignment_matrices.insert(0, bcc_assignment_matrix)

            # Pre-compute the design matrix for this molecule as this is the most
            # expensive step in the optimization.
            design_matrix_precursor = cls._compute_design_matrix_precursor(
                esp_record.grid_coordinates, ordered_conformer
            )

            target_residuals = cls.compute_residuals(
                design_matrix_precursor,
                fixed_charges,
                cls._electronic_property(esp_record),
            )

            trainable_assignment_matrix = numpy.hstack(trainable_assignment_matrices)

            yield cls._create_term_object(
                design_matrix_precursor @ trainable_assignment_matrix,
                target_residuals,
            )

    @classmethod
    def vectorize_collections(
        cls,
        bcc_collection: Optional[BCCCollection] = None,
        trainable_bcc_parameters: Optional[List[str]] = None,
        vsite_collection: Optional[VirtualSiteCollection] = None,
        trainable_vsite_parameters: Optional[List[VirtualSiteChargeKey]] = None,
    ) -> numpy.ndarray:
        """Returns a flat vector containing any bond charge correction and virtual
        site charge increment parameters to be trained

        The array is ordered to match the design matrices produced by
        ``compute_objective_terms`` such that ``term.design_matrix @ charge_vector``
        yield a vector of residuals.

        Parameters
        ----------
        bcc_collection
            A collection of bond charge correction parameters that should perturb the
            base set of charges for each molecule in the ``esp_records`` list.
        trainable_bcc_parameters
            A list of SMIRKS patterns referencing those parameters in the
            ``bcc_collection`` that should be trained.
        vsite_collection
            A collection of virtual site parameters that should create virtual sites
            for each molecule in the ``esp_records`` list and perturb the base charges
            on each atom.
        trainable_vsite_parameters
            A list of virtual site keys (tuples of the form ``(smirks, type, name)``)
            referencing those parameters in the ``vsite_collection`` that should be
            trained.

        Returns
        -------
            An array of the form `[bcc_q_0, ..., bcc_q_n, vsite_q_0, ..., vsite_q_m`
            containing the bond charge correction and virtual site charge increment
            parameters
        """

        charge_increments = []

        if bcc_collection is not None:

            charge_increments.extend(
                parameter.value
                for parameter in bcc_collection.parameters
                if parameter.smirks in trainable_bcc_parameters
            )

        if vsite_collection is not None:

            charge_increments.extend(
                parameter.charge_increments[charge_index]
                for i, parameter in enumerate(vsite_collection.parameters)
                for charge_index in range(len(parameter.charge_increments))
                if (
                    parameter.smirks,
                    parameter.type,
                    parameter.name,
                    charge_index,
                )
                in trainable_vsite_parameters
            )

        return numpy.array(charge_increments).reshape((-1, 1))


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
            set of charge increment parameters with shape=(n_atoms, 1) in units of [e].
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
    def compute_charge_increment_contribution(
        cls,
        design_matrix_precursor: numpy.ndarray,
        assignment_matrix: numpy.ndarray,
        charge_increments: numpy.ndarray,
    ) -> numpy.ndarray:
        """Computes the contribution of a set of charge increments parameters to
        the total ESP [Hartree / e].

        Parameters
        ----------
        design_matrix_precursor
            The inverse distances between all atoms in a molecule and the set of
            grid points which the ESP values were calculated on in units of [1 / Bohr].
        assignment_matrix
            The matrix which maps the charge increment parameters onto atoms / virtual
            sites in the molecule.
        charge_increments
            The values of the charge increments in units of [e].

        Returns
        -------
            The contribution of the charge increment parameters to the total ESP
            with shape=(n_grid_points, 1) in units of [Hartree / e].
        """
        return super(ESPOptimization, cls).compute_charge_increment_contribution(
            design_matrix_precursor, assignment_matrix, charge_increments
        )

    @classmethod
    def _compute_design_matrix_precursor(
        cls, grid_coordinates: numpy.ndarray, conformer: numpy.ndarray
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
            set of charge increment parameters with shape=(n_atoms, 1) in units of [e].
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
    def compute_charge_increment_contribution(
        cls,
        design_matrix_precursor: numpy.ndarray,
        assignment_matrix: numpy.ndarray,
        charge_increments: numpy.ndarray,
    ) -> numpy.ndarray:
        """Computes the contribution of a set of charge increments parameters to
        the total electric field [Hartree / (e . a0)].

        Parameters
        ----------
        design_matrix_precursor
            The vector field which points from the atoms in a molecule to the grid
            points which the electronic property was calculated on with
            shape=(n_grid_points, 3, n_bcc_parameters) and units of [1 / Bohr ^ 2].
        assignment_matrix
            The matrix which maps the charge increment parameters onto atoms / virtual
            sites in the molecule.
        charge_increments
            The values of the charge increments in units of [e].

        Returns
        -------
            The contribution of the charge increment parameters to the total
            electric field with shape=(n_grid_points, 3) in units of
            [Hartree / (e . a0)].
        """
        return super(
            ElectricFieldOptimization, cls
        ).compute_charge_increment_contribution(
            design_matrix_precursor, assignment_matrix, charge_increments
        )

    @classmethod
    def _compute_design_matrix_precursor(
        cls, grid_coordinates: numpy.ndarray, conformer: numpy.ndarray
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
