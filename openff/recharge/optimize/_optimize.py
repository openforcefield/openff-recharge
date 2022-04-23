import abc
from collections import defaultdict
from typing import TYPE_CHECKING, Generator, List, Optional, Tuple, Type, TypeVar, Union

import numpy
from openff.units import unit
from typing_extensions import Literal

from openff.recharge.charges.bcc import BCCCollection, BCCGenerator
from openff.recharge.charges.library import (
    LibraryChargeCollection,
    LibraryChargeGenerator,
)
from openff.recharge.charges.qc import QCChargeGenerator, QCChargeSettings
from openff.recharge.charges.vsite import (
    VirtualSiteChargeKey,
    VirtualSiteCollection,
    VirtualSiteGenerator,
    VirtualSiteGeometryKey,
)
from openff.recharge.esp.storage import MoleculeESPRecord
from openff.recharge.utilities.geometry import (
    compute_inverse_distance_matrix,
    compute_vector_field,
)
from openff.recharge.utilities.tensors import (
    TensorType,
    append_zero,
    concatenate,
    to_numpy,
    to_torch,
)

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule


_VSITE_ATTRIBUTES = ("distance", "in_plane_angle", "out_of_plane_angle")

_TERM_T = TypeVar("_TERM_T", bound="ObjectiveTerm")


class ObjectiveTerm(abc.ABC):
    """A base for classes that stores precalculated values used to compute the terms of
    an objective function.

    See the ``predict`` and ``loss`` functions for more details.
    """

    @classmethod
    @abc.abstractmethod
    def _objective(cls) -> Type["Objective"]:
        """The objective class that this term is associated with."""
        raise NotImplementedError()

    def __init__(
        self,
        atom_charge_design_matrix: Optional[TensorType],
        #
        vsite_charge_assignment_matrix: Optional[TensorType],
        vsite_fixed_charges: Optional[TensorType],
        #
        vsite_coord_assignment_matrix: Optional[TensorType],
        vsite_fixed_coords: Optional[TensorType],
        #
        vsite_local_coordinate_frame: Optional[TensorType],
        #
        grid_coordinates: Optional[TensorType],
        reference_values: TensorType,
    ):
        """

        Parameters
        ----------
        atom_charge_design_matrix
            A matrix with shape=(n_grid_points, n_bcc_charges + n_vsite_charges) that
            yields the atom contributions to electrostatic property of interest when
            left multiplying by a vector of partial charges parameters [e] with
            shape=(n_bcc_charges + n_vsite_charges, 1).

            It is usually constructed by ``X=RT`` where ``R`` is either an inverse
            distance [1 / Bohr] or vector field [1 / Bohr^2] matrix and ``T`` is an
            assignment matrix that computes the partial charge on each atom as the sum
            of one or more charges from the charge vector.
        vsite_charge_assignment_matrix
            A matrix with shape=(n_vsites, n_vsite_charges) that that computes the
            partial charge on each virtual site as the sum of one or more charges from
            the a vector of virtual site charge parameters [e], i.e

            ``vsite_charges = vsite_charge_assignment_matrix @ v_site_charge_parameters``
        vsite_fixed_charges
            A vector with shape=(n_vsites, 1) of the partial charges assigned to each
            virtual site that will remain fixed during an optimization.
        vsite_coord_assignment_matrix
            A matrix with shape=(n_vsites, 3) of indices into a flat vector of virtual
            site local frame coordinate parameters that will be trained, such that

            ``vsite_local_coords == vsite_coord_params[vsite_coord_assignment_matrix]
                                    + vsite_fixed_coords``

            An entry of -1 indicates a value of ``0.0`` should be used.
        vsite_fixed_coords
            A vector with shape=(n_vsites, 3) of virtual site local frame coordinates
            virtual site that will remain fixed during an optimization, such that

            ``vsite_local_coords == vsite_coord_params[vsite_coord_assignment_matrix]
                                    + vsite_fixed_coords``
        vsite_local_coordinate_frame
            A tenser with shape=(4, n_vsites, 3) and units of [A] that stores the local
            frames of all virtual sites  whereby ``local_frames[0]`` is an array of the
            origins of each frame, ``local_frames[1]`` the x-directions,
            ``local_frames[2]`` the y-directions, and ``local_frames[2]`` the
            z-directions.
        grid_coordinates
            A matrix with shape=(n_grid_points, 3) and units of [A] of the grid
            coordinates that the electrostatic property will be evaluated on.
        reference_values
            A vector with shape=(n_grid_points, n_dim) of the reference values of the
            electrostatic property of interest evaluated on a grid of points.
        """

        # TODO: Proper v-site mutual exclusive exception.
        # TODO: Shape validation.
        assert (
            vsite_local_coordinate_frame is None
            and vsite_charge_assignment_matrix is None
            and vsite_fixed_charges is None
            and vsite_coord_assignment_matrix is None
            and vsite_fixed_coords is None
            and grid_coordinates is None
        ) or (
            vsite_local_coordinate_frame is not None
            and vsite_charge_assignment_matrix is not None
            and vsite_fixed_charges is not None
            and vsite_coord_assignment_matrix is not None
            and vsite_fixed_coords is not None
            and grid_coordinates is not None
        ), "all virtual site terms must be provided or none must be"

        self.atom_charge_design_matrix = atom_charge_design_matrix

        self.vsite_charge_assignment_matrix = vsite_charge_assignment_matrix
        self.vsite_fixed_charges = vsite_fixed_charges

        self.vsite_coord_assignment_matrix = vsite_coord_assignment_matrix
        self.vsite_fixed_coords = vsite_fixed_coords

        self.vsite_local_coordinate_frame = vsite_local_coordinate_frame

        self.grid_coordinates = grid_coordinates
        self.reference_values = reference_values

        self._grid_batches = None

    def to_backend(self, backend: Literal["numpy", "torch"]):
        """Converts the tensors associated with this term to a particular backend.

        Parameters
        ----------
        backend
            The backend to convert the tensors to.
        """
        converter = to_torch if backend == "torch" else to_numpy

        self.atom_charge_design_matrix = converter(self.atom_charge_design_matrix)

        self.vsite_charge_assignment_matrix = converter(
            self.vsite_charge_assignment_matrix
        )
        self.vsite_fixed_charges = converter(self.vsite_fixed_charges)

        self.vsite_coord_assignment_matrix = converter(
            self.vsite_coord_assignment_matrix
        )
        self.vsite_fixed_coords = converter(self.vsite_fixed_coords)

        self.vsite_local_coordinate_frame = converter(self.vsite_local_coordinate_frame)

        self.grid_coordinates = converter(self.grid_coordinates)
        self.reference_values = converter(self.reference_values)

    @classmethod
    def combine(cls: Type[_TERM_T], *terms: _TERM_T) -> _TERM_T:
        """Combines multiple objective term objects into a single object that can
        be evaluated more efficiently by stacking the cached terms in a way that
        allows vectorized operations.

        Notes
        -----
        * This feature is very experimental and should only be used if you know what
          you are doing.

        Parameters
        ----------
        terms
            The terms to combine.

        Returns
        -------
            The combined term.
        """

        if any(term._grid_batches is not None for term in terms):
            raise RuntimeError("Several of the terms have already been combined")

        return_value = cls(
            atom_charge_design_matrix=concatenate(
                *(term.atom_charge_design_matrix for term in terms)
            ),
            #
            vsite_charge_assignment_matrix=concatenate(
                *(term.vsite_charge_assignment_matrix for term in terms)
            ),
            vsite_fixed_charges=concatenate(
                *(term.vsite_fixed_charges for term in terms)
            ),
            #
            vsite_coord_assignment_matrix=concatenate(
                *(term.vsite_coord_assignment_matrix for term in terms)
            ),
            vsite_fixed_coords=concatenate(
                *(term.vsite_fixed_coords for term in terms)
            ),
            #
            vsite_local_coordinate_frame=concatenate(
                *(term.vsite_local_coordinate_frame for term in terms), dimension=1
            ),
            #
            grid_coordinates=concatenate(*(term.grid_coordinates for term in terms)),
            reference_values=concatenate(*(term.reference_values for term in terms)),
        )

        if return_value.grid_coordinates is not None:

            return_value._grid_batches = [(0, 0)]

            n_grid_points, n_vsites = 0, 0

            for term in terms:

                n_grid_points += len(term.grid_coordinates)
                n_vsites += len(term.vsite_fixed_charges)

                return_value._grid_batches.append((n_grid_points, n_vsites))

        return return_value

    def predict(
        self,
        charge_parameters: TensorType,
        vsite_coordinate_parameters: Optional[TensorType],
    ):
        """Predict the value of the electrostatic property of interest using the
        current values of the parameter.

        Parameters
        ----------
        charge_parameters
            A vector with shape=(n_bcc_charges + n_vsite_charges, 1) and units of [e]
            that contains the current values of both the BCC and virtual site charge
            increment parameters being trained.

            The ordering of the parameters should match the order used to generate
            the ``atom_charge_design_matrix``.
        vsite_coordinate_parameters
            A flat vector with shape=(n_bcc_charges + n_vsite_charges, 1)
            that contains the current values of both the virtual site coordinate
            parameters being trained.

            All distances should be in units of [A] and all angles in units of degrees.

            The ordering of the parameters should match the order used to generate
            the ``vsite_coord_assignment_matrix``.

        Returns
        -------
            The predicted value of the electrostatic property represented by this term
            with the same units and shape as ``reference_values``.
        """

        if (
            self.vsite_local_coordinate_frame is None
            and vsite_coordinate_parameters is not None
        ):

            raise RuntimeError(
                "Virtual site parameters were provided but this term was not set-up to"
                "handle such particles."
            )

        if self._objective()._flatten_charges():
            charge_parameters = charge_parameters.flatten()

        if self.atom_charge_design_matrix is not None:
            atom_contribution = self.atom_charge_design_matrix @ charge_parameters
        else:
            atom_contribution = 0.0

        if (
            self.vsite_local_coordinate_frame is not None
            and self.vsite_local_coordinate_frame.shape[1] > 0
        ):

            trainable_coordinates = append_zero(vsite_coordinate_parameters.flatten())[
                self.vsite_coord_assignment_matrix
            ]

            vsite_local_coordinates = self.vsite_fixed_coords + trainable_coordinates
            # noinspection PyTypeChecker
            vsite_coordinates = VirtualSiteGenerator.convert_local_coordinates(
                vsite_local_coordinates,
                self.vsite_local_coordinate_frame,
                backend="numpy"
                if isinstance(vsite_local_coordinates, numpy.ndarray)
                else "torch",
            )

            n_vsite_charges = self.vsite_charge_assignment_matrix.shape[1]

            vsite_charges = (
                self.vsite_charge_assignment_matrix
                @ charge_parameters[-n_vsite_charges:]
                + self.vsite_fixed_charges
            )

            if self._objective()._flatten_charges():
                vsite_charges = vsite_charges.flatten()

            if self._grid_batches is None or len(self._grid_batches) <= 1:

                design_matrix_precursor = (
                    self._objective()._compute_design_matrix_precursor(
                        self.grid_coordinates, vsite_coordinates
                    )
                )

                vsite_contribution = design_matrix_precursor @ vsite_charges

            else:

                vsite_contribution = concatenate(
                    *(
                        self._objective()._compute_design_matrix_precursor(
                            self.grid_coordinates[
                                self._grid_batches[i][0] : self._grid_batches[i + 1][0],
                                :,
                            ],
                            vsite_coordinates[
                                self._grid_batches[i][1] : self._grid_batches[i + 1][1],
                                :,
                            ],
                        )
                        @ vsite_charges[
                            self._grid_batches[i][1] : self._grid_batches[i + 1][1]
                        ]
                        for i in range(len(self._grid_batches) - 1)
                    )
                )

        else:
            vsite_contribution = 0.0

        return atom_contribution + vsite_contribution

    def loss(
        self,
        charge_parameters: TensorType,
        vsite_coordinate_parameters: Optional[TensorType],
    ) -> TensorType:
        """Evaluate the L2 loss function (i.e ``(target_values - predict(q, c)) ** 2)``
        using the current values of the parameters being trained.

        Parameters
        ----------
        charge_parameters
            A vector with shape=(n_bcc_charges + n_vsite_charges, 1) and units of [e]
            that contains the current values of both the BCC and virtual site charge
            increment parameters being trained.

            The ordering of the parameters should match the order used to generate
            the ``atom_charge_design_matrix``.
        vsite_coordinate_parameters
            A flat vector with shape=(n_bcc_charges + n_vsite_charges, 1)
            that contains the current values of both the virtual site coordinate
            parameters being trained.

            All distances should be in units of [A] and all angles in units of degrees.

            The ordering of the parameters should match the order used to generate
            the ``vsite_coord_assignment_matrix``.

        Returns
        -------
            The L2 loss function.
        """

        delta = self.reference_values - self.predict(
            charge_parameters, vsite_coordinate_parameters
        )

        return (delta * delta).sum()


class Objective(abc.ABC):
    """A utility class which contains helper functions for computing the
    contributions to a least squares objective function which captures the
    deviation of the ESP computed using molecular partial charges and the ESP
    computed by a QM calculation."""

    @classmethod
    @abc.abstractmethod
    def _objective_term(cls) -> Type[ObjectiveTerm]:
        """The objective term class associated with this objective."""
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def _flatten_charges(cls) -> bool:
        """Whether to operate on a flattened array of charges rather than the
        2D array. This is mainly to be used when the design matrix is a tensor."""
        raise NotImplementedError()

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
            The grid coordinates which the electronic property was computed on in
            units of Angstrom.
        conformer
            The coordinates of the molecule conformer the property was computed for in
            units of Angstrom.
        """

    @classmethod
    @abc.abstractmethod
    def _electrostatic_property(cls, record: MoleculeESPRecord) -> numpy.ndarray:
        """Returns the value of the electronic property being refit from an ESP
        record.

        Parameters
        ----------
        record
            The record which stores the electronic property being optimized against.
        """

    @classmethod
    def _compute_library_charge_terms(
        cls,
        molecule: "Molecule",
        charge_collection: LibraryChargeCollection,
        charge_parameter_keys: List[Tuple[str, Tuple[int, ...]]],
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:

        assignment_matrix = LibraryChargeGenerator.build_assignment_matrix(
            molecule, charge_collection
        )

        flat_collection_keys = [
            (parameter.smiles, i)
            for parameter in charge_collection.parameters
            for i in range(len(parameter.value))
        ]
        flat_collection_values = [
            charge
            for parameter in charge_collection.parameters
            for charge in parameter.value
        ]

        trainable_parameter_indices = [
            flat_collection_keys.index((smirks, i))
            for smirks, indices in charge_parameter_keys
            for i in indices
        ]
        fixed_parameter_indices = [
            i
            for i in range(len(flat_collection_keys))
            if i not in trainable_parameter_indices
        ]

        fixed_assignment_matrix = assignment_matrix[:, fixed_parameter_indices]
        fixed_charge_values = numpy.array(
            [[flat_collection_values[index]] for index in fixed_parameter_indices]
        )

        fixed_charges = fixed_assignment_matrix @ fixed_charge_values

        trainable_assignment_matrix = assignment_matrix[:, trainable_parameter_indices]

        return trainable_assignment_matrix, fixed_charges.reshape(-1, 1)

    @classmethod
    def _compute_bcc_charge_terms(
        cls,
        molecule: "Molecule",
        bcc_collection: BCCCollection,
        bcc_parameter_keys: List[str],
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:

        flat_collection_keys = [
            parameter.smirks for parameter in bcc_collection.parameters
        ]

        trainable_parameter_indices = [
            flat_collection_keys.index(key) for key in bcc_parameter_keys
        ]
        fixed_parameter_indices = [
            i
            for i in range(len(bcc_collection.parameters))
            if i not in trainable_parameter_indices
        ]

        assignment_matrix = BCCGenerator.build_assignment_matrix(
            molecule, bcc_collection
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
    def _compute_vsite_coord_terms(
        cls,
        molecule: "Molecule",
        conformer: numpy.ndarray,
        vsite_collection: VirtualSiteCollection,
        vsite_coordinate_parameter_keys: List[VirtualSiteGeometryKey],
    ) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:

        parameters_by_key = {
            (parameter.smirks, parameter.type, parameter.name): parameter
            for parameter in vsite_collection.parameters
        }

        _, assigned_parameter_map = VirtualSiteGenerator._apply_virtual_sites(
            molecule, vsite_collection
        )

        if len(assigned_parameter_map) == 0:

            return (
                numpy.zeros((0, 3)),
                numpy.zeros((0, 3)),
                numpy.zeros((4, 0, 3)),
            )

        assigned_parameters = defaultdict(list)

        for _, parameter_keys in assigned_parameter_map.items():
            for parameter_key, orientations in parameter_keys.items():
                assigned_parameters[parameter_key].extend(orientations)

        assigned_parameters = [
            (parameters_by_key[parameter_key], orientations)
            for parameter_key, orientations in assigned_parameters.items()
        ]

        local_coordinate_parameters = []
        local_coordinate_indices = []

        for atom_indices, parameter_key, parameter in [
            (orientation, parameter_key, parameters_by_key[parameter_key])
            for parent_atom_index, parameter_keys in assigned_parameter_map.items()
            for parameter_key, orientations in parameter_keys.items()
            for orientation in orientations
        ]:

            local_coordinate_parameters.append(
                [
                    0.0
                    if (*parameter_key, attribute) in vsite_coordinate_parameter_keys
                    else parameter.local_frame_coordinates[0, i]
                    for i, attribute in enumerate(_VSITE_ATTRIBUTES)
                ]
            )
            # noinspection PyTypeChecker
            local_coordinate_indices.append(
                [
                    -1
                    if (*parameter_key, attribute)
                    not in vsite_coordinate_parameter_keys
                    else vsite_coordinate_parameter_keys.index(
                        (*parameter_key, attribute)
                    )
                    for attribute in _VSITE_ATTRIBUTES
                ]
            )

        local_coordinate_frame = VirtualSiteGenerator.build_local_coordinate_frames(
            conformer, assigned_parameters
        )

        # noinspection PyTypeChecker
        return (
            numpy.array(local_coordinate_indices),
            numpy.array(local_coordinate_parameters),
            local_coordinate_frame,
        )

    @classmethod
    def _compute_vsite_charge_terms(
        cls,
        molecule: "Molecule",
        vsite_collection: VirtualSiteCollection,
        vsite_charge_parameter_keys: List[VirtualSiteChargeKey],
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:

        flat_collection_keys = [
            (parameter.smirks, parameter.type, parameter.name, i)
            for parameter in vsite_collection.parameters
            for i in range(len(parameter.charge_increments))
        ]
        flat_collection_values = [
            charge
            for parameter in vsite_collection.parameters
            for charge in parameter.charge_increments
        ]

        trainable_parameter_indices = [
            flat_collection_keys.index(key) for key in vsite_charge_parameter_keys
        ]
        fixed_parameter_indices = [
            i
            for i in range(len(flat_collection_keys))
            if i not in trainable_parameter_indices
        ]

        assignment_matrix = VirtualSiteGenerator.build_charge_assignment_matrix(
            molecule, vsite_collection
        )

        fixed_assignment_matrix = assignment_matrix[:, fixed_parameter_indices]
        fixed_charge_values = numpy.array(
            [[flat_collection_values[index]] for index in fixed_parameter_indices]
        )

        fixed_charges = fixed_assignment_matrix @ fixed_charge_values

        trainable_assignment_matrix = assignment_matrix[:, trainable_parameter_indices]

        return trainable_assignment_matrix, fixed_charges.reshape(-1, 1)

    @classmethod
    def compute_objective_terms(
        cls,
        esp_records: List[MoleculeESPRecord],
        charge_collection: Optional[
            Union[QCChargeSettings, LibraryChargeCollection]
        ] = None,
        charge_parameter_keys: Optional[List[Tuple[str, Tuple[int, ...]]]] = None,
        bcc_collection: Optional[BCCCollection] = None,
        bcc_parameter_keys: Optional[List[str]] = None,
        vsite_collection: Optional[VirtualSiteCollection] = None,
        vsite_charge_parameter_keys: Optional[List[VirtualSiteChargeKey]] = None,
        vsite_coordinate_parameter_keys: Optional[List[VirtualSiteGeometryKey]] = None,
    ) -> Generator[ObjectiveTerm, None, None]:
        """Pre-calculates the terms that contribute to the total objective function.

        This function assumes that the array (/tensor) of values to train will have shape
        (n_charge_parameter_keys + n_bcc_parameter_keys + vsite_charge_parameter_keys, 1)
        with the values in the array corresponding to the values pointed to by the keys
        starting with library charge values (if any), followed by BCCs (if any) and
        finally any v-site charge increments (if any). See the ``vectorize`` method of
        the collections for an easy way to generate such an array.

        Notes
        -----
        It is critical that the order of the values of the array match the order of the
        keys provided here otherwise the wrong parameters will be applied to the wrong
        atoms.

        Parameters
        ----------
        esp_records
            The calculated records that contain the reference ESP and electric field
            data to train against.
        charge_collection
            Optionally either i) a collection settings that define how to compute a
            set of base charges (e.g. AM1) or ii) a collection of library charges to
            be applied. This base charges may be perturbed by any supplied
            ``bcc_collection`` or ``vsite_collection``. If no value is provided, all
            base charges will be set to zero.
        charge_parameter_keys
            A list of tuples of the form ``(smiles, (idx_0, ...))`` that define those
            parameters in the ``charge_collection`` that should be trained.

            Here ``idx_i`` is an index into the ``value`` field of the parameter uniquely
            identified by the ``smiles`` key.

            This argument can only be used when the ``charge_collection`` is a library
            charge collection.

            The order of these keys **must** match the order of the charges in the
            vector of charges being trained. See for e.g.
            ``LibraryChargeCollection.vectorize``.
        bcc_collection
            A collection of bond charge correction parameters that should perturb the
            base set of charges for each molecule in the ``esp_records`` list.
        bcc_parameter_keys
            A list of SMIRKS patterns that define those parameters in the
            ``bcc_collection`` that should be trained.

            The order of these keys **must** match the order of the charges in the
            vector of charges being trained. See for e.g. ``BCCCollection.vectorize``.
        vsite_collection
            A collection of virtual site parameters that should create virtual sites
            for each molecule in the ``esp_records`` list and perturb the base charges
            on each atom.
        vsite_charge_parameter_keys
            A list of tuples of the form ``(smirks, type, name, idx)`` that define
            those charge increment parameters in the ``vsite_collection`` that should be
            trained.

            Here ``idx`` is an index into the ``charge_increments`` field of the
            parameter uniquely identified by the other terms of the key.

            The order of these keys **must** match the order of the charges in the
            vector of charges being trained. See for e.g.
            ``VirtualSiteCollection.vectorize_charge_increments``.
        vsite_coordinate_parameter_keys
            A list of tuples of the form ``(smirks, type, name, attr)``) that define
            those local frame coordinate parameters in the ``vsite_collection`` that
            should be trained.

            Here ``attr`` should be one of ``{'distance', 'in_plane_angle',
            'out_of_plane_angle'}``.

            The order of these keys **must** match the order of the charges in the
            vector of charges being trained. See for e.g.
            ``VirtualSiteCollection.vectorize_coordinates``.

        Returns
        -------
            The precalculated terms which may be used to compute the full
            contribution to the objective function.
        """

        from openff.toolkit.topology import Molecule

        for esp_record in esp_records:

            molecule: Molecule = Molecule.from_mapped_smiles(
                esp_record.tagged_smiles, allow_undefined_stereo=True
            )
            ordered_conformer = esp_record.conformer

            fixed_atom_charges = numpy.zeros((molecule.n_atoms, 1))

            # Pre-compute the design matrix precursor (e.g inverse distance matrix for
            # ESP) for this molecule as this is an 'expensive' step in the optimization.
            design_matrix_precursor = cls._compute_design_matrix_precursor(
                esp_record.grid_coordinates, ordered_conformer
            )

            (
                atom_charge_design_matrices,
                vsite_charge_assignment_matrix,
                vsite_fixed_charges,
                vsite_coord_assignment_matrix,
                vsite_fixed_coords,
                vsite_local_coordinate_frame,
                grid_coordinates,
            ) = ([], None, None, None, None, None, None)

            if charge_collection is None:
                pass
            elif isinstance(charge_collection, QCChargeSettings):
                assert (
                    charge_parameter_keys is None
                ), "charges generated using `QCChargeSettings` cannot be trained"

                fixed_atom_charges += QCChargeGenerator.generate(
                    molecule, [ordered_conformer * unit.angstrom], charge_collection
                )

            elif isinstance(charge_collection, LibraryChargeCollection):

                (
                    library_assignment_matrix,
                    library_fixed_charges,
                ) = cls._compute_library_charge_terms(
                    molecule,
                    charge_collection,
                    charge_parameter_keys,
                )

                fixed_atom_charges += library_fixed_charges

                atom_charge_design_matrices.append(
                    design_matrix_precursor @ library_assignment_matrix
                )
            else:
                raise NotImplementedError()

            if bcc_collection is not None:

                (
                    bcc_assignment_matrix,
                    bcc_fixed_charges,
                ) = cls._compute_bcc_charge_terms(
                    molecule, bcc_collection, bcc_parameter_keys
                )

                fixed_atom_charges += bcc_fixed_charges

                atom_charge_design_matrices.append(
                    design_matrix_precursor @ bcc_assignment_matrix
                )

            if vsite_collection is not None:

                (
                    vsite_charge_assignment_matrix,
                    vsite_fixed_charges,
                ) = cls._compute_vsite_charge_terms(
                    molecule, vsite_collection, vsite_charge_parameter_keys
                )
                (
                    vsite_coord_assignment_matrix,
                    vsite_fixed_coords,
                    vsite_local_coordinate_frame,
                ) = cls._compute_vsite_coord_terms(
                    molecule,
                    ordered_conformer,
                    vsite_collection,
                    vsite_coordinate_parameter_keys or [],
                )

                fixed_atom_charges += vsite_fixed_charges[: molecule.n_atoms]
                vsite_fixed_charges = vsite_fixed_charges[molecule.n_atoms :]

                atom_charge_assignment_matrix = vsite_charge_assignment_matrix[
                    : molecule.n_atoms
                ]
                atom_charge_design_matrices.append(
                    design_matrix_precursor @ atom_charge_assignment_matrix
                )

                vsite_charge_assignment_matrix = vsite_charge_assignment_matrix[
                    molecule.n_atoms :
                ]

                grid_coordinates = esp_record.grid_coordinates

            atom_charge_design_matrix = (
                None
                if len(atom_charge_design_matrices) == 0
                else numpy.concatenate(atom_charge_design_matrices, axis=-1)
            )

            if cls._flatten_charges():
                fixed_atom_charges = fixed_atom_charges.flatten()

            reference_values = (
                cls._electrostatic_property(esp_record)
                - design_matrix_precursor @ fixed_atom_charges
            )

            yield cls._objective_term()(
                atom_charge_design_matrix,
                vsite_charge_assignment_matrix,
                vsite_fixed_charges,
                vsite_coord_assignment_matrix,
                vsite_fixed_coords,
                vsite_local_coordinate_frame,
                grid_coordinates,
                reference_values,
            )


class ESPObjectiveTerm(ObjectiveTerm):
    """A class that stores precalculated values used to compute the difference between
    a reference set of electrostatic potentials and a set computed using a set of fixed
    partial charges.

    See the ``predict`` and ``loss`` functions for more details.
    """

    @classmethod
    def _objective(cls) -> Type["ESPObjective"]:
        return ESPObjective


class ESPObjective(Objective):
    """A utility class which contains helper functions for computing the
    contributions to a least squares objective function which captures the
    deviation of the ESP computed using molecular partial charges and the ESP
    computed by a QM calculation."""

    @classmethod
    def _objective_term(cls) -> Type[ESPObjectiveTerm]:
        return ESPObjectiveTerm

    @classmethod
    def _flatten_charges(cls) -> bool:
        return False

    @classmethod
    def _compute_design_matrix_precursor(
        cls, grid_coordinates: numpy.ndarray, conformer: numpy.ndarray
    ):
        # Pre-compute the inverse distance between each atom in the molecule
        # and each grid point.
        inverse_distance_matrix = compute_inverse_distance_matrix(
            grid_coordinates, conformer
        )
        # Care must be taken to ensure that length units are converted from [Angstrom]
        # to [Bohr].
        inverse_distance_matrix = unit.convert(
            inverse_distance_matrix, unit.angstrom**-1, unit.bohr**-1
        )

        return inverse_distance_matrix

    @classmethod
    def _electrostatic_property(cls, record: MoleculeESPRecord) -> numpy.ndarray:
        return record.esp


class ElectricFieldObjectiveTerm(ObjectiveTerm):
    """A class that stores precalculated values used to compute the difference between
    a reference set of electric field vectors and a set computed using a set of fixed
    partial charges.

    See the ```predict`` and ``loss`` functions for more details.
    """

    @classmethod
    def _objective(cls) -> Type["ElectricFieldObjective"]:
        return ElectricFieldObjective


class ElectricFieldObjective(Objective):
    """A utility class which contains helper functions for computing the
    contributions to a least squares objective function which captures the
    deviation of the electric field computed using molecular partial charges and the
    electric field computed by a QM calculation."""

    @classmethod
    def _objective_term(cls) -> Type[ElectricFieldObjectiveTerm]:
        return ElectricFieldObjectiveTerm

    @classmethod
    def _flatten_charges(cls) -> bool:
        return True

    @classmethod
    def _compute_design_matrix_precursor(
        cls, grid_coordinates: numpy.ndarray, conformer: numpy.ndarray
    ):
        vector_field = compute_vector_field(
            unit.convert(conformer, unit.angstrom, unit.bohr),
            unit.convert(grid_coordinates, unit.angstrom, unit.bohr),
        )

        return vector_field

    @classmethod
    def _electrostatic_property(cls, record: MoleculeESPRecord) -> numpy.ndarray:
        return record.electric_field
