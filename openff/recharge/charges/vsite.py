"""Generate virtual sites for molecules from a collection of virtual site parameters."""

import abc
import copy
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union, overload

import numpy
from openff.units import unit
from openff.utilities import requires_package
from pydantic import BaseModel, Field, constr, validator

from openff.recharge.aromaticity import AromaticityModels
from openff.recharge.charges.exceptions import ChargeAssignmentError

if TYPE_CHECKING:
    try:
        import torch
    except ImportError:
        torch = None

    from openff.toolkit import Molecule
    from openff.toolkit.typing.engines.smirnoff import VirtualSiteHandler

ExclusionPolicy = Literal["none", "parents"]

VirtualSiteKey = Tuple[str, str, str]
VirtualSiteChargeKey = Tuple[str, str, str, int]
VirtualSiteGeometryKey = Tuple[
    str, str, str, Literal["distance", "in_plane_angle", "out_of_plane_angle"]
]

_DEGREES_TO_RADIANS = numpy.pi / 180.0


class _VirtualSiteParameter(BaseModel, abc.ABC):
    """The base class for virtual site parameters."""

    type: Literal["base-virtual-site"]

    smirks: constr(min_length=1) = Field(
        ...,
        description="A SMIRKS pattern that encodes the chemical environment that "
        "this parameter should be applied to.",
    )
    name: Optional[str] = Field(
        None, description="An optional name associated with this virtual site."
    )

    distance: float = Field(
        ...,
        description="The distance to place the virtual site along its associated basis.",
    )
    charge_increments: Tuple[float, ...] = Field(
        ...,
        description="The amount of charge [e] to be transferred from the virtual site "
        "to each tagged atom that forms the basis for the virtual site.",
    )

    sigma: float = Field(
        ..., description="The LJ sigma parameter [A] associated with the virtual site."
    )
    epsilon: float = Field(
        ...,
        description="The LJ espilon [kJ / mol] parameter associated with the virtual "
        "site.",
    )

    match: Literal["once", "all-permutations"] = Field(..., description="...")

    @classmethod
    @abc.abstractmethod
    def local_frame_weights(cls) -> numpy.ndarray:
        """Returns a matrix of the weights to apply to a matrix of the coordinates of
        the virtual sites' parent atoms to yield the origin, x and y vectors of the
        virtual sites local frame with shape=(3, n_parent_atoms)."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def local_frame_coordinates(self) -> numpy.ndarray:
        """Returns a 1 X 3 array of the spherical coordinates (``[d, theta, phi]``) of
        this virtual site with respect to its local frame.
        """
        raise NotImplementedError()


class BondChargeSiteParameter(_VirtualSiteParameter):
    type: Literal["BondCharge"] = "BondCharge"

    @classmethod
    def local_frame_weights(cls) -> numpy.ndarray:
        return numpy.array([[1.0, 0.0], [-1.0, 1.0], [-1.0, 1.0]])

    @property
    def local_frame_coordinates(self) -> numpy.ndarray:
        # distance, theta, phi
        return numpy.array([[self.distance, 180.0, 0.0]])


class MonovalentLonePairParameter(_VirtualSiteParameter):
    type: Literal["MonovalentLonePair"] = "MonovalentLonePair"

    in_plane_angle: float = Field(
        ...,
        description="The angle [deg] to move the virtual site in the plane defined "
        "by the tagged atoms by.",
    )
    out_of_plane_angle: float = Field(
        ...,
        description="The angle [deg] to move the virtual site out of the plane "
        "defined by the tagged atoms by.",
    )

    @classmethod
    def local_frame_weights(cls) -> numpy.ndarray:
        return numpy.array([[1.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])

    @property
    def local_frame_coordinates(self) -> numpy.ndarray:
        # distance, theta, phi
        return numpy.array(
            [[self.distance, self.in_plane_angle, self.out_of_plane_angle]]
        )


class DivalentLonePairParameter(_VirtualSiteParameter):
    type: Literal["DivalentLonePair"] = "DivalentLonePair"

    out_of_plane_angle: float = Field(
        ...,
        description="The angle [deg] to move the virtual site out of the plane "
        "defined by the tagged atoms by.",
    )

    @classmethod
    def local_frame_weights(cls) -> numpy.ndarray:
        return numpy.array([[1.0, 0.0, 0.0], [-1.0, 0.5, 0.5], [-1.0, 1.0, 0.0]])

    @property
    def local_frame_coordinates(self) -> numpy.ndarray:
        # distance, theta, phi
        return numpy.array([[self.distance, 180.0, self.out_of_plane_angle]])


class TrivalentLonePairParameter(_VirtualSiteParameter):
    type: Literal["TrivalentLonePair"] = "TrivalentLonePair"

    @classmethod
    def local_frame_weights(cls) -> numpy.ndarray:
        return numpy.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [-1.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                [-1.0, 1.0, 0.0, 0.0],
            ]
        )

    @property
    def local_frame_coordinates(self) -> numpy.ndarray:
        # distance, theta, phi
        return numpy.array([[self.distance, 180.0, 0.0]])


VirtualSiteParameterType = Union[
    BondChargeSiteParameter,
    MonovalentLonePairParameter,
    DivalentLonePairParameter,
    TrivalentLonePairParameter,
]


class VirtualSiteCollection(BaseModel):
    """A collection of virtual site parameters that are based off of the SMIRNOFF
    specification."""

    parameters: List[VirtualSiteParameterType] = Field(
        ...,
        description="The virtual site parameters to apply.",
    )
    aromaticity_model: AromaticityModels = Field(
        AromaticityModels.MDL,
        description="The model to use when assigning aromaticity.",
    )

    exclusion_policy: ExclusionPolicy = Field("parents", description="...")

    @validator("aromaticity_model")
    def validate_aromaticity_model(cls, value):
        assert value == AromaticityModels.MDL, "only MDL aromaticity model is supported"

    def to_smirnoff(self) -> "VirtualSiteHandler":
        """Converts this collection of virtual site parameters to a SMIRNOFF virtual
        site parameter handler.

        Returns
        -------
            The constructed parameter handler.
        """
        from openff.toolkit.typing.engines.smirnoff import VirtualSiteHandler

        # noinspection PyTypeChecker
        parameter_handler = VirtualSiteHandler(
            version="0.3", exclusion_policy=self.exclusion_policy
        )

        for parameter in reversed(self.parameters):
            parameter_kwargs = dict(
                smirks=parameter.smirks,
                type=parameter.type,
                name=parameter.name,
                distance=parameter.distance * unit.angstrom,
                charge_increment=[
                    charge * unit.elementary_charge
                    for charge in parameter.charge_increments
                ],
                sigma=parameter.sigma * unit.angstrom,
                epsilon=parameter.epsilon * unit.kilojoules_per_mole,
                match=parameter.match.replace("-", "_").lower(),
            )

            if parameter.type == "MonovalentLonePair":
                parameter_kwargs["outOfPlaneAngle"] = (
                    parameter.out_of_plane_angle * unit.degrees
                )
                parameter_kwargs["inPlaneAngle"] = (
                    parameter.in_plane_angle * unit.degrees
                )

            elif parameter.type == "DivalentLonePair":
                parameter_kwargs["outOfPlaneAngle"] = (
                    parameter.out_of_plane_angle * unit.degrees
                )

            parameter_handler.add_parameter(parameter_kwargs=parameter_kwargs)

        return parameter_handler

    @classmethod
    @requires_package("openmm")
    def from_smirnoff(
        cls,
        parameter_handler: "VirtualSiteHandler",
        aromaticity_model=AromaticityModels.MDL,
    ) -> "VirtualSiteCollection":
        """Attempts to convert a SMIRNOFF virtual site parameter handler to a virtual
        site parameter collection.

        Parameters
        ----------
        parameter_handler
            The parameter handler to convert.
        aromaticity_model
            The model which describes how aromaticity should be assigned
            when applying the virtual site correction parameters.

        Returns
        -------
            The converted virtual site collection.
        """

        assert (
            aromaticity_model == AromaticityModels.MDL
        ), "only MDL aromaticity model is supported"

        parameters = []

        for smirnoff_parameter in reversed(parameter_handler.parameters):
            base_kwargs = dict(
                smirks=smirnoff_parameter.smirks,
                name=smirnoff_parameter.name,
                distance=smirnoff_parameter.distance.m_as(unit.angstrom),
                charge_increments=tuple(
                    charge.m_as(unit.elementary_charge)
                    for charge in smirnoff_parameter.charge_increment
                ),
                sigma=smirnoff_parameter.sigma.m_as(unit.angstrom),
                epsilon=smirnoff_parameter.epsilon.m_as(unit.kilojoules_per_mole),
                match=smirnoff_parameter.match.replace("_", "-").lower(),
            )

            if smirnoff_parameter.type == "BondCharge":
                parameter = BondChargeSiteParameter(**base_kwargs)

            elif smirnoff_parameter.type == "MonovalentLonePair":
                parameter = MonovalentLonePairParameter(
                    **base_kwargs,
                    out_of_plane_angle=smirnoff_parameter.outOfPlaneAngle.m_as(
                        unit.degrees
                    ),
                    in_plane_angle=smirnoff_parameter.inPlaneAngle.m_as(unit.degrees),
                )

            elif smirnoff_parameter.type == "DivalentLonePair":
                parameter = DivalentLonePairParameter(
                    **base_kwargs,
                    out_of_plane_angle=smirnoff_parameter.outOfPlaneAngle.m_as(
                        unit.degrees
                    ),
                )

            elif smirnoff_parameter.type == "TrivalentLonePair":
                parameter = TrivalentLonePairParameter(**base_kwargs)

            else:
                raise NotImplementedError()

            parameters.append(parameter)

        return VirtualSiteCollection(
            parameters=parameters,
            aromaticity_model=aromaticity_model,
            exclusion_policy=parameter_handler.exclusion_policy.lower(),
        )

    def vectorize_coordinates(
        self, parameter_keys: List[VirtualSiteGeometryKey]
    ) -> numpy.ndarray:
        """Returns a flat vector of the local frame coordinate values associated with a
        specified set of 'keys'.

        Parameters
        ----------
        parameter_keys
            A list of parameter 'keys' of the form ``(smirks, type, name, attr)`` that
            specify which local frame coordinate to include in the returned vector.

            The allowed attributes are ``distance``, ``in_plane_angle``,
            ``out_of_plane_angle``

        Returns
        -------
            A vector of local frame coordinate with shape=(n_keys, 1)
        """

        parameters_by_key = {
            (parameter.smirks, parameter.type, parameter.name): parameter
            for parameter in self.parameters
        }

        parameter_values = numpy.array(
            [
                [getattr(parameters_by_key[tuple(parameter_key)], attribute)]
                for *parameter_key, attribute in parameter_keys
            ]
        )

        return parameter_values

    def vectorize_charge_increments(
        self, parameter_keys: List[VirtualSiteChargeKey]
    ) -> numpy.ndarray:
        """Returns a flat vector of the charge increment values associated with a
        specified set of 'keys'.

        Parameters
        ----------
        parameter_keys
            A list of parameter 'keys' of the form ``(smirks, type, name, idx)`` that
            specify which charge increments to include in the returned vector, where
            `idx` is an integer index into a parameters' ``charge_increments`` tuple.

        Returns
        -------
            A vector of charge increments with shape=(n_keys, 1)
        """

        parameters_by_key = {
            (parameter.smirks, parameter.type, parameter.name): parameter
            for parameter in self.parameters
        }

        return numpy.array(
            [
                [
                    parameters_by_key[tuple(parameter_key)].charge_increments[
                        charge_index
                    ]
                ]
                for *parameter_key, charge_index in parameter_keys
            ]
        )


class VirtualSiteGenerator:
    @classmethod
    def _apply_virtual_sites(
        cls, molecule: "Molecule", vsite_collection: VirtualSiteCollection
    ) -> Tuple["Molecule", Dict[int, Dict[VirtualSiteKey, List[Tuple[int, ...]]]]]:
        """Applies a virtual site collection to a molecule.

        Parameters
        ----------
        molecule
            The molecule to build the virtual sites for.
        vsite_collection
            The v-site collection to use to create the virtual sites.

        Returns
        -------
            An OpenFF molecule with virtual sites as well as a dictionary that maps each
            virtual site back to the parameter that yielded it.
        """

        parameter_handler = vsite_collection.to_smirnoff()

        off_topology = copy.deepcopy(molecule).to_topology()
        parameter_handler.create_openff_virtual_sites(off_topology)

        vsite_matches = parameter_handler.find_matches(off_topology)
        assigned_vsite_keys = defaultdict(lambda: defaultdict(list))

        for vsite_match in vsite_matches:
            vsite_parameter = vsite_match.parameter_type
            vsite_key = (
                vsite_parameter.smirks,
                vsite_parameter.type,
                vsite_parameter.name,
            )

            parent_atom_index = vsite_match.environment_match.reference_atom_indices[
                vsite_parameter.parent_index
            ]

            assigned_vsite_keys[parent_atom_index][vsite_key].append(
                vsite_match.environment_match.reference_atom_indices
            )

        off_molecule = next(off_topology.reference_molecules)

        return off_molecule, {
            parent_atom_index: {
                key: [*orientations] for key, orientations in keys.items()
            }
            for parent_atom_index, keys in assigned_vsite_keys.items()
        }

    @classmethod
    def _build_charge_increment_array(
        cls, vsite_collection: VirtualSiteCollection
    ) -> Tuple[numpy.ndarray, List[VirtualSiteChargeKey]]:
        """Returns a flat vector of the charge increments contained within a virtual site
        collection as well as a list of keys that map each value back to its original
        parameter.

        Parameters
        ----------
        vsite_collection
            The collection containing the v-site parameters.

        Returns
        -------
            ...
        """

        charge_values = []
        charge_keys = []

        for parameter in vsite_collection.parameters:
            for i, charge_increment in enumerate(parameter.charge_increments):
                charge_values.append(charge_increment)

                charge_keys.append(
                    (parameter.smirks, parameter.type, parameter.name, i)
                )

        return numpy.array(charge_values), charge_keys

    @classmethod
    def _validate_charge_assignment_matrix(
        cls,
        assignment_matrix: numpy.ndarray,
    ):
        """Validates the charge increment assignment matrix.

        Parameters
        ----------
        assignment_matrix
            The assignment matrix to validate with
            shape=(n_atoms + n_vsites, n_charge_increments)
        """

        non_zero_assignments = assignment_matrix.sum(axis=0).astype(bool)

        if non_zero_assignments.any():
            raise ChargeAssignmentError(
                "An internal error occurred. The v-site charge increments alter the "
                "total charge of the molecule"
            )

    @classmethod
    def build_charge_assignment_matrix(
        cls,
        molecule: "Molecule",
        vsite_collection: VirtualSiteCollection,
    ) -> numpy.ndarray:
        """Generates a matrix that specifies which v-site charge increments have been
        applied to which atoms in the molecule.

        The matrix takes the form ...

        Parameters
        ----------
        molecule
            The molecule to assign the v-site charge increments to.
        vsite_collection
            The v-site parameters that may be assigned.

        Returns
        -------
            The assignment matrix with shape=(n_atoms + n_vsites, n_charge_increments)
            where ...
        """

        from openff.toolkit.typing.engines.smirnoff import VirtualSiteHandler

        off_molecule, assigned_vsite_keys = cls._apply_virtual_sites(
            molecule, vsite_collection
        )

        _, all_vsite_keys = cls._build_charge_increment_array(vsite_collection)

        n_particles = off_molecule.n_particles
        n_corrections = len(all_vsite_keys)

        assignment_matrix = numpy.zeros((n_particles, n_corrections))

        for vsite_particle in [
            vsite_particle
            for vsite in off_molecule.virtual_sites
            for vsite_particle in vsite.particles
        ]:
            vsite_index = vsite_particle.molecule_particle_index

            type_parent_index = VirtualSiteHandler.VirtualSiteType.type_to_parent_index(
                vsite_particle.virtual_site.type
            )
            parent_atom_index = vsite_particle.orientation[type_parent_index]

            vsite_parameter_keys = assigned_vsite_keys[parent_atom_index]

            for vsite_parameter_key in vsite_parameter_keys:
                for i, atom_index in enumerate(vsite_particle.orientation):
                    # noinspection PyTypeChecker
                    vsite_parameter_index = all_vsite_keys.index(
                        (*vsite_parameter_key, i)
                    )

                    assignment_matrix[atom_index, vsite_parameter_index] += 1
                    assignment_matrix[vsite_index, vsite_parameter_index] -= 1

        cls._validate_charge_assignment_matrix(assignment_matrix)
        return assignment_matrix

    @classmethod
    def apply_charge_assignment_matrix(
        cls,
        assignment_matrix: numpy.ndarray,
        vsite_collection: VirtualSiteCollection,
    ) -> numpy.ndarray:
        """Applies an assignment matrix to a list of virtual site parameters to yield the
        final charges increments due to the virtual sites for a molecule.

        Parameters
        ----------
        assignment_matrix
            The virtual site charge increment assignment matrix constructed using
            ``build_charge_assignment_matrix`` that describes how the virtual site
            charge increments should be applied. This should have
            shape=(n_atoms + n_vsites, n_charge_increments)
        vsite_collection
            The virtual site parameters that may be assigned.

        Returns
        -------
            The charge increments with shape=(n_atoms + n_vsites, 1).
        """

        correction_values, _ = cls._build_charge_increment_array(vsite_collection)
        charge_corrections = assignment_matrix @ correction_values

        if not numpy.isclose(charge_corrections.sum(), 0.0):
            raise ChargeAssignmentError(
                "An internal error occurred. The bond charge corrections were applied "
                "in such a way so that the total charge of the molecule will be "
                "altered."
            )

        return charge_corrections.reshape(-1, 1)

    @classmethod
    def generate_charge_increments(
        cls,
        molecule: "Molecule",
        vsite_collection: VirtualSiteCollection,
    ) -> numpy.ndarray:
        """Generate a set of charge increments due to virtual sites for a molecule.

        Parameters
        ----------
        molecule
            The molecule to generate the charge increments for.
        vsite_collection
            The virtual site parameters that may be assigned.

        Returns
        -------
            The charge increments with shape=(n_atoms + n_vsites, 1) that should be
            applied to the molecule.
        """

        assignment_matrix = cls.build_charge_assignment_matrix(
            molecule, vsite_collection
        )

        generated_corrections = cls.apply_charge_assignment_matrix(
            assignment_matrix, vsite_collection
        )

        return generated_corrections

    @classmethod
    def build_local_coordinate_frames(
        cls,
        conformer: numpy.ndarray,
        assigned_parameters: List[
            Tuple[VirtualSiteParameterType, List[Tuple[int, ...]]]
        ],
    ) -> numpy.ndarray:
        """Builds an orthonormal coordinate frame for each virtual particle
        based on the type of virtual site and the coordinates of the parent atoms.

        Notes
        -----
        * See `the OpenMM documentation for further information
          <http://docs.openmm.org/7.0.0/userguide/theory.html#virtual-sites>`_.

        Parameters
        ----------
        conformer
            The conformer of the molecule that the virtual sites are being added to
            with shape=(n_atoms, 3) and units of [A].
        assigned_parameters
            A dictionary of the form ``assigned_parameters[atom_indices] = parameters``
            where ``atom_indices`` is a tuple of indices corresponding to the atoms
            that the virtual site is orientated on, and ``parameters`` is a list of the
            parameters that describe the virtual sites.

        Returns
        -------
            An array storing the local frames of all virtual sites with
            shape=(4, n_vsites, 3) whereby ``local_frames[0]`` is an array of the
            origins of each frame, ``local_frames[1]`` the x-directions,
            ``local_frames[2]`` the y-directions, and ``local_frames[2]`` the
            z-directions.
        """

        stacked_frames = [[], [], [], []]

        for orientation, vsite_parameter in (
            (orientation, vsite_parameter)
            for vsite_parameter, orientations in assigned_parameters
            for orientation in orientations
        ):
            parent_coordinates = conformer[orientation, :]

            weighted_coordinates = (
                vsite_parameter.local_frame_weights() @ parent_coordinates
            )

            origin = weighted_coordinates[0, :]

            xy_plane = weighted_coordinates[1:, :]
            xy_plane_norm = xy_plane / numpy.sqrt(
                (xy_plane * xy_plane).sum(-1)
            ).reshape(-1, 1)

            x_hat = xy_plane_norm[0, :]
            z_hat = numpy.cross(x_hat, xy_plane[1, :])
            y_hat = numpy.cross(z_hat, x_hat)

            stacked_frames[0].append(origin.reshape(1, -1))
            stacked_frames[1].append(x_hat.reshape(1, -1))
            stacked_frames[2].append(y_hat.reshape(1, -1))
            stacked_frames[3].append(z_hat.reshape(1, -1))

        local_frames = numpy.stack([numpy.vstack(frames) for frames in stacked_frames])
        return local_frames

    @classmethod
    @overload
    def convert_local_coordinates(
        cls,
        local_frame_coordinates: numpy.ndarray,
        local_coordinate_frames: numpy.ndarray,
        backend: Literal["numpy"],
    ) -> numpy.ndarray:
        ...

    @classmethod
    @overload
    def convert_local_coordinates(
        cls,
        local_frame_coordinates: "torch.Tensor",
        local_coordinate_frames: "torch.Tensor",
        backend: Literal["torch"],
    ) -> "torch.Tensor":
        ...

    @classmethod
    def convert_local_coordinates(
        cls,
        local_frame_coordinates,
        local_coordinate_frames,
        backend: Literal["numpy", "torch"] = "numpy",
    ) -> numpy.ndarray:
        """Converts a set of local virtual site coordinates defined in a spherical
        coordinate system into a full set of cartesian coordinates.

        Parameters
        ----------
        local_frame_coordinates
            An array containing the local coordinates with shape=(n_vsites, 3) and with
            columns of distance [A], 'in plane angle' [deg] and 'out of plane'
            angle [deg].
        local_coordinate_frames
            The orthonormal basis associated with each of the virtual sites with
            shape=(4, n_vsites, 3). See the ``build_local_coordinate_frames`` function
            for more details.
        backend
            The framework to use when performing mathematical operations.

        Returns
        -------
            An array of the cartesian coordinates of the virtual sites with
            shape=(n_vsites, 3) and units of [A].
        """

        if backend == "numpy":
            np = numpy
        elif backend == "torch":
            import torch

            np = torch
        else:
            raise NotImplementedError()

        d = local_frame_coordinates[:, 0].reshape(-1, 1)

        theta = (local_frame_coordinates[:, 1] * _DEGREES_TO_RADIANS).reshape(-1, 1)
        phi = (local_frame_coordinates[:, 2] * _DEGREES_TO_RADIANS).reshape(-1, 1)

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        # Here we use cos(phi) in place of sin(phi) and sin(phi) in place of cos(phi)
        # this is because we want phi=0 to represent a 0 degree angle from the x-y plane
        # rather than 0 degrees from the z-axis.
        vsite_positions = (
            local_coordinate_frames[0]
            + d * cos_theta * cos_phi * local_coordinate_frames[1]
            + d * sin_theta * cos_phi * local_coordinate_frames[2]
            + d * sin_phi * local_coordinate_frames[3]
        )

        return vsite_positions

    @classmethod
    def generate_positions(
        cls,
        molecule: "Molecule",
        vsite_collection: VirtualSiteCollection,
        conformer: unit.Quantity,
    ) -> unit.Quantity:
        """Computes the positions of a set of virtual sites relative to a provided
        molecule in a given conformer.

        Parameters
        ----------
        molecule
            The molecule to apply virtual sites to.
        vsite_collection
            The virtual site parameters to apply to the molecule
        conformer
            The conformer [A] of the molecule with shape=(n_atoms, 3) that the virtual
            sites should be placed relative to.

        Returns
        -------
            An array of virtual site positions [A] with shape=(n_vsites, 3).
        """

        conformer = conformer.to(unit.angstrom).m

        vsite_parameters_by_key = {
            (parameter.smirks, parameter.type, parameter.name): parameter
            for parameter in vsite_collection.parameters
        }

        # Extract the values of the assigned parameters.
        _, assigned_parameter_map = cls._apply_virtual_sites(molecule, vsite_collection)
        assigned_parameters = defaultdict(list)

        for _, parameter_keys in assigned_parameter_map.items():
            for parameter_key, orientations in parameter_keys.items():
                assigned_parameters[parameter_key].extend(orientations)

        assigned_parameters = [
            (vsite_parameters_by_key[parameter_key], orientations)
            for parameter_key, orientations in assigned_parameters.items()
        ]

        local_frame_coordinates = numpy.vstack(
            [
                parameter.local_frame_coordinates
                for parameter, orientations in assigned_parameters
                for _ in orientations
            ]
        )

        # Construct the global cartesian coordinates of the v-sites.
        local_coordinate_frames = cls.build_local_coordinate_frames(
            conformer, assigned_parameters
        )
        vsite_positions = VirtualSiteGenerator.convert_local_coordinates(
            local_frame_coordinates, local_coordinate_frames, backend="numpy"
        )

        return vsite_positions * unit.angstrom
