import abc
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union

import numpy
from pydantic import BaseModel, Field, constr

from openff.recharge.aromaticity import AromaticityModel, AromaticityModels
from openff.recharge.charges.exceptions import UnableToAssignChargeError
from openff.recharge.utilities import requires_package
from openff.recharge.utilities.openeye import import_oechem

if TYPE_CHECKING:

    from openeye.oechem import OEMol
    from openff.toolkit.topology import Molecule
    from openff.toolkit.typing.engines.smirnoff import VirtualSiteHandler

ExclusionPolicy = Literal["none", "parents"]

VirtualSiteKey = Tuple[str, str, str]


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


class BondChargeSiteParameter(_VirtualSiteParameter):

    type: Literal["BondCharge"] = "BondCharge"


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


class DivalentLonePairParameter(_VirtualSiteParameter):

    type: Literal["DivalentLonePair"] = "DivalentLonePair"

    out_of_plane_angle: float = Field(
        ...,
        description="The angle [deg] to move the virtual site out of the plane "
        "defined by the tagged atoms by.",
    )


class TrivalentLonePairParameter(_VirtualSiteParameter):

    type: Literal["TrivalentLonePair"] = "TrivalentLonePair"


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

    @requires_package("openff.toolkit")
    @requires_package("simtk")
    def to_smirnoff(self) -> "VirtualSiteHandler":
        """Converts this collection of virtual site parameters to a SMIRNOFF virtual
        site parameter handler.

        Returns
        -------
            The constructed parameter handler.
        """

        from openff.toolkit.typing.engines.smirnoff import VirtualSiteHandler
        from simtk import unit

        # noinspection PyTypeChecker
        parameter_handler = VirtualSiteHandler(
            version="0.3", exclusion_policy=self.exclusion_policy
        )

        for parameter in self.parameters:

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
    @requires_package("simtk")
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

        from simtk import unit

        parameters = []

        for smirnoff_parameter in parameter_handler.parameters:

            base_kwargs = dict(
                smirks=smirnoff_parameter.smirks,
                name=smirnoff_parameter.name,
                distance=smirnoff_parameter.distance.value_in_unit(unit.angstrom),
                charge_increments=tuple(
                    charge.value_in_unit(unit.elementary_charge)
                    for charge in smirnoff_parameter.charge_increment
                ),
                sigma=smirnoff_parameter.sigma.value_in_unit(unit.angstrom),
                epsilon=smirnoff_parameter.epsilon.value_in_unit(
                    unit.kilojoules_per_mole
                ),
                match=smirnoff_parameter.match.replace("_", "-").lower(),
            )

            if smirnoff_parameter.type == "BondCharge":
                parameter = BondChargeSiteParameter(**base_kwargs)

            elif smirnoff_parameter.type == "MonovalentLonePair":

                parameter = MonovalentLonePairParameter(
                    **base_kwargs,
                    out_of_plane_angle=smirnoff_parameter.outOfPlaneAngle.value_in_unit(
                        unit.degrees
                    ),
                    in_plane_angle=smirnoff_parameter.inPlaneAngle.value_in_unit(
                        unit.degrees
                    ),
                )

            elif smirnoff_parameter.type == "DivalentLonePair":

                parameter = DivalentLonePairParameter(
                    **base_kwargs,
                    out_of_plane_angle=smirnoff_parameter.outOfPlaneAngle.value_in_unit(
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


class VirtualSiteGenerator:
    @classmethod
    def _apply_virtual_sites(
        cls, oe_molecule: "OEMol", vsite_collection: VirtualSiteCollection
    ) -> Tuple["Molecule", Dict[Tuple[int, ...], List[VirtualSiteKey]]]:
        """Applies a virtual site collection to a molecule.

        Parameters
        ----------
        oe_molecule
            The molecule to build the virtual sites for.
        vsite_collection
            The v-site collection to use to create the virtual sites.

        Returns
        -------
            An OpenFF molecule with virtual sites as well as a dictionary that maps each
            virtual site back to the parameter that yielded it.
        """

        from openff.toolkit.topology import Molecule

        parameter_handler = vsite_collection.to_smirnoff()

        off_topology = parameter_handler.create_openff_virtual_sites(
            Molecule.from_openeye(oe_molecule).to_topology()
        )
        off_molecule = next(off_topology.reference_molecules)

        term_topology = parameter_handler._term_map_topology(off_topology)

        assigned_vsite_keys = defaultdict(set)

        for (_, atom_indices), vsite_keys in term_topology.items():

            for vsite_key in vsite_keys:

                (_, (smirks, (vsite_type, (vsite_name, _)))) = vsite_key
                assigned_vsite_keys[atom_indices].add((smirks, vsite_type, vsite_name))

        return off_molecule, {
            atom_indices: [*keys] for atom_indices, keys in assigned_vsite_keys.items()
        }

    @classmethod
    def _build_charge_increment_array(
        cls, vsite_collection: VirtualSiteCollection
    ) -> Tuple[numpy.ndarray, List[Tuple[str, str, str, int]]]:
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

            raise UnableToAssignChargeError(
                "An internal error occurred. The v-site charge increments alter the "
                "total charge of the molecule"
            )

    @classmethod
    def build_charge_assignment_matrix(
        cls,
        oe_molecule: "OEMol",
        vsite_collection: VirtualSiteCollection,
    ) -> numpy.ndarray:
        """Generates a matrix that specifies which v-site charge increments have been
        applied to which atoms in the molecule.

        The matrix takes the form ...

        Parameters
        ----------
        oe_molecule
            The molecule to assign the v-site charge increments to.
        vsite_collection
            The v-site parameters that may be assigned.

        Returns
        -------
            The assignment matrix with shape=(n_atoms + n_vsites, n_charge_increments)
            where ...
        """

        oechem = import_oechem()

        # Make a copy of the molecule to assign the aromatic flags to.
        oe_molecule = oechem.OEMol(oe_molecule)
        # Assign aromaticity flags to ensure correct smirks matches.
        AromaticityModel.assign(oe_molecule, vsite_collection.aromaticity_model)

        off_molecule, assigned_vsite_keys = cls._apply_virtual_sites(
            oe_molecule, vsite_collection
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
            vsite_parameter_keys = assigned_vsite_keys[vsite_particle.orientation]

            for vsite_parameter_key in vsite_parameter_keys:

                for i, atom_index in enumerate(vsite_particle.orientation):

                    # noinspection PyTypeChecker
                    vsite_parameter_index = all_vsite_keys.index(
                        (*vsite_parameter_key, i)
                    )

                    assignment_matrix[atom_index, vsite_parameter_index] -= 1
                    assignment_matrix[vsite_index, vsite_parameter_index] += 1

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

            raise UnableToAssignChargeError(
                "An internal error occurred. The bond charge corrections were applied "
                "in such a way so that the total charge of the molecule will be "
                "altered."
            )

        return charge_corrections.reshape(-1, 1)

    @classmethod
    def generate_charge_increments(
        cls,
        oe_molecule: "OEMol",
        vsite_collection: VirtualSiteCollection,
    ) -> numpy.ndarray:
        """Generate a set of charge increments due to virtual sites for a molecule.

        Parameters
        ----------
        oe_molecule
            The molecule to generate the charge increments for.
        vsite_collection
            The virtual site parameters that may be assigned.

        Returns
        -------
            The charge increments with shape=(n_atoms + n_vsites, 1) that should be
            applied to the molecule.
        """

        assignment_matrix = cls.build_charge_assignment_matrix(
            oe_molecule, vsite_collection
        )

        generated_corrections = cls.apply_charge_assignment_matrix(
            assignment_matrix, vsite_collection
        )

        return generated_corrections

    @classmethod
    @requires_package("simtk")
    def generate_positions(
        cls,
        oe_molecule: "OEMol",
        vsite_collection: VirtualSiteCollection,
        conformer: numpy.ndarray,
    ):
        """Computes the positions of a set of virtual sites relative to a provided
        molecule in a given conformer.

        Parameters
        ----------
        oe_molecule
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

        from simtk import unit

        off_molecule, _ = cls._apply_virtual_sites(oe_molecule, vsite_collection)

        vsite_positions = (
            off_molecule.compute_virtual_site_positions_from_atom_positions(
                conformer * unit.angstrom
            )
        )

        # noinspection PyUnresolvedReferences
        return vsite_positions.value_in_unit(unit.angstrom)
