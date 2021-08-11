"""This module contains classes which will generate partial charges using a combination
of a cheaper QM method and a set of bond charge corrections.
"""
import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy
from openff.utilities import get_data_file_path, requires_package
from pydantic import BaseModel, Field, constr

from openff.recharge.aromaticity import AromaticityModel, AromaticityModels
from openff.recharge.charges.charges import ChargeGenerator, ChargeSettings
from openff.recharge.charges.exceptions import UnableToAssignChargeError
from openff.recharge.conformers import ConformerGenerator, ConformerSettings
from openff.recharge.utilities.exceptions import (
    UnsupportedBCCSmirksError,
    UnsupportedBCCValueError,
)
from openff.recharge.utilities.openeye import import_oechem, match_smirks

if TYPE_CHECKING:
    from openeye import oechem
    from openff.toolkit.typing.engines.smirnoff.parameters import (
        ChargeIncrementModelHandler,
    )


class BCCParameter(BaseModel):
    """An object which encodes the value of a bond-charge correction, the chemical
    environment to which it should be applied, and provenance about its source.
    """

    smirks: constr(min_length=1) = Field(
        ...,
        description="A SMIRKS pattern that encodes the chemical environment that "
        "this correction should be applied to.",
    )
    value: float = Field(..., description="The value [e] of this correction.")

    provenance: Optional[Dict[str, Any]] = Field(
        None, description="Provenance information about this bond charge correction."
    )


class BCCCollection(BaseModel):
    """The settings which describes which BCCs should be applied,
    as well as information about how they should be applied."""

    parameters: List[BCCParameter] = Field(
        ..., description="The bond charge corrections to apply."
    )
    aromaticity_model: AromaticityModels = Field(
        AromaticityModels.AM1BCC,
        description="The model to use when assigning aromaticity.",
    )

    @requires_package("openff.toolkit")
    @requires_package("simtk")
    def to_smirnoff(self) -> "ChargeIncrementModelHandler":
        """Converts this collection of bond charge correction parameters to
        a SMIRNOFF bond charge increment parameter handler.

        Notes
        -----
        * The AM1BCC charges applied by this handler will likely not match
          those computed using the built-in OpenEye implementation as that
          implementation uses a custom aromaticity model not supported by
          SMIRNOFF. This is in addition to potential conversion errors of the
          parameters into the SMIRKS language.

        * The aromaticity model defined by the collection will be ignored as
          this handler will parse the aromaticity model directly from the
          top level aromaticity node of the the SMIRNOFF specification.

        Returns
        -------
            The constructed parameter handler.
        """
        from openff.toolkit.typing.engines.smirnoff.parameters import (
            ChargeIncrementModelHandler,
        )
        from simtk import unit

        # noinspection PyTypeChecker
        bcc_parameter_handler = ChargeIncrementModelHandler(version="0.3")

        bcc_parameter_handler.number_of_conformers = 500
        bcc_parameter_handler.partial_charge_method = "am1elf10"

        for bcc_parameter in reversed(self.parameters):
            bcc_parameter_handler.add_parameter(
                {
                    "smirks": bcc_parameter.smirks,
                    "charge_increment": [
                        bcc_parameter.value * unit.elementary_charge,
                        -bcc_parameter.value * unit.elementary_charge,
                    ],
                }
            )

        return bcc_parameter_handler

    @classmethod
    @requires_package("simtk")
    def from_smirnoff(
        cls,
        parameter_handler: "ChargeIncrementModelHandler",
        aromaticity_model=AromaticityModels.MDL,
    ) -> "BCCCollection":
        """Attempts to convert a SMIRNOFF bond charge increment parameter handler
        to a bond charge parameter collection.

        Notes
        -----
        * Only bond charge corrections (i.e. corrections whose SMIRKS only involve
          two tagged atoms) are supported currently.

        Parameters
        ----------
        parameter_handler
            The parameter handler to convert.
        aromaticity_model
            The model which describes how aromaticity should be assigned
            when applying the bond charge correction parameters.

        Returns
        -------
            The converted bond charge correction collection.
        """
        from simtk import unit

        bcc_parameters = []

        for off_parameter in reversed(parameter_handler.parameters):

            smirks = off_parameter.smirks

            if len(off_parameter.charge_increment) not in [1, 2]:
                raise UnsupportedBCCSmirksError(
                    smirks, len(off_parameter.charge_increment)
                )

            forward_value = off_parameter.charge_increment[0].value_in_unit(
                unit.elementary_charge
            )
            reverse_value = -forward_value

            if len(off_parameter.charge_increment) > 1:
                reverse_value = off_parameter.charge_increment[1].value_in_unit(
                    unit.elementary_charge
                )

            if not numpy.isclose(forward_value, -reverse_value):
                raise UnsupportedBCCValueError(
                    smirks,
                    forward_value,
                    reverse_value,
                )

            bcc_parameters.append(
                BCCParameter(smirks=smirks, value=forward_value, provenance={})
            )

        return cls(parameters=bcc_parameters, aromaticity_model=aromaticity_model)

    def vectorize(self, smirks: List[str]) -> numpy.ndarray:
        """Returns a flat vector of the charge increment values associated with each
        SMIRKS pattern in a specified list.

        Parameters
        ----------
        smirks
            A list of SMIRKS patterns corresponding to the BCC values to include
            in the returned vector.

        Returns
        -------
            A flat vector of charge increments with shape=(n_smirks, 1)
        """

        parameters = {parameter.smirks: parameter for parameter in self.parameters}
        return numpy.array([[parameters[pattern].value] for pattern in smirks])


class BCCGenerator:
    """A class for generating the bond charge corrections which should
    be applied to a molecule."""

    @classmethod
    def _validate_assignment_matrix(
        cls,
        oe_molecule: "oechem.OEMol",
        assignment_matrix: numpy.ndarray,
        bcc_counts_matrix: numpy.ndarray,
        bcc_collection: BCCCollection,
    ):
        """Validates the assignment matrix.

        Parameters
        ----------
        oe_molecule
            The molecule which the assignment matrix was generated
            for.
        assignment_matrix
            The assignment matrix to validate with
            shape=(n_atoms, n_bond_charge_corrections)
        bcc_counts_matrix
            A matrix which contains the number of times that a bond charge
            correction is applied to a given atom with
            shape=(n_atoms, n_bond_charge_corrections).
        bcc_collection
            The collection of parameters to be assigned.
        """

        # Check for unassigned atoms
        n_atoms = len(list(oe_molecule.GetAtoms()))

        unassigned_atoms = numpy.where(~bcc_counts_matrix[:n_atoms].any(axis=1))[0]

        if len(unassigned_atoms) > 0:
            unassigned_atom_string = ", ".join(map(str, unassigned_atoms))

            raise UnableToAssignChargeError(
                f"Atoms {unassigned_atom_string} could not be assigned a bond "
                f"charge correction atom type."
            )

        # Check for non-zero contributions from charge corrections
        non_zero_assignments = assignment_matrix.sum(axis=0).nonzero()[0]

        if len(non_zero_assignments) > 0:

            correction_smirks = [
                bcc_collection.parameters[index].smirks
                for index in non_zero_assignments
            ]

            raise UnableToAssignChargeError(
                f"An internal error occurred. The {correction_smirks} were applied in "
                f"such a way so that the bond charge corrections alter the total "
                f"charge of the molecule"
            )

        # Ensure all bonds have been assigned a BCC.
        n_assignments = bcc_counts_matrix.sum(axis=1)
        atoms = {atom.GetIdx(): atom for atom in oe_molecule.GetAtoms()}

        unassigned_atoms = {
            index: (len([*atoms[index].GetBonds()]), n_assignments[index])
            for index in range(len(atoms))
            if len([*atoms[index].GetBonds()]) != n_assignments[index]
        }

        if len(unassigned_atoms) > 0:

            unassigned_atom_string = "\n".join(
                [
                    f"atom {index}: expected={n_expected} assigned={n_actual}"
                    for index, (n_expected, n_actual) in unassigned_atoms.items()
                ]
            )

            raise UnableToAssignChargeError(
                f"Bond charge corrections could not be applied to all bonds in the "
                f"molecule:\n\n{unassigned_atom_string}"
            )

    @classmethod
    def build_assignment_matrix(
        cls,
        oe_molecule: "oechem.OEMol",
        bcc_collection: BCCCollection,
    ) -> numpy.ndarray:
        """Generates a matrix that specifies which bond charge corrections have been
        applied to which atoms in the molecule.

        The matrix takes the form `[atom_index, bcc_index]` where `atom_index` is the
        index of an atom in the molecule and `bcc_index` is the index of a bond charge
        correction. Each value in the matrix can either be positive or negative
        depending on the direction the BCC was applied in.

        Parameters
        ----------
        oe_molecule
            The molecule to assign the bond charge corrections to.
        bcc_collection
            The bond charge correction parameters that may be assigned.

        Returns
        -------
            The assignment matrix with shape=(n_atoms, n_bond_charge_corrections)
            where `n_atoms` is the number of atoms in the molecule and
            `n_bond_charge_corrections` is the number of bond charges corrections
            to apply.
        """

        oechem = import_oechem()

        # Make a copy of the molecule to assign the aromatic flags to.
        oe_molecule = oechem.OEMol(oe_molecule)
        # Assign aromaticity flags to ensure correct smirks matches.
        AromaticityModel.assign(oe_molecule, bcc_collection.aromaticity_model)

        bond_charge_corrections = bcc_collection.parameters

        atoms = {atom.GetIdx(): atom for atom in oe_molecule.GetAtoms()}

        assignment_matrix = numpy.zeros((len(atoms), len(bond_charge_corrections)))
        bcc_counts_matrix = numpy.zeros((len(atoms), len(bond_charge_corrections)))

        matched_bonds = set()

        for index in range(len(bond_charge_corrections)):

            bond_correction = bond_charge_corrections[index]

            matches = match_smirks(bond_correction.smirks, oe_molecule, False)

            for matched_indices in matches:

                forward_matched_bond = (matched_indices[0], matched_indices[1])
                reverse_matched_bond = (matched_indices[1], matched_indices[0])

                if (
                    forward_matched_bond in matched_bonds
                    or reverse_matched_bond in matched_bonds
                ):
                    continue

                assignment_matrix[matched_indices[0], index] += 1
                assignment_matrix[matched_indices[1], index] -= 1

                bcc_counts_matrix[matched_indices[0], index] += 1
                bcc_counts_matrix[matched_indices[1], index] += 1

                matched_bonds.add(forward_matched_bond)
                matched_bonds.add(reverse_matched_bond)

        # Validate the assignments
        cls._validate_assignment_matrix(
            oe_molecule, assignment_matrix, bcc_counts_matrix, bcc_collection
        )

        return assignment_matrix

    @classmethod
    def apply_assignment_matrix(
        cls,
        assignment_matrix: numpy.ndarray,
        bcc_collection: BCCCollection,
    ) -> numpy.ndarray:
        """Applies an assignment matrix to a list of bond charge corrections
        yield the final bond-charge corrections for a molecule.

        Parameters
        ----------
        assignment_matrix
            The bond-charge correction matrix constructed using
            ``build_assignment_matrix`` which describes how the
            bond charge corrections should be applied. This
            should have shape=(n_atoms, n_bond_charge_corrections)
        bcc_collection
            The bond charge correction parameters which may be assigned.

        Returns
        -------
            The bond-charge corrections with shape=(n_atoms, 1).
        """
        # Create a vector of the corrections to apply
        correction_values = numpy.array(
            [
                [bond_charge_correction.value]
                for bond_charge_correction in bcc_collection.parameters
            ]
        )

        # Apply the corrections
        charge_corrections = assignment_matrix @ correction_values

        if not numpy.isclose(charge_corrections.sum(), 0.0):

            raise UnableToAssignChargeError(
                "An internal error occurred. The bond charge corrections were applied "
                "in such a way so that the total charge of the molecule will be "
                "altered."
            )

        return charge_corrections

    @classmethod
    def applied_corrections(
        cls,
        *oe_molecules: "oechem.OEMol",
        bcc_collection: BCCCollection,
    ) -> List[BCCParameter]:
        """Returns the bond charge corrections which will be applied
        to a given molecule.

        Parameters
        ----------
        oe_molecules
            The molecule which the bond charge corrections would be applied to.
        bcc_collection
            The bond charge correction parameters which may be assigned.
        """

        applied_corrections = []

        for oe_molecule in oe_molecules:

            assignment_matrix = cls.build_assignment_matrix(oe_molecule, bcc_collection)
            applied_correction_indices = numpy.where(assignment_matrix.any(axis=0))[0]

            applied_corrections.extend(
                bcc_collection.parameters[index]
                for index in applied_correction_indices
                if bcc_collection.parameters[index] not in applied_corrections
            )

        applied_corrections.sort(key=lambda x: bcc_collection.parameters.index(x))
        return applied_corrections

    @classmethod
    def generate(
        cls,
        oe_molecule: "oechem.OEMol",
        bcc_collection: BCCCollection,
    ) -> numpy.ndarray:
        """Generate a set of charge increments for a molecule.

        Parameters
        ----------
        oe_molecule
            The molecule to generate the bond-charge corrections for.
        bcc_collection
            The bond charge correction parameters which may be assigned.

        Returns
        -------
            The bond-charge corrections which should be applied to the
            molecule.
        """

        assignment_matrix = cls.build_assignment_matrix(oe_molecule, bcc_collection)

        generated_corrections = cls.apply_assignment_matrix(
            assignment_matrix, bcc_collection
        )

        return generated_corrections


def original_am1bcc_corrections() -> BCCCollection:
    """Returns the bond charge corrections originally reported
    in the literture [1]_.

    References
    ----------
    [1] Jakalian, A., Jack, D. B., & Bayly, C. I. (2002). Fast, efficient
        generation of high-quality atomic charges. AM1-BCC model: II.
        Parameterization and validation. Journal of computational chemistry,
        23(16), 1623–1641.
    """
    bcc_file_path = get_data_file_path(
        os.path.join("bcc", "original-am1-bcc.json"), "openff.recharge"
    )

    with open(bcc_file_path) as file:
        bcc_dictionaries = json.load(file)

    bond_charge_corrections = [
        BCCParameter(**dictionary) for dictionary in bcc_dictionaries
    ]

    return BCCCollection(
        parameters=bond_charge_corrections, aromaticity_model=AromaticityModels.AM1BCC
    )


def compare_openeye_parity(oe_molecule: "oechem.OEMol") -> bool:
    """A utility function to compute the bond charge corrections
    on a molecule using both the internal AM1BCC implementation,
    and the OpenEye AM1BCC implementation.

    This method is mainly only to be used for testing and validation
    purposes.

    Parameters
    ----------
    oe_molecule
        The molecule to compute the charges of.

    Returns
    -------
        Whether the internal and OpenEye implementations are in
        agreement for this molecule.
    """

    bond_charge_corrections = original_am1bcc_corrections()

    # Generate a conformer for the molecule.
    conformers = ConformerGenerator.generate(
        oe_molecule,
        ConformerSettings(method="omega", sampling_mode="sparse", max_conformers=1),
    )

    # Generate a set of reference charges using the OpenEye implementation
    reference_charges = ChargeGenerator.generate(
        oe_molecule,
        conformers,
        ChargeSettings(theory="am1bcc", symmetrize=False, optimize=False),
    )

    # Generate a set of charges using this frameworks functions
    am1_charges = ChargeGenerator.generate(
        oe_molecule,
        conformers,
        ChargeSettings(theory="am1", symmetrize=False, optimize=False),
    )

    # Determine the values of the OE BCCs
    reference_charge_corrections = reference_charges - am1_charges

    # Compute the internal BCCs
    assignment_matrix = BCCGenerator.build_assignment_matrix(
        oe_molecule, bond_charge_corrections
    )
    charge_corrections = BCCGenerator.apply_assignment_matrix(
        assignment_matrix, bond_charge_corrections
    )

    implementation_charges = am1_charges + charge_corrections

    # Check that their is no difference between the implemented and
    # reference charges.
    return numpy.allclose(reference_charges, implementation_charges) and numpy.allclose(
        charge_corrections, reference_charge_corrections
    )
