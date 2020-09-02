"""This module contains classes which will generate partial charges using a combination
of a cheaper QM method and a set of bond charge corrections.
"""
import json
import os
from enum import Enum
from typing import Any, Dict, List

import numpy
from openeye import oechem
from pydantic import BaseModel, Field, constr

from openff.recharge.charges.charges import ChargeGenerator, ChargeSettings
from openff.recharge.charges.exceptions import UnableToAssignChargeError
from openff.recharge.conformers import ConformerGenerator, ConformerSettings
from openff.recharge.utilities import get_data_file_path
from openff.recharge.utilities.openeye import match_smirks


class AromaticityModels(Enum):
    """An enumeration of the available aromaticity models.

    These include

    * AM1BCC - the aromaticity model defined in the original AM1BCC publications _[1].
    * MDL - The MDL aromaticity model.

    References
    ----------
    [1] Jakalian, A., Jack, D. B., & Bayly, C. I. (2002). Fast, efficient generation
        of high-quality atomic charges. AM1-BCC model: II. Parameterization and
        validation. Journal of computational chemistry, 23(16), 1623–1641.
    """

    AM1BCC = "AM1BCC"
    MDL = "MDL"


class AromaticityModel:
    """A class which will assign aromatic flags to a molecule based upon
    a specified aromatic model."""

    @classmethod
    def _set_aromatic(
        cls, ring_matches: List[Dict[int, int]], oe_molecule: oechem.OEMol
    ):
        """Flag all specified ring atoms and all ring bonds between those atoms
        as being aromatic.

        Parameters
        ----------
        ring_matches
            The indices of the atoms in each of the rings to flag as aromatic.
        oe_molecule
            The molecule to assign the aromatic flags to.
        """

        atoms = {atom.GetIdx(): atom for atom in oe_molecule.GetAtoms()}
        bonds = {
            tuple(sorted((bond.GetBgnIdx(), bond.GetEndIdx()))): bond
            for bond in oe_molecule.GetBonds()
        }

        for ring_match in ring_matches:

            ring_atom_indices = {match for match in ring_match.values()}

            for matched_atom_index in ring_atom_indices:
                atoms[matched_atom_index].SetAromatic(True)

            for (index_a, index_b), bond in bonds.items():

                if index_a not in ring_atom_indices or index_b not in ring_atom_indices:
                    continue

                if not bond.IsInRing():
                    continue

                bond.SetAromatic(True)

    @classmethod
    def _assign_am1bcc(cls, oe_molecule: oechem.OEMol):
        """Applies aromaticity flags based upon the aromaticity model
        outlined in the original AM1BCC publications _[1].

        Parameters
        ----------
        oe_molecule
            The molecule to assign aromatic flags to.

        References
        ----------
        [1] Jakalian, A., Jack, D. B., & Bayly, C. I. (2002). Fast, efficient generation
            of high-quality atomic charges. AM1-BCC model: II. Parameterization and
            validation. Journal of computational chemistry, 23(16), 1623–1641.
        """

        x_type = "[#6X3,#7X2,#15X2,#7X3+1,#15X3+1,#8X2+1,#16X2+1:N]"
        y_type = "[#6X2-1,#7X2-1,#8X2,#16X2,#7X3,#15X3:N]"
        z_type = x_type

        # Case 1)
        case_1_smirks = (
            f"{x_type.replace('N', '1')}1"
            f"=@{x_type.replace('N', '2')}"
            f"-@{x_type.replace('N', '3')}"
            f"=@{x_type.replace('N', '4')}"
            f"-@{x_type.replace('N', '5')}"
            f"=@{x_type.replace('N', '6')}-@1"
        )

        case_1_matches = match_smirks(case_1_smirks, oe_molecule, unique=True)
        case_1_atoms = {
            match for matches in case_1_matches for match in matches.values()
        }

        cls._set_aromatic(case_1_matches, oe_molecule)

        # Track the ar6 assignments as there is no atom attribute to
        # safely determine if an atom is in a six member ring when
        # that same atom is also in a five member ring.
        ar6_assignments = {*case_1_atoms}

        # Case 2)
        case_2_smirks = (
            f"{x_type.replace('N', '1')}1"
            f"=@{x_type.replace('N', '2')}"
            f"-@{x_type.replace('N', '3')}"
            f"=@{x_type.replace('N', '4')}"
            f"-@{x_type.replace('N', '5')}"
            f":@{x_type.replace('N', '6')}-@1"
        )

        previous_case_2_atoms = None
        case_2_atoms = {}

        while previous_case_2_atoms != case_2_atoms:

            case_2_matches = match_smirks(case_2_smirks, oe_molecule, unique=True)

            # Enforce the ar6 condition
            case_2_matches = [
                case_2_match
                for case_2_match in case_2_matches
                if case_2_match[4] in ar6_assignments
                and case_2_match[5] in ar6_assignments
            ]

            previous_case_2_atoms = case_2_atoms
            case_2_atoms = {
                match for matches in case_2_matches for match in matches.values()
            }

            ar6_assignments.update(case_2_atoms)
            cls._set_aromatic(case_2_matches, oe_molecule)

        # Case 3)
        case_3_smirks = (
            f"{x_type.replace('N', '1')}1"
            f"=@{x_type.replace('N', '2')}"
            f"-@{x_type.replace('N', '3')}"
            f":@{x_type.replace('N', '4')}"
            f"~@{x_type.replace('N', '5')}"
            f":@{x_type.replace('N', '6')}-@1"
        )

        previous_case_3_atoms = None
        case_3_atoms = {}

        while previous_case_3_atoms != case_3_atoms:

            case_3_matches = match_smirks(case_3_smirks, oe_molecule, unique=True)

            # Enforce the ar6 condition
            case_3_matches = [
                case_3_match
                for case_3_match in case_3_matches
                if case_3_match[2] in ar6_assignments
                and case_3_match[3] in ar6_assignments
                and case_3_match[4] in ar6_assignments
                and case_3_match[5] in ar6_assignments
            ]

            previous_case_3_atoms = case_3_atoms
            case_3_atoms = {
                match for matches in case_3_matches for match in matches.values()
            }

            ar6_assignments.update(case_3_atoms)

            cls._set_aromatic(case_3_matches, oe_molecule)

        # Case 4)
        case_4_smirks = (
            "[#6+1:1]1"
            f"-@{x_type.replace('N', '2')}"
            f"=@{x_type.replace('N', '3')}"
            f"-@{x_type.replace('N', '4')}"
            f"=@{x_type.replace('N', '5')}"
            f"-@{x_type.replace('N', '6')}"
            f"=@{x_type.replace('N', '7')}-@1"
        )

        case_4_matches = match_smirks(case_4_smirks, oe_molecule, unique=True)
        case_4_atoms = {
            match for matches in case_4_matches for match in matches.values()
        }

        cls._set_aromatic(case_4_matches, oe_molecule)

        # Case 5)
        case_5_smirks = (
            f"{y_type.replace('N', '1')}1"
            f"-@{z_type.replace('N', '2')}"
            f"=@{z_type.replace('N', '3')}"
            f"-@{x_type.replace('N', '4')}"
            f"=@{x_type.replace('N', '5')}-@1"
        )

        ar_6_ar_7_matches = {
            *case_1_atoms,
            *case_2_atoms,
            *case_3_atoms,
            *case_4_atoms,
        }

        case_5_matches = match_smirks(case_5_smirks, oe_molecule, unique=True)
        case_5_matches = [
            matches
            for matches in case_5_matches
            if matches[1] not in ar_6_ar_7_matches
            and matches[2] not in ar_6_ar_7_matches
        ]

        cls._set_aromatic(case_5_matches, oe_molecule)

    @classmethod
    def assign(cls, oe_molecule: oechem.OEMol, model: AromaticityModels):
        """Clears the current aromaticity flags on a molecule and assigns
        new ones based on the specified aromaticity model.

        Parameters
        ----------
        oe_molecule
            The molecule to assign aromatic flags to.
        model
            The aromaticity model to apply.
        """

        oechem.OEClearAromaticFlags(oe_molecule)

        if model == AromaticityModels.AM1BCC:
            cls._assign_am1bcc(oe_molecule)
        elif model == AromaticityModels.MDL:
            oechem.OEAssignAromaticFlags(oe_molecule, oechem.OEAroModel_MDL)
        else:
            raise NotImplementedError()


class BCCParameter(BaseModel):
    """An object which encodes the value of a bond-charge correction, the chemical
    environment to which it should be applied, and provenance about its source.
    """

    smirks: constr(min_length=1) = Field(
        ...,
        description="A SMIRKS pattern which encodes the chemical environment that "
        "this correction should be applied to.",
    )
    value: float = Field(..., description="The value of this correction.")

    provenance: Dict[str, Any] = Field(
        ..., description="Provenance information about this bond charge correction."
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


class BCCGenerator:
    """A class for generating the bond charge corrections which should
    be applied to a molecule."""

    @classmethod
    def _validate_assignment_matrix(
        cls,
        oe_molecule: oechem.OEMol,
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
        unassigned_atoms = numpy.where(~bcc_counts_matrix.any(axis=1))[0]

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
        oe_molecule: oechem.OEMol,
        bcc_collection: BCCCollection,
    ) -> numpy.ndarray:
        """Generated a matrix which indicates which bond charge
        corrections have been applied to each atom in the molecule.

        The matrix takes the form `[atom_index, bcc_index]` where
        `atom_index` is the index of an atom in the molecule and
        `bcc_index` is the index of a bond charge correction. Each
        value in the matrix can either be positive or negative
        depending on the direction the BCC was applied in.

        Parameters
        ----------
        oe_molecule
            The molecule to assign the bond charge corrections to.
        bcc_collection
            The bond charge correction parameters which may be assigned.

        Returns
        -------
            The assignment matrix with shape=(n_atoms, n_bond_charge_corrections)
            where `n_atoms` is the number of atoms in the molecule and
            `n_bond_charge_corrections` is the number of bond charges corrections
            to apply.
        """

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
        *oe_molecules: oechem.OEMol,
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

        return applied_corrections

    @classmethod
    def generate(
        cls,
        oe_molecule: oechem.OEMol,
        bcc_collection: BCCCollection,
    ) -> numpy.ndarray:
        """Generate the partial charges for a molecule. If no conformer
        generator is provided those conformers on the provided molecule
        will be used, otherwise, new conformers will be generated using the
        generator.

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

        # Build the assignment matrix
        assignment_matrix = cls.build_assignment_matrix(oe_molecule, bcc_collection)

        # Determine which bond charge corrections have actually been applied to the
        # molecule
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
    bcc_file_path = get_data_file_path(os.path.join("bcc", "original-am1-bcc.json"))

    with open(bcc_file_path) as file:
        bcc_dictionaries = json.load(file)

    bond_charge_corrections = [
        BCCParameter(**dictionary) for dictionary in bcc_dictionaries
    ]

    return BCCCollection(
        parameters=bond_charge_corrections, aromaticity_model=AromaticityModels.AM1BCC
    )


def compare_openeye_parity(oe_molecule: oechem.OEMol) -> bool:
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
