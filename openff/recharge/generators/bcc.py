"""This module contains classes which will generate partial charges using a combination
of a cheaper QM method and a set of bond charge corrections.
"""
import abc
import json
import os
from typing import List, Optional, Tuple, Type

import numpy
from openeye import oechem, oequacpac

from openff.recharge.conformers.conformers import ConformerGenerator
from openff.recharge.generators.exceptions import (
    OEQuacpacError,
    UnableToAssignChargeError,
)
from openff.recharge.models import BondChargeCorrection
from openff.recharge.utilities import get_data_file_path
from openff.recharge.utilities.exceptions import MissingConformersError
from openff.recharge.utilities.openeye import call_openeye, match_smirks


class _BaseBCC(abc.ABC):
    """A base class for those classes which will compute a set of partial charges on
    a molecule by first computing partial charges from a cheaper QM method, and then
    correcting these using a set of bond charge corrections."""

    @classmethod
    @abc.abstractmethod
    def _generate_qm_charges(cls, oe_molecule: oechem.OEMol):
        """Compute the partial charges on a molecule using a cheaper
        QM method.

        Parameters
        ----------
        oe_molecule
            The molecule to compute the charges for. This molecule should
            already have partial charges assigned to it.
        """

    @classmethod
    @abc.abstractmethod
    def _assign_aromaticity(cls, oe_molecule: oechem.OEMol):
        """Assign aromatic flags to a molecule. Existing aromatic flags will
        be cleared.

        Parameters
        ----------
        oe_molecule
            The molecule to assign flags to.
        """

    @classmethod
    def _build_assignment_matrix(
        cls,
        oe_molecule: oechem.OEMol,
        bond_charge_corrections: List[BondChargeCorrection],
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
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
        bond_charge_corrections
            The bond charge corrections to assign.

        Returns
        -------
            The assignment matrix (with shape=`(n_atoms, n_bond_charge_corrections)`
            where `n_atoms` is the number of atoms in the molecule and
            `n_bond_charge_corrections` is the number of bond charges corrections
            to apply) as well as an absolute copy of the assignment matrix.
        """
        atoms = {atom.GetIdx(): atom for atom in oe_molecule.GetAtoms()}

        assignment_matrix = numpy.zeros((len(atoms), len(bond_charge_corrections)))
        absolute_assignment_matrix = numpy.zeros(
            (len(atoms), len(bond_charge_corrections))
        )

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

                absolute_assignment_matrix[matched_indices[0], index] += 1
                absolute_assignment_matrix[matched_indices[1], index] += 1

                matched_bonds.add(forward_matched_bond)
                matched_bonds.add(reverse_matched_bond)

        return assignment_matrix, absolute_assignment_matrix

    @classmethod
    def generate(
        cls,
        oe_molecule: oechem.OEMol,
        bond_charge_corrections: List[BondChargeCorrection],
        conformer_generator: Optional[Type[ConformerGenerator]] = None,
    ) -> Tuple[oechem.OEMol, List[BondChargeCorrection]]:
        """Generate the partial charges for a molecule. If no conformer
        generator is provided those conformers on the provided molecule
        will be used, otherwise, new conformers will be generated using the
        generator.

        Parameters
        ----------
        oe_molecule
            The molecule to compute the charges for.
        bond_charge_corrections
            The bond charge corrections to apply.
        conformer_generator
            The optional conformer to use to generate charges.

        Returns
        -------
            The molecule complete with partial charges and the conformers on the
            used in the charge calculation (if any), and the bond charge corrections
            which were applied to the molecule.
        """

        # Make a copy of the molecule to work on.
        oe_molecule = oechem.OEMol(oe_molecule)

        # Assign aromaticity flags to ensure correct smirks matches.
        cls._assign_aromaticity(oe_molecule)

        # Generate conformers to compute the charges for.
        if conformer_generator is not None:
            oe_molecule = conformer_generator.generate(oe_molecule)
        elif oe_molecule.NumConfs() == 0:
            raise MissingConformersError()

        # Generate the QM partial charges
        cls._generate_qm_charges(oe_molecule)

        # Assign types to each of the atoms in the molecule.
        assignment_matrix, absolute_assignment_matrix = cls._build_assignment_matrix(
            oe_molecule, bond_charge_corrections
        )

        # Check for unassigned atoms
        unassigned_atoms = numpy.where(~assignment_matrix.any(axis=1))[0]

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
                bond_charge_corrections[index].smirks for index in non_zero_assignments
            ]

            raise UnableToAssignChargeError(
                f"An internal error occurred. The {correction_smirks} were applied in "
                f"such a way so that the bond charge corrections alter the total "
                f"charge of the molecule"
            )

        # Ensure all bonds have been assigned a BCC.
        n_assignments = absolute_assignment_matrix.sum(axis=1)
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

        # Determine which bond charge corrections have actually been applied to the
        # molecule
        applied_correction_indices = numpy.where(assignment_matrix.any(axis=0))[0]

        applied_corrections = [
            bond_charge_corrections[index] for index in applied_correction_indices
        ]

        # Create a vector of the corrections to apply
        correction_values = numpy.array(
            [
                [bond_charge_correction.value]
                for bond_charge_correction in bond_charge_corrections
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

        atoms = {atom.GetIdx(): atom for atom in oe_molecule.GetAtoms()}

        for atom_index in range(len(charge_corrections)):

            new_partial_charge = (
                atoms[atom_index].GetPartialCharge() + charge_corrections[atom_index, 0]
            )

            atoms[atom_index].SetPartialCharge(new_partial_charge)

        return oe_molecule, applied_corrections


class AM1BCC(_BaseBCC):
    """A class which will compute the partial charges on a molecule by first
    computing AM1 level partial charges, and then applying a set of bond charge
    corrections on top of these.
    """

    @classmethod
    def _generate_qm_charges(cls, oe_molecule: oechem.OEMol):

        call_openeye(
            oequacpac.OEAssignCharges,
            oe_molecule,
            oequacpac.OEAM1Charges(optimize=True, symmetrize=True),
            exception_type=OEQuacpacError,
        )

    @classmethod
    def _set_aromatic(cls, atom_indices, oe_molecule: oechem.OEMol):
        """Flag all specified atoms and all ring bonds between those atoms
        as being aromatic."""

        atoms = {atom.GetIdx(): atom for atom in oe_molecule.GetAtoms()}

        for matched_atom_index in atom_indices:
            atoms[matched_atom_index].SetAromatic(True)

        bonds = {
            tuple(sorted((bond.GetBgnIdx(), bond.GetEndIdx()))): bond
            for bond in oe_molecule.GetBonds()
        }

        for (index_a, index_b), bond in bonds.items():

            if index_a not in atom_indices or index_b not in atom_indices:
                continue

            if not bond.IsInRing():
                continue

            bond.SetAromatic(True)

    @classmethod
    def _assign_aromaticity(cls, oe_molecule: oechem.OEMol):
        """Assign aromatic flags based upon the model proposed on the 2000 AM1BCC
        paper by A. Jakalian, D. B. Jack and C. I. Bayly."""

        oechem.OEClearAromaticFlags(oe_molecule)

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

        cls._set_aromatic(case_1_atoms, oe_molecule)

        # Case 2)
        case_2_smirks = (
            f"{x_type.replace('N', '1')}1"
            f"=@{x_type.replace('N', '2')}"
            f"-@{x_type.replace('N', '3')}"
            f"=@{x_type.replace('N', '4')}"
            f"-@[#6ar6:5]"
            f":@[#6ar6:6]-@1"
        )

        case_2_matches = match_smirks(case_2_smirks, oe_molecule, unique=True)
        case_2_atoms = {
            match for matches in case_2_matches for match in matches.values()
        }

        cls._set_aromatic(case_2_atoms, oe_molecule)

        # Case 3)
        case_3_smirks = (
            f"{x_type.replace('N', '1')}1"
            f"=@{x_type.replace('N', '2')}"
            f"-@[#6ar6:3]"
            f":@[#6ar6:4]"
            f":@[#6ar6:5]"
            f":@[#6ar6:6]-@1"
        )

        case_3_matches = match_smirks(case_3_smirks, oe_molecule, unique=True)
        case_3_atoms = {
            match for matches in case_3_matches for match in matches.values()
        }

        cls._set_aromatic(case_3_atoms, oe_molecule)

        # Case 4)
        case_4_smirks = (
            "[#6+1r7:1]1"
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

        cls._set_aromatic(case_4_atoms, oe_molecule)

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

        case_5_atoms = {
            match for matches in case_5_matches for match in matches.values()
        }

        cls._set_aromatic(case_5_atoms, oe_molecule)

    @staticmethod
    def original_corrections() -> List[BondChargeCorrection]:
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
            BondChargeCorrection(**dictionary) for dictionary in bcc_dictionaries
        ]

        return bond_charge_corrections
