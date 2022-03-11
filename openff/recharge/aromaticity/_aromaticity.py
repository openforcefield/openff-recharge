from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Tuple

from openff.recharge.utilities.molecule import find_ring_bonds
from openff.recharge.utilities.toolkits import apply_mdl_aromaticity_model, match_smirks

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule


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
        cls,
        ring_matches: List[Dict[int, int]],
        is_bond_in_ring: Dict[Tuple[int, int], bool],
        is_atom_aromatic: Dict[int, bool],
        is_bond_aromatic: Dict[Tuple[int, int], bool],
    ):
        """Flag all specified ring atoms and all ring bonds between those atoms
        as being aromatic.

        Parameters
        ----------
        ring_matches
            The indices of the atoms in each of the rings to flag as aromatic.
        is_atom_aromatic
            The atom aromaticity flags to update.
        is_bond_aromatic
            The bond aromaticity flags to update.
        """

        for ring_match in ring_matches:

            ring_atom_indices = {match for match in ring_match.values()}

            for matched_atom_index in ring_atom_indices:
                is_atom_aromatic[matched_atom_index] = True

            for index_a, index_b in is_bond_aromatic:

                if index_a not in ring_atom_indices or index_b not in ring_atom_indices:
                    continue

                # noinspection PyTypeChecker
                if not is_bond_in_ring[tuple(sorted((index_a, index_b)))]:
                    continue

                is_bond_aromatic[(index_a, index_b)] = True

    @classmethod
    def _assign_am1bcc(
        cls, molecule: "Molecule"
    ) -> Tuple[Dict[int, bool], Dict[Tuple[int, int], bool]]:
        """Applies aromaticity flags based upon the aromaticity model
        outlined in the original AM1BCC publications _[1].

        Parameters
        ----------
        molecule
            The molecule to generate aromatic flags for.

        References
        ----------
        [1] Jakalian, A., Jack, D. B., & Bayly, C. I. (2002). Fast, efficient generation
            of high-quality atomic charges. AM1-BCC model: II. Parameterization and
            validation. Journal of computational chemistry, 23(16), 1623–1641.
        """

        is_atom_aromatic = {i: False for i in range(molecule.n_atoms)}
        is_bond_aromatic = {
            (bond.atom1_index, bond.atom2_index): False for bond in molecule.bonds
        }

        is_bond_in_ring = find_ring_bonds(molecule)

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

        case_1_matches = match_smirks(
            case_1_smirks, molecule, is_atom_aromatic, is_bond_aromatic, unique=True
        )
        case_1_atoms = {
            match for matches in case_1_matches for match in matches.values()
        }

        cls._set_aromatic(
            case_1_matches, is_bond_in_ring, is_atom_aromatic, is_bond_aromatic
        )

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

            case_2_matches = match_smirks(
                case_2_smirks, molecule, is_atom_aromatic, is_bond_aromatic, unique=True
            )

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
            cls._set_aromatic(
                case_2_matches, is_bond_in_ring, is_atom_aromatic, is_bond_aromatic
            )

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

            case_3_matches = match_smirks(
                case_3_smirks, molecule, is_atom_aromatic, is_bond_aromatic, unique=True
            )

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

            cls._set_aromatic(
                case_3_matches, is_bond_in_ring, is_atom_aromatic, is_bond_aromatic
            )

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

        case_4_matches = match_smirks(
            case_4_smirks, molecule, is_atom_aromatic, is_bond_aromatic, unique=True
        )
        case_4_atoms = {
            match for matches in case_4_matches for match in matches.values()
        }

        cls._set_aromatic(
            case_4_matches, is_bond_in_ring, is_atom_aromatic, is_bond_aromatic
        )

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

        case_5_matches = match_smirks(
            case_5_smirks, molecule, is_atom_aromatic, is_bond_aromatic, unique=True
        )
        case_5_matches = [
            matches
            for matches in case_5_matches
            if matches[1] not in ar_6_ar_7_matches
            and matches[2] not in ar_6_ar_7_matches
        ]

        cls._set_aromatic(
            case_5_matches, is_bond_in_ring, is_atom_aromatic, is_bond_aromatic
        )

        return is_atom_aromatic, is_bond_aromatic

    @classmethod
    def apply(
        cls, molecule: "Molecule", model: AromaticityModels
    ) -> Tuple[Dict[int, bool], Dict[Tuple[int, int], bool]]:
        """Returns whether each atom and bond in a molecule is aromatic or not according
        to a given aromaticity model.

        Parameters
        ----------
        molecule
            The molecule to generate aromatic flags for.
        model
            The aromaticity model to apply.

        Returns
        -------
            A dictionary of the form ``is_atom_aromatic[atom_index] = is_aromatic`` and
            ``is_bond_aromatic[(atom_index_a, atom_index_b)] = is_aromatic``.
        """

        if model == AromaticityModels.AM1BCC:
            is_atom_aromatic, is_bond_aromatic = cls._assign_am1bcc(molecule)
        elif model == AromaticityModels.MDL:
            is_atom_aromatic, is_bond_aromatic = apply_mdl_aromaticity_model(molecule)
        else:
            raise NotImplementedError()

        return is_atom_aromatic, is_bond_aromatic
