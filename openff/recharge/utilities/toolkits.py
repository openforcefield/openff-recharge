"""Uniform API for performing common cheminformatics tasks / queries"""
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Tuple, cast

import numpy
from openff.toolkit.utils import ToolkitUnavailableException
from openff.units import unit
from openff.units.elements import SYMBOLS
from openff.utilities import MissingOptionalDependencyError, requires_package
from openff.utilities.utilities import requires_oe_module

if TYPE_CHECKING:
    from openff.toolkit import Molecule


class VdWRadiiType(Enum):
    Bondi = "Bondi"


def _bond_key(index_a: int, index_b: int) -> Tuple[int, int]:
    return cast(Tuple[int, int], tuple(sorted((index_a, index_b))))


@requires_oe_module("oechem")
def _oe_match_smirks(
    smirks: str,
    molecule: "Molecule",
    is_atom_aromatic: Dict[int, bool],
    is_bond_aromatic: Dict[Tuple[int, int], bool],
    unique: bool,
    kekulize: bool = False,
) -> List[Dict[int, int]]:
    oe_molecule = molecule.to_openeye()

    oe_atoms = {oe_atom.GetIdx(): oe_atom for oe_atom in oe_molecule.GetAtoms()}
    oe_bonds = {
        _bond_key(bond.GetBgnIdx(), bond.GetEndIdx()): bond
        for bond in oe_molecule.GetBonds()
    }

    for index, is_aromatic in is_atom_aromatic.items():
        oe_atoms[index].SetAromatic(is_aromatic)
    for indices, is_aromatic in is_bond_aromatic.items():
        oe_bonds[_bond_key(*indices)].SetAromatic(is_aromatic)

    from openeye import oechem

    query = oechem.OEQMol()
    assert oechem.OEParseSmarts(query, smirks), f"failed to parse {smirks}"

    substructure_search = oechem.OESubSearch(query)
    substructure_search.SetMaxMatches(0)

    matches = []

    for match in substructure_search.Match(oe_molecule, unique):
        matched_indices = {
            atom_match.pattern.GetMapIdx() - 1: atom_match.target.GetIdx()
            for atom_match in match.GetAtoms()
            if atom_match.pattern.GetMapIdx() != 0
        }

        matches.append(matched_indices)

    return matches


@requires_package("rdkit")
def _rd_match_smirks(
    smirks: str,
    molecule: "Molecule",
    is_atom_aromatic: Dict[int, bool],
    is_bond_aromatic: Dict[Tuple[int, int], bool],
    unique: bool,
    kekulize: bool = False
) -> List[Dict[int, int]]:
    from rdkit import Chem

    rd_molecule: Chem.Mol = molecule.to_rdkit()
    Chem.SanitizeMol(rd_molecule, Chem.SANITIZE_ALL ^ Chem.SANITIZE_SETAROMATICITY)
    if kekulize:
        Chem.Kekulize(rd_molecule)

    rd_atoms = {rd_atom.GetIdx(): rd_atom for rd_atom in rd_molecule.GetAtoms()}
    rd_bonds = {
        _bond_key(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()): bond
        for bond in rd_molecule.GetBonds()
    }

    for index, is_aromatic in is_atom_aromatic.items():
        rd_atoms[index].SetIsAromatic(is_aromatic)
    for indices, is_aromatic in is_bond_aromatic.items():
        rd_bonds[_bond_key(*indices)].SetIsAromatic(is_aromatic)

    query = Chem.MolFromSmarts(smirks)
    assert query is not None, f"failed to parse {smirks}"

    max_matches = numpy.iinfo(numpy.uintc).max

    full_matches = rd_molecule.GetSubstructMatches(
        query, uniquify=unique, maxMatches=max_matches, useChirality=True
    )

    matches = [
        {
            query_atom.GetAtomMapNum() - 1: index
            for index, query_atom in zip(match, query.GetAtoms())
            if query_atom.GetAtomMapNum() != 0
        }
        for match in full_matches
    ]

    return matches


def match_smirks(
    smirks: str,
    molecule: "Molecule",
    is_atom_aromatic: Dict[int, bool],
    is_bond_aromatic: Dict[Tuple[int, int], bool],
    unique: bool = False,
    kekulize: bool = False,
) -> List[Dict[int, int]]:
    """Attempt to find the indices (optionally unique) of all atoms which
    match a particular SMIRKS pattern.

    Parameters
    ----------
    smirks
        The SMIRKS pattern to match.
    molecule
        The molecule to match against.
    is_atom_aromatic
        A dictionary of the form ``is_atom_aromatic[atom_index] = is_aromatic``.
    is_bond_aromatic
        A dictionary of the form
        ``is_bond_aromatic[(atom_index_a, atom_index_b)] = is_aromatic``.
    unique
        Whether to return only unique matches.
    kekulize
        Whether to kekulize the molecule before matching.
        This is only supported by the RDKit backend.

    Returns
    -------
        A list of all the matches where each match is stored as a dictionary of
        the smirks indices and their corresponding matched atom indices.
    """

    try:
        return _oe_match_smirks(
            smirks, molecule, is_atom_aromatic, is_bond_aromatic, unique, kekulize
        )
    except (
        ModuleNotFoundError,
        MissingOptionalDependencyError,
        ToolkitUnavailableException,
    ):
        return _rd_match_smirks(
            smirks, molecule, is_atom_aromatic, is_bond_aromatic, unique, kekulize
        )


def compute_vdw_radii(
    molecule: "Molecule", radii_type: VdWRadiiType = VdWRadiiType.Bondi
) -> unit.Quantity:
    """Computes the vdW radii of each atom in a molecule

    Parameters
    ----------
    molecule
        The molecule containing the atoms
    radii_type
        The type of vdW radii to compute.

    Returns
    -------
        A list of the vdW radii [A] of each atom.
    """

    if radii_type == VdWRadiiType.Bondi:
        _BONDI_RADII = {
            "H": 1.20,
            "C": 1.70,
            "N": 1.55,
            "O": 1.52,
            "F": 1.47,
            "P": 1.80,
            "S": 1.80,
            "Cl": 1.75,
            "Br": 1.85,
            "I": 1.98,
            "He": 1.40,
            "Ar": 1.88,
            "Na": 2.27,
            "K": 1.75,
        }

        return [
            _BONDI_RADII[SYMBOLS[atom.atomic_number]] for atom in molecule.atoms
        ] * unit.angstrom
    else:
        raise NotImplementedError()


@requires_oe_module("oechem")
def _oe_apply_mdl_aromaticity_model(
    molecule: "Molecule",
) -> Tuple[Dict[int, bool], Dict[Tuple[int, int], bool]]:
    oe_molecule = molecule.to_openeye()

    from openeye import oechem

    oechem.OEClearAromaticFlags(oe_molecule)
    oechem.OEAssignAromaticFlags(oe_molecule, oechem.OEAroModel_MDL)

    is_atom_aromatic = {
        atom.GetIdx(): atom.IsAromatic() for atom in oe_molecule.GetAtoms()
    }
    is_bond_aromatic = {
        (bond.GetBgnIdx(), bond.GetEndIdx()): bond.IsAromatic()
        for bond in oe_molecule.GetBonds()
    }

    return is_atom_aromatic, is_bond_aromatic


@requires_package("rdkit")
def _rd_apply_mdl_aromaticity_model(
    molecule: "Molecule",
) -> Tuple[Dict[int, bool], Dict[Tuple[int, int], bool]]:
    rd_molecule = molecule.to_rdkit()

    from rdkit import Chem

    Chem.SetAromaticity(rd_molecule, Chem.AromaticityModel.AROMATICITY_MDL)

    is_atom_aromatic = {
        atom.GetIdx(): atom.GetIsAromatic() for atom in rd_molecule.GetAtoms()
    }
    is_bond_aromatic = {
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()): bond.GetIsAromatic()
        for bond in rd_molecule.GetBonds()
    }

    return is_atom_aromatic, is_bond_aromatic


def apply_mdl_aromaticity_model(
    molecule: "Molecule",
) -> Tuple[Dict[int, bool], Dict[Tuple[int, int], bool]]:
    """Returns whether each atom and bond in a molecule is aromatic or not according
    to the MDL aromaticity model.

    Parameters
    ----------
    molecule
        The molecule to generate aromatic flags for.

    Returns
    -------
        A dictionary of the form ``is_atom_aromatic[atom_index] = is_aromatic`` and
        ``is_bond_aromatic[(atom_index_a, atom_index_b)] = is_aromatic``.
    """

    try:
        return _oe_apply_mdl_aromaticity_model(molecule)
    except (
        ModuleNotFoundError,
        MissingOptionalDependencyError,
        ToolkitUnavailableException,
    ):
        return _rd_apply_mdl_aromaticity_model(molecule)


@requires_oe_module("oechem")
def _oe_get_atom_symmetries(molecule: "Molecule") -> List[int]:
    from openeye import oechem

    oe_mol = molecule.to_openeye()
    oechem.OEPerceiveSymmetry(oe_mol)

    symmetry_classes_by_index = {
        a.GetIdx(): a.GetSymmetryClass() for a in oe_mol.GetAtoms()
    }
    return [symmetry_classes_by_index[i] for i in range(molecule.n_atoms)]


@requires_package("rdkit")
def _rd_get_atom_symmetries(molecule: "Molecule") -> List[int]:
    from rdkit import Chem

    rd_mol = molecule.to_rdkit()
    return list(Chem.CanonicalRankAtoms(rd_mol, breakTies=False))


def get_atom_symmetries(molecule: "Molecule") -> List[int]:
    """Returns indices that describe which atoms in a molecule a topologically
    symmetrical.

    Parameters
    ----------
    molecule
        The molecule of interest.

    Returns
    -------
        A list of `n_atom` indices, whereby atoms with the same index are considered
        to be topologically symmetrical.
    """

    try:
        return _oe_get_atom_symmetries(molecule)
    except (
        ImportError,
        ToolkitUnavailableException,
        MissingOptionalDependencyError,
    ):
        return _rd_get_atom_symmetries(molecule)


@requires_oe_module("oechem")
def _oe_molecule_to_tagged_smiles(molecule: "Molecule", indices: List[int]) -> str:
    from openeye import oechem

    oe_mol: oechem.OEMol = molecule.to_openeye()
    oe_atoms: Dict[int, oechem.OEAtomBase] = {
        oe_atom.GetIdx(): oe_atom for oe_atom in oe_mol.GetAtoms()
    }

    for i, index in enumerate(indices):
        oe_atoms[i].SetMapIdx(index)

    return oechem.OEMolToSmiles(oe_mol)


@requires_package("rdkit")
def _rd_molecule_to_tagged_smiles(molecule: "Molecule", indices: List[int]) -> str:
    from rdkit import Chem

    rd_mol: Chem.Mol = molecule.to_rdkit()
    rd_atoms: Dict[int, Chem.Atom] = {
        rd_atom.GetIdx(): rd_atom for rd_atom in rd_mol.GetAtoms()
    }

    for i, index in enumerate(indices):
        rd_atoms[i].SetAtomMapNum(index)

    return Chem.MolToSmiles(rd_mol)


def molecule_to_tagged_smiles(molecule: "Molecule", indices: List[int]) -> str:
    """Returns a SMILES pattern whereby each atom has been tagged with a specified
    index

    Parameters
    ----------
    molecule
        The molecule of interest.
    indices
        The indices to associate with each atom in the molecule.

    Returns
    -------
        A list of `n_atom` indices, whereby atoms with the same index are considered
        to be topologically symmetrical.
    """

    assert (
        len(indices) == molecule.n_atoms
    ), "number of atom indices does not match number of atoms"
    assert all(index > 0 for index in indices), "all indices must be > 0"

    try:
        return _oe_molecule_to_tagged_smiles(molecule, indices)
    except (
        ImportError,
        ToolkitUnavailableException,
        MissingOptionalDependencyError,
    ):
        return _rd_molecule_to_tagged_smiles(molecule, indices)
