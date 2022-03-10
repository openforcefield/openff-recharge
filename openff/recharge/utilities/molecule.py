import typing
from typing import TYPE_CHECKING, List, Tuple

from openff.units import unit

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule


def smiles_to_molecule(smiles: str, guess_stereochemistry: bool = False) -> "Molecule":
    """Attempts to parse a smiles pattern into a molecule object.

    Parameters
    ----------
    smiles
        The smiles pattern to parse.
    guess_stereochemistry
        If true, the stereochemistry of molecules which is not defined in the SMILES
        pattern will be guessed by enumerating possible stereoisomers and selecting
        the first one in the list.

    Returns
    -------
    The parsed molecule.
    """
    from openff.toolkit.topology import Molecule
    from openff.toolkit.utils import UndefinedStereochemistryError

    try:
        molecule = Molecule.from_smiles(smiles)
    except UndefinedStereochemistryError:

        if not guess_stereochemistry:
            raise

        molecule: Molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)

        stereoisomers = molecule.enumerate_stereoisomers(
            undefined_only=True, max_isomers=1
        )

        if len(stereoisomers) > 0:
            # We would ideally raise an exception here if the number of stereoisomers
            # is zero, however due to the way that the OFF toolkit perceives pyramidal
            # nitrogen stereocenters these would show up as undefined stereochemistry
            # but have no enumerated stereoisomers.
            molecule = stereoisomers[0]

    return molecule


def find_ring_bonds(molecule: "Molecule") -> typing.Dict[Tuple[int, int], bool]:
    """Finds all bonds that are parts of a ring system."""

    is_in_ring = {
        typing.cast(Tuple[int, int], tuple(sorted(match))): True
        for match in molecule.chemical_environment_matches("[*:1]@[*:2]", unique=True)
    }

    for bond in molecule.bonds:

        indices = typing.cast(
            Tuple[int, int], tuple(sorted((bond.atom1_index, bond.atom2_index)))
        )

        if indices in is_in_ring:
            continue

        is_in_ring[indices] = False

    return is_in_ring


def extract_conformers(molecule: "Molecule") -> List[unit.Quantity]:
    """Extracts the conformers of a molecule.

    Parameters
    ----------
    molecule
        The molecule to extract the conformers from.

    Returns
    -------
        A list of the extracted conformers [A], where each conformer is a numpy array
        with shape=(n_atoms, 3).
    """

    from simtk import unit as simtk_unit

    return [
        conformer.value_in_unit(simtk_unit.angstrom) * unit.angstrom
        for conformer in molecule.conformers
    ]
