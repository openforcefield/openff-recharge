import numpy
import pytest
from openff.toolkit.topology import Molecule
from openff.toolkit.utils import UndefinedStereochemistryError

from openff.recharge.tests import does_not_raise
from openff.recharge.utilities.molecule import (
    extract_conformers,
    find_ring_bonds,
    smiles_to_molecule,
)


@pytest.mark.parametrize(
    "guess_stereochemistry, expected_raises",
    [(False, pytest.raises(UndefinedStereochemistryError)), (True, does_not_raise())],
)
def test_smiles_to_molecule(guess_stereochemistry, expected_raises):
    """Tests that the `smiles_to_molecule` behaves as expected."""

    with expected_raises:
        molecule = smiles_to_molecule("ClC(Br)=C(F)Cl", guess_stereochemistry)
        assert Molecule.are_isomorphic(
            molecule,
            Molecule.from_smiles("ClC(Br)=C(F)Cl", allow_undefined_stereo=True),
            bond_stereochemistry_matching=False,
        )[0]


@pytest.mark.parametrize(
    "smiles, expected_value",
    [
        (
            "[C:1]([H:2])([H:3])([H:4])([H:5])",
            {(0, 1): False, (0, 2): False, (0, 3): False, (0, 4): False},
        ),
        (
            "[H:6][C:2]1=[C:1]([C:4](=[C:3]1[H:7])[H:8])[H:5]",
            {
                (0, 1): True,
                (1, 2): True,
                (2, 3): True,
                (0, 3): True,
                (0, 4): False,
                (1, 5): False,
                (2, 6): False,
                (3, 7): False,
            },
        ),
    ],
)
def test_find_ring_bonds(smiles, expected_value):

    ring_bonds = find_ring_bonds(Molecule.from_mapped_smiles(smiles))
    assert ring_bonds == expected_value


def test_extract_conformers():
    """Test that the `molecule_to_conformers` function returns
    a non-zero numpy array of the correct shape."""

    from openff.units import unit

    molecule = Molecule.from_smiles("[H][H]")
    molecule._conformers = []

    conformer = numpy.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    molecule.add_conformer(conformer * unit.angstrom)

    conformers = extract_conformers(molecule)
    assert len(conformers) == 1

    assert conformers[0].shape == conformer.shape
    assert numpy.allclose(conformers[0], conformer * unit.angstrom)
