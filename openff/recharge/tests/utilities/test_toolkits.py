import numpy
import pytest
from openff.toolkit import Molecule
from openff.units import unit

from openff.recharge.utilities.toolkits import (
    _oe_apply_mdl_aromaticity_model,
    _oe_get_atom_symmetries,
    _oe_match_smirks,
    _oe_molecule_to_tagged_smiles,
    _rd_apply_mdl_aromaticity_model,
    _rd_get_atom_symmetries,
    _rd_match_smirks,
    _rd_molecule_to_tagged_smiles,
    apply_mdl_aromaticity_model,
    compute_vdw_radii,
    get_atom_symmetries,
    match_smirks,
    molecule_to_tagged_smiles,
)


@pytest.mark.parametrize(
    "match_function", [match_smirks, _oe_match_smirks, _rd_match_smirks]
)
@pytest.mark.parametrize(
    "smiles, smirks, is_atom_aromatic, is_bond_aromatic, unique, expected_matches",
    [
        (
            "[C:1]([H:2])([H:3])([H:4])([H:5])",
            "[#6:1]-[#1:2]",
            None,
            None,
            True,
            [(0, 1), (0, 2), (0, 3), (0, 4)],
        ),
        (
            "[H:6][C:2]1=[C:1]([C:4](=[C:3]1[H:7])[H:8])[H:5]",
            "C1=CC=C1",
            None,
            None,
            True,
            [tuple()],
        ),
        (
            "[H:6][C:2]1=[C:1]([C:4](=[C:3]1[H:7])[H:8])[H:5]",
            "C1=CC=C1",
            None,
            None,
            False,
            [tuple(), tuple(), tuple(), tuple()],
        ),
        (
            "[H:6][C:2]1=[C:1]([C:4](=[C:3]1[H:7])[H:8])[H:5]",
            "[a:1]",
            {
                0: False,
                1: False,
                2: False,
                3: False,
                4: False,
                5: False,
                6: False,
                7: False,
            },
            {
                (0, 1): False,
                (1, 2): False,
                (2, 3): False,
                (3, 0): False,
                (0, 4): False,
                (1, 5): False,
                (2, 6): False,
                (3, 7): False,
            },
            True,
            [],
        ),
        (
            "[H:6][C:2]1=[C:1]([C:4](=[C:3]1[H:7])[H:8])[H:5]",
            "[a:1]",
            {
                0: True,
                1: True,
                2: True,
                3: True,
                4: False,
                5: False,
                6: False,
                7: False,
            },
            {
                (0, 1): True,
                (1, 2): True,
                (2, 3): True,
                (3, 0): True,
                (0, 4): False,
                (1, 5): False,
                (2, 6): False,
                (3, 7): False,
            },
            True,
            [(0,), (1,), (2,), (3,)],
        ),
    ],
)
def test_match_smirks(
    match_function,
    smiles,
    smirks,
    is_atom_aromatic,
    is_bond_aromatic,
    unique,
    expected_matches,
):
    """Test that the correct exception is raised when an invalid smirks
    pattern is provided to `match_smirks`."""

    molecule = Molecule.from_mapped_smiles(smiles)

    is_atom_aromatic = (
        {i: atom.is_aromatic for i, atom in enumerate(molecule.atoms)}
        if not is_atom_aromatic
        else is_atom_aromatic
    )
    is_bond_aromatic = (
        {
            (bond.atom1_index, bond.atom2_index): bond.is_aromatic
            for bond in molecule.bonds
        }
        if not is_bond_aromatic
        else is_bond_aromatic
    )

    matches = match_function(
        smirks, molecule, is_atom_aromatic, is_bond_aromatic, unique
    )

    assert len(matches) == len(expected_matches)
    assert {tuple(match[i] for i in range(len(match))) for match in matches} == {
        *expected_matches
    }


@pytest.mark.parametrize(
    "match_function", [match_smirks, _oe_match_smirks, _rd_match_smirks]
)
def test_match_smirks_invalid(match_function):
    """Test that the correct exception is raised when an invalid smirks
    pattern is provided to `match_smirks`."""

    molecule = Molecule.from_smiles("C")

    with pytest.raises(AssertionError, match="failed to parse X"):
        match_function(
            "X",
            molecule,
            {i: False for i in range(molecule.n_atoms)},
            {(bond.atom1_index, bond.atom2_index): False for bond in molecule.bonds},
            False,
        )


def test_compute_vdw_radii():
    molecule = Molecule.from_mapped_smiles("[C:1]([H:2])([H:3])([H:4])([H:5])")
    radii = compute_vdw_radii(molecule)

    assert numpy.allclose([1.7, 1.2, 1.2, 1.2, 1.2] * unit.angstrom, radii)


@pytest.mark.parametrize(
    "apply_function",
    [
        apply_mdl_aromaticity_model,
        _rd_apply_mdl_aromaticity_model,
        _oe_apply_mdl_aromaticity_model,
    ],
)
@pytest.mark.parametrize(
    "smiles, expected_is_atom_aromatic, expected_is_bond_aromatic",
    [
        (
            "[C:1]([H:2])([H:3])([H:4])([H:5])",
            {0: False, 1: False, 2: False, 3: False, 4: False},
            {(0, 1): False, (0, 2): False, (0, 3): False, (0, 4): False},
        ),
        (
            "[H:9][c:3]1[c:2]([c:1]([c:6]([c:5]([c:4]1[H:10])[H:11])[H:12])[H:7])[H:8]",
            {
                0: True,
                1: True,
                2: True,
                3: True,
                4: True,
                5: True,
                6: False,
                7: False,
                8: False,
                9: False,
                10: False,
                11: False,
            },
            {
                (0, 1): True,
                (0, 5): True,
                (0, 6): False,
                (1, 2): True,
                (1, 7): False,
                (2, 3): True,
                (2, 8): False,
                (3, 4): True,
                (3, 9): False,
                (4, 5): True,
                (4, 10): False,
                (5, 11): False,
            },
        ),
        (
            "[H:7][C:2]1=[C:1]([O:5][C:4](=[C:3]1[H:8])[H:9])[H:6]",
            {
                0: False,
                1: False,
                2: False,
                3: False,
                4: False,
                5: False,
                6: False,
                7: False,
                8: False,
            },
            {
                (0, 1): False,
                (0, 4): False,
                (0, 5): False,
                (1, 2): False,
                (1, 6): False,
                (2, 3): False,
                (2, 7): False,
                (3, 4): False,
                (3, 8): False,
            },
        ),
    ],
)
def test_apply_mdl_aromaticity_model(
    apply_function, smiles, expected_is_atom_aromatic, expected_is_bond_aromatic
):
    is_atom_aromatic, is_bond_aromatic = apply_function(
        Molecule.from_mapped_smiles(smiles)
    )

    assert is_atom_aromatic == expected_is_atom_aromatic
    assert is_bond_aromatic == expected_is_bond_aromatic


@pytest.mark.parametrize(
    "get_symmetries_func",
    [_oe_get_atom_symmetries, _rd_get_atom_symmetries, get_atom_symmetries],
)
def test_get_atom_symmetries(get_symmetries_func):
    molecule = Molecule.from_mapped_smiles("[H:1][C:2]([H:3])([H:4])[O:5][H:6]")

    try:
        atom_symmetries = get_symmetries_func(molecule)

    except ModuleNotFoundError as e:
        pytest.skip(f"missing optional dependency - {e.name}")
        return

    assert len({atom_symmetries[i] for i in (0, 2, 3)}) == 1
    assert len({atom_symmetries[i] for i in (1, 4, 5)}) == 3


@pytest.mark.parametrize(
    "molecule_to_tagged_smiles_func",
    [
        _oe_molecule_to_tagged_smiles,
        _rd_molecule_to_tagged_smiles,
        molecule_to_tagged_smiles,
    ],
)
def test_molecule_to_tagged_smiles(molecule_to_tagged_smiles_func):
    molecule = Molecule.from_mapped_smiles("[H:1][C:2]([H:3])([H:4])[O:5][H:6]")

    try:
        tagged_smiles = molecule_to_tagged_smiles_func(molecule, [1, 2, 1, 1, 3, 4])

    except ModuleNotFoundError as e:
        pytest.skip(f"missing optional dependency - {e.name}")
        return

    assert tagged_smiles == "[H:1][C:2]([H:1])([H:1])[O:3][H:4]"

    # Do a quick canary test to make sure the toolkit doesn't stop parsing molecules
    # with duplicate indices correctly
    recreated_map = Molecule.from_smiles(tagged_smiles).properties["atom_map"]
    assert len(recreated_map) == 6 and {*recreated_map.values()} == {1, 2, 3, 4}
