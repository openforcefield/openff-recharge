from collections import defaultdict
from typing import List

import numpy
import pytest
from openff.toolkit.topology import Molecule

from openff.recharge.charges.library import LibraryChargeParameter
from openff.recharge.charges.resp._resp import (
    _deduplicate_constraints,
    _generate_dummy_values,
    generate_resp_systems_of_equations,
    molecule_to_resp_library_charge,
)
from openff.recharge.esp import ESPSettings
from openff.recharge.esp.storage import MoleculeESPRecord
from openff.recharge.grids import GridSettings
from openff.recharge.optimize import ESPObjective


@pytest.fixture()
def mock_esp_records() -> List[MoleculeESPRecord]:

    conformer = numpy.array(
        [
            [-0.5, +0.0, +0.0],
            [+0.5, +0.0, +0.0],
            [-0.5, +1.0, +0.0],
            [+0.5, +1.0, +0.0],
            [-0.5, +2.0, +0.0],
            [+0.5, +2.0, +0.0],
            [-0.5, -1.0, +0.0],
            [-0.5, +0.0, -1.0],
            [+0.5, -1.0, +0.0],
            [+0.5, +0.0, +1.0],
        ]
    )

    grid = numpy.array([[0.0, 5.0, 0.0], [0.0, -5.0, 0.0]])
    esp = numpy.array([[2.0], [4.0]])

    return [
        MoleculeESPRecord(
            tagged_smiles=(
                "[C:1]([H:7])([H:8])([O:3][H:5])[C:2]([H:9])([H:10])([O:4][H:6])"
            ),
            conformer=conformer,
            grid_coordinates=grid,
            esp=esp,
            electric_field=None,
            esp_settings=ESPSettings(grid_settings=GridSettings()),
        ),
        MoleculeESPRecord(
            tagged_smiles=(
                "[C:1]([H:7])([H:8])([O:3][H:5])[C:2]([H:9])([H:10])([O:4][H:6])"
            ),
            conformer=conformer,
            grid_coordinates=grid * 2,
            esp=esp / 2.0,
            electric_field=None,
            esp_settings=ESPSettings(grid_settings=GridSettings()),
        ),
    ]


@pytest.mark.parametrize(
    "smiles, expected_values",
    [
        ("[Cl:1][H:2]", [0.0, 0.0]),
        ("[O-:1][H:2]", [-0.5, -0.5]),
        ("[N+:1]([H:2])([H:2])([H:2])([H:2])", [0.2, 0.2]),
        ("[N+:1]([H:2])([H:3])([H:4])([H:5])", [0.2, 0.2, 0.2, 0.2, 0.2]),
        ("[N+:1]([H:2])([H:2])[O-:3]", [0.0, 0.0, 0.0]),
        (
            "[H:1][c:9]1[c:10]([c:13]([c:16]2[c:15]([c:11]1[H:3])[c:12]([c:14]"
            "([c:17]([n+:20]2[C:19]([H:8])([H:8])[H:8])[C:18]([H:7])([H:7])[H:7])"
            "[H:6])[H:4])[H:5])[H:2]",
            [1.0 / 24.0] * 20,
        ),
    ],
)
def test_generate_dummy_values(smiles, expected_values):

    from simtk import unit as simtk_unit

    actual_values = _generate_dummy_values(smiles)
    assert actual_values == expected_values

    molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)

    total_charge = molecule.total_charge.value_in_unit(simtk_unit.elementary_charge)
    sum_charge = sum(
        actual_values[i - 1] for i in molecule.properties["atom_map"].values()
    )

    assert numpy.isclose(total_charge, sum_charge)


@pytest.mark.parametrize(
    "input_smiles, "
    "equivalize_within_methyl_carbons, "
    "equivalize_within_methyl_hydrogens, "
    "equivalize_within_other_heavy_atoms, "
    "equivalize_within_other_hydrogen_atoms, "
    "expected_groupings, "
    "expected_methyl_carbons, "
    "expected_methyl_hydrogens, "
    "expected_heavy_atoms, "
    "expected_hydrogens",
    [
        (
            "[C:1]([H:2])([H:3])([H:4])[H:5]",
            False,
            False,
            False,
            False,
            [(0,), (1,), (2,), (3,), (4,)],
            [0],
            [1, 2, 3, 4],
            [],
            [],
        ),
        (
            "[C:1]([H:2])([H:3])([H:4])[H:5]",
            False,
            True,
            False,
            False,
            [(0,), (1, 2, 3, 4)],
            [0],
            [1, 2, 3, 4],
            [],
            [],
        ),
        (
            "[C:1]([H:3])([H:4])([H:5])[C:2]([H:6])([H:7])([H:8])",
            False,
            False,
            False,
            False,
            [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,)],
            [0, 1],
            [2, 3, 4, 5, 6, 7],
            [],
            [],
        ),
        (
            "[C:1]([H:3])([H:4])([H:5])[C:2]([H:6])([H:7])([H:8])",
            True,
            False,
            False,
            False,
            [(0, 1), (2,), (3,), (4,), (5,), (6,), (7,)],
            [0, 1],
            [2, 3, 4, 5, 6, 7],
            [],
            [],
        ),
        (
            "[C:1]([H:3])([H:4])([H:5])[C:2]([H:6])([H:7])([H:8])",
            False,
            True,
            False,
            False,
            [(0,), (1,), (2, 3, 4, 5, 6, 7)],
            [0, 1],
            [2, 3, 4, 5, 6, 7],
            [],
            [],
        ),
        (
            "[C:1]([H:3])([H:4])([H:5])[C:2]([H:6])([H:7])([H:8])",
            True,
            True,
            False,
            False,
            [(0, 1), (2, 3, 4, 5, 6, 7)],
            [0, 1],
            [2, 3, 4, 5, 6, 7],
            [],
            [],
        ),
        (
            "[C:1]([H:7])([H:8])([O:3][H:5])[C:2]([H:9])([H:10])([O:4][H:6])",
            False,
            False,
            False,
            False,
            [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,)],
            [0, 1],
            [6, 7, 8, 9],
            [2, 3],
            [4, 5],
        ),
        (
            "[C:1]([H:7])([H:8])([O:3][H:5])[C:2]([H:9])([H:10])([O:4][H:6])",
            False,
            False,
            True,
            False,
            [(0,), (1,), (2, 3), (4,), (5,), (6,), (7,), (8,), (9,)],
            [0, 1],
            [6, 7, 8, 9],
            [2, 3],
            [4, 5],
        ),
        (
            "[C:1]([H:7])([H:8])([O:3][H:5])[C:2]([H:9])([H:10])([O:4][H:6])",
            False,
            False,
            False,
            True,
            [(0,), (1,), (2,), (3,), (4, 5), (6,), (7,), (8,), (9,)],
            [0, 1],
            [6, 7, 8, 9],
            [2, 3],
            [4, 5],
        ),
        (
            "[C:1]([H:7])([H:8])([O:3][H:5])[C:2]([H:9])([H:10])([O:4][H:6])",
            False,
            False,
            True,
            True,
            [(0,), (1,), (2, 3), (4, 5), (6,), (7,), (8,), (9,)],
            [0, 1],
            [6, 7, 8, 9],
            [2, 3],
            [4, 5],
        ),
        (
            "[C:1]([H:7])([H:8])([O:3][H:5])[C:2]([H:9])([H:10])([O:4][H:6])",
            True,
            True,
            True,
            True,
            [(0, 1), (2, 3), (4, 5), (6, 7, 8, 9)],
            [0, 1],
            [6, 7, 8, 9],
            [2, 3],
            [4, 5],
        ),
    ],
)
def test_molecule_to_resp_library_charge(
    input_smiles,
    equivalize_within_methyl_carbons,
    equivalize_within_methyl_hydrogens,
    equivalize_within_other_heavy_atoms,
    equivalize_within_other_hydrogen_atoms,
    expected_groupings,
    expected_methyl_carbons,
    expected_methyl_hydrogens,
    expected_heavy_atoms,
    expected_hydrogens,
):

    input_molecule: Molecule = Molecule.from_mapped_smiles(input_smiles)

    parameter = molecule_to_resp_library_charge(
        input_molecule,
        equivalize_within_methyl_carbons,
        equivalize_within_methyl_hydrogens,
        equivalize_within_other_heavy_atoms,
        equivalize_within_other_hydrogen_atoms,
    )

    output_molecule: Molecule = Molecule.from_smiles(parameter.smiles)

    _, output_to_input_index = Molecule.are_isomorphic(
        output_molecule, input_molecule, return_atom_map=True
    )

    actual_groupings_dict = defaultdict(list)

    for atom_index, map_index in output_molecule.properties["atom_map"].items():
        actual_groupings_dict[map_index].append(output_to_input_index[atom_index])

    actual_groupings = [
        tuple(sorted(group)) for group in actual_groupings_dict.values()
    ]

    assert len(actual_groupings) == len(expected_groupings)
    assert {*actual_groupings} == {*expected_groupings}

    def compare_expected(dict_key: str, expected_values: List[int]):

        actual_values = []

        for index in parameter.provenance[dict_key]:
            actual_values.extend(actual_groupings_dict[index + 1])

        assert sorted(actual_values) == sorted(expected_values)

    compare_expected("methyl-carbon-indices", expected_methyl_carbons)
    compare_expected("methyl-hydrogen-indices", expected_methyl_hydrogens)
    compare_expected("other-heavy-indices", expected_heavy_atoms)
    compare_expected("other-hydrogen-indices", expected_hydrogens)


@pytest.mark.parametrize(
    "input_matrix, input_values, expected_matrix, expected_values",
    [
        (
            numpy.array([[1, 2, 3, 4], [4, 3, 2, 1]]),
            numpy.array([[1.0], [2.0]]),
            numpy.array([[1, 2, 3, 4], [4, 3, 2, 1]]),
            numpy.array([[1.0], [2.0]]),
        ),
        (
            numpy.array([[1, 2, 3, 4], [1, 2, 3, 4], [4, 3, 2, 1]]),
            numpy.array([[1.0], [2.0], [3.0]]),
            numpy.array([[1, 2, 3, 4], [4, 3, 2, 1]]),
            numpy.array([[1.0], [3.0]]),
        ),
        (
            numpy.array([[1, 2, 3, 4], [4, 3, 2, 1], [1, 2, 3, 4]]),
            numpy.array([[1.0], [3.0], [2.0]]),
            numpy.array([[1, 2, 3, 4], [4, 3, 2, 1]]),
            numpy.array([[1.0], [3.0]]),
        ),
    ],
)
def test_deduplicate_constraints(
    input_matrix, input_values, expected_matrix, expected_values
):

    output_matrix, output_values = _deduplicate_constraints(input_matrix, input_values)

    assert output_matrix.shape == expected_matrix.shape
    assert numpy.allclose(output_matrix, expected_matrix)

    assert output_values.shape == expected_values.shape
    assert numpy.allclose(output_values, expected_values)


@pytest.mark.parametrize(
    "equivalize_between_methyl_carbons, "
    "equivalize_between_methyl_hydrogens, "
    "equivalize_between_other_heavy_atoms, "
    "equivalize_between_other_hydrogen_atoms, "
    "fix_methyl_carbons, "
    "fix_methyl_hydrogens, "
    "fix_other_heavy_atoms, "
    "fix_other_hydrogen_atoms, "
    "expected_design_matrix, "
    "expected_reference_values, "
    "expected_constraint_matrix, "
    "expected_constraint_values, "
    "expected_restraint_indices, "
    "expected_trainable_mapping, ",
    [
        (
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            numpy.array([[2, 4, 6, 16], [4, 8, 12, 32], [2, 4, 6, 16], [4, 8, 12, 32]]),
            numpy.array([[2], [4], [1], [2]]),
            numpy.array([[2, 2, 2, 4]]),
            numpy.array([[0.0]]),
            [0, 1],
            {0: 0, 1: 1, 2: 2, 3: 3},
        ),
        (
            False,
            True,
            True,
            False,
            #
            False,
            False,
            False,
            False,
            numpy.array(
                [
                    [2, 4, 6, 16, 0, 0],
                    [4, 8, 12, 32, 0, 0],
                    [0, 4, 0, 16, 2, 6],
                    [0, 8, 0, 32, 4, 12],
                ]
            ),
            numpy.array([[2], [4], [1], [2]]),
            numpy.array([[2, 2, 2, 4, 0, 0], [0, 2, 0, 4, 2, 2]]),
            numpy.array([[0.0], [0.0]]),
            [0, 1],
            {0: 0, 1: 1, 2: 2, 3: 3},
        ),
        (
            True,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            numpy.array(
                [
                    [2, 4, 6, 16, 0, 0],
                    [4, 8, 12, 32, 0, 0],
                    [2, 0, 6, 0, 16, 4],
                    [4, 0, 12, 0, 32, 8],
                ]
            ),
            numpy.array([[2], [4], [1], [2]]),
            numpy.array([[2, 2, 2, 4, 0, 0], [2, 0, 2, 0, 4, 2]]),
            numpy.array([[0.0], [0.0]]),
            [0, 1],
            {0: 0, 1: 1, 2: 2, 3: 3},
        ),
        (
            True,
            False,
            True,
            True,
            False,
            False,
            True,
            True,
            numpy.array([[2, 16, 0], [4, 32, 0], [2, 0, 16], [4, 0, 32]]),
            numpy.array([[2], [4], [1], [2]]),
            numpy.array([[2, 4, 0], [2, 0, 4]]),
            numpy.array([[0.0], [0.0]]),
            [0],
            {0: 0, 1: 3},
        ),
    ],
)
def test_generate_resp_systems_of_equations(
    mock_esp_records,
    equivalize_between_methyl_carbons,
    equivalize_between_methyl_hydrogens,
    equivalize_between_other_heavy_atoms,
    equivalize_between_other_hydrogen_atoms,
    fix_methyl_carbons,
    fix_methyl_hydrogens,
    fix_other_heavy_atoms,
    fix_other_hydrogen_atoms,
    expected_design_matrix,
    expected_reference_values,
    expected_constraint_matrix,
    expected_constraint_values,
    expected_restraint_indices,
    expected_trainable_mapping,
    monkeypatch,
):

    parameter = LibraryChargeParameter(
        smiles="[C:1]([H:4])([H:4])([O:2][H:3])[C:1]([H:4])([H:4])([O:2][H:3])",
        value=[0.0] * 4,
        provenance={
            "methyl-carbon-indices": [0],
            "methyl-hydrogen-indices": [3],
            "other-heavy-indices": [1],
            "other-hydrogen-indices": [2],
        },
    )

    monkeypatch.setattr(
        ESPObjective,
        "_compute_design_matrix_precursor",
        lambda *_: numpy.array(
            [
                [1, 1, 2, 2, 3, 3, 4, 4, 4, 4],
                [2, 2, 4, 4, 6, 6, 8, 8, 8, 8],
            ]
        ),
    )

    (
        resp_design_matrix,
        resp_reference_values,
        resp_constraint_matrix,
        resp_constraint_values,
        resp_restraint_indices,
        resp_trainable_mapping,
    ) = generate_resp_systems_of_equations(
        parameter,
        mock_esp_records,
        equivalize_between_methyl_carbons,
        equivalize_between_methyl_hydrogens,
        equivalize_between_other_heavy_atoms,
        equivalize_between_other_hydrogen_atoms,
        fix_methyl_carbons,
        fix_methyl_hydrogens,
        fix_other_heavy_atoms,
        fix_other_hydrogen_atoms,
    )

    assert resp_design_matrix.shape == expected_design_matrix.shape
    assert numpy.allclose(resp_design_matrix, expected_design_matrix)
    assert resp_reference_values.shape == expected_reference_values.shape
    assert numpy.allclose(resp_reference_values, expected_reference_values)

    assert resp_constraint_matrix.shape == expected_constraint_matrix.shape
    assert numpy.allclose(resp_constraint_matrix, expected_constraint_matrix)
    assert resp_constraint_values.shape == expected_constraint_values.shape
    assert numpy.allclose(resp_constraint_values, expected_constraint_values)

    assert sorted(expected_restraint_indices) == sorted(resp_restraint_indices)
    assert expected_trainable_mapping == resp_trainable_mapping
