import numpy
import pytest

from openff.recharge.utilities.geometry import (
    compute_inverse_distance_matrix,
    compute_vector_field,
    reorder_conformer,
)
from openff.recharge.utilities.openeye import smiles_to_molecule

try:
    import torch
except ImportError:
    torch = None

tensor_types = [numpy.array] + ([] if torch is None else [torch.tensor])


@pytest.mark.parametrize("tensor_type", tensor_types)
def test_compute_inverse_distance_matrix(tensor_type):

    points_a = tensor_type([[0.0, 1.0], [2.0, 3.0]])
    points_b = tensor_type([[0.0, 2.0], [4.0, 6.0], [6.0, 8.0]])

    expected_values = tensor_type(
        [[1.0000000, 0.1561738, 0.1084652], [0.4472136, 0.2773501, 0.1561738]]
    )

    inverse_distances = compute_inverse_distance_matrix(points_a, points_b)
    assert inverse_distances.shape == expected_values.shape

    assert numpy.allclose(expected_values, inverse_distances)


@pytest.mark.parametrize("tensor_type", tensor_types)
def test_compute_vector_field(tensor_type):

    points_a = tensor_type([[0.0, 3.0, 0.0], [0.0, 0.0, 4.0]])
    points_b = tensor_type([[4.0, 0.0, 0.0]])

    vector_field = compute_vector_field(points_a, points_b)
    assert vector_field.shape == (1, 3, 2)

    root_2 = numpy.sqrt(2)

    expected_output = tensor_type(
        [
            [
                [+4.0 / 5.0 ** 3, +4.0 / (root_2 * 4.0) ** 3],
                [-3.0 / 5.0 ** 3, +0.0],
                [+0.0, -4.0 / (root_2 * 4.0) ** 3],
            ]
        ]
    )

    assert vector_field.shape == expected_output.shape
    assert numpy.allclose(expected_output, vector_field)


def test_reorder_conformer():
    """Tests that conformers can be correctly re-ordered."""

    # Check the simple case where no re-ordering should occur.
    oe_molecule = smiles_to_molecule("[C:1]([H:2])([H:3])([H:4])[H:5]")
    conformer = numpy.array([[index, 0.0, 0.0] for index in range(5)])

    reordered_conformer = reorder_conformer(oe_molecule, conformer)
    assert numpy.allclose(reordered_conformer, conformer)

    # Check a case where things are actually re-ordered simple case where no
    # re-ordering should occur.
    oe_molecule = smiles_to_molecule("[C:5]([H:1])([H:4])([H:2])[H:3]")

    conformer = numpy.array([[index, 0.0, 0.0] for index in range(5)])
    expected_conformer = conformer[numpy.array([4, 0, 3, 1, 2])]

    reordered_conformer = reorder_conformer(oe_molecule, conformer)
    assert numpy.allclose(reordered_conformer, expected_conformer)
