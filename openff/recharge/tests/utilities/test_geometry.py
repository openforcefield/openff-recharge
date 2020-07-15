import numpy

from openff.recharge.utilities.geometry import (
    compute_inverse_distance_matrix,
    reorder_conformer,
)
from openff.recharge.utilities.openeye import smiles_to_molecule


def test_compute_inverse_distance_matrix():

    points_a = numpy.array([[0.0, 1.0], [2.0, 3.0]])
    points_b = numpy.array([[0.0, 2.0], [4.0, 6.0], [6.0, 8.0]])

    inverse_distances = compute_inverse_distance_matrix(points_a, points_b)
    assert inverse_distances.shape == (2, 3)

    expected = numpy.array(
        [[1.0000000, 0.1561738, 0.1084652], [0.4472136, 0.2773501, 0.1561738]]
    )

    assert numpy.allclose(expected, inverse_distances)


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
