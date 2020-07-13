import numpy

from openff.recharge.utilities.geometry import inverse_distance_matrix


def test_inverse_distance_matrix():

    points_a = numpy.array([[0.0, 1.0], [2.0, 3.0]])
    points_b = numpy.array([[0.0, 2.0], [4.0, 6.0], [6.0, 8.0]])

    inverse_distances = inverse_distance_matrix(points_a, points_b)
    assert inverse_distances.shape == (2, 3)

    expected = numpy.array(
        [[1.0000000, 0.1561738, 0.1084652], [0.4472136, 0.2773501, 0.1561738]]
    )

    assert numpy.allclose(expected, inverse_distances)
