import numpy



def compute_inverse_distance_matrix(points_a: numpy.ndarray, points_b: numpy.ndarray):
    """Computes a matrix of the inverse distances between all of the points
    in ``points_a`` and all of the points in ``points_b``.

    Notes
    -----
    This function is not currently vectorised but may be in the future.

    Parameters
    ----------
    points_a
        The first list of points with shape=(n_points_a, n_dim).
    points_b
        The second list of points with shape=(n_points_b, n_dim).

    Returns
    -------
        The squared distance matrix where ``d_ij = 1.0 / sqrt((a_i - b_i) ^ 2)``
        with shape=(n_points_a, n_points_b).
    """

    inverse_distances = numpy.zeros((len(points_a), len(points_b)))

    for i in range(len(points_a)):

        for j in range(len(points_b)):

            distance = numpy.sqrt(
                numpy.sum((points_a[i] - points_b[j]) * (points_a[i] - points_b[j]))
            )
            inverse_distances[i, j] = 1.0 / distance

    return inverse_distances
