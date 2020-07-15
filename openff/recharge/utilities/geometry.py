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


def reorder_conformer(
    oe_molecule: oechem.OEMol, conformer: numpy.ndarray
) -> numpy.ndarray:
    """Reorder a conformer to match the ordering of the atoms
    in a molecule. The map index on each atom in the molecule
    should match the original orderings of the atoms for which
    the conformer was generated."""

    index_map = {atom.GetIdx(): atom.GetMapIdx() for atom in oe_molecule.GetAtoms()}

    indices = numpy.array(
        [index_map[index] - 1 for index in range(oe_molecule.NumAtoms())]
    )

    reordered_conformer = conformer[indices]
    return reordered_conformer
