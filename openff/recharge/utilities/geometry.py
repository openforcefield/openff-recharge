from typing import TYPE_CHECKING

import numpy

if TYPE_CHECKING:
    from openeye import oechem

BOHR_TO_ANGSTROM = 0.529177210903  # NIST 2018 CODATA
INVERSE_ANGSTROM_TO_BOHR = BOHR_TO_ANGSTROM

ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM


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


def compute_field_vector(
    point_a: numpy.ndarray, point_b: numpy.ndarray
) -> numpy.ndarray:
    """Computes a vector which points from ``point_a`` to ``point_b`` and has
    a magnitude equal to the inverse squared distance between the points.

    Parameters
    ----------
    point_a
        The first point with shape=(1, n_dim).
    point_b
        The second point with shape=(1, n_dim).

    Returns
    -------
        The computed field vector with shape=(1, n_dim).
    """

    direction_vector = point_b - point_a
    direction_norm = numpy.linalg.norm(direction_vector)

    field_vector = direction_vector / (direction_norm * direction_norm * direction_norm)

    return field_vector


def compute_vector_field(
    points_a: numpy.ndarray, points_b: numpy.ndarray
) -> numpy.ndarray:
    """Computes a tensor containing the vectors which point from all of the points in
    ``points_a`` to all of the points in ``points_b`` and have magnitudes equal to
    the inverse squared distance between the points.

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
        The vector field tensor with shape=(n_points_b, n_dim, n_points_a) and where
        ``tensor[i, :, j] = (b_i - a_j) /  ||b_i - a_j|| ^ 3)``
    """

    assert points_a.shape[1] == points_b.shape[1]

    field_tensor = numpy.zeros((*points_b.shape, points_a.shape[0]))

    for j in range(len(points_a)):
        for i in range(len(points_b)):

            field_tensor[i, :, j] = compute_field_vector(points_a[j], points_b[i])

    return field_tensor


def reorder_conformer(
    oe_molecule: "oechem.OEMol", conformer: numpy.ndarray
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
