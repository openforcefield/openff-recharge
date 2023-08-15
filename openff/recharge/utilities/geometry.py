"""Geometry centric functions such as computing pairwise distances"""
from typing import TYPE_CHECKING, overload

import numpy

from openff.recharge.utilities.tensors import inverse_cdist, pairwise_differences

if TYPE_CHECKING:
    import torch


@overload
def compute_inverse_distance_matrix(
    points_a: numpy.ndarray, points_b: numpy.ndarray
) -> numpy.ndarray:
    ...


@overload
def compute_inverse_distance_matrix(
    points_a: "torch.Tensor", points_b: "torch.Tensor"
) -> "torch.Tensor":
    ...


def compute_inverse_distance_matrix(points_a, points_b):
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

    assert type(points_a) is type(points_b)
    return inverse_cdist(points_a, points_b)


@overload
def compute_vector_field(
    points_a: numpy.ndarray, points_b: numpy.ndarray
) -> numpy.ndarray:
    ...


@overload
def compute_vector_field(
    points_a: "torch.Tensor", points_b: "torch.Tensor"
) -> "torch.Tensor":
    ...


def compute_vector_field(points_a, points_b):
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

    directions = pairwise_differences(points_a, points_b)

    inverse_distances = inverse_cdist(points_a, points_b)
    inverse_distances_3 = inverse_distances * inverse_distances * inverse_distances

    if isinstance(inverse_distances_3, numpy.ndarray):
        return directions * inverse_distances_3.T[:, None, :]
    else:
        return directions * inverse_distances_3.transpose(0, 1)[:, None, :]
