"""Utilities for manipulating numpy and pytorch tensors using a consistent API."""

from typing import TYPE_CHECKING, Union, overload

import numpy
from openff.utilities import requires_package

if TYPE_CHECKING:
    import torch

TensorType = Union[numpy.ndarray, "torch.Tensor"]

_ZERO = None


@overload
def to_numpy(tensor: None) -> None:
    ...


@overload
def to_numpy(tensor: TensorType) -> numpy.ndarray:
    ...


def to_numpy(tensor):
    """Converts an array-like object (either numpy or pytorch) into a numpy array."""
    if tensor is None:
        return None

    if isinstance(tensor, numpy.ndarray):
        return tensor

    return tensor.detach().numpy()


@overload
def to_torch(tensor: None) -> None:
    ...


@overload
def to_torch(tensor: TensorType) -> "torch.Tensor":
    ...


@requires_package("torch")
def to_torch(tensor):
    """Converts an array-like object (either numpy or pytorch) into a pytorch tensor."""

    import torch

    if tensor is None:
        return None

    if isinstance(tensor, torch.Tensor):
        return tensor

    return torch.from_numpy(tensor)


@overload
def cdist(a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
    ...


@overload
def cdist(a: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor":
    ...


def cdist(a, b):

    assert type(a) == type(b)

    if isinstance(a, numpy.ndarray):
        return numpy.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)

    elif a.__module__.startswith("torch"):

        import torch

        return torch.cdist(a, b)

    raise NotImplementedError()


@overload
def inverse_cdist(a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
    ...


@overload
def inverse_cdist(a: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor":
    ...


def inverse_cdist(a, b):

    assert type(a) == type(b)

    if isinstance(a, numpy.ndarray):
        return 1.0 / cdist(a, b)

    elif a.__module__.startswith("torch"):
        return cdist(a, b).reciprocal()

    raise NotImplementedError()


@overload
def pairwise_differences(a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
    ...


@overload
def pairwise_differences(a: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor":
    ...


def pairwise_differences(a, b):
    """Returns a tensor containing the vectors which point from all of the points (with
    dimension of ``n_dim``) in tensor ``a`` to all of the points in tensor ``b``.

    Parameters
    ----------
    a
        The first tensor of points with shape=(n_a, n_dim).
    b
        The second tensor of points with shape=(n_b, n_dim).

    Returns
    -------
        The vector field tensor with shape=(n_points_b, n_dim, n_points_a) and where
        ``tensor[i, :, j] = (b_i - a_j)``
    """

    assert type(a) == type(b)

    if isinstance(a, numpy.ndarray):
        return numpy.einsum("ijk->jki", b[None, :, :] - a[:, None, :])

    elif a.__module__.startswith("torch"):

        import torch

        return torch.einsum("ijk->jki", b[None, :, :] - a[:, None, :])

    raise NotImplementedError()


@overload
def append_zero(a: numpy.ndarray) -> numpy.ndarray:
    ...


@overload
def append_zero(a: "torch.Tensor") -> "torch.Tensor":
    ...


def append_zero(a):

    if isinstance(a, numpy.ndarray):
        return numpy.hstack([a, 0.0])

    elif a.__module__.startswith("torch"):

        import torch

        return torch.cat([a, torch.zeros(1, dtype=a.dtype)])

    raise NotImplementedError()
