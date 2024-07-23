import numpy
import scipy
import pytest

from openff.recharge.utilities.tensors import (
    append_zero,
    cdist,
    concatenate,
    inverse_cdist,
    pairwise_differences,
    to_numpy,
    to_torch,
    as_sparse,
    as_dense,
)

try:
    import torch
except ImportError:
    torch = None

tensor_types = [numpy.array] + ([] if torch is None else [torch.tensor])


@pytest.mark.parametrize("tensor_type", tensor_types)
def test_to_numpy(tensor_type):
    input_tensor = tensor_type([[0.5, 1.0]])
    output_tensor = to_numpy(input_tensor)

    expected_tensor = numpy.array([[0.5, 1.0]])

    assert output_tensor.shape == expected_tensor.shape
    assert numpy.allclose(output_tensor, expected_tensor)


def test_to_numpy_none():
    assert to_numpy(None) is None


@pytest.mark.parametrize("tensor_type", tensor_types)
def test_to_torch(tensor_type):
    pytest.importorskip("torch")

    input_tensor = tensor_type([[0.5, 1.0]])
    output_tensor = to_torch(input_tensor)

    expected_tensor = torch.tensor([[0.5, 1.0]])

    assert output_tensor.shape == expected_tensor.shape
    assert torch.allclose(output_tensor, expected_tensor)


def test_to_torch_none():
    pytest.importorskip("torch")
    assert to_torch(None) is None


@pytest.mark.parametrize("tensor_type", tensor_types)
def test_cdist(tensor_type):
    input_tensor_a = tensor_type([[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    input_tensor_b = tensor_type([[0.0, 2.0, 0.0], [0.0, 3.0, 0.0], [0.0, 4.0, 0.0]])

    output_tensor = cdist(input_tensor_a, input_tensor_b)
    assert type(output_tensor) is type(input_tensor_a)

    output_tensor = to_numpy(output_tensor)

    expected_tensor = numpy.array(
        [[numpy.sqrt(x * x + y * y) for y in [2.0, 3.0, 4.0]] for x in [3.0, 4.0]]
    )

    assert output_tensor.shape == expected_tensor.shape
    assert numpy.allclose(output_tensor, expected_tensor)


@pytest.mark.parametrize("tensor_type", tensor_types)
def test_inverse_cdist(tensor_type):
    input_tensor_a = tensor_type([[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    input_tensor_b = tensor_type([[0.0, 2.0, 0.0], [0.0, 3.0, 0.0], [0.0, 4.0, 0.0]])

    output_tensor = inverse_cdist(input_tensor_a, input_tensor_b)
    assert type(output_tensor) is type(input_tensor_a)

    output_tensor = to_numpy(output_tensor)

    expected_tensor = numpy.array(
        [[1.0 / numpy.sqrt(x * x + y * y) for y in [2.0, 3.0, 4.0]] for x in [3.0, 4.0]]
    )

    assert output_tensor.shape == expected_tensor.shape
    assert numpy.allclose(output_tensor, expected_tensor)


@pytest.mark.parametrize("tensor_type", tensor_types)
def test_pairwise_differences(tensor_type):
    points_a = tensor_type([[0.0, 3.0, 1.0], [0.0, 4.0, 2.0]])
    points_b = tensor_type([[2.0, 0.0, 2.0], [3.0, 0.0, 4.0], [4.0, 0.0, 0.0]])

    output_tensor = pairwise_differences(points_a, points_b)

    expected_tensor = numpy.array(
        [
            [[2.0, 2.0], [-3.0, -4.0], [1.0, 0.0]],
            [[3.0, 3.0], [-3.0, -4.0], [3.0, 2.0]],
            [[4.0, 4.0], [-3.0, -4.0], [-1.0, -2.0]],
        ]
    )

    assert output_tensor.shape == expected_tensor.shape
    assert numpy.allclose(output_tensor, expected_tensor)


@pytest.mark.parametrize("tensor_type", tensor_types)
def test_append_zero(tensor_type):
    input_tensor = tensor_type([1.0, 2.0])
    output_tensor = append_zero(input_tensor)

    expected_tensor = numpy.array([1.0, 2.0, 0.0])

    assert output_tensor.shape == expected_tensor.shape
    assert numpy.allclose(output_tensor, expected_tensor)


@pytest.mark.parametrize("tensor_type", tensor_types)
def test_concatenate(tensor_type):
    input_tensors = [
        tensor_type([[1.0, 2.0]]),
        tensor_type([[3.0, 4.0]]),
        tensor_type([[5.0, 6.0]]),
    ]
    output_tensor = concatenate(*input_tensors)

    expected_tensor = numpy.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    assert output_tensor.shape == expected_tensor.shape
    assert numpy.allclose(output_tensor, expected_tensor)


def test_concatenate_none():
    assert concatenate(None, None) is None


def test_as_sparse_numpy():
    input_tensor = numpy.array([[1.0, 2.0], [3.0, 4.0]])
    output_tensor = as_sparse(input_tensor)

    assert type(output_tensor) is scipy.sparse.coo_array

def test_as_sparse_scipy():
    input_tensor = scipy.sparse.coo_array([[1.0, 2.0], [3.0, 4.0]])
    output_tensor = as_sparse(input_tensor)

    assert input_tensor is output_tensor

def test_as_sparse_torch():
    input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    output_tensor = as_sparse(input_tensor)

    assert output_tensor.is_sparse

def test_as_dense_numpy():
    input_tensor = numpy.array([[1.0, 2.0], [3.0, 4.0]])
    output_tensor = as_dense(input_tensor)

    assert input_tensor is output_tensor

def test_as_dense_scipy():
    input_tensor = scipy.sparse.coo_array([[1.0, 2.0], [3.0, 4.0]])
    output_tensor = as_dense(input_tensor)

    assert type(output_tensor) is numpy.ndarray

def test_as_dense_torch():
    input_tensor = as_sparse(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert input_tensor.is_sparse

    output_tensor = as_dense(input_tensor)
    assert not output_tensor.is_sparse