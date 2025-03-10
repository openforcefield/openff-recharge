from typing import Annotated
from openff.toolkit import Quantity
import numpy

from pydantic import (
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    WrapValidator,
)


def conformer_validator(
    value: numpy.ndarray | Quantity,
    handler: ValidatorFunctionWrapHandler,
    info: ValidationInfo,
) -> numpy.ndarray:
    if info.mode == "json":
        raise NotImplementedError()

    assert info.mode == "python"

    if isinstance(value, numpy.ndarray):
        return value
    elif isinstance(value, Quantity):
        return value.m_as("angstrom")
    else:
        raise ValueError(f"Invalid type {type(value)}")


def esp_validator(
    value: numpy.ndarray | Quantity,
    handler: ValidatorFunctionWrapHandler,
    info: ValidationInfo,
) -> numpy.ndarray:
    if info.mode == "json":
        raise NotImplementedError()

    assert info.mode == "python"

    if isinstance(value, numpy.ndarray):
        return value
    elif isinstance(value, Quantity):
        return value.m_as("hartree / e")
    else:
        raise ValueError(f"Invalid type {type(value)}")


def electric_field_validator(
    value: numpy.ndarray | Quantity,
    handler: ValidatorFunctionWrapHandler,
    info: ValidationInfo,
) -> numpy.ndarray:
    if info.mode == "json":
        raise NotImplementedError()

    assert info.mode == "python"

    if isinstance(value, numpy.ndarray):
        return value
    elif isinstance(value, Quantity):
        return value.m_as("hartree / (e * a0)")
    else:
        raise ValueError(f"Invalid type {type(value)}")


Conformer = Annotated[
    numpy.ndarray[float],
    WrapValidator(conformer_validator),
]

ESP = Annotated[
    numpy.ndarray[float],
    WrapValidator(esp_validator),
]

ElectricField = Annotated[
    numpy.ndarray[float],
    WrapValidator(electric_field_validator),
]
