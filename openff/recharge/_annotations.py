from typing import Annotated
from collections.abc import Callable
from openff.toolkit import Quantity
import numpy
from functools import partial
from pydantic import BeforeValidator


def _array_validator(
    value: numpy.ndarray | Quantity,
    unit: str,
) -> numpy.ndarray:
    if isinstance(value, numpy.ndarray):
        return value
    elif isinstance(value, Quantity):
        return value.m_as(unit)
    else:
        raise ValueError(f"Invalid type {type(value)}")


def validator_factory(unit: str) -> Callable:
    """
    Return a function that converts the input array in given implicit units.

    This is meant to be used as the argument to pydantic.BeforeValidator in an Annotated type.

    """
    return partial(_array_validator, unit=unit)


Coordinates = Annotated[
    numpy.ndarray[float],
    BeforeValidator(validator_factory(unit="angstrom")),
]

ESP = Annotated[
    numpy.ndarray[float],
    BeforeValidator(validator_factory(unit="hartree / e")),
]

ElectricField = Annotated[
    numpy.ndarray[float],
    BeforeValidator(validator_factory(unit="hartree / (e * a0)")),
]
