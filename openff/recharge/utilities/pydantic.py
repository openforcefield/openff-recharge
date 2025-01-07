"""Common utilities and types for when building pydantic models.

Notes
-----
Most of the classes in the module are based off of the discussion here:
https://github.com/samuelcolvin/pydantic/issues/380
"""

from typing import Any, cast

import numpy
from openff.units import unit, Quantity
from openff.recharge._pydantic import validator


class ArrayMeta(type):
    def __getitem__(self, t):
        return type("Array", (Array,), {"__dtype__": t})


class Array(numpy.ndarray, metaclass=ArrayMeta):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_type

    @classmethod
    def validate_type(cls, val):
        dtype = getattr(cls, "__dtype__", Any)

        if dtype is Any:
            return numpy.array(val)
        else:
            return numpy.array(val, dtype=dtype)


def wrapped_float_validator(field_name: str, expected_units: unit.Unit) -> validator:
    def validate_unit(cls, value):
        if isinstance(value, str):
            return float(value)
        elif value is None or isinstance(value, float):
            return value

        assert isinstance(value, Quantity)
        return cast(Quantity, value).to(expected_units).m

    return validator(field_name, allow_reuse=True, pre=True)(validate_unit)


def wrapped_array_validator(field_name: str, expected_units: unit.Unit) -> validator:
    def validate_unit(cls, value):
        if isinstance(value, str):
            raise NotImplementedError()
        elif value is None or isinstance(value, numpy.ndarray):
            return value

        assert isinstance(value, Quantity)
        return cast(Quantity, value).to(expected_units).m

    return validator(field_name, allow_reuse=True, pre=True)(validate_unit)
