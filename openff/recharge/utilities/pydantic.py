"""A set of common utilities and types for when building pydantic
models.

Notes
-----
Most of the classes in the module are based off of the discussion here:
https://github.com/samuelcolvin/pydantic/issues/380
"""
from typing import Any

import numpy


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
