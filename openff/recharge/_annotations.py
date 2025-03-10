from typing import Annotated, Any
from openff.toolkit import Quantity
import numpy

from pydantic import (
    ConfigDict,
    AfterValidator,
    BeforeValidator,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    WrapSerializer,
    WrapValidator,
)
from openff.interchange._annotations import (
    quantity_validator,
    _dimensionality_validator_factory,
    quantity_json_serializer,
)


def _duck_to_angstrom(value: Any):
    """Cast list or ndarray without units to Quantity[ndarray] of nanometer."""
    if isinstance(value, (list, numpy.ndarray)):
        return Quantity(value, "angstrom")
    else:
        return value


def _duck_to_hartree_e(value: Any):
    """Cast list or ndarray without units to Quantity[ndarray] of nanometer."""
    if isinstance(value, (list, numpy.ndarray)):
        return Quantity(value, "hartree / e")
    else:
        return value


def _duck_to_hartree_e_bohr(value: Any):
    """Cast list or ndarray without units to Quantity[ndarray] of nanometer."""
    if isinstance(value, (list, numpy.ndarray)):
        return Quantity(value, "hartree / (e * bohr)")
    else:
        return value


_AngstromQuantity = Annotated[
    Quantity,
    WrapValidator(quantity_validator),
    BeforeValidator(_duck_to_angstrom),
    WrapSerializer(quantity_json_serializer),
]

_ESPQuantity = Annotated[
    Quantity,
    WrapValidator(quantity_validator),
    BeforeValidator(_duck_to_hartree_e),
    WrapSerializer(quantity_json_serializer),
]

_ElectricFieldQuantity = Annotated[
    Quantity,
    WrapValidator(quantity_validator),
    BeforeValidator(_duck_to_hartree_e_bohr),
    WrapSerializer(quantity_json_serializer),
]
