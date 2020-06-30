"""Contains the core data models used by this framework."""
from typing import Any, Dict

from pydantic import BaseModel, Field, constr


class BondChargeCorrection(BaseModel):
    """An object which encodes the value of a bond-charge correction, the chemical
    environment to which it should be applied, and provenance about its source.
    """

    smirks: constr(min_length=1) = Field(
        ...,
        description="A SMIRKS pattern which encodes the chemical environment that "
        "this correction should be applied to.",
    )
    value: float = Field(..., description="The value of this correction.")

    provenance: Dict[str, Any] = Field(
        ..., description="Provenance information about this bond charge correction."
    )
