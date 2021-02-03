"""An optional module which integrates this library with the OpenFF
SMIRNOFF force field data models and specification."""
from typing import TYPE_CHECKING

import numpy

from openff.recharge.charges.bcc import AromaticityModels, BCCCollection, BCCParameter
from openff.recharge.smirnoff.exceptions import (
    UnsupportedBCCSmirksError,
    UnsupportedBCCValueError,
)
from openff.recharge.utilities import requires_package

if TYPE_CHECKING:
    from openff.toolkit.typing.engines.smirnoff.parameters import (
        ChargeIncrementModelHandler,
    )


@requires_package("openff.toolkit")
@requires_package("simtk")
def to_smirnoff(bcc_collection: BCCCollection) -> "ChargeIncrementModelHandler":
    """Converts a collection of bond charge correction parameters to
    a SMIRNOFF bond charge increment parameter handler.

    Notes
    -----
    * The AM1BCC charges applied by this handler will likely not match
      those computed using the built-in OpenEye implementation as that
      implementation uses a custom aromaticity model not supported by
      SMIRNOFF. This is in addition to potential conversion errors of the
      parameters into the SMIRKS language.

    * The aromaticity model defined by the collection will be ignored as
      this handler will parse the aromaticity model directly from the
      top level aromaticity node of the the SMIRNOFF specification.

    Parameters
    ----------
    bcc_collection
        The bond charge corrections to add to the SMIRNOFF handler.

    Returns
    -------
        The constructed parameter handler.
    """
    from openff.toolkit.typing.engines.smirnoff.parameters import (
        ChargeIncrementModelHandler,
    )
    from simtk import unit

    # noinspection PyTypeChecker
    bcc_parameter_handler = ChargeIncrementModelHandler(version="0.3")

    bcc_parameter_handler.number_of_conformers = 500
    bcc_parameter_handler.partial_charge_method = "am1elf10"

    for bcc_parameter in reversed(bcc_collection.parameters):
        bcc_parameter_handler.add_parameter(
            {
                "smirks": bcc_parameter.smirks,
                "charge_increment": [
                    bcc_parameter.value * unit.elementary_charge,
                    -bcc_parameter.value * unit.elementary_charge,
                ],
            }
        )

    return bcc_parameter_handler


@requires_package("simtk")
def from_smirnoff(
    parameter_handler: "ChargeIncrementModelHandler",
    aromaticity_model=AromaticityModels.MDL,
) -> BCCCollection:
    """Attempts to convert a SMIRNOFF bond charge increment parameter handler
    to a bond charge parameter collection.

    Notes
    -----
    * Only bond charge corrections (i.e. corrections whose SMIRKS only involve
      two tagged atoms) are supported currently.

    Parameters
    ----------
    parameter_handler
        The parameter handler to convert.
    aromaticity_model
        The model which describes how aromaticity should be assigned
        when applying the bond charge correction parameters.

    Returns
    -------
        The converted bond charge correction collection.
    """
    from simtk import unit

    bcc_parameters = []

    for off_parameter in reversed(parameter_handler.parameters):

        smirks = off_parameter.smirks

        if len(off_parameter.charge_increment) not in [1, 2]:
            raise UnsupportedBCCSmirksError(smirks, len(off_parameter.charge_increment))

        forward_value = off_parameter.charge_increment[0].value_in_unit(
            unit.elementary_charge
        )
        reverse_value = -forward_value

        if len(off_parameter.charge_increment) > 1:

            reverse_value = off_parameter.charge_increment[1].value_in_unit(
                unit.elementary_charge
            )

        if not numpy.isclose(forward_value, -reverse_value):

            raise UnsupportedBCCValueError(
                smirks,
                forward_value,
                reverse_value,
            )

        bcc_parameters.append(
            BCCParameter(smirks=smirks, value=forward_value, provenance={})
        )

    return BCCCollection(parameters=bcc_parameters, aromaticity_model=aromaticity_model)
