"""A module containing those exceptions which will be raised when generating
partial charges for a molecule."""
from openff.recharge.utilities.exceptions import RechargeException


class OEQuacpacError(RechargeException):
    """An exception raised when Quacpac fails to complete successfully."""


class MissingConformersError(RechargeException):
    """An exception raised when a molecule which does not contain any conformers
    was provided to a function which required them."""

    def __init__(self):
        super(MissingConformersError, self).__init__(
            "The provided molecule does not contain any conformers."
        )


class UnableToAssignChargeError(RechargeException):
    """An exception raised when atoms in a molecule could not correctly be
    assigned a partial charge."""
