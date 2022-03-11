"""A module containing those exceptions which will be raised when generating
partial charges for a molecule."""
from openff.recharge.utilities.exceptions import RechargeException


class ChargeAssignmentError(RechargeException):
    """An exception raised when atoms in a molecule could not correctly be
    assigned a partial charge."""
