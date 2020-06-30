"""A module containing those exceptions which will be raised when generating
conformers fails."""
from openff.recharge.utilities.exceptions import RechargeException


class OEOmegaError(RechargeException):
    """An exception raised when OMEGA fails to complete successfully."""
