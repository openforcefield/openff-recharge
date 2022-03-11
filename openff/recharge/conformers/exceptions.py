"""Exceptions raised when generating conformers"""
from openff.recharge.utilities.exceptions import RechargeException


class ConformerGenerationError(RechargeException):
    """An exception raised when atoms in a molecule could not correctly be
    assigned a partial charge."""
