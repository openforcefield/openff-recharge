"""Common exceptions raised by the framework."""
import abc


class RechargeException(BaseException):
    """The base exception from which most custom exceptions should inherit."""


class UnsupportedSMIRNOFFBCCError(RechargeException, abc.ABC):
    """The base error for when a SMIRNOFF charge increment parameter
    cannot be mapped onto a bond charge correction parameter.
    """

    def __init__(self, smirks, message):
        """

        Parameters
        ----------
        smirks
            The SMIRKS pattern off the SMIRNOFF parameter which
            cannot be converted.
        """

        super(UnsupportedSMIRNOFFBCCError, self).__init__(message)
        self.smirks = smirks


class UnsupportedBCCSmirksError(UnsupportedSMIRNOFFBCCError):
    """An error raised when a SMIRNOFF charge increment parameter
    cannot be mapped onto a bond charge correction parameter as it
    should be applied to more / less than two atoms (i.e. not a BCC).
    """

    def __init__(self, smirks, n_tagged: int):
        """

        Parameters
        ----------
        smirks
            The SMIRKS pattern off the SMIRNOFF parameter which
            cannot be converted.
        n_tagged
            The number of tagged atoms in the SMIRKS pattern.
        """

        super(UnsupportedBCCSmirksError, self).__init__(
            smirks,
            f"Only SMIRNOFF charge increments which apply to two atoms "
            f"are supported. The {smirks} applies to {n_tagged} atoms.",
        )


class UnsupportedBCCValueError(UnsupportedSMIRNOFFBCCError):
    """An error raised when a SMIRNOFF charge increment parameter
    cannot be mapped onto a bond charge correction parameter as it
    does not symetrically apply across a bond (i.e
    ``charge_charge_increment[0] != -charge_charge_increment[1].
    """

    def __init__(self, smirks, charge_increment_0, charge_increment_1):

        super(UnsupportedBCCValueError, self).__init__(
            smirks,
            f"Only SMIRNOFF charge increments which apply symmetrically to "
            f"a bond are supported "
            f"(smirks={smirks},"
            f"forward_value={charge_increment_0}, "
            f"reverse_value={charge_increment_1}).",
        )
