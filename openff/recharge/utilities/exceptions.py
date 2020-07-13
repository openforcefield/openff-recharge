"""A module containing general exceptions raised by the framework."""


class RechargeException(BaseException):
    """The base exception from which most custom exceptions should inherit."""


class MoleculeFromSmilesError(RechargeException):
    """An exception raised when attempting to create a molecule from a
    SMILES pattern."""

    def __init__(self, *args, smiles: str, **kwargs):
        """

        Parameters
        ----------
        smiles
            The SMILES pattern which could not be parsed.
        """

        super(MoleculeFromSmilesError, self).__init__(*args, **kwargs)
        self.smiles = smiles


class InvalidSmirksError(RechargeException):
    """An exception raised when an invalid smirks pattern is provided."""

    def __init__(self, *args, smirks: str, **kwargs):
        """

        Parameters
        ----------
        smirks
            The SMIRKS pattern which could not be parsed.
        """

        super(InvalidSmirksError, self).__init__(*args, **kwargs)
        self.smirks = smirks
