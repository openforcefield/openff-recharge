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


class MissingOptionalDependency(RechargeException):
    """An exception raised when an optional dependency is required
    but cannot be found.

    Attributes
    ----------
    library_name
        The name of the missing library.
    license_issue
        Whether the library was importable but was unusable due
        to a missing license.
    """

    def __init__(self, library_name: str, license_issue: bool = False):
        """

        Parameters
        ----------
        library_name
            The name of the missing library.
        license_issue
            Whether the library was importable but was unusable due
            to a missing license.
        """

        message = f"The required {library_name} module could not be imported."

        if license_issue:
            message = f"{message} This is due to a missing license."

        super(MissingOptionalDependency, self).__init__(message)

        self.library_name = library_name
        self.license_issue = license_issue
