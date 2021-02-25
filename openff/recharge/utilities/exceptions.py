"""A module containing general exceptions raised by the framework."""
_CONDA_INSTALLATION_COMMANDS = {
    "openff.toolkit": "conda install -c conda-forge openff-toolkit",
    "openeye": "conda install -c openeye openeye-toolkits",
    "qcportal": "conda install -c conda-forge qcportal",
    "cmiles": "conda install -c conda-forge cmiles",
    "psi4": "conda install -c psi4/label/dev psi4",
}


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
        conda_command = _CONDA_INSTALLATION_COMMANDS.get(
            library_name.split(".")[0], None
        )

        if license_issue:
            message = f"{message} This is due to a missing license."
        elif conda_command is not None:
            message = (
                f"{message} Try installing the package by running `{conda_command}`."
            )

        super(MissingOptionalDependency, self).__init__(message)

        self.library_name = library_name
        self.license_issue = license_issue
