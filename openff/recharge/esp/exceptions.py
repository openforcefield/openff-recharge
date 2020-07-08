from openff.recharge.utilities.exceptions import RechargeException


class Psi4Error(RechargeException):
    """An exception raised when Psi4 fails to execute."""

    def __init__(self, std_output: str, std_error: str):
        """

        Parameters
        ----------
        std_error
            The stderr from Psi4.
        std_output
            The stdout from Psi4.
        """
        super(Psi4Error, self).__init__(
            f"Psi4 failed to execute.\n\nStdErr:\n{std_error}\n\nStdOut:\n{std_output}"
        )
