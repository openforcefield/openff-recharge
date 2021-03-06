import os

import pytest

from openff.recharge.utilities import get_data_file_path, requires_package, temporary_cd
from openff.recharge.utilities.exceptions import MissingOptionalDependency


def compare_paths(path_1: str, path_2: str) -> bool:
    """Checks whether two paths are the same.

    Parameters
    ----------
    path_1
        The first path.
    path_2
        The second path.

    Returns
    -------
    True if the paths are equivalent.
    """
    return os.path.normpath(path_1) == os.path.normpath(path_2)


def test_get_data_file_path():
    """Tests that the `get_data_file_path` can correctly find
    data files.
    """

    # Test a path which should exist.
    bcc_path = get_data_file_path(os.path.join("bcc", "original-am1-bcc.json"))
    assert os.path.isfile(bcc_path)

    # Test a path which should not exist.
    with pytest.raises(FileNotFoundError):
        get_data_file_path("missing")


def test_temporary_cd():
    """Tests that temporary cd works as expected"""

    original_directory = os.getcwd()

    # Move to the parent directory
    with temporary_cd(os.pardir):

        current_directory = os.getcwd()
        expected_directory = os.path.abspath(
            os.path.join(original_directory, os.pardir)
        )

        assert compare_paths(current_directory, expected_directory)

    assert compare_paths(os.getcwd(), original_directory)

    # Move to a temporary directory
    with temporary_cd():
        assert not compare_paths(os.getcwd(), original_directory)

    assert compare_paths(os.getcwd(), original_directory)

    # Move to the same directory
    with temporary_cd(""):
        assert compare_paths(os.getcwd(), original_directory)

    assert compare_paths(os.getcwd(), original_directory)

    with temporary_cd(os.curdir):
        assert compare_paths(os.getcwd(), original_directory)

    assert compare_paths(os.getcwd(), original_directory)


def test_requires_package():
    """Tests that the ``requires_package`` utility behaves as expected."""

    def dummy_function():
        pass

    # sys should always be found so this should not raise an exception.
    requires_package("sys")(dummy_function)()

    with pytest.raises(MissingOptionalDependency) as error_info:
        requires_package("fake-lib")(dummy_function)()

    assert error_info.value.library_name == "fake-lib"
