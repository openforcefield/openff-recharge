import os

import pytest

from openff.recharge.utilities.utilities import get_data_file_path


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
