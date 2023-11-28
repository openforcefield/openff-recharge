import pytest
from click.testing import CliRunner


@pytest.fixture()
def runner() -> CliRunner:
    """Creates a new click CLI runner object and temporarily moves the working directory
    to a temporary directory"""
    click_runner = CliRunner()

    with click_runner.isolated_filesystem():
        yield click_runner
