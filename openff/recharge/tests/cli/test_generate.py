import json
import logging
import os
from multiprocessing.pool import Pool

import numpy
import pytest

from openff.recharge.cli.generate import _compute_esp
from openff.recharge.cli.generate import generate as generate_cli
from openff.recharge.conformers import ConformerGenerator, ConformerSettings
from openff.recharge.esp import ESPSettings
from openff.recharge.esp.exceptions import Psi4Error
from openff.recharge.esp.psi4 import Psi4ESPGenerator
from openff.recharge.esp.storage import MoleculeESPStore
from openff.recharge.grids import LatticeGridSettings
from openff.toolkit._tests.utils import requires_openeye


@requires_openeye
def test_generate(runner, monkeypatch):
    # Mock the Psi4 calls so the test can run even when not present.
    # This requires also mocking the multiprocessing to ensure the
    # monkeypatch on Psi4 holds.
    def mock_imap(_, func, iterable):
        return [func(x) for x in iterable]

    set_kwargs = {}

    def mock_psi4_generate(*_, **kwargs):
        set_kwargs.update(kwargs)
        return (
            numpy.zeros((5, 3)),
            numpy.zeros((1, 3)),
            numpy.zeros((1, 1)),
            numpy.zeros((1, 3)),
        )

    monkeypatch.setattr(Psi4ESPGenerator, "generate", mock_psi4_generate)
    monkeypatch.setattr(Pool, "imap", mock_imap)

    # Create a mock set of inputs.
    with open("smiles.json", "w") as file:
        json.dump(["C"], file)

    with open("esp-settings.json", "w") as file:
        file.write(ESPSettings(grid_settings=LatticeGridSettings(spacing=1.0)).json())

    with open("conformer-settings.json", "w") as file:
        file.write(ConformerSettings(method="omega", sampling_mode="sparse").json())

    result = runner.invoke(generate_cli)

    if result.exit_code != 0:
        raise result.exception

    assert os.path.isfile("esp-store.sqlite")

    esp_store = MoleculeESPStore()
    assert len(esp_store.retrieve("C")) == 1

    assert set_kwargs["minimize"] is True


@pytest.mark.parametrize("error_type", [RuntimeError])
def test_compute_esp_oe_error(error_type, caplog, monkeypatch):
    def mock_conformer_generate(*_):
        raise error_type()

    monkeypatch.setattr(ConformerGenerator, "generate", mock_conformer_generate)

    with caplog.at_level(logging.ERROR):
        _compute_esp(
            "C",
            ConformerSettings(),
            ESPSettings(grid_settings=LatticeGridSettings(spacing=1.0)),
            False,
        )

    assert "Coordinates could not be generated for" in caplog.text
    assert error_type.__name__ in caplog.text


def test_compute_esp_psi4_error(caplog, monkeypatch):
    def mock_psi4_generate(*_, **kwargs):
        raise Psi4Error("std_out", "std_err")

    monkeypatch.setattr(ConformerGenerator, "generate", lambda *args: [None])
    monkeypatch.setattr(Psi4ESPGenerator, "generate", mock_psi4_generate)

    with caplog.at_level(logging.ERROR):
        _compute_esp(
            "C",
            ConformerSettings(),
            ESPSettings(grid_settings=LatticeGridSettings(spacing=1.0)),
            False,
        )

    assert "Psi4 failed to run for conformer" in caplog.text
    assert "Psi4Error" in caplog.text
