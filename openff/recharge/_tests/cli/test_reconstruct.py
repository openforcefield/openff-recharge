import json
import os
from multiprocessing.pool import Pool

import numpy
import pytest

from openff.recharge.cli.reconstruct import _retrieve_result_records
from openff.recharge.cli.reconstruct import reconstruct as reconstruct_cli
from openff.recharge.esp import ESPSettings
from openff.recharge.esp.storage import MoleculeESPRecord, MoleculeESPStore
from openff.recharge.grids import LatticeGridSettings
from openff.recharge.utilities.molecule import smiles_to_molecule
from openff.toolkit._tests.utils import requires_openeye


def test_retrieve_result_records():
    pytest.importorskip("qcportal")

    # noinspection PyTypeChecker
    qc_results, qc_keywords = _retrieve_result_records(["1"])

    assert len(qc_results) == 1
    assert len(qc_keywords) == 1
    assert "1" in qc_keywords


@requires_openeye
def test_reconstruct(runner, monkeypatch):
    pytest.importorskip("psi4")
    pytest.importorskip("qcportal")

    # Mock the multiprocessing call to return a dummy ESP record for a faster test.
    def mock_imap(_, func, iterable):
        return [
            MoleculeESPRecord.from_molecule(
                smiles_to_molecule("O"),
                conformer=numpy.zeros((3, 3)),
                grid_coordinates=numpy.zeros((1, 3)),
                esp=numpy.zeros((1, 1)),
                electric_field=numpy.zeros((1, 3)),
                esp_settings=ESPSettings(grid_settings=LatticeGridSettings()),
            )
        ]

    monkeypatch.setattr(Pool, "imap", mock_imap)

    # Create a mock set of inputs.
    with open("record-ids.json", "w") as file:
        json.dump(["1"], file)

    with open("grid-settings.json", "w") as file:
        file.write(LatticeGridSettings(spacing=1.0).json())

    result = runner.invoke(
        reconstruct_cli,
        "--record-ids record-ids.json --grid-settings grid-settings.json",
    )

    if result.exit_code != 0:
        raise result.exception

    assert os.path.isfile("esp-store.sqlite")

    esp_store = MoleculeESPStore()
    assert len(esp_store.retrieve("O")) == 1
