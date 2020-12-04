import json
import os
from multiprocessing.pool import Pool
from typing import TYPE_CHECKING

import numpy

from openff.recharge.cli.reconstruct import _retrieve_result_records
from openff.recharge.cli.reconstruct import reconstruct as reconstruct_cli
from openff.recharge.esp import ESPSettings
from openff.recharge.esp.storage import MoleculeESPRecord, MoleculeESPStore
from openff.recharge.grids import GridSettings
from openff.recharge.utilities.openeye import smiles_to_molecule

if TYPE_CHECKING:
    from qcfractal import FractalSnowflake


def test_retrieve_result_records(qc_server: "FractalSnowflake"):

    # noinspection PyTypeChecker
    qc_results, qc_keywords = _retrieve_result_records(["1"])

    assert len(qc_results) == 1
    assert len(qc_keywords) == 1
    assert "1" in qc_keywords


def test_reconstruct(runner, monkeypatch):

    # Mock the multiprocessing call to return a dummy ESP record for a faster test.
    def mock_imap(_, func, iterable):
        return [
            MoleculeESPRecord.from_oe_molecule(
                smiles_to_molecule("O"),
                conformer=numpy.zeros((3, 3)),
                grid_coordinates=numpy.zeros((1, 3)),
                esp=numpy.zeros((1, 1)),
                electric_field=numpy.zeros((1, 3)),
                esp_settings=ESPSettings(grid_settings=GridSettings()),
            )
        ]

    monkeypatch.setattr(Pool, "imap", mock_imap)

    # Create a mock set of inputs.
    with open("record-ids.json", "w") as file:
        json.dump(["1"], file)

    with open("grid-settings.json", "w") as file:
        file.write(GridSettings(spacing=1.0).json())

    result = runner.invoke(
        reconstruct_cli,
        "--record-ids record-ids.json --grid-settings grid-settings.json",
    )

    if result.exit_code != 0:
        raise result.exception

    assert os.path.isfile("esp-store.sqlite")

    esp_store = MoleculeESPStore()
    assert len(esp_store.retrieve("O")) == 1
