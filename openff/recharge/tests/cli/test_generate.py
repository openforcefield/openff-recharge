import json
import os
from multiprocessing.pool import Pool

import numpy

from openff.recharge.cli.generate import generate as generate_cli
from openff.recharge.conformers import ConformerSettings
from openff.recharge.esp import ESPSettings
from openff.recharge.esp.psi4 import Psi4ESPGenerator
from openff.recharge.esp.storage import MoleculeESPStore
from openff.recharge.grids import GridSettings


def test_generate(runner, monkeypatch):

    # Mock the Psi4 calls so the test can run even when not present.
    # This requires also mocking the multiprocessing to ensure the
    # monkeypatch on Psi4 holds.
    def mock_map(_, func, iterable):
        return [func(x) for x in iterable]

    def mock_psi4_generate(*_):
        return numpy.zeros((1, 3)), numpy.zeros((1, 1)), numpy.zeros((1, 3))

    monkeypatch.setattr(Psi4ESPGenerator, "generate", mock_psi4_generate)
    monkeypatch.setattr(Pool, "map", mock_map)

    # Create a mock set of inputs.
    with open("smiles.json", "w") as file:
        json.dump(["C"], file)

    with open("esp-settings.json", "w") as file:
        file.write(ESPSettings(grid_settings=GridSettings(spacing=1.0)).json())

    with open("conformer-settings.json", "w") as file:
        file.write(ConformerSettings(method="omega", sampling_mode="sparse").json())

    result = runner.invoke(generate_cli)

    if result.exit_code != 0:
        raise result.exception

    assert os.path.isfile("esp-store.sqlite")

    esp_store = MoleculeESPStore()
    assert len(esp_store.retrieve("C")) == 1
