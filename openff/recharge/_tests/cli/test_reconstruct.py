import json
import os
from importlib.resources import files

import numpy
import pytest

from openff.recharge.cli.reconstruct import reconstruct as reconstruct_cli
from openff.recharge.esp import ESPSettings
from openff.recharge.esp.storage import MoleculeESPRecord, MoleculeESPStore
from openff.recharge.grids import LatticeGridSettings, GridSettingsType
from openff.recharge.utilities.molecule import smiles_to_molecule
from openff.toolkit._tests.utils import requires_openeye


# mock a data retreival from qc_archive, keep as minimal as possible.
class MockMolecule:
    def __init__(self, molecule_data):
        self.symbols = molecule_data["symbols"]
        self.geometry = molecule_data["geometry"]


class MockWavefunction:
    def __init__(self, wavefunction_data):
        self.basis = wavefunction_data["basis"]
        self.scf_orbitals_a = numpy.array(wavefunction_data["scf_orbitals_a"])
        self.scf_orbitals_b = numpy.array(wavefunction_data["scf_orbitals_b"])


class MockBaseRecord:
    def __init__(self, record_data):
        self.molecule = MockMolecule(record_data["molecule"])
        self.wavefunction = MockWavefunction(record_data["wavefunction"])
        self.properties = record_data["properties"]
        self.specification = record_data["specification"]


def load_mock_qc_result():
    with open(
        files("openff.recharge")
        / os.path.join("_tests", "data", "qc_results", "mock_qc_result.json"),
    ) as file:
        mock_qc_result_data = json.load(file)
        return MockBaseRecord(mock_qc_result_data)


# top-level function to allow it to be pickled in spawn() multiprocesses
def mock_process_result(
    result_tuple,
    grid_settings: GridSettingsType,
):
    return MoleculeESPRecord.from_molecule(
        smiles_to_molecule("O"),
        conformer=numpy.zeros((3, 3)),
        grid_coordinates=numpy.zeros((1, 3)),
        esp=numpy.zeros((1, 1)),
        electric_field=numpy.zeros((1, 3)),
        esp_settings=ESPSettings(grid_settings=LatticeGridSettings()),
    )


MOCK_QC_RESULT = load_mock_qc_result()


@requires_openeye
def test_reconstruct(runner, monkeypatch):
    pytest.importorskip("psi4")

    # mock the process result function
    def mock_retrieve_result_records(record_ids: int = 1) -> tuple[tuple, list[dict]]:
        return [MOCK_QC_RESULT]

    monkeypatch.setattr(
        "openff.recharge.cli.reconstruct._retrieve_result_records",
        mock_retrieve_result_records,
    )
    monkeypatch.setattr(
        "openff.recharge.cli.reconstruct._process_result", mock_process_result
    )

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
