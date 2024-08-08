import json
import os
from multiprocessing.pool import Pool
from qcportal.record_models import BaseRecord
from importlib_resources import files

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
    # noinspection PyTypeChecker
    qc_results, qc_keywords = _retrieve_result_records(32651863)

    assert len([*qc_results]) == 1
    assert len([*qc_keywords]) == 1

class MockMolecule:
    def __init__(self, molecule_data):
        self.schema_name = molecule_data['schema_name']
        self.schema_version = molecule_data['schema_version']
        self.symbols = molecule_data['symbols']
        self.geometry = molecule_data['geometry']
        self.name = molecule_data['name']
        self.molecular_charge = molecule_data['molecular_charge']
        self.molecular_multiplicity = molecule_data['molecular_multiplicity']

class MockWavefunction:
    def __init__(self, wavefunction_data):
        self.basis = wavefunction_data['basis']
        self.scf_orbitals_a = wavefunction_data['scf_orbitals_a']
        self.scf_orbitals_b = wavefunction_data['scf_orbitals_b']

class MockBaseRecord:
    def __init__(self, record_data):
        self.molecule = MockMolecule(record_data['molecule'])
        self.wavefunction = MockWavefunction(record_data['wavefunction'])
        self.properties = record_data['properties']
        self.specification = record_data['specification']

def load_mock_qc_result():
    with open(files("openff.recharge") / os.path.join("_tests", "data", "qc_results", "mock_qc_result.json"), "r") as file:
        mock_qc_result_data = json.load(file)
        return MockBaseRecord(mock_qc_result_data)

# Load the mock QC result once
MOCK_QC_RESULT = load_mock_qc_result()


# @requires_openeye
def test_reconstruct(runner, monkeypatch):
    pytest.importorskip("psi4")

    # REMOVE THIS MOCK
    # Mock the multiprocessing call to return a dummy ESP record for a faster test.
    # def mock_imap(_, func, iterable):
    #     return [s
    #         MoleculeESPRecord.from_molecule(
    #             smiles_to_molecule("O"),
    #             conformer=numpy.zeros((3, 3)),
    #             grid_coordinates=numpy.zeros((1, 3)),
    #             esp=numpy.zeros((1, 1)),
    #             electric_field=numpy.zeros((1, 3)),
    #             esp_settings=ESPSettings(grid_settings=LatticeGridSettings()),
    #         )
    #     ]

    # monkeypatch.setattr(Pool, "imap", mock_imap)

    # mock_retrieve_result_records here
    # Return tuple["qcportal.record_models.RecordQueryIterator", list[dict]] 
    # But the qcportal.record_models.RecordQueryIterator could be replaced by a standard 
    # python list iterator

    
    def mock_retrieve_result_records(record_ids: int=1) -> tuple[MockBaseRecord,list[dict]]:
            return [MOCK_QC_RESULT], [{}]
        
    qc_results, qc_keyword_sets = mock_retrieve_result_records()
    out = [
        (qc_result, qc_molecule, qc_keyword_sets[index])
        for index, qc_result in enumerate(qc_results)
        for qc_molecule in [qc_result.molecule]
    ]
    print(out)

   # Mock the _retrieve_result_records function to return mock data.
    monkeypatch.setattr(
        "openff.recharge.cli.reconstruct._retrieve_result_records", 
        mock_retrieve_result_records
    )

    # Create a mock set of inputs.
    with open("record-ids.json", "w") as file:
        json.dump(["1"], file)

    with open("grid-settings.json", "w") as file:
        file.write(LatticeGridSettings(spacing=1.0).json())

    # qc_result_mock = BaseRecord(
    #     id=1,
    #     wavefunction=mock_wavefunction,
    #     molecule = mock_qc_molecule,
    #     properties={"calcinfo_nalpha": 2},
    #     specification={"basis": "STO-3G", "method": "HF"},
    #     keywords = 
    # )
    result = runner.invoke(
        reconstruct_cli,
        "--record-ids record-ids.json --grid-settings grid-settings.json",
    )

    if result.exit_code != 0:
        raise result.exception

    assert os.path.isfile("esp-store.sqlite")

    esp_store = MoleculeESPStore()
    assert len(esp_store.retrieve("O")) == 1

