import numpy

from openff.recharge.esp import ESPSettings
from openff.recharge.esp.storage import MoleculeESPRecord, MoleculeESPStore
from openff.recharge.grids import GridSettings
from openff.recharge.utilities.openeye import smiles_to_molecule


def test_record_from_oe_mol():
    """Tests that a ``MoleculeESPRecord`` can be correctly created from
    an ``OEMol``."""

    oe_molecule = smiles_to_molecule("C")
    conformer = numpy.array([[1.0, 0.0, 0.0]])

    grid_coordinates = numpy.array([[0.0, 1.0, 0.0]])
    esp = numpy.array([[2.0]])

    esp_settings = ESPSettings(grid_settings=GridSettings())

    record = MoleculeESPRecord.from_oe_molecule(
        oe_molecule=oe_molecule,
        conformer=conformer,
        grid_coordinates=grid_coordinates,
        esp=esp,
        esp_settings=esp_settings,
    )

    assert record.tagged_smiles == "[H:2][C:1]([H:3])([H:4])[H:5]"

    assert numpy.allclose(record.conformer, conformer)
    assert numpy.allclose(record.esp, esp)

    assert record.esp_settings == esp_settings


def test_tagged_to_canonical_smiles():

    assert (
        MoleculeESPStore._tagged_to_canonical_smiles("[H:2][C:1]([H:3])([H:4])[H:5]")
        == "C"
    )


def test_store(tmp_path):
    """Tests that records can be stored in a store."""

    esp_store = MoleculeESPStore(f"{tmp_path}.sqlite")

    esp_store.store(
        MoleculeESPRecord(
            tagged_smiles="[Ar:1]",
            conformer=numpy.array([[0.0, 0.0, 0.0]]),
            grid_coordinates=numpy.array([[0.0, 0.0, 0.0]]),
            esp=numpy.array([[0.0]]),
            esp_settings=ESPSettings(grid_settings=GridSettings()),
        )
    )

    assert esp_store.list() == ["[Ar]"]


def test_retrieve(tmp_path):
    """Tests that records can be retrieved from a store."""

    esp_store = MoleculeESPStore(f"{tmp_path}.sqlite")
    oe_molecule = smiles_to_molecule("C")

    esp_store.store(
        MoleculeESPRecord.from_oe_molecule(
            oe_molecule,
            conformer=numpy.array([[index, 0.0, 0.0] for index in range(5)]),
            grid_coordinates=numpy.array([[0.0, 0.0, 0.0]]),
            esp=numpy.array([[0.0]]),
            esp_settings=ESPSettings(
                basis="6-31g*", method="scf", grid_settings=GridSettings(),
            ),
        ),
        MoleculeESPRecord.from_oe_molecule(
            oe_molecule,
            conformer=numpy.array([[index, 0.0, 0.0] for index in range(5)]),
            grid_coordinates=numpy.array([[0.0, 0.0, 0.0]]),
            esp=numpy.array([[0.0]]),
            esp_settings=ESPSettings(
                basis="6-31g**", method="hf", grid_settings=GridSettings(),
            ),
        ),
        MoleculeESPRecord.from_oe_molecule(
            smiles_to_molecule("CO"),
            conformer=numpy.array([[index, 0.0, 0.0] for index in range(5)]),
            grid_coordinates=numpy.array([[0.0, 0.0, 0.0]]),
            esp=numpy.array([[0.0]]),
            esp_settings=ESPSettings(
                basis="6-31g*", method="hf", grid_settings=GridSettings(),
            ),
        ),
    )

    assert len(esp_store.retrieve()) == 3

    records = esp_store.retrieve(smiles="CO")
    assert len(records) == 1
    assert records[0].tagged_smiles == "[H:3][C:1]([H:4])([H:5])[O:2][H:6]"

    records = esp_store.retrieve(basis="6-31g*")
    assert len(records) == 2
    assert records[0].esp_settings.basis == "6-31g*"
    assert records[1].esp_settings.basis == "6-31g*"

    records = esp_store.retrieve(smiles="C", basis="6-31g*")
    assert len(records) == 1
    assert records[0].esp_settings.basis == "6-31g*"
    assert records[0].tagged_smiles == "[H:2][C:1]([H:3])([H:4])[H:5]"

    records = esp_store.retrieve(method="hf")
    assert len(records) == 2
    assert records[0].esp_settings.method == "hf"
    assert records[1].esp_settings.method == "hf"

    records = esp_store.retrieve(smiles="C", basis="6-31g*")
    assert len(records) == 1
    assert records[0].esp_settings.basis == "6-31g*"
    assert records[0].tagged_smiles == "[H:2][C:1]([H:3])([H:4])[H:5]"
