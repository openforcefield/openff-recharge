import numpy
import pytest
from openff.units import unit

from openff.recharge.esp import ESPSettings, PCMSettings
from openff.recharge.esp.storage import MoleculeESPRecord, MoleculeESPStore
from openff.recharge.esp.storage.db import (
    DB_VERSION,
    DBESPSettings,
    DBGridSettings,
    DBInformation,
    DBPCMSettings,
)
from openff.recharge.esp.storage.exceptions import IncompatibleDBVersion
from openff.recharge.grids import LatticeGridSettings, MSKGridSettings
from openff.recharge.utilities.molecule import smiles_to_molecule


class TestMoleculeESPRecord:
    @pytest.fixture()
    def mock_record(self):
        return MoleculeESPRecord(
            tagged_smiles="[Ar:1]",
            esp_settings=ESPSettings(grid_settings=LatticeGridSettings()),
            conformer=numpy.array([[0.0, 5.0, 0.0]]) * unit.nanometers,
            grid_coordinates=numpy.array([[0.0, 6.0, 0.0]]) * unit.nanometers,
            esp=numpy.array([[4.0]]) * unit.hartree / unit.e,
            electric_field=numpy.array([[1.0, 2.0, 3.0]])
            * unit.hartree
            / (unit.bohr * unit.e),
        )

    def test_validate_quantity(self, mock_record):
        assert numpy.allclose(mock_record.conformer, numpy.array([[0.0, 50.0, 0.0]]))
        assert numpy.allclose(
            mock_record.grid_coordinates, numpy.array([[0.0, 60.0, 0.0]])
        )
        assert numpy.allclose(mock_record.esp, numpy.array([[4.0]]))
        assert numpy.allclose(
            mock_record.electric_field, numpy.array([[1.0, 2.0, 3.0]])
        )

    def test_conformer_quantity(self, mock_record):
        assert numpy.allclose(
            mock_record.conformer_quantity,
            numpy.array([[0.0, 5.0, 0.0]]) * unit.nanometers,
        )

    def test_grid_coordinates_quantity(self, mock_record):
        assert numpy.allclose(
            mock_record.grid_coordinates_quantity,
            numpy.array([[0.0, 6.0, 0.0]]) * unit.nanometers,
        )

    def test_esp_quantity(self, mock_record):
        assert numpy.allclose(
            mock_record.esp_quantity, numpy.array([[4.0]]) * unit.hartree / unit.e
        )

    def test_electric_field_quantity(self, mock_record):
        assert numpy.allclose(
            mock_record.electric_field_quantity,
            numpy.array([[1.0, 2.0, 3.0]]) * unit.hartree / (unit.bohr * unit.e),
        )


def test_db_version(tmp_path):
    """Tests that a version is correctly added to a new store."""

    esp_store = MoleculeESPStore(f"{tmp_path}.sqlite")

    with esp_store._get_session() as db:
        db_info = db.query(DBInformation).first()

        assert db_info is not None
        assert db_info.version == DB_VERSION

    assert esp_store.db_version == DB_VERSION


def test_provenance(tmp_path):
    """Tests that a stores provenance can be set / retrieved."""

    esp_store = MoleculeESPStore(f"{tmp_path}.sqlite")

    assert esp_store.general_provenance == {}
    assert esp_store.software_provenance == {}

    general_provenance = {"author": "Author 1"}
    software_provenance = {"psi4": "0.1.0"}

    esp_store.set_provenance(general_provenance, software_provenance)

    assert esp_store.general_provenance == general_provenance
    assert esp_store.software_provenance == software_provenance


def test_db_invalid_version(tmp_path):
    """Tests that the correct exception is raised when loading a store
    with an unsupported version."""

    esp_store = MoleculeESPStore(f"{tmp_path}.sqlite")

    with esp_store._get_session() as db:
        db_info = db.query(DBInformation).first()
        db_info.version = DB_VERSION - 1

    with pytest.raises(IncompatibleDBVersion) as error_info:
        MoleculeESPStore(f"{tmp_path}.sqlite")

    assert error_info.value.found_version == DB_VERSION - 1
    assert error_info.value.expected_version == DB_VERSION


def test_record_from_molecule():
    """Tests that a ``MoleculeESPRecord`` can be correctly created from
    a ``Molecule``."""

    molecule = smiles_to_molecule("C")
    conformer = numpy.array([[1.0, 0.0, 0.0]])

    grid_coordinates = numpy.array([[0.0, 1.0, 0.0]])
    esp = numpy.array([[2.0]])
    electric_field = numpy.array([[2.0, 3.0, 4.0]])

    esp_settings = ESPSettings(
        pcm_settings=PCMSettings(), grid_settings=LatticeGridSettings()
    )

    record = MoleculeESPRecord.from_molecule(
        molecule=molecule,
        conformer=conformer,
        grid_coordinates=grid_coordinates,
        esp=esp,
        electric_field=electric_field,
        esp_settings=esp_settings,
    )

    # smiles order might change depending on toolkit
    expected_smiles = (
        "[H:2][C:1]([H:3])([H:4])[H:5]",
        "[C:1]([H:2])([H:3])([H:4])[H:5]",
    )
    assert record.tagged_smiles in expected_smiles

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
            electric_field=numpy.array([[0.0, 0.0, 0.0]]),
            esp_settings=ESPSettings(grid_settings=LatticeGridSettings()),
        ),
        MoleculeESPRecord(
            tagged_smiles="[Ar:1]",
            conformer=numpy.array([[0.0, 0.0, 0.0]]),
            grid_coordinates=numpy.array([[0.0, 0.0, 0.0]]),
            esp=numpy.array([[0.0]]),
            electric_field=numpy.array([[0.0, 0.0, 0.0]]),
            esp_settings=ESPSettings(
                pcm_settings=PCMSettings(), grid_settings=LatticeGridSettings()
            ),
        ),
    )

    assert esp_store.list() == ["[Ar]"]


def test_unique_esp_settings(tmp_path):
    """Tests that ESP settings are stored uniquely in the DB."""

    esp_store = MoleculeESPStore(f"{tmp_path}.sqlite")
    esp_settings = ESPSettings(grid_settings=LatticeGridSettings())

    # Store duplicate settings in the same session.
    with esp_store._get_session() as db:
        db.add(DBESPSettings.unique(db, esp_settings))
        db.add(DBESPSettings.unique(db, esp_settings))

    with esp_store._get_session() as db:
        assert db.query(DBESPSettings.id).count() == 1

    # Store a duplicate setting in a new session.
    with esp_store._get_session() as db:
        db.add(DBESPSettings.unique(db, esp_settings))

    with esp_store._get_session() as db:
        assert db.query(DBESPSettings.id).count() == 1

    # Store a non-duplicate set of settings
    esp_settings.method = "dft"

    with esp_store._get_session() as db:
        db.add(DBESPSettings.unique(db, esp_settings))

    with esp_store._get_session() as db:
        assert db.query(DBESPSettings.id).count() == 2


@pytest.mark.parametrize(
    "original_grid_settings, modified_grid_settings",
    [
        (LatticeGridSettings(), LatticeGridSettings(spacing=0.7)),
        (MSKGridSettings(), MSKGridSettings(density=0.123)),
    ],
)
def test_unique_grid_settings(tmp_path, original_grid_settings, modified_grid_settings):
    """Tests that ESP settings are stored uniquely in the DB."""

    esp_store = MoleculeESPStore(f"{tmp_path}.sqlite")

    # Store duplicate settings in the same session.
    with esp_store._get_session() as db:
        db.add(DBGridSettings.unique(db, original_grid_settings))
        db.add(DBGridSettings.unique(db, original_grid_settings))

    with esp_store._get_session() as db:
        assert db.query(DBGridSettings.id).count() == 1

    # Store a duplicate setting in a new session.
    with esp_store._get_session() as db:
        db.add(DBGridSettings.unique(db, original_grid_settings))

    with esp_store._get_session() as db:
        assert db.query(DBGridSettings.id).count() == 1

    # Store a non-duplicate set of settings
    with esp_store._get_session() as db:
        db.add(DBGridSettings.unique(db, modified_grid_settings))

    with esp_store._get_session() as db:
        assert db.query(DBGridSettings.id).count() == 2


def test_unique_pcm_settings(tmp_path):
    """Tests that ESP settings are stored uniquely in the DB."""

    esp_store = MoleculeESPStore(f"{tmp_path}.sqlite")
    pcm_settings = PCMSettings()

    # Store duplicate settings in the same session.
    with esp_store._get_session() as db:
        db.add(DBPCMSettings.unique(db, pcm_settings))
        db.add(DBPCMSettings.unique(db, pcm_settings))

    with esp_store._get_session() as db:
        assert db.query(DBPCMSettings.id).count() == 1

    # Store a duplicate setting in a new session.
    with esp_store._get_session() as db:
        db.add(DBPCMSettings.unique(db, pcm_settings))

    with esp_store._get_session() as db:
        assert db.query(DBPCMSettings.id).count() == 1

    # Store a non-duplicate set of settings
    pcm_settings.cavity_area *= 2.0

    with esp_store._get_session() as db:
        db.add(DBPCMSettings.unique(db, pcm_settings))

    with esp_store._get_session() as db:
        assert db.query(DBPCMSettings.id).count() == 2


def test_lattice_grid_settings_round_trip():
    """Test the round trip to / from the DB representation
    of a ``LatticeGridSettings`` object."""

    original_grid_settings = LatticeGridSettings()

    db_grid_settings = DBGridSettings._instance_to_db(original_grid_settings)
    recreated_grid_settings = DBGridSettings.db_to_instance(db_grid_settings)

    assert original_grid_settings.type == recreated_grid_settings.type

    assert numpy.isclose(
        original_grid_settings.spacing, recreated_grid_settings.spacing
    )
    assert numpy.isclose(
        original_grid_settings.inner_vdw_scale, recreated_grid_settings.inner_vdw_scale
    )
    assert numpy.isclose(
        original_grid_settings.outer_vdw_scale, recreated_grid_settings.outer_vdw_scale
    )


def test_msk_grid_settings_round_trip():
    """Test the round trip to / from the DB representation
    of a ``LatticeGridSettings`` object."""

    original_grid_settings = MSKGridSettings()

    db_grid_settings = DBGridSettings._instance_to_db(original_grid_settings)
    recreated_grid_settings = DBGridSettings.db_to_instance(db_grid_settings)

    assert original_grid_settings.type == recreated_grid_settings.type

    assert numpy.isclose(
        original_grid_settings.density, recreated_grid_settings.density
    )


def test_pcm_settings_round_trip():
    """Test the round trip to / from the DB representation
    of a ``PCMSettings`` object."""

    original_pcm_settings = PCMSettings()

    db_pcm_settings = DBPCMSettings._instance_to_db(original_pcm_settings)
    recreated_pcm_settings = DBPCMSettings.db_to_instance(db_pcm_settings)

    assert original_pcm_settings.solver == recreated_pcm_settings.solver
    assert original_pcm_settings.solvent == recreated_pcm_settings.solvent

    assert original_pcm_settings.radii_model == recreated_pcm_settings.radii_model
    assert original_pcm_settings.radii_scaling == recreated_pcm_settings.radii_scaling

    assert numpy.isclose(
        original_pcm_settings.cavity_area, recreated_pcm_settings.cavity_area
    )


def test_retrieve(tmp_path):
    """Tests that records can be retrieved from a store."""

    esp_store = MoleculeESPStore(f"{tmp_path}.sqlite")
    molecule = smiles_to_molecule("C")

    esp_store.store(
        MoleculeESPRecord.from_molecule(
            molecule,
            conformer=numpy.array([[index, 0.0, 0.0] for index in range(5)]),
            grid_coordinates=numpy.array([[0.0, 0.0, 0.0]]),
            esp=numpy.array([[0.0]]),
            electric_field=numpy.array([[0.0, 0.0, 0.0]]),
            esp_settings=ESPSettings(
                basis="6-31g*",
                method="scf",
                grid_settings=LatticeGridSettings(),
            ),
        ),
        MoleculeESPRecord.from_molecule(
            molecule,
            conformer=numpy.array([[index, 0.0, 0.0] for index in range(5)]),
            grid_coordinates=numpy.array([[0.0, 0.0, 0.0]]),
            esp=numpy.array([[0.0]]),
            electric_field=numpy.array([[0.0, 0.0, 0.0]]),
            esp_settings=ESPSettings(
                basis="6-31g**",
                method="hf",
                grid_settings=LatticeGridSettings(),
            ),
        ),
        MoleculeESPRecord.from_molecule(
            smiles_to_molecule("CO"),
            conformer=numpy.array([[index, 0.0, 0.0] for index in range(5)]),
            grid_coordinates=numpy.array([[0.0, 0.0, 0.0]]),
            esp=numpy.array([[0.0]]),
            electric_field=numpy.array([[0.0, 0.0, 0.0]]),
            esp_settings=ESPSettings(
                basis="6-31g*",
                method="hf",
                grid_settings=LatticeGridSettings(),
                pcm_settings=PCMSettings(),
            ),
        ),
    )

    assert len(esp_store.retrieve()) == 3

    records = esp_store.retrieve(smiles="CO")
    assert len(records) == 1
    expected_smiles_co = (
        "[H:3][C:1]([H:4])([H:5])[O:2][H:6]",
        "[C:1]([O:2][H:6])([H:3])([H:4])[H:5]",
    )
    assert records[0].tagged_smiles in expected_smiles_co

    records = esp_store.retrieve(basis="6-31g*")
    assert len(records) == 2
    assert records[0].esp_settings.basis == "6-31g*"
    assert records[1].esp_settings.basis == "6-31g*"

    records = esp_store.retrieve(smiles="C", basis="6-31g*")
    assert len(records) == 1
    assert records[0].esp_settings.basis == "6-31g*"
    expected_smiles_c = (
        "[H:2][C:1]([H:3])([H:4])[H:5]",
        "[C:1]([H:2])([H:3])([H:4])[H:5]",
    )
    assert records[0].tagged_smiles in expected_smiles_c

    records = esp_store.retrieve(method="hf")
    assert len(records) == 2
    assert records[0].esp_settings.method == "hf"
    assert records[1].esp_settings.method == "hf"

    records = esp_store.retrieve(smiles="C", basis="6-31g*")
    assert len(records) == 1
    assert records[0].esp_settings.basis == "6-31g*"
    assert records[0].tagged_smiles in expected_smiles_c

    records = esp_store.retrieve(basis="6-31g*", implicit_solvent=True)
    assert len(records) == 1
    assert records[0].esp_settings.basis == "6-31g*"
    assert records[0].tagged_smiles in expected_smiles_co

    records = esp_store.retrieve(basis="6-31g*", implicit_solvent=False)
    assert len(records) == 1
    assert records[0].esp_settings.basis == "6-31g*"
    assert records[0].tagged_smiles in expected_smiles_c
