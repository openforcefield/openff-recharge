"""This module contains classes which are able to store and retrieve
calculated electrostatic potentials in unified data collections.
"""
import functools
from collections import defaultdict
from contextlib import contextmanager
from typing import TYPE_CHECKING, ContextManager, Dict, List, Optional

import numpy
from pydantic import BaseModel, Field
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from openff.recharge.esp import ESPSettings
from openff.recharge.esp.storage.db import (
    DB_VERSION,
    DBBase,
    DBConformerRecord,
    DBCoordinate,
    DBESPSettings,
    DBGeneralProvenance,
    DBGridESP,
    DBGridSettings,
    DBInformation,
    DBMoleculeRecord,
    DBPCMSettings,
    DBSoftwareProvenance,
)
from openff.recharge.esp.storage.exceptions import IncompatibleDBVersion
from openff.recharge.utilities.openeye import import_oechem, smiles_to_molecule

if TYPE_CHECKING:
    from openeye import oechem

    Array = numpy.ndarray
else:
    from openff.recharge.utilities.pydantic import Array


class MoleculeESPRecord(BaseModel):
    """A record which contains information about the molecule that the ESP
    was measured for (including the exact conformer coordinates), provenance
    about how the ESP was calculated, and the values of the ESP at each of
    the grid points."""

    tagged_smiles: str = Field(
        ...,
        description="The tagged SMILES patterns (SMARTS) which encodes both the "
        "molecule stored in this record, a map between the atoms and the molecule and "
        "their coordinates.",
    )

    conformer: Array[float] = Field(
        ...,
        description="The coordinates [Angstrom] of this conformer with "
        "shape=(n_atoms, 3).",
    )

    grid_coordinates: Array[float] = Field(
        ...,
        description="The grid coordinates [Angstrom] which the ESP was calculated on "
        "with shape=(n_grid_points, 3).",
    )
    esp: Array[float] = Field(
        ...,
        description="The value of the ESP [Hartree / e] at each of the grid "
        "coordinates with shape=(n_grid_points, 1).",
    )
    electric_field: Array[float] = Field(
        ...,
        description="The value of the electric field [Hartree / (e . a0)] at each of "
        "the grid coordinates with shape=(n_grid_points, 3).",
    )

    esp_settings: ESPSettings = Field(
        ..., description="The settings used to generate the ESP stored in this record."
    )

    @classmethod
    def from_oe_molecule(
        cls,
        oe_molecule: "oechem.OEMol",
        conformer: numpy.ndarray,
        grid_coordinates: numpy.ndarray,
        esp: numpy.ndarray,
        electric_field: numpy.ndarray,
        esp_settings: ESPSettings,
    ) -> "MoleculeESPRecord":
        """Creates a new ``MoleculeESPRecord`` from an existing molecule
        object, taking care of creating the InChI and SMARTS representations.

        Parameters
        ----------
        oe_molecule
            The molecule to store in the record.
        conformer
            The coordinates [Angstrom] of this conformer with shape=(n_atoms, 3).
        grid_coordinates
            The grid coordinates [Angstrom] which the ESP was calculated on
            with shape=(n_grid_points, 3).
        esp
            The value of the ESP [Hartree / e] at each of the grid coordinates
            with shape=(n_grid_points, 1).
        electric_field
            The value of the electric field [Hartree / (e . a0)] at each of
            the grid coordinates with shape=(n_grid_points, 3).
        esp_settings
            The settings used to generate the ESP stored in this record.

        Returns
        -------
            The created record.
        """

        oechem = import_oechem()

        # Work on a copy of the molecule
        oe_molecule = oechem.OEMol(oe_molecule)

        # Build the tagged SMILES representation of the molecule.
        for atom in oe_molecule.GetAtoms():
            atom.SetMapIdx(atom.GetIdx() + 1)

        tagged_smiles = oechem.OECreateIsoSmiString(oe_molecule)

        return MoleculeESPRecord(
            tagged_smiles=tagged_smiles,
            conformer=conformer,
            grid_coordinates=grid_coordinates,
            esp=esp,
            electric_field=electric_field,
            esp_settings=esp_settings,
        )

    class Config:
        orm_mode = True


class MoleculeESPStore:
    """A class used to store the electrostatic potentials and the grid
    on which they were evaluated for multiple molecules in multiple conformers,
    as well as to retrieve and query this stored data.

    This class currently can only store the data in a SQLite data base.
    """

    @property
    def db_version(self) -> int:

        with self._get_session() as db:
            db_info = db.query(DBInformation).first()

            return db_info.version

    @property
    def general_provenance(self) -> Dict[str, str]:
        with self._get_session() as db:

            db_info = db.query(DBInformation).first()

            return {
                provenance.key: provenance.value
                for provenance in db_info.general_provenance
            }

    @property
    def software_provenance(self) -> Dict[str, str]:
        with self._get_session() as db:

            db_info = db.query(DBInformation).first()

            return {
                provenance.key: provenance.value
                for provenance in db_info.software_provenance
            }

    def __init__(self, database_path: str = "esp-store.sqlite"):
        """

        Parameters
        ----------
        database_path
            The path to the SQLite database to store to and retrieve data from.
        """
        self._database_url = f"sqlite:///{database_path}"

        self._engine = create_engine(self._database_url, echo=False)
        DBBase.metadata.create_all(self._engine)

        self._session_maker = sessionmaker(
            autocommit=False, autoflush=False, bind=self._engine
        )

        # Validate the DB version if present, or add one if not.
        with self._get_session() as db:
            db_info = db.query(DBInformation).first()

            if not db_info:
                db_info = DBInformation(version=DB_VERSION)
                db.add(db_info)

            if db_info.version != DB_VERSION:
                raise IncompatibleDBVersion(db_info.version, DB_VERSION)

    def set_provenance(
        self,
        general_provenance: Dict[str, str],
        software_provenance: Dict[str, str],
    ):
        """Set the stores provenance information.

        Parameters
        ----------
        general_provenance
            A dictionary storing provenance about the store such as the author,
            which QCArchive data set it was generated from, when it was generated
            etc.
        software_provenance
            A dictionary storing the provenance of the software and packages used
            to generate the data in the store.
        """

        with self._get_session() as db:

            db_info: DBInformation = db.query(DBInformation).first()
            db_info.general_provenance = [
                DBGeneralProvenance(key=key, value=value)
                for key, value in general_provenance.items()
            ]
            db_info.software_provenance = [
                DBSoftwareProvenance(key=key, value=value)
                for key, value in software_provenance.items()
            ]

    @contextmanager
    def _get_session(self) -> ContextManager[Session]:

        session = self._session_maker()

        try:
            yield session
            session.commit()
        except BaseException as e:
            session.rollback()
            raise e
        finally:
            session.close()

    @classmethod
    def _db_records_to_model(
        cls, db_records: List[DBMoleculeRecord]
    ) -> List[MoleculeESPRecord]:
        """Maps a set of database records into their corresponding
        data models.

        Parameters
        ----------
        db_records
            The records to map.

        Returns
        -------
            The mapped data models.
        """
        # noinspection PyTypeChecker
        return [
            MoleculeESPRecord(
                tagged_smiles=db_conformer.tagged_smiles,
                conformer=numpy.array(
                    [
                        [db_coordinate.x, db_coordinate.y, db_coordinate.z]
                        for db_coordinate in db_conformer.coordinates
                    ]
                ),
                grid_coordinates=numpy.array(
                    [
                        [grid_esp_value.x, grid_esp_value.y, grid_esp_value.z]
                        for grid_esp_value in db_conformer.grid_esp_values
                    ]
                ),
                esp=numpy.array(
                    [
                        [grid_esp_value.value]
                        for grid_esp_value in db_conformer.grid_esp_values
                    ]
                ),
                electric_field=numpy.array(
                    [
                        [
                            grid_esp_value.field_x,
                            grid_esp_value.field_y,
                            grid_esp_value.field_z,
                        ]
                        for grid_esp_value in db_conformer.grid_esp_values
                    ]
                ),
                esp_settings=ESPSettings(
                    basis=db_conformer.esp_settings.basis,
                    method=db_conformer.esp_settings.method,
                    grid_settings=DBGridSettings.db_to_instance(
                        db_conformer.grid_settings
                    ),
                    pcm_settings=None
                    if not db_conformer.pcm_settings
                    else DBPCMSettings.db_to_instance(db_conformer.pcm_settings),
                ),
            )
            for db_record in db_records
            for db_conformer in db_record.conformers
        ]

    @classmethod
    def _store_smiles_records(
        cls, db: Session, smiles: str, records: List[MoleculeESPRecord]
    ) -> DBMoleculeRecord:
        """Stores a set of records which all store information for the same
        molecule.

        Parameters
        ----------
        db
            The current database session.
        smiles
            The smiles representation of the molecule.
        records
            The records to store.
        """

        existing_db_molecule = (
            db.query(DBMoleculeRecord).filter(DBMoleculeRecord.smiles == smiles).first()
        )

        if existing_db_molecule is not None:
            db_record = existing_db_molecule
        else:
            db_record = DBMoleculeRecord(smiles=smiles)

        # noinspection PyTypeChecker
        # noinspection PyUnresolvedReferences
        db_record.conformers.extend(
            DBConformerRecord(
                tagged_smiles=record.tagged_smiles,
                coordinates=[
                    DBCoordinate(x=coordinate[0], y=coordinate[1], z=coordinate[2])
                    for coordinate in record.conformer
                ],
                grid_esp_values=[
                    DBGridESP(
                        x=coordinate[0],
                        y=coordinate[1],
                        z=coordinate[2],
                        value=esp[0],
                        field_x=electric_field[0],
                        field_y=electric_field[1],
                        field_z=electric_field[2],
                    )
                    for coordinate, esp, electric_field in zip(
                        record.grid_coordinates, record.esp, record.electric_field
                    )
                ],
                grid_settings=DBGridSettings.unique(
                    db, record.esp_settings.grid_settings
                ),
                pcm_settings=None
                if not record.esp_settings.pcm_settings
                else DBPCMSettings.unique(db, record.esp_settings.pcm_settings),
                esp_settings=DBESPSettings.unique(db, record.esp_settings),
            )
            for record in records
        )

        if existing_db_molecule is None:
            db.add(db_record)

        return db_record

    @classmethod
    @functools.lru_cache(128)
    def _tagged_to_canonical_smiles(cls, tagged_smiles: str) -> str:
        """Converts a smiles pattern which contains atom indices into
        a canonical smiles pattern without indices.

        Parameters
        ----------
        tagged_smiles
            The tagged smiles pattern to convert.

        Returns
        -------
            The canonical smiles pattern.
        """

        oechem = import_oechem()

        oe_molecule = smiles_to_molecule(tagged_smiles)

        for atom in oe_molecule.GetAtoms():
            atom.SetMapIdx(0)

        return oechem.OECreateCanSmiString(oe_molecule)

    def store(self, *records: MoleculeESPRecord):
        """Store the electrostatic potentials calculated for
        a given molecule in the data store.

        Parameters
        ----------
        records
            The records to store.

        Returns
        -------
            The records as they appear in the store.
        """

        # Validate an re-partition the records by their smiles patterns.
        records_by_smiles: Dict[str, List[MoleculeESPRecord]] = defaultdict(list)

        for record in records:

            record = MoleculeESPRecord(**record.dict())
            smiles = self._tagged_to_canonical_smiles(record.tagged_smiles)

            records_by_smiles[smiles].append(record)

        # Store the records.
        with self._get_session() as db:

            for smiles in records_by_smiles:
                self._store_smiles_records(db, smiles, records_by_smiles[smiles])

    def retrieve(
        self,
        smiles: Optional[str] = None,
        basis: Optional[str] = None,
        method: Optional[str] = None,
        implicit_solvent: Optional[bool] = None,
    ) -> List[MoleculeESPRecord]:
        """Retrieve records stored in this data store, optionally
        according to a set of filters."""

        oechem = import_oechem()

        with self._get_session() as db:

            db_records = db.query(DBMoleculeRecord)

            if smiles is not None:

                smiles = oechem.OECreateCanSmiString(smiles_to_molecule(smiles))
                db_records = db_records.filter(DBMoleculeRecord.smiles == smiles)

            if basis is not None or method is not None or implicit_solvent is not None:

                db_records = db_records.join(DBConformerRecord)

                if basis is not None or method is not None:

                    db_records = db_records.join(
                        DBESPSettings, DBConformerRecord.esp_settings
                    )

                    if basis is not None:
                        db_records = db_records.filter(DBESPSettings.basis == basis)
                    if method is not None:
                        db_records = db_records.filter(DBESPSettings.method == method)

                if implicit_solvent is not None:

                    if implicit_solvent:
                        db_records = db_records.filter(
                            DBConformerRecord.pcm_settings_id.isnot(None)
                        )
                    else:
                        db_records = db_records.filter(
                            DBConformerRecord.pcm_settings_id.is_(None)
                        )

            db_records = db_records.all()

            records = self._db_records_to_model(db_records)

            if basis:
                records = [
                    record for record in records if record.esp_settings.basis == basis
                ]
            if method:
                records = [
                    record for record in records if record.esp_settings.method == method
                ]

            return records

    def list(self) -> List[str]:
        """Lists the molecules which exist in and may be retrieved from the
        store."""

        with self._get_session() as db:
            return [smiles for (smiles,) in db.query(DBMoleculeRecord.smiles).all()]
