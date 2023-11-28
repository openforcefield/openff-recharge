"""Utilities for storing data in a SQLite database"""
import abc
import math
from typing import TypeVar

from sqlalchemy import (
    Boolean,
    Column,
    ForeignKey,
    Integer,
    PickleType,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Query, Session, relationship, declarative_base

from openff.recharge.esp import ESPSettings, PCMSettings
from openff.recharge.grids import GridSettingsType, LatticeGridSettings, MSKGridSettings

DBBase = declarative_base()

_InstanceType = TypeVar("_InstanceType")
_DBInstanceType = TypeVar("_DBInstanceType")

DB_VERSION = 2
_DB_FLOAT_PRECISION = 100000.0


def _float_to_db_int(value: float) -> int:
    return int(math.floor(value * _DB_FLOAT_PRECISION))


def _db_int_to_float(value: int) -> float:
    return value / _DB_FLOAT_PRECISION


class _UniqueMixin:
    """A base class for records which should be unique in the
    database."""

    @classmethod
    @abc.abstractmethod
    def _hash(cls, instance: _InstanceType) -> int:
        """Returns the hash of the instance that this record represents."""
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def _query(cls, db: Session, instance: _InstanceType) -> Query:
        """Returns a query which should find existing copies of an instance."""
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def _instance_to_db(cls, instance: _InstanceType) -> _DBInstanceType:
        """Map an instance into a database version of itself."""
        raise NotImplementedError()

    @classmethod
    def unique(cls, db: Session, instance: _InstanceType) -> _DBInstanceType:
        """Creates a new database object from the specified instance if it
        does not already exist on the database, otherwise the existing
        instance is returned.
        """

        cache = getattr(db, "_unique_cache", None)

        if cache is None:
            db._unique_cache = cache = {}

        key = (cls, cls._hash(instance))

        if key in cache:
            return cache[key]

        with db.no_autoflush:
            existing_instance = cls._query(db, instance).first()

            if not existing_instance:
                existing_instance = cls._instance_to_db(instance)
                db.add(existing_instance)

        cache[key] = existing_instance
        return existing_instance


class DBGridSettings(_UniqueMixin, DBBase):
    __tablename__ = "grid_settings"

    id = Column(Integer, primary_key=True, index=True)

    type = Column(String, nullable=False)

    lattice_spacing = Column(Integer, nullable=True)
    lattice_inner_vdw_scale = Column(Integer, nullable=True)
    lattice_outer_vdw_scale = Column(Integer, nullable=True)

    msk_density = Column(Integer, nullable=True)

    @classmethod
    def _hash(cls, instance: GridSettingsType) -> int:
        if isinstance(instance, LatticeGridSettings):
            return hash(
                (
                    instance.type,
                    _float_to_db_int(instance.spacing),
                    _float_to_db_int(instance.inner_vdw_scale),
                    _float_to_db_int(instance.outer_vdw_scale),
                )
            )

        elif isinstance(instance, MSKGridSettings):
            return hash((instance.type, _float_to_db_int(instance.density)))

        else:
            raise NotImplementedError()

    @classmethod
    def _query(cls, db: Session, instance: GridSettingsType) -> Query:
        if isinstance(instance, LatticeGridSettings):
            spacing = _float_to_db_int(instance.spacing)
            inner_vdw_scale = _float_to_db_int(instance.inner_vdw_scale)
            outer_vdw_scale = _float_to_db_int(instance.outer_vdw_scale)

            return (
                db.query(DBGridSettings)
                .filter(DBGridSettings.type == instance.type)
                .filter(DBGridSettings.lattice_spacing == spacing)
                .filter(DBGridSettings.lattice_inner_vdw_scale == inner_vdw_scale)
                .filter(DBGridSettings.lattice_outer_vdw_scale == outer_vdw_scale)
            )

        elif isinstance(instance, MSKGridSettings):
            density = _float_to_db_int(instance.density)

            return (
                db.query(DBGridSettings)
                .filter(DBGridSettings.type == instance.type)
                .filter(DBGridSettings.msk_density == density)
            )

        else:
            raise NotImplementedError()

    @classmethod
    def _instance_to_db(cls, instance: GridSettingsType) -> "DBGridSettings":
        if isinstance(instance, LatticeGridSettings):
            return DBGridSettings(
                type=instance.type,
                lattice_spacing=_float_to_db_int(instance.spacing),
                lattice_inner_vdw_scale=_float_to_db_int(instance.inner_vdw_scale),
                lattice_outer_vdw_scale=_float_to_db_int(instance.outer_vdw_scale),
            )

        elif isinstance(instance, MSKGridSettings):
            return DBGridSettings(
                type=instance.type,
                msk_density=_float_to_db_int(instance.density),
            )

        else:
            raise NotImplementedError()

    @classmethod
    def db_to_instance(cls, db_instance: "DBGridSettings") -> GridSettingsType:
        if db_instance.type in ["fcc"]:
            # noinspection PyTypeChecker
            return LatticeGridSettings(
                type=db_instance.type,
                spacing=_db_int_to_float(db_instance.lattice_spacing),
                inner_vdw_scale=_db_int_to_float(db_instance.lattice_inner_vdw_scale),
                outer_vdw_scale=_db_int_to_float(db_instance.lattice_outer_vdw_scale),
            )
        elif db_instance.type == "msk":
            # noinspection PyTypeChecker
            return MSKGridSettings(
                type=db_instance.type, density=_db_int_to_float(db_instance.msk_density)
            )
        else:
            raise NotImplementedError()


class DBPCMSettings(_UniqueMixin, DBBase):
    __tablename__ = "pcm_settings"

    id = Column(Integer, primary_key=True, index=True)

    solver = Column(String(6), nullable=False)
    solvent = Column(String(20), nullable=False)

    radii_model = Column(String(8), nullable=False)
    radii_scaling = Column(Boolean, nullable=False)

    cavity_area = Column(Integer)

    @classmethod
    def _hash(cls, instance: PCMSettings) -> int:
        return hash(
            (
                instance.solver,
                instance.solvent,
                instance.radii_model,
                instance.radii_scaling,
                _float_to_db_int(instance.cavity_area),
            )
        )

    @classmethod
    def _query(cls, db: Session, instance: PCMSettings) -> Query:
        cavity_area = _float_to_db_int(instance.cavity_area)

        return (
            db.query(DBPCMSettings)
            .filter(DBPCMSettings.solver == instance.solver)
            .filter(DBPCMSettings.solvent == instance.solvent)
            .filter(DBPCMSettings.radii_model == instance.radii_model)
            .filter(DBPCMSettings.radii_scaling == instance.radii_scaling)
            .filter(DBPCMSettings.cavity_area == cavity_area)
        )

    @classmethod
    def _instance_to_db(cls, instance: PCMSettings) -> "DBPCMSettings":
        return DBPCMSettings(
            solver=instance.solver,
            solvent=instance.solvent,
            radii_model=instance.radii_model,
            radii_scaling=instance.radii_scaling,
            cavity_area=_float_to_db_int(instance.cavity_area),
        )

    @classmethod
    def db_to_instance(cls, db_instance: "DBPCMSettings") -> PCMSettings:
        # noinspection PyTypeChecker
        return PCMSettings(
            solver=db_instance.solver,
            solvent=db_instance.solvent,
            radii_model=db_instance.radii_model,
            radii_scaling=db_instance.radii_scaling,
            cavity_area=_db_int_to_float(db_instance.cavity_area),
        )


class DBESPSettings(_UniqueMixin, DBBase):
    __tablename__ = "esp_settings"
    __table_args__ = (UniqueConstraint("basis", "method"),)

    id = Column(Integer, primary_key=True, index=True)

    basis = Column(String, index=True, nullable=False)
    method = Column(String, index=True, nullable=False)

    psi4_dft_grid_settings = Column(String, nullable=False)

    @classmethod
    def _hash(cls, instance: ESPSettings) -> int:
        return hash(
            (instance.basis, instance.method, instance.psi4_dft_grid_settings.value)
        )

    @classmethod
    def _query(cls, db: Session, instance: ESPSettings) -> Query:
        return (
            db.query(DBESPSettings)
            .filter(DBESPSettings.basis == instance.basis)
            .filter(DBESPSettings.method == instance.method)
            .filter(
                DBESPSettings.psi4_dft_grid_settings
                == instance.psi4_dft_grid_settings.value
            )
        )

    @classmethod
    def _instance_to_db(cls, instance: ESPSettings) -> "DBESPSettings":
        return DBESPSettings(
            **instance.dict(
                exclude={"grid_settings", "pcm_settings", "psi4_dft_grid_settings"}
            ),
            psi4_dft_grid_settings=instance.psi4_dft_grid_settings.value
        )


class DBConformerRecord(DBBase):
    __tablename__ = "conformers"

    id = Column(Integer, primary_key=True, index=True)
    parent_id = Column(String, ForeignKey("molecules.smiles"), nullable=False)

    tagged_smiles = Column(String, nullable=False)

    coordinates = Column(PickleType, nullable=False)

    grid = Column(PickleType, nullable=False)
    esp = Column(PickleType, nullable=False)
    field = Column(PickleType, nullable=True)

    grid_settings = relationship("DBGridSettings", uselist=False)
    grid_settings_id = Column(Integer, ForeignKey("grid_settings.id"), nullable=False)

    pcm_settings = relationship("DBPCMSettings", uselist=False)
    pcm_settings_id = Column(Integer, ForeignKey("pcm_settings.id"), nullable=True)

    esp_settings = relationship("DBESPSettings", uselist=False)
    esp_settings_id = Column(Integer, ForeignKey("esp_settings.id"), nullable=False)


class DBMoleculeRecord(DBBase):
    __tablename__ = "molecules"

    smiles = Column(String, primary_key=True, index=True)
    conformers = relationship("DBConformerRecord")


class DBGeneralProvenance(DBBase):
    __tablename__ = "general_provenance"

    key = Column(String, primary_key=True, index=True, unique=True)
    value = Column(String, nullable=False)

    parent_id = Column(Integer, ForeignKey("db_info.version"))


class DBSoftwareProvenance(DBBase):
    __tablename__ = "software_provenance"

    key = Column(String, primary_key=True, index=True, unique=True)
    value = Column(String, nullable=False)

    parent_id = Column(Integer, ForeignKey("db_info.version"))


class DBInformation(DBBase):
    """A class which keeps track of the current database
    settings.
    """

    __tablename__ = "db_info"

    version = Column(Integer, primary_key=True)

    general_provenance = relationship(
        "DBGeneralProvenance", cascade="all, delete-orphan"
    )
    software_provenance = relationship(
        "DBSoftwareProvenance", cascade="all, delete-orphan"
    )
