import abc
import math
from decimal import Decimal
from typing import TypeVar

from sqlalchemy import (
    DECIMAL,
    Boolean,
    Column,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Query, Session, relationship

from openff.recharge.esp import ESPSettings, PCMSettings
from openff.recharge.grids import GridSettings

DBBase = declarative_base()

_InstanceType = TypeVar("_InstanceType")
_DBInstanceType = TypeVar("_DBInstanceType")

FLOAT_PRECISION = 100000.0


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


class DBCoordinate(DBBase):

    __tablename__ = "coordinates"

    id = Column(Integer, primary_key=True, index=True)
    parent_id = Column(Integer, ForeignKey("conformers.id"), nullable=False)

    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    z = Column(Float, nullable=False)


class DBGridESP(DBBase):

    __tablename__ = "grid_esp_values"

    id = Column(Integer, primary_key=True, index=True)
    parent_id = Column(Integer, ForeignKey("conformers.id"), nullable=False)

    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    z = Column(Float, nullable=False)

    value = Column(Float, nullable=False)


class DBGridSettings(_UniqueMixin, DBBase):

    __tablename__ = "grid_settings"

    id = Column(Integer, primary_key=True, index=True)

    type = Column(String, nullable=False)
    spacing = Column(DECIMAL, nullable=False)

    inner_vdw_scale = Column(DECIMAL, nullable=False)
    outer_vdw_scale = Column(DECIMAL, nullable=False)

    @classmethod
    def _hash(cls, instance: GridSettings) -> int:
        return hash(
            (
                instance.type,
                f"{instance.spacing:.5f}",
                f"{instance.inner_vdw_scale:.5f}",
                f"{instance.outer_vdw_scale:.5f}",
            )
        )

    @classmethod
    def _query(cls, db: Session, instance: GridSettings) -> Query:

        spacing = Decimal(round(instance.spacing, 5))
        inner_vdw_scale = Decimal(round(instance.inner_vdw_scale, 5))
        outer_vdw_scale = Decimal(round(instance.outer_vdw_scale, 5))

        return (
            db.query(DBGridSettings)
            .filter(DBGridSettings.type == instance.type)
            .filter(DBGridSettings.spacing == spacing)
            .filter(DBGridSettings.inner_vdw_scale == inner_vdw_scale)
            .filter(DBGridSettings.outer_vdw_scale == outer_vdw_scale)
        )

    @classmethod
    def _instance_to_db(cls, instance: GridSettings) -> "DBGridSettings":

        spacing = Decimal(round(instance.spacing, 5))
        inner_vdw_scale = Decimal(round(instance.inner_vdw_scale, 5))
        outer_vdw_scale = Decimal(round(instance.outer_vdw_scale, 5))

        return DBGridSettings(
            type=instance.type,
            spacing=spacing,
            inner_vdw_scale=inner_vdw_scale,
            outer_vdw_scale=outer_vdw_scale,
        )


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
                int(math.floor(instance.cavity_area * FLOAT_PRECISION)),
            )
        )

    @classmethod
    def _query(cls, db: Session, instance: PCMSettings) -> Query:

        cavity_area = int(math.floor(instance.cavity_area * FLOAT_PRECISION))

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

        cavity_area = int(math.floor(instance.cavity_area * FLOAT_PRECISION))

        return DBPCMSettings(
            solver=instance.solver,
            solvent=instance.solvent,
            radii_model=instance.radii_model,
            radii_scaling=instance.radii_scaling,
            cavity_area=cavity_area,
        )

    @classmethod
    def db_to_instance(cls, db_instance: "DBPCMSettings") -> PCMSettings:

        cavity_area = db_instance.cavity_area / FLOAT_PRECISION

        # noinspection PyTypeChecker
        return PCMSettings(
            solver=db_instance.solver,
            solvent=db_instance.solvent,
            radii_model=db_instance.radii_model,
            radii_scaling=db_instance.radii_scaling,
            cavity_area=cavity_area,
        )


class DBESPSettings(_UniqueMixin, DBBase):

    __tablename__ = "esp_settings"
    __table_args__ = (UniqueConstraint("basis", "method"),)

    id = Column(Integer, primary_key=True, index=True)

    basis = Column(String, nullable=False)
    method = Column(String, nullable=False)

    @classmethod
    def _hash(cls, instance: ESPSettings) -> int:
        return hash((instance.basis, instance.method))

    @classmethod
    def _query(cls, db: Session, instance: ESPSettings) -> Query:
        return (
            db.query(DBESPSettings)
            .filter(DBESPSettings.basis == instance.basis)
            .filter(DBESPSettings.method == instance.method)
        )

    @classmethod
    def _instance_to_db(cls, instance: ESPSettings) -> "DBESPSettings":
        return DBESPSettings(**instance.dict(exclude={"grid_settings", "pcm_settings"}))


class DBConformerRecord(DBBase):

    __tablename__ = "conformers"

    id = Column(Integer, primary_key=True, index=True)
    parent_id = Column(String, ForeignKey("molecules.smiles"), nullable=False)

    tagged_smiles = Column(String, nullable=False)

    coordinates = relationship("DBCoordinate")
    grid_esp_values = relationship("DBGridESP")

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
