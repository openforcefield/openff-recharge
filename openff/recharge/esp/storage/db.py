from sqlalchemy import (
    Column,
    Float,
    ForeignKey,
    Integer,
    PrimaryKeyConstraint,
    String,
    Table,
    UniqueConstraint,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship

from openff.recharge.esp import ESPSettings

DBBase = declarative_base()

conformer_esp_settings_table = Table(
    "conformer_esp_settings",
    DBBase.metadata,
    Column("conformer_id", Integer, ForeignKey("conformers.id")),
    Column("esp_settings_id", Integer, ForeignKey("esp_settings.id")),
    PrimaryKeyConstraint("conformer_id", "esp_settings_id"),
)


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


class DBGridSettings(DBBase):

    __tablename__ = "grid_settings"

    id = Column(Integer, primary_key=True, index=True)
    parent_id = Column(Integer, ForeignKey("conformers.id"), nullable=False)

    type = Column(String, nullable=False)
    spacing = Column(Float, nullable=False)

    inner_vdw_scale = Column(Float, nullable=False)
    outer_vdw_scale = Column(Float, nullable=False)


class DBESPSettings(DBBase):

    __tablename__ = "esp_settings"
    __table_args__ = (UniqueConstraint("basis", "method"),)

    id = Column(Integer, primary_key=True, index=True)

    basis = Column(String, nullable=False)
    method = Column(String, nullable=False)

    @classmethod
    def unique(cls, db: Session, instance: ESPSettings) -> "DBESPSettings":
        """Creates a new ``DBESPSettings`` object from the specified
        instance if it does not already exist on the database, otherwise
        the existing instance is returned.
        """

        db_esp_settings = (
            db.query(DBESPSettings)
            .filter(DBESPSettings.basis == instance.basis)
            .filter(DBESPSettings.method == instance.method)
            .first()
        )

        if db_esp_settings:
            return db_esp_settings

        return DBESPSettings(**instance.dict(exclude={"grid_settings"}))


class DBConformerRecord(DBBase):

    __tablename__ = "conformers"

    id = Column(Integer, primary_key=True, index=True)
    parent_id = Column(String, ForeignKey("molecules.smiles"), nullable=False)

    tagged_smiles = Column(String, nullable=False)

    coordinates = relationship("DBCoordinate")
    grid_esp_values = relationship("DBGridESP")

    grid_settings = relationship("DBGridSettings", uselist=False)
    esp_settings = relationship(
        "DBESPSettings", secondary=conformer_esp_settings_table, uselist=False
    )


class DBMoleculeRecord(DBBase):

    __tablename__ = "molecules"

    smiles = Column(String, primary_key=True, index=True)
    conformers = relationship("DBConformerRecord")
