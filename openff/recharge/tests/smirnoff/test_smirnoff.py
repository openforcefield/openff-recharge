import numpy
import pytest

from openff.recharge.charges.bcc import BCCGenerator, original_am1bcc_corrections
from openff.recharge.charges.charges import ChargeGenerator, ChargeSettings
from openff.recharge.conformers import ConformerGenerator, ConformerSettings
from openff.recharge.smirnoff.smirnoff import from_smirnoff, to_smirnoff
from openff.recharge.utilities.exceptions import MissingOptionalDependency
from openff.recharge.utilities.openeye import smiles_to_molecule


def test_collection_to_smirnoff():
    """Test that a collection of bcc parameters can be mapped to
    a SMIRNOFF `ChargeIncrementModelHandler` in a way that yields
    the same partial charges on a molecule."""
    pytest.importorskip("openforcefield")

    from openforcefield.topology import Molecule
    from openforcefield.typing.engines.smirnoff.parameters import ElectrostaticsHandler
    from simtk import unit
    from simtk.openmm import System

    smirnoff_handler = to_smirnoff(original_am1bcc_corrections())
    assert smirnoff_handler is not None

    off_molecule = Molecule.from_smiles("C(H)(H)(H)(H)")

    off_topology = off_molecule.to_topology()
    off_topology._ref_mol_to_charge_method = {off_molecule: None}

    omm_system = System()

    # noinspection PyTypeChecker
    ElectrostaticsHandler(method="PME", version="0.3").create_force(
        omm_system, off_topology
    )
    smirnoff_handler.create_force(omm_system, off_topology)

    off_charges = [
        omm_system.getForce(0)
        .getParticleParameters(i)[0]
        .value_in_unit(unit.elementary_charge)
        for i in range(5)
    ]

    oe_molecule = smiles_to_molecule("C")

    conformers = ConformerGenerator.generate(
        oe_molecule,
        ConformerSettings(method="omega", sampling_mode="sparse", max_conformers=1),
    )

    expected_charges = ChargeGenerator.generate(
        oe_molecule, conformers, ChargeSettings()
    ) + BCCGenerator.generate(smiles_to_molecule("C"), original_am1bcc_corrections())
    numpy.allclose(numpy.round(expected_charges[:, 0], 3), numpy.round(off_charges, 3))


def test_collection_from_smirnoff():
    """Test that a SMIRNOFF `ChargeIncrementModelHandler` can be mapped to
    a collection of bcc parameters
    """
    pytest.importorskip("openforcefield")

    from openforcefield.typing.engines.smirnoff.parameters import (
        ChargeIncrementModelHandler,
    )
    from simtk import unit

    bcc_value = 0.1 * unit.elementary_charge

    # noinspection PyTypeChecker
    parameter_handler = ChargeIncrementModelHandler(version="0.3")
    parameter_handler.add_parameter(
        {"smirks": "[#6:1]-[#6:2]", "charge_increment": [-bcc_value, bcc_value]}
    )
    parameter_handler.add_parameter(
        {"smirks": "[#1:1]-[#1:2]", "charge_increment": [bcc_value, -bcc_value]}
    )

    bcc_collection = from_smirnoff(parameter_handler)
    assert len(bcc_collection.parameters) == 2

    assert bcc_collection.parameters[0].smirks == "[#1:1]-[#1:2]"
    assert numpy.isclose(bcc_collection.parameters[0].value, 0.1)
    assert bcc_collection.parameters[1].smirks == "[#6:1]-[#6:2]"
    assert numpy.isclose(bcc_collection.parameters[1].value, -0.1)


def test_missing_dependency():
    """Test that the correct custom exception is raised when the
    OpenFF toolkit cannot be imported."""

    try:
        import openforcefield  # noqa F401
    except ImportError:
        pass
    else:
        pytest.skip(
            "This test should only be run in cases where the OpenFF Toolkit is not "
            "installed."
        )

    with pytest.raises(MissingOptionalDependency):
        to_smirnoff(original_am1bcc_corrections())
