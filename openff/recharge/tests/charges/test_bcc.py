import sys

import numpy
import pytest
from openff.utilities.exceptions import MissingOptionalDependency

from openff.recharge.charges.bcc import (
    BCCCollection,
    BCCGenerator,
    BCCParameter,
    compare_openeye_parity,
    original_am1bcc_corrections,
)
from openff.recharge.charges.charges import ChargeGenerator, ChargeSettings
from openff.recharge.charges.exceptions import UnableToAssignChargeError
from openff.recharge.conformers import ConformerGenerator, ConformerSettings
from openff.recharge.utilities.openeye import smiles_to_molecule


def test_load_original_am1_bcc():
    """Tests that the original BCC values can be parsed from the
    data directory."""
    assert len(original_am1bcc_corrections().parameters) > 0


def test_to_smirnoff():
    """Test that a collection of bcc parameters can be mapped to a SMIRNOFF
    `ChargeIncrementModelHandler` in a way that yields the same partial charges on a
    molecule.
    """

    pytest.importorskip("openff.toolkit")

    from openff.toolkit.topology import Molecule
    from openff.toolkit.typing.engines.smirnoff.parameters import ElectrostaticsHandler
    from simtk import unit
    from simtk.openmm import System

    smirnoff_handler = original_am1bcc_corrections().to_smirnoff()
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


def test_to_smirnoff_missing_dependency(monkeypatch):
    """Test that the correct custom exception is raised when the
    OpenFF toolkit cannot be imported."""

    monkeypatch.setitem(sys.modules, "openff.toolkit", None)

    with pytest.raises(MissingOptionalDependency):
        original_am1bcc_corrections().to_smirnoff()


def test_from_smirnoff():
    """Test that a SMIRNOFF `ChargeIncrementModelHandler` can be mapped to
    a collection of bcc parameters
    """
    pytest.importorskip("openff.toolkit")

    from openff.toolkit.typing.engines.smirnoff.parameters import (
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
        {"smirks": "[#1:1]-[#1:2]", "charge_increment": [bcc_value]}
    )

    bcc_collection = BCCCollection.from_smirnoff(parameter_handler)
    assert len(bcc_collection.parameters) == 2

    assert bcc_collection.parameters[0].smirks == "[#1:1]-[#1:2]"
    assert numpy.isclose(bcc_collection.parameters[0].value, 0.1)
    assert bcc_collection.parameters[1].smirks == "[#6:1]-[#6:2]"
    assert numpy.isclose(bcc_collection.parameters[1].value, -0.1)


def test_build_assignment_matrix():

    oe_molecule = smiles_to_molecule("C")

    bond_charge_corrections = [
        BCCParameter(smirks="[#6:1]-[#6:2]", value=1.0, provenance={}),
        BCCParameter(smirks="[#6:1]-[#1:2]", value=1.0, provenance={}),
    ]

    assignment_matrix = BCCGenerator.build_assignment_matrix(
        oe_molecule, BCCCollection(parameters=bond_charge_corrections)
    )

    assert assignment_matrix.shape == (5, 2)
    assert numpy.allclose(assignment_matrix[:, 0], 0)

    assert assignment_matrix[0, 1] == 4
    assert numpy.allclose(assignment_matrix[1:, 1], -1)


def test_applied_corrections():

    bcc_collection = BCCCollection(
        parameters=[
            BCCParameter(smirks="[#6:1]-[#6:2]", value=1.0, provenance={}),
            BCCParameter(smirks="[#6:1]-[#1:2]", value=1.0, provenance={}),
        ]
    )

    applied_corrections = BCCGenerator.applied_corrections(
        smiles_to_molecule("C"), bcc_collection=bcc_collection
    )

    assert len(applied_corrections) == 1
    assert applied_corrections[0] == bcc_collection.parameters[1]


def test_applied_corrections_order():
    """Ensure that the applied corrections are returned in the correct order
    when applying them to multiple molecules."""

    bcc_collection = BCCCollection(
        parameters=[
            BCCParameter(smirks="[#7:1]-[#1:2]", value=1.0, provenance={}),
            BCCParameter(smirks="[#6:1]-[#1:2]", value=1.0, provenance={}),
        ]
    )

    applied_corrections = BCCGenerator.applied_corrections(
        smiles_to_molecule("C"), smiles_to_molecule("N"), bcc_collection=bcc_collection
    )

    assert len(applied_corrections) == 2

    assert applied_corrections[0] == bcc_collection.parameters[0]
    assert applied_corrections[1] == bcc_collection.parameters[1]


def test_apply_assignment():

    settings = BCCCollection(
        parameters=[BCCParameter(smirks="[#1:1]-[#1:2]", value=0.0, provenance={})]
    )
    assignment_matrix = numpy.array([[1], [1]])

    # Test with a valid set of BCCs
    charge_corrections = BCCGenerator.apply_assignment_matrix(
        assignment_matrix, settings
    )

    assert charge_corrections.shape == (2, 1)
    assert numpy.allclose(charge_corrections, 0.0)

    # Test with invalid BCCs
    settings.parameters[0].value = 1.0

    with pytest.raises(UnableToAssignChargeError) as error_info:
        BCCGenerator.apply_assignment_matrix(assignment_matrix, settings)

    assert "the total charge of the molecule will be altered." in str(error_info.value)


def test_compare_openeye_parity():
    """Test that the OE parity functions as expected."""
    assert compare_openeye_parity(smiles_to_molecule("C"))


def test_am1_bcc_missing_parameters():
    """Tests that the correct exception is raised when generating partial charges
    for a molecule without conformers and no conformer generator.
    """
    oe_molecule = smiles_to_molecule("o1cccc1")

    with pytest.raises(UnableToAssignChargeError) as error_info:
        BCCGenerator.generate(oe_molecule, BCCCollection(parameters=[]))

    assert "could not be assigned a bond charge correction atom type" in str(
        error_info.value
    )


def test_generate():
    """Test that the full generate method can be called without
    error"""

    bond_charge_corrections = original_am1bcc_corrections()

    # Generate a small molecule
    oe_molecule = smiles_to_molecule("C")

    BCCGenerator.generate(oe_molecule, bond_charge_corrections)
