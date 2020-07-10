import numpy
import pytest

from openff.recharge.charges.bcc import (
    BCCGenerator,
    BCCSettings,
    BondChargeCorrection,
    compare_openeye_parity,
    original_am1bcc_corrections,
)
from openff.recharge.charges.exceptions import UnableToAssignChargeError
from openff.recharge.utilities.openeye import smiles_to_molecule


def test_load_original_am1_bcc():
    """Tests that the original BCC values can be parsed from the
    data directory."""
    assert len(original_am1bcc_corrections()) > 0


def test_build_assignment_matrix():

    oe_molecule = smiles_to_molecule("C")

    bond_charge_corrections = [
        BondChargeCorrection(smirks="[#6:1]-[#6:2]", value=1.0, provenance={}),
        BondChargeCorrection(smirks="[#6:1]-[#1:2]", value=1.0, provenance={}),
    ]

    assignment_matrix = BCCGenerator.build_assignment_matrix(
        oe_molecule, BCCSettings(bond_charge_corrections=bond_charge_corrections)
    )

    assert assignment_matrix.shape == (5, 2)
    assert numpy.allclose(assignment_matrix[:, 0], 0)

    assert assignment_matrix[0, 1] == 4
    assert numpy.allclose(assignment_matrix[1:, 1], -1)


def test_applied_corrections():

    oe_molecule = smiles_to_molecule("C")
    bond_charge_corrections = [
        BondChargeCorrection(smirks="[#6:1]-[#6:2]", value=1.0, provenance={}),
        BondChargeCorrection(smirks="[#6:1]-[#1:2]", value=1.0, provenance={}),
    ]
    settings = BCCSettings(bond_charge_corrections=bond_charge_corrections)

    assignment_matrix = BCCGenerator.build_assignment_matrix(oe_molecule, settings)
    applied_corrections = BCCGenerator.applied_corrections(assignment_matrix, settings)

    assert len(applied_corrections) == 1
    assert applied_corrections[0] == bond_charge_corrections[1]


def test_apply_assignment():

    settings = BCCSettings(
        bond_charge_corrections=[
            BondChargeCorrection(smirks="[#1:1]-[#1:2]", value=0.0, provenance={})
        ]
    )
    assignment_matrix = numpy.array([[1], [1]])

    # Test with a valid set of BCCs
    charge_corrections = BCCGenerator.apply_assignment_matrix(
        assignment_matrix, settings
    )

    assert charge_corrections.shape == (2, 1)
    assert numpy.allclose(charge_corrections, 0.0)

    # Test with invalid BCCs
    settings.bond_charge_corrections[0].value = 1.0

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
    oe_molecule = smiles_to_molecule("C")

    with pytest.raises(UnableToAssignChargeError) as error_info:
        BCCGenerator.generate(oe_molecule, BCCSettings(bond_charge_corrections=[]))

    assert "could not be assigned a bond charge correction atom type" in str(
        error_info.value
    )


def test_generate():
    """Test that the full generate method can be called without
    error"""

    bond_charge_corrections = original_am1bcc_corrections()

    # Generate a small molecule
    oe_molecule = smiles_to_molecule("C")

    BCCGenerator.generate(
        oe_molecule, BCCSettings(bond_charge_corrections=bond_charge_corrections)
    )
