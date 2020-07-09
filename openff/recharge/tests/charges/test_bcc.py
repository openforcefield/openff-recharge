import numpy
import pytest

from openff.recharge.charges.bcc import (
    BCCGenerator,
    BCCSettings,
    BondChargeCorrection,
    original_am1bcc_corrections,
)
from openff.recharge.charges.charges import ChargeGenerator, ChargeSettings
from openff.recharge.charges.exceptions import UnableToAssignChargeError
from openff.recharge.conformers.conformers import OmegaELF10
from openff.recharge.utilities.openeye import smiles_to_molecule, match_smirks


@pytest.fixture(scope="module")
def bond_charge_corrections():
    return original_am1bcc_corrections()


def coverage_smiles():

    return [
        "C",
        "CC",
        "C=C",
        "CC=O",
        "Cc1occc1",
        "C=Cc1occc1",
        "O=Cc1occc1",
        "OCc1occc1",
        "o1cccc1c2occc2",
        "o1cccc1c2ccccc2",
        "o1ccc2ccccc12",
        "o1ccc2ccoc12",
        "Cc1ccccc1",
        "C=Cc1ccccc1",
        "COC",
        "CC=O",
        "CO",
        "C=CC=C",
        "C=CC=O",
        "C=CO",
        "O=Cc1ccccc1",
        "C(=O)O",
        "c1ccc(cc1)c2ccccc2",
        "Oc1ccccc1",
        "c1ccc2ccccc2c1",
        "COOC",
        "o1c2ccccc2c3ccccc13",
        "Oc1occc1",
        "C(=O)C=O",
        "o1ccc2occc12",
    ]


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


@pytest.mark.parametrize("smiles", coverage_smiles())
def test_am1_bcc_oe_parity(bond_charge_corrections, smiles):
    """Test that this frameworks AM1BCC implementation matches the OpenEye
    implementation."""

    # Build the molecule
    oe_molecule = smiles_to_molecule(smiles)
    # Generate a conformer for the molecule.
    conformers = OmegaELF10.generate(oe_molecule, max_conformers=1)

    # Generate a set of reference charges using the OpenEye implementation
    reference_charges = ChargeGenerator.generate(
        oe_molecule, conformers, ChargeSettings(theory="am1bcc")
    )

    # Generate a set of charges using this frameworks functions
    am1_charges = ChargeGenerator.generate(
        oe_molecule, conformers, ChargeSettings(theory="am1")
    )
    charge_corrections = BCCGenerator.generate(
        oe_molecule, BCCSettings(bond_charge_corrections=bond_charge_corrections)
    )

    implementation_charges = am1_charges + charge_corrections

    # Check that their is no difference between the implemented and
    # reference charges.
    assert numpy.allclose(reference_charges, implementation_charges)


def test_am1_bcc_missing_parameters(bond_charge_corrections):
    """Tests that the correct exception is raised when generating partial charges
    for a molecule without conformers and no conformer generator.
    """
    oe_molecule = smiles_to_molecule("C")

    with pytest.raises(UnableToAssignChargeError) as error_info:
        BCCGenerator.generate(oe_molecule, BCCSettings(bond_charge_corrections=[]))

    assert "could not be assigned a bond charge correction atom type" in str(
        error_info.value
    )


@pytest.mark.parametrize(
    "smiles",
    [
        # # 110931
        # "C[O-]",
        # # 120931
        # "C/C=C(/C)[O-]",
        # # 130931
        # "CC(N)=O",
        # "[H]/N=C(\\C)/[O-]",
        # # # 140931
        # "CC([O-])=O",
        # # # 150931
        # "[O-]C#C",
        # # 170931
        # "[nH]1cccc1",
        # "c1ccncc1",
        # "[O-]c1occc1",
        # "[O-]c1ccccn1",
        # "[O-]c1[nH]ccc1",
        # # 160931
        # "O=C1C=CCC=C1",
        # "[O-]c1ccccc1",
        # "[O-]c1cocc1",
        # "[O-]c1coc2ccccc12",
        # "[nH]1cccc1",
        # "o1cccc1",
        "[nH]1ccc2ccccc12",
        # # 230931
        # "NC=O",
        # "C[N+]([O-])=O",
        # "[nH]1cccc1",
        # "Cn1cccc1"
    ]
)
def test_delocalised_parity(bond_charge_corrections, smiles):

    # Build the molecule
    oe_molecule = smiles_to_molecule(smiles)

    # Generate a conformer for the molecule.
    conformers = OmegaELF10.generate(oe_molecule, max_conformers=1)

    # Generate a set of reference charges using the OpenEye implementation
    reference_charges = ChargeGenerator.generate(
        oe_molecule, conformers, ChargeSettings(theory="am1bcc")
    )

    # Generate a set of charges using this frameworks functions
    am1_charges = ChargeGenerator.generate(
        oe_molecule, conformers, ChargeSettings(theory="am1")
    )

    assignment_matrix = BCCGenerator.build_assignment_matrix(
        oe_molecule, BCCSettings(bond_charge_corrections=bond_charge_corrections)
    )
    applied_corrections = BCCGenerator.applied_corrections(
        assignment_matrix, BCCSettings(bond_charge_corrections=bond_charge_corrections)
    )
    charge_corrections = BCCGenerator.apply_assignment_matrix(
        assignment_matrix, BCCSettings(bond_charge_corrections=bond_charge_corrections)
    )

    implementation_charges = am1_charges + charge_corrections
    reference_charge_corrections = reference_charges - am1_charges

    x = match_smirks("[#7X3ar5,#7X3$(*~[#8])$(*~[#8]):1]-[#1:2]", oe_molecule)

    # Check that their is no difference between the implemented and
    # reference charges.
    assert numpy.allclose(reference_charges, implementation_charges)
    assert numpy.allclose(charge_corrections, reference_charge_corrections)
