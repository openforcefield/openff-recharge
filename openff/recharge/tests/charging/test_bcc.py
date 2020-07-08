import numpy
import pytest
from openeye import oechem, oequacpac

from openff.recharge.conformers.conformers import OmegaELF10
from openff.recharge.generators.bcc import AM1BCC
from openff.recharge.generators.exceptions import (
    OEQuacpacError,
    UnableToAssignChargeError,
)
from openff.recharge.utilities.exceptions import MissingConformersError
from openff.recharge.utilities.openeye import call_openeye, smiles_to_molecule


@pytest.fixture(scope="module")
def bond_charge_corrections():
    return AM1BCC.original_corrections()


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
    assert len(AM1BCC.original_corrections()) > 0


@pytest.mark.parametrize("smiles", coverage_smiles())
def test_am1_bcc_single_existing_conformer(bond_charge_corrections, smiles):
    """Test that this frameworks AM1BCC implementation matches the OpenEye
    implementation."""

    # Build the molecule
    oe_molecule = smiles_to_molecule(smiles)
    # Generate a conformer
    oe_molecule = OmegaELF10.generate(oe_molecule, max_conformers=1)

    # Generate a set of reference charges using the OpenEye implementation
    oe_reference_molecule = oechem.OEMol(oe_molecule)

    call_openeye(
        oequacpac.OEAssignCharges,
        oe_molecule,
        oequacpac.OEAM1BCCCharges(optimize=True, symmetrize=True),
        exception_type=OEQuacpacError,
    )

    reference_charges = {
        atom.GetIdx(): atom.GetPartialCharge()
        for atom in oe_reference_molecule.GetAtoms()
    }

    # Generate a set of charges using this frameworks functions
    oe_molecule, _ = AM1BCC.generate(oe_molecule, bond_charge_corrections)

    implementation_charges = {
        atom.GetIdx(): atom.GetPartialCharge() for atom in oe_molecule.GetAtoms()
    }

    # Check that their is no difference between the implemented and
    # reference charges.
    assert {*implementation_charges} == {*reference_charges}

    differences = {
        i: (
            implementation_charges[i],
            reference_charges[i],
            implementation_charges[i] - reference_charges[i],
        )
        for i in implementation_charges
        if not numpy.isclose(implementation_charges[i], reference_charges[i])
    }

    assert len(differences) == 0


def test_am1_bcc_missing_conformer():
    """Tests that the correct exception is raised when generating partial charges
    for a molecule without conformers and no conformer generator.
    """
    oe_molecule = smiles_to_molecule("C")
    oe_molecule.DeleteConfs()

    with pytest.raises(MissingConformersError):
        AM1BCC.generate(oe_molecule, [])


def test_am1_bcc_conformer_generator(bond_charge_corrections):
    """Tests that the a conformer generator can be passed to and used
    by the AM1BCC partial charge generator.
    """
    oe_molecule = smiles_to_molecule("C")
    AM1BCC.generate(oe_molecule, bond_charge_corrections, OmegaELF10)


def test_am1_bcc_missing_parameters(bond_charge_corrections):
    """Tests that the correct exception is raised when generating partial charges
    for a molecule without conformers and no conformer generator.
    """
    oe_molecule = smiles_to_molecule("C")

    with pytest.raises(UnableToAssignChargeError) as error_info:
        AM1BCC.generate(oe_molecule, [], OmegaELF10)

    assert "could not be assigned a bond charge correction atom type" in str(
        error_info.value
    )
