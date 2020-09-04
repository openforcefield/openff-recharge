import numpy
import pytest
from openeye import oechem

from openff.recharge.charges.bcc import (
    AromaticityModel,
    AromaticityModels,
    BCCCollection,
    BCCGenerator,
    BCCParameter,
    compare_openeye_parity,
    original_am1bcc_corrections,
)
from openff.recharge.charges.exceptions import UnableToAssignChargeError
from openff.recharge.utilities.openeye import smiles_to_molecule


def test_load_original_am1_bcc():
    """Tests that the original BCC values can be parsed from the
    data directory."""
    assert len(original_am1bcc_corrections().parameters) > 0


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


@pytest.mark.parametrize(
    "smiles",
    [
        "c1ccccc1",  # benzene
        "c1ccc2ccccc2c1",  # napthelene
        "c1ccc2c(c1)ccc3ccccc23",  # phenanthrene
        "c1ccc2c(c1)ccc3c4ccccc4ccc23",  # chrysene
        "c1cc2ccc3cccc4ccc(c1)c2c34",  # pyrene
        "c1cc2ccc3ccc4ccc5ccc6ccc1c7c2c3c4c5c67",  # coronene
        "Cc1ccc2cc3ccc(C)cc3cc2c1",  # 2,7-Dimethylanthracene
    ],
)
def test_am1_bcc_aromaticity_simple(smiles):
    """Checks that the custom AM1BCC aromaticity model behaves as
    expected for simple fused hydrocarbons.
    """

    oe_molecule = smiles_to_molecule(smiles)
    AromaticityModel.assign(oe_molecule, AromaticityModels.AM1BCC)

    ring_carbons = [
        atom
        for atom in oe_molecule.GetAtoms()
        if atom.GetAtomicNum() == 6 and oechem.OEAtomIsInRingSize(atom, 6)
    ]
    ring_indices = {atom.GetIdx() for atom in ring_carbons}

    assert all(atom.IsAromatic() for atom in ring_carbons)
    assert all(
        bond.IsAromatic()
        for bond in oe_molecule.GetBonds()
        if bond.GetBgnIdx() in ring_indices and bond.GetEndIdx() in ring_indices
    )


def test_am1_bcc_aromaticity_ring_size():
    """Checks that the custom AM1BCC aromaticity model behaves as
    expected fused hydrocarbons with varying ring sizes"""

    oe_molecule = smiles_to_molecule("C1CC2=CC=CC3=C2C1=CC=C3")
    AromaticityModel.assign(oe_molecule, AromaticityModels.AM1BCC)

    atoms = {atom.GetIdx(): atom for atom in oe_molecule.GetAtoms()}

    assert [not atoms[index].IsAromatic() for index in range(2)]
    assert [atoms[index].IsAromatic() for index in range(2, 12)]


@pytest.mark.parametrize(
    "aromaticity_model",
    [AromaticityModels.AM1BCC, AromaticityModels.MDL],
)
def test_aromaticity_models(aromaticity_model):

    oe_molecule = smiles_to_molecule("C")
    AromaticityModel.assign(oe_molecule, aromaticity_model)


def test_generate():
    """Test that the full generate method can be called without
    error"""

    bond_charge_corrections = original_am1bcc_corrections()

    # Generate a small molecule
    oe_molecule = smiles_to_molecule("C")

    BCCGenerator.generate(oe_molecule, bond_charge_corrections)
