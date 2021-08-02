import pytest
from openeye import oechem

from openff.recharge.aromaticity import AromaticityModel, AromaticityModels
from openff.recharge.utilities.openeye import smiles_to_molecule


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
