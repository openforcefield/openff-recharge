import pytest

from openff.recharge.aromaticity import AromaticityModel, AromaticityModels
from openff.recharge.utilities.molecule import find_ring_bonds, smiles_to_molecule


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

    molecule = smiles_to_molecule(smiles)

    ring_bonds = {
        pair for pair, is_in_ring in find_ring_bonds(molecule).items() if is_in_ring
    }
    ring_atoms = {index for pair in ring_bonds for index in pair}

    is_atom_aromatic, is_bond_aromatic = AromaticityModel.apply(
        molecule, AromaticityModels.AM1BCC
    )

    assert all(is_atom_aromatic[index] for index in ring_atoms)
    assert all(is_bond_aromatic[pair] for pair in ring_bonds)


def test_am1_bcc_aromaticity_ring_size():
    """Checks that the custom AM1BCC aromaticity model behaves as
    expected fused hydrocarbons with varying ring sizes"""

    molecule = smiles_to_molecule("C1CC2=CC=CC3=C2C1=CC=C3")

    is_atom_aromatic, is_bond_aromatic = AromaticityModel.apply(
        molecule, AromaticityModels.AM1BCC
    )

    assert [not is_atom_aromatic[index] for index in range(2)]
    assert [is_atom_aromatic[index] for index in range(2, 12)]


@pytest.mark.parametrize(
    "aromaticity_model",
    [AromaticityModels.AM1BCC, AromaticityModels.MDL],
)
def test_aromaticity_models(aromaticity_model):
    molecule = smiles_to_molecule("C")

    is_atom_aromatic, is_bond_aromatic = AromaticityModel.apply(
        molecule, aromaticity_model
    )

    assert not any(is_atom_aromatic.values())
    assert not any(is_bond_aromatic.values())
