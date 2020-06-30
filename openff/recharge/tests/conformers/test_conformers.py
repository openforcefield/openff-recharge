from openff.recharge.conformers.conformers import OmegaELF10
from openff.recharge.utilities.openeye import smiles_to_molecule


def test_omega_elf10():
    """Tests the `OmegaELF10` generator can generate conformers without
    issue.
    """

    oe_molecule = smiles_to_molecule("CO")
    OmegaELF10.generate(oe_molecule, max_conformers=1)
