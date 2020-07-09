import pytest
from typing_extensions import Literal

from openff.recharge.charges.charges import ChargeGenerator, ChargeSettings
from openff.recharge.conformers.conformers import OmegaELF10
from openff.recharge.utilities.openeye import smiles_to_molecule


@pytest.mark.parametrize("theory", ["am1", "am1bcc"])
def test_generate_charges(theory: Literal["am1", "am1bcc"]):
    """Ensure that charges can be generated for a simple molecule using
    the `ChargeGenerator` class."""

    oe_molecule = smiles_to_molecule("C")
    conformers = OmegaELF10.generate(oe_molecule, 1)

    ChargeGenerator.generate(oe_molecule, conformers, ChargeSettings(theory=theory))
