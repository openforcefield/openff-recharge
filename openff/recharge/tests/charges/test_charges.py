import numpy
import pytest
from typing_extensions import Literal

from openff.recharge.charges.charges import ChargeGenerator, ChargeSettings
from openff.recharge.utilities.openeye import smiles_to_molecule


@pytest.mark.parametrize("theory", ["am1", "am1bcc"])
def test_generate_charges(theory: Literal["am1", "am1bcc"]):
    """Ensure that charges can be generated for a simple molecule using
    the `ChargeGenerator` class."""

    oe_molecule = smiles_to_molecule("C")
    conformer = numpy.array(
        [
            [-0.0000658, -0.0000061, 0.0000215],
            [-0.0566733, 1.0873573, -0.0859463],
            [0.6194599, -0.3971111, -0.8071615],
            [-1.0042799, -0.4236047, -0.0695677],
            [0.4415590, -0.2666354, 0.9626540],
        ]
    )

    ChargeGenerator.generate(oe_molecule, [conformer], ChargeSettings(theory=theory))
