from typing import Optional

import pytest

from openff.recharge.conformers import ConformerGenerator, ConformerSettings
from openff.recharge.utilities.openeye import smiles_to_molecule


def test_max_conformers():
    """Tests the conformer generator returns back a number of conformers less than
    or equal to the maximum.
    """
    oe_molecule = smiles_to_molecule("CCOCO")

    assert (
        len(
            ConformerGenerator.generate(
                oe_molecule, ConformerSettings(max_conformers=1)
            )
        )
        == 1
    )


@pytest.mark.parametrize("method", ["omega", "omega-elf10"])
@pytest.mark.parametrize("sampling_mode", ["dense", "sparse"])
@pytest.mark.parametrize("max_conformers", [1, None])
def test_generate_conformers(
    method: str, sampling_mode: str, max_conformers: Optional[int]
):
    """Tests conformer generator."""
    oe_molecule = smiles_to_molecule("CO")

    # noinspection PyTypeChecker
    ConformerGenerator.generate(
        oe_molecule,
        ConformerSettings(
            method=method, sampling_mode=sampling_mode, max_conformers=max_conformers
        ),
    )
