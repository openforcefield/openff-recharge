from typing import Optional

import pytest

from openff.recharge.conformers import ConformerGenerator, ConformerSettings
from openff.recharge.utilities.molecule import smiles_to_molecule


@pytest.mark.parametrize("max_conformers", [1, 2])
def test_max_conformers(max_conformers):
    """Tests the conformer generator returns a number of conformers less than
    or equal to the maximum.
    """
    molecule = smiles_to_molecule("CCOCO")

    assert (
        len(
            ConformerGenerator.generate(
                molecule, ConformerSettings(max_conformers=max_conformers)
            )
        )
        == max_conformers
    )


@pytest.mark.parametrize("method", ["omega", "omega-elf10"])
@pytest.mark.parametrize("sampling_mode", ["dense", "sparse"])
@pytest.mark.parametrize("max_conformers", [1, None])
def test_generate_conformers(
    method: str, sampling_mode: str, max_conformers: Optional[int]
):
    """Tests conformer generator."""
    molecule = smiles_to_molecule("CO")

    # noinspection PyTypeChecker
    ConformerGenerator.generate(
        molecule,
        ConformerSettings(
            method=method, sampling_mode=sampling_mode, max_conformers=max_conformers
        ),
    )
