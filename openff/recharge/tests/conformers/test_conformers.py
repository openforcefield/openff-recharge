from typing import Optional

import pytest
from openff.toolkit import Molecule
from openff.toolkit.tests.utils import requires_openeye

from openff.recharge.conformers import ConformerGenerator, ConformerSettings
from openff.recharge.conformers.exceptions import ConformerGenerationError
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


@pytest.mark.skip(
    reason=(
        "The toolkit (0.12+) sees this as a radical-containing molecule."
        "Need to use a different molecule."
    )
)
@requires_openeye
def test_generate_omega_conformers_error():
    with pytest.raises(
        ConformerGenerationError, match="Failed to generate conformers using OMEGA"
    ):
        ConformerGenerator.generate(
            Molecule.from_smiles("C(S(=O)[O-])(Cl)(Cl)Cl", allow_undefined_stereo=True),
            ConformerSettings(method="omega", sampling_mode="sparse", max_conformers=1),
        )

    with pytest.raises(
        ConformerGenerationError, match="ELF10 conformer selection failed"
    ):
        ConformerGenerator.generate(Molecule.from_smiles("[Mg]"), ConformerSettings())


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
