import pytest

from openff.recharge.esp import ESPGenerator, ESPSettings
from openff.recharge.grids import GridSettings
from openff.recharge.utilities.exceptions import MissingConformersError
from openff.recharge.utilities.openeye import smiles_to_molecule


def test_generate_missing_conformers():
    """Test that the right error is raised when conformers are missing."""

    # Define the settings to use.
    settings = ESPSettings(grid_settings=GridSettings(spacing=2.0))

    # Generate a small molecule which should finish fast.
    oe_molecule = smiles_to_molecule("C")
    oe_molecule.DeleteConfs()

    with pytest.raises(MissingConformersError):
        ESPGenerator.generate(oe_molecule, settings)
