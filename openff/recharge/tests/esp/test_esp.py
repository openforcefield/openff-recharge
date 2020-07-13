import numpy
import pytest

from openff.recharge.esp import ESPGenerator, ESPSettings
from openff.recharge.grids import GridSettings
from openff.recharge.utilities.openeye import smiles_to_molecule


def test_abstract_generate():
    """Test that the right error is raised when attempting to
    generate an esp using the abstract base class."""

    # Define the settings to use.
    settings = ESPSettings(grid_settings=GridSettings(spacing=2.0))

    # Generate a small molecule which should finish fast.
    oe_molecule = smiles_to_molecule("C")
    conformer = numpy.zeros((oe_molecule.NumAtoms(), 3))

    with pytest.raises(NotImplementedError):
        ESPGenerator.generate(oe_molecule, conformer, settings)
