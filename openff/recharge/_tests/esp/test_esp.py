import numpy
import pytest
from openff.units import unit

from openff.recharge.esp import ESPGenerator, ESPSettings
from openff.recharge.grids import LatticeGridSettings
from openff.recharge.utilities.molecule import smiles_to_molecule


def test_abstract_generate():
    """Test that the right error is raised when attempting to
    generate an esp using the abstract base class."""

    # Define the settings to use.
    settings = ESPSettings(grid_settings=LatticeGridSettings(spacing=2.0))

    # Generate a small molecule which should finish fast.
    molecule = smiles_to_molecule("C")
    conformer = numpy.zeros((molecule.n_atoms, 3)) * unit.angstrom

    with pytest.raises(NotImplementedError):
        ESPGenerator.generate(molecule, conformer, settings)
