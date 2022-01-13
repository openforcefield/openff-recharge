import numpy
import pytest
from openff.units import unit

from openff.recharge.grids import GridGenerator, GridSettings
from openff.recharge.utilities.openeye import smiles_to_molecule


class TestGridSettings:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (GridSettings(spacing=0.1), 0.1),
            (GridSettings(spacing=0.1 * unit.angstrom), 0.1),
            (GridSettings(spacing=2.0 * unit.nanometers), 20.0),
        ],
    )
    def test_validate_spacing(self, value, expected):
        assert numpy.isclose(value.spacing, expected)

    def test_spacing_quantity(self):

        value = numpy.random.random() * unit.nanometers

        assert numpy.isclose(GridSettings(spacing=value).spacing_quantity, value)


def test_generate_fcc_grid():

    # Build a simple case monatomic test case.
    oe_molecule = smiles_to_molecule("[Ar]")
    conformer = numpy.array([[0.0, 0.0, 0.0]]) * unit.angstrom

    # Select grid the grid settings so the corners of the FCC
    # lattice should be cut off.
    grid_settings = GridSettings(
        type="fcc",
        spacing=numpy.sqrt(2) * 1.8 / 2.0,
        inner_vdw_scale=0.9,
        outer_vdw_scale=1.1,
    )

    grid = GridGenerator.generate(oe_molecule, conformer, grid_settings)

    # Ensure that the grid encompasses the correct number of grid points.
    assert len(grid) == 24
    assert not numpy.allclose(grid, 0.0 * unit.angstrom)
