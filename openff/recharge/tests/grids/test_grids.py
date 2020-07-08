import numpy

from openff.recharge.grids import GridGenerator, GridSettings
from openff.recharge.utilities.openeye import smiles_to_molecule


def test_generate_fcc_grid():

    # Build a simple case monatomic test case.
    oe_molecule = smiles_to_molecule("[Ar]")
    conformer = numpy.array([[0.0, 0.0, 0.0]])

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
