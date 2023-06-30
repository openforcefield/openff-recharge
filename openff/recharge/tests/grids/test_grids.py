import numpy
import pytest
from openff.units import unit

from openff.recharge.conformers import ConformerGenerator, ConformerSettings
from openff.recharge.grids import GridGenerator, LatticeGridSettings, MSKGridSettings
from openff.recharge.tests.data import (
    ARGON_FCC_GRID,
    UNIT_CONNOLLY_SPHERE,
    WATER_MSK_GRID,
)
from openff.recharge.utilities.molecule import smiles_to_molecule
from openff.toolkit.tests.utils import requires_openeye


class TestLatticeGridSettings:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (LatticeGridSettings(spacing=0.1), 0.1),
            (LatticeGridSettings(spacing=0.1 * unit.angstrom), 0.1),
            (LatticeGridSettings(spacing=2.0 * unit.nanometers), 20.0),
        ],
    )
    def test_validate_spacing(self, value, expected):
        assert numpy.isclose(value.spacing, expected)

    def test_spacing_quantity(self):
        value = numpy.random.random() * unit.nanometers

        assert numpy.isclose(LatticeGridSettings(spacing=value).spacing_quantity, value)


class TestMSKGridSettings:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (MSKGridSettings(density=0.1), 0.1),
            (MSKGridSettings(density=0.1 / unit.angstrom**2), 0.1),
            (MSKGridSettings(density=2.0 / unit.nanometers**2), 0.02),
        ],
    )
    def test_validate_density(self, value, expected):
        assert numpy.isclose(value.density, expected)

    def test_density_quantity(self):
        value = numpy.random.random() / unit.nanometers**2

        assert numpy.isclose(MSKGridSettings(density=value).density_quantity, value)


class TestGridGenerator:
    def test_generate_connolly_sphere(self):
        actual_sphere = GridGenerator._generate_connolly_sphere(1.0, 1.0)

        # Regression tested against respyte and psiresp
        assert actual_sphere.shape == UNIT_CONNOLLY_SPHERE.shape
        assert numpy.allclose(actual_sphere, UNIT_CONNOLLY_SPHERE)

    def test_cull_points(self):
        conformer = numpy.array(
            [
                [0.0, 0.0, 0.0],
            ]
        )
        grid = numpy.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )

        inner_radii = numpy.array([[0.45]])
        outer_radii = numpy.array([[0.55]])

        culled_grid = GridGenerator._cull_points(
            conformer, grid, inner_radii, outer_radii
        )
        assert culled_grid.shape == (1, 3)
        assert numpy.allclose(culled_grid, numpy.array([[0.5, 0.0, 0.0]]))

        culled_grid = GridGenerator._cull_points(
            conformer,
            grid,
            inner_radii,
            outer_radii,
            exclusion_mask=numpy.array([[True, False, False]]),
        )
        assert culled_grid.shape == (2, 3)
        assert numpy.allclose(
            culled_grid, numpy.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        )

    def test_generate_fcc_grid(self):
        # Build a simple case monatomic test case.
        molecule = smiles_to_molecule("[Ar]")
        conformer = numpy.array([[0.0, 0.0, 0.0]]) * unit.angstrom

        # Select grid the grid settings so the corners of the FCC
        # lattice should be cut off.
        grid_settings = LatticeGridSettings(
            type="fcc",
            spacing=numpy.sqrt(2) * 1.8 / 2.0,
            inner_vdw_scale=0.9,
            outer_vdw_scale=1.1,
        )

        grid = GridGenerator.generate(molecule, conformer, grid_settings)

        assert grid.shape == ARGON_FCC_GRID.shape
        assert numpy.allclose(grid, ARGON_FCC_GRID * unit.angstrom)

    @requires_openeye
    def test_generate_msk_grid(self):
        molecule = smiles_to_molecule("O")

        [conformer] = ConformerGenerator.generate(
            molecule, ConformerSettings(max_conformers=1)
        )

        grid = GridGenerator.generate(molecule, conformer, MSKGridSettings())

        assert grid.shape == WATER_MSK_GRID.shape
        assert numpy.allclose(grid, WATER_MSK_GRID * unit.angstrom)
