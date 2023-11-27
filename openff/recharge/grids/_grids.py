import functools
import itertools
from typing import TYPE_CHECKING, Literal, Optional, Union

import numpy
from openff.units import unit
from openff.recharge._pydantic import BaseModel, Field

from openff.recharge.utilities.pydantic import wrapped_float_validator
from openff.recharge.utilities.toolkits import VdWRadiiType, compute_vdw_radii

if TYPE_CHECKING:
    from openff.toolkit import Molecule

    PositiveFloat = float
else:
    from openff.recharge._pydantic import PositiveFloat


class LatticeGridSettings(BaseModel):
    """A class which encodes the settings to use when generating a
    lattice type grid."""

    type: Literal["fcc"] = Field("fcc", description="The type of grid to generate.")

    spacing: PositiveFloat = Field(
        0.5, description="The grid spacing in units of angstroms."
    )

    inner_vdw_scale: PositiveFloat = Field(
        1.4,
        description="A scalar which defines the inner radius of the shell "
        "around the molecule to retain grid points within.",
    )
    outer_vdw_scale: PositiveFloat = Field(
        2.0,
        description="A scalar which defines the outer radius of the shell "
        "around the molecule to retain grid points within.",
    )

    @property
    def spacing_quantity(self) -> unit.Quantity:
        return self.spacing * unit.angstrom

    _validate_spacing = wrapped_float_validator("spacing", unit.angstrom)


class MSKGridSettings(BaseModel):
    """A class which encodes the settings to use when generating a standard
    Merz-Singh-Kollman (MSK) grid."""

    type: Literal["msk"] = "msk"

    density: PositiveFloat = Field(
        1.0, description="The density [/Angstrom^2] of the MSK grid."
    )

    @property
    def density_quantity(self) -> unit.Quantity:
        return self.density * unit.angstrom**-2

    _validate_density = wrapped_float_validator("density", unit.angstrom**-2)


GridSettings = LatticeGridSettings  # For backwards compatability.
GridSettingsType = Union[LatticeGridSettings, MSKGridSettings]


class GridGenerator:
    """A containing methods to generate the grids upon which the
    electrostatic potential of a molecule will be computed."""

    @classmethod
    @functools.lru_cache
    def _generate_connolly_sphere(cls, radius: float, density: float) -> numpy.ndarray:
        """Generates a set of points on a sphere with a given radius and density
        according to the method described by M. Connolly.

        Parameters
        ----------
        radius
            The radius [Angstrom] of the sphere.
        density
            The density [/Angstrom^2] of the points on the sphere.

        Returns
        -------
            The coordinates of the points on the sphere
        """

        # Estimate the number of points according to `surface area * density`
        n_points = int(4 * numpy.pi * radius * radius * density)

        n_equatorial = int(numpy.sqrt(n_points * numpy.pi))  # sqrt (density) * 2 * pi
        n_latitudinal = int(n_equatorial / 2)  # 0 to 180 def so 1/2 points

        phi_per_latitude = numpy.pi * numpy.arange(n_latitudinal + 1) / n_latitudinal

        sin_phi_per_latitude = numpy.sin(phi_per_latitude)
        cos_phi_per_latitude = numpy.cos(phi_per_latitude)

        n_longitudinal_per_latitude = numpy.maximum(
            (n_equatorial * sin_phi_per_latitude).astype(int), 1
        )

        sin_phi = numpy.repeat(sin_phi_per_latitude, n_longitudinal_per_latitude)
        cos_phi = numpy.repeat(cos_phi_per_latitude, n_longitudinal_per_latitude)

        theta = numpy.concatenate(
            [
                2 * numpy.pi * numpy.arange(1, n_longitudinal + 1) / n_longitudinal
                for n_longitudinal in n_longitudinal_per_latitude
            ]
        )

        x = radius * numpy.cos(theta) * sin_phi
        y = radius * numpy.sin(theta) * sin_phi
        z = radius * cos_phi

        return numpy.stack([x, y, z]).T

    @classmethod
    def _generate_msk_shells(
        cls, conformer: numpy.ndarray, radii: numpy.ndarray, settings: MSKGridSettings
    ) -> numpy.ndarray:
        """Generates a grid of points according to the algorithm proposed by Connolly
        using the settings proposed by Merz-Singh-Kollman.

        Parameters
        ----------
        conformer
            The conformer [Angstrom] of the molecule with shape=(n_atoms, 3).
        radii
            The radii [Angstrom] of each atom in the molecule with shape=(n_atoms, 1).
        settings
            The settings that describe how the grid should be generated.

        Returns
        -------
            The coordinates [Angstrom] of the grid with shape=(n_grid_points, 3).
        """

        shells = []

        for scale in [1.4, 1.6, 1.8, 2.0]:
            atom_spheres = [
                coordinate
                + cls._generate_connolly_sphere(radius.item() * scale, settings.density)
                for radius, coordinate in zip(radii, conformer)
            ]
            shell = numpy.vstack(atom_spheres)

            n_grid_points = len(shell)
            n_atoms = len(radii)

            # Build a mask to ensure that grid points generated around an atom aren't
            # accidentally culled due to precision issues.
            exclusion_mask = numpy.zeros((n_atoms, n_grid_points), dtype=bool)

            offset = 0

            for atom_index, atom_sphere in enumerate(atom_spheres):
                exclusion_mask[atom_index, offset : offset + len(atom_sphere)] = True
                offset += len(atom_sphere)

            shells.append(
                cls._cull_points(
                    conformer, shell, radii * scale, exclusion_mask=exclusion_mask
                )
            )

        return numpy.vstack(shells)

    @classmethod
    def _generate_lattice(
        cls,
        conformer: numpy.ndarray,
        radii: numpy.ndarray,
        settings: LatticeGridSettings,
    ) -> numpy.ndarray:
        """Generates a grid of points in on a lattice around a specified
        molecule in a given conformer.

        Parameters
        ----------
        conformer
            The conformer [Angstrom] of the molecule with shape=(n_atoms, 3).
        radii
            The radii [Angstrom] of each atom in the molecule.
        settings
            The settings that describe how the grid should be generated.

        Returns
        -------
            The coordinates [Angstrom] of the grid with shape=(n_grid_points, 3).
        """

        # Compute the center of the molecule
        conformer_center = numpy.mean(conformer, axis=0)

        # Compute the bounding box which the grid should roughly fit inside of.
        minimum_bounds = numpy.min(conformer - radii * settings.outer_vdw_scale, axis=0)
        maximum_bounds = numpy.max(conformer + radii * settings.outer_vdw_scale, axis=0)

        lattice_constant = 2.0 * settings.spacing / numpy.sqrt(2.0)

        n_cells = tuple(
            int(n)
            for n in numpy.ceil((maximum_bounds - minimum_bounds) / lattice_constant)
        )

        # Compute the coordinates of the grid.
        coordinates = []

        for x, y, z in itertools.product(
            *(range(0, n * 2 + 1) for n in n_cells), repeat=1
        ):
            a = x % 2
            b = y % 2
            c = z % 2

            is_grid_point = (
                (not a and not b and not c)
                or (a and b and not c)
                or (not a and b and c)
                or (a and not b and c)
            )

            if not is_grid_point:
                continue

            coordinate = (
                numpy.array([[x - n_cells[0], y - n_cells[1], z - n_cells[2]]])
                * 0.5
                * lattice_constant
                + conformer_center
            )

            coordinates.append(coordinate)

        return cls._cull_points(
            conformer,
            numpy.concatenate(coordinates),
            radii * settings.inner_vdw_scale,
            radii * settings.outer_vdw_scale,
        )

    @classmethod
    def _cull_points(
        cls,
        conformer: numpy.ndarray,
        grid: numpy.ndarray,
        inner_radii: numpy.ndarray,
        outer_radii: Optional[numpy.ndarray] = None,
        exclusion_mask: Optional[numpy.ndarray] = None,
    ) -> numpy.ndarray:
        """Removes all points that are either within or outside a vdW shell around a
        given conformer.
        """

        from scipy.spatial.distance import cdist

        distances = cdist(conformer, grid)
        exclusion_mask = False if exclusion_mask is None else exclusion_mask

        is_within_shell = numpy.any(
            (distances < inner_radii) & (~exclusion_mask), axis=0
        )
        is_outside_shell = False

        if outer_radii is not None:
            is_outside_shell = numpy.all(
                (distances > outer_radii) & (~exclusion_mask), axis=0
            )

        discard_point = numpy.logical_or(is_within_shell, is_outside_shell)

        return grid[~discard_point]

    @classmethod
    def generate(
        cls,
        molecule: "Molecule",
        conformer: unit.Quantity,
        settings: GridSettingsType,
    ) -> unit.Quantity:
        """Generates a grid of points in a shell around a specified
        molecule in a given conformer according a set of settings.

        Parameters
        ----------
        molecule
            The molecule to generate the grid around.
        conformer
            The conformer of the molecule with shape=(n_atoms, 3).
        settings
            The settings which describe how the grid should
            be generated.

        Returns
        -------
            The coordinates of the grid with shape=(n_grid_points, 3).
        """

        conformer = conformer.to(unit.angstrom).m

        vdw_radii = compute_vdw_radii(molecule, radii_type=VdWRadiiType.Bondi)
        radii_array = numpy.array([[radii] for radii in vdw_radii.m_as(unit.angstrom)])

        if isinstance(settings, LatticeGridSettings):
            coordinates = cls._generate_lattice(conformer, radii_array, settings)
        elif isinstance(settings, MSKGridSettings):
            coordinates = cls._generate_msk_shells(conformer, radii_array, settings)
        else:
            raise NotImplementedError()

        return coordinates * unit.angstrom
