import abc
import os
from enum import Enum
from typing import TYPE_CHECKING, Optional, Tuple

from openff.units import unit
from pydantic import BaseModel, Field
from typing_extensions import Literal

from openff.recharge.grids import GridGenerator, GridSettingsType

if TYPE_CHECKING:
    from openeye import oechem

    PositiveFloat = float
else:
    from pydantic import PositiveFloat


class DFTGridSettings(Enum):
    """An enumeration of the possible DFT grid settings to use when computing
    properties using PSI4.

    * Default - The values of `dft_spherical_points`, `dft_radial_points`,
      and `dft_pruning_scheme` are not explicitly set and are left for Psi4 to
      select.
    * Medium - `dft_spherical_points=434`, `dft_radial_points=85`,
      `dft_pruning_scheme=robust` [1]_.
    * Fine - `dft_spherical_points=590`, `dft_radial_points=99`,
      `dft_pruning_scheme=robust` [2]_.

    References
    ----------
    [1] http://forum.psicode.org/t/dft-scf-not-converging/1725/7 (accessed 22/09/2020)
    [2] http://www.psicode.org/psi4manual/1.3.2/dft.html#grid-selection
        (accessed 22/09/2020)
    """

    Default = "default"
    Medium = "medium"
    Fine = "fine"


class PCMSettings(BaseModel):
    """A class which describes the polarizable continuum model (PCM)
    to include in the calculation of an ESP.
    """

    solver: Literal["CPCM", "IEFPCM"] = Field("CPCM", description="The solver to use.")

    solvent: Literal["Water"] = Field(
        "Water",
        description="The solvent to simulate. This controls the dielectric constant "
        "of the model.",
    )

    radii_model: Literal["Bondi", "UFF", "Allinger"] = Field(
        "Bondi",
        description="The type of atomic radii to use when computing the molecular "
        "cavity.",
    )
    radii_scaling: bool = Field(
        True, description="Whether to scale the atomic radii by a factor of 1.2."
    )

    cavity_area: PositiveFloat = Field(
        0.3, description="The average area of the surface partition for the cavity."
    )


class ESPSettings(BaseModel):
    """A class which contains the settings to use in an ESP calculation."""

    basis: str = Field(
        "6-31g*", description="The basis set to use in the ESP calculation."
    )
    method: str = Field("hf", description="The method to use in the ESP calculation.")

    grid_settings: GridSettingsType = Field(
        ...,
        description="The settings to use when generating the grid to generate the "
        "electrostatic potential on.",
    )

    pcm_settings: Optional[PCMSettings] = Field(
        None,
        description="The settings to use if including a polarizable continuum "
        "model in the ESP calculation.",
    )

    psi4_dft_grid_settings: DFTGridSettings = Field(
        DFTGridSettings.Default,
        description="The DFT grid settings to use when performing computations with "
        "Psi4.",
    )


class ESPGenerator(abc.ABC):
    """A base class for classes which are able to generate the electrostatic
    potential of a molecule on a specified grid.
    """

    @classmethod
    @abc.abstractmethod
    def _generate(
        cls,
        oe_molecule: "oechem.OEMol",
        conformer: unit.Quantity,
        grid: unit.Quantity,
        settings: ESPSettings,
        directory: str = None,
    ) -> Tuple[unit.Quantity, unit.Quantity]:
        """The implementation of the public ``generate`` function which
        should return the ESP for the provided conformer.

        Parameters
        ----------
        oe_molecule
            The molecule to generate the ESP for.
        conformer
            The conformer of the molecule to generate the ESP for.
        grid
            The grid to generate the ESP on with shape=(n_grid_points, 3).
        settings
            The settings to use when generating the ESP.
        directory
            The directory to run the calculation in. If none is specified,
            a temporary directory will be created and used.

        Returns
        -------
            The ESP [Hartree / e] at each grid point with shape=(n_grid_points, 1)
            and the electric field [Hartree / (e . a0)] with shape=(n_grid_points, 3).
        """
        raise NotImplementedError

    @classmethod
    def generate(
        cls,
        oe_molecule: "oechem.OEMol",
        conformer: unit.Quantity,
        settings: ESPSettings,
        directory: str = None,
    ) -> Tuple[unit.Quantity, unit.Quantity, unit.Quantity]:
        """Generate the electrostatic potential (ESP) on a grid defined by
        a provided set of settings.

        Parameters
        ----------
        oe_molecule
            The molecule to generate the ESP for.
        conformer
            The molecule conformer to generate the ESP of.
        settings
            The settings to use when generating the ESP.
        directory
            The directory to run the calculation in. If none is specified,
            a temporary directory will be created and used.

        Returns
        -------
            The grid [Angstrom] which the ESP  was generated on with
            shape=(n_grid_points, 3), the ESP [Hartree / e] with shape=(n_grid_points, 1)
            and the electric field [Hartree / (e . a0)] with shape=(n_grid_points, 3) at
            each grid point with for each conformer present on the specified molecule.
        """

        if directory is not None and len(directory) > 0:
            os.makedirs(directory, exist_ok=True)

        grid = GridGenerator.generate(oe_molecule, conformer, settings.grid_settings)
        esp, electric_field = cls._generate(
            oe_molecule, conformer, grid, settings, directory
        )

        return grid, esp, electric_field
