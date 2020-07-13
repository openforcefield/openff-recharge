import abc
import os
from typing import Tuple

import numpy
from openeye.oechem import OEMol
from pydantic import BaseModel, Field

from openff.recharge.grids import GridGenerator, GridSettings


class ESPSettings(BaseModel):
    """A class which contains the settings to use in an ESP calculation.
    """

    basis: str = Field(
        "6-31g*", description="The basis set to use in the ESP calculation."
    )
    method: str = Field("hf", description="The method to use in the ESP calculation.")

    grid_settings: GridSettings = Field(
        ...,
        description="The settings to use when generating the grid to generate the "
        "electrostatic potential on.",
    )


class ESPGenerator(abc.ABC):
    """A base class for classes which are able to generate the electrostatic
    potential of a molecule on a specified grid.
    """

    @classmethod
    @abc.abstractmethod
    def _generate(
        cls,
        oe_molecule: OEMol,
        conformer: numpy.ndarray,
        grid: numpy.ndarray,
        settings: ESPSettings,
        directory: str = None,
    ) -> numpy.ndarray:
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
            The ESP [Hartree / e] at each grid point with shape=(n_grid_points, 1).
        """
        raise NotImplementedError

    @classmethod
    def generate(
        cls,
        oe_molecule: OEMol,
        conformer: numpy.ndarray,
        settings: ESPSettings,
        directory: str = None,
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
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
            shape=(n_grid_points, 3) and the ESP [Hartree / e] at each grid point with
            shape=(n_grid_points, 1) for each conformer present on the specified
            molecule.
        """

        if directory is not None and len(directory) > 0:
            os.makedirs(directory, exist_ok=True)

        grid = GridGenerator.generate(oe_molecule, conformer, settings.grid_settings)
        esp = cls._generate(oe_molecule, conformer, grid, settings, directory)

        return grid, esp
