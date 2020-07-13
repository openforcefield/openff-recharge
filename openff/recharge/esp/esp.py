import abc
from typing import List, Tuple

import numpy
from openeye.oechem import OEMol
from pydantic import BaseModel, Field

from openff.recharge.grids import GridGenerator, GridSettings
from openff.recharge.utilities.exceptions import MissingConformersError
from openff.recharge.utilities.openeye import molecule_to_conformers


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

    @classmethod
    def generate(
        cls, oe_molecule: OEMol, settings: ESPSettings, directory: str = None
    ) -> List[Tuple[numpy.ndarray, numpy.ndarray]]:
        """Generate the electrostatic potential (ESP) on a grid defined by
        a provided set of settings.

        Parameters
        ----------
        oe_molecule
            The molecule to generate the ESP for.
        settings
            The settings to use when generating the ESP.
        directory
            The directory to run the calculation in. If none is specified,
            a temporary directory will be created and used.

        Returns
        -------
            The grid which the ESP [Hartree / e] was generated on with
            shape=(n_grid_points, 3) and the ESP  at each grid point with
            shape=(n_grid_points, 1) for each conformer present on the specified
            molecule.
        """

        if oe_molecule.NumConfs() == 0:
            raise MissingConformersError()

        conformers = molecule_to_conformers(oe_molecule)

        values = []

        for conformer in conformers:

            grid = GridGenerator.generate(
                oe_molecule, conformer, settings.grid_settings
            )
            esp = cls._generate(oe_molecule, conformer, grid, settings, directory)

            values.append((grid, esp))

        return values
