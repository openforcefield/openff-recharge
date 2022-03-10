"""A module for generating conformers for molecules."""
import logging
from typing import TYPE_CHECKING, List, Optional

import numpy
from openff.units import unit
from pydantic import BaseModel, Field
from typing_extensions import Literal

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule

_logger = logging.getLogger()


class ConformerSettings(BaseModel):
    """The settings to use when generating conformers for a
    particular molecule.
    """

    method: Literal["omega", "omega-elf10"] = Field(
        "omega-elf10", description="The method to use to generate the conformers."
    )
    sampling_mode: Literal["sparse", "dense"] = Field(
        "dense", description="The mode in which to generate the conformers."
    )

    max_conformers: Optional[int] = Field(
        5, description="The maximum number of conformers to generate."
    )


class ConformerGenerator:
    """A class to generate a set of conformers for a molecule according to
    a specified set of settings.
    """

    @classmethod
    def _generate_omega_conformers(
        cls,
        molecule: "Molecule",
        settings: ConformerSettings,
    ) -> List[unit.Quantity]:

        oe_molecule = molecule.to_openeye()

        from openeye import oeomega, oequacpac

        if settings.sampling_mode == "sparse":
            omega_options = oeomega.OEOmegaOptions(oeomega.OEOmegaSampling_Sparse)
        elif settings.sampling_mode == "dense":
            omega_options = oeomega.OEOmegaOptions(oeomega.OEOmegaSampling_Dense)
        else:
            raise NotImplementedError()

        omega = oeomega.OEOmega(omega_options)
        omega.SetIncludeInput(False)
        omega.SetCanonOrder(False)

        assert omega(oe_molecule), "OMEGA failed to execute"

        if settings.method == "omega-elf10":

            # Select a subset of the OMEGA generated conformers using the ELF10 method.
            oe_elf_options = oequacpac.OEELFOptions()
            oe_elf_options.SetElfLimit(10)
            oe_elf_options.SetPercent(2.0)

            oe_elf = oequacpac.OEELF(oe_elf_options)

            assert oe_elf.Select(oe_molecule), "ELF10 conformer selection failed"

        conformers = []

        for oe_conformer in oe_molecule.GetConfs():

            conformer = numpy.zeros((oe_molecule.NumAtoms(), 3))

            for atom_index, coordinates in oe_conformer.GetCoords().items():
                conformer[atom_index, :] = coordinates

            conformers.append(conformer * unit.angstrom)

        return conformers

    @classmethod
    def generate(
        cls,
        molecule: "Molecule",
        settings: ConformerSettings,
    ) -> List[unit.Quantity]:
        """Generates a set of conformers for a given molecule.

        Notes
        -----
        * All operations are performed on a copy of the original molecule so that it
        will not be mutated by this function.

        Parameters
        ----------
        molecule
            The molecule to generate conformers for.
        settings
            The settings to generate the conformers according to.
        """

        if "omega" in settings.method:
            conformers = cls._generate_omega_conformers(molecule, settings)
        else:
            raise NotImplementedError()

        if settings.max_conformers:
            conformers = conformers[0 : min(settings.max_conformers, len(conformers))]

        return conformers
