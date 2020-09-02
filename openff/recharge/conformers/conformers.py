"""A module for generating conformers for molecules."""
import logging
from typing import List, Optional

import numpy
from openeye import oechem, oeomega, oequacpac
from pydantic import BaseModel, Field
from typing_extensions import Literal

from openff.recharge.charges.exceptions import OEQuacpacError
from openff.recharge.conformers.exceptions import OEOmegaError
from openff.recharge.utilities.openeye import call_openeye, molecule_to_conformers

logger = logging.getLogger()


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
    def generate(
        cls,
        oe_molecule: oechem.OEMol,
        settings: ConformerSettings,
    ) -> List[numpy.ndarray]:
        """Generates a set of conformers for a given molecule.

        Notes
        -----
        * All operations are performed on a copy of the original molecule so that it
        will not be mutated by this function.

        Parameters
        ----------
        oe_molecule
            The molecule to generate conformers for.
        settings
            The settings to generate the conformers according to.
        """

        oe_molecule = oechem.OEMol(oe_molecule)

        # Enable dense sampling of conformers
        if settings.sampling_mode == "sparse":
            omega_options = oeomega.OEOmegaOptions(oeomega.OEOmegaSampling_Sparse)
        elif settings.sampling_mode == "dense":
            omega_options = oeomega.OEOmegaOptions(oeomega.OEOmegaSampling_Dense)
        else:
            raise NotImplementedError()

        omega = oeomega.OEOmega(omega_options)
        omega.SetIncludeInput(False)
        omega.SetCanonOrder(False)

        call_openeye(omega, oe_molecule, exception_type=OEOmegaError)

        if settings.method == "omega-elf10":

            # Select a subset of the OMEGA generated conformers using the ELF10 method.
            charge_engine = oequacpac.OEELFCharges(
                oequacpac.OEAM1BCCCharges(), 10, 2.0, True
            )

            call_openeye(
                oequacpac.OEAssignCharges,
                oe_molecule,
                charge_engine,
                exception_type=OEQuacpacError,
            )

        conformers = molecule_to_conformers(oe_molecule)

        if settings.max_conformers:
            conformers = conformers[0 : min(settings.max_conformers, len(conformers))]

        return conformers
