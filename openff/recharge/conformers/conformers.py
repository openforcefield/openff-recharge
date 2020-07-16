"""A module for generating conformers with a focus on generating extended conformers
(i.e. conformers where strongly polar groups are not interacting)."""
import abc
import logging
from typing import List, Optional

import numpy
from openeye import oechem, oeomega, oequacpac

from openff.recharge.charges.exceptions import OEQuacpacError
from openff.recharge.conformers.exceptions import OEOmegaError
from openff.recharge.utilities.openeye import call_openeye, molecule_to_conformers

logger = logging.getLogger()


class ConformerGenerator(abc.ABC):
    """A base class for classes which will generate charges for a
    given molecule"""

    @classmethod
    @abc.abstractmethod
    def generate(cls, oe_molecule: oechem.OEMol) -> oechem.OEMol:
        """Generates a set of conformers for a given molecule..

        Notes
        -----
        * All operations are performed on a copy of the original molecule so that it
        will not be mutated by this function. The generated conformers will be
        available on the return molecule instead.

        * Any existing conformers on the given molecule will be discarded.

        Parameters
        ----------
        oe_molecule
            The molecule to generate conformers for.
        """


class OmegaELF10(ConformerGenerator):
    """A conformer generator which initial generates a diverse set of conformers
     using the OpenEye Omega toolkit, and then prunes those conformers based
     upon the OpenEYE ELF10 implementation."""

    @classmethod
    def generate(
        cls, oe_molecule: oechem.OEMol, max_conformers: Optional[int] = 5,
    ) -> List[numpy.ndarray]:
        """
        Parameters
        ----------
        oe_molecule
            The molecule to generate conformers for.
        max_conformers
            The maximum number of conformers to be generated.
        """

        oe_molecule = oechem.OEMol(oe_molecule)

        # Enable dense sampling of conformers
        omega_options = oeomega.OEOmegaOptions(oeomega.OEOmegaSampling_Dense)

        omega = oeomega.OEOmega(omega_options)
        omega.SetIncludeInput(False)
        omega.SetCanonOrder(False)

        call_openeye(omega, oe_molecule, exception_type=OEOmegaError)

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

        if max_conformers:
            conformers = conformers[0 : min(max_conformers, len(conformers))]

        return conformers
