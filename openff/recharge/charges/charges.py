"""This module contains classes which will generate partial charges using a combination
of a cheaper QM method and a set of bond charge corrections.
"""
from typing import TYPE_CHECKING, List

import numpy
from pydantic import BaseModel, Field
from typing_extensions import Literal

from openff.recharge.charges.exceptions import OEQuacpacError
from openff.recharge.utilities.openeye import (
    call_openeye,
    import_oechem,
    import_oequacpac,
)

if TYPE_CHECKING:
    from openeye import oechem


class ChargeSettings(BaseModel):
    """The settings to use when assigning partial charges from
    quantum chemical calculations.
    """

    theory: Literal["am1", "am1bcc"] = Field(
        "am1", description="The level of theory to use when computing the charges."
    )

    symmetrize: bool = Field(
        True,
        description="Whether the partial charges should be made equal for bond-"
        "topology equivalent atoms.",
    )
    optimize: bool = Field(
        True,
        description="Whether to optimize the input conformer during the charge"
        "calculation.",
    )


class ChargeGenerator:
    """A class which will compute the partial charges of a molecule
    from a quantum chemical calculation."""

    @classmethod
    def generate(
        cls,
        oe_molecule: "oechem.OEMol",
        conformers: List[numpy.ndarray],
        settings: ChargeSettings,
    ) -> numpy.ndarray:
        """Generates the averaged partial charges from multiple conformers of a specified
        molecule.

        Notes
        -----
        * Virtual sites will be assigned a partial charge of 0.0 e.

        Parameters
        ----------
        oe_molecule
            The molecule to compute the partial charges for.
        conformers
            The conformers to use in the partial charge calculations
            where each conformer should have a shape=(n_atoms + n_vsites, 3).
        settings
            The settings to use in the charge generation.

        Returns
        -------
            The computed partial charges.
        """

        oechem = import_oechem()
        oequacpac = import_oequacpac()

        # Make a copy of the molecule so as not to perturb the original.
        oe_molecule = oechem.OEMol(oe_molecule)

        # Apply the conformers to the molecule
        oe_molecule.DeleteConfs()

        for conformer in conformers:
            conformer = conformer[: oe_molecule.NumAtoms()]
            oe_molecule.NewConf(oechem.OEFloatArray(conformer.flatten()))

        # Compute the partial charges.
        if settings.theory == "am1":
            call_openeye(
                oequacpac.OEAssignCharges,
                oe_molecule,
                oequacpac.OEAM1Charges(
                    optimize=settings.optimize, symmetrize=settings.symmetrize
                ),
                exception_type=OEQuacpacError,
            )
        elif settings.theory == "am1bcc":
            call_openeye(
                oequacpac.OEAssignCharges,
                oe_molecule,
                oequacpac.OEAM1BCCCharges(
                    optimize=settings.optimize, symmetrize=settings.symmetrize
                ),
                exception_type=OEQuacpacError,
            )
        else:
            raise NotImplementedError()

        # Retrieve the partial charges from the molecule.
        atoms = {atom.GetIdx(): atom for atom in oe_molecule.GetAtoms()}
        charges = numpy.array(
            [
                [atoms[index].GetPartialCharge()]
                for index in range(oe_molecule.NumAtoms())
            ]
        )

        n_vsites = len(conformers[0]) - oe_molecule.NumAtoms()

        if n_vsites:
            charges = numpy.vstack((charges, numpy.zeros((n_vsites, 1))))

        return charges
