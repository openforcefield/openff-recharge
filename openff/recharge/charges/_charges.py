"""This module contains classes which will generate partial charges using a combination
of a cheaper QM method and a set of bond charge corrections.
"""
import copy
from typing import TYPE_CHECKING, List

import numpy
from openff.units import unit
from pydantic import BaseModel, Field
from typing_extensions import Literal

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule


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
    def _generate_omega_charges(
        cls,
        molecule: "Molecule",
        conformer: numpy.ndarray,
        settings: ChargeSettings,
    ) -> numpy.ndarray:

        oe_molecule = molecule.to_openeye()

        from openeye import oechem, oequacpac

        oe_molecule.DeleteConfs()
        oe_molecule.NewConf(oechem.OEFloatArray(conformer.flatten()))

        if settings.theory == "am1":
            assert oequacpac.OEAssignCharges(
                oe_molecule,
                oequacpac.OEAM1Charges(
                    optimize=settings.optimize, symmetrize=settings.symmetrize
                ),
            ), f"QUACPAC failed to generate {settings.theory} charges"
        elif settings.theory == "am1bcc":
            oequacpac.OEAssignCharges(
                oe_molecule,
                oequacpac.OEAM1BCCCharges(
                    optimize=settings.optimize, symmetrize=settings.symmetrize
                ),
            ), f"QUACPAC failed to generate {settings.theory} charges"
        else:
            raise NotImplementedError()

        atoms = {atom.GetIdx(): atom for atom in oe_molecule.GetAtoms()}
        return numpy.array(
            [
                [atoms[index].GetPartialCharge()]
                for index in range(oe_molecule.NumAtoms())
            ]
        )

    @classmethod
    def generate(
        cls,
        molecule: "Molecule",
        conformers: List[unit.Quantity],
        settings: ChargeSettings,
    ) -> numpy.ndarray:
        """Generates the averaged partial charges from multiple conformers of a
        specified molecule.

        Notes
        -----
        * Virtual sites will be assigned a partial charge of 0.0 e.

        Parameters
        ----------
        molecule
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

        from simtk import unit as simtk_unit

        # Make a copy of the molecule so as not to perturb the original.
        molecule = copy.deepcopy(molecule)

        conformer_charges = []

        for conformer in conformers:

            conformer = conformer[: molecule.n_atoms].m_as(unit.angstrom)

            if settings.theory == "am1" and settings.optimize and settings.symmetrize:
                charge_method = "am1-mulliken"
            elif (
                settings.theory == "am1bcc"
                and settings.optimize
                and settings.symmetrize
            ):
                charge_method = "am1-mulliken"
            elif (
                settings.theory == "am1bcc"
                and not settings.optimize
                and not settings.symmetrize
            ):
                charge_method = "am1bccnosymspt"
            else:
                charge_method = None

            if charge_method:
                molecule.assign_partial_charges(
                    charge_method, use_conformers=[conformer * simtk_unit.angstrom]
                )
                conformer_charges.append(
                    numpy.array(
                        [
                            *molecule.partial_charges.value_in_unit(
                                simtk_unit.elementary_charge
                            )
                        ]
                    )
                )
            else:
                conformer_charges.append(
                    cls._generate_omega_charges(molecule, conformer, settings)
                )

        charges = numpy.mean(conformer_charges, axis=0).reshape(-1, 1)
        n_vsites = len(conformers[0]) - molecule.n_atoms

        if n_vsites:
            charges = numpy.vstack((charges, numpy.zeros((n_vsites, 1))))

        return charges
