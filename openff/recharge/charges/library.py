from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy
from openff.utilities import requires_package
from pydantic import BaseModel, Field, constr

from openff.recharge.charges.exceptions import UnableToAssignChargeError

if TYPE_CHECKING:
    from openeye import oechem
    from openff.toolkit.typing.engines.smirnoff import LibraryChargeHandler


class LibraryChargeParameter(BaseModel):
    """An object which encodes the values of a set of charges applied to each atom in
    a molecule.
    """

    smiles: constr(min_length=1) = Field(
        ...,
        description="An indexed SMILES pattern that encodes and labels the **full** "
        "molecule that the charges should be applied to.",
    )
    value: List[float] = Field(..., description="The values [e] of the charges.")

    provenance: Optional[Dict[str, Any]] = Field(
        None, description="Provenance information about this parameter."
    )


class LibraryChargeCollection(BaseModel):
    """A library of charges sets that can be applied to molecules."""

    parameters: List[LibraryChargeParameter] = Field(
        ..., description="The library charges to apply."
    )

    @requires_package("openff.toolkit")
    @requires_package("simtk")
    def to_smirnoff(self) -> "LibraryChargeHandler":
        """Converts this collection of library charge parameters to
        a SMIRNOFF library charge parameter handler.

        Returns
        -------
            The constructed parameter handler.
        """
        from openff.toolkit.typing.engines.smirnoff.parameters import (
            LibraryChargeHandler,
        )
        from simtk import unit

        # noinspection PyTypeChecker
        parameter_handler = LibraryChargeHandler(version="0.3")

        for parameter in reversed(self.parameters):

            parameter_handler.add_parameter(
                {
                    "smirks": parameter.smiles,
                    "charge": parameter.value * unit.elementary_charge,
                }
            )

        return parameter_handler

    @classmethod
    @requires_package("simtk")
    def from_smirnoff(
        cls, parameter_handler: "LibraryChargeHandler"
    ) -> "LibraryChargeCollection":
        """Attempts to convert a SMIRNOFF library charge parameter handler
        to a library charge parameter collection.

        Parameters
        ----------
        parameter_handler
            The parameter handler to convert.

        Returns
        -------
            The converted bond charge correction collection.
        """
        from simtk import unit

        return cls(
            parameters=[
                LibraryChargeParameter(
                    smiles=off_parameter.smirks,
                    value=[
                        charge.value_in_unit(unit.elementary_charge)
                        for charge in off_parameter.charge
                    ],
                )
                for off_parameter in reversed(parameter_handler.parameters)
            ]
        )  # [py/call-to-non-callable]

    def vectorize(self, keys: List[Tuple[str, Tuple[int, ...]]]) -> numpy.ndarray:
        """Returns a flat vector of the charge increment values associated with each
        SMILES pattern in a specified list.

        Parameters
        ----------
        keys
            A list of tuples of the form ``(smiles, idx)`` that define those parameters
            in the ``charge_collection`` that should be trained.

            Here ``idx`` is an index into the ``value`` field of the parameter uniquely
            identified by the ``smiles`` key.

        Returns
        -------
            A flat vector of charge increments with shape=(n_smiles_i, 1) where
            `n_smiles_i` corresponds to the number of tagged atoms in SMILES pattern
            `i`.
        """

        parameters: Dict[Tuple[str, int], LibraryChargeParameter] = {
            (parameter.smiles, i): parameter.value[i]
            for parameter in self.parameters
            for i in range(len(parameter.value))
        }
        return numpy.array(
            [[parameters[(smiles, i)]] for smiles, indices in keys for i in indices]
        )


class LibraryChargeGenerator:
    """A class for generating the library charges which should be applied to a
    molecule.
    """

    @classmethod
    def build_assignment_matrix(
        cls,
        oe_molecule: "oechem.OEMol",
        charge_collection: LibraryChargeCollection,
    ) -> numpy.ndarray:
        """Generates a matrix that specifies which library charge have been
        applied to which atoms in the molecule.

        The matrix takes the form `[atom_index, charge_index]` where `atom_index` is the
        index of an atom in the molecule and `charge_index` is an index into a fully
        vectorized view of the charge collection.

        Parameters
        ----------
        oe_molecule
            The molecule to assign the bond charge corrections to.
        charge_collection
            The library charge parameters that may be assigned.

        Returns
        -------
            The assignment matrix with shape=(n_atoms, n_library_charges)
            where `n_atoms` is the number of atoms in the molecule and
            `n_library_charges` is the **total** number of library charges.
        """

        from openff.toolkit.topology import Molecule

        molecule: Molecule = Molecule.from_openeye(oe_molecule)
        charge_index = 0

        n_total_charges = sum(
            len(parameter.value) for parameter in charge_collection.parameters
        )

        assignment_matrix = numpy.zeros((molecule.n_atoms, n_total_charges))

        for parameter in charge_collection.parameters:

            smiles_molecule: Molecule = Molecule.from_mapped_smiles(parameter.smiles)

            are_isomorphic, atom_map = Molecule.are_isomorphic(
                molecule, smiles_molecule, return_atom_map=True
            )

            if not are_isomorphic:
                charge_index += len(parameter.value)
                continue

            for i in range(molecule.n_atoms):
                assignment_matrix[i, charge_index + atom_map[i]] = 1

            return assignment_matrix

        raise UnableToAssignChargeError(
            f"Atoms {list(range(molecule.n_atoms))} could not be assigned a library "
            f"charge."
        )

    @classmethod
    def apply_assignment_matrix(
        cls,
        assignment_matrix: numpy.ndarray,
        charge_collection: LibraryChargeCollection,
    ) -> numpy.ndarray:
        """Applies an assignment matrix to a list of bond charge corrections
        yield the final bond-charge corrections for a molecule.

        Parameters
        ----------
        assignment_matrix
            The library charge assignment matrix constructed using
            ``build_assignment_matrix`` which describes how the library charges should
            be applied. This should have shape=(n_atoms, n_library_charges)
        charge_collection
            The library charge parameters which may be assigned.

        Returns
        -------
            The library charges with shape=(n_atoms, 1).
        """

        all_values = numpy.array(
            [
                [charge]
                for parameter in charge_collection.parameters
                for charge in parameter.value
            ]
        )

        return assignment_matrix @ all_values

    @classmethod
    def generate(
        cls,
        oe_molecule: "oechem.OEMol",
        charge_collection: LibraryChargeCollection,
    ) -> numpy.ndarray:
        """Generate a set of charge increments for a molecule.

        Parameters
        ----------
        oe_molecule
            The molecule to generate the bond-charge corrections for.
        charge_collection
            The set of library charge parameters which may be assigned.

        Returns
        -------
            The library charges [e] that should be applied to the molecule with
            shape=(n_atoms, 1).
        """

        return cls.apply_assignment_matrix(
            cls.build_assignment_matrix(oe_molecule, charge_collection),
            charge_collection,
        )
