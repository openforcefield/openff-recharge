"""Generate partial charges for molecules from a collection of library parameters."""

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
import warnings

import numpy
from openff.units import unit
from openff.utilities import requires_package
from openff.toolkit.utils.exceptions import AtomMappingWarning
from openff.recharge._pydantic import BaseModel, Field, constr, validator

from openff.recharge.charges.exceptions import ChargeAssignmentError

if TYPE_CHECKING:
    from openff.toolkit import Molecule
    from openff.toolkit.typing.engines.smirnoff import LibraryChargeHandler


class LibraryChargeParameter(BaseModel):
    """An object which encodes the values of a set of charges applied to each atom in
    a molecule.
    """

    smiles: constr(min_length=1) = Field(
        ...,
        description="An indexed SMILES pattern that encodes and labels the **full** "
        "molecule that the charges should be applied to. Each index should correspond "
        "to a value in the ``value`` field. Multiple atoms can be assigned the same "
        "index in order to indicate that they should have equivalent charges.",
    )
    value: List[float] = Field(..., description="The values [e] of the charges.")

    provenance: Optional[Dict[str, Any]] = Field(
        None, description="Provenance information about this parameter."
    )

    @validator("smiles")
    def _validate_smiles(cls, value):
        try:
            from openff.toolkit import Molecule
        except ImportError:
            return value

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=AtomMappingWarning)
            molecule = Molecule.from_smiles(value, allow_undefined_stereo=True)

        atom_map = molecule.properties.get("atom_map", None)
        assert atom_map is not None, "SMILES pattern does not contain index map"

        assert len(atom_map) == molecule.n_atoms, "not all atoms contain a map index"
        assert {*atom_map.values()} == {
            i + 1 for i in range(len({*atom_map.values()}))
        }, "map indices must start from 1 and be continuous"

        return value

    @validator("value")
    def _validate_value(cls, value, values):
        if "smiles" not in values:
            return value

        try:
            from openff.toolkit import Molecule
        except ImportError:
            return value

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=AtomMappingWarning)
            molecule = Molecule.from_smiles(
                values["smiles"], allow_undefined_stereo=True
            )

        n_expected = len({*molecule.properties["atom_map"].values()})

        assert n_expected == len(value), (
            f"expected {n_expected} charges, " f"found {len(value)}"
        )

        total_charge = molecule.total_charge.m_as(unit.elementary_charge)

        sum_charge = sum(value[i - 1] for i in molecule.properties["atom_map"].values())

        assert numpy.isclose(total_charge, sum_charge), (
            f"sum of values {sum_charge} does not match "
            f"expected charge {total_charge}"
        )

        return value

    def copy_value_from_other(self, other: "LibraryChargeParameter"):
        """Assigns this parameters value from another library charge parameter.

        Notes
        -----
            * This function requires that both parameters should be applied to the exact
              same molecule.
            * If multiple values in the other parameter map to a single value in this
              parameter than the average value will be used.
        """

        from openff.toolkit import Molecule

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=AtomMappingWarning)
            self_molecule = Molecule.from_smiles(
                self.smiles, allow_undefined_stereo=True
            )
            other_molecule = Molecule.from_smiles(
                other.smiles, allow_undefined_stereo=True
            )

        are_isomorphic, self_to_other_atom_map = Molecule.are_isomorphic(
            self_molecule,
            other_molecule,
            return_atom_map=True,
            atom_stereochemistry_matching=False,
            bond_stereochemistry_matching=False,
        )
        assert are_isomorphic, "parameters are incompatible"

        applied_other_values = {
            i: other.value[other_molecule.properties["atom_map"][i] - 1]
            for i in range(other_molecule.n_atoms)
        }
        applied_self_values = {
            i: applied_other_values[self_to_other_atom_map[i]]
            for i in range(other_molecule.n_atoms)
        }

        self_values = defaultdict(list)

        for atom_index, charge_index in self_molecule.properties["atom_map"].items():
            self_values[charge_index - 1].append(applied_self_values[atom_index])

        self.value = [
            float(numpy.mean(self_values[i])) for i in range(len(self_values))
        ]

    def generate_constraint_matrix(
        self, trainable_indices: Optional[List[int]] = None
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Returns a matrix that when applied to ``value`` will yield the total charge
        on the molecule, as well as the value of the expected total charge.

        Parameters
        ----------
        trainable_indices
            An optional list of indices into ``value`` that describe which parameters
            are currently being trained.

        Returns
        -------
            The constraint matrix with shape=(1, ``n_values``) as well as an array
            containing the expected total charge (or the total charge minus the sum of
            any fixed charges if ``trainable_indices`` is specified``). ``n_values`` will
            be equal to the length of ``trainable_indices`` if provided, or otherwise to
            the length of ``self.value``.
        """
        from openff.toolkit import Molecule

        trainable_indices = (
            trainable_indices
            if trainable_indices is not None
            else list(range(len(self.value)))
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=AtomMappingWarning)
            molecule: Molecule = Molecule.from_smiles(
                self.smiles, allow_undefined_stereo=True
            )

        total_charge = molecule.total_charge.m_as(unit.elementary_charge)

        constraint_matrix = numpy.zeros((1, len(self.value)))

        for _atom_index, map_index in molecule.properties["atom_map"].items():
            constraint_matrix[0, map_index - 1] += 1

        for i, (value, n_times) in enumerate(
            zip(self.value, constraint_matrix.flatten())
        ):
            if i in trainable_indices:
                continue

            total_charge -= n_times * value

        return constraint_matrix[:, trainable_indices], numpy.array([[total_charge]])


class LibraryChargeCollection(BaseModel):
    """A library of charges sets that can be applied to molecules."""

    parameters: List[LibraryChargeParameter] = Field(
        ..., description="The library charges to apply."
    )

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
    @requires_package("openff.toolkit")
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
        return cls(
            parameters=[
                LibraryChargeParameter(
                    smiles=off_parameter.smirks,
                    value=[
                        charge.m_as(unit.elementary_charge)
                        for charge in off_parameter.charge
                    ],
                )
                for off_parameter in reversed(parameter_handler.parameters)
            ]
        )

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
    def _validate_assignment_matrix(
        cls,
        molecule: "Molecule",
        assignment_matrix: numpy.ndarray,
        charge_collection: LibraryChargeCollection,
    ):
        """Ensure that an assignment matrix yields sensible charges on a molecule."""
        total_charge = cls.apply_assignment_matrix(
            assignment_matrix, charge_collection
        ).sum()
        expected_charge = molecule.total_charge.m_as(unit.elementary_charge)

        if not numpy.isclose(total_charge, expected_charge):
            raise ChargeAssignmentError(
                f"The assigned charges yield a total charge ({total_charge:.4f}) that "
                f"does not match the expected value ({expected_charge:.4f})."
            )

    @classmethod
    def build_assignment_matrix(
        cls,
        molecule: "Molecule",
        charge_collection: LibraryChargeCollection,
    ) -> numpy.ndarray:
        """Generates a matrix that specifies which library charge have been
        applied to which atoms in the molecule.

        The matrix takes the form `[atom_index, charge_index]` where `atom_index` is the
        index of an atom in the molecule and `charge_index` is an index into a fully
        vectorized view of the charge collection.

        Parameters
        ----------
        molecule
            The molecule to assign the bond charge corrections to.
        charge_collection
            The library charge parameters that may be assigned.

        Returns
        -------
            The assignment matrix with shape=(n_atoms, n_library_charges)
            where `n_atoms` is the number of atoms in the molecule and
            `n_library_charges` is the **total** number of library charges.
        """

        from openff.toolkit import Molecule

        charge_index = 0

        n_total_charges = sum(
            len(parameter.value) for parameter in charge_collection.parameters
        )

        assignment_matrix = numpy.zeros((molecule.n_atoms, n_total_charges))

        for parameter in charge_collection.parameters:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=AtomMappingWarning)
                smiles_molecule: Molecule = Molecule.from_smiles(
                    parameter.smiles, allow_undefined_stereo=True
                )

            are_isomorphic, atom_map = Molecule.are_isomorphic(
                molecule, smiles_molecule, return_atom_map=True
            )

            if not are_isomorphic:
                charge_index += len(parameter.value)
                continue

            value_map = {
                i: smiles_molecule.properties["atom_map"][atom_map[i]] - 1
                for i in range(smiles_molecule.n_atoms)
            }

            for i in range(molecule.n_atoms):
                assignment_matrix[i, charge_index + value_map[i]] = 1

            cls._validate_assignment_matrix(
                molecule, assignment_matrix, charge_collection
            )
            return assignment_matrix

        raise ChargeAssignmentError(
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
        molecule: "Molecule",
        charge_collection: LibraryChargeCollection,
    ) -> numpy.ndarray:
        """Generate a set of charge increments for a molecule.

        Parameters
        ----------
        molecule
            The molecule to generate the bond-charge corrections for.
        charge_collection
            The set of library charge parameters which may be assigned.

        Returns
        -------
            The library charges [e] that should be applied to the molecule with
            shape=(n_atoms, 1).
        """

        return cls.apply_assignment_matrix(
            cls.build_assignment_matrix(molecule, charge_collection),
            charge_collection,
        )
