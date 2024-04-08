import functools
import warnings
from typing import TYPE_CHECKING

import numpy
from openff.toolkit.utils.exceptions import AtomMappingWarning
from openff.units import unit

from openff.recharge.charges.library import (
    LibraryChargeCollection,
    LibraryChargeParameter,
)
from openff.recharge.charges.resp.solvers import IterativeSolver, RESPNonLinearSolver
from openff.recharge.esp.storage import MoleculeESPRecord
from openff.recharge.optimize import ESPObjective, ESPObjectiveTerm
from openff.recharge.utilities.toolkits import (
    get_atom_symmetries,
    molecule_to_tagged_smiles,
)

if TYPE_CHECKING:
    from openff.toolkit import Molecule


def _generate_dummy_values(smiles: str) -> list[float]:
    """A convenience method for generating a list of dummy values for a
    ``LibraryChargeParameter`` that sums to the correct total charge.
    """

    from openff.toolkit import Molecule

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AtomMappingWarning)
        molecule: Molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)

    total_charge = molecule.total_charge.m_as(unit.elementary_charge)
    per_atom_charge = total_charge / molecule.n_atoms

    n_values = len(set(molecule.properties["atom_map"].values()))
    return [per_atom_charge] * n_values


def molecule_to_resp_library_charge(
    molecule: "Molecule",
    equivalize_within_methyl_carbons: bool,
    equivalize_within_methyl_hydrogens: bool,
    equivalize_within_other_heavy_atoms: bool,
    equivalize_within_other_hydrogen_atoms: bool,
) -> LibraryChargeParameter:
    """Creates a library charge parameter from a molecule where each atom as been
    assigned a map index that represents which equivalence group the atom is in.

    Notes
    -----
    * The ``value`` of the returned parameter will contain a set of dummy values that
      sum to the correct total charge.
    * The ``provenance`` dictionary contains the indices into the ``value`` array of
      any charges applied to methyl(ene) carbons / hydrogens and other heavy atoms
      / hydrogens
    * Topologically symmetry is detected using the ``get_atom_symmetries`` utility.

    Parameters
    ----------
    molecule
        The molecule to create the SMILES pattern from.
    equivalize_within_methyl_carbons
        Whether all topologically symmetric methyl(ene) carbons (i.e. those matched by
        '[#6X4H3,#6H4,#6X4H2:1]') in **the same** conformer should be assigned an
        equivalent charge.
    equivalize_within_methyl_hydrogens
        Whether all topologically symmetric methyl(ene) hydrogens (i.e. those attached to
        a methyl(ene) carbon) in **the same** conformer should be assigned an
        equivalent charge.
    equivalize_within_other_heavy_atoms
        Whether all topologically symmetric heavy atoms that are not methyl(ene) carbons
        in **the same** conformer should be assigned an equivalent charge.
    equivalize_within_other_hydrogen_atoms
        Whether all topologically symmetric hydrogens that are not methyl(ene) hydrogens
        in **the same** conformer should be assigned an equivalent charge.

    Returns
    -------
        The library charge with atoms assigned their correct symmetry groups.
    """

    atom_symmetries = get_atom_symmetries(molecule)
    max_symmetry_group = max(atom_symmetries) + 1

    methyl_carbons = [
        index
        for index, in molecule.chemical_environment_matches("[#6X4H3,#6H4,#6X4H2:1]")
    ]
    methyl_hydrogens = [
        atom.molecule_atom_index
        for index in methyl_carbons
        for atom in molecule.atoms[index].bonded_atoms
        if atom.atomic_number == 1
    ]
    other_heavy_atoms = [
        i
        for i, atom in enumerate(molecule.atoms)
        if atom.atomic_number != 1 and i not in methyl_carbons
    ]
    other_hydrogen_atoms = [
        i
        for i, atom in enumerate(molecule.atoms)
        if atom.atomic_number == 1 and i not in methyl_hydrogens
    ]

    atoms_not_to_equivalize = (
        ([] if equivalize_within_methyl_carbons else methyl_carbons)
        + ([] if equivalize_within_methyl_hydrogens else methyl_hydrogens)
        + ([] if equivalize_within_other_heavy_atoms else other_heavy_atoms)
        + ([] if equivalize_within_other_hydrogen_atoms else other_hydrogen_atoms)
    )

    for index in atoms_not_to_equivalize:
        atom_symmetries[index] = max_symmetry_group
        max_symmetry_group += 1

    symmetry_groups = sorted(set(atom_symmetries))

    atom_indices = [symmetry_groups.index(group) + 1 for group in atom_symmetries]
    tagged_smiles = molecule_to_tagged_smiles(molecule, atom_indices)

    return LibraryChargeParameter(
        smiles=tagged_smiles,
        value=_generate_dummy_values(tagged_smiles),
        provenance={
            "methyl-carbon-indices": sorted(
                {atom_indices[i] - 1 for i in methyl_carbons}
            ),
            "methyl-hydrogen-indices": sorted(
                {atom_indices[i] - 1 for i in methyl_hydrogens}
            ),
            "other-heavy-indices": sorted(
                {atom_indices[i] - 1 for i in other_heavy_atoms}
            ),
            "other-hydrogen-indices": sorted(
                {atom_indices[i] - 1 for i in other_hydrogen_atoms}
            ),
        },
    )


def _deduplicate_constraints(
    constraint_matrix: numpy.ndarray, constraint_values: numpy.ndarray
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Removes duplicate rows from a constraint matrix and corresponding values are.

    Parameters
    ----------
    constraint_matrix
        The constraint matrix to de-duplicate with shape=(n_constraints, n_values).
    constraint_values
        The expected values of the constraints with shape=(n_constraints, 1)

    Returns
    -------
        A de-duplicated representation of the constraint matrix and values.
    """

    found_constraints = set()

    deduplicated_rows = []
    deduplicated_values = []

    for constraint_row, constraint_value in zip(constraint_matrix, constraint_values):
        constraint_row = tuple(int(i) for i in constraint_row)

        if constraint_row in found_constraints:
            continue

        deduplicated_rows.append(constraint_row)

        assert constraint_value.shape == (1,)
        deduplicated_values.append([float(constraint_value[0])])

        found_constraints.add(constraint_row)

    return numpy.array(deduplicated_rows), numpy.array(deduplicated_values)


def generate_resp_systems_of_equations(
    charge_parameter: LibraryChargeParameter,
    qc_data_records: list[MoleculeESPRecord],
    equivalize_between_methyl_carbons: bool,
    equivalize_between_methyl_hydrogens: bool,
    equivalize_between_other_heavy_atoms: bool,
    equivalize_between_other_hydrogen_atoms: bool,
    fix_methyl_carbons: bool,
    fix_methyl_hydrogens: bool,
    fix_other_heavy_atoms: bool,
    fix_other_hydrogen_atoms: bool,
) -> tuple[
    numpy.ndarray,
    numpy.ndarray,
    numpy.ndarray,
    numpy.ndarray,
    list[int],
    dict[int, int],
]:
    """Generates the matrices that encode the systems of equations that form the RESP
    loss function.

    Parameters
    ----------
    charge_parameter
        The parameter that will be fit to ESP data.
    qc_data_records
        The records containing the reference QC ESP data.
    equivalize_between_methyl_carbons
        Whether all topologically symmetric methyl(ene) carbons (i.e. those matched by
        '[#6X4H3,#6H4,#6X4H2:1]') in **different** conformers should be assigned an
        equivalent charge.
    equivalize_between_methyl_hydrogens
        Whether all topologically symmetric methyl(ene) hydrogens (i.e. those attached to
        a methyl(ene) carbon) in **different** conformers should be assigned an
        equivalent charge.
    equivalize_between_other_heavy_atoms
        Whether all topologically symmetric heavy atoms that are not methyl(ene) carbons
        in **different** conformers should be assigned an equivalent charge.
    equivalize_between_other_hydrogen_atoms
        Whether all topologically symmetric hydrogens that are not methyl(ene) hydrogens
        should be assigned an equivalent charge.
    fix_methyl_carbons
        Whether to fix the charges on methyl(ene) carbons.
    fix_methyl_hydrogens
        Whether to fix the charges on methyl(ene) hydrogens.
    fix_other_heavy_atoms
        Whether to fix the charges on heavy atoms that are not methyl(ene) carbons.
    fix_other_hydrogen_atoms
        Whether to fix the charges on heavy atoms that are not methyl(ene) hydrogens.

    Returns
    -------
        A tuple of:

        * a design matrix with shape=(n_grid_points, n_not_fixed_charges)
        * a vector of reference ESP values with shape=(n_grid_points, 1)
        * a constraint matrix with shape=(n_constraints, n_not_fixed_charges)
        * a vector of values the constraints should equal with
          shape=(n_constraints, 1)
        * the indices of the `n_trainable_charges` that should be restrained
        * a dictionary that maps the indices of the not-fixed charges back to their
          original indices in the ``charge_parameter.value`` list.
    """

    methyl_carbons = charge_parameter.provenance["methyl-carbon-indices"]
    methyl_hydrogens = charge_parameter.provenance["methyl-hydrogen-indices"]
    other_heavy_atoms = charge_parameter.provenance["other-heavy-indices"]
    other_hydrogen_atoms = charge_parameter.provenance["other-hydrogen-indices"]

    charges_to_train = sorted(
        ([] if fix_methyl_carbons else methyl_carbons)
        + ([] if fix_methyl_hydrogens else methyl_hydrogens)
        + ([] if fix_other_heavy_atoms else other_heavy_atoms)
        + ([] if fix_other_hydrogen_atoms else other_hydrogen_atoms)
    )

    objective_terms = list(
        ESPObjective.compute_objective_terms(
            qc_data_records,
            charge_collection=LibraryChargeCollection(parameters=[charge_parameter]),
            charge_parameter_keys=[(charge_parameter.smiles, tuple(charges_to_train))],
        )
    )

    (
        constraint_matrix,
        constraint_vector,
    ) = charge_parameter.generate_constraint_matrix(charges_to_train)

    # We need to handle the special case of not equivalizing certain charges between
    # conformers. This is done by augmenting the design matrix to accommodate a set of
    # 'dummy' charges per conformer which are not equivalized between conformers.
    charges_not_to_equivalize = (
        ([] if equivalize_between_methyl_carbons else methyl_carbons)
        + ([] if equivalize_between_methyl_hydrogens else methyl_hydrogens)
        + ([] if equivalize_between_other_heavy_atoms else other_heavy_atoms)
        + ([] if equivalize_between_other_hydrogen_atoms else other_hydrogen_atoms)
    )
    charges_not_to_equivalize = [
        charges_to_train.index(i)
        for i in charges_not_to_equivalize
        if i in charges_to_train
    ]

    n_dummy_charges = (len(objective_terms) - 1) * len(charges_not_to_equivalize)

    conformer_constraint_matrices = []
    conformer_constraint_vectors = [constraint_vector] * len(objective_terms)

    for conformer_index, objective_term in enumerate(objective_terms):
        design_matrix = objective_term.atom_charge_design_matrix.copy()

        padded_design_matrix = numpy.pad(
            objective_term.atom_charge_design_matrix,
            pad_width=[(0, 0), (0, n_dummy_charges)],
            mode="constant",
        )
        padded_constraint = numpy.pad(
            constraint_matrix,
            pad_width=[(0, 0), (0, n_dummy_charges)],
            mode="constant",
        )

        if conformer_index > 0:
            old_order = list(range(padded_constraint.shape[1]))
            new_order = [*old_order]

            for i, charge_index in enumerate(charges_not_to_equivalize):
                new_order[
                    design_matrix.shape[1]
                    + (conformer_index - 1) * len(charges_not_to_equivalize)
                    + i
                ] = charge_index

            padded_design_matrix[:, old_order] = padded_design_matrix[:, new_order]
            padded_design_matrix[:, charges_not_to_equivalize] = 0.0
            padded_constraint[:, old_order] = padded_constraint[:, new_order]
            padded_constraint[:, charges_not_to_equivalize] = 0.0

        conformer_constraint_matrices.append(padded_constraint)
        objective_term.atom_charge_design_matrix = padded_design_matrix

    combined_term = ESPObjectiveTerm.combine(*objective_terms)

    restraint_indices = [
        charges_to_train.index(i)
        for i in (methyl_carbons + other_heavy_atoms)
        if i in charges_to_train
    ]
    charge_array_to_value = {i: index for i, index in enumerate(charges_to_train)}

    combined_constraint_matrix, combined_constraint_values = _deduplicate_constraints(
        numpy.vstack(conformer_constraint_matrices),
        numpy.vstack(conformer_constraint_vectors),
    )

    return (
        combined_term.atom_charge_design_matrix,
        combined_term.reference_values,
        combined_constraint_matrix,
        combined_constraint_values,
        restraint_indices,
        charge_array_to_value,
    )


@functools.lru_cache(10000)
def _canonicalize_smiles(smiles: str) -> str:
    """Attempts to canonicalize a SMILES pattern, stripping any map indices in the
    process
    """

    from openff.toolkit import Molecule

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AtomMappingWarning)
        return Molecule.from_smiles(smiles, allow_undefined_stereo=True).to_smiles(
            mapped=False
        )


def generate_resp_charge_parameter(
    qc_data_records: list[MoleculeESPRecord], solver: RESPNonLinearSolver | None
) -> LibraryChargeParameter:
    """Generates a set of RESP charges for a molecule in multiple conformers.

    Notes
    -----
    * Methyl(ene) carbons are detected as any carbon matched by '[#6X4H3,#6H4,#6X4H2:1]',
      methyl(ene) hydrogens are any hydrogen attached to a methyl(ene) carbon, otherwise
      the atom is treated as any other heavy atom / hydrogen.
    * All heavy atom charge and non-methyl(ene) hydrogen charge will be equivalized in
      stage 1 of the fit both within and between multiple conformers, while methyl(ene)
      hydrogen charge will not be either within or between multiple conformers.
    * All methyl(ene) charges (both carbon and hydrogen) will be equivalized both within
      and between multiple conformers in stage 2 of the fit.
    * All atom charge will be free to vary during stage 1 of the fit, while only
      methyl(ene) charges (both carbon and hydrogen) will be free to vary during stage 2.
    * A value of b=0.1 is used for the hyperbolic restraints applied to all heavy atom
      charges in both stages, while a value of a=0.0005 and 0.001 will be used for
      stages 1 and 2 respectively.

    Parameters
    ----------
    qc_data_records
        The computed ESP for a molecule in different conformers.
    solver
        The solver to use when finding the charges that minimize the RESP loss function.
        By default, the iterative solver described in the RESP publications is used.

    Returns
    -------
        The RESP charges generated for the molecule.
    """

    from openff.toolkit import Molecule

    solver = IterativeSolver() if solver is None else solver

    unique_smiles = {
        _canonicalize_smiles(record.tagged_smiles) for record in qc_data_records
    }
    assert (
        len(unique_smiles) == 1
    ), "all QC records must be generated for the same molecule"

    molecule = Molecule.from_smiles(
        next(iter(unique_smiles)), allow_undefined_stereo=True
    )

    b = 0.1

    ###################################################################################
    #                                       STAGE 1                                   #
    #                                                                                 #
    # * METHYL(ENE) C   : constrain equiv within=[✓] between=[✓] conformers fixed=[x] #
    # * METHYL(ENE) H   : constrain equiv within=[x] between=[x] conformers fixed=[x] #
    # * HEAVY ATOMS     : constrain equiv within=[✓] between=[✓] conformers fixed=[x] #
    # * HYDROGEN        : constrain equiv within=[✓] between=[✓] conformers fixed=[x] #
    #                                                                                 #
    ###################################################################################

    a = 0.0005

    resp_parameter_1 = molecule_to_resp_library_charge(
        molecule,
        equivalize_within_methyl_hydrogens=False,
        equivalize_within_methyl_carbons=True,
        equivalize_within_other_hydrogen_atoms=True,
        equivalize_within_other_heavy_atoms=True,
    )
    (
        design_matrix_1,
        reference_esp_1,
        constraint_matrix_1,
        constraint_vector_1,
        restraint_indices_1,
        _,
    ) = generate_resp_systems_of_equations(
        resp_parameter_1,
        qc_data_records,
        equivalize_between_methyl_hydrogens=False,
        equivalize_between_methyl_carbons=True,
        equivalize_between_other_hydrogen_atoms=True,
        equivalize_between_other_heavy_atoms=True,
        fix_methyl_hydrogens=False,
        fix_methyl_carbons=False,
        fix_other_hydrogen_atoms=False,
        fix_other_heavy_atoms=False,
    )
    resp_charges_1 = solver.solve(
        design_matrix_1,
        reference_esp_1,
        constraint_matrix_1,
        constraint_vector_1,
        a,
        b,
        restraint_indices_1,
        len(qc_data_records),
    )

    resp_parameter_1.value = (
        resp_charges_1[: len(resp_parameter_1.value)].flatten().tolist()
    )

    ###################################################################################
    #                                       STAGE 2                                   #
    #                                                                                 #
    # * METHYL(ENE) C   : constrain equiv within=[✓] between=[✓] conformers fixed=[x] #
    # * METHYL(ENE) H   : constrain equiv within=[✓] between=[✓] conformers fixed=[x] #
    # * HEAVY ATOMS     : constrain equiv within=[-] between=[-] conformers fixed=[✓] #
    # * HYDROGEN        : constrain equiv within=[-] between=[-] conformers fixed=[✓] #
    #                                                                                 #
    ###################################################################################

    a = 0.001

    resp_parameter_2 = molecule_to_resp_library_charge(
        molecule,
        equivalize_within_methyl_hydrogens=True,
        equivalize_within_methyl_carbons=True,
        equivalize_within_other_hydrogen_atoms=True,
        equivalize_within_other_heavy_atoms=True,
    )
    resp_parameter_2.copy_value_from_other(resp_parameter_1)

    (
        design_matrix_2,
        reference_esp_2,
        constraint_matrix_2,
        constraint_vector_2,
        restraint_indices_2,
        charge_map,
    ) = generate_resp_systems_of_equations(
        resp_parameter_2,
        qc_data_records,
        equivalize_between_methyl_hydrogens=True,
        equivalize_between_methyl_carbons=True,
        equivalize_between_other_hydrogen_atoms=True,
        equivalize_between_other_heavy_atoms=True,
        fix_methyl_hydrogens=False,
        fix_methyl_carbons=False,
        fix_other_hydrogen_atoms=True,
        fix_other_heavy_atoms=True,
    )

    resp_charges_2 = solver.solve(
        design_matrix_2,
        reference_esp_2,
        constraint_matrix_2,
        constraint_vector_2,
        a,
        b,
        restraint_indices_2,
        len(qc_data_records),
    )

    for array_index, value_index in charge_map.items():
        resp_parameter_2.value[value_index] = float(resp_charges_2[array_index].item(0))

    return resp_parameter_2
