from typing import Callable, Tuple, Type

import numpy
import pytest
from openff.units import unit
from typing_extensions import Literal

from openff.recharge.charges.bcc import BCCCollection, BCCParameter
from openff.recharge.charges.vsite import (
    BondChargeSiteParameter,
    MonovalentLonePairParameter,
    VirtualSiteCollection,
)
from openff.recharge.esp import ESPSettings
from openff.recharge.esp.storage import MoleculeESPRecord
from openff.recharge.grids import GridSettings
from openff.recharge.optimize import (
    ElectricFieldObjective,
    ElectricFieldObjectiveTerm,
    ESPObjective,
    ESPObjectiveTerm,
)
from openff.recharge.optimize._optimize import Objective, ObjectiveTerm
from openff.recharge.utilities.openeye import smiles_to_molecule

try:
    import torch
except ImportError:
    torch = None

backends = ["numpy"] + ([] if torch is None else ["torch"])

BOHR_TO_ANGSTROM = unit.convert(1.0, unit.bohr, unit.angstrom)


@pytest.fixture()
def hcl_esp_record() -> MoleculeESPRecord:

    oe_molecule = smiles_to_molecule("[H]Cl")

    return MoleculeESPRecord.from_oe_molecule(
        oe_molecule,
        conformer=numpy.array([[-4, 0, 0], [0, 0, 0]]) * BOHR_TO_ANGSTROM,
        grid_coordinates=numpy.array([[-4, 3, 0], [4, 3, 0]]) * BOHR_TO_ANGSTROM,
        esp=numpy.array([[2.0], [2.0]]),
        electric_field=numpy.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
        esp_settings=ESPSettings(grid_settings=GridSettings()),
    )


@pytest.fixture()
def hcl_parameters() -> Tuple[BCCCollection, VirtualSiteCollection]:

    bcc_collection = BCCCollection(
        parameters=[
            BCCParameter(smirks="[#17:1]-[#1:2]", value=-1.0, provenance={}),
        ]
    )
    vsite_collection = VirtualSiteCollection(
        parameters=[
            BondChargeSiteParameter(
                smirks="[#17:1]-[#1:2]",
                name="EP",
                distance=4.0 * BOHR_TO_ANGSTROM,
                match="once",
                charge_increments=(0.5, 0.1),
                sigma=1.0,
                epsilon=0.0,
            ),
        ]
    )
    return bcc_collection, vsite_collection


@pytest.mark.parametrize(
    "input_type, output_type, backend",
    (
        []
        if torch is None
        else [
            (numpy.array, torch.Tensor, "torch"),
            (torch.tensor, torch.Tensor, "torch"),
            (numpy.array, numpy.ndarray, "numpy"),
            (torch.tensor, numpy.ndarray, "numpy"),
        ]
    ),
)
def test_term_to_backend(
    input_type: Callable, output_type: Type, backend: Literal["numpy", "torch"]
):

    objective_term = ESPObjectiveTerm(
        atom_charge_design_matrix=input_type([[1.0, 2.0]]),
        vsite_charge_assignment_matrix=input_type([[1], [-1]]),
        vsite_fixed_charges=input_type([[1.0]]),
        vsite_coord_assignment_matrix=input_type([[-1, 1, -1]]),
        vsite_fixed_coords=input_type([[0.0, 0.0, 0.0]]),
        vsite_local_coordinate_frame=input_type(
            [[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0]]]
        ),
        grid_coordinates=input_type([[0.0, 0.0, 0.0]]),
        reference_values=input_type([[0.0]]),
    )
    objective_term.to_backend(backend)

    assert isinstance(objective_term.atom_charge_design_matrix, output_type)

    assert isinstance(objective_term.vsite_charge_assignment_matrix, output_type)
    assert isinstance(objective_term.vsite_fixed_charges, output_type)

    assert isinstance(objective_term.vsite_coord_assignment_matrix, output_type)
    assert isinstance(objective_term.vsite_fixed_coords, output_type)

    assert isinstance(objective_term.grid_coordinates, output_type)
    assert isinstance(objective_term.reference_values, output_type)


@pytest.mark.parametrize(
    "term_class, design_matrix_precursor, expected_loss",
    [
        (
            ESPObjectiveTerm,
            # Assumes two atoms at (0,0,0) and (4,0,0) and one grid point at (0,3,0)
            numpy.array([[1.0 / 3.0, 1.0 / 5.0]]),
            # Target value (1.0) - design matrix @ charges (2.0)
            (1.0 - (2.0 / 3.0 - 2.0 / 5.0)) ** 2,
        ),
        (
            ElectricFieldObjectiveTerm,
            numpy.array(
                [[[+0.0, -4.0 / 125.0], [+3.0 / 27.0, +3.0 / 125.0], [+0.0, +0.0]]]
            ),
            (
                (1.0 - 2.0 * (0.0 - -4.0 / 125.0)) ** 2
                + (1.0 - 2.0 * (3.0 / 27.0 - 3.0 / 125.0)) ** 2
                + 1.0
            ),
        ),
    ],
)
@pytest.mark.parametrize("backend", backends)
def test_term_loss_bcc_only(
    term_class: Type[ObjectiveTerm],
    design_matrix_precursor: numpy.ndarray,
    expected_loss: numpy.ndarray,
    backend: Literal["numpy", "torch"],
):

    charge_values = (
        numpy.array([[2.0]]) if backend == "numpy" else torch.tensor([[2.0]])
    )

    term = term_class(
        atom_charge_design_matrix=design_matrix_precursor @ numpy.array([[1], [-1]]),
        vsite_charge_assignment_matrix=None,
        vsite_fixed_charges=None,
        vsite_coord_assignment_matrix=None,
        vsite_fixed_coords=None,
        vsite_local_coordinate_frame=None,
        grid_coordinates=None,
        reference_values=numpy.ones((1, 1 if design_matrix_precursor.ndim == 2 else 3)),
    )
    term.to_backend(backend)

    output_loss = float(term.loss(charge_values, None))

    assert numpy.isclose(expected_loss, output_loss)


@pytest.mark.parametrize(
    "term_class, expected_loss",
    [
        # Assumes an atom at (0,0,0), v-site at (4,0,0) and a grid point at (0,3,0)
        (
            ESPObjectiveTerm,
            # Target value (1.0) - inverse distance @ charges (2.0)
            (1.0 - 2.0 / 5.0) ** 2,
        ),
        (
            ElectricFieldObjectiveTerm,
            (
                # Target value (1.0) - vector field @ charges (2.0)
                (1.0 - 2.0 * -4.0 / 125.0) ** 2
                + (1.0 - 2.0 * 3.0 / 125.0) ** 2
                + 1.0
            ),
        ),
    ],
)
@pytest.mark.parametrize("backend", backends)
def test_term_evaluate_vsite_only(
    term_class: Type[ObjectiveTerm],
    expected_loss: numpy.ndarray,
    backend: Literal["numpy", "torch"],
):

    charge_values = (
        numpy.array([[2.0]]) if backend == "numpy" else torch.tensor([[2.0]])
    )
    coordinate_values = (
        numpy.array([[4.0 * BOHR_TO_ANGSTROM]])
        if backend == "numpy"
        else torch.tensor([[4.0 * BOHR_TO_ANGSTROM]])
    )

    term = term_class(
        atom_charge_design_matrix=None,
        vsite_charge_assignment_matrix=numpy.array([[-1.0]]),
        # In practice this should be zero as there is only one charge increment and
        # it is 'trainable', however for testing we give it a value to ensure this
        # term is correctly included.
        vsite_fixed_charges=numpy.array([[4.0]]),
        vsite_coord_assignment_matrix=numpy.array([[0, -1, -1]]),
        vsite_fixed_coords=numpy.array([[0.0, 180.0, 0.0]]),
        vsite_local_coordinate_frame=numpy.array(
            [
                [[0.0, 0.0, 0.0]],
                [[-1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0]],
            ]
        ),
        grid_coordinates=numpy.array([[0.0, 3.0, 0.0]]) * BOHR_TO_ANGSTROM,
        reference_values=numpy.ones((1, 1 if term_class == ESPObjectiveTerm else 3)),
    )
    term.to_backend(backend)

    output_loss = float(term.loss(charge_values, coordinate_values))

    assert numpy.isclose(expected_loss, output_loss)


@pytest.mark.parametrize(
    "term_class, design_matrix_precursor, expected_loss",
    [
        # Assumes two atoms at (0,0,0) and (4,0,0), a v-site a (-4,0,0) and a grid point
        # at (0,3,0). The first atom should receive a q=2.5, the second q=-2.0, and the
        # v-site q=-0.5
        (
            ESPObjectiveTerm,
            numpy.array([[1.0 / 3.0, 1.0 / 5.0]]),
            # Target value (1.0) - design matrix @ charges (2.0)
            (1.0 - (2.5 / 3.0 - 2.0 / 5.0 - 0.5 / 5.0)) ** 2,
        ),
        (
            ElectricFieldObjectiveTerm,
            numpy.array([[[0.0, -4.0 / 125.0], [3.0 / 27.0, 3.0 / 125.0], [0.0, 0.0]]]),
            (
                (1.0 - (2.5 * 0.0 + -2.0 * -4.0 / 125.0 + -0.5 * 4.0 / 125.0)) ** 2
                + (1.0 - (2.5 * 3.0 / 27.0 + -2.0 * 3.0 / 125.0 + -0.5 * 3.0 / 125.0))
                ** 2
                + 1.0
            ),
        ),
    ],
)
@pytest.mark.parametrize("backend", backends)
def test_term_evaluate_bcc_and_vsite(
    term_class, design_matrix_precursor, expected_loss, backend
):

    charge_values = (
        numpy.array([[2.0], [0.5]])
        if backend == "numpy"
        else torch.tensor([[2.0], [0.5]])
    )

    coordinate_values = (
        numpy.array([[-4.0 * BOHR_TO_ANGSTROM]])
        if backend == "numpy"
        else torch.tensor([[-4.0 * BOHR_TO_ANGSTROM]])
    )

    term = term_class(
        atom_charge_design_matrix=design_matrix_precursor
        @ numpy.array([[1, 1], [-1, 0]]),
        vsite_charge_assignment_matrix=numpy.array([[-1.0]]),
        vsite_fixed_charges=numpy.array([[0.0]]),
        vsite_coord_assignment_matrix=numpy.array([[0, -1, -1]]),
        vsite_fixed_coords=numpy.array([[0.0, 180.0, 0.0]]),
        vsite_local_coordinate_frame=numpy.array(
            [
                [[0.0, 0.0, 0.0]],
                [[-1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0]],
            ]
        ),
        grid_coordinates=numpy.array([[0.0, 3.0, 0.0]]) * BOHR_TO_ANGSTROM,
        reference_values=numpy.ones((1, 1 if term_class == ESPObjectiveTerm else 3)),
    )
    term.to_backend(backend)

    output_loss = float(term.loss(charge_values, coordinate_values))

    assert numpy.isclose(expected_loss, output_loss)


@pytest.mark.parametrize("objective_class", [ESPObjective, ElectricFieldObjective])
@pytest.mark.parametrize("backend", backends)
def test_combine_terms(objective_class, backend, hcl_parameters):

    bcc_collection, vsite_collection = hcl_parameters

    objective_terms_generator = objective_class.compute_objective_terms(
        esp_records=[
            MoleculeESPRecord.from_oe_molecule(
                smiles_to_molecule("[H]Cl"),
                conformer=numpy.random.random((2, 3)),
                grid_coordinates=numpy.random.random((grid_size, 3)),
                esp=numpy.random.random((grid_size, 1)),
                electric_field=numpy.random.random((grid_size, 3)),
                esp_settings=ESPSettings(grid_settings=GridSettings()),
            )
            for grid_size in [4, 5]
        ],
        charge_settings=None,
        bcc_collection=bcc_collection,
        bcc_parameter_keys=["[#17:1]-[#1:2]"],
        vsite_collection=vsite_collection,
        vsite_charge_parameter_keys=[("[#17:1]-[#1:2]", "BondCharge", "EP", 0)],
        vsite_coordinate_parameter_keys=[
            ("[#17:1]-[#1:2]", "BondCharge", "EP", "distance")
        ],
    )
    objective_terms = [*objective_terms_generator]

    # Define a dummy set of parameters
    charge_values = (
        numpy.random.random((2, 1)) if backend == "numpy" else torch.rand((2, 1))
    )
    coordinate_values = (
        numpy.random.random((1, 1)) if backend == "numpy" else torch.rand((1, 1))
    )

    # Compute the total loss by summation.
    summed_loss = numpy.zeros(1) if backend == "numpy" else torch.zeros(1)

    for objective_term in objective_terms:

        objective_term.to_backend(backend)
        summed_loss += objective_term.loss(charge_values, coordinate_values)

    # Combine the terms and re-compute the loss
    combined_term = objective_class._objective_term().combine(*objective_terms)
    combined_loss = combined_term.loss(charge_values, coordinate_values)

    assert numpy.isclose(float(summed_loss), float(combined_loss))


def test_compute_bcc_charge_terms():

    oe_molecule = smiles_to_molecule("C#C")

    bcc_collection = BCCCollection(
        parameters=[
            BCCParameter(smirks="[#6:1]-[#1:2]", value=1.0, provenance={}),
            BCCParameter(smirks="[#6:1]#[#6:2]", value=2.0, provenance={}),
        ]
    )

    assignment_matrix, fixed_charges = Objective._compute_bcc_charge_terms(
        oe_molecule, bcc_collection, ["[#6:1]-[#1:2]"]
    )

    assert assignment_matrix.shape == (4, 1)
    assert numpy.allclose(
        assignment_matrix, numpy.array([[1.0], [1.0], [-1.0], [-1.0]])
    )

    assert fixed_charges.shape == (4, 1)
    assert numpy.allclose(fixed_charges, numpy.array([[2], [-2], [0], [0]]))


def test_compute_vsite_charge_terms():

    oe_molecule = smiles_to_molecule("C#C")

    vsite_collection = VirtualSiteCollection(
        parameters=[
            BondChargeSiteParameter(
                smirks="[#6:1]-[#1:2]",
                name="EP",
                distance=-0.1,
                match="once",
                charge_increments=(0.0, 1.0),
                sigma=1.0,
                epsilon=0.0,
            ),
            BondChargeSiteParameter(
                smirks="[#6:1]#[#6:2]",
                name="EP",
                distance=1.0,
                match="once",
                charge_increments=(-2.0, -2.0),
                sigma=1.0,
                epsilon=0.0,
            ),
        ]
    )

    assignment_matrix, fixed_charges = Objective._compute_vsite_charge_terms(
        oe_molecule,
        vsite_collection,
        [
            ("[#6:1]-[#1:2]", "BondCharge", "EP", 0),
            ("[#6:1]#[#6:2]", "BondCharge", "EP", 1),
        ],
    )

    assert assignment_matrix.shape == (7, 2)
    assert numpy.allclose(
        assignment_matrix,
        numpy.array([[1, 0], [1, 1], [0, 0], [0, 0], [0, -1], [-1, 0], [-1, 0]]),
    )

    assert fixed_charges.shape == (7, 1)
    assert numpy.allclose(
        fixed_charges, numpy.array([[-2.0], [0.0], [1.0], [1.0], [2.0], [-1.0], [-1.0]])
    )


def test_compute_vsite_coord_terms():

    oe_molecule = smiles_to_molecule("C=O")

    conformer = numpy.array(
        [
            [+1.0, +0.0, +0.0],
            [+0.0, +0.0, +0.0],
            [-1.0, +0.0, +1.0],
            [-1.0, +0.0, -1.0],
        ]
    )

    vsite_collection = VirtualSiteCollection(
        parameters=[
            MonovalentLonePairParameter(
                smirks="[O:1]=[C:2]-[H:3]",
                name="EP",
                charge_increments=(0.0, 0.0, 0.0),
                sigma=0.0,
                match="once",
                epsilon=0.0,
                distance=1.0,
                in_plane_angle=180.0,
                out_of_plane_angle=45.0,
            )
        ]
    )

    assignment_matrix, fixed_coords, local_frame = Objective._compute_vsite_coord_terms(
        oe_molecule,
        conformer,
        vsite_collection,
        [
            ("[O:1]=[C:2]-[H:3]", "MonovalentLonePair", "EP", "out_of_plane_angle"),
            ("[O:1]=[C:2]-[H:3]", "MonovalentLonePair", "EP", "distance"),
        ],
    )

    assert assignment_matrix.shape == (2, 3)
    assert numpy.allclose(assignment_matrix, numpy.array([[1, -1, 0], [1, -1, 0]]))

    assert fixed_coords.shape == (2, 3)
    assert numpy.allclose(
        fixed_coords, numpy.array([[0.0, 180.0, 0.0], [0.0, 180.0, 0.0]])
    )

    assert local_frame.shape == (4, 2, 3)


def test_compute_esp_objective_terms(hcl_esp_record, hcl_parameters):

    bcc_collection, vsite_collection = hcl_parameters

    objective_terms_generator = ESPObjective.compute_objective_terms(
        [hcl_esp_record],
        None,
        bcc_collection=bcc_collection,
        bcc_parameter_keys=["[#17:1]-[#1:2]"],
        vsite_collection=vsite_collection,
        vsite_charge_parameter_keys=[("[#17:1]-[#1:2]", "BondCharge", "EP", 0)],
        vsite_coordinate_parameter_keys=[
            ("[#17:1]-[#1:2]", "BondCharge", "EP", "distance")
        ],
    )
    objective_terms = [*objective_terms_generator]

    assert len(objective_terms) == 1
    objective_term = objective_terms[0]

    expected_design_matrix = numpy.array(
        [
            [1.0 / 5.0 - 1.0 / 3.0, 1.0 / 5.0],
            [1.0 / 5.0 - 1.0 / numpy.sqrt(3.0 * 3.0 + 8.0 * 8.0), 1.0 / 5.0],
        ]
    )
    assert (
        objective_term.atom_charge_design_matrix.shape == expected_design_matrix.shape
    )
    assert numpy.allclose(
        objective_term.atom_charge_design_matrix, expected_design_matrix
    )

    assert objective_term.vsite_charge_assignment_matrix.shape == (1, 1)
    assert numpy.isclose(objective_term.vsite_charge_assignment_matrix, -1)

    assert objective_term.vsite_fixed_charges.shape == (1, 1)
    assert numpy.isclose(objective_term.vsite_fixed_charges, -0.1)

    assert objective_term.vsite_coord_assignment_matrix.shape == (1, 3)
    assert numpy.allclose(
        objective_term.vsite_coord_assignment_matrix, numpy.array([[0, -1, -1]])
    )

    assert objective_term.vsite_fixed_coords.shape == (1, 3)
    assert numpy.allclose(
        objective_term.vsite_fixed_coords, numpy.array([[0.0, 180.0, 0.0]])
    )

    assert objective_term.vsite_local_coordinate_frame.shape == (4, 1, 3)

    assert (
        hcl_esp_record.grid_coordinates.shape == objective_term.grid_coordinates.shape
    )
    assert numpy.allclose(
        hcl_esp_record.grid_coordinates, objective_term.grid_coordinates
    )

    assert objective_term.reference_values.shape == (2, 1)
    assert numpy.allclose(
        objective_term.reference_values,
        numpy.array(
            [[2.0 - (0.1 / 3.0)], [2.0 - (0.1 / numpy.sqrt(3.0 * 3.0 + 8.0 * 8.0))]]
        ),
    )


def test_compute_field_objective_terms(hcl_esp_record, hcl_parameters):

    bcc_collection, vsite_collection = hcl_parameters

    objective_terms_generator = ElectricFieldObjective.compute_objective_terms(
        [hcl_esp_record],
        None,
        bcc_collection=bcc_collection,
        bcc_parameter_keys=["[#17:1]-[#1:2]"],
        vsite_collection=vsite_collection,
        vsite_charge_parameter_keys=[("[#17:1]-[#1:2]", "BondCharge", "EP", 0)],
        vsite_coordinate_parameter_keys=[
            ("[#17:1]-[#1:2]", "BondCharge", "EP", "distance")
        ],
    )
    objective_terms = [*objective_terms_generator]

    assert len(objective_terms) == 1
    objective_term = objective_terms[0]

    # Distance between H and the grid point at (4, 3, 0)
    h_distance = numpy.sqrt(3.0 * 3.0 + 8.0 * 8.0)

    expected_design_matrix = (
        numpy.array(
            [
                [[0.0, -4.0 / 5.0 ** 3], [3.0 / 3.0 ** 3, 3.0 / 5.0 ** 3], [0.0, 0.0]],
                [
                    [8.0 / h_distance ** 3, 4.0 / 5.0 ** 3],
                    [3.0 / h_distance ** 3, 3.0 / 5.0 ** 3],
                    [0.0, 0.0],
                ],
            ]
        )
        @ numpy.array([[-1, 0], [1, 1]])
    )

    assert (
        objective_term.atom_charge_design_matrix.shape == expected_design_matrix.shape
    )
    assert numpy.allclose(
        objective_term.atom_charge_design_matrix, expected_design_matrix
    )

    assert objective_term.vsite_charge_assignment_matrix.shape == (1, 1)
    assert numpy.isclose(objective_term.vsite_charge_assignment_matrix, -1)

    assert objective_term.vsite_fixed_charges.shape == (1, 1)
    assert numpy.isclose(objective_term.vsite_fixed_charges, -0.1)

    assert objective_term.vsite_coord_assignment_matrix.shape == (1, 3)
    assert numpy.allclose(
        objective_term.vsite_coord_assignment_matrix, numpy.array([[0, -1, -1]])
    )

    assert objective_term.vsite_fixed_coords.shape == (1, 3)
    assert numpy.allclose(
        objective_term.vsite_fixed_coords, numpy.array([[0.0, 180.0, 0.0]])
    )

    assert objective_term.vsite_local_coordinate_frame.shape == (4, 1, 3)

    assert (
        hcl_esp_record.grid_coordinates.shape == objective_term.grid_coordinates.shape
    )
    assert numpy.allclose(
        hcl_esp_record.grid_coordinates, objective_term.grid_coordinates
    )

    assert objective_term.reference_values.shape == (2, 3)
    assert numpy.allclose(
        objective_term.reference_values,
        numpy.array(
            [
                [1.0 - 0.0, 2.0 - 0.1 * 3.0 / 3.0 ** 3, 3.0 - 0.0],
                [
                    1.0 - 0.1 * 8.0 / h_distance ** 3,
                    2.0 - 0.1 * 3.0 / h_distance ** 3,
                    3.0 - 0.0,
                ],
            ]
        ),
    )
