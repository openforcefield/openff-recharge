from typing import List, Optional, Tuple

import numpy
import pytest

from openff.recharge.charges.bcc import BCCCollection, BCCParameter
from openff.recharge.charges.vsite import (
    BondChargeSiteParameter,
    VirtualSiteChargeKey,
    VirtualSiteCollection,
)
from openff.recharge.esp import ESPSettings
from openff.recharge.esp.storage import MoleculeESPRecord
from openff.recharge.grids import GridSettings
from openff.recharge.optimize import ESPOptimization
from openff.recharge.optimize.optimize import ElectricFieldOptimization, _Optimization
from openff.recharge.utilities.geometry import (
    ANGSTROM_TO_BOHR,
    INVERSE_ANGSTROM_TO_BOHR,
    compute_vector_field,
)
from openff.recharge.utilities.openeye import smiles_to_molecule


def mock_esp_record(
    smiles: str,
    conformer: numpy.ndarray,
    grid: numpy.ndarray,
    esp_value: float,
    ef_value: Tuple[float, float, float],
) -> MoleculeESPRecord:

    oe_molecule = smiles_to_molecule(smiles)

    return MoleculeESPRecord.from_oe_molecule(
        oe_molecule,
        conformer=conformer,
        grid_coordinates=grid,
        esp=numpy.array([[esp_value]] * len(grid)),
        electric_field=numpy.array([ef_value] * len(grid)),
        esp_settings=ESPSettings(grid_settings=GridSettings()),
    )


def test_compute_esp_residuals():

    v_difference = ESPOptimization.compute_residuals(
        numpy.array([[1.0 / 5.0, 0.0], [0.0, 1.0 / 5.0]]),
        numpy.array([[5.0], [10.0]]),
        numpy.array([[1.0], [2.0]]),
    )

    assert numpy.allclose(v_difference, 0.0)


def test_compute_bcc_terms():

    oe_molecule = smiles_to_molecule("C#C")

    bcc_collection = BCCCollection(
        parameters=[
            BCCParameter(smirks="[#6:1]-[#1:2]", value=1.0, provenance={}),
            BCCParameter(smirks="[#6:1]#[#6:2]", value=2.0, provenance={}),
        ]
    )

    assignment_matrix, fixed_charges = _Optimization._compute_bcc_terms(
        oe_molecule, bcc_collection, ["[#6:1]-[#1:2]"]
    )

    assert assignment_matrix.shape == (4, 1)
    assert numpy.allclose(
        assignment_matrix, numpy.array([[1.0], [1.0], [-1.0], [-1.0]])
    )

    assert fixed_charges.shape == (4, 1)
    assert numpy.allclose(fixed_charges, numpy.array([[2], [-2], [0], [0]]))


def test_compute_vsite_terms():

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

    assignment_matrix, fixed_charges = _Optimization._compute_vsite_terms(
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
        numpy.array([[-1, 0], [-1, -1], [0, 0], [0, 0], [0, 1], [1, 0], [1, 0]]),
    )

    assert fixed_charges.shape == (7, 1)
    assert numpy.allclose(
        fixed_charges, numpy.array([[2.0], [0.0], [-1.0], [-1.0], [-2.0], [1.0], [1.0]])
    )


def test_vectorize_collections():

    bcc_collection = BCCCollection(
        parameters=[
            BCCParameter(smirks=f"[#{element}:1]-[#1:2]", value=value, provenance={})
            for element, value in [(9, 1.0), (17, 2.0), (35, 3.0)]
        ]
    )
    vsite_collection = VirtualSiteCollection(
        parameters=[
            BondChargeSiteParameter(
                smirks=f"[#1:1]-[#{element}:2]",
                name="EP",
                distance=-4.0,
                match="once",
                charge_increments=values,
                sigma=1.0,
                epsilon=0.0,
            )
            for element, values in [(9, (4.0, 5.0)), (17, (6.0, 7.0)), (35, (8.0, 9.0))]
        ]
    )

    parameter_vector = _Optimization.vectorize_collections(
        bcc_collection=bcc_collection,
        trainable_bcc_parameters=["[#17:1]-[#1:2]"],
        vsite_collection=vsite_collection,
        trainable_vsite_parameters=[
            ("[#1:1]-[#9:2]", "BondCharge", "EP", 1),
            ("[#1:1]-[#35:2]", "BondCharge", "EP", 0),
        ],
    )
    assert parameter_vector.shape == (3, 1)
    assert numpy.allclose(parameter_vector, numpy.array([[2.0], [5.0], [8.0]]))


@pytest.mark.parametrize(
    "bcc_collection, bcc_keys, vsite_collection, vsite_keys, expected_design_matrix, "
    "expected_residuals",
    [
        (
            BCCCollection(
                parameters=[
                    BCCParameter(smirks="[#17:1]-[#1:2]", value=1.0, provenance={}),
                ]
            ),
            ["[#17:1]-[#1:2]"],
            None,
            [],
            numpy.array([[-2.0 / 15.0], [2.0 / 15.0]]),
            numpy.array([[2.0], [2.0]]),
        ),
        (
            None,
            [],
            VirtualSiteCollection(
                parameters=[
                    BondChargeSiteParameter(
                        smirks="[#1:1]-[#17:2]",
                        name="EP",
                        distance=-4.0,
                        match="once",
                        charge_increments=(0.0, 1.0),
                        sigma=1.0,
                        epsilon=0.0,
                    ),
                ]
            ),
            [("[#1:1]-[#17:2]", "BondCharge", "EP", i) for i in [0, 1]],
            numpy.array([[-2.0 / 15.0, 0.0], [2.0 / 15.0, 0.0]]),
            numpy.array([[2.0], [2.0]]),
        ),
    ],
)
def test_compute_esp_objective_terms(
    bcc_collection: Optional[BCCCollection],
    bcc_keys: List[str],
    vsite_collection: Optional[VirtualSiteCollection],
    vsite_keys: List[VirtualSiteChargeKey],
    expected_design_matrix: numpy.ndarray,
    expected_residuals: numpy.ndarray,
):

    esp_record = mock_esp_record(
        "[H]Cl",
        numpy.array([[-2, 0, 0], [2, 0, 0]]),
        numpy.array([[-2, 3, 0], [2, 0, 3]]),
        2.0,
        (1.0, 1.0, 1.0),
    )

    objective_terms_generator = ESPOptimization.compute_objective_terms(
        [esp_record],
        None,
        bcc_collection=bcc_collection,
        trainable_bcc_parameters=bcc_keys,
        vsite_collection=vsite_collection,
        trainable_vsite_parameters=vsite_keys,
    )
    objective_terms = [*objective_terms_generator]

    assert len(objective_terms) == 1
    objective_term = objective_terms[0]

    assert objective_term.design_matrix.shape == expected_design_matrix.shape

    assert numpy.allclose(
        objective_term.design_matrix / INVERSE_ANGSTROM_TO_BOHR,
        expected_design_matrix,
    )

    assert objective_term.target_residuals.shape == expected_residuals.shape
    assert numpy.allclose(objective_term.target_residuals, expected_residuals)


@pytest.mark.parametrize(
    "bcc_collection, bcc_keys, vsite_collection, vsite_keys, "
    "expected_assignment_matrix, expected_residuals",
    [
        (
            BCCCollection(
                parameters=[
                    BCCParameter(smirks="[#17:1]-[#1:2]", value=1.0, provenance={}),
                ]
            ),
            ["[#17:1]-[#1:2]"],
            None,
            [],
            numpy.array([[-1.0], [1.0]]),
            numpy.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
        ),
        (
            None,
            [],
            VirtualSiteCollection(
                parameters=[
                    BondChargeSiteParameter(
                        smirks="[#1:1]-[#17:2]",
                        name="EP",
                        distance=-4.0,
                        match="once",
                        charge_increments=(0.0, 1.0),
                        sigma=1.0,
                        epsilon=0.0,
                    ),
                ]
            ),
            [("[#1:1]-[#17:2]", "BondCharge", "EP", i) for i in [0, 1]],
            numpy.array([[-1, 0], [0, -1], [1, 1]]),
            numpy.array([[2.0], [2.0]]),
        ),
    ],
)
def test_compute_electric_field_objective_terms(
    bcc_collection: Optional[BCCCollection],
    bcc_keys: List[str],
    vsite_collection: Optional[VirtualSiteCollection],
    vsite_keys: List[VirtualSiteChargeKey],
    expected_assignment_matrix: numpy.ndarray,
    expected_residuals: numpy.ndarray,
):

    esp_record = mock_esp_record(
        "[H]Cl",
        numpy.array([[-2, 0, 0], [2, 0, 0]]),
        numpy.array([[-2, 3, 0], [2, 0, 3]]),
        2.0,
        (1.0, 2.0, 3.0),
    )

    objective_terms_generator = ElectricFieldOptimization.compute_objective_terms(
        [esp_record],
        None,
        bcc_collection=bcc_collection,
        trainable_bcc_parameters=bcc_keys,
        vsite_collection=vsite_collection,
        trainable_vsite_parameters=vsite_keys,
    )
    objective_terms = [*objective_terms_generator]

    assert len(objective_terms) == 1
    objective_term = objective_terms[0]

    expected_vector_field = compute_vector_field(
        (
            esp_record.conformer
            if vsite_collection is None
            else (numpy.vstack([esp_record.conformer, numpy.array([[2, 0, 0]])]))
        )
        * ANGSTROM_TO_BOHR,
        esp_record.grid_coordinates * ANGSTROM_TO_BOHR,
    )
    expected_design_matrix = expected_vector_field @ expected_assignment_matrix[:, :]

    assert objective_term.design_matrix.shape == expected_design_matrix.shape

    assert numpy.allclose(
        objective_term.design_matrix,
        expected_design_matrix,
    )

    assert objective_term.target_residuals.shape == expected_residuals.shape
    assert numpy.allclose(objective_term.target_residuals, expected_residuals)
