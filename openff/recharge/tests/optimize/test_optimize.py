from typing import List, Optional

import numpy

from openff.recharge.charges.bcc import BCCCollection, BCCParameter
from openff.recharge.charges.charges import ChargeSettings
from openff.recharge.esp import ESPSettings
from openff.recharge.esp.storage import MoleculeESPRecord, MoleculeESPStore
from openff.recharge.grids import GridSettings
from openff.recharge.optimize import ESPOptimization
from openff.recharge.optimize.optimize import ElectricFieldOptimization
from openff.recharge.utilities.openeye import smiles_to_molecule


class MockMoleculeESPStore(MoleculeESPStore):
    def retrieve(
        self,
        smiles: Optional[str] = None,
        basis: Optional[str] = None,
        method: Optional[str] = None,
        implicit_solvent: Optional[bool] = None,
    ) -> List[MoleculeESPRecord]:

        oe_molecule = smiles_to_molecule("C#C")
        conformer = numpy.array(
            [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
        )

        return [
            MoleculeESPRecord.from_oe_molecule(
                oe_molecule,
                conformer=conformer,
                grid_coordinates=numpy.zeros((1, 3)),
                esp=numpy.zeros((1, 1)),
                electric_field=numpy.zeros((1, 3)),
                esp_settings=ESPSettings(grid_settings=GridSettings()),
            )
        ]


def test_compute_esp_residuals():

    v_difference = ESPOptimization.compute_residuals(
        numpy.array([[1.0 / 5.0, 0.0], [0.0, 1.0 / 5.0]]),
        numpy.array([[5.0], [10.0]]),
        numpy.array([[1.0], [2.0]]),
    )

    assert numpy.allclose(v_difference, 0.0)


def test_compute_esp_objective_terms(tmp_path):

    bcc_collection = BCCCollection(
        parameters=[
            BCCParameter(smirks="[#6:1]-[#1:2]", value=1.0, provenance={}),
            BCCParameter(smirks="[#6:1]#[#6:2]", value=2.0, provenance={}),
        ]
    )

    objective_terms_generator = ESPOptimization.compute_objective_terms(
        ["C#C"],
        MockMoleculeESPStore(f"{tmp_path}.sqlite"),
        bcc_collection,
        [1],
        ChargeSettings(),
    )

    objective_terms = [*objective_terms_generator]

    assert len(objective_terms) == 1
    objective_term = objective_terms[0]

    assert objective_term.design_matrix.shape == (1, 1)
    assert objective_term.target_residuals.shape == (1, 1)


def test_compute_electric_field_objective_terms(tmp_path):

    bcc_collection = BCCCollection(
        parameters=[
            BCCParameter(smirks="[#6:1]-[#1:2]", value=1.0, provenance={}),
            BCCParameter(smirks="[#6:1]#[#6:2]", value=2.0, provenance={}),
        ]
    )

    objective_terms_generator = ElectricFieldOptimization.compute_objective_terms(
        ["C#C"],
        MockMoleculeESPStore(f"{tmp_path}.sqlite"),
        bcc_collection,
        [1],
        ChargeSettings(),
    )

    objective_terms = [*objective_terms_generator]

    assert len(objective_terms) == 1
    objective_term = objective_terms[0]

    assert objective_term.design_matrix.shape == (1, 3, 1)
    assert objective_term.target_residuals.shape == (1, 3)
