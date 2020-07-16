from typing import List, Optional

import numpy

from openff.recharge.charges.bcc import BCCCollection, BCCParameter
from openff.recharge.charges.charges import ChargeSettings
from openff.recharge.esp import ESPSettings
from openff.recharge.esp.storage import MoleculeESPRecord, MoleculeESPStore
from openff.recharge.grids import GridSettings
from openff.recharge.optimize import ESPOptimization
from openff.recharge.utilities.openeye import smiles_to_molecule


class MockMoleculeESPStore(MoleculeESPStore):
    def retrieve(
        self,
        smiles: Optional[str] = None,
        basis: Optional[str] = None,
        method: Optional[str] = None,
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
                esp_settings=ESPSettings(grid_settings=GridSettings()),
            )
        ]


def test_inverse_distance_matrix():

    esp_record = MoleculeESPRecord(
        tagged_smiles="[Ar:1]",
        conformer=numpy.array([[0.0, 0.0, 0.0]]),
        grid_coordinates=numpy.array([[5.0, 0.0, 0.0]]),
        esp=numpy.zeros((1, 1)),
        esp_settings=ESPSettings(grid_settings=GridSettings()),
    )

    inverse_distance_matrix = ESPOptimization.inverse_distance_matrix(esp_record)

    distance_matrix = 1.0 / inverse_distance_matrix
    assert numpy.isclose(distance_matrix, 5.0)


def test_compute_v_difference():

    v_difference = ESPOptimization.compute_v_difference(
        numpy.array([[1.0 / 5.0, 0.0], [0.0, 1.0 / 5.0]]),
        numpy.array([[5.0], [10.0]]),
        numpy.array([[1.0], [2.0]]),
    )

    assert numpy.allclose(v_difference, 0.0)


def test_compute_objective_function():

    objective_function = ESPOptimization.compute_objective_function(
        numpy.array([[6.0]]), numpy.array([[2.0]]),
    )

    assert numpy.isclose(objective_function, 16.0)


def test_precalculate(tmp_path):

    bcc_collection = BCCCollection(
        parameters=[
            BCCParameter(smirks="[#6:1]-[#1:2]", value=1.0, provenance={}),
            BCCParameter(smirks="[#6:1]#[#6:2]", value=2.0, provenance={}),
        ]
    )

    precalculated_terms = ESPOptimization.precalculate(
        ["C#C"],
        MockMoleculeESPStore(f"{tmp_path}.sqlite"),
        bcc_collection,
        [1],
        ChargeSettings(),
    )

    assert len(precalculated_terms) == 1
    precalculated_term = precalculated_terms[0]

    assert precalculated_term.assignment_matrix.shape == (4, 1)
    assert precalculated_term.inverse_distance_matrix.shape == (1, 4)
    assert precalculated_term.v_difference.shape == (1, 1)
