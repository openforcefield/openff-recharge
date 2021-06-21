from typing import List, Optional

import numpy
from openff.toolkit.typing.engines.smirnoff import ForceField

from openff.recharge.charges.bcc import (
    BCCCollection,
    BCCParameter,
    VSiteSMIRNOFFCollection,
)
from openff.recharge.charges.charges import ChargeSettings
from openff.recharge.esp import ESPSettings
from openff.recharge.esp.storage import MoleculeESPRecord, MoleculeESPStore
from openff.recharge.grids import GridSettings, GridGenerator
from openff.recharge.optimize import ESPOptimization
from openff.recharge.optimize.optimize import ElectricFieldOptimization
from openff.recharge.utilities.openeye import smiles_to_molecule
from openff.recharge.smirnoff import from_smirnoff_virtual_sites


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


class MockMoleculeESPStoreWater(MoleculeESPStore):
    def retrieve(
        self,
        smiles: Optional[str] = None,
        basis: Optional[str] = None,
        method: Optional[str] = None,
        implicit_solvent: Optional[bool] = None,
    ) -> List[MoleculeESPRecord]:

        oe_molecule = smiles_to_molecule("O")
        conformer = numpy.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        grid = GridGenerator.generate(oe_molecule, conformer, GridSettings())
        n_pts = grid.shape[0]
        return [
            MoleculeESPRecord.from_oe_molecule(
                oe_molecule,
                conformer=conformer,
                grid_coordinates=grid,
                esp=numpy.zeros((n_pts, 1)),
                electric_field=numpy.zeros((n_pts, 3)),
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
        vsite_collection=None,
    )

    objective_terms = [*objective_terms_generator]

    assert len(objective_terms) == 1
    objective_term = objective_terms[0]

    assert objective_term.design_matrix.shape == (1, 1)
    assert objective_term.target_residuals.shape == (1, 1)


def vsite_collection_from_smirnoff_xml(xml_filename: str) -> VSiteSMIRNOFFCollection:
    """
    Generate a virtual site collection from a SMIRNOFF force field

    Parameters
    ----------
    xml_filename : str
        Name of the SMIRNOFF XML file

    Returns
    -------
    VSiteSMIRNOFFCollection:
        The virtual site collection
    """

    FF = ForceField(xml_filename, allow_cosmetic_attributes=True)

    name = "VirtualSites"
    VSH = FF.get_parameter_handler(name)

    vsite_collection = from_smirnoff_virtual_sites(
        parameter_handler=VSH, parameter_handler_name=name
    )

    return vsite_collection


def test_compute_esp_objective_terms_with_vsites(tmp_path):

    bcc_collection = BCCCollection(
        parameters=[
            BCCParameter(smirks="[#6:1]-[#1:2]", value=1.0, provenance={}),
            BCCParameter(smirks="[#6:1]#[#6:2]", value=2.0, provenance={}),
        ]
    )

    # this adds 5 charge increments from two vsites, only one applies to C#C
    vsite_collection = vsite_collection_from_smirnoff_xml(
        "./smirnoff_with_vsite.offxml"
    )

    objective_terms_generator = ESPOptimization.compute_objective_terms(
        ["C#C"],
        MockMoleculeESPStore(f"{tmp_path}.sqlite"),
        bcc_collection,
        [],
        ChargeSettings(),
        vsite_collection=vsite_collection,
    )

    objective_terms = [*objective_terms_generator]

    assert len(objective_terms) == 1
    objective_term = objective_terms[0]

    # 2 from bcc collection, 5 (only 2 assigned) from vsite FF
    # this contains A and b, solve for q

    assert objective_term.design_matrix.shape == (1, 7)
    assert objective_term.target_residuals.shape == (1, 1)


def test_compute_esp_objective_terms_tip4(tmp_path):

    bcc_collection = BCCCollection(
        parameters=[
            BCCParameter(smirks="[#8:1]-[#1:2]", value=0.0, provenance={}),
        ]
    )

    # this adds 5 charge increments from two vsites, only one applies to C#C
    vsite_collection = vsite_collection_from_smirnoff_xml("./tip4.offxml")

    store = MockMoleculeESPStoreWater(f"{tmp_path}.sqlite")
    objective_terms_generator = ESPOptimization.compute_objective_terms(
        ["O"],
        store,
        bcc_collection,
        [],
        ChargeSettings(),
        vsite_collection=vsite_collection,
    )

    objective_terms = [*objective_terms_generator]

    assert len(objective_terms) == 1
    objective_term = objective_terms[0]


    # 1 from bcc collection, 3 from vsite FF
    # this contains A and b, solve for q

    import scipy.optimize
    import numpy as np

    def fun(x, A, b):
        return ((np.dot(A, x.reshape(-1, 1)) - b) ** 2).sum()

    x0 = np.zeros((objective_term.design_matrix.shape[1],))
    ret = scipy.optimize.least_squares(
        fun,
        x0,
        args=(objective_term.design_matrix, objective_term.target_residuals),
        verbose=True,
    )
    print(ret.x)
    # assert objective_term.design_matrix.shape == (1, 5)
    # assert objective_term.target_residuals.shape == (1, 1)


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
        vsite_collection=None,
    )

    objective_terms = [*objective_terms_generator]

    assert len(objective_terms) == 1
    objective_term = objective_terms[0]

    assert objective_term.design_matrix.shape == (1, 3, 1)
    assert objective_term.target_residuals.shape == (1, 3)


def test_compute_electric_field_objective_terms_with_vsites(tmp_path):

    bcc_collection = BCCCollection(
        parameters=[
            BCCParameter(smirks="[#6:1]-[#1:2]", value=1.0, provenance={}),
            BCCParameter(smirks="[#6:1]#[#6:2]", value=2.0, provenance={}),
        ]
    )

    vsite_collection = vsite_collection_from_smirnoff_xml(
        "./smirnoff_with_vsite.offxml"
    )

    objective_terms_generator = ElectricFieldOptimization.compute_objective_terms(
        ["C#C"],
        MockMoleculeESPStore(f"{tmp_path}.sqlite"),
        bcc_collection,
        [1],
        ChargeSettings(),
        vsite_collection=vsite_collection,
    )

    objective_terms = [*objective_terms_generator]

    assert len(objective_terms) == 1
    objective_term = objective_terms[0]

    assert objective_term.design_matrix.shape == (1, 3, 6)
    assert objective_term.target_residuals.shape == (1, 3)
