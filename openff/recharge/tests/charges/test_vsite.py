import copy
from typing import TYPE_CHECKING

import numpy
import pytest
from openff.toolkit import Molecule
from openff.units import unit

from openff.recharge.charges.exceptions import ChargeAssignmentError
from openff.recharge.charges.vsite import (
    BondChargeSiteParameter,
    DivalentLonePairParameter,
    MonovalentLonePairParameter,
    TrivalentLonePairParameter,
    VirtualSiteCollection,
    VirtualSiteGenerator,
)
from openff.recharge.tests import does_not_raise
from openff.recharge.utilities.molecule import smiles_to_molecule

pytest.importorskip("openff.toolkit")

if TYPE_CHECKING:
    from openff.toolkit.typing.engines.smirnoff import ForceField, VirtualSiteHandler


def _vsite_handler_to_string(vsite_handler: "VirtualSiteHandler") -> str:
    from openff.toolkit import ForceField

    force_field = ForceField()

    vsite_handler = copy.deepcopy(vsite_handler)
    force_field._parameter_handlers["VirtualSites"] = vsite_handler

    for parameter in vsite_handler.parameters:
        for attribute_name in parameter._get_parameter_attributes():
            attribute = getattr(parameter, attribute_name)

            if not isinstance(attribute, unit.Quantity):
                continue

            setattr(parameter, attribute_name, attribute)

    return force_field.to_string("XML")


@pytest.fixture(scope="module")
def vsite_force_field() -> "ForceField":
    from openff.toolkit.typing.engines.smirnoff import ForceField

    force_field = ForceField()

    vsite_handler: "VirtualSiteHandler" = force_field.get_parameter_handler(
        "VirtualSites"
    )

    vsite_handler.add_parameter(
        parameter_kwargs={
            "smirks": "[#6:2]=[#8:1]",
            "name": "EP",
            "type": "BondCharge",
            "distance": 0.7 * unit.nanometers,
            "match": "all_permutations",
            "charge_increment1": 0.2 * unit.elementary_charge,
            "charge_increment2": 0.1 * unit.elementary_charge,
            "sigma": 1.0 * unit.angstrom,
            "epsilon": 2.0 / 4.184 * unit.kilocalorie_per_mole,
        }
    )
    vsite_handler.add_parameter(
        parameter_kwargs={
            "smirks": "[#1:2]-[#8X2H2+0:1]-[#1:3]",
            "name": "EP",
            "type": "MonovalentLonePair",
            "distance": -0.0106 * unit.nanometers,
            "outOfPlaneAngle": 90.0 * unit.degrees,
            "inPlaneAngle": numpy.pi * unit.radians,
            "match": "all_permutations",
            "charge_increment1": 0.0 * unit.elementary_charge,
            "charge_increment2": 1.0552 * 0.5 * unit.elementary_charge,
            "charge_increment3": 1.0552 * 0.5 * unit.elementary_charge,
            "sigma": 0.0 * unit.nanometers,
            "epsilon": 0.5 * unit.kilojoules_per_mole,
        }
    )
    vsite_handler.add_parameter(
        parameter_kwargs={
            "smirks": "[#1:2]-[#8X2H2+0:1]-[#1:3]",
            "name": "EP",
            "type": "DivalentLonePair",
            "distance": -0.0106 * unit.nanometers,
            "outOfPlaneAngle": 0.1 * unit.degrees,
            "match": "all_permutations",
            "charge_increment1": 0.0 * unit.elementary_charge,
            "charge_increment2": 1.0552 * 0.5 * unit.elementary_charge,
            "charge_increment3": 1.0552 * 0.5 * unit.elementary_charge,
            "sigma": 1.0 * unit.angstrom,
            "epsilon": 0.5 * unit.kilojoules_per_mole,
        }
    )
    vsite_handler.add_parameter(
        parameter_kwargs={
            "smirks": "[#1:2][#7:1]([#1:3])[#1:4]",
            "name": "EP",
            "type": "TrivalentLonePair",
            "distance": 0.5 * unit.nanometers,
            "match": "once",
            "charge_increment1": 0.2 * unit.elementary_charge,
            "charge_increment2": 0.0 * unit.elementary_charge,
            "charge_increment3": 0.0 * unit.elementary_charge,
            "charge_increment4": 0.0 * unit.elementary_charge,
            "sigma": 1.0 * unit.angstrom,
            "epsilon": 0.5 * unit.kilojoules_per_mole,
        }
    )

    return force_field


@pytest.fixture(scope="module")
def vsite_collection(vsite_force_field: "ForceField") -> VirtualSiteCollection:
    return VirtualSiteCollection.from_smirnoff(vsite_force_field["VirtualSites"])


@pytest.mark.skip(
    reason="Virtual site code has not yet been refactored for version 0.11.0"
)
class TestVirtualSiteParameter:
    @pytest.mark.parametrize(
        "parameter_type, n_positions",
        [
            (BondChargeSiteParameter, 2),
            (MonovalentLonePairParameter, 3),
            (DivalentLonePairParameter, 3),
            (TrivalentLonePairParameter, 4),
        ],
    )
    def test_local_frame_weights(self, parameter_type, n_positions):
        assert parameter_type.local_frame_weights().shape == (3, n_positions)

    @pytest.mark.parametrize(
        "parameter, expected_value",
        [
            (
                BondChargeSiteParameter(
                    smirks="[*:1]-[*:2]",
                    name="EP",
                    charge_increments=(0.0, 0.0),
                    sigma=0.0,
                    epsilon=0.0,
                    match="once",
                    distance=2.0,
                ),
                numpy.array([[2.0, 180.0, 0.0]]),
            ),
            (
                MonovalentLonePairParameter(
                    smirks="[*:1]-[*:2]-[*:3]",
                    name="EP",
                    charge_increments=(0.0, 0.0, 0.0),
                    sigma=0.0,
                    epsilon=0.0,
                    match="once",
                    distance=2.0,
                    in_plane_angle=30.0,
                    out_of_plane_angle=60.0,
                ),
                numpy.array([[2.0, 30.0, 60.0]]),
            ),
            (
                DivalentLonePairParameter(
                    smirks="[*:1]-[*:2]-[*:3]",
                    name="EP",
                    charge_increments=(0.0, 0.0, 0.0),
                    sigma=0.0,
                    epsilon=0.0,
                    match="once",
                    distance=2.0,
                    out_of_plane_angle=60.0,
                ),
                numpy.array([[2.0, 180.0, 60.0]]),
            ),
            (
                TrivalentLonePairParameter(
                    smirks="[*:1](-[*:2])(-[*:3])(-[*:4])",
                    name="EP",
                    charge_increments=(0.0, 0.0, 0.0),
                    sigma=0.0,
                    epsilon=0.0,
                    match="once",
                    distance=2.0,
                ),
                numpy.array([[2.0, 180.0, 0.0]]),
            ),
        ],
    )
    def test_local_frame_coordinates(self, parameter, expected_value):
        assert parameter.local_frame_coordinates.shape == expected_value.shape
        assert numpy.allclose(parameter.local_frame_coordinates, expected_value)


@pytest.mark.skip(
    reason="Virtual site code has not yet been refactored for version 0.11.0"
)
class TestVirtualSiteCollection:
    def test_to_smirnoff(
        self, vsite_force_field: "ForceField", vsite_collection: VirtualSiteCollection
    ):
        """Test that a collection of v-site parameters can be mapped to a SMIRNOFF
        `VirtualSiteHandler`.
        """

        expected_xml = _vsite_handler_to_string(vsite_force_field["VirtualSites"])

        smirnoff_handler = vsite_collection.to_smirnoff()
        assert smirnoff_handler is not None

        actual_xml = _vsite_handler_to_string(smirnoff_handler)

        assert expected_xml == actual_xml

    def test_from_smirnoff(self, vsite_force_field):
        """Test that a SMIRNOFF `VirtualSiteHandler` can be mapped to
        a collection of v-site parameters
        """

        # noinspection PyTypeChecker
        parameter_handler = vsite_force_field["VirtualSites"]

        vsite_collection = VirtualSiteCollection.from_smirnoff(parameter_handler)
        assert len(vsite_collection.parameters) == len(parameter_handler.parameters)

        vsite_parameters = {
            parameter.type: parameter for parameter in vsite_collection.parameters
        }
        assert {*vsite_parameters} == {
            "BondCharge",
            "MonovalentLonePair",
            "DivalentLonePair",
            "TrivalentLonePair",
        }

        bond_charge: BondChargeSiteParameter = vsite_parameters["BondCharge"]
        assert bond_charge.smirks == "[#6:2]=[#8:1]"
        assert bond_charge.name == "EP"
        assert numpy.isclose(bond_charge.distance, 7.0)
        assert bond_charge.match == "all-permutations"
        assert len(bond_charge.charge_increments) == 2
        assert numpy.isclose(bond_charge.sigma, 1.0)
        assert numpy.isclose(bond_charge.epsilon, 2.0)

        monovalent: MonovalentLonePairParameter = vsite_parameters["MonovalentLonePair"]
        assert monovalent.smirks == "[#1:2]-[#8X2H2+0:1]-[#1:3]"
        assert monovalent.name == "EP"
        assert numpy.isclose(monovalent.distance, -0.106)
        assert monovalent.match == "all-permutations"
        assert len(monovalent.charge_increments) == 3
        assert numpy.isclose(monovalent.sigma, 0.0)
        assert numpy.isclose(monovalent.epsilon, 0.5)
        assert numpy.isclose(monovalent.out_of_plane_angle, 90.0)
        assert numpy.isclose(monovalent.in_plane_angle, 180.0)

        divalent: DivalentLonePairParameter = vsite_parameters["DivalentLonePair"]
        assert divalent.smirks == "[#1:2]-[#8X2H2+0:1]-[#1:3]"
        assert divalent.name == "EP"
        assert numpy.isclose(divalent.distance, -0.106)
        assert divalent.match == "all-permutations"
        assert len(divalent.charge_increments) == 3
        assert numpy.isclose(divalent.sigma, 1.0)
        assert numpy.isclose(divalent.epsilon, 0.5)
        assert numpy.isclose(divalent.out_of_plane_angle, 0.1)

        trivalent: TrivalentLonePairParameter = vsite_parameters["TrivalentLonePair"]
        assert trivalent.smirks == "[#1:2][#7:1]([#1:3])[#1:4]"
        assert trivalent.name == "EP"
        assert numpy.isclose(trivalent.distance, 5.0)
        assert trivalent.match == "once"
        assert len(trivalent.charge_increments) == 4
        assert numpy.isclose(trivalent.sigma, 1.0)
        assert numpy.isclose(trivalent.epsilon, 0.5)

    def test_smirnoff_parity(
        self, vsite_force_field: "ForceField", vsite_collection: VirtualSiteCollection
    ):
        import openmm.unit

        molecule = smiles_to_molecule("N")

        openmm_system = vsite_force_field.create_openmm_system(molecule.to_topology())
        openmm_force = [
            force
            for force in openmm_system.getForces()
            if isinstance(force, openmm.NonbondedForce)
        ][0]

        openff_charges = numpy.array(
            [
                openmm_force.getParticleParameters(i)[0].value_in_unit(
                    openmm.unit.elementary_charge
                )
                for i in range(openmm_force.getNumParticles())
            ]
        ).reshape(-1, 1)

        recharges = VirtualSiteGenerator.generate_charge_increments(
            molecule, vsite_collection
        )

        assert openff_charges.shape == recharges.shape
        assert numpy.allclose(openff_charges, recharges)

    def test_vectorize_charge_increments(self, vsite_collection):
        parameter_vector = vsite_collection.vectorize_charge_increments(
            parameter_keys=[
                ("[#6:2]=[#8:1]", "BondCharge", "EP", 1),
                ("[#1:2]-[#8X2H2+0:1]-[#1:3]", "DivalentLonePair", "EP", 1),
                ("[#1:2][#7:1]([#1:3])[#1:4]", "TrivalentLonePair", "EP", 0),
            ],
        )
        assert parameter_vector.shape == (3, 1)
        assert numpy.allclose(
            parameter_vector, numpy.array([[0.1], [1.0552 * 0.5], [0.2]])
        )

    def test_vectorize_coordinates(self, vsite_collection):
        parameter_vector = vsite_collection.vectorize_coordinates(
            parameter_keys=[
                ("[#1:2][#7:1]([#1:3])[#1:4]", "TrivalentLonePair", "EP", "distance"),
                (
                    "[#1:2]-[#8X2H2+0:1]-[#1:3]",
                    "MonovalentLonePair",
                    "EP",
                    "in_plane_angle",
                ),
                (
                    "[#1:2]-[#8X2H2+0:1]-[#1:3]",
                    "MonovalentLonePair",
                    "EP",
                    "out_of_plane_angle",
                ),
                ("[#6:2]=[#8:1]", "BondCharge", "EP", "distance"),
            ],
        )
        assert parameter_vector.shape == (4, 1)
        assert numpy.allclose(
            parameter_vector, numpy.array([[5.0], [180.0], [90.0], [7.0]])
        )


@pytest.mark.skip(
    reason="Virtual site code has not yet been refactored for version 0.11.0"
)
class TestVirtualSiteGenerator:
    def test_apply_virtual_sites(self, vsite_collection):
        molecule = smiles_to_molecule("N")

        molecule, assigned_vsite_keys = VirtualSiteGenerator._apply_virtual_sites(
            molecule, vsite_collection
        )

        assert molecule.n_virtual_sites == 1

        orientations = molecule.virtual_sites[0].orientations
        assert len(orientations) == 1

        assert assigned_vsite_keys == {
            orientations[0][0]: {
                ("[#1:2][#7:1]([#1:3])[#1:4]", "TrivalentLonePair", "EP"): [
                    (0, 1, 2, 3)
                ]
            }
        }

    def test_build_charge_array(self, vsite_collection):
        charge_values, charge_keys = VirtualSiteGenerator._build_charge_increment_array(
            vsite_collection
        )

        assert len(charge_values) == len(charge_keys)
        assert len(charge_values) == 2 + 3 + 3 + 4

        assert not numpy.allclose(charge_values, 0.0)
        assert charge_values.shape == (12,)

        vsite_smirks = set()

        for charge_key in charge_keys:
            assert len(charge_key) == 4
            vsite_smirks.add(charge_key[0])

        assert len(vsite_smirks) == 3

    @pytest.mark.parametrize(
        "assignment_matrix, expected_raises",
        [
            (numpy.array([[1, 2], [-1, -2]]), does_not_raise()),
            (
                numpy.array([[1, 2], [0, -2]]),
                pytest.raises(
                    ChargeAssignmentError,
                    match="The v-site charge increments alter the",
                ),
            ),
        ],
    )
    def test_validate_charge_assignment_matrix(
        self, assignment_matrix, expected_raises
    ):
        with expected_raises:
            VirtualSiteGenerator._validate_charge_assignment_matrix(assignment_matrix)

    @pytest.mark.parametrize(
        "smiles, expected_matrix",
        [
            (
                "C(=O)=O",
                numpy.hstack(
                    [
                        numpy.zeros((5, 10)),
                        numpy.array([[0, 2], [1, 0], [1, 0], [-1, -1], [-1, -1]]),
                    ]
                ),
            ),
            (
                "O",
                numpy.hstack(
                    [
                        # Two v-sites added because DivalentLonePair set to all permutations
                        numpy.zeros((5, 4)),
                        numpy.array(
                            [
                                [2, 0, 0],
                                [0, 1, 1],
                                [0, 1, 1],
                                [-1, -1, -1],
                                [-1, -1, -1],
                            ]
                        ),
                        numpy.zeros((5, 5)),
                    ]
                ),
            ),
            (
                "N",
                numpy.hstack(
                    [
                        numpy.array(
                            [
                                [1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1],
                                [-1, -1, -1, -1],
                            ]
                        ),
                        numpy.zeros((5, 8)),
                    ]
                ),
            ),
        ],
    )
    def test_charge_assignment_matrix(self, smiles, expected_matrix, vsite_collection):
        molecule = smiles_to_molecule(smiles)

        assignment_matrix = VirtualSiteGenerator.build_charge_assignment_matrix(
            molecule, vsite_collection
        )

        assert assignment_matrix.shape == expected_matrix.shape
        assert numpy.allclose(assignment_matrix, expected_matrix)

    @pytest.mark.parametrize(
        "smiles, expected_increments",
        [
            ("C(=O)=O", numpy.array([[0.2], [0.2], [0.2], [-0.3], [-0.3]])),
            ("O", numpy.array([[0.0], [1.0552], [1.0552], [-1.0552], [-1.0552]])),
            ("N", numpy.array([[0.2], [0.0], [0.0], [0.0], [-0.2]])),
        ],
    )
    def test_generate_charge_increments(
        self, smiles, expected_increments, vsite_collection
    ):
        molecule = smiles_to_molecule(smiles)

        actual_increments = VirtualSiteGenerator.generate_charge_increments(
            molecule, vsite_collection
        )

        assert actual_increments.shape == expected_increments.shape
        assert numpy.allclose(actual_increments, expected_increments)

    def test_local_coordinate_frames(self):
        # Build a dummy 'formaldehyde' conformer and associated v-site parameter.
        conformer = numpy.array(
            [
                [+1.0, +0.0, +0.0],
                [+0.0, +0.0, +0.0],
                [-1.0, +0.0, +1.0],
                [-1.0, +0.0, -1.0],
            ]
        )
        parameter = MonovalentLonePairParameter(
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

        assigned_parameters = [(parameter, [(0, 1, 2), (0, 1, 3)])]

        actual_coordinate_frames = VirtualSiteGenerator.build_local_coordinate_frames(
            conformer, assigned_parameters
        )
        expected_coordinate_frames = numpy.array(
            [
                [[+1.0, +0.0, +0.0], [+1.0, +0.0, +0.0]],
                [[-1.0, +0.0, +0.0], [-1.0, +0.0, +0.0]],
                [[+0.0, +0.0, +1.0], [+0.0, +0.0, -1.0]],
                [[+0.0, +1.0, +0.0], [+0.0, -1.0, +0.0]],
            ]
        )

        assert actual_coordinate_frames.shape == (4, 2, 3)
        assert numpy.allclose(actual_coordinate_frames, expected_coordinate_frames)

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_convert_local_coordinates(self, backend):
        local_frame_coordinates = numpy.array([[1.0, 45.0, 45.0]])
        local_coordinate_frames = numpy.array(
            [
                [[+0.0, +0.0, +0.0]],
                [[+1.0, +0.0, +0.0]],
                [[+0.0, +1.0, +0.0]],
                [[+0.0, +0.0, +1.0]],
            ]
        )

        if backend == "torch":
            pytest.importorskip("torch")
            import torch

            local_coordinate_frames = torch.from_numpy(local_coordinate_frames)
            local_frame_coordinates = torch.from_numpy(local_frame_coordinates)

        actual_coordinates = VirtualSiteGenerator.convert_local_coordinates(
            local_frame_coordinates, local_coordinate_frames, backend
        )

        if backend == "numpy":
            assert isinstance(actual_coordinates, numpy.ndarray)
        else:
            assert isinstance(actual_coordinates, torch.Tensor)
            actual_coordinates = actual_coordinates.numpy()

        expected_coordinates = numpy.array([[0.5, 0.5, 1.0 / numpy.sqrt(2.0)]])

        assert actual_coordinates.shape == (1, 3)
        assert numpy.allclose(actual_coordinates, expected_coordinates)

    def test_generate_positions(self, vsite_collection):
        molecule = smiles_to_molecule("N")

        conformer = (
            numpy.array(
                [
                    [0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.5, 0.0, +numpy.sqrt(0.75)],
                    [0.5, 0.0, -numpy.sqrt(0.75)],
                ]
            )
            * unit.angstrom
        )

        vsite_position = VirtualSiteGenerator.generate_positions(
            molecule, vsite_collection, conformer
        )

        assert vsite_position.shape == (1, 3)
        assert numpy.allclose(
            vsite_position, numpy.array([0.0, 6.0, 0.0]) * unit.angstrom
        )

    def test_generate_multiple_positions(self):
        vsite_collection = VirtualSiteCollection(
            parameters=[
                BondChargeSiteParameter(
                    smirks="[#6A:2]-[#17:1]",
                    name="EP1",
                    distance=0.5,
                    charge_increments=(0.05, 0.0),
                    sigma=0.0,
                    epsilon=0.0,
                    match="all-permutations",
                ),
                BondChargeSiteParameter(
                    smirks="[#6A:2]-[#17:1]",
                    name="EP2",
                    distance=2.0,
                    charge_increments=(-0.05, 0.0),
                    sigma=0.0,
                    epsilon=0.0,
                    match="all-permutations",
                ),
                BondChargeSiteParameter(
                    smirks="[#6A:2]-[#35:1]",
                    name="EP1",
                    distance=1.0,
                    charge_increments=(0.05, 0.0),
                    sigma=0.0,
                    epsilon=0.0,
                    match="all-permutations",
                ),
                BondChargeSiteParameter(
                    smirks="[#6A:2]-[#35:1]",
                    name="EP2",
                    distance=3.0,
                    charge_increments=(-0.05, 0.0),
                    sigma=0.0,
                    epsilon=0.0,
                    match="all-permutations",
                ),
            ]
        )

        molecule = Molecule.from_mapped_smiles("[C:1]([Cl:2])([Cl:3])([Br:4])([Br:5])")

        conformer = (
            numpy.array(
                [
                    [+0.0, +0.0, 0.0],
                    [-1.0, +0.0, 0.0],
                    [+0.0, -1.0, 0.0],
                    [+1.0, +0.0, 0.0],
                    [+0.0, +1.0, 0.0],
                ]
            )
            * unit.angstrom
        )

        actual_vsite_positions = VirtualSiteGenerator.generate_positions(
            molecule, vsite_collection, conformer
        )
        expected_vsite_positions = (
            numpy.array(
                [
                    [+4.0, +0.0, 0.0],
                    [+0.0, +4.0, 0.0],
                    [+2.0, +0.0, 0.0],
                    [+0.0, +2.0, 0.0],
                    [-3.0, +0.0, 0.0],
                    [+0.0, -3.0, 0.0],
                    [-1.5, +0.0, 0.0],
                    [+0.0, -1.5, 0.0],
                ]
            )
            * unit.angstrom
        )

        assert actual_vsite_positions.shape == expected_vsite_positions.shape
        assert numpy.allclose(actual_vsite_positions, expected_vsite_positions)
