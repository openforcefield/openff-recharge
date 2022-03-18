import numpy
import pytest

from openff.recharge.charges.exceptions import ChargeAssignmentError
from openff.recharge.charges.library import (
    LibraryChargeCollection,
    LibraryChargeGenerator,
    LibraryChargeParameter,
)


@pytest.fixture()
def mock_charge_collection() -> LibraryChargeCollection:
    return LibraryChargeCollection(
        parameters=[
            LibraryChargeParameter(smiles="[Cl:1][Cl:2]", value=[-0.1, 0.1]),
            LibraryChargeParameter(smiles="[H:1][O:2][H:3]", value=[0.3, 0.4, -0.7]),
        ]
    )


class TestLibraryChargeCollection:
    def test_to_smirnoff(self, mock_charge_collection):

        pytest.importorskip("openff.toolkit")

        from simtk import unit

        charge_handler = mock_charge_collection.to_smirnoff()

        assert len(charge_handler.parameters) == 2

        assert charge_handler.parameters[0].smirks == "[H:1][O:2][H:3]"
        assert numpy.allclose(
            [
                charge.value_in_unit(unit.elementary_charge)
                for charge in charge_handler.parameters[0].charge
            ],
            [0.3, 0.4, -0.7],
        )

        assert charge_handler.parameters[-1].smirks == "[Cl:1][Cl:2]"
        assert numpy.allclose(
            [
                charge.value_in_unit(unit.elementary_charge)
                for charge in charge_handler.parameters[-1].charge
            ],
            [-0.1, 0.1],
        )

    def test_from_smirnoff(self):

        pytest.importorskip("openff.toolkit")

        from openff.toolkit.typing.engines.smirnoff import LibraryChargeHandler
        from simtk import unit

        # noinspection PyTypeChecker
        charge_handler = LibraryChargeHandler(version="0.3")
        charge_handler.add_parameter(
            {
                "smirks": "[H:1][O:2][H:3]",
                "charge": [
                    0.0 * unit.elementary_charge,
                    0.1 * unit.elementary_charge,
                    0.2 * unit.elementary_charge,
                ],
            }
        )
        charge_handler.add_parameter(
            {
                "smirks": "[Cl:1][Cl:2]",
                "charge": [0.3 * unit.elementary_charge, 0.4 * unit.elementary_charge],
            }
        )

        charge_collection = LibraryChargeCollection.from_smirnoff(charge_handler)

        assert len(charge_collection.parameters) == 2

        assert charge_collection.parameters[0].smiles == "[Cl:1][Cl:2]"
        assert numpy.allclose(charge_collection.parameters[0].value, [0.3, 0.4])

        assert charge_collection.parameters[-1].smiles == "[H:1][O:2][H:3]"
        assert numpy.allclose(charge_collection.parameters[-1].value, [0.0, 0.1, 0.2])

    @pytest.mark.parametrize(
        "keys, expected_value",
        [
            ([("[Cl:1][Cl:2]", (0, 1))], numpy.array([[-0.1], [0.1]])),
            (
                [("[Cl:1][Cl:2]", (1, 0)), ("[H:1][O:2][H:3]", (0, 1, 2))],
                numpy.array([[0.1], [-0.1], [0.3], [0.4], [-0.7]]),
            ),
            (
                [("[H:1][O:2][H:3]", (0, 2, 1)), ("[Cl:1][Cl:2]", (0, 1))],
                numpy.array([[0.3], [-0.7], [0.4], [-0.1], [0.1]]),
            ),
        ],
    )
    def test_vectorize(self, mock_charge_collection, keys, expected_value):

        charge_vector = mock_charge_collection.vectorize(keys)

        assert charge_vector.shape == expected_value.shape
        assert numpy.allclose(charge_vector, expected_value)


class TestLibraryChargeGenerator:
    def test_validate_assignment_matrix(self, mock_charge_collection):

        from openff.toolkit.topology import Molecule

        molecule = Molecule.from_mapped_smiles("[H:1][O:2][H:3]")

        with pytest.raises(ChargeAssignmentError, match="charges yield a total charge"):

            LibraryChargeGenerator._validate_assignment_matrix(
                molecule,
                numpy.array([[0, 0, 1, 2, 3], [0, 0, 4, 5, 6], [0, 0, 7, 8, 9]]),
                mock_charge_collection,
            )

    @pytest.mark.parametrize(
        "smiles, expected_value",
        [
            ("[Cl:1][Cl:2]", numpy.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])),
            (
                "[H:1][O:2][H:3]",
                numpy.array([[0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]),
            ),
            (
                "[H:2][O:1][H:3]",
                numpy.array([[0, 0, 0, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 1]]),
            ),
        ],
    )
    def test_build_assignment_matrix(
        self, mock_charge_collection, smiles, expected_value
    ):

        pytest.importorskip("openff.toolkit")

        from openff.toolkit.topology import Molecule

        molecule = Molecule.from_mapped_smiles(smiles)

        assignment_matrix = LibraryChargeGenerator.build_assignment_matrix(
            molecule, mock_charge_collection
        )

        assert expected_value.shape == assignment_matrix.shape
        assert numpy.allclose(expected_value, assignment_matrix)

    def test_build_assignment_matrix_equivalent_atoms(self):

        from openff.toolkit.topology import Molecule

        molecule = Molecule.from_mapped_smiles("[C:1]([H:3])([H:4])([H:5])[O:2][H:6]")

        charge_collection = LibraryChargeCollection(
            parameters=[
                LibraryChargeParameter(
                    smiles="[C:1]([H:2])([H:2])([H:2])[O:3][H:4]",
                    value=[0.15, -0.05, -0.2, 0.2],
                ),
            ]
        )

        assignment_matrix = LibraryChargeGenerator.build_assignment_matrix(
            molecule, charge_collection
        )

        assert assignment_matrix.shape == (6, 4)
        assert numpy.allclose(
            assignment_matrix,
            numpy.array(
                [
                    [1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 1, 0, 0],
                    [0, 1, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                ]
            ),
        )

    @pytest.mark.parametrize(
        "smiles, expected_value",
        [
            ("[Cl:1][Cl:2]", numpy.array([[-0.1], [0.1]])),
            ("[H:1][O:2][H:3]", numpy.array([[0.3], [0.4], [-0.7]])),
            ("[H:2][O:1][H:3]", numpy.array([[0.4], [0.3], [-0.7]])),
        ],
    )
    def test_generate(self, mock_charge_collection, smiles, expected_value):

        pytest.importorskip("openff.toolkit")

        from openff.toolkit.topology import Molecule

        molecule = Molecule.from_mapped_smiles(smiles)

        actual_charges = LibraryChargeGenerator.generate(
            molecule, mock_charge_collection
        )

        assert expected_value.shape == actual_charges.shape
        assert numpy.allclose(expected_value, actual_charges)
