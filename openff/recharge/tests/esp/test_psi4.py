import numpy
import pytest

from openff.recharge.esp import ESPSettings
from openff.recharge.esp.exceptions import Psi4Error
from openff.recharge.esp.psi4 import Psi4ESPGenerator
from openff.recharge.grids import GridSettings
from openff.recharge.utilities.openeye import smiles_to_molecule


def test_generate_input():
    """Test that the correct input is generated from the
    jinja template."""

    # Define the settings to use.
    settings = ESPSettings(grid_settings=GridSettings())

    # Create a closed shell molecule.
    oe_molecule = smiles_to_molecule("[Cl-]")
    conformer = numpy.array([[0.0, 0.0, 0.0]])

    input_contents = Psi4ESPGenerator._generate_input(oe_molecule, conformer, settings)

    expected_output = "\n".join(
        [
            "molecule mol {",
            "  noreorient",
            "  nocom",
            "  -1 1",
            "  Cl  0.000000000  0.000000000  0.000000000",
            "}",
            "",
            "set basis 6-31g*",
            "E,wfn = prop('hf', properties = ['GRID_ESP', 'GRID_FIELD'], "
            "return_wfn=True)",
        ]
    )

    assert expected_output == input_contents

    # Create an open shell molecule.
    oe_molecule = smiles_to_molecule("[B]")
    conformer = numpy.array([[0.0, 0.0, 0.0]])

    input_contents = Psi4ESPGenerator._generate_input(oe_molecule, conformer, settings)

    expected_output = "\n".join(
        [
            "molecule mol {",
            "  noreorient",
            "  nocom",
            "  0 2",
            "  B  0.000000000  0.000000000  0.000000000",
            "}",
            "",
            "set basis 6-31g*",
            "E,wfn = prop('uhf', properties = ['GRID_ESP', 'GRID_FIELD'], "
            "return_wfn=True)",
        ]
    )

    assert expected_output == input_contents


def test_generate():
    """Perform a test run of Psi4."""

    # Define the settings to use.
    settings = ESPSettings(grid_settings=GridSettings(spacing=2.0))

    # Generate a small molecule which should finish fast.
    oe_molecule = smiles_to_molecule("C")
    oe_molecule.DeleteConfs()

    conformer = numpy.array(
        [
            [-0.0000658, -0.0000061, 0.0000215],
            [-0.0566733, 1.0873573, -0.0859463],
            [0.6194599, -0.3971111, -0.8071615],
            [-1.0042799, -0.4236047, -0.0695677],
            [0.4415590, -0.2666354, 0.9626540],
        ]
    )

    grid, esp = Psi4ESPGenerator.generate(oe_molecule, conformer, settings)

    assert len(grid) == len(esp)
    assert len(grid) > 0

    assert not numpy.all(numpy.isclose(esp, 0.0))


def test_ps4_error():
    """Tests that the correct custom error is raised when Psi4
    fails to run"""

    # Define the settings to use.
    settings = ESPSettings(grid_settings=GridSettings(spacing=2.0))

    # Generate a small molecule with an invalid conformer.
    oe_molecule = smiles_to_molecule("C")
    conformer = numpy.zeros((5, 3))

    with pytest.raises(Psi4Error) as error_info:
        Psi4ESPGenerator.generate(oe_molecule, conformer, settings)

    error_message = str(error_info.value)

    assert len(error_message) > 0
    assert "StdOut" in error_message
    assert "StdErr" in error_message
