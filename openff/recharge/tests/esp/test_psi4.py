from typing import List

import numpy
import pytest

from openff.recharge.esp import DFTGridSettings, ESPSettings, PCMSettings
from openff.recharge.esp.exceptions import Psi4Error
from openff.recharge.esp.psi4 import Psi4ESPGenerator
from openff.recharge.grids import GridSettings
from openff.recharge.utilities.openeye import smiles_to_molecule


def test_generate_input_base():
    """Test that the correct input is generated from the
    jinja template."""
    pytest.importorskip("psi4")

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
            "set {",
            "  basis 6-31g*",
            "}",
            "",
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
            "set {",
            "  basis 6-31g*",
            "}",
            "",
            "E,wfn = prop('uhf', properties = ['GRID_ESP', 'GRID_FIELD'], "
            "return_wfn=True)",
        ]
    )

    assert expected_output == input_contents


@pytest.mark.parametrize(
    "dft_grid_settings, expected_grid_settings",
    [
        (DFTGridSettings.Default, []),
        (
            DFTGridSettings.Medium,
            [
                "",
                "  dft_spherical_points 434",
                "  dft_radial_points 85",
                "  dft_pruning_scheme robust",
            ],
        ),
        (
            DFTGridSettings.Fine,
            [
                "",
                "  dft_spherical_points 590",
                "  dft_radial_points 99",
                "  dft_pruning_scheme robust",
            ],
        ),
    ],
)
def test_generate_input_dft_settings(
    dft_grid_settings: DFTGridSettings, expected_grid_settings: List[str]
):
    """Test that the correct input is generated from the
    jinja template."""
    pytest.importorskip("psi4")

    # Define the settings to use.
    settings = ESPSettings(
        psi4_dft_grid_settings=dft_grid_settings, grid_settings=GridSettings()
    )

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
            "set {",
            "  basis 6-31g*",
            *expected_grid_settings,
            "}",
            "",
            "E,wfn = prop('hf', properties = ['GRID_ESP', 'GRID_FIELD'], "
            "return_wfn=True)",
        ]
    )

    assert expected_output == input_contents


def test_generate_input_pcm():
    """Test that the correct input is generated from the
    jinja template."""
    pytest.importorskip("psi4")

    # Define the settings to use.
    settings = ESPSettings(pcm_settings=PCMSettings(), grid_settings=GridSettings())

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
            "set {",
            "  basis 6-31g*",
            "",
            "  pcm true",
            "  pcm_scf_type total",
            "}",
            "pcm = {",
            "  Units = Angstrom",
            "  Medium {",
            "  SolverType = CPCM",
            "  Solvent = Water",
            "  }",
            "",
            "  Cavity {",
            "  RadiiSet = Bondi",
            "  Type = GePol",
            "  Scaling = True",
            "  Area = 0.3",
            "  Mode = Implicit",
            "  }",
            "}",
            "",
            "E,wfn = prop('hf', properties = ['GRID_ESP', 'GRID_FIELD'], "
            "return_wfn=True)",
        ]
    )

    assert expected_output == input_contents


@pytest.mark.parametrize("enable_pcm", [False, True])
def test_generate(enable_pcm):
    """Perform a test run of Psi4."""
    pytest.importorskip("psi4")

    # Define the settings to use.
    settings = ESPSettings(grid_settings=GridSettings(spacing=2.0))

    if enable_pcm:
        settings.pcm_settings = PCMSettings()

    # Generate a small molecule which should finish fast.
    oe_molecule = smiles_to_molecule("C")

    conformer = numpy.array(
        [
            [-0.0000658, -0.0000061, 0.0000215],
            [-0.0566733, 1.0873573, -0.0859463],
            [0.6194599, -0.3971111, -0.8071615],
            [-1.0042799, -0.4236047, -0.0695677],
            [0.4415590, -0.2666354, 0.9626540],
        ]
    )

    grid, esp, electric_field = Psi4ESPGenerator.generate(
        oe_molecule, conformer, settings
    )

    assert grid.shape[0] > 0
    assert grid.shape[1] == 3

    assert esp.shape == (len(grid), 1)
    assert electric_field.shape == (len(grid), 3)

    assert not numpy.all(numpy.isclose(esp, 0.0))
    assert not numpy.all(numpy.isclose(electric_field, 0.0))


def test_ps4_error():
    """Tests that the correct custom error is raised when Psi4
    fails to run"""
    pytest.importorskip("psi4")

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
