from typing import List

import numpy
import pytest
from openff.units import unit

from openff.recharge.esp import DFTGridSettings, ESPSettings, PCMSettings
from openff.recharge.esp.exceptions import Psi4Error
from openff.recharge.esp.psi4 import Psi4ESPGenerator
from openff.recharge.grids import LatticeGridSettings
from openff.recharge.utilities.molecule import smiles_to_molecule


@pytest.mark.parametrize(
    "compute_esp, compute_field, expected_properties",
    [
        (False, False, "[]"),
        (True, False, "['GRID_ESP']"),
        (False, True, "['GRID_FIELD']"),
        (True, True, "['GRID_ESP', 'GRID_FIELD']"),
    ],
)
def test_generate_input_base(compute_esp, compute_field, expected_properties):
    """Test that the correct input is generated from the
    jinja template."""
    pytest.importorskip("psi4")

    # Define the settings to use.
    settings = ESPSettings(grid_settings=LatticeGridSettings())

    # Create a closed shell molecule.
    molecule = smiles_to_molecule("[Cl-]")
    conformer = numpy.array([[0.0, 0.0, 0.0]]) * unit.angstrom

    input_contents = Psi4ESPGenerator._generate_input(
        molecule, conformer, settings, False, compute_esp, compute_field
    )

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
            f"E,wfn = prop('hf', properties = {expected_properties}, "
            "return_wfn=True)",
            "mol.save_xyz_file('final-geometry.xyz',1)",
        ]
    )

    assert expected_output == input_contents

    # Create an open shell molecule.
    molecule = smiles_to_molecule("[B]")
    conformer = numpy.array([[0.0, 0.0, 0.0]]) * unit.angstrom

    input_contents = Psi4ESPGenerator._generate_input(
        molecule, conformer, settings, True, compute_esp, compute_field
    )

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
            "optimize('scf')",
            "",
            f"E,wfn = prop('uhf', properties = {expected_properties}, "
            "return_wfn=True)",
            "mol.save_xyz_file('final-geometry.xyz',1)",
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
        psi4_dft_grid_settings=dft_grid_settings, grid_settings=LatticeGridSettings()
    )

    # Create a closed shell molecule.
    molecule = smiles_to_molecule("[Cl-]")
    conformer = numpy.array([[0.0, 0.0, 0.0]]) * unit.angstrom

    input_contents = Psi4ESPGenerator._generate_input(
        molecule, conformer, settings, False, True, True
    )

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
            "mol.save_xyz_file('final-geometry.xyz',1)",
        ]
    )

    assert expected_output == input_contents


def test_generate_input_pcm():
    """Test that the correct input is generated from the
    jinja template."""
    pytest.importorskip("psi4")

    # Define the settings to use.
    settings = ESPSettings(
        pcm_settings=PCMSettings(), grid_settings=LatticeGridSettings()
    )

    # Create a closed shell molecule.
    molecule = smiles_to_molecule("[Cl-]")
    conformer = numpy.array([[0.1, 0.0, 0.0]]) * unit.nanometer

    input_contents = Psi4ESPGenerator._generate_input(
        molecule, conformer, settings, False, True, True
    )

    expected_output = "\n".join(
        [
            "molecule mol {",
            "  noreorient",
            "  nocom",
            "  -1 1",
            "  Cl  1.000000000  0.000000000  0.000000000",
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
            "mol.save_xyz_file('final-geometry.xyz',1)",
        ]
    )

    assert expected_output == input_contents


@pytest.mark.parametrize("enable_pcm, minimize", [(False, True), (True, False)])
def test_generate(enable_pcm, minimize):
    """Perform a test run of Psi4."""
    pytest.importorskip("psi4")

    # Define the settings to use.
    settings = ESPSettings(grid_settings=LatticeGridSettings(spacing=2.0))

    if enable_pcm:
        settings.pcm_settings = PCMSettings()

    # Generate a small molecule which should finish fast.
    molecule = smiles_to_molecule("C")

    input_conformer = (
        numpy.array(
            [
                [-0.0000658, -0.0000061, 0.0000215],
                [-0.0566733, 1.0873573, -0.0859463],
                [0.6194599, -0.3971111, -0.8071615],
                [-1.0042799, -0.4236047, -0.0695677],
                [0.4415590, -0.2666354, 0.9626540],
            ]
        )
        * unit.angstrom
    )

    output_conformer, grid, esp, electric_field = Psi4ESPGenerator.generate(
        molecule, input_conformer, settings, minimize=minimize
    )

    assert grid.shape[0] > 0
    assert grid.shape[1] == 3

    assert esp.shape == (len(grid), 1)
    assert electric_field.shape == (len(grid), 3)

    assert output_conformer.shape == input_conformer.shape
    assert numpy.allclose(output_conformer, input_conformer) != minimize

    assert not numpy.all(numpy.isclose(esp, 0.0))
    assert not numpy.all(numpy.isclose(electric_field, 0.0))


def test_generate_no_properties():
    """Perform a test run of Psi4."""
    pytest.importorskip("psi4")

    # Define the settings to use.
    settings = ESPSettings(grid_settings=LatticeGridSettings(spacing=2.0))

    # Generate a small molecule which should finish fast.
    molecule = smiles_to_molecule("C")

    input_conformer = (
        numpy.array(
            [
                [-0.0000658, -0.0000061, 0.0000215],
                [-0.0566733, 1.0873573, -0.0859463],
                [0.6194599, -0.3971111, -0.8071615],
                [-1.0042799, -0.4236047, -0.0695677],
                [0.4415590, -0.2666354, 0.9626540],
            ]
        )
        * unit.angstrom
    )

    output_conformer, grid, esp, electric_field = Psi4ESPGenerator.generate(
        molecule,
        input_conformer,
        settings,
        minimize=False,
        compute_esp=False,
        compute_field=False,
    )

    assert esp is None
    assert electric_field is None


def test_ps4_error():
    """Tests that the correct custom error is raised when Psi4
    fails to run"""
    pytest.importorskip("psi4")

    # Define the settings to use.
    settings = ESPSettings(grid_settings=LatticeGridSettings(spacing=2.0))

    # Generate a small molecule with an invalid conformer.
    molecule = smiles_to_molecule("C")
    conformer = numpy.zeros((5, 3)) * unit.angstrom

    with pytest.raises(Psi4Error) as error_info:
        Psi4ESPGenerator.generate(molecule, conformer, settings)

    error_message = str(error_info.value)

    assert len(error_message) > 0
    assert "StdOut" in error_message
    assert "StdErr" in error_message
