"""Compute ESP and electric field data using Psi4"""
import os
import subprocess
from typing import TYPE_CHECKING, Tuple

import jinja2
import numpy
from openff.units import unit
from openff.utilities import get_data_file_path, temporary_cd

from openff.recharge.esp import ESPGenerator, ESPSettings
from openff.recharge.esp.exceptions import Psi4Error

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule


class Psi4ESPGenerator(ESPGenerator):
    """An class which will compute the electrostatic potential of
    a molecule using Psi4.
    """

    @classmethod
    def _generate_input(
        cls,
        molecule: "Molecule",
        conformer: unit.Quantity,
        settings: ESPSettings,
    ) -> str:
        """Generate the input files for Psi4.

        Parameters
        ----------
        molecule
            The molecule to generate the ESP for.
        conformer
            The conformer of the molecule to generate the ESP for.
        settings
            The settings to use when generating the ESP.

        Returns
        -------
            The contents of the input file.
        """

        from simtk import unit as simtk_unit

        # Compute the total formal charge on the molecule.
        formal_charge = sum(
            atom.formal_charge.value_in_unit(simtk_unit.elementary_charge)
            for atom in molecule.atoms
        )

        # Compute the spin multiplicity
        total_atomic_number = sum(atom.atomic_number for atom in molecule.atoms)
        spin_multiplicity = 1 if (formal_charge + total_atomic_number) % 2 == 0 else 2

        # Store the atoms and coordinates in a jinja friendly dict.
        conformer = conformer.to(unit.angstrom).m

        atoms = [
            {
                "element": atom.element.symbol,
                "x": conformer[index, 0],
                "y": conformer[index, 1],
                "z": conformer[index, 2],
            }
            for index, atom in enumerate(molecule.atoms)
        ]

        # Format the jinja template
        template_path = get_data_file_path(
            os.path.join("psi4", "input.dat"), "openff.recharge"
        )

        with open(template_path) as file:
            template = jinja2.Template(file.read())

        enable_pcm = settings.pcm_settings is not None

        template_inputs = {
            "charge": formal_charge,
            "spin": spin_multiplicity,
            "atoms": atoms,
            "basis": settings.basis,
            "method": settings.method,
            "enable_pcm": enable_pcm,
            "dft_settings": settings.psi4_dft_grid_settings.value,
        }

        if enable_pcm:

            template_inputs.update(
                {
                    "pcm_solver": settings.pcm_settings.solver,
                    "pcm_solvent": settings.pcm_settings.solvent,
                    "pcm_radii_set": settings.pcm_settings.radii_model,
                    "pcm_scaling": settings.pcm_settings.radii_scaling,
                    "pcm_area": settings.pcm_settings.cavity_area,
                }
            )

        rendered_template = template.render(template_inputs)
        # Remove the white space after the for loop
        rendered_template = rendered_template.replace("  \n}", "}")

        return rendered_template

    @classmethod
    def _generate(
        cls,
        molecule: "Molecule",
        conformer: unit.Quantity,
        grid: unit.Quantity,
        settings: ESPSettings,
        directory: str = None,
    ) -> Tuple[unit.Quantity, unit.Quantity]:

        # Perform the calculation in a temporary directory
        with temporary_cd(directory):

            # Store the input file.
            input_contents = cls._generate_input(molecule, conformer, settings)

            with open("input.dat", "w") as file:
                file.write(input_contents)

            # Store the grid to file.
            grid = grid.to(unit.angstrom).m
            numpy.savetxt("grid.dat", grid, delimiter=" ", fmt="%16.10f")

            # Attempt to run the calculation
            psi4_process = subprocess.Popen(
                ["psi4", "input.dat", "output.dat"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            std_output, std_error = psi4_process.communicate()
            exit_code = psi4_process.returncode

            if exit_code != 0:
                raise Psi4Error(std_output.decode(), std_error.decode())

            esp = numpy.loadtxt("grid_esp.dat").reshape(-1, 1) * unit.hartree / unit.e
            electric_field = (
                numpy.loadtxt("grid_field.dat") * unit.hartree / (unit.e * unit.bohr)
            )

        return esp, electric_field
