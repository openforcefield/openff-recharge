import os
import subprocess
from typing import Tuple

import jinja2
import numpy
from openeye import oechem

from openff.recharge.esp import ESPGenerator, ESPSettings
from openff.recharge.esp.exceptions import Psi4Error
from openff.recharge.utilities import get_data_file_path, temporary_cd


class Psi4ESPGenerator(ESPGenerator):
    """An class which will compute the electrostatic potential of
    a molecule using Psi4.
    """

    @classmethod
    def _generate_input(
        cls, oe_molecule: oechem.OEMol, conformer: numpy.ndarray, settings: ESPSettings
    ) -> str:
        """Generate the input files for Psi4.

        Parameters
        ----------
        oe_molecule
            The molecule to generate the ESP for.
        conformer
            The conformer of the molecule to generate the ESP for.
        settings
            The settings to use when generating the ESP.

        Returns
        -------
            The contents of the input file.
        """

        # Compute the total formal charge on the molecule.
        formal_charge = sum(atom.GetFormalCharge() for atom in oe_molecule.GetAtoms())

        # Compute the spin multiplicity
        total_atomic_number = sum(
            atom.GetAtomicNum() for atom in oe_molecule.GetAtoms()
        )

        spin_multiplicity = 1 if (formal_charge + total_atomic_number) % 2 == 0 else 2

        # Store the atoms and coordinates in a jinja friendly dict.
        atoms = [
            {
                "element": oechem.OEGetAtomicSymbol(atom.GetAtomicNum()),
                "x": conformer[atom.GetIdx(), 0],
                "y": conformer[atom.GetIdx(), 1],
                "z": conformer[atom.GetIdx(), 2],
            }
            for atom in oe_molecule.GetAtoms()
        ]

        # Format the jinja template
        template_path = get_data_file_path(os.path.join("psi4", "input.dat"))

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
        oe_molecule: oechem.OEMol,
        conformer: numpy.ndarray,
        grid: numpy.ndarray,
        settings: ESPSettings,
        directory: str = None,
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:

        # Perform the calculation in a temporary directory
        with temporary_cd(directory):

            # Store the input file.
            input_contents = cls._generate_input(oe_molecule, conformer, settings)

            with open("input.dat", "w") as file:
                file.write(input_contents)

            # Store the grid to file.
            # noinspection PyTypeChecker
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

            esp = numpy.loadtxt("grid_esp.dat").reshape(-1, 1)
            electric_field = numpy.loadtxt("grid_field.dat").reshape(-1, 1)

        return esp, electric_field
