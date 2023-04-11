import os.path
from glob import glob

import numpy
from openff.toolkit.topology import Molecule
from openff.units import unit
from simtk import unit as simtk_unit

from openff.recharge.charges.library import (
    LibraryChargeCollection,
    LibraryChargeGenerator,
)
from openff.recharge.charges.resp import generate_resp_charge_parameter
from openff.recharge.charges.resp.solvers import IterativeSolver
from openff.recharge.esp import ESPSettings
from openff.recharge.esp.storage import MoleculeESPRecord
from openff.recharge.grids import MSKGridSettings


def main():

    qc_data_settings = ESPSettings(
        method="hf", basis="6-31G*", grid_settings=MSKGridSettings()
    )

    respyte_outputs = glob(os.path.join("respyte-data", "output-*"))

    for _, respyte_output in enumerate(respyte_outputs):

        molecule = Molecule.from_file(os.path.join(respyte_output, "mol1_conf1.mol2"))

        respyte_charges = numpy.round(
            numpy.array(
                molecule.partial_charges.value_in_unit(simtk_unit.elementary_charge)
            ),
            4,
        )

        qc_data_record = MoleculeESPRecord.from_molecule(
            molecule,
            numpy.loadtxt(os.path.join(respyte_output, "xyz.txt")) * unit.bohr,
            numpy.loadtxt(os.path.join(respyte_output, "grid.txt")) * unit.bohr,
            numpy.loadtxt(os.path.join(respyte_output, "esp.txt")).reshape(-1, 1)
            * unit.hartree
            / unit.e,
            None,
            qc_data_settings,
        )

        resp_charge_parameter = generate_resp_charge_parameter(
            [qc_data_record], IterativeSolver()
        )
        resp_charges = LibraryChargeGenerator.generate(
            molecule, LibraryChargeCollection(parameters=[resp_charge_parameter])
        )
        resp_charges = numpy.round(resp_charges.flatten(), 4)

        def float_to_str(x):
            return f"{x:.4f}"

        print(molecule.to_smiles(isomeric=False, explicit_hydrogens=True, mapped=True))
        print("\n")
        print(f"RESPYTE CHARGES: {' '.join(map(float_to_str, respyte_charges))}")
        print(f"RESP    CHARGES: {' '.join(map(float_to_str, resp_charges))}")
        print("\n")
        print(f"RESPYTE SUM: {respyte_charges.sum():.4f}")
        print(f"RESP    SUM: {resp_charges.sum():.4f}")
        print("\n")


if __name__ == "__main__":
    main()
