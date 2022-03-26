import numpy
from openff.toolkit.topology import Molecule

from openff.recharge.charges.library import (
    LibraryChargeCollection,
    LibraryChargeGenerator,
)
from openff.recharge.charges.resp import generate_resp_charge_parameter
from openff.recharge.charges.resp.solvers import IterativeSolver
from openff.recharge.esp import ESPSettings
from openff.recharge.esp.psi4 import Psi4ESPGenerator
from openff.recharge.esp.storage import MoleculeESPRecord
from openff.recharge.grids import MSKGridSettings
from openff.recharge.utilities.molecule import extract_conformers


def main():

    qc_data_settings = ESPSettings(
        method="hf", basis="6-31G*", grid_settings=MSKGridSettings()
    )

    molecule: Molecule = Molecule.from_mapped_smiles(
        "[C:1]([H:5])([H:6])([H:8])[C:2]([H:7])([H:9])[O:3][H:4]"
    )
    molecule.generate_conformers(n_conformers=1)

    [input_conformer] = extract_conformers(molecule)

    conformer, grid, esp, electric_field = Psi4ESPGenerator.generate(
        molecule, input_conformer, qc_data_settings, minimize=True
    )

    qc_data_record = MoleculeESPRecord.from_molecule(
        molecule, conformer, grid, esp, None, qc_data_settings
    )

    resp_solver = IterativeSolver()
    # While by default the iterative approach to finding the set of charges that minimize
    # the RESP loss function as described in the original papers is used, others such as
    # on that calls out to SciPy are available, e.g.
    # resp_solver = SciPySolver(method="SLSQP")

    resp_charge_parameter = generate_resp_charge_parameter(
        [qc_data_record], resp_solver
    )
    resp_charges = LibraryChargeGenerator.generate(
        molecule, LibraryChargeCollection(parameters=[resp_charge_parameter])
    )

    print(f"RESP SMILES         : {resp_charge_parameter.smiles}")
    print(f"RESP VALUES (UNIQUE): {resp_charge_parameter.value}")
    print("")
    print(f"RESP CHARGES        : {numpy.round(resp_charges, 4)}")


if __name__ == "__main__":
    main()
