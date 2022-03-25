from openff.toolkit.topology import Molecule
from tqdm import tqdm

from openff.recharge.charges.library import (
    LibraryChargeCollection,
    LibraryChargeGenerator,
)
from openff.recharge.charges.resp import generate_resp_charge_parameter
from openff.recharge.conformers import ConformerGenerator, ConformerSettings
from openff.recharge.esp import ESPSettings
from openff.recharge.esp.psi4 import Psi4ESPGenerator
from openff.recharge.esp.storage import MoleculeESPRecord
from openff.recharge.grids import MSKGridSettings


def main():

    qc_data_settings = ESPSettings(
        method="hf", basis="6-31G*", grid_settings=MSKGridSettings()
    )
    qc_data_records = []

    molecule = Molecule.from_mapped_smiles(
        "[H:5][O:3][C:1]([H:7])([H:8])[C:2]([H:9])([H:10])[O:4][H:6]"
    )
    conformers = ConformerGenerator.generate(
        molecule, ConformerSettings(max_conformers=5)
    )

    for conformer in tqdm(conformers):

        grid, esp, electric_field = Psi4ESPGenerator.generate(
            molecule=molecule, conformer=conformer, settings=qc_data_settings
        )
        qc_data_record = MoleculeESPRecord.from_molecule(
            molecule, conformer, grid, esp, electric_field, qc_data_settings
        )

        qc_data_records.append(qc_data_record)

    from openff.recharge.charges.resp.solvers import IterativeSolver

    resp_solver = IterativeSolver()
    # from openff.recharge.charges.resp.solvers import SciPySolver
    # resp_solver = SciPySolver(method="SLSQP")

    resp_charge_parameter = generate_resp_charge_parameter(qc_data_records, resp_solver)

    resp_charges = LibraryChargeGenerator.generate(
        molecule, LibraryChargeCollection(parameters=[resp_charge_parameter])
    )

    print(f"RESP SMILES         : {resp_charge_parameter.smiles}")
    print(f"RESP VALUES (UNIQUE): {resp_charge_parameter.value}")
    print("")
    print(f"RESP CHARGES        : {resp_charges}")


if __name__ == "__main__":
    main()
