from tqdm import tqdm

from openff.recharge.conformers import ConformerGenerator, ConformerSettings
from openff.recharge.esp import ESPSettings
from openff.recharge.esp.psi4 import Psi4ESPGenerator
from openff.recharge.esp.storage import MoleculeESPRecord, MoleculeESPStore
from openff.recharge.grids import LatticeGridSettings
from openff.recharge.utilities.molecule import smiles_to_molecule


def main():
    # Load in the molecule that we would like to generate the electrostatic properties
    # for.
    molecule = smiles_to_molecule("OCC(O)CO")

    # Define the grid that the electrostatic properties will be trained on and the
    # level of theory to compute the properties at.
    grid_settings = LatticeGridSettings(
        type="fcc", spacing=0.5, inner_vdw_scale=1.4, outer_vdw_scale=2.0
    )
    # Generating an MSK style grid is also supported:
    # from openff.recharge.grids import MSKGridSettings
    # grid_settings = MSKGridSettings(density=1.0)

    esp_settings = ESPSettings(basis="6-31G*", method="hf", grid_settings=grid_settings)

    # Generate a set of conformers for the molecule. We will compute the ESP and
    # electric field for the molecule in each conformer.
    conformers = ConformerGenerator.generate(
        molecule, ConformerSettings(max_conformers=10)
    )

    # Create a database to store the computed electrostatic properties in to make
    # training and testing against the data easier.
    qc_data_store = MoleculeESPStore()

    # Compute and store the electrostatic properties for each conformer
    records = []

    for conformer in tqdm(conformers):
        conformer, grid, esp, electric_field = Psi4ESPGenerator.generate(
            molecule,
            conformer,
            esp_settings,
            # Minimize the input conformer prior to evaluating the ESP / EF
            minimize=True,
        )

        record = MoleculeESPRecord.from_molecule(
            molecule, conformer, grid, esp, electric_field, esp_settings
        )
        records.append(record)

    qc_data_store.store(*records)

    # Retrieve the stored properties.
    print(qc_data_store.retrieve())


if __name__ == "__main__":
    main()
