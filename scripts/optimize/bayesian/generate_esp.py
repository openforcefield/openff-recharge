from openff.recharge.conformers.conformers import OmegaELF10
from openff.recharge.esp import ESPSettings
from openff.recharge.esp.psi4 import Psi4ESPGenerator
from openff.recharge.esp.storage import MoleculeESPRecord, MoleculeESPStore
from openff.recharge.grids import GridSettings
from openff.recharge.utilities.openeye import smiles_to_molecule


def main():

    # Define the molecules to include in the training set.
    training_smiles = ["C", "CC", "CCC", "CO", "CCO", "CCCO"]

    # Create a compact store for the computes ESP values.
    esp_store = MoleculeESPStore()

    # Compute the ESP of each molecule of interest .
    grid_settings = GridSettings(
        type="fcc", spacing=0.5, inner_vdw_scale=1.4, outer_vdw_scale=2.0
    )
    esp_settings = ESPSettings(
        basis="aug-cc-pV(D+d)Z", method="pw6b95", grid_settings=grid_settings
    )

    # Generate the ESP.
    for smiles in training_smiles:

        oe_molecule = smiles_to_molecule(smiles)
        conformers = OmegaELF10.generate(oe_molecule, 5)

        for conformer in conformers:
            grid_coordinates, esp = Psi4ESPGenerator.generate(
                oe_molecule, conformer, esp_settings
            )

            esp_store.store(
                MoleculeESPRecord.from_oe_molecule(
                    oe_molecule, conformer, grid_coordinates, esp, esp_settings
                )
            )


if __name__ == "__main__":
    main()
