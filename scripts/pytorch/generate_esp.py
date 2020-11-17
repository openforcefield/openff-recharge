import functools
from multiprocessing import Pool
from typing import List

from openff.recharge.conformers import ConformerGenerator, ConformerSettings
from openff.recharge.esp import ESPSettings
from openff.recharge.esp.psi4 import Psi4ESPGenerator
from openff.recharge.esp.storage import MoleculeESPRecord, MoleculeESPStore
from openff.recharge.grids import GridSettings
from openff.recharge.utilities.openeye import smiles_to_molecule

N_PROCESSES = 4


def compute_esp(
    smiles: str, max_conformers: int, settings: ESPSettings
) -> List[MoleculeESPRecord]:
    """Compute the ESP for a molecule in different conformers.

    Parameters
    ----------
    smiles
        The SMILES representation of the molecule.
    max_conformers
        The target number of conformers to compute the ESP of.
    settings
        The settings to use when generating the ESP.
    """

    oe_molecule = smiles_to_molecule(smiles)

    # Generate a set of conformers for the molecule.
    conformers = ConformerGenerator.generate(
        oe_molecule, ConformerSettings(max_conformers=max_conformers)
    )

    esp_records = []

    for conformer in conformers:

        grid_coordinates, esp, electric_field = Psi4ESPGenerator.generate(
            oe_molecule, conformer, settings
        )

        esp_records.append(
            MoleculeESPRecord.from_oe_molecule(
                oe_molecule, conformer, grid_coordinates, esp, electric_field, settings
            )
        )

    return esp_records


def main():

    # Define the molecules to include in the training set.
    smiles = {
        "CC1(C)C(=O)[C@]2(C)CC[C@H]1C2",
        "CCCCCCO",
        "CCCCC",
        "CCCCCCCC",
        "COC(C)=O",
        "CCO",
        "OCCO",
        "C=CCCCC",
        "CO",
        "CCC(C)(C)O",
        "CC1CCCCC1",
        "CCCOC(C)=O",
        "CCOc1ccccc1",
        "CCCCOCCCC",
        "OCCCCO",
        "O=Cc1ccccc1",
        "CCc1ccccc1",
        "CCCCOCCO",
        "CCCO",
        "CCOC(=O)CC(=O)OCC",
        "CCCCCO",
        "CCCCCC(C)=O",
        "CC(C)OC(C)C",
        "COC=O",
        "CCCCO",
        "CCC(C)=O",
        "Cc1ccccc1",
        "CCOCCO",
        "CCCCC(=O)OCC",
        "CCCCOC(C)=O",
        "OCCCO",
        "COCCOCCOC",
        "CCOC(C)=O",
        "C1COCCO1",
        "CCCCCCCO",
        "c1ccccc1",
        "CCCCC(=O)OC",
        "CCCCOC=O",
        "OCCOCCO",
        "CCC(=O)OC",
        "CCCC(=O)OC",
        "CC(C)(C)O",
        "CCCCCCCCO",
        "O=C1CCCC1",
        "C=C(C)[C@H]1CC=C(C)CC1",
        "COC(C)(C)C",
        "OCCCc1ccccc1",
        "CC(=O)C(C)=O",
        "CC(C)CO",
        "COc1ccccc1",
        "CCCC(C)=O",
        "Cc1ccccc1C",
        "CC(C)CC(C)(C)C",
        "OCc1ccccc1",
        "CCCCCC",
        "COCCOCCO",
        "OCC(O)CO",
        "CC/C=C\\CCO",
        "OCCc1ccccc1",
        "CCCOC(=O)CCC",
        "CC(=O)CC(C)C",
        "CC(C)=O",
        "CCCCCCC",
        "C1CCOC1",
        "C1CCCCC1",
        "C1CCCC1",
        "CCOC(=O)CC",
        "CC(C)O",
        "CCOC=O",
        "CCCOC=O",
        "CCCOC(=O)CC",
        "C1CCOCC1",
        "CCOC(C)(C)C",
        "CCOC(C)(C)CC",
        "CCCCCCCCC",
    }

    # Create a compact store for the computes ESP values.
    esp_store = MoleculeESPStore()

    # Compute the ESP of each molecule of interest .
    grid_settings = GridSettings(
        type="fcc", spacing=0.5, inner_vdw_scale=1.4, outer_vdw_scale=2.0
    )
    esp_settings = ESPSettings(
        basis="aug-cc-pV(D+d)Z", method="pw6b95", grid_settings=grid_settings
    )

    with Pool(processes=N_PROCESSES) as pool:

        esp_store.store(
            *[
                esp_record
                for esp_records in pool.map(
                    functools.partial(
                        compute_esp, max_conformers=5, settings=esp_settings
                    ),
                    smiles,
                )
                for esp_record in esp_records
            ]
        )


if __name__ == "__main__":
    main()
