import functools
import json
import logging
from multiprocessing import Pool
from typing import List

import click

from openff.recharge.charges.exceptions import OEQuacpacError
from openff.recharge.conformers import ConformerGenerator, ConformerSettings
from openff.recharge.conformers.exceptions import OEOmegaError
from openff.recharge.esp import ESPSettings
from openff.recharge.esp.exceptions import Psi4Error
from openff.recharge.esp.psi4 import Psi4ESPGenerator
from openff.recharge.esp.storage import MoleculeESPRecord, MoleculeESPStore
from openff.recharge.utilities.openeye import smiles_to_molecule


def _compute_esp(
    smiles: str, conformer_settings: ConformerSettings, settings: ESPSettings
) -> List[MoleculeESPRecord]:
    """Compute the ESP for a molecule in different conformers.

    Parameters
    ----------
    smiles
        The SMILES representation of the molecule.
    conformer_settings
        The settings to use when generating the conformers.
    settings
        The settings to use when generating the ESP.
    """

    logger = logging.getLogger(__name__)
    logger.info(f"Processing {smiles}")

    oe_molecule = smiles_to_molecule(smiles)

    # Generate a set of conformers for the molecule.
    try:
        conformers = ConformerGenerator.generate(oe_molecule, conformer_settings)
    except (OEOmegaError, OEQuacpacError):
        logger.exception(f"Coordinates could not be generated for {smiles}.")
        return []

    esp_records = []

    for index, conformer in enumerate(conformers):

        try:
            grid_coordinates, esp, electric_field = Psi4ESPGenerator.generate(
                oe_molecule, conformer, settings
            )
        except Psi4Error:
            logger.exception(f"Psi4 failed to run for conformer {index} of {smiles}.")
            continue

        esp_records.append(
            MoleculeESPRecord.from_oe_molecule(
                oe_molecule, conformer, grid_coordinates, esp, electric_field, settings
            )
        )

    logger.info(f"Finished processing {smiles}")

    return esp_records


@click.command(help="Generate electrostatic property data for a set of SMILES.")
@click.option(
    "--smiles",
    default="smiles.json",
    type=click.Path(exists=True, dir_okay=False),
    help="The path to a JSON file containing the set of SMILES patterns.",
    show_default=True,
)
@click.option(
    "--esp-settings",
    default="esp-settings.json",
    type=click.Path(exists=True, dir_okay=False),
    help="The path to the JSON serialized ESP calculation settings.",
    show_default=True,
)
@click.option(
    "--conf-settings",
    default="conformer-settings.json",
    type=click.Path(exists=True, dir_okay=False),
    help="The path to the JSON serialized conformer generation settings.",
    show_default=True,
)
@click.option(
    "--n-procs",
    "n_processors",
    type=int,
    default=1,
    help="The number of processes to compute the ESP across.",
    show_default=True,
)
def generate(smiles: str, esp_settings: str, conf_settings: str, n_processors: int):

    logging.basicConfig(level=logging.INFO)

    esp_store = MoleculeESPStore()

    # Load in the SMILES patterns to compute the ESP properties for.
    with open(smiles) as file:
        smiles = json.load(file)

    # Load in the conformer generation and ESP settings.
    esp_settings = ESPSettings.parse_file(esp_settings)
    conformer_settings = ConformerSettings.parse_file(conf_settings)

    with Pool(processes=n_processors) as pool:

        for esp_records in pool.imap(
            functools.partial(
                _compute_esp,
                conformer_settings=conformer_settings,
                settings=esp_settings,
            ),
            smiles,
        ):

            if len(esp_records) == 0:
                continue

            esp_store.store(*esp_records)
