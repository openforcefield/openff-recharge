import functools
import json
import logging
import os
import pwd
from datetime import datetime
from multiprocessing import Pool
from typing import TYPE_CHECKING, Dict, List, Tuple

import click
from tqdm import tqdm

import openff.recharge
from openff.recharge.esp.qcresults import from_qcportal_results
from openff.recharge.esp.storage import MoleculeESPStore
from openff.recharge.grids import GridSettings, GridSettingsType

if TYPE_CHECKING:
    import qcelemental.models
    import qcportal.models


QCFractalResults = List[
    Tuple["qcelemental.models.Molecule", "qcportal.models.ResultRecord"]
]
QCFractalKeywords = Dict[str, "qcportal.models.KeywordSet"]


def _retrieve_result_records(
    record_ids: List["qcportal.models.ObjectId"],
) -> Tuple[QCFractalResults, QCFractalKeywords]:
    import qcportal

    # Pull down the individual result records.
    results = []

    paginating = True
    page_index = 0

    client = qcportal.PortalClient()

    while paginating:
        page_results = client.query_results(
            id=record_ids,
            limit=client.server_info["query_limit"],
            skip=page_index,
        )

        results.extend(
            [(page_result.get_molecule(), page_result) for page_result in page_results]
        )

        paginating = len(page_results) > 0
        page_index += client.server_info["query_limit"]

    # Fetch the corresponding record keywords.
    keyword_ids = list({result.keywords for (_, result) in results})
    keywords = {
        keyword_id: client.query_keywords(keyword_id)[0] for keyword_id in keyword_ids
    }

    return results, keywords


def _process_result(
    result_tuple: Tuple[
        "qcportal.models.ResultRecord",
        "qcelemental.models.Molecule",
        "qcportal.models.KeywordSet",
    ],
    grid_settings: GridSettingsType,
):
    return from_qcportal_results(*result_tuple, grid_settings=grid_settings)


@click.command(
    help="Compute the ESP from a set of wave-functions stored in a QCFractal instance."
)
@click.option(
    "--record-ids",
    "record_ids_path",
    type=click.Path(exists=True, dir_okay=False),
    help="The path to a JSON serialized list of the ids of the result records that "
    "contain the wave-functions to reconstruct the ESP / electric field from.",
)
@click.option(
    "--grid-settings",
    "grid_settings_path",
    type=click.Path(exists=True, dir_okay=False),
    help="The path to the JSON serialized settings which define the grid to reconstruct "
    "the ESP / electric field on.",
)
@click.option(
    "--n-procs",
    "n_processors",
    type=int,
    default=1,
    help="The number of processes to compute the ESP across.",
    show_default=True,
)
def reconstruct(record_ids_path: str, grid_settings_path: str, n_processors: int):
    import openeye
    import psi4
    import qcelemental
    import qcportal

    logging.basicConfig(level=logging.INFO)

    # Load in the record ids.
    with open(record_ids_path) as file:
        record_ids = json.load(file)

    # Load in the ESP settings.
    grid_settings = GridSettings.parse_file(grid_settings_path)

    # Pull down the QCA result records.
    qc_results, qc_keyword_sets = _retrieve_result_records(record_ids)

    with Pool(processes=n_processors) as pool:
        esp_records = list(
            tqdm(
                pool.imap(
                    functools.partial(_process_result, grid_settings=grid_settings),
                    [
                        (qc_result, qc_molecule, qc_keyword_sets[qc_result.keywords])
                        for (qc_molecule, qc_result) in qc_results
                    ],
                ),
                total=len(qc_results),
            )
        )

    # Store the ESP records in a data store.
    esp_store = MoleculeESPStore()
    esp_store.set_provenance(
        general_provenance={
            "user": pwd.getpwuid(os.getuid()).pw_name,
            "date": datetime.now().strftime("%d-%m-%Y"),
            "record_ids": ",".join(record_ids),
        },
        software_provenance={
            "openff-recharge": openff.recharge.__version__,
            "openeye": openeye.__version__,
            "psi4": psi4.__version__,
            "qcportal": qcportal.__version__,
            "qcelemental": qcelemental.__version__,
        },
    )

    esp_store.store(*esp_records)
