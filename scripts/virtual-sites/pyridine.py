import numpy

from openff.recharge.charges.bcc import original_am1bcc_corrections
from openff.recharge.charges.charges import ChargeSettings
from openff.recharge.charges.vsite import (
    DivalentLonePairParameter,
    VirtualSiteCollection,
)
from openff.recharge.conformers import ConformerGenerator, ConformerSettings
from openff.recharge.esp import ESPSettings
from openff.recharge.esp.psi4 import Psi4ESPGenerator
from openff.recharge.esp.storage import MoleculeESPRecord
from openff.recharge.grids import GridSettings
from openff.recharge.optimize import ESPOptimization
from openff.recharge.utilities.openeye import smiles_to_molecule


def main():

    # Load in the molecule to train
    oe_molecule = smiles_to_molecule("c1ccncc1")

    conformer = ConformerGenerator.generate(
        oe_molecule, ConformerSettings(max_conformers=1)
    )[0]

    # Generate a set of ESP data to train the v-site charge to.
    esp_settings = ESPSettings(
        method="hf", basis="6-31G*", grid_settings=GridSettings(spacing=0.7)
    )
    grid, esp, electric_field = Psi4ESPGenerator.generate(
        oe_molecule=oe_molecule, conformer=conformer, settings=esp_settings
    )

    esp_record = MoleculeESPRecord.from_oe_molecule(
        oe_molecule, conformer, grid, esp, electric_field, esp_settings
    )

    # Define the parameters to train
    bcc_collection = original_am1bcc_corrections()

    vsite_collection = VirtualSiteCollection(
        parameters=[
            DivalentLonePairParameter(
                smirks="[#6r6:1]@[#7r6:2]@[#6r6:3]",
                name="EP",
                distance=0.3,
                out_of_plane_angle=0.0,
                charge_increments=(0.0, 0.0, 0.0),
                sigma=0.0,
                epsilon=0.0,
                match="once",
            )
        ]
    )

    vsite_parameter_keys = [
        (parameter.smirks, parameter.type, parameter.name, 1)
        for parameter in vsite_collection.parameters
    ]

    # Define the terms that contribute to the objective function.
    objective_terms_generator = ESPOptimization.compute_objective_terms(
        esp_records=[esp_record],
        charge_settings=ChargeSettings(),
        bcc_collection=bcc_collection,
        trainable_bcc_parameters=[],
        vsite_collection=vsite_collection,
        trainable_vsite_parameters=vsite_parameter_keys,
    )
    objective_term = next(iter(objective_terms_generator))

    # Train the parameters.
    initial_parameters = ESPOptimization.vectorize_collections(
        bcc_collection=bcc_collection,
        trainable_bcc_parameters=[],
        vsite_collection=vsite_collection,
        trainable_vsite_parameters=vsite_parameter_keys,
    )

    final_parameters, *_ = numpy.linalg.lstsq(
        objective_term.design_matrix, objective_term.target_residuals, rcond=None
    )

    print("INITIAL: ", initial_parameters)
    print("FINAL:   ", final_parameters)


if __name__ == "__main__":
    main()
