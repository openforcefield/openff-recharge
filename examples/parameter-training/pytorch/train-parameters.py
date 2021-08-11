import numpy
import torch

from openff.recharge.charges.bcc import BCCCollection, BCCParameter
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
from openff.recharge.optimize import ESPObjective
from openff.recharge.utilities.openeye import smiles_to_molecule
from openff.recharge.utilities.tensors import to_torch


def print_parameters(keys, values):

    for key, value in zip(keys, values):
        print(key, f"{float(value):.5f}")


def main():

    # Load in the molecule of interest and generate a set of reference QC data to train
    # against.
    oe_molecule = smiles_to_molecule("c1ccncc1")

    conformer = ConformerGenerator.generate(
        oe_molecule, ConformerSettings(max_conformers=1)
    )[0]

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
    bcc_collection = BCCCollection(
        parameters=[
            BCCParameter(smirks="[#6:1]-[#1:2]", value=0.0, provenance={}),
            BCCParameter(smirks="[#6:1]@[#6:2]", value=0.0, provenance={}),
            BCCParameter(smirks="[#6:1]@[#7:2]", value=0.0, provenance={}),
        ]
    )
    bcc_parameter_keys = ["[#6:1]-[#1:2]", "[#6:1]@[#7:2]"]

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
    vsite_charge_parameter_keys = [
        # Only train the nitrogen-vsite charge increment.
        ("[#6r6:1]@[#7r6:2]@[#6r6:3]", "DivalentLonePair", "EP", 1)
    ]
    vsite_coordinate_parameter_keys = [
        ("[#6r6:1]@[#7r6:2]@[#6r6:3]", "DivalentLonePair", "EP", "distance")
    ]

    # Construct the terms in our objective function that we will aim to minimize. See
    # also the ``ElectricFieldObjective`` objective class.
    objective_terms_generator = ESPObjective.compute_objective_terms(
        esp_records=[esp_record],
        charge_settings=ChargeSettings(),
        bcc_collection=bcc_collection,
        bcc_parameter_keys=bcc_parameter_keys,
        vsite_collection=vsite_collection,
        vsite_charge_parameter_keys=vsite_charge_parameter_keys,
        vsite_coordinate_parameter_keys=vsite_coordinate_parameter_keys,
    )

    # Convert all of the arrays to PyTorch tensors ready for PyTorch training.
    objective_term = next(iter(objective_terms_generator))
    objective_term.to_backend("torch")

    # Vectorize our BCC and virtual site parameters into flat tensors that can be
    # provided to and trained by a PyTorch optimizer.
    initial_charge_increments = numpy.vstack(
        [
            bcc_collection.vectorize(bcc_parameter_keys),
            vsite_collection.vectorize_charge_increments(vsite_charge_parameter_keys),
        ]
    )
    initial_charge_increments = to_torch(initial_charge_increments)
    initial_charge_increments.requires_grad = True

    initial_vsite_coordinates = vsite_collection.vectorize_coordinates(
        vsite_coordinate_parameter_keys
    )
    initial_vsite_coordinates = to_torch(initial_vsite_coordinates)
    initial_vsite_coordinates.requires_grad = True

    print("INITIAL".center(80, "="))

    print_parameters(
        [*bcc_parameter_keys, *vsite_charge_parameter_keys], initial_charge_increments
    )
    print_parameters(vsite_coordinate_parameter_keys, initial_vsite_coordinates)

    # Optimize the parameters.
    lr = 1e-2
    n_epochs = 300

    print("TRAINING".center(80, "="))

    optimizer = torch.optim.Adam(
        [initial_charge_increments, initial_vsite_coordinates], lr=lr
    )

    for epoch in range(n_epochs + 1):

        loss = objective_term.evaluate(
            initial_charge_increments, initial_vsite_coordinates
        )

        # Add a light restraint to the v-site distance to stop it from becoming too
        # unphysical
        loss += (0.1 * (initial_vsite_coordinates - 0.3) ** 2).sum()

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: loss={loss.item()}")

    print("FINAL".center(80, "="))

    print_parameters(
        [*bcc_parameter_keys, *vsite_charge_parameter_keys], initial_charge_increments
    )
    print_parameters(vsite_coordinate_parameter_keys, initial_vsite_coordinates)


if __name__ == "__main__":
    main()
