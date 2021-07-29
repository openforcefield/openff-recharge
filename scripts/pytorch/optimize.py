import numpy
import torch
import torch.optim

from openff.recharge.charges.bcc import (
    AromaticityModels,
    BCCCollection,
    BCCGenerator,
    original_am1bcc_corrections,
)
from openff.recharge.charges.charges import ChargeSettings
from openff.recharge.esp.storage import MoleculeESPStore
from openff.recharge.optimize.optimize import ElectricFieldOptimization, ESPOptimization
from openff.recharge.utilities.openeye import smiles_to_molecule


def main():

    # Define the type of optimization
    optimization_class = ESPOptimization
    # optimization_class = ElectricFieldOptimization

    # Load the pre-computed ESP values.
    esp_records = MoleculeESPStore().retrieve()

    # Determine which BCC parameters are exercised by the training set,
    # and fix any with a fixed zero value
    bcc_parameters = BCCGenerator.applied_corrections(
        *[smiles_to_molecule(record.tagged_smiles) for record in esp_records],
        bcc_collection=original_am1bcc_corrections(),
    )
    bcc_collection = BCCCollection(
        parameters=bcc_parameters, aromaticity_model=AromaticityModels.MDL
    )

    trainable_bcc_parameters = [
        parameter.smirks
        for parameter in bcc_parameters
        if parameter.provenance["code"][0:2] != parameter.provenance["code"][-2:]
    ]

    # Define the starting parameters
    initial_parameters = torch.from_numpy(
        optimization_class.vectorize_collections(
            bcc_collection, trainable_bcc_parameters
        )
    )
    current_parameters = initial_parameters.clone().requires_grad_(True)

    # Precalculate the expensive operations which are needed to
    # evaluate the objective function, but do not depend on the
    # current parameters.
    objective_term_generator = optimization_class.compute_objective_terms(
        esp_records=esp_records,
        charge_settings=ChargeSettings(),
        bcc_collection=bcc_collection,
        trainable_bcc_parameters=trainable_bcc_parameters,
    )
    objective_terms = [*objective_term_generator]

    design_matrix = torch.from_numpy(
        numpy.vstack(
            [objective_term.design_matrix for objective_term in objective_terms]
        )
    )
    target_residuals = torch.from_numpy(
        numpy.vstack(
            [objective_term.target_residuals for objective_term in objective_terms]
        )
    )

    # Optimize the parameters.
    lr = 1e-2
    n_epochs = 100

    # Defines an Adam optimizer to update the parameters
    optimizer = torch.optim.Adam([current_parameters], lr=lr)

    for epoch in range(n_epochs):

        if issubclass(optimization_class, ElectricFieldOptimization):
            delta = target_residuals - design_matrix @ current_parameters.flatten()
        elif issubclass(optimization_class, ESPOptimization):
            delta = target_residuals - design_matrix @ current_parameters
        else:
            raise NotImplementedError()

        loss = (delta * delta).sum()

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        print(f"Epoch {epoch}: loss={loss.item()}")

    print("\n----------\nParameters\n----------\n")

    # Save out the new parameters.
    final_bcc_collection = bcc_collection.copy(deep=True)

    for initial_parameter, final_parameter in zip(
        bcc_collection.parameters, final_bcc_collection.parameters
    ):

        if final_parameter.smirks not in trainable_bcc_parameters:
            continue

        final_parameter_index = trainable_bcc_parameters.index(final_parameter.smirks)

        final_parameter.value = float(
            current_parameters.detach().numpy()[final_parameter_index]
        )

        print(
            initial_parameter.smirks.ljust(25),
            f"initial={initial_parameter.value:+.4f}",
            f"final={final_parameter.value:+.4f}",
        )

    with open("final-parameters.json", "w") as file:
        file.write(final_bcc_collection.json())


if __name__ == "__main__":
    main()
