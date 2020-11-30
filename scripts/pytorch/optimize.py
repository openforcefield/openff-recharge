import numpy
import torch
import torch.optim
from matplotlib import pyplot

from openff.recharge.charges.bcc import (
    BCCCollection,
    BCCGenerator,
    original_am1bcc_corrections,
    AromaticityModels,
)
from openff.recharge.charges.charges import ChargeSettings
from openff.recharge.esp.storage import MoleculeESPStore
from openff.recharge.optimize.optimize import ElectricFieldOptimization, ESPOptimization
from openff.recharge.utilities.openeye import smiles_to_molecule


def main():

    # Define the type of optimization
    # optimization_class = ESPOptimization
    optimization_class = ElectricFieldOptimization

    # Load the data base of ESP values.
    esp_store = MoleculeESPStore()
    # Define the molecules to include in the training set.
    smiles = [*esp_store.list()]

    # Determine which BCC parameters are exercised by the training set,
    # and fix any with a fixed zero value
    bcc_parameters = BCCGenerator.applied_corrections(
        *[smiles_to_molecule(smiles_pattern) for smiles_pattern in smiles],
        bcc_collection=original_am1bcc_corrections(),
    )
    fixed_parameter_indices = [
        index
        for index in range(len(bcc_parameters))
        if bcc_parameters[index].provenance["code"][0:2]
        == bcc_parameters[index].provenance["code"][-2:]
    ]

    bcc_collection = BCCCollection(
        parameters=bcc_parameters,
        aromaticity_model=AromaticityModels.MDL
    )
    charge_settings = ChargeSettings()

    # Define the starting parameters
    initial_parameters = torch.tensor(
        [
            [bcc_parameters[index].value]
            for index in range(len(bcc_parameters))
            if index not in fixed_parameter_indices
        ],
        dtype=torch.float64,
    )
    current_parameters = initial_parameters.clone().detach().requires_grad_(True)

    # Precalculate the expensive operations which are needed to
    # evaluate the objective function, but do not depend on the
    # current parameters.
    objective_term_generator = optimization_class.compute_objective_terms(
        smiles, esp_store, bcc_collection, fixed_parameter_indices, charge_settings
    )
    objective_terms = [*objective_term_generator]

    design_matrix = torch.from_numpy(
        numpy.vstack(objective_term.design_matrix for objective_term in objective_terms)
    )
    target_residuals = torch.from_numpy(
        numpy.vstack(
            objective_term.target_residuals for objective_term in objective_terms
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

    print(f"Initial parameters: {initial_parameters.detach().numpy()}")
    print(f"Final parameters: {current_parameters.detach().numpy()}")

    # Save out the new parameters.
    final_bcc_collection = bcc_collection.copy(deep=True)

    parameter_index = 0

    for index in range(len(final_bcc_collection.parameters)):

        if index in fixed_parameter_indices:
            continue

        final_bcc_collection.parameters[index].value = float(
            current_parameters.detach().numpy()[parameter_index]
        )

        parameter_index += 1

    for index, parameter in reversed(list(enumerate(final_bcc_collection.parameters))):

        if index in fixed_parameter_indices:
            continue

        print(parameter.smirks, parameter.value)

    with open("final-parameters.json", "w") as file:
        file.write(final_bcc_collection.json())


if __name__ == "__main__":
    main()
