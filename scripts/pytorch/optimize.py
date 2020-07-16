import torch
import torch.optim

from openff.recharge.charges.bcc import (
    BCCCollection,
    BCCGenerator,
    original_am1bcc_corrections,
)
from openff.recharge.charges.charges import ChargeSettings
from openff.recharge.esp.storage import MoleculeESPStore
from openff.recharge.optimize.optimize import ESPOptimization
from openff.recharge.utilities.openeye import smiles_to_molecule


def main():

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

    bcc_collection = BCCCollection(parameters=bcc_parameters)
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
    precalculated_terms = ESPOptimization.precalculate(
        smiles, esp_store, bcc_collection, fixed_parameter_indices, charge_settings
    )

    for precalculated_term in precalculated_terms:
        precalculated_term.assignment_matrix = torch.from_numpy(
            precalculated_term.assignment_matrix
        )
        precalculated_term.inverse_distance_matrix = torch.from_numpy(
            precalculated_term.inverse_distance_matrix
        )
        precalculated_term.v_difference = torch.from_numpy(
            precalculated_term.v_difference
        )

    # Optimize the parameters.
    lr = 1e-2
    n_epochs = 250

    # Defines an Adam optimizer to update the parameters
    optimizer = torch.optim.Adam([current_parameters], lr=lr)

    for epoch in range(n_epochs):

        loss = torch.zeros(1, dtype=torch.float64)

        for precalculated_term in precalculated_terms:

            # noinspection PyTypeChecker
            v_correction = ESPOptimization.compute_v_correction(
                precalculated_term.inverse_distance_matrix,
                precalculated_term.assignment_matrix,
                current_parameters,
            )

            loss += ESPOptimization.compute_objective_function(
                precalculated_term.v_difference, v_correction
            )

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        print(f"Epoch {epoch}: loss={loss.item()}")

    print(f"Initial parameters: {initial_parameters.detach().numpy()}")
    print(f"Final parameters: {current_parameters.detach().numpy()}")


if __name__ == "__main__":
    main()
