import os

import arviz
import numpy
import pyro
import torch
from pyro.distributions import HalfCauchy, MultivariateNormal, Normal
from pyro.infer import MCMC, NUTS

from openff.recharge.charges.bcc import (
    BCCCollection,
    BCCGenerator,
    original_am1bcc_corrections,
)
from openff.recharge.charges.charges import ChargeSettings
from openff.recharge.esp.storage import MoleculeESPStore
from openff.recharge.optimize import ESPOptimization
from openff.recharge.utilities import temporary_cd
from openff.recharge.utilities.openeye import smiles_to_molecule


def model(n_parameters, design_matrix):

    sigma = pyro.sample("sigma", HalfCauchy(10.0))

    parameters = pyro.sample(
        "parameters",
        Normal(torch.zeros((n_parameters, 1)), torch.ones((n_parameters, 1))),
    )

    residuals = pyro.sample(
        "residuals",
        MultivariateNormal(
            design_matrix @ parameters, torch.eye(n_parameters) * sigma * sigma
        ),
    )

    return residuals


def conditioned_model(model, n_parameters, design_matrix, observed):
    return pyro.condition(model, data={"residuals": observed})(
        n_parameters, design_matrix
    )


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
    initial_sigma = torch.tensor([0.01])
    initial_parameters = torch.tensor(
        [
            [bcc_parameters[index].value]
            for index in range(len(bcc_parameters))
            if index not in fixed_parameter_indices
        ]
    )

    # Precalculate the expensive operations which are needed to
    # evaluate the objective function, but do not depend on the
    # current parameters.
    precalculated_terms = ESPOptimization.precalculate(
        smiles, esp_store, bcc_collection, fixed_parameter_indices, charge_settings
    )

    # Compute the design matrix.
    design_matrix = torch.tensor(
        numpy.vstack(
            [
                precalculated.inverse_distance_matrix @ precalculated.assignment_matrix
                for precalculated in precalculated_terms
            ]
        ),
        dtype=torch.float32,
    )
    observed = torch.tensor(
        numpy.vstack(
            [precalculated.v_difference for precalculated in precalculated_terms]
        ),
        dtype=torch.float32,
    )

    # Perform the MCMC inference.
    nuts_kernel = NUTS(conditioned_model, jit_compile=True)

    mcmc = MCMC(
        nuts_kernel,
        num_samples=1000,
        warmup_steps=100,
        num_chains=1,
        initial_params={"parameters": initial_parameters, "sigma": initial_sigma},
    )
    mcmc.run(model, len(initial_parameters), design_matrix, observed)

    data = arviz.from_pyro(mcmc)
    os.makedirs("pyro", exist_ok=True)

    with temporary_cd("pyro"):
        axes = arviz.plot_trace(data)
        figure = axes[0][0].figure
        figure.savefig("trace.png")

        axes = arviz.plot_pair(data, kind="kde")
        figure = axes[0][0].figure
        figure.savefig("corner.png")


if __name__ == "__main__":
    main()
