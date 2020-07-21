import functools
import os

import arviz
import numpy
from tqdm import tqdm

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


class MetropolisSampler:
    def __init__(
        self, potential_fn, proposal_sizes, acceptance_target=0.5, tune_frequency=100,
    ):
        self._potential_fn = potential_fn

        self._acceptance_target = acceptance_target
        self._proposal_sizes = proposal_sizes
        self._tune_frequency = tune_frequency

        self._proposed_moves = numpy.zeros(proposal_sizes.shape)
        self._accepted_moves = numpy.zeros(proposal_sizes.shape)

    def step(self, current_parameters, current_ln_p, adapt):

        # Choose a random parameter to change
        parameter_index = numpy.random.randint(0, len(current_parameters), 1)

        # Sample the new parameters from a normal distribution.
        proposed_parameters = current_parameters.copy()

        proposal_value = (numpy.random.random() * 2.0 - 1.0) * self._proposal_sizes[
            parameter_index
        ]
        proposed_parameters[parameter_index] = (
            current_parameters[parameter_index] + proposal_value
        )

        proposed_log_p = self._potential_fn(proposed_parameters)

        alpha = proposed_log_p - current_ln_p

        random_number = numpy.log(numpy.random.random())
        accept = random_number < alpha

        # Update the bookkeeping
        self._proposed_moves[parameter_index] += 1

        if accept:
            self._accepted_moves[parameter_index] += 1

            current_parameters = proposed_parameters
            current_ln_p = proposed_log_p

        # Tune the proposals if needed
        total_proposed_moves = numpy.sum(self._proposed_moves)

        if (
            adapt
            and self._tune_frequency > 0
            and total_proposed_moves > 0
            and total_proposed_moves % self._tune_frequency == 0
        ):
            self._tune_proposals()

        return current_parameters, current_ln_p, accept

    def _tune_proposals(self):
        """Attempt to tune the move proposals to reach the
        `acceptance_target`.
        """

        divisor = numpy.maximum(1, self._proposed_moves)
        acceptance_rates = self._accepted_moves / divisor

        for parameter_index, rate in enumerate(acceptance_rates):

            scale = 0.9 if rate < self._acceptance_target else 1.1
            scale = 1.0 if self._proposed_moves[parameter_index] == 0 else scale

            self._proposal_sizes[parameter_index] *= scale

        self._proposed_moves = numpy.zeros(self._proposal_sizes.shape)
        self._accepted_moves = numpy.zeros(self._proposal_sizes.shape)


def compute_ln_posterior(parameters, observed, design_matrix):

    sigma = parameters[-1:, :]
    parameters = parameters[:-1, :]

    # Compute the log prior for the bcc parameters
    ln_bcc_prior = -0.5 * (numpy.log(2.0 * numpy.pi) + parameters * parameters)
    # Compute the sigma prior
    ln_sigma_prior = -numpy.log(10.0)

    # Compute the likelihood
    delta_term = observed - design_matrix @ parameters

    ln_prefactor = -design_matrix.shape[0] * numpy.log(sigma)
    ln_likelihood = ln_prefactor - 0.5 / (sigma * sigma) * (delta_term.T @ delta_term)

    ln_posterior = ln_bcc_prior.sum() + ln_sigma_prior + ln_likelihood
    return ln_posterior


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
    initial_sigma = 0.01
    initial_parameters = numpy.array(
        [
            [bcc_parameters[index].value]
            for index in range(len(bcc_parameters))
            if index not in fixed_parameter_indices
        ]
        + [[initial_sigma]]
    )

    current_parameters = initial_parameters.copy()

    # Precalculate the expensive operations which are needed to
    # evaluate the objective function, but do not depend on the
    # current parameters.
    precalculated_terms = ESPOptimization.precalculate(
        smiles, esp_store, bcc_collection, fixed_parameter_indices, charge_settings
    )

    # Compute the design matrix.
    design_matrix = numpy.vstack(
        precalculated.inverse_distance_matrix @ precalculated.assignment_matrix
        for precalculated in precalculated_terms
    )
    observed = numpy.vstack(
        precalculated.v_difference for precalculated in precalculated_terms
    )

    # Construct the potential function
    potential_fn = functools.partial(
        compute_ln_posterior, observed=observed, design_matrix=design_matrix
    )

    sampler = MetropolisSampler(
        potential_fn=potential_fn, proposal_sizes=current_parameters.copy() / 10.0
    )

    initial_ln_p = potential_fn(current_parameters)
    current_ln_p = initial_ln_p

    parameter_trace = []

    for epoch in tqdm(range(50000)):

        current_parameters, current_ln_p, _ = sampler.step(
            current_parameters, current_ln_p, epoch < 5000
        )

        if epoch >= 5000:
            parameter_trace.append(current_parameters)

    data = arviz.convert_to_inference_data(
        {
            i
            if i < len(current_parameters) - 1
            else "sigma": numpy.hstack(parameter_trace)[i, :]
            for i in range(len(current_parameters))
        }
    )
    os.makedirs("reference", exist_ok=True)

    with temporary_cd("reference"):
        axes = arviz.plot_trace(data)
        figure = axes[0][0].figure
        figure.savefig("trace.png")

        axes = arviz.plot_pair(data, kind="kde")
        figure = axes[0][0].figure
        figure.savefig("corner.png")


if __name__ == "__main__":
    main()
