import arviz
import pyro
import pyro.distributions
import torch
import torch.distributions
from matplotlib import pyplot
from pyro.infer import MCMC, NUTS

from openff.recharge.charges.bcc import BCCCollection, BCCParameter
from openff.recharge.charges import ChargeSettings
from openff.recharge.charges.vsite import BondChargeSiteParameter, VirtualSiteCollection
from openff.recharge.conformers import ConformerGenerator, ConformerSettings
from openff.recharge.esp import ESPSettings
from openff.recharge.esp.psi4 import Psi4ESPGenerator
from openff.recharge.esp.storage import MoleculeESPRecord
from openff.recharge.grids import GridSettings
from openff.recharge.optimize import ESPObjective
from openff.recharge.utilities.openeye import smiles_to_molecule


def main():

    # Calculate the ESP of chloromethane using Psi4.
    oe_molecule = smiles_to_molecule("CCl")
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

    # Define the parameters to be trained.
    bcc_collection = BCCCollection(
        parameters=[
            BCCParameter(smirks="[#1:1]-[#6:2]", value=0.0),
            BCCParameter(smirks="[#17:1]-[#6:2]", value=0.0),
        ]
    )
    bcc_parameters_to_train = ["[#1:1]-[#6:2]", "[#17:1]-[#6:2]"]

    vsite_collection = VirtualSiteCollection(
        parameters=[
            BondChargeSiteParameter(
                smirks="[#17:1]-[#6:2]",
                name="EP",
                distance=2.3,
                charge_increments=(0.4, 0.0),
                sigma=0.0,
                epsilon=0.0,
                match="once",
            )
        ]
    )
    vsite_charge_parameter_keys = [
        # Train only the Cl-EP charge correction.
        ("[#17:1]-[#6:2]", "BondCharge", "EP", 0)
    ]
    vsite_coordinate_parameter_keys = [
        # Train the Cl-EP distance.
        ("[#17:1]-[#6:2]", "BondCharge", "EP", "distance")
    ]

    n_charge_parameters = len(bcc_parameters_to_train) + len(
        vsite_charge_parameter_keys
    )
    n_coord_parameters = len(vsite_coordinate_parameter_keys)

    objective_terms_generator = ESPObjective.compute_objective_terms(
        esp_records=[esp_record],
        charge_settings=ChargeSettings(),
        bcc_collection=bcc_collection,
        bcc_parameter_keys=bcc_parameters_to_train,
        vsite_collection=vsite_collection,
        vsite_charge_parameter_keys=vsite_charge_parameter_keys,
        vsite_coordinate_parameter_keys=vsite_coordinate_parameter_keys,
    )
    objective_term = next(iter(objective_terms_generator))
    objective_term.to_backend("torch")

    # Define our 'model' - i.e. the prior and likelihood functions that combined
    # define the posterior distribution we aim to draw samples from.
    def model():

        # Place priors on the virtual site charges increments and distance.
        charge_increment = pyro.sample(
            "charge_increment",
            pyro.distributions.Normal(
                # Restrain the partial charges close to 0.0 using a tighter normal
                # distribution with sigma=0.1
                torch.zeros((n_charge_parameters, 1)),
                torch.ones((n_charge_parameters, 1)) * 0.1,
            ),
        )
        distance = pyro.sample(
            "distance",
            pyro.distributions.Normal(
                # Use a normal distribution centered at one and with a sigma of 0.5
                # to stop the distance collapsing to 0 or growing too large.
                torch.ones((n_coord_parameters, 1)),
                torch.ones((n_coord_parameters, 1)) * 0.5,
            ),
        )

        sigma = pyro.sample(
            # Place a weakly informative prior on sigma.
            "sigma",
            pyro.distributions.HalfCauchy(torch.tensor([[1.0]])),
        )

        # Evaluate the ESP of our model.
        predicted = objective_term.predict(charge_increment, distance)

        return pyro.sample(
            "predicted_residuals",
            pyro.distributions.Normal(loc=predicted, scale=sigma),
            obs=objective_term.reference_values,
        )

    # Train the parameters and plot the sampled traces.
    nuts_kernel = NUTS(model)

    mcmc = MCMC(nuts_kernel, num_samples=500, warmup_steps=200, num_chains=1)
    mcmc.run()

    pyro_data = arviz.from_pyro(mcmc)

    arviz.plot_pair(pyro_data, kind="kde")
    pyplot.show()

    arviz.plot_trace(pyro_data)
    pyplot.show()


if __name__ == "__main__":
    main()
