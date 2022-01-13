import numpy
from tqdm import tqdm

from openff.recharge.charges import ChargeSettings
from openff.recharge.charges.bcc import BCCCollection, BCCParameter
from openff.recharge.conformers import ConformerGenerator, ConformerSettings
from openff.recharge.esp import ESPSettings
from openff.recharge.esp.psi4 import Psi4ESPGenerator
from openff.recharge.esp.storage import MoleculeESPRecord
from openff.recharge.grids import GridSettings
from openff.recharge.optimize import ESPObjective, ESPObjectiveTerm
from openff.recharge.utilities.openeye import smiles_to_molecule


def main():

    # Load in the molecules to train
    training_set = ["C", "CC", "CCC", "CCCC"]

    # Generate reference QC data for each molecule in the set.
    qc_data_settings = ESPSettings(
        method="hf", basis="6-31G*", grid_settings=GridSettings(spacing=0.7)
    )
    qc_data_records = []

    for smiles in tqdm(training_set):

        oe_molecule = smiles_to_molecule(smiles)

        conformers = ConformerGenerator.generate(
            oe_molecule, ConformerSettings(max_conformers=5)
        )

        for conformer in tqdm(conformers):

            grid, esp, electric_field = Psi4ESPGenerator.generate(
                oe_molecule=oe_molecule, conformer=conformer, settings=qc_data_settings
            )
            qc_data_record = MoleculeESPRecord.from_oe_molecule(
                oe_molecule, conformer, grid, esp, electric_field, qc_data_settings
            )

            qc_data_records.append(qc_data_record)

    # Define a set of parameters to train
    bcc_collection = BCCCollection(
        parameters=[
            BCCParameter(smirks="[#6X4:1]-[#6X4:2]", value=0.0),
            BCCParameter(smirks="[#6X4:1]-[#1:2]", value=0.0),
        ]
    )
    bcc_parameters_to_train = ["[#6X4:1]-[#1:2]"]

    # Construct the terms in our objective function that we will aim to minimize. See
    # also the ``ElectricFieldObjective`` objective class.
    objective_terms_generator = ESPObjective.compute_objective_terms(
        esp_records=qc_data_records,
        # Here we use AM1-mulliken charges as the base charges to correct.
        charge_settings=ChargeSettings(theory="am1"),
        bcc_collection=bcc_collection,
        bcc_parameter_keys=bcc_parameters_to_train,
    )
    # Combine all the terms in our objective function (i.e. the difference between
    # the reference and predicted ESP values for each molecule in each conformer) into
    # a single object.
    objective_term = ESPObjectiveTerm.combine(*objective_terms_generator)

    # Train the parameters.
    trained_values, *_ = numpy.linalg.lstsq(
        objective_term.atom_charge_design_matrix,
        objective_term.reference_values,
        rcond=None,
    )

    print("TRAINED PARAMETERS".center(80, "-"))

    for parameter_smirks, trained_value in zip(bcc_parameters_to_train, trained_values):

        print(
            parameter_smirks,
            f" INITIAL={0.0:.4f} " f" FINAL={float(trained_value):.4f}",
        )


if __name__ == "__main__":
    main()
