import os
import pickle
import json
from collections import defaultdict
from multiprocessing import Pool
from typing import Dict, List

import numpy
from openeye import oechem, oeomega
from tqdm import tqdm

from openff.recharge.charges.bcc import (
    BCCGenerator,
    BCCSettings,
    BondChargeCorrection,
    original_am1bcc_corrections,
)
from openff.recharge.charges.charges import ChargeGenerator, ChargeSettings
from openff.recharge.charges.exceptions import UnableToAssignChargeError
from openff.recharge.conformers.conformers import OmegaELF10
from openff.recharge.utilities.exceptions import RechargeException
from openff.recharge.utilities.openeye import match_smirks, smiles_to_molecule

output_stream = oechem.oeosstream()

oechem.OEThrow.SetOutputStream(output_stream)
oechem.OEThrow.Clear()


class UnableToProcessError(RechargeException):
    """An exception raised when a smiles pattern could not
    be processed ready for partial charge assignment."""


def process_molecule(smiles: str) -> Dict[str, List[str]]:

    # Drop salts / ionic liquids which are not currently supported.
    # by the assignment code.
    if "." in smiles:
        return {}

    bond_charge_corrections = original_am1bcc_corrections()
    bcc_smirks = [bcc.smirks for bcc in bond_charge_corrections]

    oe_molecule = smiles_to_molecule(smiles)

    allowed_elements = [1, 6, 7, 8, 9, 17, 35]

    # Skip heavy molecules for now.
    if oechem.OECount(oe_molecule, oechem.OEIsHeavy()) > 25:
        return {}

    # Ignore molecule which contain currently unsupported elements
    if any(
        atom.GetAtomicNum() not in allowed_elements for atom in oe_molecule.GetAtoms()
    ):
        return {}

    matched_smirks = []

    for smirks in bcc_smirks:

        if len(match_smirks(smirks, oe_molecule)) == 0:
            continue

        matched_smirks.append(smirks)

    if len(matched_smirks) == 0:
        return {}

    smiles = oechem.OECreateIsoSmiString(oe_molecule)
    return {smiles: matched_smirks}


def find_smiles_per_smirks() -> Dict[str, List[str]]:

    input_molecule_stream = oechem.oemolistream()
    assert input_molecule_stream.open("NCI-Open_2012-05-01.sdf")

    smiles = [
        oechem.OECreateCanSmiString(oe_molecule)
        for oe_molecule in input_molecule_stream.GetOEMols()
    ]

    smirks_per_smiles = {}

    with Pool(processes=4) as pool:

        for match in tqdm(pool.imap(process_molecule, smiles), total=len(smiles)):
            smirks_per_smiles.update(match)

    smiles_per_smirks = defaultdict(list)

    for smiles, smirks_matches in smirks_per_smiles.items():
        for smirks in smirks_matches:
            smiles_per_smirks[smirks].append(smiles)

    return smiles_per_smirks


def has_parity(smiles: str, bond_charge_corrections: List[BondChargeCorrection]):

    try:

        # Build a molecule from the smiles pattern.
        oe_molecule = smiles_to_molecule(smiles)

        # Attempt to resolve an unspecified stereochemistry issues
        # by just selecting one of the possible stereoisomers.
        unspecified_stereochemistry = any(
            entity.IsChiral() and not entity.HasStereoSpecified()
            for entity in [*oe_molecule.GetAtoms(), *oe_molecule.GetBonds()]
        )

        if unspecified_stereochemistry:

            enantiomer = next(
                iter(oeomega.OEFlipper(oe_molecule.GetActive(), 12, True))
            )
            oe_molecule = oechem.OEMol(enantiomer)

        # Generate a conformer for the molecule.
        conformers = OmegaELF10.generate(oe_molecule, max_conformers=1)

        # Generate a set of reference charges using the OpenEye implementation
        reference_charges = ChargeGenerator.generate(
            oe_molecule, conformers, ChargeSettings(theory="am1bcc")
        )

        # Generate of base set of AM1 charges using the OpenEye implementation
        am1_charges = ChargeGenerator.generate(
            oe_molecule, conformers, ChargeSettings(theory="am1")
        )

    except RechargeException as e:
        raise UnableToProcessError(str(e))

    try:
        assignment_matrix = BCCGenerator.build_assignment_matrix(
            oe_molecule, BCCSettings(bond_charge_corrections=bond_charge_corrections)
        )
    except UnableToAssignChargeError:
        print(f"Unable to build the assignment matrix for {smiles}.")
        return False

    try:
        charge_corrections = BCCGenerator.apply_assignment_matrix(
            assignment_matrix,
            BCCSettings(bond_charge_corrections=bond_charge_corrections),
        )
    except UnableToAssignChargeError:
        print(f"Unable to build the apply the assignment matrix for {smiles}.")
        return False

    implementation_charges = am1_charges + charge_corrections
    reference_charge_corrections = reference_charges - am1_charges

    # Check that their is no difference between the implemented and
    # reference charges.
    if not numpy.allclose(
        reference_charges, implementation_charges
    ) or not numpy.allclose(charge_corrections, reference_charge_corrections):
        return False

    return True


def main():

    # Build a dictionary of all of the smiles patterns which match
    # the current list of bond charge corrections.
    if not os.path.isfile("smiles_per_smirks.pkl"):

        smiles_per_smirks = find_smiles_per_smirks()

        with open("smiles_per_smirks.pkl", "wb") as file:
            pickle.dump(smiles_per_smirks, file)

    else:

        with open("smiles_per_smirks.pkl", "rb") as file:
            smiles_per_smirks = pickle.load(file)

    # Check if any smirks weren't covered.
    all_bcc_codes = {bcc.provenance["code"] for bcc in original_am1bcc_corrections()}
    covered_codes = {
        bcc.provenance["code"]
        for bcc in original_am1bcc_corrections()
        if bcc.smirks not in smiles_per_smirks
    }

    missed_codes = all_bcc_codes - covered_codes
    print(f"Codes without coverage: {missed_codes}")

    passed_smiles = set()
    failed_smiles = set()

    for smirks in smiles_per_smirks:

        print(f"Generating set for {smirks}")

        number_of_matches = 0

        current_n_passed = len(passed_smiles)
        current_n_failed = len(failed_smiles)

        while number_of_matches < 20 and len(smiles_per_smirks[smirks]) > 0:

            smiles = smiles_per_smirks[smirks].pop(0)

            if smiles in passed_smiles or smiles in failed_smiles:
                continue

            try:

                parity = has_parity(smiles, original_am1bcc_corrections())

            except UnableToProcessError:
                continue

            if parity:
                passed_smiles.add(smiles)
            else:
                failed_smiles.add(smiles)

            number_of_matches += 1

        print(
            f"{len(passed_smiles) - current_n_passed} molecules passed, "
            f"{len(failed_smiles) - current_n_failed} failed."
        )

    with open("passed-smiles.json", "w") as file:
        json.dump([*passed_smiles], file)
    with open("failed-smiles.json", "w") as file:
        json.dump([*failed_smiles], file)


if __name__ == "__main__":
    main()
