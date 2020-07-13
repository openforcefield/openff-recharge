"""A script to compare this frameworks AM1BCC implementation with the
built-in OpenEye implementation."""
import functools
import json
import os
from collections import defaultdict
from multiprocessing import Pool
from typing import Dict, List

from openeye import oechem
from tqdm import tqdm

from openff.recharge.charges.bcc import (
    BondChargeCorrection,
    compare_openeye_parity,
    original_am1bcc_corrections,
)
from openff.recharge.utilities.exceptions import RechargeException
from openff.recharge.utilities.openeye import match_smirks, smiles_to_molecule

output_stream = oechem.oeosstream()

oechem.OEThrow.SetOutputStream(output_stream)
oechem.OEThrow.Clear()


def apply_filter(smiles: str) -> bool:
    """Filters out molecules which should not be included in the
    validation set. This include molecules which contain elements
    which the bond charge corrections don't cover, over molecules
    which would be to heavily to validate against swiftly.

    Parameters
    ----------
    smiles
        The SMILES pattern to apply the filter to.

    Returns
    -------
        Whether to include the molecule or not.
    """

    oe_molecule = smiles_to_molecule(smiles)
    allowed_elements = [1, 6, 7, 8, 9, 17, 35]

    return oechem.OECount(oe_molecule, oechem.OEIsHeavy()) <= 25 and all(
        atom.GetAtomicNum() in allowed_elements for atom in oe_molecule.GetAtoms()
    )


def import_nci_molecule() -> List[str]:

    print("Importing NCI molecules.")

    input_molecule_stream = oechem.oemolistream()
    input_molecule_stream.open("NCI-Open_2012-05-01.sdf")

    nci_smiles = [
        oechem.OECreateIsoSmiString(oe_molecule)
        for oe_molecule in input_molecule_stream.GetOEMols()
    ]

    with Pool(processes=4) as pool:

        filter_generator = tqdm(
            pool.imap(apply_filter, nci_smiles), total=len(nci_smiles)
        )

        nci_smiles = [
            smiles for smiles, retain in zip(nci_smiles, filter_generator) if retain
        ]

    # Split any joined smiles (such as salts) and retain only the largest
    # of the encoded molecules.
    nci_smiles = [
        smiles
        if "." not in smiles
        else sorted(smiles.split("."), key=lambda x: len(x), reverse=True)[0]
        for smiles in nci_smiles
    ]

    return nci_smiles


def coverage_molecules() -> List[str]:

    return [
        "CC1=NN(C(=O)C1)C",
        "c1cc[nH]c1",
        "CC(=O)N",
        "C#[N+][O-]",
        "c1cc(oc1)C=O",
        "C=Cc1ccccc1",
        "[N+](=O)([O-])[O-]",
        "c1ccc2ccccc2c1",
        "C[N+](=CN)C",
        "c1ccc(cc1)C=O",
        "C[C@@H](C(=O)NO)N",
        "C=CN1CCCC1=O",
        "c1([nH]nnn1)N",
        "c1cc[n+](cc1)O",
        "C(=O)N",
        "[H]/N=C(\\c1ccncc1)/NO",
        "COOC",
        "C=CC=O",
        "C1CNC(=O)N(C1=O)O",
        "c1ccc2c(c1)cco2",
        "C[N+](=O)[O-]",
        "c1cc(oc1)CO",
        "C[O-]",
        "Cn1cccc1",
        "Cc1ccccc1",
        "[H]/N=C\\NO",
        "CC(=O)[O-]",
        "c1ccnc(c1)[O-]",
        "c1c[nH]nc1",
        "[H]/N=C(/NO)\\C/C(=N\\[H])/NO",
        "c1cc[n+](c(c1)N)CCC(=O)[O-]",
        "c1ccc(cc1)O",
        "c1ccc2c(c1)c3ccccc3o2",
        "c1coc2c1cco2",
        "CO",
        "[N+](=O)(O)[O-]",
        "N(=O)(=O)[O-]",
        "c1c(o[nH]c1=O)N",
        "c1cc(oc1)[O-]",
        "c1ccc2c(c1)c(co2)[O-]",
        "C=CC=C",
        "c1coc2c1occ2",
        "Cc1ccco1",
        "C#C[O-]",
        "c1cc[n-]c1",
        "c1ccoc1",
        "COC",
        "c1ccc(cc1)c2ccco2",
        "c1ccncc1",
        "c1cc([nH]c1)[O-]",
        "[H]/N=C(\\C)/[O-]",
        "C/C=C(/C)\\[O-]",
        "C=Cc1ccco1",
        "C(=O)(N)NO",
        "c1ccc(cc1)[O-]",
        "c1cc(oc1)O",
        "CN(=O)=O",
        "c1ccc(cc1)c2ccccc2",
        "C(C(=O)NO)N",
        "c1cc2cccc3c2c(c1)CC3",
        "[H]/N=C(/N)\\Nc1[nH]nnn1",
        "c1cc[nH+]cc1",
        "C[N+](C)(C)[O-]",
        "CONC(=O)N",
        "c1ccc2c(c1)cc[nH]2",
        "c1ccc(cc1)/N=C\\NO",
        "C=CO",
        "c1cocc1[O-]",
        "CC(=O)NO",
        "C[N+](=C)C",
        "C(=O)C=O",
        "C=C",
        "CC1=NC(=NC1=[N+]=[N-])Cl",
        "c1cc[n+](cc1)[O-]",
        "CN(C)O",
        "N(=O)(=O)O",
        "CC=O",
        "c1cc(oc1)c2ccco2",
        "CC",
        "C1C=CC(=O)C=C1",
        "C",
        "C(=O)O",
        "C[N+](=C)[O-]",
        # 220125
        "O=CN[N+]#C",
        # 150125
        "C#C[N+]#C",
        # 240171
        "FN=C",
        # # 250191 - Omega cannot generate a conformer.
        # "C#[NH+]",
        # 150171
        "FC#C",
        # 240125
        "C=N[N+]#C",
        # 220171
        "FNC=O",
        # 220631
        "[O-]NC=O",
        # 240631,
        "[O-]N=C",
        # 140125
        "O=C[N+]#C",
        # 130125
        "N=C[N+]#C",
    ]


def match_bcc_parameters(
    smiles: str, bond_charge_corrections: List[BondChargeCorrection]
) -> List[str]:
    """Returns the list of bond charge correction SMIRKS patterns which a molecule
    defined by it's SMILES pattern will exercise.

    Parameters
    ----------
    smiles
        The SMILES pattern to match against.
    bond_charge_corrections
        The bond charge correction parameters to match.
    Returns
    -------
        The SMIRKS patterns of the matched bond charge correction parameters
    """

    oe_molecule = smiles_to_molecule(smiles, True)

    matched_smirks = [
        bond_charge_correction.smirks
        for bond_charge_correction in bond_charge_corrections
        if len(match_smirks(bond_charge_correction.smirks, oe_molecule)) > 0
    ]

    return matched_smirks


def map_smiles_to_smirks(smiles: List[str]) -> Dict[str, List[str]]:
    """Builds a dictionary of the SMILES patterns which are
    exercised by a given set of bond charge corrections.

    Parameters
    ----------
    smiles
        The SMILES patterns to match to the bond charge corrections.

    Returns
    -------
        A dictionary with keys of the SMIRKS patterns of the bond charge
        corrections and values of lists of the SMILES patterns which exercise
        that particular SMIRKS.
    """

    print("Mapping smiles to smirks.")

    match_function = functools.partial(
        match_bcc_parameters, bond_charge_corrections=original_am1bcc_corrections()
    )

    with Pool(processes=4) as pool:

        smirks_per_smiles = tqdm(pool.imap(match_function, smiles), total=len(smiles))

        smiles_per_smirks = defaultdict(list)

        for smiles_pattern, smirks_matches in zip(smiles, smirks_per_smiles):
            for smirks in smirks_matches:
                smiles_per_smirks[smirks].append(smiles_pattern)

    return smiles_per_smirks


def main():

    # Construct a list of molecule from both the NCI 2012 Open set and
    # a list of hand curated SMILES patterns chosen to exercise the more
    # uncommon bond charge correction parameters, and determine which
    # bond charge corrections they exercise.
    if not os.path.isfile("smiles_per_smirks.json"):

        coverage_smiles = [*{*import_nci_molecule(), *coverage_molecules()}]
        smiles_per_smirks = map_smiles_to_smirks(coverage_smiles)

        with open("smiles_per_smirks.json", "w") as file:
            json.dump(smiles_per_smirks, file)

    else:

        with open("smiles_per_smirks.json", "r") as file:
            smiles_per_smirks = json.load(file)

    # Check which bond charge correction parameters weren't covered by the set.
    all_bcc_codes = {bcc.provenance["code"] for bcc in original_am1bcc_corrections()}

    covered_codes = {
        bcc.provenance["code"]
        for bcc in original_am1bcc_corrections()
        if bcc.smirks in smiles_per_smirks
    }

    missed_codes = all_bcc_codes - covered_codes
    print(f"Codes without coverage: {missed_codes}")

    # Select molecules from the above curated list and check whether the
    # charges generated by this framework match the OpenEye implementation.
    passed_smiles = set()
    failed_smiles = set()

    for smirks in smiles_per_smirks:

        print(f"Validating {smirks}")

        number_of_matches = 0

        current_n_passed = len(passed_smiles)
        current_n_failed = len(failed_smiles)

        while number_of_matches < 20 and len(smiles_per_smirks[smirks]) > 0:

            smiles = smiles_per_smirks[smirks].pop(0)

            if smiles in passed_smiles or smiles in failed_smiles:
                continue

            try:

                oe_molecule = smiles_to_molecule(smiles, guess_stereochemistry=True)
                identical_charges = compare_openeye_parity(oe_molecule)

            except RechargeException:
                continue

            if identical_charges:
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
