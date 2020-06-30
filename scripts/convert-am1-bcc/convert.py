"""This script converts the original AM1BCC values [1]_ into the data model expected by
this framework.

References
----------
[1] Jakalian, A., Jack, D. B., & Bayly, C. I. (2002). Fast, efficient generation of
    high-quality atomic charges. AM1-BCC model: II. Parameterization and validation.
    Journal of computational chemistry, 23(16), 1623–1641.
"""
import json
import logging
from typing import Dict, List

import pandas

from openff.recharge.models import BondChargeCorrection

logging.basicConfig()
logger = logging.getLogger(__name__)


def build_bond_charge_corrections(
    atom_codes: Dict[str, str],
    bond_codes: Dict[str, str],
    bcc_overrides: Dict[str, float],
) -> List[BondChargeCorrection]:

    # Convert the atom and bond codes into the six number codes used
    # in the AM1BCC paper.
    all_codes = []

    for first_atom_code in atom_codes:
        for bond_code in bond_codes:
            for last_atom_code in atom_codes:

                code = f"{first_atom_code}{bond_code}{last_atom_code}"
                all_codes.append(code)

    # Remove any BCCs defined for bond or atom codes which haven't yet been
    # specified.
    bcc_frame = pandas.read_csv("am1bcc.csv")

    bcc_frame["BCC"] = bcc_frame["BCC"].round(4)
    bcc_frame = bcc_frame.sort_values(by=["Index"])

    unconverted_codes = bcc_frame[~bcc_frame["Code"].isin(all_codes)]

    for unconverted_code in unconverted_codes["Code"].unique():
        logger.warning(f"{unconverted_code} was not converted.")

    bcc_frame = bcc_frame[bcc_frame["Code"].isin(all_codes)]

    # Convert the data frame into a collection of correction objects.
    bond_charge_corrections = {}

    for _, bcc_row in bcc_frame.iterrows():

        code = str(bcc_row["Code"])[0:6]

        first_atom_code = code[0:2]
        bond_code = code[2:4]
        last_atom_code = code[4:6]

        smirks = (
            f"{atom_codes[first_atom_code]}"
            f"{bond_codes[bond_code]}"
            f"{atom_codes[last_atom_code].replace(':1', ':2')}"
        )

        value = bcc_overrides.get(code, bcc_row["BCC"])

        bond_charge_corrections[code] = BondChargeCorrection(
            smirks=smirks, value=value, provenance={"code": code}
        )

    return [
        bond_charge_corrections[code]
        for code in all_codes
        if code in bond_charge_corrections
    ]


def main():

    atom_codes = {
        # C4 Tetravalent carbon
        "11": "[#6X4:1]",
        # # C1,2 Univalent or divalent carbon
        "15": "[#6X1,#6X2:1]",
        # C3=C Trivalent carbon, double bonded to carbon
        "12": "[#6X3$(*=[#6]):1]",
        # C3=N,P Trivalent carbon, double bonded to nitrogen or phosphorus
        "13": "[#6X3$(*=[#7,#15]):1]",
        # C3=O,S Trivalent carbon, double bonded to oxygen or sulfur
        "14": "[#6X3$(*=[#8,#16]):1]",
        # Carlp Aromatic carbon bonded to an aromatic oxygen or nitrogen with a lone pair
        "17": "[#6a$(*~[#7aX2,#8aX2]):1]",
        # Car Aromatic carbon
        "16": "[#6a:1]",
        # O1ester,acid Double-bonded oxygen in an ester or acid
        "32": "[#8X1$(*=[#6X3]-[#8X2]):1]",
        # O1,2 Univalent or divalent oxygen
        "31": "[#8X1,#8X2:1]",
        # H1 Hydrogen
        "91": "[#1:1]",
    }
    bond_codes = {
        "01": "-",  # Single bond
        "02": "=",  # Double bond
        "03": "#",  # Triple bond
        "07": ":",  # 'Single' aromatic Bond
        "08": ":",  # 'Double' aromatic Bond
    }

    bcc_overrides = {"110112": 0.0024, "120114": -0.0172}

    bond_charge_corrections = build_bond_charge_corrections(
        atom_codes, bond_codes, bcc_overrides
    )
    bond_charge_corrections = [
        bond_charge_correction.dict()
        for bond_charge_correction in bond_charge_corrections
    ]

    with open("original-am1-bcc.json", "w") as file:
        json.dump(bond_charge_corrections, file)


if __name__ == '__main__':
    main()
