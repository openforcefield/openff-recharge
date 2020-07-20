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

import numpy
import pandas
from openeye import oechem

from openff.recharge.charges.bcc import BCCParameter
from openff.recharge.utilities.exceptions import InvalidSmirksError
from openff.recharge.utilities.openeye import call_openeye

logging.basicConfig()
logger = logging.getLogger(__name__)


def build_bond_charge_corrections(
    atom_codes: Dict[str, str],
    bond_codes: Dict[str, str],
    bcc_overrides: Dict[str, float],
    custom_bcc_smirks: Dict[str, str],
) -> List[BCCParameter]:

    # Convert the atom and bond codes into the six number codes used
    # in the AM1BCC paper.
    all_codes = [*custom_bcc_smirks]

    for first_atom_code in atom_codes:
        for bond_code in bond_codes:
            for last_atom_code in atom_codes:

                code = f"{first_atom_code}{bond_code}{last_atom_code}"

                if code not in all_codes:
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

        if code in custom_bcc_smirks:
            smirks = custom_bcc_smirks.pop(code)

        # Validate the smirks
        query = oechem.OEQMol()
        call_openeye(
            oechem.OEParseSmarts,
            query,
            smirks,
            exception_type=InvalidSmirksError,
            exception_kwargs={"smirks": smirks},
        )

        value = bcc_overrides.get(code, bcc_row["BCC"])

        bond_charge_corrections[code] = BCCParameter(
            smirks=smirks, value=value, provenance={"code": code}
        )

    assert len(custom_bcc_smirks) == 0

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
        # N3hdeloc Trivalent nitrogen with a highly delocalized lone pair
        "23": "[#7X3ar5,#7X3+1,#7X3+0$(*-[#6X3$(*=[#7X3+1])]),$([#7X3](-[#8X1-1])=[#8X1]),$([#7X3](=[#8X1])=[#8X1]):1]",
        # N3deloc Trivalent nitrogen with a delocalized lone pair
        "22": "[#7X2-1$(*-[#6X3$(*=[#8X1,#16X1])]),#7X3$(*-[#6X3$(*=[#8X1,#16X1])]):1]",
        # N2,3,4 Amine nitrogen
        "21": "[#7X4,#7X3,#7X2-1:1]",
        # N1,2 Univalent or cationic divalent nitrogen
        "25": "[#7X1,#7X2+1:1]",
        # N2 Neutral divalent nitrogen
        "24": "[#7X2+0,#7X2-1ar5:1]",
        # O1lact Double-bonded oxygen in a lactone or lactam
        "33": "[#8X1$(*=[#6r]@[#7r,#8r]):1]",
        # O1ester,acid Double-bonded oxygen in an ester or acid
        "32": "[#8X1$(*=[#6X3]-[#8X2]):1]",
        # O1,2 Univalent or divalent oxygen
        "31": "[#8X1,#8X2:1]",
        # # P3,4 Trivalent or tetravalent double-bonded phosphorus 42
        # "42": "[#15X3$(*=[*]),#15X4$(*=[*]):1]",
        # # P2,3 Divalent or trivalent phosphorus
        # "41": "[#15X2,#15X3:1]",
        # # S4 Tetravalent sulfur
        "53": "[#16X4:1]",
        # # S3 Trivalent sulfur
        "52": "[#16X3:1]",
        # # S1,2 Univalent or divalent sulfur
        "51": "[#16X1,#16X2:1]",
        # Si4 Tetravalent silicon
        "61": "[#14X4:1]",
        # F1 Fluorine
        "71": "[#9:1]",
        # Cl1 Chlorine
        "72": "[#17:1]",
        # Br1 Bromine
        "73": "[#35:1]",
        # # # I1 Iodine
        "74": "[#53:1]",
        # H1 Hydrogen
        "91": "[#1:1]",
    }
    bond_codes = {
        "01": "-",  # Single bond
        "02": "=",  # Double bond
        "03": "#",  # Triple bond
        "06": "-",  # 'Single' Dative bond
        "07": ":",  # 'Single' aromatic Bond
        "08": ":",  # 'Double' aromatic Bond
        "09": "~",  # Single bond with charge or delocalized bond
    }

    custom_bcc_smirks = {
        # 'Delocalised' S-O
        # "310951": "",
        # "310952": "",
        # "310953": "",

        # 'Delocalised' S-S
        # "510953": "",

        # 'Delocalised' S-C
        "110951": "[#6X4:1]-[#16X1-1:2]",
        "150951": "[#6X1,#6X2:1]-[#16X1-1:2]",
        "120951": "[#6X3$(*=[#6]):1]-[#16X1-1:2]",
        "130951": "[#6X3$(*=[#7,#15]):1]-[#16X1-1:2]",
        "140951": "[#6X3:1](~[#8X1,#16X1])(~[#16X1:2])",
        "170951": "[#6a$(*~[#7aX2,#8aX2]):1]-[#16X1-1:2]",
        "160951": "[#6a:1]-[#16X1-1:2]",

        # 'Delocalised' N-O
        "230931": "[$([#7X3](-[#8X1])=[#8X1]),$([#7X3](=[#8X1])=[#8X1]):1]~[#8X1:2]",
        # Dative N-O.
        "230631": "[#7X3+1:1]-[#8X1-1:2]",
        "220631": "[#7X3$(*-[#6X3$(*=[#8])]):1]-[#8X1-1:2]",
        "210631": "[#7X4+1:1]-[#8X1-1:2]",
        "250631": "[#7X1,#7X2+1:1]-[#8X1-1:2]",
        "240631": "[#7X2+0:1]-[#8X1-1:2]",
        # 'Delocalised' C-O.
        "110931": "[#6X4:1]-[#8X1-1:2]",
        "150931": "[#6X1,#6X2:1]-[#8X1-1:2]",
        "120931": "[#6X3$(*=[#6]):1]-[#8X1-1:2]",
        "130931": "[#6X3$(*=[#7,#15]):1]-[#8X1-1:2]",
        "140931": "[#6X3:1](~[#8X1,#16X1])(~[#8X1:2])",
        "170931": "[#6a$(*~[#7aX2,#8aX2]):1]-[#8X1-1:2]",
        "160931": "[#6a:1]-[#8X1-1:2]",
    }

    bcc_overrides = {"110112": 0.0024, "120114": -0.0172}

    bcc_parameters = build_bond_charge_corrections(
        atom_codes, bond_codes, bcc_overrides, custom_bcc_smirks
    )

    # Remove duplicate parameters caused by the duplication of the aromatic
    # bond type.
    bcc_parameters_per_smirks = {}
    unique_bcc_parameters = []

    for bcc_parameter in bcc_parameters:

        if bcc_parameter.smirks in bcc_parameters_per_smirks:

            assert numpy.isclose(
                bcc_parameter.value,
                bcc_parameters_per_smirks[bcc_parameter.smirks].value,
            )
            assert bcc_parameter.provenance["code"][2:4] in ["07", "08"]

            continue

        unique_bcc_parameters.append(bcc_parameter)
        bcc_parameters_per_smirks[bcc_parameter.smirks] = bcc_parameter

    with open("original-am1-bcc.json", "w") as file:

        json.dump(
            [bcc_parameter.dict() for bcc_parameter in unique_bcc_parameters], file
        )


if __name__ == "__main__":
    main()
