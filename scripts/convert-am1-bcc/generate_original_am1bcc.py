"""This script converts the original AM1BCC values [1]_ into the data model expected by
this framework.

References
----------
[1] Jakalian, A., Jack, D. B., & Bayly, C. I. (2002). Fast, efficient generation of
    high-quality atomic charges. AM1-BCC model: II. Parameterization and validation.
    Journal of computational chemistry, 23(16), 1623–1641.
"""
import click
import json
import itertools
import logging

import numpy as np
import pandas as pd
from openeye import oechem

from openff.recharge.charges.bcc import BCCParameter

logging.basicConfig()
logger = logging.getLogger(__name__)


GENERAL_ATOM_CODES = {
    # from more general to more specific
    # patterns will get matched with more specific ones so order is important!
    "X": {
        # === HALOGENS ===
        # F1 Fluorine
        "71": [r"[#9:1]"],
        # Cl1 Chlorine
        "72": [r"[#17:1]"],
        # Br1 Bromine
        "73": [r"[#35:1]"],
        # I1 Iodine
        "74": [r"[#53:1]"],
        # === OTHER ===
        # Si4 Tetravalent silicon
        "61": [r"[#14X4:1]"],
        # H1 Hydrogen
        "91": [r"[#1:1]"],
    },
    "S": {
        # === SULFUR ===
        # S1,2 Univalent or divalent sulfur
        "51": [r"[#16X1:1]", "[#16X2:1]"],
        # S3 Trivalent sulfur
        "52": [r"[#16X3:1]"],
        # S4 Tetravalent sulfur
        "53": [r"[#16X4:1]"],
    },
    "P": {
        # === PHOSPHORUS ===
        # P2,3 Divalent or trivalent phosphorus
        # "41": [r"[#15:1]", "[#15X2:1]", "[15X1:1]", "[#15X3:1]"],
        "41": [r"[#15:1]"],
        # P3,4 Trivalent or tetravalent double-bonded phosphorus
        # "42": [r"[15X4:1]", "[#15X3;$(*=[*]):1]"],
        "42": [r"[#15X4,#15X3$(*=[*]):1]"],
    },
    "O": {
        # === OXYGEN ===
        # O1,2 Univalent or divalent oxygen
        # "31": [r"[#8:1]", "[#8X1:1]", "[#8X2:1]"],
        "31": [r"[#8:1]"],
        # O1ester,acid Double-bonded oxygen in an ester or acid
        "32": [r"[#8X1$(*=[#6X3]-[#8X2]):1]"],
        # O1lact Double-bonded oxygen in a lactone or lactam
        "33": [r"[#8X1$(*=[#6r]@[#7r,#8r]):1]"],
    },
    "C": {
        # === CARBON ===
        # C4 Tetravalent carbon
        "11": [r"[#6X4:1]"],
        # C1,2 Univalent or divalent carbon
        "15": [r"[#6X1:1]", "[#6X2:1]"],
        # C3=C Trivalent carbon, double bonded to carbon
        "12": [r"[#6X3:1]"],
        # Car Aromatic carbon
        "16": [
            r"[#6a:1]",
            # or planar ring with two continuous single bonds and at least two double bonds
            r"[#6X3R$([*](-,:[R]-,=,:[R])=,:[R]):1]",
        ],
        # Carlp Aromatic carbon bonded to an aromatic oxygen or nitrogen with a lone pair
        "17": [
            r"[#6aX3$(*~[#7aX2,#8aX2]):1]",
            r"[#6X3R$(*~[#7aX2,#8aX2])&$([*](-,:[R]-,=,:[R])=,:[R]):1]",
        ],
        # C3=N,P Trivalent carbon, double bonded to nitrogen or phosphorus
        "13": [
            r"[#6X3$(*=[#7,#15]):1]",
        ],
        # C3=O,S Trivalent carbon, double bonded to oxygen or sulfur
        "14": [r"[#6X3$(*=[#8X1,#8X2+1,#16+0]):1]"],
        "12b": [r"[#6X3A$(*=[#6]):1]"],
    },
    "N": {
        # === NITROGEN ===
        # N1,2 Univalent or cationic divalent nitrogen
        "25": [r"[#7:1]", r"[#7X1,#7X2+1:1]"],
        # N2 Neutral divalent nitrogen
        "24": [r"[#7X2+0,#7X2-1ar5:1]"],
        # N2,3,4 Amine nitrogen
        "21": [
            r"[#7X4:1]",
            r"[#7X3:1]",
            r"[#7X2-1A:1]",
        ],  # "[#7X2$([*](-,:[*])-,:[*]):1]"],
        # N3deloc Trivalent nitrogen with a delocalized lone pair
        "22": [
            r"[#7X2$(*(-,:[*])-,:[#6X3$(*=[#8,#16])]):1]",  # N2-C(=O,S),
        ],
        "23": [
            r"[#7X3r5$([*](-,:[r5])(-,:[r5])-[!#1&!#6X4&!#6a]):1]",
            # "[#7X3R$([*](-[!#1&!])(-,:[R]=,:[R])-,:[R]=,:[R]):1]",
            r"[#7X3r5$([*]1(-,:[r5]=,:[r5]-,:[r5]=,:[r5]:,-1)-[!#1&!#6a!#6X4]):1]",
            r"[#7X3+1:1]",
            r"[#7X3$([*](~[#8X1])~[#8X1]):1]",
            # "[#7X3$([*]~[#7X2]=[#7X3]):1]",
            r"[#7X3+0$(*-[#6X3$(*=[#7X3+1])]):1]",
        ],
        "22b": [
            r"[#7X3+0$(*-[#6X3$(*(=,:[#8,#16])-[!#8&!#16])]):1]",  # amide
            # "[#7X3+0$(*-[#6X3$(*(=[#8X2+1]-[#7X2-1])-[!#8&!#16])]):1]",
        ],
        "23b": [
            r"[#7X3ar5:1]",
        ],
    },
}

SPECIFIC_PATTERNS = {
    # P-O
    r"[#8X1-1:1]~[#15&!$(*=*):2]": "310941",
    r"[#8X2:1]-[#15X4+0,#15X3;!$(*=*):2]": "310141",
    r"[#8X1:1]-,=[#15X4$([*](-[#8X1-1])-[#8X1-1,#8X2,#16X2]):2]": "310942",
    r"[#8X1-1:1]-[#15X4$([*](=[#8X1])-[#8X1-1]):2]": "310942",
    r"[#8X1-1:1]-,:[#15X4$([*](-[#8X1-1])(-[#8X1-1])=[#8]):2]": "310942",
    r"[#8X1:1]=[#15X4$([*](-[#8X1-1])(-[#6])-[#6,#1]):2]": "310942",
    # ...
    # 'Delocalised' S-O
    r"[#16X1,#16X2;$(*=[#8X1]);$(*-[#8X1-1]):2]~[#8X1:1]": "310951",
    r"[#16X3$(*=[#8X1]);$(*-[#8X1-1]):2]~[#8X1:1]": "310952",
    r"[#16X4$(*~[#8X1]);$(*-[#8X1-1]):2]~[#8X1:1]": "310953",
    # 'Delocalised' S-S
    r"[#16X1-1$([*]-S),#16X1+0$([*]=S):1]-,=[#16X4:2]": "510953",
    # 'Delocalised' S-C
    r"[#6a:1]-[#16X1-1:2]": "160951",
    r"[#6a$(*~[#7aX2,#8aX2]):1]-[#16X1-1:2]": "170951",
    r"[#6X3:1](~[#8X1,#16X1])(~[#16X1:2])": "140951",
    r"[#6X3$(*=[#7,#15]):1]-[#16X1-1:2]": "130951",
    r"[#6X3$(*=[#6]):1]-[#16X1-1:2]": "120951",
    r"[#6X1,#6X2:1]-[#16X1-1:2]": "150951",
    r"[#6X1,#6X2$([*]=[#8,#16]):1]=[#16:2]": "150951",
    r"[#6X4:1]-[#16X1-1:2]": "110951",
    # 'Delocalised' C-O
    r"[#6a:1]-[#8X1-1:2]": "160931",
    r"[#6a$(*~[#7aX2,#8aX2]):1]-[#8X1-1:2]": "170931",
    r"[#6X3:1](~[#8X1,#16X1])(~[#8X1:2])": "140931",
    r"[#6X3$(*=[#7,#15]):1]-[#8X1-1:2]": "130931",
    r"[#6X3$(*=[#6]):1]-[#8X1-1:2]": "120931",
    r"[#6X1,#6X2:1]-[#8X1-1:2]": "150931",
    r"[#6X2$([*]=[#8,#16]):1]=[#8X1:2]": "150931",
    r"[#6X4:1]-[#8X1-1:2]": "110931",
    # 'Delocalised' N-O
    r"[#7X2+0:1]-[#8X1-1:2]": "240631",
    r"[#7X3+1:1]-[#8X1-1:2]": "230631",
    (
        r"[#7X4+1,#7X3+0$([*]-[#6X3]=,:[#7X2,#6X3]),#7X3+0$([*]-[#6X3]-,=,:[#7a,#6a]),"
        r"#7X3+0$(*-[#6X4,#1]),#7X3+0$([*](-[#7X3])-[#7X3]),"
        r"#7X3+0$([*](-[#8])-[#8]):1]-[#8X1-1:2]"
    ): "210631",
    r"[#7X3+0$(*-[#6X3$(*=[#8,#16])]):1]-[#8X1-1:2]": "220631",
    (
        r"[$([#7X3](-[#8X1])=[#8X1]),$([#7X3](=[#8X1])=[#8X1]),"
        r"$([#7X3](-[#8X1])-[#8X1]):1]~[#8X1:2]"
    ): "230931",
}

GENERAL_BOND_CODES = {
    "09": "~",  # Single bond with charge or delocalized bond
    "08": ":",  # 'Double' aromatic Bond
    "07": ":",  # 'Single' aromatic Bond
    "06": "-",  # 'Single' Dative bond
    "03": "#",  # Triple bond
    "02": "=",  # Double bond
    "01": "-",  # Single bond
}

BCC_OVERRIDES = {
    "110112": 0.0024,
    "120114": -0.0172,
}


def build_bccs(
    bcc_value_file: str = "am1bcc.csv",
):
    bcc_frame = pd.read_csv(bcc_value_file)
    bcc_frame[r"Code"] = bcc_frame[r"Code"].astype(str)
    bcc_frame[r"BCC"] = bcc_frame[r"BCC"].round(4)

    BCC_VALUES = dict(zip(bcc_frame[r"Code"], bcc_frame[r"BCC"]))
    BCC_VALUES.update(BCC_OVERRIDES)

    flat_atom_codes = {}
    for element_smirks in GENERAL_ATOM_CODES.values():
        for atom_code, all_atom_smirks in element_smirks.items():
            not_atom_map = [smirks[1:-3] for smirks in all_atom_smirks]
            final_smirks = f"[{','.join(not_atom_map)}:1]"
            flat_atom_codes[final_smirks] = atom_code[:2]

    # create bcc patterns from less to more specific
    bcc_patterns = []
    for (atom_smirks1, atom_code1), (
        atom_smirks2,
        atom_code2,
    ) in itertools.combinations_with_replacement(flat_atom_codes.items(), 2):
        for bond_code, bond_smirks in GENERAL_BOND_CODES.items():
            smirks = atom_smirks1 + bond_smirks + atom_smirks2.replace(":1", ":2")
            code = atom_code1 + bond_code + atom_code2
            bcc_patterns.append((smirks, code))

            if atom_smirks2 != atom_smirks1:
                smirks2 = atom_smirks2 + bond_smirks + atom_smirks1.replace(":1", ":2")
                code2 = atom_code2 + bond_code + atom_code1
                bcc_patterns.append((smirks2, code2))

    for smirks, code in SPECIFIC_PATTERNS.items():
        bcc_patterns.append((smirks, code))

    # create BCCs from more to less specific
    bond_charge_corrections = []
    unique_patterns = set()
    for smirks, code in bcc_patterns[::-1]:
        if smirks in unique_patterns:
            continue
        if code not in BCC_VALUES:
            if code[:2] == code[-2:]:
                value = 0
            else:
                continue
        else:
            value = BCC_VALUES[code]

        # Validate the smirks
        query = oechem.OEQMol()
        assert oechem.OEParseSmarts(query, smirks)

        bcc = BCCParameter(smirks=smirks, value=value, provenance={"code": code})
        bond_charge_corrections.append(bcc)
        unique_patterns.add(smirks)

    # TODO: reverse if we reverse the order of BCCs to go from less to more specific
    return bond_charge_corrections[::-1]


@click.command()
@click.option(
    "--bcc-value-file",
    default="am1bcc.csv",
    help="The name of the file containing the BCC values.",
)
@click.option(
    "--output-file",
    default="openeye-am1-bcc.json",
    help="The name of the output file to write the BCCs to.",
)
def create_openeye_bccs(
    bcc_value_file: str = "am1bcc.csv",
    output_file: str = "openeye-am1-bcc.json",
):
    bcc_parameters = build_bccs(bcc_value_file=bcc_value_file)

    # Remove duplicate parameters caused by the duplication of the aromatic
    # bond type.
    bcc_parameters_per_smirks = {}
    unique_bcc_parameters = []

    for bcc_parameter in bcc_parameters:
        if bcc_parameter.smirks in bcc_parameters_per_smirks:
            code = bcc_parameter.provenance[r"code"]
            assert np.isclose(
                bcc_parameter.value,
                bcc_parameters_per_smirks[bcc_parameter.smirks].value,
            ), (bcc_parameter, bcc_parameters_per_smirks[bcc_parameter.smirks])
            is_aromatic = code[2:4] in [r"07", "08"]
            is_self_to_self = code[:2] == code[-2:]
            assert is_aromatic or is_self_to_self, bcc_parameter
            continue
        unique_bcc_parameters.append(bcc_parameter)
        bcc_parameters_per_smirks[bcc_parameter.smirks] = bcc_parameter

    with open(output_file, "w") as f:
        json.dump(
            # reverse for now while recharge still goes from more to less specific
            {"parameters": [bcc.dict() for bcc in bcc_parameters][::-1]},
            f,
            indent=4,
        )


if __name__ == "__main__":
    create_openeye_bccs()
