"""A set of utilities to aid in interfacing with the OpenEye toolkits."""
import logging
import re
from typing import Any, Callable, Dict, List, Type, TypeVar

import numpy
from openeye import oechem

from openff.recharge.utilities.exceptions import (
    InvalidSmirksError,
    MoleculeFromSmilesError,
    RechargeException,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


def call_openeye(
    oe_callable: Callable[[T], bool],
    *args: T,
    exception_type: Type[RechargeException] = RuntimeError,
    exception_kwargs: Dict[str, Any] = None
):
    """Wraps a call to an OpenEye function, either capturing the output in an
    exception if the function does not complete successfully, or redirecting it
    to the logger.

    Parameters
    ----------
    oe_callable
        The OpenEye function to call.
    args
        The arguments to pass to the OpenEye function.
    exception_type:
        The type of exception to raise when the function does not
        successfully complete.
    exception_kwargs
        The keyword arguments to pass to the exception.
    """

    if exception_kwargs is None:
        exception_kwargs = {}

    output_stream = oechem.oeosstream()

    oechem.OEThrow.SetOutputStream(output_stream)
    oechem.OEThrow.Clear()

    status = oe_callable(*args)

    oechem.OEThrow.SetOutputStream(oechem.oeerr)

    output_string = output_stream.str().decode("UTF-8")

    output_string = output_string.replace("Warning: ", "")
    output_string = re.sub("^: +", "", output_string, flags=re.MULTILINE)
    output_string = re.sub("\n$", "", output_string)

    if not status:

        # noinspection PyArgumentList
        raise exception_type("\n" + output_string, **exception_kwargs)

    elif len(output_string) > 0:
        logger.debug(output_string)


def smiles_to_molecule(smiles: str) -> oechem.OEMol:
    """Attempts to parse a smiles pattern into a molecule object.

    Parameters
    ----------
    smiles: str
        The smiles pattern to parse.

    Returns
    -------
    The parsed molecule.
    """

    oe_molecule = oechem.OEMol()

    call_openeye(
        oechem.OESmilesToMol,
        oe_molecule,
        smiles,
        exception_type=MoleculeFromSmilesError,
        exception_kwargs={"smiles": smiles},
    )
    call_openeye(
        oechem.OEAddExplicitHydrogens,
        oe_molecule,
        exception_type=MoleculeFromSmilesError,
        exception_kwargs={"smiles": smiles},
    )
    call_openeye(
        oechem.OEPerceiveChiral,
        oe_molecule,
        exception_type=MoleculeFromSmilesError,
        exception_kwargs={"smiles": smiles},
    )

    return oe_molecule


def match_smirks(
    smirks: str, oe_molecule: oechem.OEMol, unique: bool = False
) -> List[Dict[int, int]]:
    """Attempt to find the indices (optionally unique) of all atoms which
    match a particular SMIRKS pattern.

    Parameters
    ----------
    smirks
        The SMIRKS pattern to match.
    oe_molecule
        The molecule to match against.
    unique
        Whether to return back only unique matches.

    Returns
    -------
        A list of all the matches where each match is stored as a dictionary of
        the smirks indices and their corresponding matched atom indices.
    """

    query = oechem.OEQMol()
    call_openeye(
        oechem.OEParseSmarts,
        query,
        smirks,
        exception_type=InvalidSmirksError,
        exception_kwargs={"smirks": smirks},
    )

    substructure_search = oechem.OESubSearch(query)
    substructure_search.SetMaxMatches(0)

    matches = []

    for match in substructure_search.Match(oe_molecule, unique):

        matched_indices = {
            atom_match.pattern.GetMapIdx() - 1: atom_match.target.GetIdx()
            for atom_match in match.GetAtoms()
            if atom_match.pattern.GetMapIdx() != 0
        }

        matches.append(matched_indices)

    return matches


def molecule_to_conformers(oe_molecule: oechem.OEMol) -> List[numpy.ndarray]:
    """Extracts the conformers of a molecule and stores them
    inside of a numpy array.

    Parameters
    ----------
    oe_molecule
        The molecule to extract the conformers from.

    Returns
    -------
        A list of the extracted conformers, where each
        conformer is a numpy array with shape=(n_atoms, 3).
    """

    conformers = []

    for oe_conformer in oe_molecule.GetConfs():

        conformer = numpy.zeros((oe_molecule.NumAtoms(), 3))

        for atom_index, coordinates in oe_conformer.GetCoords().items():
            conformer[atom_index, :] = coordinates

        conformers.append(conformer)

    return conformers
