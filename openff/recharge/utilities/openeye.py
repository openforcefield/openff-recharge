"""A set of utilities to aid in interfacing with the OpenEye toolkits."""
import logging
import re
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple, Type, TypeVar, Union

import numpy
from openeye import oechem, oeomega

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


def smiles_to_molecule(
    smiles: str, guess_stereochemistry: bool = False
) -> oechem.OEMol:
    """Attempts to parse a smiles pattern into a molecule object.

    Parameters
    ----------
    smiles
        The smiles pattern to parse.
    guess_stereochemistry
        If true, the stereochemistry of molecules which is not
        defined in the SMILES pattern will be guessed using the
        OpenEye ``OEFlipper`` utility.

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

    unspecified_stereochemistry = any(
        entity.IsChiral() and not entity.HasStereoSpecified()
        for entity in [*oe_molecule.GetAtoms(), *oe_molecule.GetBonds()]
    )

    if unspecified_stereochemistry and guess_stereochemistry:

        stereoisomer = next(iter(oeomega.OEFlipper(oe_molecule.GetActive(), 12, True)))
        oe_molecule = oechem.OEMol(stereoisomer)

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


def _oe_draw_system(
    image,
    smiles: Tuple[str, ...],
    molecule_width: float,
    molecule_height: float,
    offset,
):
    from openeye import oechem, oedepict

    render_options = oedepict.OE2DMolDisplayOptions(
        molecule_width, molecule_height, oedepict.OEScale_AutoScale
    )

    for index, molecule_smiles in enumerate(smiles):

        oe_molecule = oechem.OEMol()
        oechem.OEParseSmiles(oe_molecule, molecule_smiles)
        oedepict.OEPrepareDepiction(oe_molecule)

        molecule_offset = offset + oedepict.OE2DPoint(molecule_width * index, 0.0)

        frame = oedepict.OEImageFrame(
            image, molecule_width, molecule_height, molecule_offset
        )

        molecule_display = oedepict.OE2DMolDisplay(oe_molecule, render_options)
        oedepict.OERenderMolecule(frame, molecule_display, False)

    width = molecule_width * len(smiles)

    frame = oedepict.OEImageFrame(image, width, molecule_height, offset)
    oedepict.OEDrawBorder(frame, oedepict.OELightGreyPen)


def smiles_to_image_grid(
    smiles_patterns: List[Union[str, Tuple[str, ...]]],
    file_path: str,
    image_width: int,
    molecules_per_row: int,
):
    """Creates a an image grid of a list of substances described by their
    SMILES patterns. These may either be the smiles pattern of a single
    molecule or a tuple of the smiles in a mixture.

    Parameters
    ----------
    smiles_patterns
        The SMILES patterns of the molecules. The list can either contain
        a list of single SMILES strings, or a tuple of SMILES strings. If
        tuples of SMILES are provided, these smiles will be grouped together
        in the output. All tuples in the list must have the same length.
    file_path
        The file path to save the pdf to.
    image_width
        The width of the image grid.
    molecules_per_row
        The maximum number of molecules (not mixtures) to draw per row.
    """

    if len(smiles_patterns) == 0:
        return

    # try:
    from openeye import oechem, oedepict

    # except ImportError as e:
    #     raise MissingOptionalDependency(e.path, False)
    #
    # unlicensed_library = (
    #     "openeye.oechem"
    #     if not oechem.OEChemIsLicensed()
    #     else "openeye.oedepict"
    #     if not oedepict.OEDepictIsLicensed()
    #     else None
    # )
    #
    # if unlicensed_library is not None:
    #     raise MissingOptionalDependency(unlicensed_library, True)
    # Make sure the list of smiles is a list of tuple of strings.
    system_smiles_patterns = {
        tuple(sorted(smiles if isinstance(smiles, tuple) else (smiles,)))
        for smiles in smiles_patterns
    }

    system_smiles_patterns = sorted(
        system_smiles_patterns, key=lambda x: len(x), reverse=True
    )

    if len(system_smiles_patterns[0]) > molecules_per_row:
        raise NotImplementedError()

    # Split the systems into a grid ready to be rendered.
    row_index = 0
    column_index = 0

    system_grid = defaultdict(dict)

    for system_index, system_smiles in enumerate(system_smiles_patterns):

        system_grid[row_index][column_index] = system_smiles

        if system_index == len(system_smiles_patterns) - 1:
            break

        column_index += len(system_smiles)

        if (
            column_index + len(system_smiles_patterns[system_index + 1])
            > molecules_per_row
        ):

            column_index = 0
            row_index += 1

    # Determine the width / height per molecule and hence the final image size.
    molecule_width = image_width / molecules_per_row
    molecule_height = molecule_width

    image_height = len(system_grid) * molecule_height

    image = oedepict.OEImage(image_width, image_height)

    for row_index, columns in system_grid.items():

        for column_index, system_smiles in columns.items():

            _oe_draw_system(
                image,
                system_smiles,
                molecule_width,
                molecule_height,
                oedepict.OE2DPoint(
                    column_index * molecule_width, row_index * molecule_height
                ),
            )

    oedepict.OEWriteImage(file_path, image)
