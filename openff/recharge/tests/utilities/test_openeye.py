import numpy
import pytest
from openeye import oechem

from openff.recharge.utilities.exceptions import (
    InvalidSmirksError,
    MoleculeFromSmilesError,
)
from openff.recharge.utilities.openeye import (
    match_smirks,
    molecule_to_conformers,
    smiles_to_molecule,
)


def test_smiles_to_molecule():
    """Tests that the `smiles_to_molecule` behaves as expected."""

    # Test a smiles pattern which should be able to be parsed.
    smiles_to_molecule("CO")

    # Test a bad smiles pattern.
    with pytest.raises(MoleculeFromSmilesError) as error_info:
        smiles_to_molecule("X")

    assert error_info.value.smiles == "X"


def test_guess_stereoisomer():
    """Tests that the stereochemistry of a molecule can be
    randomly guessed when not provided in the SMILES pattern."""

    oe_molecule = smiles_to_molecule("C(F)(Cl)Br")
    assert oechem.OECreateIsoSmiString(oe_molecule) == "C(F)(Cl)Br"

    oe_molecule = smiles_to_molecule("C(F)(Cl)Br", guess_stereochemistry=True)

    assert oechem.OECreateIsoSmiString(oe_molecule) in [
        "[C@H](F)(Cl)Br",
        "[C@@H](F)(Cl)Br",
    ]


def test_match_smirks():
    """Test that the correct exception is raised when an invalid smirks
    pattern is provided to `match_smirks`."""

    # Test indexed matching
    matches = match_smirks("[#6:1]-[#1:2]", smiles_to_molecule("C"), unique=True)

    assert len(matches) == 4
    assert all(match[0] == 0 for match in matches)
    assert all(match[1] != 0 for match in matches)

    # Test unique matching
    matches = match_smirks("c1ccccc1", smiles_to_molecule("c1ccccc1C"), unique=True)
    assert len(matches) == 1

    # Test not unique matching
    matches = match_smirks("c1ccccc1", smiles_to_molecule("c1ccccc1C"), unique=False)
    assert len(matches) == 12


def test_match_smirks_invalid():
    """Test that the correct exception is raised when an invalid smirks
    pattern is provided to `match_smirks`."""

    with pytest.raises(InvalidSmirksError) as error_info:
        match_smirks("X", smiles_to_molecule("C"))

    assert error_info.value.smirks == "X"


def test_molecule_to_conformer():
    """Test that the `molecule_to_conformers` function returns
    a non-zero numpy array of the correct shape."""

    oe_molecule = smiles_to_molecule("[H][H]")
    oe_molecule.DeleteConfs()

    conformer = numpy.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    oe_molecule.NewConf(oechem.OEFloatArray(conformer.flatten()))

    conformers = molecule_to_conformers(oe_molecule)
    assert len(conformers) == 1

    assert conformers[0].shape == conformer.shape
    assert numpy.allclose(conformers[0], conformer)
