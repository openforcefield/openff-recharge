from typing import Tuple

import numpy
import pytest
from openff.toolkit import Molecule
from openff.units import unit

from openff.recharge.charges.exceptions import ChargeAssignmentError
from openff.recharge.charges.qc import (
    QCChargeGenerator,
    QCChargeSettings,
    QCChargeTheory,
)
from openff.toolkit._tests.utils import (
    requires_openeye,
)


@pytest.fixture(scope="module")
def methane() -> Tuple[Molecule, numpy.ndarray]:
    molecule: Molecule = Molecule.from_mapped_smiles(
        "[C:1]([H:2])([H:3])([H:4])([H:5])"
    )
    molecule.generate_conformers(n_conformers=1)

    conformer = molecule.conformers[0].m_as(unit.angstrom)

    return molecule, conformer


def test_check_connectivity(methane):
    molecule, conformer = methane
    conformer = conformer * unit.angstrom

    QCChargeGenerator._check_connectivity(molecule, conformer)

    with pytest.raises(
        ChargeAssignmentError, match="The connectivity of the molecule changed"
    ):
        QCChargeGenerator._check_connectivity(molecule, conformer * 10.0)


def test_symmetrize_charges(methane):
    molecule, _ = methane

    actual_charges = QCChargeGenerator._symmetrize_charges(
        molecule, numpy.array([[-10.0], [1.0], [2.0], [3.0], [4.0]])
    )
    assert actual_charges.shape == (5, 1)

    assert numpy.allclose(
        actual_charges, numpy.array([[-10.0], [2.5], [2.5], [2.5], [2.5]])
    )


@pytest.mark.parametrize("theory", ["GFN1-xTB"])
def test_generate_xtb_charges(methane, theory: QCChargeTheory):
    molecule, conformer = methane
    conformer = conformer * unit.angstrom

    default_charges = QCChargeGenerator._generate_xtb_charges(
        molecule, conformer, QCChargeSettings(theory=theory)
    )
    assert default_charges.shape == (5, 1)
    assert not numpy.allclose(default_charges, 0.0)

    asym_charges = QCChargeGenerator._generate_xtb_charges(
        molecule, conformer, QCChargeSettings(theory=theory, symmetrize=False)
    )
    assert len({*default_charges[1:].flatten()}) == 1
    assert len({*asym_charges[1:].flatten()}) == 4

    unopt_charges = QCChargeGenerator._generate_xtb_charges(
        molecule, conformer, QCChargeSettings(theory=theory, optimize=False)
    )
    assert not numpy.allclose(default_charges, unopt_charges)


@requires_openeye
@pytest.mark.parametrize("theory", ["am1", "am1bcc"])
def test_generate_omega_charges(methane, theory):
    molecule, conformer = methane

    default_charges = QCChargeGenerator._generate_omega_charges(
        molecule, conformer, QCChargeSettings(theory=theory)
    )
    assert default_charges.shape == (5, 1)
    assert not numpy.allclose(default_charges, 0.0)

    asym_charges = QCChargeGenerator._generate_omega_charges(
        molecule, conformer, QCChargeSettings(theory=theory, symmetrize=False)
    )
    assert len({*default_charges[1:].flatten()}) == 1
    assert len({*asym_charges[1:].flatten()}) == 4

    unopt_charges = QCChargeGenerator._generate_omega_charges(
        molecule, conformer, QCChargeSettings(optimize=False)
    )
    assert not numpy.allclose(default_charges, unopt_charges)


@requires_openeye
@pytest.mark.parametrize("theory", ["am1", "am1bcc", "GFN1-xTB"])
def test_generate_charges(theory: QCChargeTheory, methane):
    """Ensure that charges can be generated for a simple molecule using
    the `QCChargeGenerator` class."""

    molecule, conformer = methane

    charges = QCChargeGenerator.generate(
        molecule, [conformer * unit.angstrom], QCChargeSettings(theory=theory)
    )
    assert charges.shape == (5, 1)
    assert not numpy.allclose(charges, 0.0)
