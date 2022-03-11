from typing import Tuple

import numpy
import pytest
from openff.toolkit.topology import Molecule
from openff.units import unit
from typing_extensions import Literal

from openff.recharge.charges import ChargeGenerator, ChargeSettings


@pytest.fixture(scope="module")
def methane() -> Tuple[Molecule, numpy.ndarray]:

    from simtk import unit as simtk_unit

    molecule: Molecule = Molecule.from_mapped_smiles(
        "[C:1]([H:2])([H:3])([H:4])([H:5])"
    )
    molecule.generate_conformers(n_conformers=1)

    conformer = molecule.conformers[0].value_in_unit(simtk_unit.angstrom)

    return molecule, conformer


@pytest.mark.parametrize("theory", ["am1", "am1bcc"])
def test_generate_omega_charges(methane, theory):

    molecule, conformer = methane

    default_charges = ChargeGenerator._generate_omega_charges(
        molecule, conformer, ChargeSettings()
    )
    assert default_charges.shape == (5, 1)
    assert not numpy.allclose(default_charges, 0.0)

    asym_charges = ChargeGenerator._generate_omega_charges(
        molecule, conformer, ChargeSettings(symmetrize=False)
    )
    assert len({*default_charges[1:].flatten()}) == 1
    assert len({*asym_charges[1:].flatten()}) == 4

    unopt_charges = ChargeGenerator._generate_omega_charges(
        molecule, conformer, ChargeSettings(optimize=False)
    )
    assert not numpy.allclose(default_charges, unopt_charges)


@pytest.mark.parametrize("theory", ["am1", "am1bcc"])
def test_generate_charges(theory: Literal["am1", "am1bcc"], methane, monkeypatch):
    """Ensure that charges can be generated for a simple molecule using
    the `ChargeGenerator` class."""

    molecule, conformer = methane

    def mock_generate_omega_charges(*args, **kwargs):
        raise NotImplementedError()

    monkeypatch.setattr(
        ChargeGenerator, "_generate_omega_charges", mock_generate_omega_charges
    )

    charges = ChargeGenerator.generate(
        molecule, [conformer * unit.angstrom], ChargeSettings(theory=theory)
    )
    assert charges.shape == (5, 1)
    assert not numpy.allclose(charges, 0.0)
