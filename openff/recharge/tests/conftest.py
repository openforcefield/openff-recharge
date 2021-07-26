from typing import TYPE_CHECKING

import numpy
import pytest

if TYPE_CHECKING:

    from qcfractal import FractalSnowflake
    from qcfractal.interface.collections import Dataset


@pytest.fixture(scope="module")
def pcm_input_string() -> str:

    return "\n".join(
        [
            "units = angstrom",
            "codata = 2010",
            "medium {",
            "solvertype = cpcm",
            "nonequilibrium = false",
            "solvent = h2o",
            "matrixsymm = true",
            "correction = 0.0",
            "diagonalscaling = 1.07",
            "proberadius = 0.52917721067}",
            "cavity {",
            "type = gepol",
            "area = 0.3",
            "scaling = true",
            "radiiset = bondi",
            "minradius = 52.917721067",
            "mode = implicit}",
        ]
    )


@pytest.fixture(scope="module")
def qc_server() -> "FractalSnowflake":

    pytest.importorskip("qcfractal")

    from qcfractal import FractalSnowflake

    with FractalSnowflake() as server:
        yield server


@pytest.fixture(scope="module")
def qc_data_set(qc_server: "FractalSnowflake", pcm_input_string: str) -> "Dataset":

    import qcfractal.interface as qcportal
    from qcengine.programs.psi4 import Psi4Harness

    client = qc_server.client()

    # Patch QCEngine because it cannot detect the psi4 version on linux.
    old_version_function = Psi4Harness.get_version
    Psi4Harness.get_version = lambda self: "1.4a3.dev1"

    # Mock a data set.
    data_set = qcportal.collections.Dataset("test-set", client=client)
    data_set.add_keywords(
        "vacuum",
        "psi4",
        qcportal.models.KeywordSet(values={}),
    )
    data_set.add_keywords(
        "pcm",
        "psi4",
        qcportal.models.KeywordSet(
            values={"pcm": True, "pcm__input": pcm_input_string}
        ),
    )
    data_set.add_entry(
        "water",
        qcportal.Molecule.from_data(
            dict(
                name="H2O",
                symbols=["O", "H", "H"],
                geometry=numpy.array(
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 2.0, 0.0]]
                ),
                molecular_charge=0.0,
                molecular_multiplicity=1,
                connectivity=[(0, 1, 1.0), (0, 2, 1.0)],
                fix_com=True,
                fix_orientation=True,
                fix_symmetry="c1",
                extras=dict(
                    canonical_isomeric_explicit_hydrogen_mapped_smiles="[H:2][O:1][H:3]"
                ),
            )
        ),
    )

    data_set.save()

    # Compute the set.
    x = data_set.compute(
        program="psi4",
        method="scf",
        basis="sto-3g",
        protocols={"wavefunction": "orbitals_and_eigenvalues"},
        keywords="vacuum",
    )
    y = data_set.compute(
        program="psi4",
        method="scf",
        basis="sto-3g",
        protocols={"wavefunction": "orbitals_and_eigenvalues"},
        keywords="pcm",
    )

    print(x, y)

    qc_server.await_results()

    Psi4Harness.get_version = old_version_function

    return data_set
