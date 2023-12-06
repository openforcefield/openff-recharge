import pytest

# Ensure QCPortal is imported before any OpenEye modules, see
# https://github.com/conda-forge/qcfractal-feedstock/issues/43
import qcportal  # noqa


@pytest.fixture
def public_client():
    """Setup a new connection to the public qcarchive client."""

    return qcportal.PortalClient("https://api.qcarchive.molssi.org:443/")


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
