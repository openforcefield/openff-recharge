from json import JSONDecodeError

import numpy
import pytest

from openff.recharge.esp.qcresults import (
    InvalidPCMKeywordError,
    MissingQCWaveFunctionError,
    _parse_pcm_input,
    from_qcportal_results,
)
from openff.recharge.grids import LatticeGridSettings

pytest.importorskip("qcportal")


@pytest.mark.parametrize("with_field", [False, True])
def test_from_qcportal_results(with_field):
    pytest.importorskip("psi4")

    from qcportal import FractalClient
    from qcportal.models import ObjectId

    qc_result = FractalClient().query_results(id=ObjectId("32651863"))[0]
    qc_keyword_set = FractalClient().query_keywords(id=ObjectId("2"))[0]

    esp_record = from_qcportal_results(
        qc_result=qc_result,
        qc_molecule=qc_result.get_molecule(),
        qc_keyword_set=qc_keyword_set,
        grid_settings=LatticeGridSettings(spacing=2.0),
        compute_field=with_field,
    )

    assert not numpy.allclose(esp_record.esp, 0.0, rtol=1.0e-9)
    assert (
        esp_record.electric_field is None
        if not with_field
        else not numpy.allclose(esp_record.electric_field, 0.0, rtol=1.0e-9)
    )


def test_missing_wavefunction():
    from qcportal import FractalClient
    from qcportal.models import ObjectId, ResultRecord

    qc_result = FractalClient().query_results(id=ObjectId("1"))[0]
    qc_molecule = qc_result.get_molecule()
    qc_keyword_set = FractalClient().query_keywords(id=ObjectId("2"))[0]

    # Delete the wavefunction
    qc_result = ResultRecord(
        **qc_result.dict(exclude={"wavefunction"}), wavefunction=None
    )

    with pytest.raises(MissingQCWaveFunctionError):
        from_qcportal_results(
            qc_result, qc_molecule, qc_keyword_set, LatticeGridSettings()
        )


def test_parse_pcm_input():
    value = """
        units = angstrom
        codata = 2010
        medium {
     solvertype = iefpcm
     nonequilibrium = false
     solvent = h2o
     matrixsymm = true
     correction = 0.0
     diagonalscaling = 1.07
     proberadius = 0.52917721067}
        cavity {
     type = gepol
     area = 0.4
     scaling = false
     radiiset = uff
     minradius = 52.917721067
     mode = implicit}"""

    pcm_settings = _parse_pcm_input(value)
    assert pcm_settings is not None

    assert pcm_settings.solver == "IEFPCM"
    assert pcm_settings.solvent == "Water"

    assert pcm_settings.cavity_area == 0.4

    assert pcm_settings.radii_model == "UFF"
    assert pcm_settings.radii_scaling is False


def test_parse_invalid_pcm_input():
    value = """
        units = angstrom
        codata = 2011
        medium {
     solvertype = iefpcm
     nonequilibrium = false
     solvent = h2o
     matrixsymm = true
     correction = 0.0
     diagonalscaling = 1.07
     proberadius = 0.52917721067}
        cavity {
     type = gepol
     area = 0.4
     scaling = false
     radiiset = uff
     minradius = 52.917721067
     mode = implicit}"""

    with pytest.raises(InvalidPCMKeywordError):
        _parse_pcm_input(value)

    value = """
        units = angstrom
        codata = 2011
        medium {"""

    with pytest.raises(JSONDecodeError):
        _parse_pcm_input(value)
