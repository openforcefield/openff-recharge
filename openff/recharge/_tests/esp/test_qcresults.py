from json import JSONDecodeError
import qcportal

import numpy
import pytest

from openff.recharge.esp.qcresults import (
    InvalidPCMKeywordError,
    MissingQCWaveFunctionError,
    _parse_pcm_input,
    from_qcportal_results,
)
from openff.recharge.grids import LatticeGridSettings


@pytest.mark.parametrize("with_field", [False, True])
def test_from_qcportal_results(with_field, public_client):
    pytest.importorskip("psi4")

    qc_result: qcportal.singlepoint.SinglepointRecord = [
        *public_client.query_records(record_id="32651863")
    ][0]
    qc_keyword_set = public_client.query_keywords(id=ObjectId("2"))[0]  # noqa

    esp_record = from_qcportal_results(
        qc_result=qc_result,
        qc_molecule=qc_result.molecule,
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


def test_missing_wavefunction(public_client):
    from qcportal.singlepoint import SinglepointRecord

    qc_result = [*public_client.query_records(record_id="32651863")][0]
    qc_molecule = qc_result.molecule
    qc_keyword_set = public_client.query_keywords(id=ObjectId("2"))[0]  # noqa

    # Delete the wavefunction
    qc_result = SinglepointRecord(
        **qc_result.dict(exclude={"wavefunction_"}), wavefunction_=None
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
