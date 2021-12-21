from json import JSONDecodeError

import numpy
import pytest

from openff.recharge.esp import PCMSettings
from openff.recharge.esp.qcarchive import (
    InvalidPCMKeywordError,
    MissingQCMoleculesError,
    MissingQCResultsError,
    MissingQCWaveFunctionError,
    _compare_pcm_settings,
    _parse_pcm_input,
    from_qcfractal_result,
    retrieve_qcfractal_results,
)
from openff.recharge.grids import GridSettings
from openff.recharge.tests import does_not_raise

pytest.importorskip("qcportal")


@pytest.mark.parametrize("pcm_settings", [None, PCMSettings()])
def test_retrieve_results(
    pcm_settings,
    pcm_input_string: str,
):

    qc_results, qc_keywords = retrieve_qcfractal_results(
        "OpenFF BCC Refit Study COH v1.0",
        ["CO"],
        "pw6b95",
        "aug-cc-pV(D+d)Z",
        pcm_settings,
    )

    assert len(qc_results) == 1
    assert len(qc_keywords) == 1

    assert ("2" if pcm_settings is None else "12") in qc_keywords

    if pcm_settings:
        assert qc_keywords["12"].values["pcm__input"].replace(" ", "").replace(
            "\n", ""
        ) == pcm_input_string.replace(" ", "").replace("\n", "")


@pytest.mark.parametrize(
    "raise_error, expected_raises",
    [(True, pytest.raises(MissingQCMoleculesError)), (False, does_not_raise())],
)
def test_missing_smiles(raise_error, expected_raises):

    with expected_raises:

        retrieve_qcfractal_results(
            "OpenFF BCC Refit Study COH v1.0",
            ["O"],
            "pw6b95",
            "aug-cc-pV(D+d)Z",
            None,
            error_on_missing=raise_error,
        )


@pytest.mark.parametrize(
    "raise_error, expected_raises",
    [(True, pytest.raises(MissingQCResultsError)), (False, does_not_raise())],
)
def test_missing_result(raise_error, expected_raises):

    with expected_raises:

        retrieve_qcfractal_results(
            "OpenFF BCC Refit Study COH v1.0",
            ["CO"],
            "scf",
            "6-31g",
            None,
            error_on_missing=raise_error,
        )


def test_from_qcfractal_result():

    qc_results, qc_keywords = retrieve_qcfractal_results(
        "OpenFF BCC Refit Study COH v1.0", ["CO"], "pw6b95", "aug-cc-pV(D+d)Z", None
    )

    esp_record = from_qcfractal_result(
        qc_result=qc_results[0][1],
        qc_molecule=qc_results[0][1].get_molecule(),
        qc_keyword_set=qc_keywords["2"],
        grid_settings=GridSettings(spacing=2.0),
    )

    assert not numpy.allclose(esp_record.esp, 0.0, rtol=1.0e-9)
    assert not numpy.allclose(esp_record.electric_field, 0.0, rtol=1.0e-9)


def test_missing_wavefunction():

    from qcportal import FractalClient
    from qcportal.models import ResultRecord

    qc_result = FractalClient().query_results(id="1")[0]
    qc_molecule = qc_result.get_molecule()
    qc_keyword_set = FractalClient().query_keywords(id="2")[0]

    # Delete the wavefunction
    qc_result = ResultRecord(
        **qc_result.dict(exclude={"wavefunction"}), wavefunction=None
    )

    with pytest.raises(MissingQCWaveFunctionError):
        from_qcfractal_result(qc_result, qc_molecule, qc_keyword_set, GridSettings())


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


@pytest.mark.parametrize(
    "settings, expectation",
    [
        (PCMSettings(), True),
        (PCMSettings(cavity_area=0.6), False),
        (PCMSettings(solver="IEFPCM"), False),
    ],
)
def test_compare_pcm_settings(settings, expectation):
    assert _compare_pcm_settings(settings, PCMSettings()) == expectation
