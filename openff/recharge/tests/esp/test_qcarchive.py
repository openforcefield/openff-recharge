from json import JSONDecodeError
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:

    from qcfractal import FractalSnowflake
    from qcfractal.interface.collections import Dataset


@pytest.mark.parametrize("pcm_settings", [None, PCMSettings()])
def test_retrieve_results(
    pcm_settings,
    qc_server: "FractalSnowflake",
    qc_data_set: "Dataset",
    pcm_input_string: str,
):

    qc_results, qc_keywords = retrieve_qcfractal_results(
        qc_data_set.name, ["O"], "scf", "sto-3g", pcm_settings, qc_server.get_address()
    )

    assert len(qc_results) == 1
    assert len(qc_keywords) == 1

    assert ("1" if pcm_settings is None else "2") in qc_keywords

    if pcm_settings:
        assert qc_keywords["2"].values["pcm__input"] == pcm_input_string


@pytest.mark.parametrize(
    "raise_error, expected_raises",
    [(True, pytest.raises(MissingQCMoleculesError)), (False, does_not_raise())],
)
def test_missing_smiles(
    raise_error, expected_raises, qc_server: "FractalSnowflake", qc_data_set: "Dataset"
):

    with expected_raises:

        retrieve_qcfractal_results(
            qc_data_set.name,
            ["CO"],
            "scf",
            "sto-3g",
            None,
            qc_server.get_address(),
            error_on_missing=raise_error,
        )


@pytest.mark.parametrize(
    "raise_error, expected_raises",
    [(True, pytest.raises(MissingQCResultsError)), (False, does_not_raise())],
)
def test_missing_result(
    raise_error, expected_raises, qc_server: "FractalSnowflake", qc_data_set: "Dataset"
):

    with expected_raises:

        retrieve_qcfractal_results(
            qc_data_set.name,
            ["O"],
            "scf",
            "6-31g",
            None,
            qc_server.get_address(),
            error_on_missing=raise_error,
        )


def test_from_qcfractal_result(qc_server: "FractalSnowflake", qc_data_set: "Dataset"):

    qc_result = qc_server.client().query_results(id="1")[0]
    qc_molecule = qc_result.get_molecule()
    qc_keyword_set = qc_server.client().query_keywords(id="1")[0]

    esp_record = from_qcfractal_result(
        qc_result=qc_result,
        qc_molecule=qc_molecule,
        qc_keyword_set=qc_keyword_set,
        grid_settings=GridSettings(spacing=2.0),
    )

    # Generated using psi4=1.4a2.dev1091+d1fb616 on OSX
    expected_esp = numpy.array(
        [
            [-0.0111092658],
            [0.0053052239],
            [-0.0111092658],
            [0.0053052239],
            [0.0186486188],
            [-0.0193442669],
            [-0.0193442669],
            [0.0104731058],
            [0.0104731058],
            [-0.0111092658],
            [0.0053052239],
            [-0.0111092658],
            [0.0053052239],
            [0.0186486188],
        ]
    )
    expected_field = numpy.array(
        [
            [0.0020060914, 0.0030422968, -0.0033180173],
            [-0.0046893486, -0.0054572100, -0.0042860469],
            [0.0020060914, -0.0033180173, 0.0030422968],
            [-0.0046893486, -0.0042860469, -0.0054572100],
            [-0.0065091907, 0.0025283942, 0.0025283942],
            [0.0000000000, 0.0068926703, 0.0013326787],
            [0.0000000000, 0.0013326787, 0.0068926703],
            [0.0000000000, -0.0039543542, 0.0029573799],
            [0.0000000000, 0.0029573799, -0.0039543542],
            [-0.0020060914, 0.0030422968, -0.0033180173],
            [0.0046893486, -0.0054572100, -0.0042860469],
            [-0.0020060914, -0.0033180173, 0.0030422968],
            [0.0046893486, -0.0042860469, -0.0054572100],
            [0.0065091907, 0.0025283942, 0.0025283942],
        ]
    )

    assert numpy.allclose(esp_record.esp, expected_esp, rtol=1.0e-9)
    assert numpy.allclose(esp_record.electric_field, expected_field, rtol=1.0e-9)


def test_missing_wavefunction(qc_server: "FractalSnowflake", qc_data_set: "Dataset"):

    from qcportal.models import ResultRecord

    qc_result = qc_server.client().query_results(id="1")[0]
    qc_molecule = qc_result.get_molecule()
    qc_keyword_set = qc_server.client().query_keywords(id="1")[0]

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
