from typing import TYPE_CHECKING

import numpy
import pytest

from openff.recharge.esp import DFTGridSettings, ESPSettings
from openff.recharge.esp.qcarchive import (
    MissingQCMoleculesError,
    MissingQCResultsError,
    MissingQCWaveFunctionError,
    from_qcarchive,
)
from openff.recharge.grids import GridSettings

if TYPE_CHECKING:

    from qcfractal import FractalSnowflake
    from qcfractal.interface.collections import Dataset


@pytest.fixture(scope="module")
def qc_server() -> "FractalSnowflake":

    pytest.importorskip("qcfractal")

    from qcfractal import FractalSnowflake

    with FractalSnowflake() as server:
        yield server


@pytest.fixture(scope="module")
def qc_data_set(qc_server: "FractalSnowflake") -> "Dataset":

    import qcfractal.interface as qcportal

    client = qc_server.client()

    # Mock a data set.
    data_set = qcportal.collections.Dataset("test-set", client=client)
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
                extras=dict(
                    canonical_isomeric_explicit_hydrogen_mapped_smiles="[H:2][O:1][H:3]"
                ),
            )
        ),
    )
    data_set.save()

    # Compute the set.
    data_set.compute(
        program="psi4",
        method="scf",
        basis="sto-3g",
        protocols={"wavefunction": "orbitals_and_eigenvalues"},
    )

    qc_server.await_results()
    return data_set


def test_from_qcarchive(qc_server: "FractalSnowflake", qc_data_set: "Dataset"):

    esp_records = from_qcarchive(
        qc_data_set.name,
        ["O"],
        ESPSettings(
            method="scf", basis="sto-3g", grid_settings=GridSettings(spacing=2.0)
        ),
        qc_server.get_address(),
    )

    assert len(esp_records) == 1
    esp_record = esp_records[0]

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
    # expected_field = numpy.array(
    #     [
    #         [0.0020060914, 0.0030422968, -0.0033180173],
    #         [-0.0046893486, -0.0054572100, -0.0042860469],
    #         [0.0020060914, -0.0033180173, 0.0030422968],
    #         [-0.0046893486, -0.0042860469, -0.0054572100],
    #         [-0.0065091907, 0.0025283942, 0.0025283942],
    #         [0.0000000000, 0.0068926703, 0.0013326787],
    #         [0.0000000000, 0.0013326787, 0.0068926703],
    #         [0.0000000000, -0.0039543542, 0.0029573799],
    #         [0.0000000000, 0.0029573799, -0.0039543542],
    #         [-0.0020060914, 0.0030422968, -0.0033180173],
    #         [0.0046893486, -0.0054572100, -0.0042860469],
    #         [-0.0020060914, -0.0033180173, 0.0030422968],
    #         [0.0046893486, -0.0042860469, -0.0054572100],
    #         [0.0065091907, 0.0025283942, 0.0025283942],
    #     ]
    # )

    assert numpy.allclose(esp_record.esp, expected_esp, rtol=1.0e-9)


def test_missing_smiles(qc_server: "FractalSnowflake", qc_data_set: "Dataset"):

    with pytest.raises(MissingQCMoleculesError):

        from_qcarchive(
            qc_data_set.name,
            ["CO"],
            ESPSettings(
                method="scf", basis="sto-3g", grid_settings=GridSettings(spacing=2.0)
            ),
            qc_server.get_address(),
        )


def test_missing_result(qc_server: "FractalSnowflake", qc_data_set: "Dataset"):

    with pytest.raises(MissingQCResultsError):

        from_qcarchive(
            qc_data_set.name,
            ["O"],
            ESPSettings(
                method="scf", basis="6-31G", grid_settings=GridSettings(spacing=2.0)
            ),
            qc_server.get_address(),
        )


def test_missing_wavefunction(
    qc_server: "FractalSnowflake", qc_data_set: "Dataset", monkeypatch
):

    import qcportal
    from qcportal.models import ResultRecord

    old_query = qcportal.FractalClient.query_results

    def remove_wavefunction(self, *args, **kwargs):

        return [
            ResultRecord(**result.dict(exclude={"wavefunction"}), wavefunction=None)
            for result in old_query(self, *args, **kwargs)
        ]

    monkeypatch.setattr(qcportal.FractalClient, "query_results", remove_wavefunction)

    with pytest.raises(MissingQCWaveFunctionError):

        from_qcarchive(
            qc_data_set.name,
            ["O"],
            ESPSettings(
                method="scf", basis="sto-3g", grid_settings=GridSettings(spacing=2.0)
            ),
            qc_server.get_address(),
        )


def test_non_default_dft_grid():

    with pytest.raises(NotImplementedError):

        from_qcarchive(
            "",
            None,
            ESPSettings(
                method="scf",
                basis="sto-3g",
                grid_settings=GridSettings(),
                psi4_dft_grid_settings=DFTGridSettings.Fine,
            ),
            "",
        )
