"""Unit tests for run information reading (replacement of printRunParameter)."""

import numpy as np
import pytest

from eventdisplay_ml import run_info
from eventdisplay_ml.scripts import print_run_info

EPOCH_FILE_TEXT = """\
EPOCH <epoch name> <run min> <run max>
* EPOCH V5 2009-09-14 2012-07-31
* EPOCH V6_2023_2024w 2023-11-01 2024-04-22
* EPOCH V6_2024_2025s 2024-04-23 2024-11-15 60423 60629
* ATMOSPHERE 61 2023-11-01 2024-04-22
* ATMOSPHERE 62 2024-04-23  2024-11-15
"""


@pytest.fixture(name="epoch_file")
def epoch_file_fixture(tmp_path):
    """Return a small epoch file."""
    path = tmp_path / "VERITAS.Epochs.runparameter"
    path.write_text(EPOCH_FILE_TEXT, encoding="utf-8")
    return path


def test_sql_date_to_mjd_matches_known_days():
    """MJD of a day matches the values stored in the run parameters."""
    assert run_info.sql_date_to_mjd("2023-11-07") == 60255.0
    assert run_info.sql_date_to_mjd("2000-01-01 12:30:00") == 51544.0


def test_read_epoch_file_finds_epoch_and_atmosphere(epoch_file):
    """Epoch and atmosphere are selected by the run MJD."""
    assert run_info.read_epoch_file(epoch_file, "EPOCH", 60255.0, 60255.0) == "V6_2023_2024w"
    assert run_info.read_epoch_file(epoch_file, "ATMOSPHERE", 60255.0, 60255.0) == "61"
    assert run_info.read_epoch_file(epoch_file, "EPOCH", 60500.0, 60500.0) == "V6_2024_2025s"


def test_read_epoch_file_fails_without_covering_interval(epoch_file):
    """A run outside all intervals must fail, not fall back to another epoch."""
    with pytest.raises(ValueError, match="No EPOCH interval"):
        run_info.read_epoch_file(epoch_file, "EPOCH", 50000.0, 50000.0)


def test_read_epoch_file_requires_run_mjds(epoch_file):
    """Missing run MJDs must fail."""
    with pytest.raises(ValueError, match="MJDs not available"):
        run_info.read_epoch_file(epoch_file, "EPOCH", 0.0, 0.0)


def test_read_epoch_file_rejects_invalid_keyword(epoch_file):
    """Only EPOCH and ATMOSPHERE are valid keywords."""
    with pytest.raises(ValueError, match="Invalid keyword"):
        run_info.read_epoch_file(epoch_file, "GAIN", 60255.0, 60255.0)


def test_default_epoch_file_requires_environment(monkeypatch, tmp_path):
    """The default epoch file is located through VERITAS_EVNDISP_AUX_DIR."""
    monkeypatch.delenv("VERITAS_EVNDISP_AUX_DIR", raising=False)
    with pytest.raises(ValueError, match="VERITAS_EVNDISP_AUX_DIR"):
        run_info.default_epoch_file()
    monkeypatch.setenv("VERITAS_EVNDISP_AUX_DIR", str(tmp_path))
    assert (
        run_info.default_epoch_file() == tmp_path / "ParameterFiles" / "VERITAS.Epochs.runparameter"
    )


class _FakeBranch:
    def __init__(self, rows):
        self._rows = rows

    def array(self, entry_start, entry_stop, library):
        return list(self._rows[entry_start:entry_stop])


class _FakeTree:
    def __init__(self, rows, n_entries=None):
        self._rows = rows
        self.num_entries = len(rows) if n_entries is None else n_entries

    def __getitem__(self, name):
        assert name == "TelElevation"
        return _FakeBranch(self._rows)


def test_read_mean_elevation_ignores_unpointed_telescopes():
    """Telescopes at or below 5 deg elevation are excluded from the mean."""
    rows = [np.array([70.0, 60.0, 2.0, 0.0, 0.0]), np.array([68.0, 62.0, 3.0, 0.0, 0.0])]
    root_file = {"telconfig": _FakeTree([None] * 4), "data": _FakeTree(rows)}
    assert run_info.read_mean_elevation(root_file) == pytest.approx(65.0)


def test_read_mean_elevation_without_valid_telescope():
    """No pointing telescope returns the printRunParameter sentinel value."""
    root_file = {"telconfig": _FakeTree([None] * 4), "data": _FakeTree([np.zeros(4)])}
    assert run_info.read_mean_elevation(root_file) == -999.0


def test_format_run_info_uses_runinfo_column_layout():
    """Columns follow printRunParameter -runinfo; floats use stream-style formatting."""
    info = {
        "epoch": "V6_2023_2024w",
        "major_epoch": "V6",
        "atmosphere": "61",
        "run_type": "observing",
        "telescopes": "1234",
        "duration": 900.0,
        "target": "Crab",
        "zenith": 23.39644,
        "wobble": 0.5,
        "wobble_int": 50,
    }
    assert run_info.format_run_info(info).split("\t") == [
        "V6_2023_2024w",
        "V6",
        "61",
        "observing",
        "1234",
        "900",
        "Crab",
        "23.3964",
        "0.5",
        "50",
    ]


def test_get_run_info_updates_epoch_from_file(monkeypatch, epoch_file):
    """--updated replaces stored epoch and atmosphere by the epoch file values."""
    parameters = {
        "fInstrumentEpoch": "V6_old",
        "fAtmosphereID": 99,
        "fDBRunType": "",
        "fTelToAnalyze": [0, 1, 3],
        "fRunDuration": 900.0,
        "fTargetName": "Crab",
        "fWobbleNorth": 0.3,
        "fWobbleEast": 0.4,
        "fDBDataStartTimeMJD": 60255.0,
        "fDBDataStoppTimeMJD": 60255.0,
    }

    class _Open:
        def __enter__(self):
            return "root_file"

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(run_info.uproot, "open", lambda _path: _Open())
    monkeypatch.setattr(run_info, "read_run_parameters", lambda _file: parameters)
    monkeypatch.setattr(run_info, "read_mean_elevation", lambda _file: 60.0)

    stored = run_info.get_run_info("run.root")
    assert (stored["epoch"], stored["atmosphere"]) == ("V6_old", "99")
    assert stored["run_type"] == "no_db_runtype"
    assert stored["telescopes"] == "124"
    assert stored["zenith"] == pytest.approx(30.0)
    assert stored["wobble"] == pytest.approx(0.5)
    assert stored["wobble_int"] == 50

    updated = run_info.get_run_info("run.root", update_epoch=True, epoch_file=epoch_file)
    assert (updated["epoch"], updated["major_epoch"], updated["atmosphere"]) == (
        "V6_2023_2024w",
        "V6",
        "61",
    )


def test_entrypoint_exits_nonzero_on_unreadable_run_info(monkeypatch):
    """A failed lookup must stop the calling job instead of printing a partial line."""

    def _fail(*_args, **_kwargs):
        raise KeyError("runparameterV2 not found")

    monkeypatch.setattr(print_run_info, "get_run_info", _fail)
    monkeypatch.setattr("sys.argv", ["eventdisplay-ml-print-run-info", "run.root"])
    with pytest.raises(SystemExit) as exit_info:
        print_run_info.main()
    assert exit_info.value.code == 1


def test_entrypoint_prints_one_line(monkeypatch, capsys):
    """Successful run prints exactly the formatted line."""
    monkeypatch.setattr(print_run_info, "get_run_info", lambda *_a, **_k: {"x": 1})
    monkeypatch.setattr(print_run_info, "format_run_info", lambda info: f"line {info['x']}")
    monkeypatch.setattr("sys.argv", ["eventdisplay-ml-print-run-info", "run.root", "--updated"])
    print_run_info.main()
    assert capsys.readouterr().out == "line 1\n"
