"""Unit tests for utils.py."""

import json
import logging
from types import SimpleNamespace

import joblib
import pandas as pd
import pytest

from eventdisplay_ml import utils
from eventdisplay_ml.utils import (
    discover_joblib_files,
    joblib_basename,
    load_energy_range,
    load_model_parameters,
    output_file_name,
    parse_image_selection,
    read_input_file_list,
    resolve_joblib_path,
)

# ---------------------------------------------------------------------------
# resolve_joblib_path
# ---------------------------------------------------------------------------


def test_resolve_joblib_gz_found(tmp_path):
    p = tmp_path / "model.joblib.gz"
    p.touch()
    assert resolve_joblib_path(p) == p


def test_resolve_joblib_found_when_gz_missing(tmp_path):
    p = tmp_path / "model.joblib"
    p.touch()
    result = resolve_joblib_path(tmp_path / "model.joblib")
    assert result == p


def test_resolve_prefix_finds_gz_first(tmp_path):
    gz = tmp_path / "model.joblib.gz"
    plain = tmp_path / "model.joblib"
    gz.touch()
    plain.touch()
    result = resolve_joblib_path(tmp_path / "model")
    assert result == gz


def test_resolve_prefix_falls_back_to_joblib(tmp_path):
    plain = tmp_path / "model.joblib"
    plain.touch()
    result = resolve_joblib_path(tmp_path / "model")
    assert result == plain


def test_resolve_missing_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="Could not resolve"):
        resolve_joblib_path(tmp_path / "nonexistent")


def test_resolve_gz_suffix_tries_plain_fallback(tmp_path):
    plain = tmp_path / "model.joblib"
    plain.touch()
    result = resolve_joblib_path(tmp_path / "model.joblib.gz")
    assert result == plain


# ---------------------------------------------------------------------------
# joblib file helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("model.joblib.gz", "model"),
        ("model.joblib", "model"),
        ("model", "model"),
    ],
)
def test_joblib_basename(path, expected):
    assert joblib_basename(path) == expected


def test_discover_joblib_files_returns_sorted_files_and_prefers_gz(tmp_path):
    (tmp_path / "b_model.joblib").touch()
    (tmp_path / "a_model.joblib").touch()
    (tmp_path / "a_model.joblib.gz").touch()
    (tmp_path / "notes.txt").touch()

    result = discover_joblib_files(tmp_path)

    assert result == [
        tmp_path / "a_model.joblib.gz",
        tmp_path / "b_model.joblib",
    ]


def test_discover_joblib_files_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model directory not found"):
        discover_joblib_files(tmp_path / "missing")


def test_discover_joblib_files_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No joblib files"):
        discover_joblib_files(tmp_path)


def test_discover_joblib_files_ignores_directories_with_model_suffixes(tmp_path):
    """A directory named like a model must never be returned as a model file."""
    (tmp_path / "not_a_model.joblib").mkdir()
    (tmp_path / "model.joblib.gz").touch()

    assert discover_joblib_files(tmp_path) == [tmp_path / "model.joblib.gz"]


# ---------------------------------------------------------------------------
# read_input_file_list
# ---------------------------------------------------------------------------


def test_read_input_file_list_returns_lines(tmp_path):
    f = tmp_path / "files.txt"
    f.write_text("a.root\nb.root\nc.root\n")
    result = read_input_file_list(f)
    assert result == ["a.root", "b.root", "c.root"]


def test_read_input_file_list_skips_blank_lines(tmp_path):
    f = tmp_path / "files.txt"
    f.write_text("a.root\n\nb.root\n   \n")
    result = read_input_file_list(f)
    assert result == ["a.root", "b.root"]


def test_read_input_file_list_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_input_file_list(tmp_path / "missing.txt")


def test_read_input_file_list_empty_file_raises(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("")
    with pytest.raises(ValueError, match="No input files"):
        read_input_file_list(f)


# ---------------------------------------------------------------------------
# parse_image_selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("input_str", "expected"),
    [
        ("15", [0, 1, 2, 3]),  # 0b1111
        ("14", [1, 2, 3]),  # 0b1110
        ("1", [0]),  # 0b0001
        ("8", [3]),  # 0b1000
        ("0", []),  # all bits off
    ],
)
def test_parse_image_selection_bit_coded(input_str, expected):
    assert parse_image_selection(input_str) == expected


def test_parse_image_selection_comma_separated():
    assert parse_image_selection("1,2,3") == [1, 2, 3]


def test_parse_image_selection_comma_with_spaces():
    assert parse_image_selection("0, 2, 3") == [0, 2, 3]


def test_parse_image_selection_empty_returns_none():
    assert parse_image_selection("") is None
    assert parse_image_selection(None) is None


def test_parse_image_selection_invalid_raises():
    with pytest.raises(ValueError, match="Invalid image_selection"):
        parse_image_selection("abc")


def test_parse_image_selection_malformed_comma_list_falls_through_to_clear_error():
    """Partially numeric comma lists must not be silently accepted."""
    with pytest.raises(ValueError, match="Invalid image_selection"):
        parse_image_selection("1,invalid")


# ---------------------------------------------------------------------------
# load_model_parameters
# ---------------------------------------------------------------------------


@pytest.fixture
def model_params_file(tmp_path):
    params = {
        "energy_bins_log10_tev": [
            {"E_min": -1.0, "E_max": 0.0},
            {"E_min": 0.0, "E_max": 1.0},
        ],
        "zenith_bins_deg": [{"Ze_min": 0, "Ze_max": 30}],
    }
    f = tmp_path / "params.json"
    f.write_text(json.dumps(params))
    return f, params


def test_load_model_parameters_returns_full_dict(model_params_file):
    f, params = model_params_file
    result = load_model_parameters(f)
    assert result["energy_bins_log10_tev"] == params["energy_bins_log10_tev"]
    assert result["zenith_bins_deg"] == params["zenith_bins_deg"]


def test_load_model_parameters_selects_energy_bin(model_params_file):
    f, _ = model_params_file
    result = load_model_parameters(f, energy_bin_number=1)
    assert result["energy_bins_log10_tev"] == {"E_min": 0.0, "E_max": 1.0}


def test_load_model_parameters_invalid_bin_raises(model_params_file):
    f, _ = model_params_file
    with pytest.raises(ValueError, match="Invalid energy bin number"):
        load_model_parameters(f, energy_bin_number=99)


def test_load_model_parameters_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_model_parameters("/nonexistent/path.json")


def test_load_model_parameters_none_path_has_same_clear_error():
    """Argparse omissions should not leak a TypeError from open()."""
    with pytest.raises(FileNotFoundError, match="Model parameters file not found: None"):
        load_model_parameters(None)


def test_load_model_parameters_missing_energy_bins_raises_value_error(tmp_path):
    """Selecting a bin requires the energy-bin metadata to exist."""
    params_file = tmp_path / "missing_bins.json"
    params_file.write_text(json.dumps({"zenith_bins_deg": []}))

    with pytest.raises(ValueError, match="Invalid energy bin number 0"):
        load_model_parameters(params_file, energy_bin_number=0)


# ---------------------------------------------------------------------------
# load_energy_range
# ---------------------------------------------------------------------------


def test_load_energy_range_returns_power_of_ten():
    params = {"energy_bins_log10_tev": {"E_min": -1.0, "E_max": 1.0}}
    e_min, e_max = load_energy_range(params)
    assert e_min == pytest.approx(0.1)
    assert e_max == pytest.approx(10.0)


def test_load_energy_range_missing_key_raises():
    with pytest.raises(ValueError, match="Invalid or missing energy range"):
        load_energy_range({"other_key": {}})


def test_load_energy_range_missing_inner_key_raises():
    with pytest.raises((ValueError, KeyError)):
        load_energy_range({"energy_bins_log10_tev": {"E_min": -1.0}})


# ---------------------------------------------------------------------------
# output_file_name
# ---------------------------------------------------------------------------


def test_output_file_name_basic(tmp_path):
    result = output_file_name(tmp_path / "model")
    assert str(result).endswith(".joblib.gz")


def test_output_file_name_with_ntel(tmp_path):
    result = output_file_name(tmp_path / "model", n_tel=4)
    assert "_ntel4" in str(result)
    assert str(result).endswith(".joblib.gz")


def test_output_file_name_with_energy_bin(tmp_path):
    result = output_file_name(tmp_path / "model", energy_bin_number=2)
    assert "_ebin2" in str(result)


def test_output_file_name_creates_parent_directory(tmp_path):
    nested = tmp_path / "subdir" / "model"
    output_file_name(nested)
    assert (tmp_path / "subdir").is_dir()


def test_output_file_name_returns_string(tmp_path):
    result = output_file_name(tmp_path / "model")
    assert isinstance(result, str)


def test_output_file_name_combines_multiplicity_and_energy_bin(tmp_path):
    """Keep the naming contract unique when both suffixes are supplied."""
    result = output_file_name(tmp_path / "model", n_tel=3, energy_bin_number=4)
    assert result.endswith("model_ntel3_ebin4.joblib.gz")


# ---------------------------------------------------------------------------
# memory profiling and joblib loading
# ---------------------------------------------------------------------------


def test_max_rss_uses_platform_specific_units(monkeypatch):
    """MacOS reports bytes while Linux reports KiB; normalize both to GB."""
    usage = SimpleNamespace(ru_maxrss=1024**3)
    monkeypatch.setattr(utils.resource, "getrusage", lambda *_args: usage)
    monkeypatch.setattr(utils.sys, "platform", "darwin")
    assert utils._max_rss_gb() == pytest.approx(1.0)

    monkeypatch.setattr(utils.sys, "platform", "linux")
    assert utils._max_rss_gb() == pytest.approx(1024.0)


def test_current_rss_reads_proc_statm_when_available(monkeypatch, tmp_path):
    """Current RSS uses resident pages rather than the peak when /proc exists."""
    statm = tmp_path / "statm"
    statm.write_text("100 512 0 0 0 0 0")
    monkeypatch.setattr(utils, "Path", lambda _path: statm)
    monkeypatch.setattr(utils.os, "sysconf", lambda _name: 4096)

    assert utils._current_rss_gb() == pytest.approx(512 * 4096 / 1024**3)


def test_current_rss_falls_back_to_peak_when_proc_is_unavailable(monkeypatch, tmp_path):
    """macOS-like environments without /proc retain a meaningful RSS value."""
    missing_statm = tmp_path / "missing"
    monkeypatch.setattr(utils, "Path", lambda _path: missing_statm)
    monkeypatch.setattr(utils, "_max_rss_gb", lambda: 1.25)

    assert utils._current_rss_gb() == pytest.approx(1.25)


def test_log_memory_checkpoint_is_disabled_without_side_effects(monkeypatch):
    """The profiling helper must be a no-op unless explicitly enabled."""
    monkeypatch.setattr(utils, "_current_rss_gb", lambda: pytest.fail("should not be called"))
    utils.log_memory_checkpoint("disabled", enabled=False)


def test_log_memory_checkpoint_logs_timing_rss_and_dataframe_memory(monkeypatch, caplog):
    """Enabled profiling reports both process and DataFrame memory details."""
    utils._profile_start_time = None
    utils._profile_last_time = None
    monkeypatch.setattr(utils.time, "perf_counter", lambda: 10.0)
    monkeypatch.setattr(utils, "_current_rss_gb", lambda: 1.5)
    monkeypatch.setattr(utils, "_max_rss_gb", lambda: 2.5)

    with caplog.at_level(logging.INFO):
        utils.log_memory_checkpoint("after flattening", pd.DataFrame({"x": [1, 2]}), enabled=True)

    assert "Memory checkpoint [after flattening]" in caplog.text
    assert "rss=1.50 GB" in caplog.text
    assert "max_rss=2.50 GB" in caplog.text
    assert "shape=(2, 1)" in caplog.text


def test_load_joblib_returns_payload_and_suppresses_only_shape_warning(monkeypatch):
    """Model loading tolerates NumPy's known pickle warning without hiding failures."""
    payload = {"model": "trusted-test-payload"}
    monkeypatch.setattr(joblib, "load", lambda _path: payload)

    assert utils.load_joblib("model.joblib.gz") is payload
