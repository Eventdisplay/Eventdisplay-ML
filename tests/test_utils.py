"""Unit tests for utils.py."""

import json

import pytest

from eventdisplay_ml.utils import (
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


def test_output_file_name_returns_path_object(tmp_path):
    result = output_file_name(tmp_path / "model")
    assert isinstance(result, str)
