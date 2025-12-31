"""Unit tests for utility helpers such as input file list reader."""

import json

import pytest

from eventdisplay_ml.utils import (
    load_energy_range,
    load_model_parameters,
    parse_image_selection,
    read_input_file_list,
)


def test_read_input_file_list_success(tmp_path):
    """Test successful reading of input file list."""
    test_file = tmp_path / "input_files.txt"
    test_files = ["file1.txt", "file2.txt", "file3.txt"]
    test_file.write_text("\n".join(test_files))

    result = read_input_file_list(str(test_file))
    assert result == test_files


def test_read_input_file_list_with_empty_lines(tmp_path):
    """Test reading file list with empty lines."""
    test_file = tmp_path / "input_files.txt"
    content = "file1.txt\n\nfile2.txt\n  \nfile3.txt\n"
    test_file.write_text(content)

    result = read_input_file_list(str(test_file))
    assert result == ["file1.txt", "file2.txt", "file3.txt"]


def test_read_input_file_list_with_whitespace(tmp_path):
    """Test reading file list with leading/trailing whitespace."""
    test_file = tmp_path / "input_files.txt"
    content = "  file1.txt  \nfile2.txt\t\n  file3.txt"
    test_file.write_text(content)

    result = read_input_file_list(str(test_file))
    assert result == ["file1.txt", "file2.txt", "file3.txt"]


def test_read_input_file_list_empty_file(tmp_path):
    """Test reading empty file."""
    test_file = tmp_path / "input_files.txt"
    test_file.write_text("")

    with pytest.raises(ValueError, match="Error: No input files found in the list"):
        read_input_file_list(str(test_file))


def test_read_input_file_list_file_not_found():
    """Test FileNotFoundError is raised when file does not exist."""
    with pytest.raises(FileNotFoundError, match="Error: Input file list not found"):
        read_input_file_list("/nonexistent/path/file.txt")


def test_parse_image_selection_comma_separated():
    """Test parsing comma-separated indices."""
    result = parse_image_selection("1, 2, 3")
    assert result == [1, 2, 3]


def test_parse_image_selection_bit_coded():
    """Test parsing bit-coded value."""
    result = parse_image_selection("14")  # 0b1110 -> indices 1, 2, 3
    assert result == [1, 2, 3]


def test_parse_image_selection_empty_string():
    """Test parsing empty string returns None."""
    result = parse_image_selection("")
    assert result is None


def test_parse_image_selection_invalid_comma_separated():
    """Test ValueError is raised for invalid comma-separated input."""
    with pytest.raises(ValueError, match="Invalid image_selection format"):
        parse_image_selection("1, two, 3")


def test_parse_image_selection_invalid_bit_coded():
    """Test ValueError is raised for invalid bit-coded input."""
    with pytest.raises(ValueError, match="Invalid image_selection format"):
        parse_image_selection("invalid")


def test_load_model_parameters_success(tmp_path):
    """Test loading model parameters from a valid JSON file."""
    params = {
        "energy_bins_log10_tev": [{"E_min": 1.0, "E_max": 2.0}, {"E_min": 2.0, "E_max": 3.0}],
        "other_param": 42,
    }
    param_file = tmp_path / "params.json"
    param_file.write_text(json.dumps(params))

    result = load_model_parameters(str(param_file))
    assert result["energy_bins_log10_tev"] == params["energy_bins_log10_tev"]
    assert result["other_param"] == 42


def test_load_model_parameters_with_energy_bin_number(tmp_path):
    """Test loading model parameters with a specific energy bin number."""
    params = {
        "energy_bins_log10_tev": [{"E_min": 1.0, "E_max": 2.0}, {"E_min": 2.0, "E_max": 3.0}],
        "other_param": 42,
    }
    param_file = tmp_path / "params.json"
    param_file.write_text(json.dumps(params))

    result = load_model_parameters(str(param_file), energy_bin_number=1)
    assert result["energy_bins_log10_tev"] == {"E_min": 2.0, "E_max": 3.0}
    assert result["other_param"] == 42


def test_load_model_parameters_file_not_found(tmp_path):
    """Test FileNotFoundError is raised when model parameters file does not exist."""
    non_existent_file = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError, match="Model parameters file not found"):
        load_model_parameters(str(non_existent_file))


def test_load_model_parameters_invalid_energy_bin_number(tmp_path):
    """Test ValueError is raised for invalid energy bin number."""
    params = {"energy_bins_log10_tev": [{"E_min": 1.0, "E_max": 2.0}]}
    param_file = tmp_path / "params.json"
    param_file.write_text(json.dumps(params))

    with pytest.raises(ValueError, match="Invalid energy bin number 5"):
        load_model_parameters(str(param_file), energy_bin_number=5)


def test_load_model_parameters_missing_energy_bins_key(tmp_path):
    """Test ValueError is raised if energy_bins_log10_tev key is missing when energy_bin_number is given."""
    params = {"other_param": 42}
    param_file = tmp_path / "params.json"
    param_file.write_text(json.dumps(params))

    with pytest.raises(ValueError, match="Invalid energy bin number 0"):
        load_model_parameters(str(param_file), energy_bin_number=0)


def test_load_energy_range_success(tmp_path):
    """Test loading energy range for a valid energy bin."""
    params = {
        "energy_bins_log10_tev": [
            {"E_min": 0.0, "E_max": 1.0},  # 10^0=1, 10^1=10
            {"E_min": 1.0, "E_max": 2.0},  # 10^1=10, 10^2=100
        ]
    }
    param_file = tmp_path / "params.json"
    param_file.write_text(json.dumps(params))

    result = load_energy_range(str(param_file), energy_bin_number=0)
    assert result == (1.0, 10.0)

    result = load_energy_range(str(param_file), energy_bin_number=1)
    assert result == (10.0, 100.0)


def test_load_energy_range_invalid_bin_number(tmp_path):
    """Test ValueError is raised for invalid energy bin number."""
    params = {"energy_bins_log10_tev": [{"E_min": 0.0, "E_max": 1.0}]}
    param_file = tmp_path / "params.json"
    param_file.write_text(json.dumps(params))

    with pytest.raises(ValueError, match="Invalid energy bin number 5"):
        load_energy_range(str(param_file), energy_bin_number=5)


def test_load_energy_range_missing_energy_bins_key(tmp_path):
    """Test ValueError is raised if energy_bins_log10_tev key is missing."""
    params = {"other_param": 42}
    param_file = tmp_path / "params.json"
    param_file.write_text(json.dumps(params))

    with pytest.raises(ValueError, match="Invalid energy bin number 0"):
        load_energy_range(str(param_file), energy_bin_number=0)
