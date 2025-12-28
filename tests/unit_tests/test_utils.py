"""Unit tests for utility helpers such as input file list reader."""

import pytest

from eventdisplay_ml.utils import parse_image_selection, read_input_file_list


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
