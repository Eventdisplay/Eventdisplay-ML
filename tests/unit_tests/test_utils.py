
from eventdisplay_ml.utils import read_input_file_list
import pytest


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
    
    result = read_input_file_list(str(test_file))
    assert result == []


def test_read_input_file_list_file_not_found():
    """Test FileNotFoundError is raised when file does not exist."""
    with pytest.raises(FileNotFoundError, match="Error: Input file list not found"):
        read_input_file_list("/nonexistent/path/file.txt")