import pytest

import numpy as np
import pandas as pd

from eventdisplay_ml.data_processing import _to_dense_array, _to_padded_array, flatten_data_vectorized, load_training_data


def test_to_dense_array_with_regular_lists():
    """Test _to_dense_array with regular Python lists."""
    
    col = pd.Series([[1, 2, 3], [4, 5, 6]])
    result = _to_dense_array(col)
    
    assert result.shape == (2, 3)
    assert np.array_equal(result, np.array([[1, 2, 3], [4, 5, 6]]))


def test_to_dense_array_with_variable_length():
    """Test _to_dense_array with variable-length arrays."""
    
    col = pd.Series([[1, 2], [3, 4, 5], [6]])
    result = _to_dense_array(col)
    
    assert result.shape == (3, 3)
    assert result[0, 0] == 1 and result[0, 1] == 2
    assert np.isnan(result[0, 2])
    assert result[1, 2] == 5
    assert np.isnan(result[2, 1]) and np.isnan(result[2, 2])


def test_to_dense_array_with_scalar_values():
    """Test _to_dense_array with scalar values."""
    
    col = pd.Series([1, 2, 3])
    result = _to_dense_array(col)
    
    assert result.shape == (3, 1)
    assert np.array_equal(result.flatten(), np.array([1, 2, 3]))


def test_to_dense_array_with_mixed_arrays_and_scalars():
    """Test _to_dense_array with mixed array and scalar values."""
    
    col = pd.Series([[1, 2], 3, [4, 5, 6]])
    result = _to_dense_array(col)
    
    assert result.shape == (3, 3)
    assert result[0, 0] == 1 and result[0, 1] == 2
    assert result[1, 0] == 3
    assert np.isnan(result[1, 1]) and np.isnan(result[1, 2])


def test_to_dense_array_with_numpy_arrays():
    """Test _to_dense_array with numpy arrays."""
    
    col = pd.Series([np.array([1, 2]), np.array([3, 4, 5])])
    result = _to_dense_array(col)
    
    assert result.shape == (2, 3)
    assert result[0, 0] == 1 and result[0, 1] == 2
    assert np.isnan(result[0, 2])
    assert result[1, 2] == 5


def test_to_padded_array_with_regular_lists():
    """Test _to_padded_array with regular Python lists."""
    
    arrays = [[1, 2, 3], [4, 5, 6]]
    result = _to_padded_array(arrays)
    
    assert result.shape == (2, 3)
    assert np.array_equal(result, np.array([[1, 2, 3], [4, 5, 6]]))


def test_to_padded_array_with_variable_length():
    """Test _to_padded_array with variable-length arrays."""
    
    arrays = [[1, 2], [3, 4, 5], [6]]
    result = _to_padded_array(arrays)
    
    assert result.shape == (3, 3)
    assert result[0, 0] == 1 and result[0, 1] == 2
    assert np.isnan(result[0, 2])
    assert result[1, 2] == 5
    assert np.isnan(result[2, 1]) and np.isnan(result[2, 2])


def test_to_padded_array_with_scalar_values():
    """Test _to_padded_array with scalar values."""
    
    arrays = [1, 2, 3]
    result = _to_padded_array(arrays)
    
    assert result.shape == (3, 1)
    assert np.array_equal(result.flatten(), np.array([1, 2, 3]))


def test_to_padded_array_with_mixed_arrays_and_scalars():
    """Test _to_padded_array with mixed array and scalar values."""
    
    arrays = [[1, 2], 3, [4, 5, 6]]
    result = _to_padded_array(arrays)
    
    assert result.shape == (3, 3)
    assert result[0, 0] == 1 and result[0, 1] == 2
    assert result[1, 0] == 3
    assert np.isnan(result[1, 1]) and np.isnan(result[1, 2])


def test_to_padded_array_with_numpy_arrays():
    """Test _to_padded_array with numpy arrays."""
    
    arrays = [np.array([1, 2]), np.array([3, 4, 5])]
    result = _to_padded_array(arrays)
    
    assert result.shape == (2, 3)
    assert result[0, 0] == 1 and result[0, 1] == 2
    assert np.isnan(result[0, 2])
    assert result[1, 2] == 5


def test_flatten_data_vectorized_basic():
    """Test flatten_data_vectorized with basic input."""
    
    df = pd.DataFrame({
        "DispTelList_T": [np.array([0, 1]), np.array([1, 0])],
        "Disp_T": [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
        "cosphi": [np.array([0.8, 0.6]), np.array([0.7, 0.9])],
        "sinphi": [np.array([0.6, 0.8]), np.array([0.714, 0.436])],
        "loss": [np.array([0.1, 0.2]), np.array([0.15, 0.25])],
        "dist": [np.array([1.0, 2.0]), np.array([1.5, 2.5])],
        "width": [np.array([0.5, 0.6]), np.array([0.55, 0.65])],
        "length": [np.array([2.0, 3.0]), np.array([2.5, 3.5])],
        "size": [np.array([100.0, 200.0]), np.array([150.0, 250.0])],
        "E": [np.array([10.0, 20.0]), np.array([15.0, 25.0])],
        "ES": [np.array([5.0, 10.0]), np.array([7.5, 12.5])],
        "Xoff": [1.0, 2.0],
        "Yoff": [3.0, 4.0],
        "Xoff_intersect": [0.9, 1.9],
        "Yoff_intersect": [2.9, 3.9],
        "Erec": [10.0, 20.0],
        "ErecS": [5.0, 10.0],
        "EmissionHeight": [100.0, 200.0],
    })
    
    result = flatten_data_vectorized(df, n_tel=2, training_variables=["Disp_T", "cosphi", "sinphi", "loss", "dist", "width", "length", "size", "E", "ES"])
    
    assert "Disp_T_0" in result.columns
    assert "Disp_T_1" in result.columns
    assert "disp_x_0" in result.columns
    assert "disp_y_0" in result.columns
    assert len(result) == 2


def test_flatten_data_vectorized_with_pointing_corrections():
    """Test flatten_data_vectorized with pointing corrections enabled."""
    
    df = pd.DataFrame({
        "DispTelList_T": [np.array([0, 1]), np.array([1, 0])],
        "Disp_T": [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
        "cosphi": [np.array([0.8, 0.6]), np.array([0.7, 0.9])],
        "sinphi": [np.array([0.6, 0.8]), np.array([0.714, 0.436])],
        "loss": [np.array([0.1, 0.2]), np.array([0.15, 0.25])],
        "dist": [np.array([1.0, 2.0]), np.array([1.5, 2.5])],
        "width": [np.array([0.5, 0.6]), np.array([0.55, 0.65])],
        "length": [np.array([2.0, 3.0]), np.array([2.5, 3.5])],
        "size": [np.array([100.0, 200.0]), np.array([150.0, 250.0])],
        "E": [np.array([10.0, 20.0]), np.array([15.0, 25.0])],
        "ES": [np.array([5.0, 10.0]), np.array([7.5, 12.5])],
        "cen_x": [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
        "cen_y": [np.array([5.0, 6.0]), np.array([7.0, 8.0])],
        "fpointing_dx": [np.array([0.1, 0.2]), np.array([0.15, 0.25])],
        "fpointing_dy": [np.array([0.3, 0.4]), np.array([0.35, 0.45])],
        "Xoff": [1.0, 2.0],
        "Yoff": [3.0, 4.0],
        "Xoff_intersect": [0.9, 1.9],
        "Yoff_intersect": [2.9, 3.9],
        "Erec": [10.0, 20.0],
        "ErecS": [5.0, 10.0],
        "EmissionHeight": [100.0, 200.0],
    })
    
    result = flatten_data_vectorized(df, n_tel=2, training_variables=["Disp_T", "cosphi", "sinphi", "loss", "dist", "width", "length", "size", "E", "ES", "cen_x", "cen_y", "fpointing_dx", "fpointing_dy"], apply_pointing_corrections=True)
    
    assert "cen_x_0" in result.columns
    assert "cen_y_0" in result.columns


def test_flatten_data_vectorized_with_dtype_casting():
    """Test flatten_data_vectorized with explicit dtype."""
    
    df = pd.DataFrame({
        "DispTelList_T": [np.array([0, 1]), np.array([1, 0])],
        "Disp_T": [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
        "cosphi": [np.array([0.8, 0.6]), np.array([0.7, 0.9])],
        "sinphi": [np.array([0.6, 0.8]), np.array([0.714, 0.436])],
        "loss": [np.array([0.1, 0.2]), np.array([0.15, 0.25])],
        "dist": [np.array([1.0, 2.0]), np.array([1.5, 2.5])],
        "width": [np.array([0.5, 0.6]), np.array([0.55, 0.65])],
        "length": [np.array([2.0, 3.0]), np.array([2.5, 3.5])],
        "size": [np.array([100.0, 200.0]), np.array([150.0, 250.0])],
        "E": [np.array([10.0, 20.0]), np.array([15.0, 25.0])],
        "ES": [np.array([5.0, 10.0]), np.array([7.5, 12.5])],
        "Xoff": [1.0, 2.0],
        "Yoff": [3.0, 4.0],
        "Xoff_intersect": [0.9, 1.9],
        "Yoff_intersect": [2.9, 3.9],
        "Erec": [10.0, 20.0],
        "ErecS": [5.0, 10.0],
        "EmissionHeight": [100.0, 200.0],
    })
    
    result = flatten_data_vectorized(df, n_tel=2, training_variables=["Disp_T", "cosphi", "sinphi", "loss", "dist", "width", "length", "size", "E", "ES"], dtype=np.float32)
    
    assert result["Disp_T_0"].dtype == np.float32
    assert result["Erec"].dtype == np.float32


def test_flatten_data_vectorized_derived_features():
    """Test that derived features are correctly computed."""
    
    df = pd.DataFrame({
        "DispTelList_T": [np.array([0])],
        "Disp_T": [np.array([2.0])],
        "cosphi": [np.array([0.6])],
        "sinphi": [np.array([0.8])],
        "loss": [np.array([0.5])],
        "dist": [np.array([2.0])],
        "width": [np.array([1.0])],
        "length": [np.array([2.0])],
        "size": [np.array([100.0])],
        "E": [np.array([10.0])],
        "ES": [np.array([5.0])],
        "Xoff": [1.0],
        "Yoff": [3.0],
        "Xoff_intersect": [0.9],
        "Yoff_intersect": [2.9],
        "Erec": [10.0],
        "ErecS": [5.0],
        "EmissionHeight": [100.0],
    })
    
    result = flatten_data_vectorized(df, n_tel=1, training_variables=["Disp_T", "cosphi", "sinphi", "loss", "dist", "width", "length", "size", "E", "ES"])
    
    assert "disp_x_0" in result.columns
    assert "disp_y_0" in result.columns
    assert "loss_loss_0" in result.columns
    assert "loss_dist_0" in result.columns
    assert "width_length_0" in result.columns
    assert result["disp_x_0"].iloc[0] == pytest.approx(2.0 * 0.6)
    assert result["disp_y_0"].iloc[0] == pytest.approx(2.0 * 0.8)


def test_flatten_data_vectorized_missing_disp_column_for_extra_telescopes():
    """Disp columns beyond available telescopes should be NaN-filled."""

    df = pd.DataFrame(
        {
            "DispTelList_T": [np.array([0, 1, -1])],
            "Disp_T": [np.array([1.0, 2.0])],
            "cosphi": [np.array([0.8, 0.6])],
            "sinphi": [np.array([0.6, 0.8])],
            "loss": [np.array([0.1, 0.2])],
            "dist": [np.array([1.0, 2.0])],
            "width": [np.array([0.5, 0.6])],
            "length": [np.array([2.0, 3.0])],
            "size": [np.array([100.0, 200.0])],
            "E": [np.array([10.0, 20.0])],
            "ES": [np.array([5.0, 10.0])],
            "Xoff": [1.0],
            "Yoff": [3.0],
            "Xoff_intersect": [0.9],
            "Yoff_intersect": [2.9],
            "Erec": [10.0],
            "ErecS": [5.0],
            "EmissionHeight": [100.0],
        }
    )

    result = flatten_data_vectorized(
        df,
        n_tel=3,
        training_variables=[
            "Disp_T",
            "cosphi",
            "sinphi",
            "loss",
            "dist",
            "width",
            "length",
            "size",
            "E",
            "ES",
        ],
    )

    assert result["Disp_T_2"].isna().all()


def test_load_training_data_empty_files(tmp_path, mocker):
    """Test load_training_data with no matching data."""
    
    mock_file = tmp_path / "test.root"
    mock_file.touch()
    
    mock_root_file = mocker.MagicMock()
    mock_root_file.__enter__.return_value = {"data": None}
    mocker.patch("uproot.open", return_value=mock_root_file)
    
    result = load_training_data([str(mock_file)], n_tel=2, max_events=100)
    
    assert result.empty


def test_load_training_data_filters_by_n_tel(mocker):
    """Test load_training_data filters events by DispNImages."""
    
    df_raw = pd.DataFrame({
        "DispNImages": [2, 3, 2, 4],
        "MCxoff": [0.1, 0.2, 0.3, 0.4],
        "MCyoff": [0.5, 0.6, 0.7, 0.8],
        "MCe0": [100.0, 200.0, 150.0, 250.0],
        "DispTelList_T": [np.array([0, 1]), np.array([0, 1]), np.array([0, 1]), np.array([0, 1])],
        "Disp_T": [np.array([1.0, 2.0]), np.array([1.5, 2.5]), np.array([1.0, 2.0]), np.array([1.5, 2.5])],
        "cosphi": [np.array([0.8, 0.6]), np.array([0.7, 0.9]), np.array([0.8, 0.6]), np.array([0.7, 0.9])],
        "sinphi": [np.array([0.6, 0.8]), np.array([0.714, 0.436]), np.array([0.6, 0.8]), np.array([0.714, 0.436])],
        "loss": [np.array([0.1, 0.2]), np.array([0.15, 0.25]), np.array([0.1, 0.2]), np.array([0.15, 0.25])],
        "dist": [np.array([1.0, 2.0]), np.array([1.5, 2.5]), np.array([1.0, 2.0]), np.array([1.5, 2.5])],
        "width": [np.array([0.5, 0.6]), np.array([0.55, 0.65]), np.array([0.5, 0.6]), np.array([0.55, 0.65])],
        "length": [np.array([2.0, 3.0]), np.array([2.5, 3.5]), np.array([2.0, 3.0]), np.array([2.5, 3.5])],
        "size": [np.array([100.0, 200.0]), np.array([150.0, 250.0]), np.array([100.0, 200.0]), np.array([150.0, 250.0])],
        "E": [np.array([10.0, 20.0]), np.array([15.0, 25.0]), np.array([10.0, 20.0]), np.array([15.0, 25.0])],
        "ES": [np.array([5.0, 10.0]), np.array([7.5, 12.5]), np.array([5.0, 10.0]), np.array([7.5, 12.5])],
        "Xoff": [1.0, 2.0, 1.0, 2.0],
        "Yoff": [3.0, 4.0, 3.0, 4.0],
        "Xoff_intersect": [0.9, 1.9, 0.9, 1.9],
        "Yoff_intersect": [2.9, 3.9, 2.9, 3.9],
        "Erec": [10.0, 20.0, 10.0, 20.0],
        "ErecS": [5.0, 10.0, 5.0, 10.0],
        "EmissionHeight": [100.0, 200.0, 100.0, 200.0],
    })
    
    mock_tree = mocker.MagicMock()
    mock_tree.arrays.return_value = df_raw
    
    mock_root_file = mocker.MagicMock()
    mock_root_file.__enter__.return_value = {"data": mock_tree}
    mock_root_file.__exit__.return_value = None
    
    mocker.patch("uproot.open", return_value=mock_root_file)
    
    result = load_training_data(["dummy.root"], n_tel=2, max_events=-1)
    
    assert len(result) == 2
    assert "MCxoff" in result.columns
    assert "MCyoff" in result.columns
    assert "MCe0" in result.columns


def test_load_training_data_respects_max_events(mocker):
    """Test load_training_data respects max_events limit."""
    
    df_raw = pd.DataFrame({
        "DispNImages": [2] * 10,
        "MCxoff": np.arange(10, dtype=float) * 0.1,
        "MCyoff": np.arange(10, dtype=float) * 0.1,
        "MCe0": np.ones(10) * 100.0,
        "DispTelList_T": [np.array([0, 1])] * 10,
        "Disp_T": [np.array([1.0, 2.0])] * 10,
        "cosphi": [np.array([0.8, 0.6])] * 10,
        "sinphi": [np.array([0.6, 0.8])] * 10,
        "loss": [np.array([0.1, 0.2])] * 10,
        "dist": [np.array([1.0, 2.0])] * 10,
        "width": [np.array([0.5, 0.6])] * 10,
        "length": [np.array([2.0, 3.0])] * 10,
        "size": [np.array([100.0, 200.0])] * 10,
        "E": [np.array([10.0, 20.0])] * 10,
        "ES": [np.array([5.0, 10.0])] * 10,
        "Xoff": np.ones(10),
        "Yoff": np.ones(10) * 3.0,
        "Xoff_intersect": np.ones(10) * 0.9,
        "Yoff_intersect": np.ones(10) * 2.9,
        "Erec": np.ones(10) * 10.0,
        "ErecS": np.ones(10) * 5.0,
        "EmissionHeight": np.ones(10) * 100.0,
    })
    
    mock_tree = mocker.MagicMock()
    mock_tree.arrays.return_value = df_raw
    
    mock_root_file = mocker.MagicMock()
    mock_root_file.__enter__.return_value = {"data": mock_tree}
    mock_root_file.__exit__.return_value = None
    
    mocker.patch("uproot.open", return_value=mock_root_file)
    
    result = load_training_data(["dummy.root"], n_tel=2, max_events=5)
    
    assert len(result) <= 5


def test_load_training_data_handles_missing_data_tree(mocker):
    """Test load_training_data handles files without data tree."""
    
    mock_root_file = mocker.MagicMock()
    mock_root_file.__enter__.return_value = {}
    mock_root_file.__exit__.return_value = None
    
    mocker.patch("uproot.open", return_value=mock_root_file)
    
    result = load_training_data(["dummy.root"], n_tel=2, max_events=100)
    
    assert result.empty


def test_load_training_data_handles_file_read_error(mocker):
    """Test load_training_data handles exceptions when reading files."""
    
    mocker.patch("uproot.open", side_effect=Exception("File read error"))
    
    result = load_training_data(["dummy.root"], n_tel=2, max_events=100)
    
    assert result.empty


def test_load_training_data_multiple_files(mocker):
    """Test load_training_data concatenates multiple files."""
    
    df1 = pd.DataFrame({
        "DispNImages": [2, 2],
        "MCxoff": [0.1, 0.2],
        "MCyoff": [0.5, 0.6],
        "MCe0": [100.0, 150.0],
        "DispTelList_T": [np.array([0, 1]), np.array([0, 1])],
        "Disp_T": [np.array([1.0, 2.0]), np.array([1.0, 2.0])],
        "cosphi": [np.array([0.8, 0.6]), np.array([0.8, 0.6])],
        "sinphi": [np.array([0.6, 0.8]), np.array([0.6, 0.8])],
        "loss": [np.array([0.1, 0.2]), np.array([0.1, 0.2])],
        "dist": [np.array([1.0, 2.0]), np.array([1.0, 2.0])],
        "width": [np.array([0.5, 0.6]), np.array([0.5, 0.6])],
        "length": [np.array([2.0, 3.0]), np.array([2.0, 3.0])],
        "size": [np.array([100.0, 200.0]), np.array([100.0, 200.0])],
        "E": [np.array([10.0, 20.0]), np.array([10.0, 20.0])],
        "ES": [np.array([5.0, 10.0]), np.array([5.0, 10.0])],
        "Xoff": [1.0, 1.0],
        "Yoff": [3.0, 3.0],
        "Xoff_intersect": [0.9, 0.9],
        "Yoff_intersect": [2.9, 2.9],
        "Erec": [10.0, 10.0],
        "ErecS": [5.0, 5.0],
        "EmissionHeight": [100.0, 100.0],
    })
    
    df2 = pd.DataFrame({
        "DispNImages": [2],
        "MCxoff": [0.3],
        "MCyoff": [0.7],
        "MCe0": [200.0],
        "DispTelList_T": [np.array([0, 1])],
        "Disp_T": [np.array([1.0, 2.0])],
        "cosphi": [np.array([0.8, 0.6])],
        "sinphi": [np.array([0.6, 0.8])],
        "loss": [np.array([0.1, 0.2])],
        "dist": [np.array([1.0, 2.0])],
        "width": [np.array([0.5, 0.6])],
        "length": [np.array([2.0, 3.0])],
        "size": [np.array([100.0, 200.0])],
        "E": [np.array([10.0, 20.0])],
        "ES": [np.array([5.0, 10.0])],
        "Xoff": [1.0],
        "Yoff": [3.0],
        "Xoff_intersect": [0.9],
        "Yoff_intersect": [2.9],
        "Erec": [10.0],
        "ErecS": [5.0],
        "EmissionHeight": [100.0],
    })
    
    call_count = [0]
    def mock_arrays(*args, **kwargs):
        call_count[0] += 1
        return df1 if call_count[0] == 1 else df2
    
    mock_tree = mocker.MagicMock()
    mock_tree.arrays.side_effect = mock_arrays
    
    mock_root_file = mocker.MagicMock()
    mock_root_file.__enter__.return_value = {"data": mock_tree}
    mock_root_file.__exit__.return_value = None
    
    mocker.patch("uproot.open", return_value=mock_root_file)
    
    result = load_training_data(["dummy1.root", "dummy2.root"], n_tel=2, max_events=-1)
    
    assert len(result) == 3


def test_load_training_data_computes_log_MCe0(mocker):
    """Test load_training_data correctly computes log10 of MCe0."""
    
    df_raw = pd.DataFrame({
        "DispNImages": [2],
        "MCxoff": [0.1],
        "MCyoff": [0.5],
        "MCe0": [100.0],
        "DispTelList_T": [np.array([0, 1])],
        "Disp_T": [np.array([1.0, 2.0])],
        "cosphi": [np.array([0.8, 0.6])],
        "sinphi": [np.array([0.6, 0.8])],
        "loss": [np.array([0.1, 0.2])],
        "dist": [np.array([1.0, 2.0])],
        "width": [np.array([0.5, 0.6])],
        "length": [np.array([2.0, 3.0])],
        "size": [np.array([100.0, 200.0])],
        "E": [np.array([10.0, 20.0])],
        "ES": [np.array([5.0, 10.0])],
        "Xoff": [1.0],
        "Yoff": [3.0],
        "Xoff_intersect": [0.9],
        "Yoff_intersect": [2.9],
        "Erec": [10.0],
        "ErecS": [5.0],
        "EmissionHeight": [100.0],
    })
    
    mock_tree = mocker.MagicMock()
    mock_tree.arrays.return_value = df_raw
    
    mock_root_file = mocker.MagicMock()
    mock_root_file.__enter__.return_value = {"data": mock_tree}
    mock_root_file.__exit__.return_value = None
    
    mocker.patch("uproot.open", return_value=mock_root_file)
    
    result = load_training_data(["dummy.root"], n_tel=2, max_events=-1)
    
    assert "MCe0" in result.columns
    assert result["MCe0"].iloc[0] == pytest.approx(np.log10(100.0))
