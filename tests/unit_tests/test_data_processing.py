import pytest

import numpy as np
import pandas as pd

from eventdisplay_ml.data_processing import _to_dense_array, _to_padded_array, flatten_data_vectorized


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

