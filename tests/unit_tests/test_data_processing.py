"""Unit tests for data processing utilities."""

import numpy as np
import pandas as pd
import pytest

from eventdisplay_ml.data_processing import (
    _to_dense_array,
    _to_padded_array,
    flatten_data_vectorized,
    load_training_data,
)

# ============================================================================
# Parametrized Array Conversion Tests (consolidated from 10 functions)
# ============================================================================


@pytest.mark.parametrize(
    ("input_data", "expected_shape"),
    [
        ([[1, 2, 3], [4, 5, 6]], (2, 3)),
        ([[1, 2], [3, 4, 5], [6]], (3, 3)),
        ([1, 2, 3], (3, 1)),
        ([[1, 2], 3, [4, 5, 6]], (3, 3)),
    ],
)
def test_to_dense_array(input_data, expected_shape):
    """Test _to_dense_array with various input types."""
    col = pd.Series(input_data)
    result = _to_dense_array(col)
    assert result.shape == expected_shape


@pytest.mark.parametrize(
    ("input_data", "expected_shape"),
    [
        ([[1, 2, 3], [4, 5, 6]], (2, 3)),
        ([[1, 2], [3, 4, 5], [6]], (3, 3)),
        ([1, 2, 3], (3, 1)),
        ([[1, 2], 3, [4, 5, 6]], (3, 3)),
    ],
)
def test_to_padded_array(input_data, expected_shape):
    """Test _to_padded_array with various input types."""
    result = _to_padded_array(input_data)
    assert result.shape == expected_shape


def test_to_dense_array_with_numpy_arrays(arrays_numpy):
    """Test _to_dense_array with numpy arrays."""
    col = pd.Series(arrays_numpy)
    result = _to_dense_array(col)
    assert result.shape == (2, 3)


def test_to_padded_array_with_numpy_arrays(arrays_numpy):
    """Test _to_padded_array with numpy arrays."""
    result = _to_padded_array(arrays_numpy)
    assert result.shape == (2, 3)


# ============================================================================
# Data Flattening Tests
# ============================================================================


@pytest.mark.parametrize(
    ("n_tel", "with_pointing"),
    [
        (2, False),
        (2, True),
        (1, False),
    ],
)
def test_flatten_data_vectorized(
    n_tel, with_pointing, df_two_tel_base, df_two_tel_pointing, df_one_tel_base
):
    """Test flatten_data_vectorized with various telescope counts and pointing options."""
    if with_pointing and n_tel == 2:
        df = df_two_tel_pointing
    elif n_tel == 1:
        df = df_one_tel_base
    else:
        df = df_two_tel_base

    training_vars = [
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
    ]
    if with_pointing:
        training_vars.extend(["cen_x", "cen_y", "fpointing_dx", "fpointing_dy"])

    result = flatten_data_vectorized(
        df, n_tel=n_tel, training_variables=training_vars, apply_pointing_corrections=with_pointing
    )

    assert "Disp_T_0" in result.columns
    assert "disp_x_0" in result.columns
    assert len(result) == len(df)


def test_flatten_data_vectorized_derived_features(df_one_tel_base):
    """Test that derived features are correctly computed."""
    result = flatten_data_vectorized(
        df_one_tel_base,
        n_tel=1,
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

    assert "disp_x_0" in result.columns
    assert "disp_y_0" in result.columns
    assert "loss_loss_0" in result.columns
    assert "loss_dist_0" in result.columns
    assert "width_length_0" in result.columns
    # For df_one_tel_base: Disp_T[0]=1.0, cosphi[0]=0.8, sinphi[0]=0.6
    assert result["disp_x_0"].iloc[0] == pytest.approx(1.0 * 0.8)
    assert result["disp_y_0"].iloc[0] == pytest.approx(1.0 * 0.6)


def test_flatten_data_vectorized_missing_data(df_three_tel_missing):
    """Test that missing disp columns are filled with NaN."""
    result = flatten_data_vectorized(
        df_three_tel_missing,
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


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_flatten_data_vectorized_dtype(dtype, df_two_tel_base):
    """Test flatten_data_vectorized dtype casting."""
    result = flatten_data_vectorized(
        df_two_tel_base,
        n_tel=2,
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
        dtype=dtype,
    )
    assert result["Disp_T_0"].dtype == dtype


# ============================================================================
# Data Loading Tests
# ============================================================================


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
    df_raw = pd.DataFrame(
        {
            "DispNImages": [2, 3, 2, 4],
            "MCxoff": [0.1, 0.2, 0.3, 0.4],
            "MCyoff": [0.5, 0.6, 0.7, 0.8],
            "MCe0": [100.0, 200.0, 150.0, 250.0],
            "DispTelList_T": [np.array([0, 1])] * 4,
            "Disp_T": [np.array([1.0, 2.0])] * 4,
            "cosphi": [np.array([0.8, 0.6])] * 4,
            "sinphi": [np.array([0.6, 0.8])] * 4,
            "loss": [np.array([0.1, 0.2])] * 4,
            "dist": [np.array([1.0, 2.0])] * 4,
            "width": [np.array([0.5, 0.6])] * 4,
            "length": [np.array([2.0, 3.0])] * 4,
            "size": [np.array([100.0, 200.0])] * 4,
            "E": [np.array([10.0, 20.0])] * 4,
            "ES": [np.array([5.0, 10.0])] * 4,
            "Xoff": [1.0] * 4,
            "Yoff": [3.0] * 4,
            "Xoff_intersect": [0.9] * 4,
            "Yoff_intersect": [2.9] * 4,
            "Erec": [10.0] * 4,
            "ErecS": [5.0] * 4,
            "EmissionHeight": [100.0] * 4,
        }
    )

    mock_tree = mocker.MagicMock()
    mock_tree.arrays.return_value = df_raw

    mock_root_file = mocker.MagicMock()
    mock_root_file.__enter__.return_value = {"data": mock_tree}
    mock_root_file.__exit__.return_value = None
    mocker.patch("uproot.open", return_value=mock_root_file)

    result = load_training_data(["dummy.root"], n_tel=2, max_events=-1)
    assert len(result) == 2
    assert all(col in result.columns for col in ["MCxoff", "MCyoff", "MCe0"])


@pytest.mark.parametrize(
    ("max_events", "expected_max_rows"),
    [
        (5, 5),
        (3, 3),
        (-1, 10),
    ],
)
def test_load_training_data_max_events(mocker, max_events, expected_max_rows):
    """Test load_training_data respects max_events limit."""
    df_raw = pd.DataFrame(
        {
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
        }
    )

    mock_tree = mocker.MagicMock()
    mock_tree.arrays.return_value = df_raw
    mock_root_file = mocker.MagicMock()
    mock_root_file.__enter__.return_value = {"data": mock_tree}
    mock_root_file.__exit__.return_value = None
    mocker.patch("uproot.open", return_value=mock_root_file)

    result = load_training_data(["dummy.root"], n_tel=2, max_events=max_events)
    assert len(result) <= expected_max_rows


def test_load_training_data_handles_errors(mocker):
    """Test load_training_data handles file read exceptions."""
    mocker.patch("uproot.open", side_effect=Exception("File read error"))
    result = load_training_data(["dummy.root"], n_tel=2, max_events=100)
    assert result.empty


def test_load_training_data_multiple_files(mocker):
    """Test load_training_data concatenates multiple files."""
    df1 = pd.DataFrame(
        {
            "DispNImages": [2] * 2,
            "MCxoff": [0.1, 0.2],
            "MCyoff": [0.5, 0.6],
            "MCe0": [100.0, 150.0],
            "DispTelList_T": [np.array([0, 1])] * 2,
            "Disp_T": [np.array([1.0, 2.0])] * 2,
            "cosphi": [np.array([0.8, 0.6])] * 2,
            "sinphi": [np.array([0.6, 0.8])] * 2,
            "loss": [np.array([0.1, 0.2])] * 2,
            "dist": [np.array([1.0, 2.0])] * 2,
            "width": [np.array([0.5, 0.6])] * 2,
            "length": [np.array([2.0, 3.0])] * 2,
            "size": [np.array([100.0, 200.0])] * 2,
            "E": [np.array([10.0, 20.0])] * 2,
            "ES": [np.array([5.0, 10.0])] * 2,
            "Xoff": [1.0] * 2,
            "Yoff": [3.0] * 2,
            "Xoff_intersect": [0.9] * 2,
            "Yoff_intersect": [2.9] * 2,
            "Erec": [10.0] * 2,
            "ErecS": [5.0] * 2,
            "EmissionHeight": [100.0] * 2,
        }
    )
    df2 = df1.iloc[:1].copy()
    df2.loc[0, "MCe0"] = 200.0

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


def test_load_training_data_computes_log_mce0(mocker):
    """Test load_training_data correctly computes log10 of MCe0."""
    df_raw = pd.DataFrame(
        {
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
        }
    )

    mock_tree = mocker.MagicMock()
    mock_tree.arrays.return_value = df_raw
    mock_root_file = mocker.MagicMock()
    mock_root_file.__enter__.return_value = {"data": mock_tree}
    mock_root_file.__exit__.return_value = None
    mocker.patch("uproot.open", return_value=mock_root_file)

    result = load_training_data(["dummy.root"], n_tel=2, max_events=-1)
    assert "MCe0" in result.columns
    assert result["MCe0"].iloc[0] == pytest.approx(np.log10(100.0))
