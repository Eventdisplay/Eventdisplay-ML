"""Shared pytest fixtures and utilities for unit tests."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from eventdisplay_ml.features import telescope_features

# ============================================================================
# DataFrame Factory Functions
# ============================================================================


def create_base_df(n_rows=2, n_tel=2):
    """Create a base DataFrame with required columns.

    Parameters
    ----------
    n_rows : int
        Number of rows (events) in the DataFrame
    n_tel : int
        Number of telescopes (default 2)
    """
    return pd.DataFrame(
        {
            "DispTelList_T": [np.array(list(range(n_tel))) for _ in range(n_rows)],
            "Disp_T": [np.array([float(i + 1) for i in range(n_tel)]) for _ in range(n_rows)],
            "cosphi": [np.array([0.8 - i * 0.1 for i in range(n_tel)]) for _ in range(n_rows)],
            "sinphi": [np.array([0.6 + i * 0.1 for i in range(n_tel)]) for _ in range(n_rows)],
            "loss": [np.array([0.1 + i * 0.05 for i in range(n_tel)]) for _ in range(n_rows)],
            "dist": [np.array([1.0 + i * 0.5 for i in range(n_tel)]) for _ in range(n_rows)],
            "width": [np.array([0.5 + i * 0.05 for i in range(n_tel)]) for _ in range(n_rows)],
            "length": [np.array([2.0 + i * 0.5 for i in range(n_tel)]) for _ in range(n_rows)],
            "size": [np.array([100.0 + i * 50 for i in range(n_tel)]) for _ in range(n_rows)],
            "E": [np.array([10.0 + i * 10 for i in range(n_tel)]) for _ in range(n_rows)],
            "ES": [np.array([5.0 + i * 5 for i in range(n_tel)]) for _ in range(n_rows)],
            "Xoff": np.arange(n_rows, dtype=float) + 1.0,
            "Yoff": np.arange(n_rows, dtype=float) * 3.0 + 3.0,
            "Xoff_intersect": np.arange(n_rows, dtype=float) * 0.9 + 0.9,
            "Yoff_intersect": np.arange(n_rows, dtype=float) * 0.9 + 2.9,
            "Erec": np.arange(n_rows, dtype=float) * 10.0 + 10.0,
            "ErecS": np.arange(n_rows, dtype=float) * 5.0 + 5.0,
            "EmissionHeight": np.arange(n_rows, dtype=float) * 100.0 + 100.0,
            "img2_ang": np.arange(n_rows, dtype=float) * 15.0 + 30.0,
        }
    )


@pytest.fixture
def df_two_tel_base():
    """Two-telescope DataFrame."""
    return create_base_df(n_rows=2, n_tel=2)


@pytest.fixture
def df_two_tel_pointing(df_two_tel_base):
    """Two-telescope DataFrame with pointing corrections."""
    df = df_two_tel_base.copy()
    n_rows = len(df)
    df["cen_x"] = [np.array([1.0 + i, 2.0 + i]) for i in range(n_rows)]
    df["cen_y"] = [np.array([5.0 + i, 6.0 + i]) for i in range(n_rows)]
    df["fpointing_dx"] = [np.array([0.1 + i * 0.05, 0.2 + i * 0.05]) for i in range(n_rows)]
    df["fpointing_dy"] = [np.array([0.3 + i * 0.05, 0.4 + i * 0.05]) for i in range(n_rows)]
    return df


@pytest.fixture
def df_one_tel_base():
    """Single-telescope DataFrame."""
    return create_base_df(n_rows=1, n_tel=1)


@pytest.fixture
def df_three_tel_missing():
    """Three-telescope DataFrame with missing third telescope data."""
    return pd.DataFrame(
        {
            "DispTelList_T": [np.array([0, 1, -1])],
            "Disp_T": [np.array([1.0, 2.0])],  # Only 2 values for 3 telescopes
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
            "img2_ang": [45.0],
        }
    )


@pytest.fixture
def df_raw_template_2tel():
    """Raw DataFrame template for 2-telescope events (multiple rows)."""
    return create_base_df(n_rows=4, n_tel=2)


@pytest.fixture
def df_raw_two_files():
    """Pair of DataFrames to simulate loading from two files."""
    df1 = create_base_df(n_rows=2, n_tel=2)
    df2 = create_base_df(n_rows=1, n_tel=2)
    return df1, df2


@pytest.fixture
def sample_df():
    """Create a sample DataFrame with telescope data."""
    df = pd.DataFrame(
        {
            "DispTelList_T": [[0, 1, 2, 3], [0, 1], [1, 2, 3], [0, 1, 2, 3]],
            "DispNImages": [4, 2, 3, 4],
            "mscw": [1.0, 2.0, 3.0, 4.0],
            "mscl": [5.0, 6.0, 7.0, 8.0],
            "MSCW_T": [
                np.array([1.0, 2.0, 3.0, 4.0]),
                np.array([1.0, 2.0, np.nan, np.nan]),
                np.array([1.0, 2.0, 3.0, np.nan]),
                np.array([1.0, 2.0, 3.0, 4.0]),
            ],
            "fpointing_dx": [
                np.array([0.1, 0.2, 0.3, 0.4]),
                np.array([0.1, 0.2, np.nan, np.nan]),
                np.array([0.1, 0.2, 0.3, np.nan]),
                np.array([0.1, 0.2, 0.3, 0.4]),
            ],
            "fpointing_dy": [
                np.array([0.1, 0.2, 0.3, 0.4]),
                np.array([0.1, 0.2, np.nan, np.nan]),
                np.array([0.1, 0.2, 0.3, np.nan]),
                np.array([0.1, 0.2, 0.3, 0.4]),
            ],
            "Xoff": [0.5, 0.6, 0.7, 0.8],
            "Yoff": [0.3, 0.4, 0.5, 0.6],
            "Xoff_intersect": [0.51, 0.61, 0.71, 0.81],
            "Yoff_intersect": [0.31, 0.41, 0.51, 0.61],
            "Erec": [100.0, 200.0, 300.0, 400.0],
            "ErecS": [90.0, 180.0, 270.0, 360.0],
            "EmissionHeight": [10.0, 11.0, 12.0, 13.0],
            "img2_ang": [45.0, 50.0, 55.0, 60.0],
        }
    )

    for var in telescope_features():
        df[var] = [
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([1.0, 2.0, np.nan, np.nan]),
            np.array([1.0, 2.0, 3.0, np.nan]),
            np.array([1.0, 2.0, 3.0, 4.0]),
        ]
    return df


# ============================================================================
# Mock Helper Functions
# ============================================================================


def create_mock_model(n_estimators=1, importance_shape=(3,)):
    """Create a mock model with configurable estimators.

    Parameters
    ----------
    n_estimators : int
        Number of estimators for multi-output regression
    importance_shape : tuple
        Shape of feature importances array
    """
    rng = np.random.default_rng()
    mock_model = MagicMock()
    estimators = []
    for _ in range(n_estimators):
        est = MagicMock()
        est.feature_importances_ = rng.random(importance_shape)
        estimators.append(est)
    mock_model.estimators_ = estimators
    return mock_model


# ============================================================================
# Test Data Fixtures for Array/Padding Tests
# ============================================================================


@pytest.fixture
def arrays_regular():
    """Regular 2D list with equal-length rows."""
    return [[1, 2, 3], [4, 5, 6]]


@pytest.fixture
def arrays_variable_len():
    """List of lists with variable lengths."""
    return [[1, 2], [3, 4, 5], [6]]


@pytest.fixture
def arrays_mixed():
    """Mixed arrays and scalars input."""
    return [[1, 2], 3, [4, 5, 6]]


@pytest.fixture
def arrays_numpy():
    """List of numpy arrays with varying lengths."""
    return [np.array([1, 2]), np.array([3, 4, 5])]
