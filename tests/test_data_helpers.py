"""Unit tests for pure helper functions in data_processing.py."""

import numpy as np
import pandas as pd
import pytest

from eventdisplay_ml.data_processing import (
    _calculate_array_footprint,
    _clip_size_array,
    _compute_telescope_indices_to_keep,
    _get_index,
    _ground_to_shower_coords,
    _has_field,
    _make_mirror_area_columns,
    _make_tel_active_columns,
    _to_dense_array,
    _to_numpy_1d,
    apply_clip_intervals,
    energy_in_bins,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_tel_config():
    """Minimal telescope configuration for 4 telescopes with equal mirror area."""
    return {
        "tel_ids": np.array([0, 1, 2, 3]),
        "mirror_area": np.array([100.0, 100.0, 100.0, 100.0]),
        "mirror_areas": np.array([100.0, 100.0, 100.0, 100.0]),
        "tel_x": np.array([0.0, 100.0, -100.0, 0.0]),
        "tel_y": np.array([0.0, 0.0, 0.0, 100.0]),
        "max_tel_id": 3,
    }


@pytest.fixture
def mixed_area_tel_config():
    """6-telescope config with two mirror area types: 200 m² (0,1,2) and 50 m² (3,4,5)."""
    return {
        "tel_ids": np.array([0, 1, 2, 3, 4, 5]),
        "mirror_area": np.array([200.0, 200.0, 200.0, 50.0, 50.0, 50.0]),
        "mirror_areas": np.array([200.0, 200.0, 200.0, 50.0, 50.0, 50.0]),
        "tel_x": np.zeros(6),
        "tel_y": np.zeros(6),
        "max_tel_id": 5,
    }


# ---------------------------------------------------------------------------
# _to_dense_array
# ---------------------------------------------------------------------------


def test_to_dense_array_regular_series():
    col = pd.Series([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = _to_dense_array(col)
    assert result.shape == (2, 3)
    np.testing.assert_allclose(result, [[1, 2, 3], [4, 5, 6]])


def test_to_dense_array_variable_length_pads_with_nan():
    col = pd.Series([[1.0, 2.0], [3.0, 4.0, 5.0]])
    result = _to_dense_array(col)
    assert result.shape == (2, 3)
    assert np.isnan(result[0, 2])
    np.testing.assert_allclose(result[0, :2], [1.0, 2.0])
    np.testing.assert_allclose(result[1], [3.0, 4.0, 5.0])


def test_to_dense_array_numpy_arrays():
    col = pd.Series([np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])])
    result = _to_dense_array(col)
    assert result.shape == (2, 3)


# ---------------------------------------------------------------------------
# _to_numpy_1d
# ---------------------------------------------------------------------------


def test_to_numpy_1d_from_series():
    s = pd.Series([1.0, 2.0, 3.0])
    result = _to_numpy_1d(s, np.float32)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, [1.0, 2.0, 3.0])


def test_to_numpy_1d_from_numpy_array():
    arr = np.array([4.0, 5.0, 6.0])
    result = _to_numpy_1d(arr, np.float32)
    assert result.dtype == np.float32


def test_to_numpy_1d_from_list():
    result = _to_numpy_1d([1, 2, 3], np.float64)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# _has_field
# ---------------------------------------------------------------------------


def test_has_field_dataframe_present():
    df = pd.DataFrame({"a": [1], "b": [2]})
    assert _has_field(df, "a") is True
    assert _has_field(df, "c") is False


def test_has_field_dict():
    d = {"key1": 1}
    assert _has_field(d, "key1") is True
    assert _has_field(d, "missing") is False


# ---------------------------------------------------------------------------
# _get_index
# ---------------------------------------------------------------------------


def test_get_index_from_dataframe():
    idx = pd.RangeIndex(5, 10)
    df = pd.DataFrame({"a": range(5)}, index=idx)
    result = _get_index(df, 5)
    assert list(result) == list(idx)


def test_get_index_from_non_dataframe():
    result = _get_index([1, 2, 3], 3)
    assert isinstance(result, pd.RangeIndex)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# _clip_size_array
# ---------------------------------------------------------------------------


def test_clip_size_array_clips_to_minimum():
    arr = np.array([[0.0, 0.5, 1.0, 200.0]])
    result = _clip_size_array(arr)
    assert result[0, 0] == pytest.approx(1.0)  # vmin=1 from clip_intervals
    assert result[0, 1] == pytest.approx(1.0)
    assert result[0, 2] == pytest.approx(1.0)
    assert result[0, 3] == pytest.approx(200.0)


def test_clip_size_array_preserves_nan():
    arr = np.array([[np.nan, 50.0, np.nan]])
    result = _clip_size_array(arr)
    assert np.isnan(result[0, 0])
    assert np.isnan(result[0, 2])
    assert result[0, 1] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# _ground_to_shower_coords
# ---------------------------------------------------------------------------


def test_ground_to_shower_coords_zero_input():
    x = np.array([0.0])
    y = np.array([0.0])
    sx, sy = _ground_to_shower_coords(x, y, np.array([0.0]), np.array([1.0]), np.array([1.0]))
    assert sx[0] == pytest.approx(0.0)
    assert sy[0] == pytest.approx(0.0)


def test_ground_to_shower_coords_zero_elevation_gives_zero_y():
    """sin(elevation)=0 → shower_y must be zero regardless of input."""
    x = np.array([3.0])
    y = np.array([4.0])
    _, sy = _ground_to_shower_coords(x, y, np.array([0.5]), np.array([0.866]), np.array([0.0]))
    assert sy[0] == pytest.approx(0.0)


def test_ground_to_shower_coords_output_shape():
    n = 10
    x = np.random.default_rng(0).uniform(-200, 200, n)
    y = np.random.default_rng(1).uniform(-200, 200, n)
    sin_az = np.sin(np.radians(30.0)) * np.ones(n)
    cos_az = np.cos(np.radians(30.0)) * np.ones(n)
    sin_el = np.sin(np.radians(60.0)) * np.ones(n)
    sx, sy = _ground_to_shower_coords(x, y, sin_az, cos_az, sin_el)
    assert sx.shape == (n,)
    assert sy.shape == (n,)


# ---------------------------------------------------------------------------
# _make_mirror_area_columns
# ---------------------------------------------------------------------------


def test_make_mirror_area_columns_assigns_correct_values(simple_tel_config):
    n_evt = 3
    max_tel_id = 3
    result = _make_mirror_area_columns(simple_tel_config, max_tel_id, n_evt)
    assert set(result.keys()) == {f"mirror_area_{i}" for i in range(4)}
    for i in range(4):
        np.testing.assert_allclose(result[f"mirror_area_{i}"], [100.0, 100.0, 100.0])


def test_make_mirror_area_columns_missing_key_raises():
    bad_config = {"tel_ids": np.array([0]), "tel_x": np.zeros(1), "tel_y": np.zeros(1)}
    with pytest.raises(KeyError):
        _make_mirror_area_columns(bad_config, max_tel_id=0, n_evt=1)


# ---------------------------------------------------------------------------
# _make_tel_active_columns
# ---------------------------------------------------------------------------


def test_make_tel_active_columns_marks_active_telescopes():
    # tel_list_matrix: event 0 has tel 0,1,2 active; event 1 has tel 1,3 active
    tel_list_matrix = np.array([[0.0, 1.0, 2.0, np.nan], [np.nan, 1.0, np.nan, 3.0]])
    result = _make_tel_active_columns(tel_list_matrix, max_tel_id=3, n_evt=2)
    assert result["tel_active_0"][0] == pytest.approx(1.0)
    assert result["tel_active_1"][0] == pytest.approx(1.0)
    assert result["tel_active_2"][0] == pytest.approx(1.0)
    assert result["tel_active_3"][0] == pytest.approx(0.0)  # not active in event 0
    assert result["tel_active_0"][1] == pytest.approx(0.0)  # not active in event 1
    assert result["tel_active_1"][1] == pytest.approx(1.0)
    assert result["tel_active_3"][1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _compute_telescope_indices_to_keep
# ---------------------------------------------------------------------------


def test_compute_telescope_indices_to_keep_none_returns_all(simple_tel_config):
    result = _compute_telescope_indices_to_keep(
        simple_tel_config, max_tel_id=3, max_tel_per_type=None
    )
    assert result == [0, 1, 2, 3]


def test_compute_telescope_indices_to_keep_limits_per_type(mixed_area_tel_config):
    result = _compute_telescope_indices_to_keep(
        mixed_area_tel_config, max_tel_id=5, max_tel_per_type=1
    )
    # Should keep 1 from the large-area group and 1 from the small-area group
    assert len(result) == 2


def test_compute_telescope_indices_to_keep_max_per_type_two(mixed_area_tel_config):
    result = _compute_telescope_indices_to_keep(
        mixed_area_tel_config, max_tel_id=5, max_tel_per_type=2
    )
    assert len(result) == 4  # 2 from each group of 3


def test_compute_telescope_indices_to_keep_larger_than_group_keeps_all(mixed_area_tel_config):
    result = _compute_telescope_indices_to_keep(
        mixed_area_tel_config, max_tel_id=5, max_tel_per_type=10
    )
    assert len(result) == 6  # All telescopes kept since 10 > group sizes


# ---------------------------------------------------------------------------
# apply_clip_intervals
# ---------------------------------------------------------------------------


def test_apply_clip_intervals_clips_scalar_column():
    df = pd.DataFrame({"MSCW": [-5.0, 0.0, 5.0]})
    apply_clip_intervals(df)
    # MSCW clip is (-2, 2)
    assert df["MSCW"].iloc[0] == pytest.approx(-2.0)
    assert df["MSCW"].iloc[2] == pytest.approx(2.0)
    assert df["MSCW"].iloc[1] == pytest.approx(0.0)


def test_apply_clip_intervals_preserves_nan():
    df = pd.DataFrame({"MSCW": [-5.0, np.nan, 5.0]})
    apply_clip_intervals(df)
    assert np.isnan(df["MSCW"].iloc[1])


def test_apply_clip_intervals_applies_log10():
    df = pd.DataFrame({"Erec": [0.001, 0.1, 1.0, 10.0]})
    apply_clip_intervals(df, apply_log10=["Erec"])
    # Erec should be log10-transformed for valid values
    # vmin for Erec = 1e-3, so 0.001 is at min, stays at 1e-3, log10 = -3
    assert df["Erec"].iloc[2] == pytest.approx(0.0)  # log10(1.0) = 0
    assert df["Erec"].iloc[3] == pytest.approx(1.0)  # log10(10.0) = 1


def test_apply_clip_intervals_per_telescope_column():
    df = pd.DataFrame({"tgrad_x_0": [-100.0, 0.0, 100.0], "tgrad_x_1": [0.0, 0.0, 0.0]})
    apply_clip_intervals(df, n_tel=2)
    # tgrad_x clip is (-50, 50)
    assert df["tgrad_x_0"].iloc[0] == pytest.approx(-50.0)
    assert df["tgrad_x_0"].iloc[2] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# energy_in_bins
# ---------------------------------------------------------------------------


def test_energy_in_bins_assigns_nearest_bin():
    # Use energies well inside each bin to avoid tie-breaking at shared edges.
    # Bins have centers -1.5, -0.5, 0.5, 1.5 in log10(E/TeV).
    df = pd.DataFrame({"Erec": [0.003, 0.2, 5.0, 50.0]})
    bins = [
        {"E_min": -2.0, "E_max": -1.0},  # center = -1.5
        {"E_min": -1.0, "E_max": 0.0},  # center = -0.5
        {"E_min": 0.0, "E_max": 1.0},  # center =  0.5
        {"E_min": 1.0, "E_max": 2.0},  # center =  1.5
    ]
    result = energy_in_bins(df, bins)
    assert result.iloc[0] == 0  # log10(0.003) ≈ -2.5 → nearest to bin 0
    assert result.iloc[1] == 1  # log10(0.2)   ≈ -0.7 → nearest to bin 1
    assert result.iloc[2] == 2  # log10(5.0)   ≈  0.7 → nearest to bin 2
    assert result.iloc[3] == 3  # log10(50.0)  ≈  1.7 → nearest to bin 3


def test_energy_in_bins_invalid_erec_returns_minus_one():
    df = pd.DataFrame({"Erec": [0.0, -1.0, np.nan, 1.0]})
    bins = [{"E_min": -1.0, "E_max": 1.0}]
    result = energy_in_bins(df, bins)
    assert result.iloc[0] == -1  # Erec=0 is invalid
    assert result.iloc[1] == -1  # Erec<0 is invalid


# ---------------------------------------------------------------------------
# _calculate_array_footprint
# ---------------------------------------------------------------------------


@pytest.fixture
def square_tel_config():
    """4 telescopes at corners of a 200x200m square."""
    return {
        "tel_ids": np.array([0, 1, 2, 3]),
        "tel_x": np.array([-100.0, 100.0, 100.0, -100.0]),
        "tel_y": np.array([-100.0, -100.0, 100.0, 100.0]),
    }


def test_calculate_array_footprint_two_tel_returns_distance():
    # _calculate_array_footprint builds lookup_x[0..max_id] from tel_list_matrix values,
    # so tel_config must not contain IDs beyond max(tel_list_matrix values).
    tel_config = {
        "tel_ids": np.array([0, 1]),
        "tel_x": np.array([-100.0, 100.0]),
        "tel_y": np.array([0.0, 0.0]),
    }
    tel_list_matrix = np.array([[0.0, 1.0]])
    result = _calculate_array_footprint(tel_config, tel_list_matrix)
    assert result[0] == pytest.approx(200.0)


def test_calculate_array_footprint_three_tel_returns_positive():
    tel_config = {
        "tel_ids": np.array([0, 1, 2]),
        "tel_x": np.array([0.0, 100.0, 50.0]),
        "tel_y": np.array([0.0, 0.0, 100.0]),
    }
    tel_list_matrix = np.array([[0.0, 1.0, 2.0]])
    result = _calculate_array_footprint(tel_config, tel_list_matrix)
    assert result[0] > 0.0


def test_calculate_array_footprint_single_tel_returns_minus_one():
    tel_config = {
        "tel_ids": np.array([0]),
        "tel_x": np.array([0.0]),
        "tel_y": np.array([0.0]),
    }
    # Only one valid (non-NaN) entry → not enough telescopes → -1
    tel_list_matrix = np.array([[0.0]])
    result = _calculate_array_footprint(tel_config, tel_list_matrix)
    assert result[0] == pytest.approx(-1.0)


def test_calculate_array_footprint_multiple_events(square_tel_config):
    tel_list_matrix = np.array([[0.0, 1.0], [2.0, 3.0], [0.0, np.nan]])
    result = _calculate_array_footprint(square_tel_config, tel_list_matrix)
    assert len(result) == 3
    assert result[0] == pytest.approx(200.0)
    assert result[1] == pytest.approx(200.0)
    assert result[2] == pytest.approx(-1.0)
