"""Unit tests for apply_xgb_stereo script."""

from unittest.mock import Mock

import joblib
import numpy as np
import pandas as pd
import pytest

from eventdisplay_ml.scripts.apply_xgb_stereo import (
    _pad_to_four,
    apply_image_selection,
    apply_models,
    load_models,
    process_file_chunked,
)
from eventdisplay_ml.training_variables import xgb_per_telescope_training_variables


class SimpleModel:
    """A simple picklable model for testing."""

    def __init__(self, predictions):
        self.predictions = predictions

    def predict(self, x):
        """Predict using the simple model."""
        return self.predictions


# ============================================================================
# Consolidated pad_to_four tests (11 -> 1 parametrized + 1 special case)
# ============================================================================


@pytest.mark.parametrize(
    ("input_data", "expected_first_values", "check_nans"),
    [
        (np.array([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0], [3]),
        ([1.0, 2.0], [1.0, 2.0], [2, 3]),
        (np.array([5.0]), [5.0], [1, 2, 3]),
        (np.array([]), None, [0, 1, 2, 3]),
        (np.array([1.0, 2.0, 3.0, 4.0]), [1.0, 2.0, 3.0, 4.0], []),
        (np.array([1.0, np.nan, 3.0]), [1.0], [1, 3]),
        ([1, 2.5, 3], [1.0, 2.5, 3.0], [3]),
        (np.array([-1.0, -2.5, 3.0]), [-1.0, -2.5, 3.0], [3]),
        (np.array([0.0, 1.0, 0.0]), [0.0, 1.0, 0.0], [3]),
    ],
)
def test_pad_to_four(input_data, expected_first_values, check_nans):
    """Test _pad_to_four with various input types and edge cases."""
    result = _pad_to_four(input_data)

    assert len(result) == 4
    assert result.dtype == np.float32

    if expected_first_values:
        for i, val in enumerate(expected_first_values):
            assert np.isclose(result[i], val) or (np.isnan(val) and np.isnan(result[i]))

    for nan_idx in check_nans:
        assert np.isnan(result[nan_idx])


def test_pad_to_four_with_scalar():
    """Test _pad_to_four returns scalars unchanged."""
    scalar = 3.14
    result = _pad_to_four(scalar)
    assert result == 3.14


# ============================================================================
# Image Selection Tests
# ============================================================================


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
        }
    )

    for var in xgb_per_telescope_training_variables():
        df[var] = [
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([1.0, 2.0, np.nan, np.nan]),
            np.array([1.0, 2.0, 3.0, np.nan]),
            np.array([1.0, 2.0, 3.0, 4.0]),
        ]
    return df


@pytest.mark.parametrize(
    ("selection", "expected_tel_0", "expected_n_images_0"),
    [
        (None, [0, 1, 2, 3], 4),
        ([0, 1, 2, 3], [0, 1, 2, 3], 4),
        ([0, 1], [0, 1], 2),
        ([2], [2], 1),
    ],
)
def test_apply_image_selection(sample_df, selection, expected_tel_0, expected_n_images_0):
    """Test apply_image_selection with various telescope selections."""
    result = apply_image_selection(sample_df, selection)

    if selection is None or selection == [0, 1, 2, 3]:
        pd.testing.assert_frame_equal(result, sample_df)
    else:
        assert result["DispTelList_T"].iloc[0] == expected_tel_0
        assert result["DispNImages"].iloc[0] == expected_n_images_0


def test_apply_image_selection_preserves_original(sample_df):
    """Test that apply_image_selection doesn't modify the original DataFrame."""
    original_copy = sample_df.copy(deep=True)
    apply_image_selection(sample_df, [0, 1])
    pd.testing.assert_frame_equal(sample_df, original_copy)


# ============================================================================
# Model Loading Tests
# ============================================================================


@pytest.mark.parametrize(
    ("models_to_create", "expected_in_dict"),
    [
        ([2], [2]),
        ([2, 3, 4], [2, 3, 4]),
        ([], []),
    ],
)
def test_load_models(tmp_path, models_to_create, expected_in_dict):
    """Test load_models loads available models from directory."""
    for n_tel in models_to_create:
        model_file = tmp_path / f"dispdir_bdt_ntel{n_tel}_xgboost.joblib"
        joblib.dump({"multiplicity": n_tel}, model_file)

    models = load_models(str(tmp_path))

    for n_tel in expected_in_dict:
        assert n_tel in models
        assert models[n_tel]["multiplicity"] == n_tel
    assert len(models) == len(expected_in_dict)


# ============================================================================
# Model Application Tests
# ============================================================================


@pytest.mark.parametrize(
    "n_tel_multiplicities",
    [
        ([4]),
        ([2, 3, 4]),
    ],
)
def test_apply_models(sample_df, n_tel_multiplicities):
    """Test apply_models with different telescope multiplicities."""
    models = {}
    for n_tel in n_tel_multiplicities:
        # Create enough predictions for all rows (max 4 rows in sample_df)
        models[n_tel] = SimpleModel(np.array([[0.1 * n_tel, 0.2 * n_tel, 1.5]] * 4))

    pred_xoff, pred_yoff, pred_erec = apply_models(sample_df, models)

    assert all(len(p) == len(sample_df) for p in [pred_xoff, pred_yoff, pred_erec])
    assert all(p.dtype == np.float32 for p in [pred_xoff, pred_yoff, pred_erec])


def test_apply_models_with_missing_multiplicity(sample_df):
    """Test apply_models handles missing models gracefully."""
    models = {4: SimpleModel(np.array([[0.1, 0.2, 1.5]] * 4))}
    pred_xoff, _, _ = apply_models(sample_df, models)

    assert not np.isnan(pred_xoff[0])  # Row 0 has 4 telescopes
    assert np.isnan(pred_xoff[1])  # Row 1 has 2 telescopes
    assert np.isnan(pred_xoff[2])  # Row 2 has 3 telescopes
    assert not np.isnan(pred_xoff[3])  # Row 3 has 4 telescopes


def test_apply_models_with_selection_mask(sample_df):
    """Test apply_models respects selection mask."""
    models = {4: SimpleModel(np.array([[0.1, 0.2, 1.5]] * 4))}
    selection_mask = np.array([True, False, True, False])

    pred_xoff, _, _ = apply_models(sample_df, models, selection_mask)

    assert pred_xoff[0] == 0.1  # 4 tels, mask=True
    assert pred_xoff[1] == -999.0  # 2 tels, mask=False
    assert np.isnan(pred_xoff[2])  # 3 tels (no model)
    assert pred_xoff[3] == -999.0  # 4 tels, mask=False


def test_apply_models_from_directory(sample_df, tmp_path):
    """Test apply_models loads from directory string."""
    model_file = tmp_path / "dispdir_bdt_ntel4_xgboost.joblib"
    joblib.dump(SimpleModel(np.array([[0.1, 0.2, 1.5]] * 4)), model_file)

    pred_xoff, _, _ = apply_models(sample_df, str(tmp_path))
    assert len(pred_xoff) == len(sample_df)


# ============================================================================
# File Processing Tests
# ============================================================================


def test_process_file_chunked_creates_output(sample_df, tmp_path):
    """Test process_file_chunked creates output file."""
    from unittest.mock import patch

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_file = model_dir / "dispdir_bdt_ntel4_xgboost.joblib"
    joblib.dump(SimpleModel(np.array([[0.1, 0.2, 1.5]] * 4)), model_file)

    output_file = tmp_path / "output.root"

    with patch("eventdisplay_ml.scripts.apply_xgb_stereo.uproot.iterate") as mock_iterate:
        with patch("eventdisplay_ml.scripts.apply_xgb_stereo.uproot.recreate") as mock_recreate:
            mock_iterate.return_value = [sample_df.iloc[:1].copy()]
            mock_tree = Mock()
            mock_recreate.return_value.__enter__.return_value.mktree.return_value = mock_tree

            process_file_chunked(
                input_file="input.root",
                model_dir=str(model_dir),
                output_file=str(output_file),
                image_selection="15",
            )

            assert mock_tree.extend.called


@pytest.mark.parametrize(
    ("max_events", "expected_chunks"),
    [
        (None, 2),
        (2, 1),
    ],
)
def test_process_file_chunked_respects_limits(sample_df, tmp_path, max_events, expected_chunks):
    """Test process_file_chunked respects event limits."""
    from unittest.mock import patch

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    joblib.dump(
        SimpleModel(np.array([[0.1, 0.2, 1.5]] * 4)), model_dir / "dispdir_bdt_ntel4_xgboost.joblib"
    )

    with patch("eventdisplay_ml.scripts.apply_xgb_stereo.uproot.iterate") as mock_iterate:
        with patch("eventdisplay_ml.scripts.apply_xgb_stereo.uproot.recreate") as mock_recreate:
            mock_iterate.return_value = [sample_df.iloc[:2].copy(), sample_df.iloc[2:].copy()]
            mock_tree = Mock()
            mock_recreate.return_value.__enter__.return_value.mktree.return_value = mock_tree

            kwargs = {
                "input_file": "input.root",
                "model_dir": str(model_dir),
                "output_file": str(tmp_path / "output.root"),
                "image_selection": "15",
            }
            if max_events:
                kwargs["max_events"] = max_events

            process_file_chunked(**kwargs)
            assert mock_tree.extend.call_count == expected_chunks
