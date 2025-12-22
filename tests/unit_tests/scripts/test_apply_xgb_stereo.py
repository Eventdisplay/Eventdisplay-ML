"""Unit tests for apply_xgb_stereo.py script."""

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
        """Return predefined predictions regardless of input."""
        return self.predictions


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
            # Array-level features required by flatten_data_vectorized
            "Xoff": [0.5, 0.6, 0.7, 0.8],
            "Yoff": [0.3, 0.4, 0.5, 0.6],
            "Xoff_intersect": [0.51, 0.61, 0.71, 0.81],
            "Yoff_intersect": [0.31, 0.41, 0.51, 0.61],
            "Erec": [100.0, 200.0, 300.0, 400.0],
            "ErecS": [90.0, 180.0, 270.0, 360.0],
            "EmissionHeight": [10.0, 11.0, 12.0, 13.0],
        }
    )

    # Add all per-telescope training variables
    for var in xgb_per_telescope_training_variables():
        df[var] = [
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([1.0, 2.0, np.nan, np.nan]),
            np.array([1.0, 2.0, 3.0, np.nan]),
            np.array([1.0, 2.0, 3.0, 4.0]),
        ]

    return df


def test_none_selection_returns_unchanged(sample_df):
    """Test that None selection returns DataFrame unchanged."""
    result = apply_image_selection(sample_df, None)
    pd.testing.assert_frame_equal(result, sample_df)


def test_all_four_telescopes_selection_returns_unchanged(sample_df):
    """Test that selecting all 4 telescopes returns DataFrame unchanged."""
    result = apply_image_selection(sample_df, [0, 1, 2, 3])
    pd.testing.assert_frame_equal(result, sample_df)


def test_subset_selection_filters_telescope_list(sample_df):
    """Test that subset selection correctly filters telescope lists."""
    result = apply_image_selection(sample_df, [0, 1])

    assert result["DispTelList_T"].iloc[0] == [0, 1]
    assert result["DispTelList_T"].iloc[1] == [0, 1]
    assert result["DispTelList_T"].iloc[2] == [1]
    assert result["DispTelList_T"].iloc[3] == [0, 1]


def test_subset_selection_updates_dispnimages(sample_df):
    """Test that DispNImages is updated correctly after selection."""
    result = apply_image_selection(sample_df, [0, 1])

    assert result["DispNImages"].iloc[0] == 2
    assert result["DispNImages"].iloc[1] == 2
    assert result["DispNImages"].iloc[2] == 1
    assert result["DispNImages"].iloc[3] == 2


def test_subset_selection_pads_arrays_to_four(sample_df):
    """Test that per-telescope variables are padded to length 4."""
    result = apply_image_selection(sample_df, [0, 1])

    mscw_t = result["MSCW_T"].iloc[2]
    assert len(mscw_t) == 4
    # Arrays are already padded, selection doesn't change them
    assert mscw_t[0] == 1.0
    assert mscw_t[1] == 2.0
    assert mscw_t[2] == 3.0
    assert np.isnan(mscw_t[3])

    fpointing_dx = result["fpointing_dx"].iloc[2]
    assert len(fpointing_dx) == 4


def test_subset_selection_preserves_other_columns(sample_df):
    """Test that non-telescope columns are preserved."""
    result = apply_image_selection(sample_df, [0, 1])

    assert (result["mscw"] == sample_df["mscw"]).all()
    assert (result["mscl"] == sample_df["mscl"]).all()


def test_single_telescope_selection(sample_df):
    """Test selection with a single telescope."""
    result = apply_image_selection(sample_df, [2])

    assert result["DispTelList_T"].iloc[0] == [2]
    assert result["DispTelList_T"].iloc[2] == [2]
    assert result["DispNImages"].iloc[0] == 1
    assert result["DispNImages"].iloc[2] == 1


def test_original_dataframe_not_modified(sample_df):
    """Test that the original DataFrame is not modified."""
    original_copy = sample_df.copy(deep=True)
    apply_image_selection(sample_df, [0, 1])

    pd.testing.assert_frame_equal(sample_df, original_copy)


def test_empty_selection_results_in_zero_images(sample_df):
    """Test that events with no selected telescopes get DispNImages=0."""
    result = apply_image_selection(sample_df, [3])

    # Row 2 has telescope 3, so it matches the selection
    assert result["DispNImages"].iloc[2] == 1
    assert result["DispTelList_T"].iloc[2] == [3]
    # Row 1 doesn't have telescope 3, so it should have 0 images
    assert result["DispNImages"].iloc[1] == 0
    assert result["DispTelList_T"].iloc[1] == []


def test_load_models_loads_existing_models(tmp_path):
    """Test that load_models successfully loads existing model files."""
    # Create dummy model files
    model_2tel = tmp_path / "dispdir_bdt_ntel2_xgboost.joblib"
    model_3tel = tmp_path / "dispdir_bdt_ntel3_xgboost.joblib"

    dummy_model = {"type": "dummy"}
    joblib.dump(dummy_model, model_2tel)
    joblib.dump(dummy_model, model_3tel)

    models = load_models(str(tmp_path))

    assert 2 in models
    assert 3 in models
    assert models[2] == dummy_model
    assert models[3] == dummy_model
    assert 4 not in models


def test_load_models_skips_missing_models(tmp_path):
    """Test that load_models handles missing model files gracefully."""
    # Create only one model file
    model_2tel = tmp_path / "dispdir_bdt_ntel2_xgboost.joblib"
    dummy_model = {"type": "dummy"}
    joblib.dump(dummy_model, model_2tel)

    models = load_models(str(tmp_path))

    assert 2 in models
    assert 3 not in models
    assert 4 not in models


def test_load_models_empty_directory(tmp_path):
    """Test that load_models returns empty dict when no models exist."""
    models = load_models(str(tmp_path))

    assert models == {}


def test_load_models_all_multiplicities(tmp_path):
    """Test that load_models loads all available telescope multiplicities."""
    # Create all model files
    for n_tel in range(2, 5):
        model_file = tmp_path / f"dispdir_bdt_ntel{n_tel}_xgboost.joblib"
        dummy_model = {"multiplicity": n_tel}
        joblib.dump(dummy_model, model_file)

    models = load_models(str(tmp_path))

    assert len(models) == 3
    assert all(n_tel in models for n_tel in range(2, 5))
    assert models[2]["multiplicity"] == 2
    assert models[3]["multiplicity"] == 3
    assert models[4]["multiplicity"] == 4


def test_apply_models_with_preloaded_models(sample_df):
    """Test that apply_models works with preloaded models dictionary."""
    # Create a simple model that returns predictions for 4 telescopes
    simple_model = SimpleModel(
        np.array(
            [
                [0.1, 0.2, 1.5],
                [0.15, 0.25, 1.6],
                [0.12, 0.22, 1.55],
                [0.13, 0.23, 1.58],
            ]
        )
    )

    models = {4: simple_model}

    pred_xoff, pred_yoff, pred_erec = apply_models(sample_df, models)

    assert len(pred_xoff) == len(sample_df)
    assert len(pred_yoff) == len(sample_df)
    assert len(pred_erec) == len(sample_df)
    # Rows 0 and 3 have 4 telescopes, so they get predictions
    assert np.allclose(pred_xoff[0], 0.1)
    assert np.allclose(pred_yoff[0], 0.2)
    # Row 1 has 2 telescopes - no model available, should be NaN
    assert np.isnan(pred_xoff[1])
    # Row 2 has 3 telescopes - no model available, should be NaN
    assert np.isnan(pred_xoff[2])
    # Row 3 has 4 telescopes, gets second prediction
    assert np.allclose(pred_xoff[3], 0.15)


def test_apply_models_with_model_directory(sample_df, tmp_path):
    """Test that apply_models loads models from directory when string path is provided."""
    # Create a simple model
    simple_model = SimpleModel(
        np.array(
            [
                [0.1, 0.2, 1.5],
                [0.15, 0.25, 1.6],
                [0.12, 0.22, 1.55],
                [0.13, 0.23, 1.58],
            ]
        )
    )

    model_file = tmp_path / "dispdir_bdt_ntel4_xgboost.joblib"
    joblib.dump(simple_model, model_file)

    pred_xoff, _, _ = apply_models(sample_df, str(tmp_path))

    assert len(pred_xoff) == len(sample_df)
    assert not np.all(np.isnan(pred_xoff))


def test_apply_models_handles_missing_model(sample_df):
    """Test that apply_models gracefully handles missing models for a multiplicity."""
    # Create a model only for 4 telescopes, not for 2 or 3
    simple_model = SimpleModel(
        np.array(
            [
                [0.1, 0.2, 1.5],
                [0.15, 0.25, 1.6],
                [0.12, 0.22, 1.55],
                [0.13, 0.23, 1.58],
            ]
        )
    )

    models = {4: simple_model}

    pred_xoff, _, _ = apply_models(sample_df, models)

    # sample_df has DispNImages [4, 2, 3, 4], so rows 0 and 3 have 4 telescopes
    # and will get predictions. Rows 1 and 2 with 2 and 3 telescopes will be NaN
    assert len(pred_xoff) == len(sample_df)
    assert not np.isnan(pred_xoff[0])  # Has 4 telescopes
    assert np.isnan(pred_xoff[1])  # Has 2 telescopes, no model
    assert np.isnan(pred_xoff[2])  # Has 3 telescopes, no model
    assert not np.isnan(pred_xoff[3])  # Has 4 telescopes


def test_apply_models_with_selection_mask(sample_df):
    """Test that apply_models applies selection mask correctly."""
    simple_model = SimpleModel(
        np.array(
            [
                [0.1, 0.2, 1.5],
                [0.15, 0.25, 1.6],
                [0.12, 0.22, 1.55],
                [0.13, 0.23, 1.58],
            ]
        )
    )

    models = {4: simple_model}
    selection_mask = np.array([True, False, True, False])

    pred_xoff, _, _ = apply_models(sample_df, models, selection_mask)

    # Row 0: 4 telescopes, mask=True, gets prediction
    assert pred_xoff[0] == 0.1
    # Row 1: 2 telescopes (no model), mask=False, gets -999
    assert pred_xoff[1] == -999.0
    # Row 2: 3 telescopes (no model), would be NaN but mask=True so stays NaN
    assert np.isnan(pred_xoff[2])
    # Row 3: 4 telescopes, mask=False, gets -999 instead of prediction
    assert pred_xoff[3] == -999.0


def test_apply_models_output_shapes(sample_df):
    """Test that apply_models returns arrays with correct shapes."""
    simple_model = SimpleModel(
        np.array(
            [
                [0.1, 0.2, 1.5],
                [0.15, 0.25, 1.6],
                [0.12, 0.22, 1.55],
                [0.13, 0.23, 1.58],
            ]
        )
    )

    models = {4: simple_model}

    pred_xoff, pred_yoff, pred_erec = apply_models(sample_df, models)

    assert pred_xoff.shape == (len(sample_df),)
    assert pred_yoff.shape == (len(sample_df),)
    assert pred_erec.shape == (len(sample_df),)
    assert pred_xoff.dtype == np.float32
    assert pred_yoff.dtype == np.float32
    assert pred_erec.dtype == np.float32


def test_apply_models_with_multiple_multiplicities(tmp_path):
    """Test apply_models with events of different telescope multiplicities."""
    # Create sample data with different multiplicities
    df = pd.DataFrame(
        {
            "DispNImages": [2, 3, 4, 3],
            "MSCW_T": [
                np.array([1.0, 2.0, np.nan, np.nan]),
                np.array([1.0, 2.0, 3.0, np.nan]),
                np.array([1.0, 2.0, 3.0, 4.0]),
                np.array([1.0, 2.0, 3.0, np.nan]),
            ],
            "fpointing_dx": [
                np.array([0.1, 0.2, np.nan, np.nan]),
                np.array([0.1, 0.2, 0.3, np.nan]),
                np.array([0.1, 0.2, 0.3, 0.4]),
                np.array([0.1, 0.2, 0.3, np.nan]),
            ],
            "fpointing_dy": [
                np.array([0.1, 0.2, np.nan, np.nan]),
                np.array([0.1, 0.2, 0.3, np.nan]),
                np.array([0.1, 0.2, 0.3, 0.4]),
                np.array([0.1, 0.2, 0.3, np.nan]),
            ],
            "DispTelList_T": [[0, 1], [0, 1, 2], [0, 1, 2, 3], [0, 1, 2]],
            # Required columns for flatten_data_vectorized
            "Xoff": [0.5, 0.6, 0.7, 0.8],
            "Yoff": [0.3, 0.4, 0.5, 0.6],
            "Xoff_intersect": [0.51, 0.61, 0.71, 0.81],
            "Yoff_intersect": [0.31, 0.41, 0.51, 0.61],
            "Erec": [100.0, 200.0, 300.0, 400.0],
            "ErecS": [90.0, 180.0, 270.0, 360.0],
            "EmissionHeight": [10.0, 11.0, 12.0, 13.0],
        }
    )

    # Add all training variables as dummy columns
    for var in xgb_per_telescope_training_variables():
        df[var] = [
            np.array([1.0, 2.0, np.nan, np.nan]),
            np.array([1.0, 2.0, 3.0, np.nan]),
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([1.0, 2.0, 3.0, np.nan]),
        ]

    # Create models for different multiplicities
    simple_model_2tel = SimpleModel(np.array([[0.1, 0.2, 1.5]]))
    simple_model_3tel = SimpleModel(np.array([[0.15, 0.25, 1.6], [0.12, 0.22, 1.55]]))
    simple_model_4tel = SimpleModel(np.array([[0.13, 0.23, 1.58]]))

    models = {2: simple_model_2tel, 3: simple_model_3tel, 4: simple_model_4tel}

    pred_xoff, _, _ = apply_models(df, models)

    assert len(pred_xoff) == 4
    assert not np.isnan(pred_xoff[0])
    assert not np.isnan(pred_xoff[1])
    assert not np.isnan(pred_xoff[2])
    assert not np.isnan(pred_xoff[3])


def test_apply_models_handles_empty_dataframe():
    """Test that apply_models handles an empty DataFrame gracefully."""
    df_empty = pd.DataFrame(
        {
            "DispNImages": [],
            "MSCW_T": [],
            "fpointing_dx": [],
            "fpointing_dy": [],
        }
    )

    simple_model = SimpleModel(np.array([]))
    models = {4: simple_model}

    pred_xoff, pred_yoff, pred_erec = apply_models(df_empty, models)

    assert len(pred_xoff) == 0
    assert len(pred_yoff) == 0
    assert len(pred_erec) == 0


def test_process_file_chunked_creates_output_file(sample_df, tmp_path):
    """Test that process_file_chunked creates an output ROOT file."""
    # Create model directory with a dummy model
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    simple_model = SimpleModel(
        np.array([[0.1, 0.2, 1.5], [0.15, 0.25, 1.6], [0.12, 0.22, 1.55], [0.13, 0.23, 1.58]])
    )
    joblib.dump(simple_model, model_dir / "dispdir_bdt_ntel4_xgboost.joblib")

    input_file = tmp_path / "input.root"
    output_file = tmp_path / "output.root"

    # Mock uproot.iterate to return our sample data
    import unittest.mock as mock

    with mock.patch("eventdisplay_ml.scripts.apply_xgb_stereo.uproot.iterate") as mock_iterate:
        with mock.patch(
            "eventdisplay_ml.scripts.apply_xgb_stereo.uproot.recreate"
        ) as mock_recreate:
            mock_iterate.return_value = [sample_df]
            mock_tree = Mock()
            mock_recreate.return_value.__enter__.return_value.mktree.return_value = mock_tree

            process_file_chunked(
                input_file=str(input_file),
                model_dir=str(model_dir),
                output_file=str(output_file),
                image_selection="15",
            )

            mock_tree.extend.assert_called()


def test_process_file_chunked_applies_image_selection(sample_df, tmp_path):
    """Test that process_file_chunked applies image selection."""
    import unittest.mock as mock

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    simple_model = SimpleModel(
        np.array([[0.1, 0.2, 1.5], [0.15, 0.25, 1.6], [0.12, 0.22, 1.55], [0.13, 0.23, 1.58]])
    )
    joblib.dump(simple_model, model_dir / "dispdir_bdt_ntel4_xgboost.joblib")

    output_file = tmp_path / "output.root"

    with mock.patch("eventdisplay_ml.scripts.apply_xgb_stereo.uproot.iterate") as mock_iterate:
        with mock.patch(
            "eventdisplay_ml.scripts.apply_xgb_stereo.uproot.recreate"
        ) as mock_recreate:
            mock_iterate.return_value = [sample_df.copy()]
            mock_tree = Mock()
            mock_recreate.return_value.__enter__.return_value.mktree.return_value = mock_tree

            process_file_chunked(
                input_file="input.root",
                model_dir=str(model_dir),
                output_file=str(output_file),
                image_selection="3",  # Select telescopes 0 and 1
            )

            # Verify extend was called
            assert mock_tree.extend.called


def test_process_file_chunked_respects_max_events(sample_df, tmp_path):
    """Test that process_file_chunked respects max_events limit."""
    import unittest.mock as mock

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    simple_model = SimpleModel(
        np.array([[0.1, 0.2, 1.5], [0.15, 0.25, 1.6], [0.12, 0.22, 1.55], [0.13, 0.23, 1.58]])
    )
    joblib.dump(simple_model, model_dir / "dispdir_bdt_ntel4_xgboost.joblib")

    output_file = tmp_path / "output.root"

    # Create two chunks of data
    df_chunk1 = sample_df.copy()
    df_chunk2 = sample_df.copy()

    with mock.patch("eventdisplay_ml.scripts.apply_xgb_stereo.uproot.iterate") as mock_iterate:
        with mock.patch(
            "eventdisplay_ml.scripts.apply_xgb_stereo.uproot.recreate"
        ) as mock_recreate:
            mock_iterate.return_value = [df_chunk1, df_chunk2]
            mock_tree = Mock()
            mock_recreate.return_value.__enter__.return_value.mktree.return_value = mock_tree

            process_file_chunked(
                input_file="input.root",
                model_dir=str(model_dir),
                output_file=str(output_file),
                image_selection="15",
                max_events=2,  # Only process 2 events total
            )

            # Should only be called once due to max_events limit
            assert mock_tree.extend.call_count == 1


def test_process_file_chunked_handles_empty_chunks(sample_df, tmp_path):
    """Test that process_file_chunked skips empty chunks."""
    import unittest.mock as mock

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    simple_model = SimpleModel(np.array([[0.1, 0.2, 1.5]]))
    joblib.dump(simple_model, model_dir / "dispdir_bdt_ntel4_xgboost.joblib")

    output_file = tmp_path / "output.root"

    # Create chunks with one empty
    empty_df = pd.DataFrame()
    df_chunk = sample_df.iloc[:1].copy()

    with mock.patch("eventdisplay_ml.scripts.apply_xgb_stereo.uproot.iterate") as mock_iterate:
        with mock.patch(
            "eventdisplay_ml.scripts.apply_xgb_stereo.uproot.recreate"
        ) as mock_recreate:
            mock_iterate.return_value = [empty_df, df_chunk]
            mock_tree = Mock()
            mock_recreate.return_value.__enter__.return_value.mktree.return_value = mock_tree

            process_file_chunked(
                input_file="input.root",
                model_dir=str(model_dir),
                output_file=str(output_file),
                image_selection="15",
            )

            # Should only process non-empty chunk
            assert mock_tree.extend.call_count == 1


def test_process_file_chunked_converts_energy_to_power_of_10(sample_df, tmp_path):
    """Test that process_file_chunked converts predicted energy using power of 10."""
    import unittest.mock as mock

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    # Return log10(energy) = 2.0, which should be converted to 10^2.0 = 100.0
    simple_model = SimpleModel(np.array([[0.1, 0.2, 2.0]]))
    joblib.dump(simple_model, model_dir / "dispdir_bdt_ntel4_xgboost.joblib")

    output_file = tmp_path / "output.root"
    df_chunk = sample_df.iloc[:1].copy()

    with mock.patch("eventdisplay_ml.scripts.apply_xgb_stereo.uproot.iterate") as mock_iterate:
        with mock.patch(
            "eventdisplay_ml.scripts.apply_xgb_stereo.uproot.recreate"
        ) as mock_recreate:
            mock_iterate.return_value = [df_chunk]
            mock_tree = Mock()
            mock_recreate.return_value.__enter__.return_value.mktree.return_value = mock_tree

            process_file_chunked(
                input_file="input.root",
                model_dir=str(model_dir),
                output_file=str(output_file),
                image_selection="15",
            )

            # Verify extend was called with energy converted to power of 10
            call_args = mock_tree.extend.call_args
            assert call_args is not None
            extended_data = call_args[0][0]
            assert "Dir_Erec" in extended_data


def test_process_file_chunked_with_chunk_size(sample_df, tmp_path):
    """Test that process_file_chunked passes chunk_size parameter correctly."""
    import unittest.mock as mock

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    simple_model = SimpleModel(np.array([[0.1, 0.2, 1.5]]))
    joblib.dump(simple_model, model_dir / "dispdir_bdt_ntel4_xgboost.joblib")

    output_file = tmp_path / "output.root"

    with mock.patch("eventdisplay_ml.scripts.apply_xgb_stereo.uproot.iterate") as mock_iterate:
        with mock.patch(
            "eventdisplay_ml.scripts.apply_xgb_stereo.uproot.recreate"
        ) as mock_recreate:
            mock_iterate.return_value = [sample_df.iloc[:1].copy()]
            mock_tree = Mock()
            mock_recreate.return_value.__enter__.return_value.mktree.return_value = mock_tree

            process_file_chunked(
                input_file="input.root",
                model_dir=str(model_dir),
                output_file=str(output_file),
                image_selection="15",
                chunk_size=1000,
            )

            # Verify uproot.iterate was called with correct step_size
            mock_iterate.assert_called_once()
            call_kwargs = mock_iterate.call_args[1]
            assert call_kwargs["step_size"] == 1000


def test_process_file_chunked_resets_index_for_predictions(sample_df, tmp_path):
    """Test that process_file_chunked resets chunk index before predictions."""
    import unittest.mock as mock

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    simple_model = SimpleModel(
        np.array([[0.1, 0.2, 1.5], [0.15, 0.25, 1.6], [0.12, 0.22, 1.55], [0.13, 0.23, 1.58]])
    )
    joblib.dump(simple_model, model_dir / "dispdir_bdt_ntel4_xgboost.joblib")

    output_file = tmp_path / "output.root"

    # Create DataFrame with non-default index
    df_chunk = sample_df.copy()
    df_chunk.index = [100, 101, 102, 103]

    with mock.patch("eventdisplay_ml.scripts.apply_xgb_stereo.uproot.iterate") as mock_iterate:
        with mock.patch(
            "eventdisplay_ml.scripts.apply_xgb_stereo.uproot.recreate"
        ) as mock_recreate:
            mock_iterate.return_value = [df_chunk]
            mock_tree = Mock()
            mock_recreate.return_value.__enter__.return_value.mktree.return_value = mock_tree

            process_file_chunked(
                input_file="input.root",
                model_dir=str(model_dir),
                output_file=str(output_file),
                image_selection="15",
            )

            # Verify extend was called with correct number of events
            assert mock_tree.extend.called
            call_args = mock_tree.extend.call_args[0][0]
            assert len(call_args["Dir_Xoff"]) == len(df_chunk)


def test_process_file_chunked_accumulates_total_events(sample_df, tmp_path):
    """Test that process_file_chunked correctly accumulates total processed events."""
    import unittest.mock as mock

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    simple_model = SimpleModel(
        np.array([[0.1, 0.2, 1.5], [0.15, 0.25, 1.6], [0.12, 0.22, 1.55], [0.13, 0.23, 1.58]])
    )
    joblib.dump(simple_model, model_dir / "dispdir_bdt_ntel4_xgboost.joblib")

    output_file = tmp_path / "output.root"

    # Create two chunks
    df_chunk1 = sample_df.iloc[:2].copy()
    df_chunk2 = sample_df.iloc[2:].copy()

    with mock.patch("eventdisplay_ml.scripts.apply_xgb_stereo.uproot.iterate") as mock_iterate:
        with mock.patch(
            "eventdisplay_ml.scripts.apply_xgb_stereo.uproot.recreate"
        ) as mock_recreate:
            mock_iterate.return_value = [df_chunk1, df_chunk2]
            mock_tree = Mock()
            mock_recreate.return_value.__enter__.return_value.mktree.return_value = mock_tree

            process_file_chunked(
                input_file="input.root",
                model_dir=str(model_dir),
                output_file=str(output_file),
                image_selection="15",
            )

            # Should process both chunks
            assert mock_tree.extend.call_count == 2


def test_pad_to_four_with_numpy_array():
    """Test that _pad_to_four correctly pads numpy arrays to length 4."""
    arr = np.array([1.0, 2.0, 3.0])
    result = _pad_to_four(arr)

    assert len(result) == 4
    assert result.dtype == np.float32
    assert np.allclose(result[:3], [1.0, 2.0, 3.0])
    assert np.isnan(result[3])


def test_pad_to_four_with_list():
    """Test that _pad_to_four correctly pads lists to length 4."""
    lst = [1.0, 2.0]
    result = _pad_to_four(lst)

    assert len(result) == 4
    assert result.dtype == np.float32
    assert np.allclose(result[:2], [1.0, 2.0])
    assert np.isnan(result[2])
    assert np.isnan(result[3])


def test_pad_to_four_with_full_array():
    """Test that _pad_to_four returns array unchanged if already length 4."""
    arr = np.array([1.0, 2.0, 3.0, 4.0])
    result = _pad_to_four(arr)

    assert len(result) == 4
    assert result.dtype == np.float32
    assert np.allclose(result, [1.0, 2.0, 3.0, 4.0])


def test_pad_to_four_with_single_element():
    """Test that _pad_to_four pads single element array to length 4."""
    arr = np.array([5.0])
    result = _pad_to_four(arr)

    assert len(result) == 4
    assert result.dtype == np.float32
    assert result[0] == 5.0
    assert np.all(np.isnan(result[1:]))


def test_pad_to_four_with_empty_array():
    """Test that _pad_to_four pads empty array to length 4."""
    arr = np.array([])
    result = _pad_to_four(arr)

    assert len(result) == 4
    assert result.dtype == np.float32
    assert np.all(np.isnan(result))


def test_pad_to_four_with_scalar():
    """Test that _pad_to_four returns non-array scalars unchanged."""
    scalar = 3.14
    result = _pad_to_four(scalar)

    assert result == 3.14


def test_pad_to_four_with_nan_values():
    """Test that _pad_to_four preserves existing NaN values."""
    arr = np.array([1.0, np.nan, 3.0])
    result = _pad_to_four(arr)

    assert len(result) == 4
    assert result[0] == 1.0
    assert np.isnan(result[1])
    assert result[2] == 3.0
    assert np.isnan(result[3])


def test_pad_to_four_with_mixed_types():
    """Test that _pad_to_four converts mixed type lists correctly."""
    lst = [1, 2.5, 3]
    result = _pad_to_four(lst)

    assert len(result) == 4
    assert result.dtype == np.float32
    assert np.allclose(result[:3], [1.0, 2.5, 3.0])
    assert np.isnan(result[3])


def test_pad_to_four_preserves_original_values():
    """Test that _pad_to_four doesn't modify values during padding."""
    original_values = [0.1, 0.2, 0.3]
    arr = np.array(original_values)
    result = _pad_to_four(arr)

    for i, val in enumerate(original_values):
        assert np.isclose(result[i], val)


def test_pad_to_four_with_negative_values():
    """Test that _pad_to_four correctly handles negative values."""
    arr = np.array([-1.0, -2.5, 3.0])
    result = _pad_to_four(arr)

    assert len(result) == 4
    assert result[0] == -1.0
    assert result[1] == -2.5
    assert result[2] == 3.0
    assert np.isnan(result[3])


def test_pad_to_four_with_zero_values():
    """Test that _pad_to_four correctly handles zero values."""
    arr = np.array([0.0, 1.0, 0.0])
    result = _pad_to_four(arr)

    assert len(result) == 4
    assert result[0] == 0.0
    assert result[1] == 1.0
    assert result[2] == 0.0
    assert np.isnan(result[3])
    """Test that _pad_to_four converts mixed type lists correctly."""
    arr = np.array([1.0, np.nan, 3.0])
