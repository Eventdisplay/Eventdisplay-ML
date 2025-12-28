"""Unit tests models."""

import joblib
import numpy as np
import pytest

from eventdisplay_ml.scripts.apply_xgb_stereo import (
    apply_regression_models,
    load_regression_models,
)


class SimpleModel:
    """A simple picklable model for testing."""

    def __init__(self, predictions):
        self.predictions = predictions

    def predict(self, x):
        """Predict using the simple model."""
        n = len(x)
        return self.predictions[:n]


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

    models = load_regression_models(str(tmp_path))

    for n_tel in expected_in_dict:
        assert n_tel in models
        assert models[n_tel]["multiplicity"] == n_tel
    assert len(models) == len(expected_in_dict)


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

    sample_df = sample_df.reset_index(drop=True)

    pred_xoff, pred_yoff, pred_erec = apply_regression_models(sample_df, models)

    assert all(len(p) == len(sample_df) for p in [pred_xoff, pred_yoff, pred_erec])
    assert all(p.dtype == np.float32 for p in [pred_xoff, pred_yoff, pred_erec])


def test_apply_models_with_missing_multiplicity(sample_df):
    """Test apply_models handles missing models gracefully."""
    models = {4: SimpleModel(np.array([[0.1, 0.2, 1.5]] * 4))}
    pred_xoff, _, _ = apply_regression_models(sample_df, models)

    assert not np.isnan(pred_xoff[0])  # Row 0 has 4 telescopes
    assert np.isnan(pred_xoff[1])  # Row 1 has 2 telescopes
    assert np.isnan(pred_xoff[2])  # Row 2 has 3 telescopes
    assert not np.isnan(pred_xoff[3])  # Row 3 has 4 telescopes
