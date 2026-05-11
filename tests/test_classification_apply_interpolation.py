"""Tests for energy-interpolated classification apply."""

import numpy as np
import pandas as pd

from eventdisplay_ml import data_processing, models


class DummyXGBClassifier:
    """Simple classifier returning a fixed gamma probability."""

    def __init__(self, proba):
        self._proba = float(proba)

    def predict_proba(self, x_data):
        """Return constant class probabilities for all rows."""
        n_rows = len(x_data)
        p1 = np.full(n_rows, self._proba, dtype=np.float32)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])


def test_energy_interpolation_bins_linear_weights():
    """Interpolation bins should bracket log-energy with linear alpha."""
    df = pd.DataFrame({"Erec": [10**-0.75, 10**0.25, 10**0.8]})
    bins = [
        {"E_min": -1.0, "E_max": -0.5},  # center -0.75
        {"E_min": -0.5, "E_max": 0.0},  # center -0.25
        {"E_min": 0.0, "E_max": 0.5},  # center 0.25
    ]

    e_bin_lo, e_bin_hi, e_alpha = data_processing.energy_interpolation_bins(df, bins)

    np.testing.assert_array_equal(e_bin_lo, np.array([0, 1, 2], dtype=np.int32))
    np.testing.assert_array_equal(e_bin_hi, np.array([0, 2, 2], dtype=np.int32))
    np.testing.assert_allclose(e_alpha, np.array([0.0, 1.0, 0.0], dtype=np.float32), atol=1e-7)


def test_apply_classification_models_interpolates_probabilities_and_thresholds(monkeypatch):
    """Classification apply should linearly interpolate between neighboring energy-bin models."""
    df = pd.DataFrame(
        {
            "e_bin_lo": [0, 0],
            "e_bin_hi": [1, 1],
            "e_alpha": [0.25, 0.75],
            "dummy": [1.0, 2.0],
        }
    )

    model_configs = {
        "models": {
            0: {
                "model": DummyXGBClassifier(0.2),
                "features": ["dummy"],
                "thresholds": {50: 0.4},
            },
            1: {
                "model": DummyXGBClassifier(1.0),
                "features": ["dummy"],
                "thresholds": {50: 0.8},
            },
        }
    }

    monkeypatch.setattr(models, "flatten_feature_data", lambda *args, **kwargs: df[["dummy"]])

    class_probability, is_gamma = models.apply_classification_models(df, model_configs, [50])

    np.testing.assert_allclose(class_probability, np.array([0.4, 0.8], dtype=np.float32), atol=1e-7)

    # Thresholds are interpolated the same way: [0.5, 0.7]
    np.testing.assert_array_equal(is_gamma[50], np.array([0, 1], dtype=np.uint8))
