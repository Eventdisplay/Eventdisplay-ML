"""Tests for classification performance plotting helpers."""

import csv

import numpy as np
import pandas as pd
import pytest

from eventdisplay_ml.scripts import plot_classification_performance_metrics as perf


def _efficiency(background_efficiency):
    return pd.DataFrame(
        {
            "threshold": np.linspace(0.0, 1.0, 3),
            "signal_efficiency": np.array([1.0, 0.8, 0.0]),
            "background_efficiency": np.asarray(background_efficiency, dtype=float),
        }
    )


def test_zenith_uniformity_summary_reports_worst_bin():
    data_joblib = {
        "models": {
            "xgboost": {
                "efficiency": _efficiency([1.0, 0.02, 0.0]),
                "efficiency_ze0": _efficiency([1.0, 0.01, 0.0]),
                "efficiency_ze1": _efficiency([1.0, 0.04, 0.0]),
            }
        }
    }

    summary = perf.zenith_uniformity_summary(data_joblib, ebin=3, target_signal_efficiency=0.8)

    assert summary["energy_bin"] == 3
    assert summary["worst_zenith_bin"] == 1
    assert summary["worst_zenith_background_efficiency"] == pytest.approx(0.04)
    assert summary["worst_to_best_background_efficiency_ratio"] == pytest.approx(4.0)
    assert summary["worst_to_overall_background_efficiency_ratio"] == pytest.approx(2.0)


def test_write_zenith_uniformity_summary(tmp_path):
    output_path = tmp_path / "summary.csv"
    rows = [
        {
            "energy_bin": 0,
            "target_signal_efficiency": 0.8,
            "overall_signal_efficiency": 0.8,
            "overall_background_efficiency": 0.02,
        }
    ]

    perf.write_zenith_uniformity_summary(rows, output_path)

    with output_path.open() as input_file:
        loaded_rows = list(csv.DictReader(input_file))
    assert loaded_rows[0]["energy_bin"] == "0"
    assert loaded_rows[0]["overall_background_efficiency"] == "0.02"
