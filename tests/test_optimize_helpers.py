"""Tests for pure helper functions in scripts/optimize_classification.py."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import joblib
import numpy as np
import pandas as pd
import pytest

from eventdisplay_ml.scripts import optimize_classification as opt


@pytest.fixture
def efficiency_surface_inputs():
    energy = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)
    zenith = np.array([0.0, 0.0, 60.0, 60.0], dtype=float)
    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    return energy, zenith, values


def test_validate_and_convert_zenith_helpers():
    opt._validate_source_index(2.5)
    assert opt._zenith_to_inverse_cosine(60.0) == pytest.approx(2.0)
    with pytest.raises(ValueError, match=r"within \[2, 5\]"):
        opt._validate_source_index(5.5)


def test_build_uniform_axis_and_edges():
    axis = opt._build_uniform_axis(0.0, 1.0, 0.4)
    edges = opt._build_bin_edges_from_centers([1.0])
    assert axis.tolist() == pytest.approx([0.0, 0.4, 0.8, 1.0])
    assert edges.tolist() == pytest.approx([0.5, 1.5])
    with pytest.raises(ValueError, match="positive"):
        opt._build_uniform_axis(0.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="At least one bin center"):
        opt._build_bin_edges_from_centers([])


def test_reshape_surface_and_rate_interpolator(efficiency_surface_inputs):
    energy, zenith, values = efficiency_surface_inputs
    energy_axis, zenith_axis, surface = opt._reshape_surface(energy, zenith, values)
    interpolator, _, _ = opt._build_rate_interpolator(energy, zenith, values)
    sampled = opt._sample_rate_interpolator(interpolator, np.array([0.5]), np.array([60.0]))
    assert energy_axis.tolist() == [0.0, 1.0]
    assert zenith_axis.tolist() == [0.0, 60.0]
    assert surface[1, 0] == pytest.approx(3.0)
    assert sampled[0] == pytest.approx(3.5)
    with pytest.raises(ValueError, match="rectangular grid"):
        opt._reshape_surface([0.0, 1.0, 0.0], [0.0, 0.0, 60.0], [1.0, 2.0, 3.0])


def _constant_bg(points):
    return np.full(len(points), 0.5)


def test_mesh_li_ma_and_cut_optimization():
    log10_energy, zenith = opt._mesh_energy_zenith(np.array([0.0, 1.0]), np.array([0.0, 60.0]))
    li_ma = opt._li_ma_significance(np.array([30.0]), np.array([60.0]), opt._ALPHA)
    best = opt._optimize_cut_2d(
        np.array([0.0]),
        np.array([2.0]),
        np.array([1.0]),
        opt._ALPHA,
        100.0,
        np.array([0.2, 0.8]),
        _constant_bg,
    )
    assert log10_energy.shape == zenith.shape == (4,)
    assert li_ma[0] > 0
    assert best.tolist() == [0.8]


def _linear_interpolator(points):
    return points[:, 0] + points[:, 1]


def test_evaluate_efficiency_interpolator_and_model_zenith_centers():
    values = opt._evaluate_efficiency_interpolator(_linear_interpolator, [2.0], [0.5], (0.0, 1.0))
    centers = opt._model_zenith_bin_centers(
        [{"Ze_min": 0, "Ze_max": 20}, {"Ze_min": 20, "Ze_max": 40}]
    )
    assert values[0] == pytest.approx(1.5)
    assert centers.tolist() == pytest.approx([10.0, 30.0])
    with pytest.raises(ValueError, match="No zenith binning"):
        opt._model_zenith_bin_centers([])


def test_extract_load_and_fill_rates(monkeypatch, efficiency_surface_inputs):
    energy, zenith, values = efficiency_surface_inputs
    graph = MagicMock()
    graph.values.return_value = (energy, zenith, values)
    root_handle = MagicMock()
    root_handle.__enter__.return_value = {"gONRate": graph, "gBGRate": graph}
    root_handle.__exit__.return_value = False
    monkeypatch.setattr(opt.uproot, "open", lambda *_: root_handle)

    loaded = opt._load_rates(Path("rates.root"))
    filled = opt._fill_rate(
        Path("rates.root"),
        {0.5: (0.0, 1.0)},
        [{"Ze_min": 0, "Ze_max": 20}],
        source_strength=0.5,
        source_index=opt._CRAB_INDEX,
        energy_bin_width=0.5,
        inverse_cosine_zenith_bin_width=0.5,
    )

    assert loaded[2].tolist() == values.tolist()
    assert np.all(filled.signal_rate >= 0.0)
    assert filled.model_energy_axis.tolist() == [0.5]


def test_load_multi_bin_roc_and_main(tmp_path, monkeypatch):
    efficiency = pd.DataFrame(
        {
            "signal_efficiency": [0.2, 0.8],
            "background_efficiency": [0.1, 0.05],
            "threshold": [0.8, 0.2],
        }
    )
    roc_path = tmp_path / "roc_ebin0.joblib"
    roc_path_1 = tmp_path / "roc_ebin1.joblib"
    joblib.dump(
        {
            "energy_bins_log10_tev": {"E_min": -1.0, "E_max": 0.0},
            "zenith_bins_deg": [{"Ze_min": 0, "Ze_max": 20}],
            "models": {"xgboost": {"efficiency": efficiency}},
        },
        roc_path,
    )
    joblib.dump(
        {
            "energy_bins_log10_tev": {"E_min": 0.0, "E_max": 1.0},
            "zenith_bins_deg": [{"Ze_min": 0, "Ze_max": 20}],
            "models": {"xgboost": {"efficiency": efficiency}},
        },
        roc_path_1,
    )
    bg_interp, thresh_interp, energy_bins_map, zenith_bins_deg = opt._load_multi_bin_roc(
        [str(roc_path), str(roc_path_1)]
    )
    assert energy_bins_map[-0.5] == (-1.0, 0.0)
    assert zenith_bins_deg[0]["Ze_max"] == pytest.approx(20.0)
    assert np.isfinite(bg_interp(np.array([[0.0, 0.5]])))[0]
    assert np.isfinite(thresh_interp(np.array([[0.0, 0.5]])))[0]

    rate_grid = opt.RateGrid(
        log10_energy_tev=np.array([0.0]),
        zenith_deg=np.array([10.0]),
        on_rate=np.array([5.0]),
        background_rate=np.array([2.0]),
        signal_rate=np.array([3.0]),
        energy_axis=np.array([0.0]),
        zenith_axis=np.array([10.0]),
        model_energy_axis=np.array([0.0]),
        model_zenith_axis=np.array([10.0]),
        model_log10_energy=np.array([0.0]),
        model_zenith_deg=np.array([10.0]),
        model_signal_rate=np.array([3.0]),
        model_background_rate=np.array([2.0]),
    )

    class DummyTable(dict):
        def __init__(self):
            super().__init__()
            self.meta = {}
            self.output = None

        def write(self, output, format=None, overwrite=False):  # noqa: A002
            self.output = (output, format, overwrite)

    monkeypatch.setattr(
        opt,
        "_load_multi_bin_roc",
        lambda *_: (
            lambda points: np.full(len(points), 0.2),
            lambda points: np.full(len(points), 0.7),
            {0.0: (-0.5, 0.5)},
            [{"Ze_min": 0, "Ze_max": 20}],
        ),
    )
    monkeypatch.setattr(opt, "_fill_rate", lambda *_: rate_grid)
    monkeypatch.setattr(opt, "_optimize_cut_2d", lambda *_: np.array([0.6]))
    monkeypatch.setattr(opt, "_interpolate_efficiency_surface", lambda *_: np.array([0.6]))
    monkeypatch.setattr(opt, "_evaluate_efficiency_interpolator", lambda *_: np.array([0.2]))
    monkeypatch.setattr(opt, "Table", DummyTable)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "rates.root",
            str(roc_path),
            str(roc_path_1),
            "1.0",
            "--output",
            str(tmp_path / "optimized.ecsv"),
        ],
    )

    opt.main()
