"""Stereo-regression contracts across infrastructure shared with classification.

The classification rewrite must not change the shared configuration, feature,
flattening, streaming, or evaluation behavior used by production regression.
"""

from unittest.mock import MagicMock

import awkward as ak
import numpy as np
import pandas as pd

from eventdisplay_ml import config, data_processing, evaluate, features, models


def test_stereo_feature_schema_keeps_target_order_and_excludes_pointing_offsets():
    """Pin the regression schema consumed by ROOT input, training, and apply."""
    assert features.target_features("stereo_analysis") == [
        "Xoff_residual",
        "Yoff_residual",
        "E_residual",
    ]
    assert features.features("stereo_analysis", training=True) == [
        "MCxoff",
        "MCyoff",
        "MCe0",
        *features.features("stereo_analysis", training=False),
    ]
    excluded = features.excluded_features("stereo_analysis", ntel=3)
    assert excluded == {
        "fpointing_dx_0",
        "fpointing_dx_1",
        "fpointing_dx_2",
        "fpointing_dy_0",
        "fpointing_dy_1",
        "fpointing_dy_2",
    }


def test_configure_apply_stereo_keeps_primary_and_high_multiplicity_scalers_separate(
    monkeypatch,
):
    """Ensure shared CLI loading never loses or cross-wires regression scalers."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "apply-xgb-stereo",
            "--input_file",
            "input.root",
            "--model_prefix",
            "two_image",
            "--model_prefix_high_multiplicity",
            "high_image",
            "--output_file",
            "prediction.root",
        ],
    )
    primary_scalers = {
        "target_mean": {"Xoff_residual": 1.0, "Yoff_residual": 2.0, "E_residual": 3.0},
        "target_std": {"Xoff_residual": 4.0, "Yoff_residual": 5.0, "E_residual": 6.0},
    }
    high_scalers = {
        "target_mean": {"Xoff_residual": -1.0, "Yoff_residual": -2.0, "E_residual": -3.0},
        "target_std": {"Xoff_residual": 7.0, "Yoff_residual": 8.0, "E_residual": 9.0},
    }
    loader = MagicMock(
        side_effect=[
            ({"xgboost": {"model": "two"}}, primary_scalers),
            ({"xgboost": {"model": "high"}}, high_scalers),
        ]
    )
    monkeypatch.setattr(config, "load_models", loader)

    result = config.configure_apply("stereo_analysis")

    assert result["target_mean"] == primary_scalers["target_mean"]
    assert result["target_std"] == primary_scalers["target_std"]
    assert result["target_mean_high_multiplicity"] == high_scalers["target_mean"]
    assert result["target_std_high_multiplicity"] == high_scalers["target_std"]
    assert result["models"]["xgboost"]["model"] == "two"
    assert result["models_high_multiplicity"]["xgboost"]["model"] == "high"


def test_flatten_feature_data_for_regression_removes_targets_and_pointing(monkeypatch):
    """Inference must never expose training truth or pointing corrections to a model."""
    flattened = pd.DataFrame(
        {
            "physics_feature": [1.0],
            "Xoff_residual": [9.0],
            "Yoff_residual": [8.0],
            "E_residual": [7.0],
            "fpointing_dx_0": [6.0],
            "fpointing_dy_0": [5.0],
        }
    )
    flatten = MagicMock(return_value=flattened)
    monkeypatch.setattr(data_processing, "flatten_telescope_data_vectorized", flatten)

    result = data_processing.flatten_feature_data(
        pd.DataFrame({"raw": [1]}),
        ntel=2,
        analysis_type="stereo_analysis",
        training=False,
        tel_config={"max_tel_id": 0},
        preview_rows=0,
    )

    assert result.columns.tolist() == ["physics_feature"]
    assert flatten.call_args.kwargs["analysis_type"] == "stereo_analysis"
    assert flatten.call_args.kwargs["training"] is False


def test_stereo_training_chunk_filters_invalid_energy_and_keeps_truth_baseline_alignment(
    monkeypatch,
):
    """Regression residuals must be row-aligned after both validity filters."""
    raw = ak.Array(
        [
            {"MCxoff": 11.0, "MCyoff": 21.0, "MCe0": 100.0},
            {"MCxoff": 12.0, "MCyoff": 22.0, "MCe0": 100.0},
            {"MCxoff": 13.0, "MCyoff": 23.0, "MCe0": 0.0},
            {"MCxoff": 14.0, "MCyoff": 24.0, "MCe0": 1000.0},
        ]
    )
    monkeypatch.setattr(
        data_processing,
        "flatten_telescope_data_vectorized",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                "Xoff_weighted_bdt": [1.0, 2.0, 3.0, 4.0],
                "Yoff_weighted_bdt": [10.0, 20.0, 30.0, 40.0],
                "ErecS": [10.0, 0.0, 100.0, 100.0],
            }
        ),
    )

    result = data_processing._flatten_training_chunk(
        raw,
        {"observatory": "veritas"},
        "stereo_analysis",
        {"max_tel_id": 0},
        False,
        "test chunk",
    )

    # Row 1 has invalid ErecS; row 2 has invalid MC energy.  The remaining
    # rows must still use their original corresponding truth and baseline.
    assert result.index.tolist() == [0, 3]
    np.testing.assert_allclose(result["Xoff_residual"], [10.0, 10.0])
    np.testing.assert_allclose(result["Yoff_residual"], [11.0, -16.0])
    np.testing.assert_allclose(result["E_residual"], [1.0, 1.0])
    np.testing.assert_allclose(result["ErecS"], [10.0, 100.0])


def test_extra_columns_keeps_stereo_energy_linear_for_residual_targets(monkeypatch):
    """The shared extra-column helper must not log-transform regression baselines."""
    monkeypatch.setattr(
        data_processing,
        "calculate_geomagnetic_angles",
        lambda *_args, **_kwargs: np.array([0.1, 0.2], dtype=np.float32),
    )
    raw = pd.DataFrame(
        {
            "Xoff": [1.0, 2.0],
            "Yoff": [3.0, 4.0],
            "Xoff_intersect": [0.5, 1.5],
            "Yoff_intersect": [2.5, 3.5],
            "DispNImages": [2, 3],
            "img2_ang": [0.2, 0.3],
            "Erec": [10.0, 100.0],
            "ErecS": [3.0, 30.0],
            "EmissionHeight": [8.0, 9.0],
            "ArrayPointing_Azimuth": [0.0, 1.0],
            "ArrayPointing_Elevation": [70.0, 71.0],
        }
    )

    result = data_processing.extra_columns(raw, "stereo_analysis", True, raw.index)

    np.testing.assert_allclose(result["Erec"], [10.0, 100.0])
    np.testing.assert_allclose(result["ErecS"], [3.0, 30.0])
    np.testing.assert_allclose(result["Diff_Xoff"], [0.5, 0.5])
    np.testing.assert_allclose(result["Diff_Yoff"], [0.5, 0.5])


def test_stereo_streaming_resets_chunk_indices_and_obeys_global_max_events(monkeypatch):
    """Chunked ROOT application must preserve exactly the requested event rows."""
    input_root = MagicMock()
    input_root.__enter__.return_value = {"data": MagicMock()}
    input_root.__exit__.return_value = False
    output_root = MagicMock()
    output_root.__enter__.return_value = output_root
    output_root.__exit__.return_value = False
    tree = MagicMock()
    applied_chunks = []
    chunks = [
        ak.Array([{"ErecS": 1.0}, {"ErecS": 2.0}]),
        ak.Array([{"ErecS": 3.0}, {"ErecS": 4.0}]),
    ]

    monkeypatch.setattr(models.uproot, "open", lambda *_args: input_root)
    monkeypatch.setattr(models.uproot, "recreate", lambda *_args: output_root)
    monkeypatch.setattr(models.uproot, "iterate", lambda *_args, **_kwargs: chunks)
    monkeypatch.setattr(
        models.data_processing, "read_telescope_config", lambda *_args: {"max_tel_id": 3}
    )
    monkeypatch.setattr(
        models.data_processing, "_resolve_branch_aliases", lambda *_args: (["ErecS"], {})
    )
    monkeypatch.setattr(models.data_processing, "_ensure_fpointing_fields", lambda chunk: chunk)
    monkeypatch.setattr(models.features, "features", lambda *_args, **_kwargs: ["ErecS"])
    monkeypatch.setattr(models, "_output_tree", lambda *_args: tree)
    monkeypatch.setattr(
        models,
        "_apply_model",
        lambda analysis_type, chunk, *_args: applied_chunks.append((analysis_type, chunk.copy())),
    )

    models.process_file_chunked(
        "stereo_analysis",
        {
            "input_file": "input.root",
            "output_file": "output.root",
            "max_events": 3,
            "chunk_size": 2,
        },
    )

    assert [len(chunk) for _, chunk in applied_chunks] == [2, 1]
    assert [chunk.index.tolist() for _, chunk in applied_chunks] == [[0, 1], [0]]
    assert [chunk["ErecS"].tolist() for _, chunk in applied_chunks] == [[1.0, 2.0], [3.0]]
    assert all(analysis_type == "stereo_analysis" for analysis_type, _ in applied_chunks)


def test_regression_resolution_uses_dataframe_indices_for_baseline_reconstruction(monkeypatch):
    """Diagnostics must pair residuals with matching non-contiguous event rows."""
    y_test = pd.DataFrame(
        {
            "Xoff_residual": [1.0, -2.0],
            "Yoff_residual": [3.0, 4.0],
            "E_residual": [0.5, -0.5],
        },
        index=[101, 909],
    )
    df = pd.DataFrame(
        {
            "Xoff_weighted_bdt": [100.0, 10.0],
            "Yoff_weighted_bdt": [200.0, 20.0],
            "ErecS": [10.0, 100.0],
        },
        index=[909, 101],
    )
    captured = []
    real_dataframe = pd.DataFrame

    def capture_first_dataframe(*args, **kwargs):
        frame = real_dataframe(*args, **kwargs)
        if not captured:
            captured.append(frame.copy())
        return frame

    monkeypatch.setattr(evaluate.pd, "DataFrame", capture_first_dataframe)
    evaluate.calculate_resolution(
        y_test.copy(),
        y_test.copy(),
        df,
        percentiles=[68],
        log_e_min=-2,
        log_e_max=3,
        n_bins=1,
        name="xgboost",
    )

    reconstructed = captured[0]
    np.testing.assert_allclose(reconstructed["MCxoff_true"], [11.0, 98.0])
    np.testing.assert_allclose(reconstructed["MCyoff_true"], [23.0, 204.0])
    np.testing.assert_allclose(reconstructed["MCe0"], [2.5, 0.5])
