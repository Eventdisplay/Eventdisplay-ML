"""Unit tests for features.py."""

import pytest

from eventdisplay_ml.features import (
    clip_intervals,
    excluded_features,
    features,
    features_tmva_style,
    target_features,
    telescope_features,
)

# ---------------------------------------------------------------------------
# target_features
# ---------------------------------------------------------------------------


def test_target_features_stereo_returns_three_residuals():
    result = target_features("stereo_analysis")
    assert result == ["Xoff_residual", "Yoff_residual", "E_residual"]


def test_target_features_classification_returns_empty():
    assert target_features("classification") == []


def test_target_features_unknown_raises():
    with pytest.raises(ValueError, match="Unknown analysis type"):
        target_features("unknown_type")


# ---------------------------------------------------------------------------
# excluded_features
# ---------------------------------------------------------------------------


def test_excluded_features_stereo_contains_pointing_corrections():
    result = excluded_features("stereo_analysis", ntel=2)
    assert "fpointing_dx_0" in result
    assert "fpointing_dx_1" in result
    assert "fpointing_dy_0" in result
    assert "fpointing_dy_1" in result


def test_excluded_features_stereo_scales_with_ntel():
    result2 = excluded_features("stereo_analysis", ntel=2)
    result4 = excluded_features("stereo_analysis", ntel=4)
    assert len(result4) > len(result2)
    assert "fpointing_dx_3" in result4
    assert "fpointing_dx_3" not in result2


def test_excluded_features_classification_contains_energy_and_position():
    result = excluded_features("classification", ntel=2)
    assert "Erec" in result
    assert "size_0" in result
    assert "cen_x_0" in result
    assert "cen_y_0" in result
    assert "E_0" in result
    assert "ES_0" in result


def test_excluded_features_classification_returns_set():
    result = excluded_features("classification", ntel=2)
    assert isinstance(result, set)


def test_excluded_features_unknown_raises():
    with pytest.raises(ValueError, match="Unknown analysis type"):
        excluded_features("mystery", ntel=4)


# ---------------------------------------------------------------------------
# telescope_features
# ---------------------------------------------------------------------------


def test_telescope_features_classification_excludes_disp_vars():
    result = telescope_features("classification")
    assert "Disp_T" not in result
    assert "DispXoff_T" not in result
    assert "size" in result


def test_telescope_features_stereo_includes_disp_vars():
    result = telescope_features("stereo_analysis")
    assert "Disp_T" in result
    assert "DispXoff_T" in result
    assert "DispYoff_T" in result
    assert "cen_x" in result
    assert "E" in result


def test_telescope_features_stereo_is_superset_of_classification():
    stereo = set(telescope_features("stereo_analysis"))
    cls = set(telescope_features("classification"))
    assert cls.issubset(stereo)


# ---------------------------------------------------------------------------
# features / _regression_features / _classification_features
# ---------------------------------------------------------------------------


def test_features_stereo_training_includes_mc_truth():
    result = features("stereo_analysis", training=True)
    assert "MCxoff" in result
    assert "MCyoff" in result
    assert "MCe0" in result


def test_features_stereo_inference_excludes_mc_truth():
    result = features("stereo_analysis", training=False)
    assert "MCxoff" not in result
    assert "MCyoff" not in result
    assert "MCe0" not in result


def test_features_stereo_both_have_standard_branches():
    for training in (True, False):
        result = features("stereo_analysis", training=training)
        assert "Erec" in result
        assert "ErecS" in result
        assert "EmissionHeight" in result
        assert "DispNImages" in result


def test_features_classification_returns_list():
    result = features("classification")
    assert isinstance(result, list)
    assert len(result) > 0


def test_features_classification_contains_mscw_mscl():
    result = features("classification")
    assert "MSCW" in result
    assert "MSCL" in result


def test_features_unknown_raises():
    with pytest.raises(ValueError, match="Unknown analysis type"):
        features("invalid_type")


# ---------------------------------------------------------------------------
# features_tmva_style
# ---------------------------------------------------------------------------


def test_features_tmva_style_classification_returns_list():
    result = features_tmva_style("classification")
    assert isinstance(result, list)
    assert "MSCW" in result
    assert "MSCL" in result
    assert "EmissionHeight" in result


def test_features_tmva_style_stereo_raises():
    with pytest.raises(ValueError, match="TMVA-style"):
        features_tmva_style("stereo_analysis")


def test_features_tmva_style_unknown_raises():
    with pytest.raises(ValueError, match="Unknown analysis type"):
        features_tmva_style("bogus")


# ---------------------------------------------------------------------------
# clip_intervals
# ---------------------------------------------------------------------------


def test_clip_intervals_returns_dict():
    result = clip_intervals()
    assert isinstance(result, dict)


def test_clip_intervals_key_variables_present():
    result = clip_intervals()
    for key in ("size", "Erec", "EmissionHeight", "MSCW", "MSCL", "tgrad_x"):
        assert key in result, f"Expected '{key}' in clip_intervals"


def test_clip_intervals_size_has_positive_lower_bound():
    result = clip_intervals()
    vmin, vmax = result["size"]
    assert vmin == pytest.approx(1.0)
    assert vmax is None


def test_clip_intervals_emission_height_bounded():
    result = clip_intervals()
    vmin, vmax = result["EmissionHeight"]
    assert vmin == pytest.approx(0.0)
    assert vmax == pytest.approx(100.0)


def test_clip_intervals_erec_has_lower_bound_only():
    result = clip_intervals()
    vmin, vmax = result["Erec"]
    assert vmin > 0
    assert vmax is None


def test_clip_intervals_values_are_tuples_of_length_two():
    for key, val in clip_intervals().items():
        assert len(val) == 2, f"clip_intervals['{key}'] should be a 2-tuple"
