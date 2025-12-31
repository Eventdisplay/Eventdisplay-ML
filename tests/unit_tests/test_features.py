"""Unit tests for training variables selection utilities."""

import pytest

import eventdisplay_ml.features


def test_target_features_stereo_analysis():
    result = eventdisplay_ml.features.target_features("stereo_analysis")
    assert result == ["MCxoff", "MCyoff", "MCe0"]


def test_target_features_classification_exact():
    result = eventdisplay_ml.features.target_features("classification")
    assert result == []


def test_target_features_classification_in_name():
    result = eventdisplay_ml.features.target_features("my_classification_run")
    assert result == []


def test_target_features_invalid_type():
    with pytest.raises(ValueError, match="Unknown analysis type"):
        eventdisplay_ml.features.target_features("unknown_type")


def test_excluded_features_stereo_analysis():
    ntel = 3
    result = eventdisplay_ml.features.excluded_features("stereo_analysis", ntel)
    expected = {
        "fpointing_dx_0",
        "fpointing_dx_1",
        "fpointing_dx_2",
        "fpointing_dy_0",
        "fpointing_dy_1",
        "fpointing_dy_2",
    }
    assert result == expected


def test_excluded_features_classification_exact():
    ntel = 2
    result = eventdisplay_ml.features.excluded_features("classification", ntel)
    expected = {
        "Erec",
        "size_0",
        "size_1",
        "E_0",
        "E_1",
        "ES_0",
        "ES_1",
        "fpointing_dx_0",
        "fpointing_dx_1",
        "fpointing_dy_0",
        "fpointing_dy_1",
    }
    assert result == expected


def test_excluded_features_classification_in_name():
    ntel = 1
    result = eventdisplay_ml.features.excluded_features("my_classification_run", ntel)
    expected = {
        "Erec",
        "size_0",
        "E_0",
        "ES_0",
        "fpointing_dx_0",
        "fpointing_dy_0",
    }
    assert result == expected


def test_excluded_features_invalid_type():
    with pytest.raises(ValueError, match="Unknown analysis type"):
        eventdisplay_ml.features.excluded_features("unknown_type", 2)


def test_telescope_features_classification():
    result = eventdisplay_ml.features.telescope_features("classification")
    expected = [
        "cen_x",
        "cen_y",
        "cosphi",
        "sinphi",
        "loss",
        "dist",
        "width",
        "length",
        "asym",
        "tgrad_x",
        "R_core",
        "fpointing_dx",
        "fpointing_dy",
    ]
    assert result == expected


def test_telescope_features_stereo_analysis():
    result = eventdisplay_ml.features.telescope_features("stereo_analysis")
    expected = [
        "cen_x",
        "cen_y",
        "cosphi",
        "sinphi",
        "loss",
        "dist",
        "width",
        "length",
        "asym",
        "tgrad_x",
        "R_core",
        "fpointing_dx",
        "fpointing_dy",
        "size",
        "E",
        "ES",
        "Disp_T",
        "DispXoff_T",
        "DispYoff_T",
        "DispWoff_T",
    ]
    assert result == expected


def test_telescope_features_other_analysis_type():
    result = eventdisplay_ml.features.telescope_features("regression")
    expected = [
        "cen_x",
        "cen_y",
        "cosphi",
        "sinphi",
        "loss",
        "dist",
        "width",
        "length",
        "asym",
        "tgrad_x",
        "R_core",
        "fpointing_dx",
        "fpointing_dy",
        "size",
        "E",
        "ES",
        "Disp_T",
        "DispXoff_T",
        "DispYoff_T",
        "DispWoff_T",
    ]
    assert result == expected


def test__regression_features_training_true():
    result = eventdisplay_ml.features._regression_features(training=True)
    # Should start with target features
    assert result[:3] == ["MCxoff", "MCyoff", "MCe0"]
    # Should contain all regression features
    expected_features = [
        "cen_x",
        "cen_y",
        "cosphi",
        "sinphi",
        "loss",
        "dist",
        "width",
        "length",
        "asym",
        "tgrad_x",
        "R_core",
        "fpointing_dx",
        "fpointing_dy",
        "size",
        "E",
        "ES",
        "Disp_T",
        "DispXoff_T",
        "DispYoff_T",
        "DispWoff_T",
        "DispNImages",
        "DispTelList_T",
        "Xoff",
        "Yoff",
        "Xoff_intersect",
        "Yoff_intersect",
        "Erec",
        "ErecS",
        "EmissionHeight",
    ]
    # All expected features should be present after the target features
    for feat in expected_features:
        assert feat in result


def test__regression_features_training_false():
    result = eventdisplay_ml.features._regression_features(training=False)
    # Should NOT start with target features
    assert "MCxoff" not in result
    assert "MCyoff" not in result
    assert "MCe0" not in result
    # Should contain all regression features
    expected_features = [
        "cen_x",
        "cen_y",
        "cosphi",
        "sinphi",
        "loss",
        "dist",
        "width",
        "length",
        "asym",
        "tgrad_x",
        "R_core",
        "fpointing_dx",
        "fpointing_dy",
        "size",
        "E",
        "ES",
        "Disp_T",
        "DispXoff_T",
        "DispYoff_T",
        "DispWoff_T",
        "DispNImages",
        "DispTelList_T",
        "Xoff",
        "Yoff",
        "Xoff_intersect",
        "Yoff_intersect",
        "Erec",
        "ErecS",
        "EmissionHeight",
    ]
    for feat in expected_features:
        assert feat in result
    # Should have the same length as training + 3 (for the targets)
    result_training = eventdisplay_ml.features._regression_features(training=True)
    assert len(result_training) == len(result) + 3


def test__classification_features_training_true():
    result = eventdisplay_ml.features._classification_features(training=True)
    expected_tel_features = [
        "cen_x",
        "cen_y",
        "cosphi",
        "sinphi",
        "loss",
        "dist",
        "width",
        "length",
        "asym",
        "tgrad_x",
        "R_core",
        "fpointing_dx",
        "fpointing_dy",
    ]
    expected_array_features = [
        "DispNImages",
        "DispTelList_T",
        "EChi2S",
        "EmissionHeight",
        "EmissionHeightChi2",
        "MSCW",
        "MSCL",
        "ArrayPointing_Elevation",
    ]
    # Should contain all telescope and array features, but not "Erec"
    for feat in expected_tel_features + expected_array_features:
        assert feat in result
    assert "Erec" not in result
    # Should start with telescope features
    assert result[: len(expected_tel_features)] == expected_tel_features
    # Should have correct length
    assert len(result) == len(expected_tel_features) + len(expected_array_features)


def test__classification_features_training_false():
    result = eventdisplay_ml.features._classification_features(training=False)
    expected_tel_features = [
        "cen_x",
        "cen_y",
        "cosphi",
        "sinphi",
        "loss",
        "dist",
        "width",
        "length",
        "asym",
        "tgrad_x",
        "R_core",
        "fpointing_dx",
        "fpointing_dy",
    ]

    expected_array_features = [
        "DispNImages",
        "DispTelList_T",
        "EChi2S",
        "EmissionHeight",
        "EmissionHeightChi2",
        "MSCW",
        "MSCL",
        "ArrayPointing_Elevation",
    ]
    # Should contain all telescope and array features, and "Erec"
    for feat in expected_tel_features + expected_array_features:
        assert feat in result
    assert "Erec" in result
    # "Erec" should be the last feature
    assert result[-1] == "Erec"
    # Should have correct length
    assert len(result) == len(expected_tel_features) + len(expected_array_features) + 1
