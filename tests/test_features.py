"""Unit tests for eventdisplay_ml.features module."""

from eventdisplay_ml import features


def test_tmva_classification_feature_list_uses_ze_bin_not_pointing():
    tmva_features = features.features_tmva_style("classification", training=True)
    assert "ze_bin" in tmva_features
    assert "ArrayPointing_Elevation" not in tmva_features
    assert "ArrayPointing_Azimuth" not in tmva_features
