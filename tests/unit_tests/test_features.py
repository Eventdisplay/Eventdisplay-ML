"""Unit tests for eventdisplay_ml.features."""

import pytest

import eventdisplay_ml.features

# Constants for expected features
TARGETS = ["MCxoff", "MCyoff", "MCe0"]
TEL_CLASS = [
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
TEL_STEREO = [*TEL_CLASS, "size", "E", "ES", "Disp_T", "DispXoff_T", "DispYoff_T", "DispWoff_T"]
ARRAY_CLASS = [
    "DispNImages",
    "DispTelList_T",
    "EChi2S",
    "EmissionHeight",
    "EmissionHeightChi2",
    "MSCW",
    "MSCL",
    "ArrayPointing_Elevation",
]
REGRESSION = [
    *TEL_STEREO,
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


@pytest.mark.parametrize(
    ("analysis", "training", "targets", "features", "erec_last"),
    [
        ("stereo_analysis", True, TARGETS, REGRESSION, False),
        ("stereo_analysis", False, [], REGRESSION, False),
        ("classification", True, [], [*TEL_CLASS, *ARRAY_CLASS], False),
        ("classification", False, [], [*TEL_CLASS, *ARRAY_CLASS], True),
        ("my_classification_run", True, [], [*TEL_CLASS, *ARRAY_CLASS], False),
    ],
)
def test_features(analysis, training, targets, features, erec_last):
    result = eventdisplay_ml.features.features(analysis, training=training)
    for feat in features:
        assert feat in result
    if targets:
        assert result[: len(targets)] == targets
        assert len(result) == len(features) + len(targets)
    else:
        assert result[: len(TEL_CLASS)] == TEL_CLASS
        # For stereo_analysis, False, Erec is present in REGRESSION
        if analysis == "stereo_analysis" and not training:
            assert "Erec" in result
            assert len(result) == len(REGRESSION)
        elif erec_last:
            assert "Erec" in result
            assert result[-1] == "Erec"
            assert len(result) == len(TEL_CLASS) + len(ARRAY_CLASS) + 1
        else:
            assert "Erec" not in result
            assert len(result) == len(TEL_CLASS) + len(ARRAY_CLASS)


def test_features_wrong_analysis_type():
    with pytest.raises(ValueError, match="Unknown analysis type"):
        eventdisplay_ml.features.features("unknown_type", training=True)


@pytest.mark.parametrize(
    ("training", "expected_targets", "expected_features"),
    [
        (True, TARGETS, REGRESSION),
        (False, [], REGRESSION),
    ],
)
def test_regression_features(training, expected_targets, expected_features):
    result = eventdisplay_ml.features._regression_features(training=training)
    for feat in expected_features:
        assert feat in result
    if training:
        assert result[:3] == expected_targets
        assert len(result) == len(expected_features) + 3
    else:
        for t in expected_targets:
            assert t not in result
        assert len(result) == len(expected_features)


@pytest.mark.parametrize(
    ("training", "erec_last"),
    [
        (True, False),
        (False, True),
    ],
)
def test_classification_features(training, erec_last):
    result = eventdisplay_ml.features._classification_features(training=training)
    for feat in TEL_CLASS + ARRAY_CLASS:
        assert feat in result
    if erec_last:
        assert "Erec" in result
        assert result[-1] == "Erec"
        assert len(result) == len(TEL_CLASS) + len(ARRAY_CLASS) + 1
    else:
        assert "Erec" not in result
        assert result[: len(TEL_CLASS)] == TEL_CLASS
        assert len(result) == len(TEL_CLASS) + len(ARRAY_CLASS)


@pytest.mark.parametrize(
    ("analysis", "expected"),
    [
        ("classification", TEL_CLASS),
        ("stereo_analysis", TEL_STEREO),
        ("regression", TEL_STEREO),
    ],
)
def test_telescope_features(analysis, expected):
    assert eventdisplay_ml.features.telescope_features(analysis) == expected


@pytest.mark.parametrize(
    ("analysis", "expected"),
    [
        ("stereo_analysis", TARGETS),
        ("classification", []),
        ("my_classification_run", []),
    ],
)
def test_target_features(analysis, expected):
    assert eventdisplay_ml.features.target_features(analysis) == expected


def test_target_features_invalid_type():
    with pytest.raises(ValueError, match="Unknown analysis type"):
        eventdisplay_ml.features.target_features("unknown_type")


@pytest.mark.parametrize(
    ("analysis", "ntel", "expected"),
    [
        (
            "stereo_analysis",
            3,
            {f"fpointing_dx_{i}" for i in range(3)} | {f"fpointing_dy_{i}" for i in range(3)},
        ),
        (
            "classification",
            2,
            {"Erec"}
            | {f"size_{i}" for i in range(2)}
            | {f"E_{i}" for i in range(2)}
            | {f"ES_{i}" for i in range(2)}
            | {f"fpointing_dx_{i}" for i in range(2)}
            | {f"fpointing_dy_{i}" for i in range(2)},
        ),
        (
            "my_classification_run",
            1,
            {"Erec", "size_0", "E_0", "ES_0", "fpointing_dx_0", "fpointing_dy_0"},
        ),
    ],
)
def test_excluded_features(analysis, ntel, expected):
    assert eventdisplay_ml.features.excluded_features(analysis, ntel) == expected


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


def test_features_stereo_analysis_training_true():
    result = eventdisplay_ml.features.features("stereo_analysis", training=True)
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
    for feat in expected_features:
        assert feat in result
    # Should have correct length
    assert len(result) == len(expected_features) + 3


def test_features_stereo_analysis_training_false():
    result = eventdisplay_ml.features.features("stereo_analysis", training=False)
    # Should NOT start with target features
    assert "MCxoff" not in result
    assert "MCyoff" not in result
    assert "MCe0" not in result
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
    # Should have correct length
    assert len(result) == len(expected_features)


def test_features_classification_training_true():
    result = eventdisplay_ml.features.features("classification", training=True)
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
    for feat in expected_tel_features + expected_array_features:
        assert feat in result
    assert "Erec" not in result
    assert result[: len(expected_tel_features)] == expected_tel_features
    assert len(result) == len(expected_tel_features) + len(expected_array_features)


def test_features_classification_training_false():
    result = eventdisplay_ml.features.features("classification", training=False)
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
    for feat in expected_tel_features + expected_array_features:
        assert feat in result
    assert "Erec" in result
    assert result[-1] == "Erec"
    assert len(result) == len(expected_tel_features) + len(expected_array_features) + 1


def test_features_classification_in_name_training_true():
    result = eventdisplay_ml.features.features("my_classification_run", training=True)
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
    for feat in expected_tel_features + expected_array_features:
        assert feat in result
    assert "Erec" not in result
    assert result[: len(expected_tel_features)] == expected_tel_features
    assert len(result) == len(expected_tel_features) + len(expected_array_features)
