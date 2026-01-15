"""Features used for XGB training and prediction."""


def target_features(analysis_type):
    """
    Get target features based on analysis type.

    Parameters
    ----------
    analysis_type : str
        Type of analysis.

    Returns
    -------
    list
        List of target feature names.
    """
    if analysis_type == "stereo_analysis":
        return ["MCxoff", "MCyoff", "MCe0"]  # sequence matters
    if "classification" in analysis_type:
        return []
    raise ValueError(f"Unknown analysis type: {analysis_type}")


def excluded_features(analysis_type, ntel):
    """
    Features not to be used for training/prediction.

    Parameters
    ----------
    analysis_type : str
        Type of analysis.
    ntel : int
        Number of telescopes.

    Returns
    -------
    set
        Set of excluded feature names.
    """
    if analysis_type == "stereo_analysis":
        return {
            "Xoff",  # Use Xoff_weighted_bdt instead
            "Yoff",  # Use Yoff_weighted_bdt instead
            *[f"fpointing_dx_{i}" for i in range(ntel)],
            *[f"fpointing_dy_{i}" for i in range(ntel)],
        }
    if "classification" in analysis_type:
        return {
            "Erec",
            *[f"cen_x_{i}" for i in range(ntel)],
            *[f"cen_y_{i}" for i in range(ntel)],
            *[f"size_{i}" for i in range(ntel)],
            *[f"E_{i}" for i in range(ntel)],
            *[f"ES_{i}" for i in range(ntel)],
            *[f"fpointing_dx_{i}" for i in range(ntel)],
            *[f"fpointing_dy_{i}" for i in range(ntel)],
        }
    raise ValueError(f"Unknown analysis type: {analysis_type}")


def telescope_features(analysis_type):
    """
    Telescope-type features.

    Disp variables with different indexing logic in data preparation.

    Parameters
    ----------
    analysis_type : str
        Type of analysis, e.g. ``"classification"`` or ``"stereo_analysis"``.

    Returns
    -------
    list
        List of telescope-level feature names.
    """
    var = [
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
    if analysis_type == "classification":
        return var

    return [
        *var,
        "size",
        "cen_x",
        "cen_y",
        "E",
        "ES",
        "Disp_T",
        "DispXoff_T",
        "DispYoff_T",
        "DispWoff_T",
    ]


def _regression_features(training):
    """Regression features."""
    var = [
        *telescope_features("stereo_analysis"),
        "DispNImages",
        "DispTelList_T",
        "Xoff",
        "Yoff",
        "Xoff_intersect",
        "Yoff_intersect",
        "Erec",
        "ErecS",
        "EmissionHeight",
        "ArrayPointing_Elevation",
    ]
    if training:
        return [*target_features("stereo_analysis"), *var]
    return var


def _classification_features():
    """Classification features."""
    var_tel = telescope_features("classification")
    var_array = [
        "DispNImages",
        "DispTelList_T",
        "EChi2S",
        "EmissionHeight",
        "EmissionHeightChi2",
        "MSCW",
        "MSCL",
        "ArrayPointing_Elevation",
    ]
    # energy used to bin the models, but not as feature
    return var_tel + var_array + ["Erec"]


def features(analysis_type, training=True):
    """
    Get features based on analysis type.

    Parameters
    ----------
    analysis_type : str
        Type of analysis.
    training : bool, optional
        If True (default), return features including target features.
        If False, return only non-target features (i.e. features used
        for prediction).

    Returns
    -------
    list
        List of feature names.
    """
    if analysis_type == "stereo_analysis":
        return _regression_features(training)
    if "classification" in analysis_type:
        return _classification_features()
    raise ValueError(f"Unknown analysis type: {analysis_type}")
