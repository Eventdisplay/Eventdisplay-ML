"""Features used for XGB training and prediction."""


def telescope_features(analysis_type, training):
    """
    Telescope-type features.

    Disp variables with different indexing logic in data preparation.
    """
    var = [
        "cen_x",
        "cen_y",
        "cosphi",
        "sinphi",
        "loss",
        "size",
        "dist",
        "width",
        "length",
        "asym",
        "tgrad_x",
        "R_core",
    ]
    if analysis_type == "classification":
        return var

    var = [*var, "E", "ES", "Disp_T", "DispXoff_T", "DispYoff_T", "DispWoff_T"]
    if not training:
        var += ["fpointing_dx", "fpointing_dy"]
    return var


def _regression_features(training):
    """Regression features."""
    var = [
        *telescope_features("stereo_analysis", training),
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
    if training:
        return ["MCxoff", "MCyoff", "MCe0", *var]
    return var


def _classification_features(training):
    """Classification features."""
    var_tel = telescope_features("classification", training)
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
    if training:
        return var_tel + var_array
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
        If True (default), return training features. If False, return
        all features including target features.

    Returns
    -------
    list
        List of feature names.
    """
    if analysis_type == "stereo_analysis":
        return _regression_features(training)
    if "classification" in analysis_type:
        return _classification_features(training)
    raise ValueError(f"Unknown analysis type: {analysis_type}")
