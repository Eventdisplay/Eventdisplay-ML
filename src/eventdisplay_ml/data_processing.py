"""
Data processing for XGBoost analysis.

Provides common functions for flattening and preprocessing telescope array data.
"""

import logging

import numpy as np
import pandas as pd
import uproot

from eventdisplay_ml import features, utils

_logger = logging.getLogger(__name__)


def flatten_telescope_data_vectorized(df, n_tel, features, analysis_type, training=True):
    """
    Vectorized flattening of telescope array columns.

    Converts per-telescope arrays into individual feature columns, handles
    telescope indexing via DispTelList_T, and creates derived features.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing telescope data.
    n_tel : int
        Number of telescopes to flatten for.
    features : list[str]
        List of training variable names to flatten.
    analysis_type : str
        Type of analysis (e.g., "stereo_analysis").
    training : bool, optional
        If True, indicates training mode. Default is True.

    Returns
    -------
    pandas.DataFrame
        Flattened DataFrame with per-telescope columns suffixed by ``_{i}``
        for telescope index ``i``, plus derived features, and array features.
    """
    flat_features = {}
    tel_list_matrix = _to_dense_array(df["DispTelList_T"])
    n_evt = len(df)

    for var in features:
        data = _to_dense_array(df[var]) if var in df else np.full((n_evt, n_tel), np.nan)
        for i in range(n_tel):
            col_name = f"{var}_{i}"

            if var.startswith("Disp"):
                # Case 1: Simple index i
                if i < data.shape[1]:
                    flat_features[col_name] = data[:, i]
                else:
                    flat_features[col_name] = np.full(len(df), np.nan)
            else:
                # Case 2: Index lookup via DispTelList_T
                target_tel_indices = tel_list_matrix[:, i].astype(int)

                row_indices = np.arange(len(df))
                valid_mask = (target_tel_indices >= 0) & (target_tel_indices < data.shape[1])
                result = np.full(len(df), np.nan)
                result[valid_mask] = data[row_indices[valid_mask], target_tel_indices[valid_mask]]

                flat_features[col_name] = result

    df_flat = flatten_telescope_variables(n_tel, flat_features, df.index)
    return pd.concat([df_flat, extra_columns(df, analysis_type, training)], axis=1)


def _to_padded_array(arrays):
    """Convert list of variable-length arrays to fixed-size numpy array, padding with NaN."""
    max_len = max(len(arr) if hasattr(arr, "__len__") else 1 for arr in arrays)
    result = np.full((len(arrays), max_len), np.nan)
    for i, arr in enumerate(arrays):
        if hasattr(arr, "__len__"):
            result[i, : len(arr)] = arr
        else:
            result[i, 0] = arr
    return result


def _to_dense_array(col):
    """
    Convert a column of variable-length telescope data to a dense 2D numpy array.

    Handles uproot's awkward-style variable-length arrays from ROOT files
    by converting to plain Python lists first to avoid per-element iteration overhead.

    Parameters
    ----------
    col : pandas.Series
        Column containing variable-length arrays.

    Returns
    -------
    numpy.ndarray
        2D numpy array with shape (n_events, max_telescopes), padded with NaN.
    """
    arrays = col.tolist() if hasattr(col, "tolist") else list(col)
    try:
        return np.vstack(arrays)
    except (ValueError, TypeError):
        return _to_padded_array(arrays)


def flatten_feature_data(group_df, ntel, analysis_type, training):
    """Get flattened features for a group of events with given telescope multiplicity."""
    df_flat = flatten_telescope_data_vectorized(
        group_df,
        ntel,
        features.telescope_features(analysis_type),
        analysis_type=analysis_type,
        training=training,
    )
    excluded_columns = set(features.target_features(analysis_type)) | set(
        features.excluded_features(analysis_type, ntel)
    )
    return df_flat.drop(columns=excluded_columns, errors="ignore")


def load_training_data(model_configs, file_list, analysis_type):
    """
    Load and flatten training data from the mscw file for the requested telescope multiplicity.

    Parameters
    ----------
    model_configs : dict
        Dictionary containing model configuration parameters.
    file_list : str
        Path to text file containing list of input mscw files.
    analysis_type : str
        Type of analysis (e.g., "stereo_analysis").

    Returns
    -------
    pandas.DataFrame
        Flattened DataFrame ready for training.
    """
    max_events = model_configs.get("max_events", None)
    n_tel = model_configs["n_tel"]
    random_state = model_configs.get("random_state", None)

    _logger.info(f"--- Loading and Flattening Data for {analysis_type} for n_tel = {n_tel} ---")
    _logger.info(
        "Max events to process: "
        f"{max_events if max_events is not None and max_events > 0 else 'All available'}"
    )
    input_files = utils.read_input_file_list(file_list)

    branch_list = features.features(analysis_type, training=True)
    _logger.info(f"Branch list: {branch_list}")
    if max_events is not None and max_events > 0:
        max_events_per_file = max_events // len(input_files)
    else:
        max_events_per_file = None
    _logger.info(f"Max events per file: {max_events_per_file}")

    dfs = []
    for f in input_files:
        try:
            with uproot.open(f) as root_file:
                if "data" not in root_file:
                    _logger.warning(f"File: {f} does not contain a 'data' tree.")
                    continue

                _logger.info(f"Processing file: {f}")
                tree = root_file["data"]
                df = tree.arrays(branch_list, cut=model_configs.get("pre_cuts", None), library="pd")
                _logger.info(f"Number of events after event cut: {len(df)}")
                if max_events_per_file and len(df) > max_events_per_file:
                    df = df.sample(n=max_events_per_file, random_state=random_state)
                if not df.empty:
                    dfs.append(df)
        except Exception as e:
            raise FileNotFoundError(f"Error opening or reading file {f}: {e}") from e

    if len(dfs) == 0:
        raise ValueError("No data loaded from input files.")

    data_tree = pd.concat(dfs, ignore_index=True)
    _logger.info(f"Total events for n_tel={n_tel}: {len(data_tree)}")

    df_flat = flatten_telescope_data_vectorized(
        data_tree,
        n_tel,
        features.telescope_features(analysis_type),
        analysis_type,
        training=True,
    )

    if analysis_type == "stereo_analysis":
        df_flat["MCxoff"] = data_tree["MCxoff"]
        df_flat["MCyoff"] = data_tree["MCyoff"]
        df_flat["MCe0"] = np.log10(data_tree["MCe0"])
    elif analysis_type == "classification":
        df_flat["ze_bin"] = zenith_in_bins(
            90.0 - data_tree["ArrayPointing_Elevation"], model_configs.get("zenith_bins_deg", [])
        )

    df_flat.dropna(axis=1, how="all", inplace=True)
    _logger.info(f"Final events for n_tel={n_tel} after cleanup: {len(df_flat)}")

    print_variable_statistics(df_flat)

    return df_flat


def apply_image_selection(df, selected_indices, analysis_type, training=False):
    """
    Filter and pad telescope lists for selected indices.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing telescope data.
    selected_indices : list[int] or None
        List of selected telescope indices. If None or all 4 telescopes
        are selected, the DataFrame is returned unchanged.
    analysis_type : str, optional
        Type of analysis (e.g., "stereo_analysis")
    training : bool, optional
        If True, indicates training mode. Default is False.

    Returns
    -------
    pandas.DataFrame
        DataFrame with updated "DispTelList_T" and "DispNImages" columns,
        and per-telescope variables padded to length 4 with NaN.
    """
    if selected_indices is None or len(selected_indices) == 4:
        return df

    selected_set = set(selected_indices)

    def calculate_intersection(tel_list):
        return [tel_idx for tel_idx in tel_list if tel_idx in selected_set]

    df = df.copy()
    df["DispTelList_T_new"] = df["DispTelList_T"].apply(calculate_intersection)
    df["DispNImages_new"] = df["DispTelList_T_new"].apply(len)

    _logger.info(
        f"\n{df[['DispNImages', 'DispTelList_T', 'DispNImages_new', 'DispTelList_T_new']].head(20).to_string()}"
    )

    df["DispTelList_T"] = df["DispTelList_T_new"]
    df["DispNImages"] = df["DispNImages_new"]
    df = df.drop(columns=["DispTelList_T_new", "DispNImages_new"])

    pad_vars = features.telescope_features(analysis_type)

    for var_name in pad_vars:
        if var_name in df.columns:
            df[var_name] = df[var_name].apply(_pad_to_four)

    return df


def _pad_to_four(arr_like):
    """Pad a per-telescope array-like to length 4 with NaN values."""
    if isinstance(arr_like, (list, np.ndarray)):
        arr = np.asarray(arr_like, dtype=np.float32)
        pad = max(0, 4 - arr.shape[0])
        if pad:
            arr = np.pad(arr, (0, pad), mode="constant", constant_values=np.nan)
        return arr
    return arr_like


def flatten_telescope_variables(n_tel, flat_features, index):
    """Generate dataframe for telescope variables flattened for n_tel telescopes."""
    df_flat = pd.DataFrame(flat_features, index=index)
    df_flat = df_flat.astype(np.float32)

    new_cols = {}
    for i in range(n_tel):
        if f"Disp_T_{i}" in df_flat:
            new_cols[f"disp_x_{i}"] = df_flat[f"Disp_T_{i}"] * df_flat[f"cosphi_{i}"]
            new_cols[f"disp_y_{i}"] = df_flat[f"Disp_T_{i}"] * df_flat[f"sinphi_{i}"]
        new_cols[f"loss_loss_{i}"] = df_flat[f"loss_{i}"] ** 2
        new_cols[f"loss_dist_{i}"] = df_flat[f"loss_{i}"] * df_flat[f"dist_{i}"]
        new_cols[f"width_length_{i}"] = df_flat[f"width_{i}"] / (df_flat[f"length_{i}"] + 1e-6)

        if f"size_{i}" in df_flat:
            df_flat[f"size_{i}"] = np.log10(np.clip(df_flat[f"size_{i}"], 1e-6, None))
        if f"E_{i}" in df_flat:
            df_flat[f"E_{i}"] = np.log10(np.clip(df_flat[f"E_{i}"], 1e-6, None))
        if f"ES_{i}" in df_flat:
            df_flat[f"ES_{i}"] = np.log10(np.clip(df_flat[f"ES_{i}"], 1e-6, None))

        # pointing corrections
        if f"cen_x_{i}" in df_flat and f"fpointing_dx_{i}" in df_flat:
            df_flat[f"cen_x_{i}"] = df_flat[f"cen_x_{i}"] + df_flat[f"fpointing_dx_{i}"]
        if f"cen_y_{i}" in df_flat and f"fpointing_dy_{i}" in df_flat:
            df_flat[f"cen_y_{i}"] = df_flat[f"cen_y_{i}"] + df_flat[f"fpointing_dy_{i}"]
        df_flat = df_flat.drop(columns=[f"fpointing_dx_{i}", f"fpointing_dy_{i}"], errors="ignore")

    return pd.concat([df_flat, pd.DataFrame(new_cols, index=index)], axis=1)


def extra_columns(df, analysis_type, training):
    """Add extra columns required for analysis type."""
    if analysis_type == "stereo_analysis":
        return pd.DataFrame(
            {
                "Xoff_weighted_bdt": df["Xoff"].astype(np.float32),
                "Yoff_weighted_bdt": df["Yoff"].astype(np.float32),
                "Xoff_intersect": df["Xoff_intersect"].astype(np.float32),
                "Yoff_intersect": df["Yoff_intersect"].astype(np.float32),
                "Diff_Xoff": (df["Xoff"] - df["Xoff_intersect"]).astype(np.float32),
                "Diff_Yoff": (df["Yoff"] - df["Yoff_intersect"]).astype(np.float32),
                "Erec": np.log10(np.clip(df["Erec"], 1e-6, None)).astype(np.float32),
                "ErecS": np.log10(np.clip(df["ErecS"], 1e-6, None)).astype(np.float32),
                "EmissionHeight": df["EmissionHeight"].astype(np.float32),
            },
            index=df.index,
        )

    if "classification" in analysis_type:
        data = {
            "MSCW": df["MSCW"].astype(np.float32),
            "MSCL": df["MSCL"].astype(np.float32),
            "EChi2S": np.log10(np.clip(df["EChi2S"], 1e-6, None)).astype(np.float32),
            "EmissionHeight": df["EmissionHeight"].astype(np.float32),
            "EmissionHeightChi2": np.log10(np.clip(df["EmissionHeightChi2"], 1e-6, None)).astype(
                np.float32
            ),
        }
        if not training:
            data["ze_bin"] = df["ze_bin"].astype(np.float32)
        return pd.DataFrame(data, index=df.index)

    raise ValueError(f"Unknown analysis_type: {analysis_type}")


def zenith_in_bins(zenith_angles, bins):
    """Apply zenith binning based on zenith angles and given bin edges."""
    if isinstance(bins[0], dict):
        bins = [b["Ze_min"] for b in bins] + [bins[-1]["Ze_max"]]
    bins = np.asarray(bins, dtype=float)
    idx = np.clip(np.digitize(zenith_angles, bins) - 1, 0, len(bins) - 2)
    return idx.astype(np.int32)


def energy_in_bins(df_chunk, bins):
    """Apply energy binning based on reconstructed energy and given limits."""
    centers = np.array([(b["E_min"] + b["E_max"]) / 2 if b is not None else np.nan for b in bins])
    valid = (df_chunk["Erec"].to_numpy() > 0) & ~np.isnan(centers).all()
    e_bin = np.full(len(df_chunk), -1, dtype=np.int32)
    log_e = np.log10(df_chunk.loc[valid, "Erec"].to_numpy())
    distances = np.abs(log_e[:, None] - centers)
    distances[:, np.isnan(centers)] = np.inf

    e_bin[valid] = np.argmin(distances, axis=1)
    df_chunk["e_bin"] = e_bin
    return df_chunk["e_bin"]


def print_variable_statistics(df):
    """
    Print min, max, mean, and RMS for each variable in the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing variables loaded using branch_list.
    """
    for col in df.columns:
        data = df[col].dropna().to_numpy()
        if data.size == 0:
            print(f"{col}: No data")
            continue
        min_val = np.min(data)
        max_val = np.max(data)
        mean_val = np.mean(data)
        rms_val = np.sqrt(np.mean(np.square(data)))
        _logger.info(
            f"{col:25s} min: {min_val:10.4g}  max: {max_val:10.4g}  mean: {mean_val:10.4g}  rms: {rms_val:10.4g}"
        )
