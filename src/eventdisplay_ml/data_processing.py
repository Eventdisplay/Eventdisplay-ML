"""
Data processing for XGBoost analysis.

Provides common functions for flattening and preprocessing telescope array data.
"""

import logging

import numpy as np
import pandas as pd
import uproot

from eventdisplay_ml import features, utils
from eventdisplay_ml.geomag import calculate_geomagnetic_angles

_logger = logging.getLogger(__name__)


def flatten_telescope_data_vectorized(df, n_tel, features, analysis_type, training=True):
    """
    Vectorized flattening of telescope array columns.

    Converts per-telescope arrays into individual feature columns indexed by actual
    telescope ID. Features are mapped to telescope indices directly, so size_2 always
    corresponds to telescope 2, with -99 fill values for missing telescopes.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing telescope data.
    n_tel : int
        Number of telescopes to flatten for (maximum telescope index).
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
        for telescope index ``i`` (0-3), plus derived features, and array features.
        Missing telescopes are filled with -99.
    """
    flat_features = {}
    tel_list_matrix = _to_dense_array(df["DispTelList_T"])
    n_evt = len(df)
    default_value = -99.0

    for var in features:
        data = _to_dense_array(df[var]) if var in df else np.full((n_evt, n_tel), np.nan)

        for tel_idx in range(4):  # Always iterate over all 4 possible telescope indices
            col_name = f"{var}_{tel_idx}"

            if var.startswith("Disp"):
                # Case 1: Simple index - look up data by direct index
                if tel_idx < data.shape[1]:
                    flat_features[col_name] = data[:, tel_idx]
                else:
                    flat_features[col_name] = np.full(len(df), default_value)
            else:
                # Case 2: Index lookup via DispTelList_T
                # For each event, find if tel_idx is in its DispTelList_T
                # If yes, extract data at that position in the data array
                result = np.full(len(df), default_value, dtype=np.float32)

                for evt_idx in range(n_evt):
                    tel_list = tel_list_matrix[evt_idx]
                    if tel_idx in tel_list:
                        # Find position of tel_idx in tel_list
                        pos_in_list = np.where(tel_list == tel_idx)[0]
                        if len(pos_in_list) > 0 and pos_in_list[0] < data.shape[1]:
                            result[evt_idx] = data[evt_idx, pos_in_list[0]]

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
    """
    Get flattened features for events.

    All events are flattened to 4 telescopes with features indexed by actual
    telescope ID. The ntel parameter is kept for compatibility.
    """
    df_flat = flatten_telescope_data_vectorized(
        group_df,
        ntel,
        features.telescope_features(analysis_type),
        analysis_type=analysis_type,
        training=training,
    )
    excluded_columns = set(features.target_features(analysis_type)) | set(
        features.excluded_features(analysis_type, 4)  # Always use 4 for all telescope exclusions
    )
    return df_flat.drop(columns=excluded_columns, errors="ignore")


def load_training_data(model_configs, file_list, analysis_type):
    """
    Load and flatten training data from the mscw file.

    Processes all events regardless of telescope multiplicity. Features are created
    for all 4 telescopes (0-3) with default value -99 for missing telescopes.

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
    random_state = model_configs.get("random_state", None)

    _logger.info(f"--- Loading and Flattening Data for {analysis_type} ---")
    _logger.info("Processing all events regardless of multiplicity")
    _logger.info(
        "Max events to process: "
        f"{max_events if max_events is not None and max_events > 0 else 'All available'}"
    )
    if analysis_type == "classification":
        _logger.info(f"Adding zenith binning: {model_configs.get('zenith_bins_deg', [])}")

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
                df_file = tree.arrays(
                    branch_list, cut=model_configs.get("pre_cuts", None), library="pd"
                )
                if df_file.empty:
                    continue

                if max_events_per_file and len(df_file) > max_events_per_file:
                    df_file = df_file.sample(n=max_events_per_file, random_state=random_state)

                n_before = tree.num_entries
                _logger.info(
                    f"Number of events before / after event cut: {n_before} / {len(df_file)}"
                    f" (fraction retained: {len(df_file) / n_before:.4f})"
                )

                # Always flatten to 4 telescopes
                df_flat = flatten_telescope_data_vectorized(
                    df_file,
                    4,  # Always use 4 for maximum telescope coverage
                    features.telescope_features(analysis_type),
                    analysis_type,
                    training=True,
                )
                if analysis_type == "stereo_analysis":
                    df_flat["MCxoff"] = df_file["MCxoff"]
                    df_flat["MCyoff"] = df_file["MCyoff"]
                    df_flat["MCe0"] = np.log10(df_file["MCe0"])
                elif analysis_type == "classification":
                    df_flat["ze_bin"] = zenith_in_bins(
                        90.0 - df_file["ArrayPointing_Elevation"],
                        model_configs.get("zenith_bins_deg", []),
                    )

                dfs.append(df_flat)

                del df_file
        except Exception as e:
            raise FileNotFoundError(f"Error opening or reading file {f}: {e}") from e

    df_final = pd.concat(dfs, ignore_index=True)
    df_final.dropna(axis=1, how="all", inplace=True)
    _logger.info(f"Total events loaded: {len(df_final)}")

    # Log multiplicity distribution
    if "DispNImages" in df_final.columns:
        mult_counts = df_final["DispNImages"].value_counts().sort_index()
        for mult, count in mult_counts.items():
            _logger.info(f"\tDispNImages={int(mult)}: {count} events")

    if analysis_type == "classification":
        counts = df_final["ze_bin"].value_counts().sort_index()
        for zb, n in counts.items():
            _logger.info(f"\tze_bin={zb}: {n} events")

    if len(df_final) == 0:
        raise ValueError("No data loaded from input files.")

    print_variable_statistics(df_final)

    return df_final


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


def apply_clip_intervals(df, n_tel=None, apply_log10=None):
    """
    Apply clip intervals to matching columns in dataframe.

    Modifies the dataframe in place. Handles -99 default values for missing telescopes
    by preserving them throughout clipping and log10 transformation.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to apply clipping to (modified in place).
    n_tel : int, optional
        Number of telescopes. If provided, applies to per-telescope columns (var_0, var_1, etc.).
    apply_log10 : list, optional
        List of variable base names to apply log10 transformation after clipping.
    """
    if apply_log10 is None:
        apply_log10 = []

    clip_intervals = features.clip_intervals()

    for var_base, (vmin, vmax) in clip_intervals.items():
        if n_tel is not None:
            # Apply to per-telescope columns
            for i in range(n_tel):
                col_name = f"{var_base}_{i}"
                if col_name in df.columns:
                    # Preserve -99 default values
                    mask_valid = df[col_name] != -99.0

                    # Clip only valid values
                    df.loc[mask_valid, col_name] = df.loc[mask_valid, col_name].clip(vmin, vmax)

                    # Apply log10 to valid values only
                    if var_base in apply_log10:
                        mask_to_log = mask_valid & (df[col_name] > 0)
                        df.loc[mask_to_log, col_name] = np.log10(df.loc[mask_to_log, col_name])
        else:
            # Apply to non-telescope columns
            if var_base in df.columns:
                # Preserve -99 default values
                mask_valid = df[var_base] != -99.0

                # Clip only valid values
                df.loc[mask_valid, var_base] = df.loc[mask_valid, var_base].clip(vmin, vmax)

                # Apply log10 to valid values only
                if var_base in apply_log10:
                    mask_to_log = mask_valid & (df[var_base] > 0)
                    df.loc[mask_to_log, var_base] = np.log10(df.loc[mask_to_log, var_base])


def flatten_telescope_variables(n_tel, flat_features, index):
    """Generate dataframe for telescope variables flattened for all 4 telescopes.

    Always creates features for telescopes 0-3, using -99 as default value for missing data.
    """
    df_flat = pd.DataFrame(flat_features, index=index)
    df_flat = df_flat.astype(np.float32)

    new_cols = {}
    for i in range(4):  # Always iterate over all 4 telescopes
        if f"Disp_T_{i}" in df_flat:
            new_cols[f"disp_x_{i}"] = df_flat[f"Disp_T_{i}"] * df_flat[f"cosphi_{i}"]
            new_cols[f"disp_y_{i}"] = df_flat[f"Disp_T_{i}"] * df_flat[f"sinphi_{i}"]
        if f"loss_{i}" in df_flat and f"dist_{i}" in df_flat:
            new_cols[f"loss_loss_{i}"] = df_flat[f"loss_{i}"] ** 2
            new_cols[f"loss_dist_{i}"] = df_flat[f"loss_{i}"] * df_flat[f"dist_{i}"]
        if f"width_{i}" in df_flat and f"length_{i}" in df_flat:
            new_cols[f"size_dist2_{i}"] = df_flat[f"width_{i}"] / (df_flat[f"length_{i}"] + 1e-6)
            new_cols[f"width_length_{i}"] = df_flat[f"width_{i}"] / (df_flat[f"length_{i}"] + 1e-6)

    df_flat = pd.concat([df_flat, pd.DataFrame(new_cols, index=index)], axis=1)

    apply_clip_intervals(df_flat, n_tel=4, apply_log10=["size", "E", "ES", "size_dist2"])

    for i in range(4):  # Always iterate over all 4 telescopes
        if f"cen_x_{i}" in df_flat and f"fpointing_dx_{i}" in df_flat:
            df_flat[f"cen_x_{i}"] = df_flat[f"cen_x_{i}"] + df_flat[f"fpointing_dx_{i}"]
        if f"cen_y_{i}" in df_flat and f"fpointing_dy_{i}" in df_flat:
            df_flat[f"cen_y_{i}"] = df_flat[f"cen_y_{i}"] + df_flat[f"fpointing_dy_{i}"]
        df_flat = df_flat.drop(columns=[f"fpointing_dx_{i}", f"fpointing_dy_{i}"], errors="ignore")

    return df_flat


def extra_columns(df, analysis_type, training):
    """Add extra columns required for analysis type."""
    if analysis_type == "stereo_analysis":
        data = {
            "Xoff_weighted_bdt": df["Xoff"].astype(np.float32),
            "Yoff_weighted_bdt": df["Yoff"].astype(np.float32),
            "Xoff_intersect": df["Xoff_intersect"].astype(np.float32),
            "Yoff_intersect": df["Yoff_intersect"].astype(np.float32),
            "Diff_Xoff": (df["Xoff"] - df["Xoff_intersect"]).astype(np.float32),
            "Diff_Yoff": (df["Yoff"] - df["Yoff_intersect"]).astype(np.float32),
            "Erec": df["Erec"].astype(np.float32),
            "ErecS": df["ErecS"].astype(np.float32),
            "EmissionHeight": df["EmissionHeight"].astype(np.float32),
            "Geomagnetic_Angle": calculate_geomagnetic_angles(
                df["ArrayPointing_Azimuth"], df["ArrayPointing_Elevation"]
            ),
        }
    elif "classification" in analysis_type:
        data = {
            "MSCW": df["MSCW"].astype(np.float32),
            "MSCL": df["MSCL"].astype(np.float32),
            "EChi2S": df["EChi2S"].astype(np.float32),
            "EmissionHeight": df["EmissionHeight"].astype(np.float32),
            "EmissionHeightChi2": df["EmissionHeightChi2"].astype(np.float32),
        }
        if not training:
            data["ze_bin"] = df["ze_bin"].astype(np.float32)

    df_extra = pd.DataFrame(data, index=df.index)
    apply_clip_intervals(
        df_extra,
        apply_log10=[
            "EChi2S",
            "EmissionHeightChi2",
            "Erec",
            "ErecS",
        ],
    )
    return df_extra


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
