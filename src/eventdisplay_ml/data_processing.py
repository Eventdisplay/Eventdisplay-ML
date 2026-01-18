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

# Default fill value for missing telescope-dependent data
DEFAULT_FILL_VALUE = np.nan


def read_telescope_config(root_file):
    """
    Read telescope configuration from ROOT file.

    Parameters
    ----------
    root_file : uproot file handle
        Open ROOT file containing the telconfig tree.

    Returns
    -------
    dict
        Dictionary with telescope configuration:
        - 'n_tel': Total number of telescopes
        - 'tel_ids': Array of telescope IDs
        - 'mirror_areas': Array of mirror areas for each telescope
        - 'tel_x': Array of telescope X positions
        - 'tel_y': Array of telescope Y positions
        - 'max_tel_id': Maximum telescope ID
        - 'tel_types': Dictionary mapping mirror area to list of telescope IDs
    """
    if "telconfig" not in root_file:
        _logger.warning("No telconfig tree found in ROOT file, using default 4 telescopes")
        return {
            "n_tel": 4,
            "tel_ids": np.array([0, 1, 2, 3]),
            "mirror_areas": np.array([1.0, 1.0, 1.0, 1.0]),
            "tel_x": np.array([0.0, 0.0, 0.0, 0.0]),
            "tel_y": np.array([0.0, 0.0, 0.0, 0.0]),
            "max_tel_id": 3,
            "tel_types": {1.0: [0, 1, 2, 3]},
        }

    telconfig_tree = root_file["telconfig"]
    telconfig_data = telconfig_tree.arrays(
        ["NTel", "TelID", "MirrorArea", "TelX", "TelY"], library="np"
    )

    n_tel = int(telconfig_data["NTel"][0])
    tel_ids = telconfig_data["TelID"]
    mirror_areas = telconfig_data["MirrorArea"]
    tel_x = telconfig_data["TelX"]
    tel_y = telconfig_data["TelY"]
    max_tel_id = int(np.max(tel_ids))

    # Group telescopes by mirror area (telescope type)
    tel_types = {}
    for tel_id, mirror_area in zip(tel_ids, mirror_areas):
        if mirror_area not in tel_types:
            tel_types[mirror_area] = []
        tel_types[mirror_area].append(int(tel_id))

    _logger.info(f"Telescope configuration: {n_tel} telescopes, max TelID: {max_tel_id}")
    _logger.info(f"Telescope types by mirror area: {tel_types}")

    return {
        "n_tel": n_tel,
        "tel_ids": tel_ids,
        "mirror_areas": mirror_areas,
        "tel_x": tel_x,
        "tel_y": tel_y,
        "max_tel_id": max_tel_id,
        "tel_types": tel_types,
    }


def _resolve_branch_aliases(tree, branch_list):
    """Resolve branch name aliases (e.g. R_core vs R) and drop missing optional branches."""
    resolved = list(branch_list)
    rename_map = {}
    missing_optional = set()

    has_r_core = "R_core" in tree.keys()
    has_r_fallback = "R" in tree.keys()

    if "R_core" in resolved and not has_r_core and has_r_fallback:
        resolved = ["R" if b == "R_core" else b for b in resolved]
        rename_map["R"] = "R_core"
        _logger.info("Branch 'R_core' not found; using 'R' as fallback")
    elif "R_core" in resolved and not has_r_core and not has_r_fallback:
        _logger.warning("Branches 'R_core' and fallback 'R' not found; column will be missing")

    # Remove mirror_areas - it's synthesized from telconfig, not read from file
    if "mirror_areas" in resolved:
        resolved = [b for b in resolved if b != "mirror_areas"]
        _logger.info("Branch 'mirror_areas' will be synthesized from telescope configuration")

    # Remove telescope relative coordinates - they're computed from tel_config and data columns
    for synth_feat in ("tel_rel_x", "tel_rel_y", "tel_shower_x", "tel_shower_y"):
        if synth_feat in resolved:
            resolved = [b for b in resolved if b != synth_feat]
            _logger.info(
                f"Branch '{synth_feat}' will be computed from telescope config and Xcore/Ycore/pointing"
            )

    for opt in ("fpointing_dx", "fpointing_dy", "E", "Erec", "ErecS"):
        if opt in resolved and opt not in tree.keys():
            resolved = [b for b in resolved if b != opt]
            missing_optional.add(opt)
            _logger.info(f"Branch '{opt}' not found; will fill with defaults")

    return resolved, rename_map, missing_optional


def _ensure_fpointing_columns(df):
    """Ensure fpointing_dx and fpointing_dy exist; fill zeros if missing."""

    def _zero_like_tel_list(tel_list):
        length = len(tel_list) if hasattr(tel_list, "__len__") else 0
        return np.zeros(length, dtype=np.float32)

    if "fpointing_dx" not in df:
        df["fpointing_dx"] = df["DispTelList_T"].apply(_zero_like_tel_list)
    if "fpointing_dy" not in df:
        df["fpointing_dy"] = df["DispTelList_T"].apply(_zero_like_tel_list)


def _ensure_optional_scalar_columns(df, missing_optional):
    """Fill optional scalar columns like Erec/ErecS with defaults when missing."""
    if "Erec" in missing_optional and "Erec" not in df:
        df["Erec"] = np.full(len(df), DEFAULT_FILL_VALUE, dtype=np.float32)
    if "ErecS" in missing_optional and "ErecS" not in df:
        df["ErecS"] = np.full(len(df), DEFAULT_FILL_VALUE, dtype=np.float32)


def _calculate_shower_coordinates(tel_x, tel_y, shower_x, shower_y, elevation_deg, azimuth_deg):
    """
    Calculate relative telescope positions in shower coordinate system.

    Parameters
    ----------
    tel_x, tel_y : float
        Telescope position in ground coordinates (meters).
    shower_x, shower_y : float
        Shower core position in ground coordinates (meters).
    elevation_deg : float
        Pointing elevation in degrees.
    azimuth_deg : float
        Pointing azimuth in degrees.

    Returns
    -------
    tuple of float
        (rel_x, rel_y, shower_x_coord, shower_y_coord) where:
        - rel_x, rel_y: relative position in ground frame (tel - core)
        - shower_x_coord, shower_y_coord: relative position in shower coordinates
    """
    # Convert to radians
    elev = np.radians(elevation_deg)
    azim = np.radians(azimuth_deg)

    # Relative position in ground frame
    rel_x = tel_x - shower_x
    rel_y = tel_y - shower_y

    # Shower coordinate system:
    # - shower_y_axis points in shower direction (towards zenith direction of pointing)
    # - shower_x_axis is perpendicular (east-west relative to shower direction)
    # The shower direction in ground coords: (sin(elev)*cos(azim), sin(elev)*sin(azim))
    # But we use elevation as zenith angle, so pointing vector is rotated

    # Rotation from ground (east-north) to shower (along-across) coordinates
    # Along shower: cos(azim)*cos(elev)*dx + sin(azim)*cos(elev)*dy
    # Across shower (perpendicular): -sin(azim)*dx + cos(azim)*dy

    # Elevation affects how much of the shower is in the horizontal plane
    cos_elev = np.cos(elev)

    # Along shower direction (scaled by cos(elev) - shower at zenith has no horizontal extent)
    shower_y_coord = np.cos(azim) * cos_elev * rel_x + np.sin(azim) * cos_elev * rel_y
    # Across shower direction (perpendicular component in horizontal plane)
    shower_x_coord = -np.sin(azim) * rel_x + np.cos(azim) * rel_y

    return rel_x, rel_y, shower_x_coord, shower_y_coord


def flatten_telescope_data_vectorized(
    df, n_tel, features, analysis_type, training=True, tel_config=None
):
    """
    Vectorized flattening of telescope array columns.

    Converts per-telescope arrays into individual feature columns indexed by actual
    telescope ID. Features are mapped to telescope indices directly, with NaN fill
    values for missing telescopes.

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
    tel_config : dict, optional
        Telescope configuration dictionary with 'max_tel_id' and 'tel_types'.

    Returns
    -------
    pandas.DataFrame
        Flattened DataFrame with per-telescope columns suffixed by ``_{i}``
        for telescope index ``i``, plus derived features, and array features.
        Missing telescopes are filled with NaN.
    """
    flat_features = {}
    tel_list_matrix = _to_dense_array(df["DispTelList_T"])
    n_evt = len(df)
    default_value = DEFAULT_FILL_VALUE

    # Determine max telescope ID from config or use default
    max_tel_id = tel_config["max_tel_id"] if tel_config else (n_tel - 1)

    for var in features:
        # Special handling for mirror_areas: lookup from tel_config instead of reading from df
        if var == "mirror_areas" and tel_config:
            # Create a mapping from telescope ID to mirror area
            tel_id_to_mirror = dict(zip(tel_config["tel_ids"], tel_config["mirror_areas"]))
            for tel_idx in range(max_tel_id + 1):
                col_name = f"{var}_{tel_idx}"
                if tel_idx in tel_id_to_mirror:
                    mirror_val = float(tel_id_to_mirror[tel_idx])
                    # Convert zero to 100
                    if mirror_val == 0.0:
                        mirror_val = 100.0
                    flat_features[col_name] = np.full(n_evt, mirror_val, dtype=np.float32)
                else:
                    flat_features[col_name] = np.full(n_evt, default_value, dtype=np.float32)
            continue

        # Special handling for telescope relative coordinates from tel_config
        if var in ("tel_rel_x", "tel_rel_y", "tel_shower_x", "tel_shower_y") and tel_config:
            if (
                "Xcore" in df.columns
                and "Ycore" in df.columns
                and "ArrayPointing_Elevation" in df.columns
                and "ArrayPointing_Azimuth" in df.columns
            ):
                _logger.info(f"Computing synthetic feature: {var}")

                # Extract event data once as numpy arrays (vectorized)
                core_x = df["Xcore"].to_numpy(dtype=np.float64)
                core_y = df["Ycore"].to_numpy(dtype=np.float64)
                elev = df["ArrayPointing_Elevation"].to_numpy(dtype=np.float64)
                azim = df["ArrayPointing_Azimuth"].to_numpy(dtype=np.float64)

                # Mark invalid core positions
                valid_mask = (core_x >= 0) & (core_y >= 0)

                # Create mappings from telescope ID to position
                tel_id_to_x = dict(zip(tel_config["tel_ids"], tel_config["tel_x"]))
                tel_id_to_y = dict(zip(tel_config["tel_ids"], tel_config["tel_y"]))

                # Vectorized coordinate transformation for all telescopes
                for tel_idx in range(max_tel_id + 1):
                    col_name = f"{var}_{tel_idx}"
                    if tel_idx in tel_id_to_x:
                        tel_x = float(tel_id_to_x[tel_idx])
                        tel_y = float(tel_id_to_y[tel_idx])

                        # Vectorized calculation for all events at once
                        rel_x = tel_x - core_x
                        rel_y = tel_y - core_y

                        # Convert angles to radians (vectorized)
                        elev_rad = np.radians(elev)
                        azim_rad = np.radians(azim)
                        cos_elev = np.cos(elev_rad)

                        # Vectorized shower coordinate transformation
                        # Along shower direction (scaled by cos(elev) - shower at zenith has no horizontal extent)
                        shower_y_coord = (
                            np.cos(azim_rad) * cos_elev * rel_x
                            + np.sin(azim_rad) * cos_elev * rel_y
                        )
                        # Across shower direction (perpendicular component in horizontal plane)
                        shower_x_coord = -np.sin(azim_rad) * rel_x + np.cos(azim_rad) * rel_y

                        # Select output based on variable type
                        if var == "tel_rel_x":
                            result = rel_x
                        elif var == "tel_rel_y":
                            result = rel_y
                        elif var == "tel_shower_x":
                            result = shower_x_coord
                        else:  # tel_shower_y
                            result = shower_y_coord

                        # Apply validity mask: set invalid core positions to DEFAULT_FILL_VALUE
                        result = np.where(valid_mask, result, default_value)
                        flat_features[col_name] = result.astype(np.float32)
                    else:
                        flat_features[col_name] = np.full(n_evt, default_value, dtype=np.float32)
                continue
            # Missing required columns, skip
            _logger.warning(f"Cannot compute {var}: missing Xcore/Ycore or pointing direction")
            for tel_idx in range(max_tel_id + 1):
                col_name = f"{var}_{tel_idx}"
                flat_features[col_name] = np.full(n_evt, default_value, dtype=np.float32)
            continue

        data = _to_dense_array(df[var]) if var in df else np.full((n_evt, n_tel), np.nan)

        for tel_idx in range(max_tel_id + 1):  # Iterate over all possible telescope indices
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

    df_flat = flatten_telescope_variables(n_tel, flat_features, df.index, tel_config)
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


def flatten_feature_data(group_df, ntel, analysis_type, training, tel_config=None):
    """
    Get flattened features for events.

    All events are flattened with features indexed by actual telescope ID.

    Parameters
    ----------
    group_df : pandas.DataFrame
        DataFrame with events to flatten.
    ntel : int
        Maximum telescope index.
    analysis_type : str
        Type of analysis.
    training : bool
        Whether in training mode.
    tel_config : dict, optional
        Telescope configuration dictionary.
    """
    df_flat = flatten_telescope_data_vectorized(
        group_df,
        ntel,
        features.telescope_features(analysis_type),
        analysis_type=analysis_type,
        training=training,
        tel_config=tel_config,
    )
    max_tel_id = tel_config["max_tel_id"] if tel_config else ntel - 1
    excluded_columns = set(features.target_features(analysis_type)) | set(
        features.excluded_features(analysis_type, max_tel_id + 1)
    )
    return df_flat.drop(columns=excluded_columns, errors="ignore")


def load_training_data(model_configs, file_list, analysis_type):
    """
    Load and flatten training data from the mscw file.

    Processes all events regardless of telescope multiplicity. Features are created
    for all telescopes with default value NaN for missing telescopes.
    Reads telescope configuration from the ROOT file to determine the number
    and types of telescopes.

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

    tel_config = None  # Will be read from first file
    dfs = []
    for f in input_files:
        try:
            with uproot.open(f) as root_file:
                if "data" not in root_file:
                    _logger.warning(f"File: {f} does not contain a 'data' tree.")
                    continue

                # Read telescope configuration from first file
                if tel_config is None:
                    tel_config = read_telescope_config(root_file)
                    model_configs["tel_config"] = tel_config  # Store for later use

                _logger.info(f"Processing file: {f}")
                tree = root_file["data"]
                resolved_branch_list, rename_map, missing_optional = _resolve_branch_aliases(
                    tree, branch_list
                )
                df_file = tree.arrays(
                    resolved_branch_list, cut=model_configs.get("pre_cuts", None), library="pd"
                )
                if rename_map:
                    df_file.rename(columns=rename_map, inplace=True)
                _ensure_optional_scalar_columns(df_file, missing_optional)
                _ensure_fpointing_columns(df_file)
                if df_file.empty:
                    continue

                if max_events_per_file and len(df_file) > max_events_per_file:
                    df_file = df_file.sample(n=max_events_per_file, random_state=random_state)

                n_before = tree.num_entries
                _logger.info(
                    f"Number of events before / after event cut: {n_before} / {len(df_file)}"
                    f" (fraction retained: {len(df_file) / n_before:.4f})"
                )

                # Flatten using telescope configuration
                df_flat = flatten_telescope_data_vectorized(
                    df_file,
                    tel_config["max_tel_id"] + 1,
                    features.telescope_features(analysis_type),
                    analysis_type,
                    training=True,
                    tel_config=tel_config,
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
    Apply clip intervals to matching columns.

    Modifies the dataframe in place. Handles NaN default values for missing telescopes
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
            for i in range(n_tel):
                col_name = f"{var_base}_{i}"
                if col_name in df.columns:
                    mask_valid = df[col_name].notna()
                    df.loc[mask_valid, col_name] = df.loc[mask_valid, col_name].clip(vmin, vmax)
                    if var_base in apply_log10:
                        mask_to_log = mask_valid & (df[col_name] > 0)
                        df.loc[mask_to_log, col_name] = np.log10(df.loc[mask_to_log, col_name])
        else:
            if var_base in df.columns:
                mask_valid = df[var_base].notna()
                df.loc[mask_valid, var_base] = df.loc[mask_valid, var_base].clip(vmin, vmax)
                if var_base in apply_log10:
                    mask_to_log = mask_valid & (df[var_base] > 0)
                    df.loc[mask_to_log, var_base] = np.log10(df.loc[mask_to_log, var_base])


def flatten_telescope_variables(n_tel, flat_features, index, tel_config=None):
    """Generate dataframe for telescope variables flattened for all telescopes.

    Creates features for all telescope IDs, using NaN as default value for missing data.

    Parameters
    ----------
    n_tel : int
        Maximum telescope index (for backward compatibility).
    flat_features : dict
        Dictionary of flattened feature arrays.
    index : pandas.Index
        DataFrame index.
    tel_config : dict, optional
        Telescope configuration with 'max_tel_id' key.
    """
    df_flat = pd.DataFrame(flat_features, index=index)
    df_flat = df_flat.astype(np.float32)

    # Determine max telescope ID from config or use n_tel
    max_tel_id = tel_config["max_tel_id"] if tel_config else (n_tel - 1)

    new_cols = {}
    for i in range(max_tel_id + 1):  # Iterate over all possible telescopes
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

    apply_clip_intervals(
        df_flat, n_tel=max_tel_id + 1, apply_log10=["size", "E", "ES", "size_dist2"]
    )

    for i in range(max_tel_id + 1):  # Iterate over all possible telescope indices
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
