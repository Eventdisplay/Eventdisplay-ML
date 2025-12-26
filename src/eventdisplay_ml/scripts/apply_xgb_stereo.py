"""
Evaluate XGBoost BDTs for stereo reconstruction (direction, energy).

Applies trained XGBoost models to predict Xoff, Yoff, and energy
for each event from an input mscw file. The output ROOT file contains
one row per input event, maintaining the original event order.
"""

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import uproot

from eventdisplay_ml.data_processing import flatten_data_vectorized
from eventdisplay_ml.training_variables import (
    xgb_all_regression_training_variables,
    xgb_per_telescope_training_variables,
)
from eventdisplay_ml.utils import parse_image_selection

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def apply_image_selection(df, selected_indices):
    """
    Filter and pad telescope lists for selected indices.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing telescope data.
    selected_indices : list[int] or None
        List of selected telescope indices. If None or all 4 telescopes
        are selected, the DataFrame is returned unchanged.

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

    pad_vars = [
        *xgb_per_telescope_training_variables(),
        "fpointing_dx",
        "fpointing_dy",
    ]
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


def load_models(model_dir):
    """
    Load XGBoost models for different telescope multiplicities from a directory.

    Parameters
    ----------
    model_dir : str
        Path to the directory containing the trained model files
        named ``dispdir_bdt_ntel{n_tel}_xgboost.joblib``.

    Returns
    -------
    dict[int, Any]
        A dictionary mapping the number of telescopes (n_tel) to the
        corresponding loaded model objects. Only models whose files
        exist in ``model_dir`` are included.
    """
    models = {}
    model_dir_path = Path(model_dir)
    for n_tel in range(2, 5):
        model_filename = model_dir_path / f"dispdir_bdt_ntel{n_tel}_xgboost.joblib"
        if model_filename.exists():
            _logger.info(f"Loading model: {model_filename}")
            models[n_tel] = joblib.load(model_filename)
        else:
            _logger.warning(f"Model not found: {model_filename}")
    return models


def apply_models(df, models_or_dir, selection_mask=None):
    """
    Apply trained XGBoost models to a DataFrame chunk.

    Parameters
    ----------
    df : pandas.DataFrame
        Chunk of events to process.
    models_or_dir : dict[int, Any] or str
        Either a preloaded models dictionary (as returned by :func:`load_models`)
        or a path to a model directory. If a string is provided, models are
        loaded on the fly to satisfy test expectations.
    selection_mask : pandas.Series or None
        Optional mask; False entries are marked with -999 in outputs.

    Returns
    -------
    pred_xoff : numpy.ndarray
        Array of predicted Xoff values for each event in the chunk, aligned
        with the index of ``df``.
    pred_yoff : numpy.ndarray
        Array of predicted Yoff values for each event in the chunk, aligned
        with the index of ``df``.
    pred_erec : numpy.ndarray
        Array of predicted Erec values for each event in the chunk, aligned
        with the index of ``df``.
    """
    n_events = len(df)
    pred_xoff = np.full(n_events, np.nan, dtype=np.float32)
    pred_yoff = np.full(n_events, np.nan, dtype=np.float32)
    pred_erec = np.full(n_events, np.nan, dtype=np.float32)
    if isinstance(models_or_dir, str):
        models = load_models(models_or_dir)
    else:
        models = models_or_dir

    grouped = df.groupby("DispNImages")

    for n_tel, group_df in grouped:
        n_tel = int(n_tel)
        if int(n_tel) < 2:
            continue
        if n_tel not in models:
            _logger.warning(
                f"No model available for n_tel={n_tel}, skipping {len(group_df)} events"
            )
            continue

        _logger.info(f"Processing {len(group_df)} events with n_tel={n_tel}")

        training_vars_with_pointing = [
            *xgb_per_telescope_training_variables(),
            "fpointing_dx",
            "fpointing_dy",
        ]
        df_flat = flatten_data_vectorized(
            group_df,
            n_tel,
            training_vars_with_pointing,
            apply_pointing_corrections=True,
            dtype=np.float32,
        )

        excluded_columns = ["MCxoff", "MCyoff", "MCe0"]
        for n in range(n_tel):
            excluded_columns.append(f"fpointing_dx_{n}")
            excluded_columns.append(f"fpointing_dy_{n}")

        feature_cols = [col for col in df_flat.columns if col not in excluded_columns]
        x_features = df_flat[feature_cols]

        model = models[n_tel]
        predictions = model.predict(x_features)

        for i, idx in enumerate(group_df.index):
            pred_xoff[idx] = predictions[i, 0]
            pred_yoff[idx] = predictions[i, 1]
            pred_erec[idx] = predictions[i, 2]

    if selection_mask is not None:
        pred_xoff = np.where(selection_mask, pred_xoff, -999.0)
        pred_yoff = np.where(selection_mask, pred_yoff, -999.0)
        pred_erec = np.where(selection_mask, pred_erec, -999.0)

    return pred_xoff, pred_yoff, pred_erec


def process_file_chunked(
    input_file,
    model_dir,
    output_file,
    image_selection,
    max_events=None,
    chunk_size=500000,
):
    """
    Stream events from an input ROOT file in chunks, apply XGBoost models, write events.

    Parameters
    ----------
    input_file : str
        Path to the input ROOT file containing a "data" TTree.
    model_dir : str
        Directory containing the trained XGBoost model files named
        ``dispdir_bdt_ntel{n_tel}_xgboost.joblib`` for different telescope
        multiplicities.
    output_file : str
        Path to the output ROOT file to create.
    image_selection : str
        String specifying which telescope indices to select, passed to
        :func:`parse_image_selection` to obtain the corresponding indices
        used by :func:`apply_image_selection`.
    max_events : int, optional
        Maximum number of events to process. If None (default), all
        available events in the input file are processed.
    chunk_size : int, optional
        Number of events to read and process per chunk. Larger values reduce
        I/O overhead but increase memory usage. Default is 500000.

    Returns
    -------
    None
        This function writes results directly to ``output_file`` and does not
        return a value.
    """
    models = load_models(model_dir)
    branch_list = [*xgb_all_regression_training_variables(), "fpointing_dx", "fpointing_dy"]
    selected_indices = parse_image_selection(image_selection)

    _logger.info(f"Chunk size: {chunk_size}")
    if max_events:
        _logger.info(f"Maximum events to process: {max_events}")

    with uproot.recreate(output_file) as root_file:
        tree = root_file.mktree(
            "StereoAnalysis",
            {"Dir_Xoff": np.float32, "Dir_Yoff": np.float32, "Dir_Erec": np.float32},
        )

        total_processed = 0

        for df_chunk in uproot.iterate(
            f"{input_file}:data",
            branch_list,
            library="pd",
            step_size=chunk_size,
        ):
            if df_chunk.empty:
                continue

            df_chunk = apply_image_selection(df_chunk, selected_indices)
            if df_chunk.empty:
                continue

            if max_events is not None and total_processed >= max_events:
                break

            # Reset index to local chunk indices (0, 1, 2, ...) to avoid
            # index out-of-bounds when indexing chunk-sized output arrays
            df_chunk = df_chunk.reset_index(drop=True)

            pred_xoff, pred_yoff, pred_erec = apply_models(df_chunk, models)

            tree.extend(
                {
                    "Dir_Xoff": np.asarray(pred_xoff, dtype=np.float32),
                    "Dir_Yoff": np.asarray(pred_yoff, dtype=np.float32),
                    "Dir_Erec": np.power(10.0, pred_erec, dtype=np.float32),
                }
            )

            total_processed += len(df_chunk)
            _logger.info(f"Processed {total_processed} events so far")

    _logger.info(f"Streaming complete. Total processed events written: {total_processed}")


def main():
    """Parse CLI arguments and run inference on an input ROOT file."""
    parser = argparse.ArgumentParser(
        description=("Apply XGBoost Multi-Target BDTs for Stereo Reconstruction")
    )
    parser.add_argument(
        "--input-file",
        required=True,
        metavar="INPUT.root",
        help="Path to input mscw ROOT file",
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        metavar="MODEL_DIR",
        help="Directory containing XGBoost models",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        metavar="OUTPUT.root",
        help="Output ROOT file path for predictions",
    )
    parser.add_argument(
        "--image-selection",
        type=str,
        default="15",
        help=(
            "Optional telescope selection. Can be bit-coded (e.g., 14 for telescopes 1,2,3) "
            "or comma-separated indices (e.g., '1,2,3'). "
            "Keeps events with all selected telescopes or 4-telescope events. "
            "Default is 15, which selects all 4 telescopes."
        ),
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Maximum number of events to process (default: all events)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500000,
        help="Number of events to process per chunk (default: 500000)",
    )
    args = parser.parse_args()

    _logger.info("--- XGBoost Multi-Target Stereo Analysis Evaluation ---")
    _logger.info(f"Input file: {args.input_file}")
    _logger.info(f"Model directory: {args.model_dir}")
    _logger.info(f"Output file: {args.output_file}")
    _logger.info(f"Image selection: {args.image_selection}")

    process_file_chunked(
        input_file=args.input_file,
        model_dir=args.model_dir,
        output_file=args.output_file,
        image_selection=args.image_selection,
        max_events=args.max_events,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
