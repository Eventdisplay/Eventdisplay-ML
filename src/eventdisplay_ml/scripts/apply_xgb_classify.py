"""
Apply XGBoost classification model.

Applies trained XGBoost classification models to input data and outputs
for each event the predicted signal probability.

Takes into account telescope multiplicity and training in energy bins.

"""

import argparse
import logging

import numpy as np
import uproot

from eventdisplay_ml.data_processing import (
    apply_image_selection,
    energy_in_bins,
    zenith_in_bins,
)
from eventdisplay_ml.features import features
from eventdisplay_ml.models import (
    apply_classification_models,
    load_classification_models,
)
from eventdisplay_ml.utils import parse_image_selection

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def process_file_chunked(
    input_file,
    output_file,
    models,
    model_parameters,
    image_selection,
    max_events=None,
    chunk_size=500000,
):
    """
    Stream events from an input file in chunks, apply XGBoost models, write events.

    Parameters
    ----------
    input_file : str
        Path to the input file containing a "data" TTree.
    output_file : str
        Path to the output file to create.
    models : dict
        Dictionary of loaded XGBoost models for classification.
    model_parameters : dict
        Model parameters defining energy and zenith angle bins.
    image_selection : str
        String specifying which telescope indices to select.
    max_events : int, optional
        Maximum number of events to process.
    chunk_size : int, optional
        Number of events to read and process per chunk.
    """
    branch_list = features("classification", training=False)
    selected_indices = parse_image_selection(image_selection)

    _logger.info(f"Chunk size: {chunk_size}")
    if max_events:
        _logger.info(f"Maximum events to process: {max_events}")

    bin_centers = np.array(
        [(b["E_min"] + b["E_max"]) / 2 for b in model_parameters["energy_bins_log10_tev"]]
    )

    with uproot.recreate(output_file) as root_file:
        tree = root_file.mktree("Classification", {"IsGamma": np.float32})
        total_processed = 0

        for df_chunk in uproot.iterate(
            f"{input_file}:data",
            branch_list,
            library="pd",
            step_size=chunk_size,
        ):
            if df_chunk.empty:
                continue

            df_chunk = apply_image_selection(
                df_chunk, selected_indices, analysis_type="classification"
            )
            if df_chunk.empty:
                continue

            if max_events is not None and total_processed >= max_events:
                break

            # energy bins (closest center)
            valid_energy_mask = df_chunk["Erec"].values > 0
            df_chunk["e_bin"] = -1
            log_e = np.log10(df_chunk.loc[valid_energy_mask, "Erec"].values)
            distances = np.abs(log_e[:, np.newaxis] - bin_centers)
            df_chunk.loc[valid_energy_mask, "e_bin"] = np.argmin(distances, axis=1)

            df_chunk["e_bin"] = energy_in_bins(df_chunk, model_parameters["energy_bins_log10_tev"])
            df_chunk["ze_bin"] = zenith_in_bins(
                90.0 - df_chunk["ArrayPointing_Elevation"].values,
                model_parameters["zenith_bins_deg"],
            )

            # Reset index to local chunk indices (0, 1, 2, ...) to avoid
            # index out-of-bounds when indexing chunk-sized output arrays
            df_chunk = df_chunk.reset_index(drop=True)

            pred_proba = apply_classification_models(df_chunk, models)

            tree.extend(
                {
                    "IsGamma": np.asarray(pred_proba, dtype=np.float32),
                }
            )

            total_processed += len(df_chunk)
            _logger.info(f"Processed {total_processed} events so far")

    _logger.info(f"Streaming complete. Total processed events written: {total_processed}")


def main():
    """Apply XGBoost classification."""
    parser = argparse.ArgumentParser(description=("Apply XGBoost Classification"))
    parser.add_argument(
        "--input-file",
        required=True,
        metavar="INPUT.root",
        help="Path to input mscw file",
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        metavar="MODEL_DIR",
        help="Directory containing XGBoost models",
    )
    parser.add_argument(
        "--model-parameters",
        type=str,
        help=("Path to model parameter file (JSON) defining which models to load. "),
    )
    parser.add_argument(
        "--output-file",
        required=True,
        metavar="OUTPUT.root",
        help="Output file path for predictions",
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

    _logger.info("--- XGBoost Classification Evaluation ---")
    _logger.info(f"Input file: {args.input_file}")
    _logger.info(f"Model directory: {args.model_dir}")
    _logger.info(f"Output file: {args.output_file}")
    _logger.info(f"Image selection: {args.image_selection}")

    models, model_par = load_classification_models(args.model_dir, args.model_parameters)

    process_file_chunked(
        input_file=args.input_file,
        output_file=args.output_file,
        models=models,
        model_parameters=model_par,
        image_selection=args.image_selection,
        max_events=args.max_events,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
