"""
Evaluate XGBoost BDTs for stereo reconstruction (direction, energy).

Applies trained XGBoost models to predict Xoff, Yoff, and energy
for each event from an input mscw file. The output ROOT file contains
one row per input event, maintaining the original event order.
"""

import argparse
import logging

import numpy as np
import uproot

from eventdisplay_ml.data_processing import apply_image_selection
from eventdisplay_ml.features import features
from eventdisplay_ml.models import apply_regression_models, load_regression_models
from eventdisplay_ml.utils import parse_image_selection

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def process_file_chunked(
    input_file,
    models,
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
    models : dict
        Dictionary of loaded XGBoost models for regression.
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
    branch_list = features("stereo_analysis", training=False)
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

            df_chunk = apply_image_selection(
                df_chunk, selected_indices, analysis_type="stereo_analysis"
            )
            if df_chunk.empty:
                continue

            if max_events is not None and total_processed >= max_events:
                break

            # Reset index to local chunk indices (0, 1, 2, ...) to avoid
            # index out-of-bounds when indexing chunk-sized output arrays
            df_chunk = df_chunk.reset_index(drop=True)

            pred_xoff, pred_yoff, pred_erec = apply_regression_models(df_chunk, models)

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
    """Apply XGBoost stereo models to input data."""
    parser = argparse.ArgumentParser(
        description=("Apply XGBoost Multi-Target BDTs for Stereo Reconstruction")
    )
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

    _logger.info("--- XGBoost Multi-Target Stereo Analysis Evaluation ---")
    _logger.info(f"Input file: {args.input_file}")
    _logger.info(f"Model directory: {args.model_dir}")
    _logger.info(f"Output file: {args.output_file}")
    _logger.info(f"Image selection: {args.image_selection}")

    process_file_chunked(
        input_file=args.input_file,
        models=load_regression_models(args.model_dir),
        output_file=args.output_file,
        image_selection=args.image_selection,
        max_events=args.max_events,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
