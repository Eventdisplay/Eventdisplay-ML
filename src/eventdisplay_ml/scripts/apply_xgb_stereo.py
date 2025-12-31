"""
Apply XGBoost BDTs stereo reconstruction (direction, energy).

Applies trained XGBoost models to predict Xoff, Yoff, and energy
for each event from an input mscw file. The output ROOT file contains
one row per input event, maintaining the original event order.
"""

import argparse
import logging

from eventdisplay_ml.models import load_models, process_file_chunked

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def main():
    """Apply XGBoost stereo models."""
    parser = argparse.ArgumentParser(description=("Apply XGBoost Stereo Reconstruction"))
    parser.add_argument(
        "--input_file",
        required=True,
        metavar="INPUT.root",
        help="Path to input mscw file",
    )
    parser.add_argument(
        "--model_prefix",
        required=True,
        metavar="MODEL_PREFIX",
        help=("Path to directory containing regression models  (without n_tel suffix)."),
    )
    parser.add_argument(
        "--model_name",
        default="xgboost",
        help="Model name to load (default: xgboost)",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        metavar="OUTPUT.root",
        help="Output file path for predictions",
    )
    parser.add_argument(
        "--image_selection",
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
        "--max_events",
        type=int,
        default=None,
        help="Maximum number of events to process (default: all events)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=500000,
        help="Number of events to process per chunk (default: 500000)",
    )
    args = parser.parse_args()

    _logger.info("--- XGBoost Stereo Analysis Evaluation ---")
    _logger.info(f"Input file: {args.input_file}")
    _logger.info(f"Model prefix: {args.model_prefix}")
    _logger.info(f"Output file: {args.output_file}")
    _logger.info(f"Image selection: {args.image_selection}")

    process_file_chunked(
        analysis_type="stereo_analysis",
        input_file=args.input_file,
        output_file=args.output_file,
        models=load_models("stereo_analysis", args.model_prefix),
        image_selection=args.image_selection,
        max_events=args.max_events,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
