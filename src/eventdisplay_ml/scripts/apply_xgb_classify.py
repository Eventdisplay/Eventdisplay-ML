"""
Apply XGBoost classification model.

Applies trained XGBoost classification models to input data and outputs
for each event the predicted signal probability.

Takes into account telescope multiplicity and training in energy bins.

"""

import argparse
import logging

from eventdisplay_ml.models import load_models, process_file_chunked

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def main():
    """Apply XGBoost classification."""
    parser = argparse.ArgumentParser(description=("Apply XGBoost Classification"))
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
        help=(
            "Path to directory containing XGBoost classification models "
            "(without n_tel and energy bin suffix)."
        ),
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

    _logger.info("--- XGBoost Classification Evaluation ---")
    _logger.info(f"Input file: {args.input_file}")
    _logger.info(f"Model prefix: {args.model_prefix}")
    _logger.info(f"Output file: {args.output_file}")
    _logger.info(f"Image selection: {args.image_selection}")

    models, model_par = load_models("classification", args.model_prefix)

    process_file_chunked(
        analysis_type="classification",
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
