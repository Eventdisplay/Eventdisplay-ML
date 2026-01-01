"""
Apply XGBoost BDTs stereo reconstruction (direction, energy).

Applies trained XGBoost models to predict Xoff, Yoff, and energy
for each event from an input mscw file. The output file contains
one row per input event, maintaining the original event order.
"""

import logging

from eventdisplay_ml.config import configure_apply
from eventdisplay_ml.models import process_file_chunked

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def main():
    """Apply XGBoost."""
    analysis_type = "stereo_analysis"

    model_configs = configure_apply(analysis_type)

    process_file_chunked(analysis_type, model_configs)


if __name__ == "__main__":
    main()
