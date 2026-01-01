"""
Apply XGBoost classification model.

Applies trained XGBoost classification models to input data and outputs
for each event. The output file contains the predicted signal probability.

"""

import logging

from eventdisplay_ml.config import configure_apply
from eventdisplay_ml.models import process_file_chunked

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def main():
    """Apply XGBoost."""
    analysis_type = "classification"

    model_configs = configure_apply(analysis_type)

    process_file_chunked(analysis_type, model_configs)


if __name__ == "__main__":
    main()
