"""
Train XGBoost BDTs for gamma/hadron classification.

Uses image and stereo parameters to train classification BDTs to separate
gamma-ray events from hadronic background events.

Separate BDTs are trained for 2, 3, and 4 telescope multiplicity events.
"""

import logging

from eventdisplay_ml.config import configure_training
from eventdisplay_ml.data_processing import load_training_data
from eventdisplay_ml.models import save_models, train_classification

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def main():
    """Run the training pipeline."""
    analysis_type = "classification"

    model_configs = configure_training(analysis_type)

    df = [
        load_training_data(model_configs, file_list, analysis_type)
        for file_list in (
            model_configs["input_signal_file_list"],
            model_configs["input_background_file_list"],
        )
    ]

    model_configs = train_classification(df, model_configs)

    save_models(model_configs)

    _logger.info(f"XGBoost {analysis_type} model trained successfully.")


if __name__ == "__main__":
    main()
