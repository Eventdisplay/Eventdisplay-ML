"""
Train XGBoost BDTs for stereo reconstruction (direction, energy).

Uses x,y offsets calculated from intersection and dispBDT methods plus
image parameters to train multi-target regression BDTs to predict x,y offsets.

Uses energy related values to estimate event energy.

Trains a single BDT on all telescope multiplicity events.
"""

import logging

from eventdisplay_ml.config import configure_training
from eventdisplay_ml.data_processing import load_training_data
from eventdisplay_ml.models import save_models, train_regression

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def main():
    """Run the training pipeline."""
    analysis_type = "stereo_analysis"

    model_configs = configure_training(analysis_type)

    df = load_training_data(model_configs, model_configs["input_file_list"], analysis_type)

    model_configs = train_regression(df, model_configs)

    save_models(model_configs)

    _logger.info(f"XGBoost {analysis_type} model trained successfully.")


if __name__ == "__main__":
    main()
