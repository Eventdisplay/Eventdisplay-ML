"""
Train XGBoost BDTs for stereo reconstruction (direction, energy).

Uses residuals relative to DispBDT predictions as training targets. The model learns
to correct the DispBDT baseline by predicting residuals:
  - Xoff_residual = MCxoff - Xoff_DispBDT
  - Yoff_residual = MCyoff - Yoff_DispBDT
  - E_residual = log10(MCe0) - log10(ErecS_DispBDT)

During inference, the predicted residuals are added back to the DispBDT baseline
to produce the final direction and energy estimates.

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
