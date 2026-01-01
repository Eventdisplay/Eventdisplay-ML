"""
Train XGBoost BDTs stereo reconstruction (direction, energy).

Uses x,y offsets calculated from intersection and dispBDT methods plus
image parameters to train multi-target regression BDTs to predict x,y offsets.

Uses energy related values to estimate event energy.

Separate BDTs are trained for 2, 3, and 4 telescope multiplicity events.
"""

import logging

import xgboost as xgb
from sklearn.model_selection import train_test_split

from eventdisplay_ml.config import configure_training
from eventdisplay_ml.data_processing import load_training_data
from eventdisplay_ml.evaluate import evaluate_regression_model
from eventdisplay_ml.models import save_models

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def train(df, model_configs):
    """
    Train a single XGBoost model for multi-target regression.

    Parameters
    ----------
    df : pd.DataFrame
        Training data.
    model_configs : dict
        Dictionary of model configurations.
    """
    n_tel = model_configs["n_tel"]
    if df.empty:
        _logger.warning(f"Skipping training for n_tel={n_tel} due to empty data.")
        return None

    x_cols = df.columns.difference(model_configs["targets"])
    _logger.info(f"Features ({len(x_cols)}): {x_cols}")
    model_configs["features"] = list(x_cols)
    x_data, y_data = df[x_cols], df[model_configs["targets"]]

    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        train_size=model_configs.get("train_test_fraction", 0.5),
        random_state=model_configs.get("random_state", None),
    )

    _logger.info(f"n_tel={n_tel}: Training events: {len(x_train)}, Testing events: {len(x_test)}")

    for name, cfg in model_configs.get("models", {}).items():
        _logger.info(f"Training {name} for n_tel={n_tel}...")
        model = xgb.XGBRegressor(**cfg.get("hyper_parameters", {}))
        model.fit(x_train, y_train)
        evaluate_regression_model(model, x_test, y_test, df, x_cols, y_data, name)
        cfg["model"] = model

    return model_configs


def main():
    """Run the training pipeline."""
    analysis_type = "stereo_analysis"

    model_configs = configure_training(analysis_type)

    df = load_training_data(model_configs, model_configs["input_file_list"], analysis_type)

    model_configs = train(df, model_configs)

    save_models(model_configs)

    _logger.info(f"XGBoost {analysis_type} model trained successfully.")


if __name__ == "__main__":
    main()
