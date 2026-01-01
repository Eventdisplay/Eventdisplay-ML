"""
Train XGBoost BDTs for gamma/hadron classification.

Uses image and stereo parameters to train classification BDTs to separate
gamma-ray events from hadronic background events.

Separate BDTs are trained for 2, 3, and 4 telescope multiplicity events.
"""

import logging

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from eventdisplay_ml.config import configure_training
from eventdisplay_ml.data_processing import load_training_data
from eventdisplay_ml.evaluate import (
    evaluate_classification_model,
    evaluation_efficiency,
)
from eventdisplay_ml.models import save_models

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def train(df, model_configs):
    """
    Train a single XGBoost model for gamma/hadron classification.

    Parameters
    ----------
    df : list of pd.DataFrame
        Training data.
    model_configs : dict
        Dictionary of model configurations.
    """
    n_tel = model_configs["n_tel"]
    if df[0].empty or df[1].empty:
        _logger.warning(f"Skipping training for n_tel={n_tel} due to empty data.")
        return None

    df[0]["label"] = 1
    df[1]["label"] = 0
    full_df = pd.concat([df[0], df[1]], ignore_index=True)
    x_data = full_df.drop(columns=["label"])
    _logger.info(f"Features ({len(x_data.columns)}): {', '.join(x_data.columns)}")
    model_configs["features"] = list(x_data.columns)
    y_data = full_df["label"]
    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        train_size=model_configs.get("train_test_fraction", 0.5),
        random_state=model_configs.get("random_state", None),
        stratify=y_data,
    )

    _logger.info(f"n_tel={n_tel}: Training events: {len(x_train)}, Testing events: {len(x_test)}")

    for name, cfg in model_configs.get("models", {}).items():
        _logger.info(f"Training {name} for n_tel={n_tel}...")
        model = xgb.XGBClassifier(**cfg.get("hyper_parameters", {}))
        model.fit(x_train, y_train)
        evaluate_classification_model(model, x_test, y_test, full_df, x_data.columns.tolist(), name)
        cfg["model"] = model
        cfg["efficiency"] = evaluation_efficiency(name, model, x_test, y_test)

    return model_configs


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

    model_configs = train(df, model_configs)

    save_models(model_configs)

    _logger.info(f"XGBoost {analysis_type} model trained successfully.")


if __name__ == "__main__":
    main()
