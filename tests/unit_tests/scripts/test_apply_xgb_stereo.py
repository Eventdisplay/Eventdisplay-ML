"""Unit tests for apply_xgb_stereo script."""

from unittest.mock import Mock, patch

import joblib
import numpy as np
import pytest

from eventdisplay_ml.models import load_regression_models
from eventdisplay_ml.scripts.apply_xgb_stereo import (
    process_file_chunked,
)


class SimpleModel:
    """A simple picklable model for testing."""

    def __init__(self, predictions):
        self.predictions = predictions

    def predict(self, x):
        """Predict using the simple model."""
        n = len(x)
        return self.predictions[:n]


def test_process_file_chunked_creates_output(sample_df, tmp_path):
    """Test process_file_chunked creates output file."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_file = model_dir / "dispdir_bdt_ntel4_xgboost.joblib"
    joblib.dump(SimpleModel(np.array([[0.1, 0.2, 1.5]] * 4)), model_file)

    output_file = tmp_path / "output.root"

    models = load_regression_models(str(model_dir))

    with patch("eventdisplay_ml.scripts.apply_xgb_stereo.uproot.iterate") as mock_iterate:
        with patch("eventdisplay_ml.scripts.apply_xgb_stereo.uproot.recreate") as mock_recreate:
            mock_iterate.return_value = [sample_df.iloc[:1].copy()]
            mock_tree = Mock()
            mock_recreate.return_value.__enter__.return_value.mktree.return_value = mock_tree

            process_file_chunked(
                input_file="input.root",
                models=models,
                output_file=str(output_file),
                image_selection="15",
            )

            assert mock_tree.extend.called


@pytest.mark.parametrize(
    ("max_events", "expected_chunks"),
    [
        (None, 2),
        (2, 1),
    ],
)
def test_process_file_chunked_respects_limits(sample_df, tmp_path, max_events, expected_chunks):
    """Test process_file_chunked respects event limits."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    joblib.dump(
        SimpleModel(np.array([[0.1, 0.2, 1.5]] * 4)), model_dir / "dispdir_bdt_ntel4_xgboost.joblib"
    )

    models = load_regression_models(str(model_dir))

    with patch("eventdisplay_ml.scripts.apply_xgb_stereo.uproot.iterate") as mock_iterate:
        with patch("eventdisplay_ml.scripts.apply_xgb_stereo.uproot.recreate") as mock_recreate:
            mock_iterate.return_value = [sample_df.iloc[:2].copy(), sample_df.iloc[2:].copy()]
            mock_tree = Mock()
            mock_recreate.return_value.__enter__.return_value.mktree.return_value = mock_tree

            kwargs = {
                "input_file": "input.root",
                "models": models,
                "output_file": str(tmp_path / "output.root"),
                "image_selection": "15",
            }
            if max_events:
                kwargs["max_events"] = max_events

            process_file_chunked(**kwargs)
            assert mock_tree.extend.call_count == expected_chunks
