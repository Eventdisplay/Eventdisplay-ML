import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from pathlib import Path
from eventdisplay_ml.scripts.train_xgb_stereo import train


@pytest.fixture
def sample_df():
    """Create a sample DataFrame with training data."""
    np.random.seed(42)
    data = {
        "feature1": np.random.randn(100),
        "feature2": np.random.randn(100),
        "feature3": np.random.randn(100),
        "MCxoff": np.random.randn(100),
        "MCyoff": np.random.randn(100),
        "MCe0": np.random.randn(100),
    }
    return pd.DataFrame(data)

@pytest.fixture
def empty_df():
    """Create an empty DataFrame."""
    return pd.DataFrame()

@patch("eventdisplay_ml.scripts.train_xgb_stereo.dump")
@patch("eventdisplay_ml.scripts.train_xgb_stereo.evaluate_model")
@patch("eventdisplay_ml.scripts.train_xgb_stereo.MultiOutputRegressor")
def test_train_with_valid_data(mock_multi_output, mock_evaluate, mock_dump, sample_df, tmp_path):
    """Test train function with valid data."""
    mock_model = MagicMock()
    mock_multi_output.return_value = mock_model

    train(sample_df, n_tel=3, output_dir=tmp_path, train_test_fraction=0.8)

    assert mock_multi_output.called
    assert mock_model.fit.called
    assert mock_dump.called
    assert mock_evaluate.called

@patch("eventdisplay_ml.scripts.train_xgb_stereo.dump")
@patch("eventdisplay_ml.scripts.train_xgb_stereo.evaluate_model")
def test_train_with_empty_data(mock_evaluate, mock_dump, empty_df, caplog):
    """Test train function with empty DataFrame."""
    train(empty_df, n_tel=2, output_dir="/tmp", train_test_fraction=0.7)

    assert mock_dump.call_count == 0
    assert mock_evaluate.call_count == 0
    assert "Skipping training" in caplog.text

@patch("eventdisplay_ml.scripts.train_xgb_stereo.dump")
@patch("eventdisplay_ml.scripts.train_xgb_stereo.evaluate_model")
@patch("eventdisplay_ml.scripts.train_xgb_stereo.MultiOutputRegressor")
def test_train_output_filename(mock_multi_output, mock_evaluate, mock_dump, sample_df, tmp_path):
    """Test that output filename is correctly formatted."""
    mock_model = MagicMock()
    mock_multi_output.return_value = mock_model

    train(sample_df, n_tel=4, output_dir=tmp_path, train_test_fraction=0.8)

    # Verify dump was called with correct filename
    call_args = mock_dump.call_args
    output_path = call_args[0][1]
    assert "dispdir_bdt_ntel4_xgboost.joblib" in str(output_path)

@patch("eventdisplay_ml.scripts.train_xgb_stereo.dump")
@patch("eventdisplay_ml.scripts.train_xgb_stereo.evaluate_model")
@patch("eventdisplay_ml.scripts.train_xgb_stereo.MultiOutputRegressor")
def test_train_feature_selection(mock_multi_output, mock_evaluate, mock_dump, sample_df, tmp_path):
    """Test that features are correctly separated from targets."""
    mock_model = MagicMock()
    mock_multi_output.return_value = mock_model

    train(sample_df, n_tel=2, output_dir=tmp_path, train_test_fraction=0.8)

    # Verify fit was called with correct shapes
    fit_call = mock_model.fit.call_args
    x_train, y_train = fit_call[0]
    
    # Should have 3 features (feature1, feature2, feature3)
    assert x_train.shape[1] == 3
    # Should have 3 targets (MCxoff, MCyoff, MCe0)
    assert y_train.shape[1] == 3

@patch("eventdisplay_ml.scripts.train_xgb_stereo.dump")
@patch("eventdisplay_ml.scripts.train_xgb_stereo.evaluate_model")
@patch("eventdisplay_ml.scripts.train_xgb_stereo.MultiOutputRegressor")
def test_train_test_split_fraction(mock_multi_output, mock_evaluate, mock_dump, sample_df, tmp_path):
    """Test that train/test split respects the fraction parameter."""
    mock_model = MagicMock()
    mock_multi_output.return_value = mock_model

    train(sample_df, n_tel=2, output_dir=tmp_path, train_test_fraction=0.6)

    fit_call = mock_model.fit.call_args
    x_train, y_train = fit_call[0]
    
    # With 0.6 train fraction and 100 samples, expect ~60 training samples
    assert 50 <= len(x_train) <= 70