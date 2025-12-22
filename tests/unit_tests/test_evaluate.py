import logging
import numpy as np
import pandas as pd

from unittest.mock import MagicMock
from eventdisplay_ml.evaluate import shap_feature_importance


def test_shap_feature_importance_basic(caplog):
    """Test shap_feature_importance logs SHAP importance values."""
    
    caplog.set_level(logging.INFO)
    
    # Mock model and estimators
    mock_booster = MagicMock()
    mock_est = MagicMock()
    mock_est.get_booster.return_value = mock_booster
    
    mock_model = MagicMock()
    mock_model.estimators_ = [mock_est]
    
    # Create sample data
    x_sample_data = pd.DataFrame({
        'feature_1': np.random.rand(100),
        'feature_2': np.random.rand(100),
        'feature_3': np.random.rand(100),
    })
    
    # Mock SHAP values (100 samples, 4 features + 1 bias)
    shap_values = np.random.rand(100, 4)
    mock_booster.predict.return_value = np.hstack([shap_values, np.random.rand(100, 1)])
    
    target_names = ['target_1']
    
    shap_feature_importance(mock_model, x_sample_data, target_names, max_points=100, n_top=2)
    
    assert "Builtin XGBoost SHAP Importance for target_1" in caplog.text
    assert "feature_" in caplog.text


def test_shap_feature_importance_multiple_targets(caplog):
    """Test shap_feature_importance with multiple targets."""
    
    caplog.set_level(logging.INFO)
    
    # Mock model with multiple estimators
    mock_booster_1 = MagicMock()
    mock_est_1 = MagicMock()
    mock_est_1.get_booster.return_value = mock_booster_1
    
    mock_booster_2 = MagicMock()
    mock_est_2 = MagicMock()
    mock_est_2.get_booster.return_value = mock_booster_2
    
    mock_model = MagicMock()
    mock_model.estimators_ = [mock_est_1, mock_est_2]
    
    x_sample_data = pd.DataFrame({
        'feat_a': np.random.rand(50),
        'feat_b': np.random.rand(50),
    })
    
    shap_vals = np.random.rand(50, 3)
    mock_booster_1.predict.return_value = np.hstack([shap_vals, np.random.rand(50, 1)])
    mock_booster_2.predict.return_value = np.hstack([shap_vals, np.random.rand(50, 1)])
    
    target_names = ['target_x', 'target_y']
    
    shap_feature_importance(mock_model, x_sample_data, target_names, max_points=100, n_top=2)
    
    assert "Builtin XGBoost SHAP Importance for target_x" in caplog.text
    assert "Builtin XGBoost SHAP Importance for target_y" in caplog.text


def test_shap_feature_importance_sampling(caplog):
    """Test shap_feature_importance samples data correctly."""
    
    caplog.set_level(logging.INFO)
    
    mock_booster = MagicMock()
    mock_est = MagicMock()
    mock_est.get_booster.return_value = mock_booster
    
    mock_model = MagicMock()
    mock_model.estimators_ = [mock_est]
    
    # Create large dataset
    x_data = pd.DataFrame({
        'f1': np.random.rand(10000),
        'f2': np.random.rand(10000),
    })
    
    shap_vals = np.random.rand(5000, 3)
    mock_booster.predict.return_value = np.hstack([shap_vals, np.random.rand(5000, 1)])
    
    shap_feature_importance(mock_model, x_data, ['target'], max_points=5000, n_top=1)
    
    # Verify predict was called with sampled data
    assert mock_booster.predict.called
    call_args = mock_booster.predict.call_args[0][0]
    assert call_args.num_row() <= 5000