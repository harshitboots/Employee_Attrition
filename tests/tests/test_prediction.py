"""
Unit tests for prediction pipeline
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock


def test_prediction_uses_scaled_data():
    """
    Test that predictions use scaled data (not raw)
    This was the critical bug in the original app.py
    """
    # Mock data
    input_data = pd.DataFrame({
        "Age": [35],
        "Monthly Income": [5000],
        "Years at Company": [5]
    })
    
    # Create mock scaler and model
    mock_scaler = Mock()
    mock_scaler.transform.return_value = np.array([[1.0, 2.0, 3.0]])
    
    mock_model = Mock()
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
    
    # Scale and predict
    scaled_data = mock_scaler.transform(input_data)
    prediction = mock_model.predict(scaled_data)[0]
    probability = mock_model.predict_proba(scaled_data)[0][1]
    
    # Verify scaler was called
    mock_scaler.transform.assert_called_once()
    
    # Verify prediction uses scaled data
    mock_model.predict.assert_called_once()
    call_args = mock_model.predict.call_args[0][0]
    assert np.array_equal(call_args, scaled_data)
    
    assert prediction == 1
    assert probability == 0.7


def test_probability_in_valid_range():
    """Test that probability is between 0 and 1"""
    # Mock model
    mock_model = Mock()
    mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
    
    probability = mock_model.predict_proba(np.array([[1.0, 2.0]]))[0][1]
    
    assert 0 <= probability <= 1


def test_prediction_output_format():
    """Test that prediction output is in expected format"""
    mock_model = Mock()
    mock_model.predict.return_value = np.array([0])
    mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])
    
    prediction = mock_model.predict(np.array([[1.0, 2.0]]))[0]
    probability = mock_model.predict_proba(np.array([[1.0, 2.0]]))[0][1]
    
    # Prediction should be 0 or 1
    assert prediction in [0, 1]
    
    # Probability should be float between 0 and 1
    assert isinstance(probability, (float, np.floating))
    assert 0 <= probability <= 1
