"""
Integration tests for the full prediction pipeline
"""
import pytest
import pandas as pd
import numpy as np
from features import create_engineered_features


def test_end_to_end_feature_engineering():
    """Test complete feature engineering pipeline"""
    employee_data = {
        "Age": 35,
        "Monthly Income": 5000,
        "Years at Company": 5,
        "Number of Promotions": 1,
        "Distance from Home": 20,
        "Job Satisfaction": 3,
        "Work-Life Balance": 3,
        "Employee Recognition": 2,
        "Overtime": 1,
    }
    
    result = create_engineered_features(employee_data.copy())
    
    # Verify all engineered features exist
    assert "Income_per_Tenure_Year" in result
    assert "Promotion_Deprivation" in result
    assert "Wellbeing_Score" in result
    assert "Stress_Score" in result
    
    # Verify feature values are reasonable
    assert result["Income_per_Tenure_Year"] > 0
    assert result["Wellbeing_Score"] >= 3
    assert result["Wellbeing_Score"] <= 12
    assert result["Stress_Score"] >= 0


def test_high_risk_employee():
    """Test feature engineering for high-risk employee profile"""
    high_risk_employee = {
        "Monthly Income": 3000,  # Low income
        "Years at Company": 15,  # Long tenure
        "Number of Promotions": 0,  # No promotions
        "Job Satisfaction": 1,  # Low satisfaction
        "Work-Life Balance": 1,  # Poor balance
        "Employee Recognition": 1,  # Low recognition
        "Overtime": 1,  # Works overtime
        "Distance from Home": 50,  # Long commute
    }
    
    result = create_engineered_features(high_risk_employee.copy())
    
    # High-risk indicators
    assert result["Income_per_Tenure_Year"] < 500  # Low income for tenure
    assert result["Promotion_Deprivation"] > 10  # Stagnation
    assert result["Wellbeing_Score"] <= 5  # Poor wellbeing
    assert result["Stress_Score"] > 1  # High stress


def test_low_risk_employee():
    """Test feature engineering for low-risk employee profile"""
    low_risk_employee = {
        "Monthly Income": 8000,  # High income
        "Years at Company": 3,  # Short tenure
        "Number of Promotions": 1,  # Some promotion
        "Job Satisfaction": 4,  # High satisfaction
        "Work-Life Balance": 4,  # Good balance
        "Employee Recognition": 4,  # High recognition
        "Overtime": 0,  # No overtime
        "Distance from Home": 10,  # Short commute
    }
    
    result = create_engineered_features(low_risk_employee.copy())
    
    # Low-risk indicators
    assert result["Income_per_Tenure_Year"] > 1500  # Good income
    assert result["Promotion_Deprivation"] < 0  # Good career progress
    assert result["Wellbeing_Score"] >= 10  # Good wellbeing
    assert result["Stress_Score"] < 0.5  # Low stress
