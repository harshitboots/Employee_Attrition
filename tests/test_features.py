"""
Unit tests for feature engineering module
"""
import pytest
from features import (
    calculate_income_per_tenure_year,
    calculate_promotion_deprivation,
    calculate_wellbeing_score,
    calculate_stress_score,
    create_engineered_features
)


class TestIncomePerTenureYear:
    """Tests for income per tenure year calculation"""
    
    def test_normal_case(self):
        """Test normal calculation"""
        result = calculate_income_per_tenure_year(5000, 5)
        assert result == pytest.approx(833.33, rel=0.01)
    
    def test_zero_tenure(self):
        """Test handles zero tenure gracefully"""
        result = calculate_income_per_tenure_year(5000, 0)
        assert result == 5000
    
    def test_high_income(self):
        """Test with high income"""
        result = calculate_income_per_tenure_year(10000, 10)
        assert result == pytest.approx(909.09, rel=0.01)
    
    def test_low_income(self):
        """Test with low income"""
        result = calculate_income_per_tenure_year(1500, 1)
        assert result == pytest.approx(750, rel=0.01)


class TestPromotionDeprivation:
    """Tests for promotion deprivation calculation"""
    
    def test_normal_case(self):
        """Test normal calculation"""
        result = calculate_promotion_deprivation(10, 2)
        assert result == 4  # 10 - (2 * 3)
    
    def test_no_promotions(self):
        """Test with no promotions"""
        result = calculate_promotion_deprivation(10, 0)
        assert result == 10
    
    def test_high_promotions(self):
        """Test with high promotion count"""
        result = calculate_promotion_deprivation(5, 3)
        assert result == -4  # 5 - (3 * 3)
    
    def test_short_tenure(self):
        """Test with short tenure"""
        result = calculate_promotion_deprivation(1, 0)
        assert result == 1


class TestWellbeingScore:
    """Tests for wellbeing score calculation"""
    
    def test_all_high(self):
        """Test all high scores"""
        result = calculate_wellbeing_score(4, 4, 4)
        assert result == 12
    
    def test_all_low(self):
        """Test all low scores"""
        result = calculate_wellbeing_score(1, 1, 1)
        assert result == 3
    
    def test_mixed(self):
        """Test mixed scores"""
        result = calculate_wellbeing_score(3, 2, 4)
        assert result == 9
    
    def test_average(self):
        """Test average scores"""
        result = calculate_wellbeing_score(2, 3, 2)
        assert result == 7


class TestStressScore:
    """Tests for stress score calculation"""
    
    def test_overtime_and_distance(self):
        """Test with overtime and distance"""
        result = calculate_stress_score(1, 50)
        assert result == pytest.approx(1.5)
    
    def test_no_overtime_no_distance(self):
        """Test with no overtime and no distance"""
        result = calculate_stress_score(0, 0)
        assert result == pytest.approx(0)
    
    def test_overtime_only(self):
        """Test with overtime only"""
        result = calculate_stress_score(1, 0)
        assert result == pytest.approx(1)
    
    def test_distance_only(self):
        """Test with distance only"""
        result = calculate_stress_score(0, 100)
        assert result == pytest.approx(1.0)


class TestCreateEngineeredFeatures:
    """Tests for create_engineered_features function"""
    
    def test_complete_input(self):
        """Test with complete input"""
        data = {
            "Monthly Income": 5000,
            "Years at Company": 5,
            "Number of Promotions": 1,
            "Job Satisfaction": 3,
            "Work-Life Balance": 3,
            "Employee Recognition": 2,
            "Overtime": 1,
            "Distance from Home": 20
        }
        result = create_engineered_features(data.copy())
        
        assert "Income_per_Tenure_Year" in result
        assert "Promotion_Deprivation" in result
        assert "Wellbeing_Score" in result
        assert "Stress_Score" in result
    
    def test_partial_input(self):
        """Test with partial input (should use defaults)"""
        data = {"Monthly Income": 5000}
        result = create_engineered_features(data.copy())
        
        assert "Income_per_Tenure_Year" in result
        assert result["Income_per_Tenure_Year"] == 5000
