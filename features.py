"""
Feature Engineering Module for Employee Attrition Prediction

This module contains all feature engineering logic used in both 
the notebooks and the Streamlit app. Centralizing this logic ensures
consistency and makes it easier to test and maintain.
"""


def calculate_income_per_tenure_year(monthly_income, years_at_company):
    """
    Calculate income per tenure year.
    
    High income relative to tenure may indicate undercompensation,
    which is a flight risk indicator.
    
    Args:
        monthly_income (float): Employee's monthly income
        years_at_company (int): Years employee has been at the company
        
    Returns:
        float: Income per tenure year (handles zero tenure case)
    """
    if years_at_company == 0:
        return monthly_income
    return monthly_income / (years_at_company + 1)


def calculate_promotion_deprivation(years_at_company, num_promotions):
    """
    Calculate promotion deprivation score.
    
    Employees with long tenure but few promotions show career stagnation,
    a key attrition risk factor.
    
    Args:
        years_at_company (int): Years employee has been at the company
        num_promotions (int): Number of promotions received
        
    Returns:
        float: Promotion deprivation score
    """
    return years_at_company - (num_promotions * 3)


def calculate_wellbeing_score(job_satisfaction, work_life_balance, employee_recognition):
    """
    Calculate employee wellbeing score.
    
    Combines job satisfaction, work-life balance, and recognition
    into a single composite metric (range: 3-12, higher is better).
    
    Args:
        job_satisfaction (int): Job satisfaction level (1-4)
        work_life_balance (int): Work-life balance level (1-4)
        employee_recognition (int): Recognition level (1-4)
        
    Returns:
        int: Wellbeing score (3-12)
    """
    return job_satisfaction + work_life_balance + employee_recognition


def calculate_stress_score(overtime, distance_from_home):
    """
    Calculate stress score.
    
    Combines overtime and commute distance into a burnout risk metric.
    
    Args:
        overtime (int): 1 if works overtime, 0 otherwise
        distance_from_home (float): Distance from home in km
        
    Returns:
        float: Stress score
    """
    return overtime + (distance_from_home / 100)


def create_engineered_features(data_dict):
    """
    Create all engineered features from raw input.
    
    Args:
        data_dict (dict): Dictionary containing all feature values
        
    Returns:
        dict: Updated dictionary with engineered features added
    """
    data_dict["Income_per_Tenure_Year"] = calculate_income_per_tenure_year(
        data_dict.get("Monthly Income", 0),
        data_dict.get("Years at Company", 0)
    )
    
    data_dict["Promotion_Deprivation"] = calculate_promotion_deprivation(
        data_dict.get("Years at Company", 0),
        data_dict.get("Number of Promotions", 0)
    )
    
    data_dict["Wellbeing_Score"] = calculate_wellbeing_score(
        data_dict.get("Job Satisfaction", 0),
        data_dict.get("Work-Life Balance", 0),
        data_dict.get("Employee Recognition", 0)
    )
    
    data_dict["Stress_Score"] = calculate_stress_score(
        data_dict.get("Overtime", 0),
        data_dict.get("Distance from Home", 0)
    )
    
    return data_dict
