# ============================================================
# EMPLOYEE ATTRITION PREDICTION — STREAMLIT APP
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import logging
from features import create_engineered_features

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Load saved model artefacts ───────────────────────────────
try:
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    logger.info("✓ Model loaded successfully")
except FileNotFoundError:
    st.error("❌ Model file not found: best_model.pkl")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    logger.error(f"Model loading failed: {str(e)}")
    st.stop()

try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    logger.info("✓ Scaler loaded successfully")
except FileNotFoundError:
    st.error("❌ Scaler file not found: scaler.pkl")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading scaler: {str(e)}")
    logger.error(f"Scaler loading failed: {str(e)}")
    st.stop()

try:
    with open("feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
    logger.info("✓ Feature columns loaded successfully")
except FileNotFoundError:
    st.error("❌ Feature columns file not found: feature_columns.pkl")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading feature columns: {str(e)}")
    logger.error(f"Feature columns loading failed: {str(e)}")
    st.stop()

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Employee Attrition Predictor")
st.markdown("Fill in the employee details below to predict whether they are at risk of leaving.")
st.markdown("---")

# ── Input form ───────────────────────────────────────────────
st.subheader("👤 Employee Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 65, 30)
    monthly_income = st.number_input("Monthly Income (£)", min_value=1000, max_value=20000, value=5000, step=500)
    years_at_company = st.slider("Years at Company", 0, 40, 5)
    company_tenure = st.slider("Company Tenure (years)", 0, 40, 5)
    distance_from_home = st.slider("Distance from Home (km)", 1, 100, 15)
    num_promotions = st.slider("Number of Promotions", 0, 10, 1)
    num_dependents = st.slider("Number of Dependents", 0, 10, 1)

with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    overtime = st.selectbox("Overtime", ["No", "Yes"])
    remote_work = st.selectbox("Remote Work", ["No", "Yes"])
    leadership = st.selectbox("Leadership Opportunities", ["No", "Yes"])
    innovation = st.selectbox("Innovation Opportunities", ["No", "Yes"])
    job_role = st.selectbox("Job Role", [
        "Data Scientist", "Finance Manager", "Healthcare Representative",
        "Human Resources", "Laboratory Technician", "Manager",
        "Manufacturing Director", "Marketing Manager", "Research Director",
        "Research Scientist", "Sales Executive", "Sales Representative",
        "Software Engineer"
    ])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

st.markdown("---")
st.subheader("📊 Work & Satisfaction")

col3, col4 = st.columns(2)

with col3:
    job_satisfaction = st.selectbox("Job Satisfaction", ["Low", "Medium", "High", "Very High"])
    work_life_balance = st.selectbox("Work-Life Balance", ["Poor", "Fair", "Good", "Excellent"])
    performance_rating = st.selectbox("Performance Rating", ["Low", "Below Average", "Average", "High", "Very High"])
    education_level = st.selectbox("Education Level", [
        "High School", "Associate Degree", "Bachelor's Degree", "Master's Degree", "PhD"
    ])

with col4:
    job_level = st.selectbox("Job Level", ["Entry", "Mid", "Senior", "Manager", "Director"])
    company_size = st.selectbox("Company Size", ["Small", "Medium", "Large"])
    company_reputation = st.selectbox("Company Reputation", ["Poor", "Fair", "Good", "Excellent"])
    employee_recognition = st.selectbox("Employee Recognition", ["Low", "Medium", "High", "Very High"])

st.markdown("---")

# ── Predict button ───────────────────────────────────────────
if st.button("🔮 Predict Attrition Risk", use_container_width=True):
    try:
        # Validate inputs
        if age < 18 or age > 65:
            st.error("❌ Age must be between 18 and 65")
            st.stop()
        if monthly_income <= 0:
            st.error("❌ Monthly income must be greater than 0")
            st.stop()
        if years_at_company < 0:
            st.error("❌ Years at company cannot be negative")
            st.stop()

        # Build input dict
        input_dict = {
            "Age": age,
            "Years at Company": years_at_company,
            "Monthly Income": monthly_income,
            "Number of Promotions": num_promotions,
            "Distance from Home": distance_from_home,
            "Number of Dependents": num_dependents,
            "Company Tenure": company_tenure,
            "Gender": 1 if gender == "Female" else 0,
            "Overtime": 1 if overtime == "Yes" else 0,
            "Remote Work": 1 if remote_work == "Yes" else 0,
            "Leadership Opportunities": 1 if leadership == "Yes" else 0,
            "Innovation Opportunities": 1 if innovation == "Yes" else 0,
            "Work-Life Balance": {"Poor": 1, "Fair": 2, "Good": 3, "Excellent": 4}[work_life_balance],
            "Job Satisfaction": {"Low": 1, "Medium": 2, "High": 3, "Very High": 4}[job_satisfaction],
            "Performance Rating": {"Low": 1, "Below Average": 2, "Average": 3, "High": 4, "Very High": 5}[performance_rating],
            "Education Level": {"High School": 1, "Associate Degree": 2, "Bachelor's Degree": 3, "Master's Degree": 4, "PhD": 5}[education_level],
            "Job Level": {"Entry": 1, "Mid": 2, "Senior": 3, "Manager": 4, "Director": 5}[job_level],
            "Company Size": {"Small": 1, "Medium": 2, "Large": 3}[company_size],
            "Company Reputation": {"Poor": 1, "Fair": 2, "Good": 3, "Excellent": 4}[company_reputation],
            "Employee Recognition": {"Low": 1, "Medium": 2, "High": 3, "Very High": 4}[employee_recognition],
        }

        # One-hot encode Job Role and Marital Status
        all_roles = [
            "Data Scientist", "Finance Manager", "Healthcare Representative",
            "Human Resources", "Laboratory Technician", "Manager",
            "Manufacturing Director", "Marketing Manager", "Research Director",
            "Research Scientist", "Sales Executive", "Sales Representative",
            "Software Engineer"
        ]
        all_marital = ["Single", "Married", "Divorced"]

        for role in all_roles[1:]:
            input_dict[f"Job Role_{role}"] = 1 if job_role == role else 0

        for status in all_marital[1:]:
            input_dict[f"Marital Status_{status}"] = 1 if marital_status == status else 0

        # Create engineered features using the module
        input_dict = create_engineered_features(input_dict)

        # Align to training columns
        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        # ✅ FIX: Scale the data before prediction (CRITICAL BUG FIX)
        input_scaled = scaler.transform(input_df)
        
        # ✅ Use scaled data for prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        logger.info(f"Prediction made: {prediction}, Probability: {probability:.2%}")

        # ── Result ───────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🎯 Prediction Result")

        if prediction == 1:
            st.error(f"🚨 **AT RISK** — This employee is likely to leave")
            st.metric("Attrition Probability", f"{probability*100:.1f}%")
        else:
            st.success(f"✅ **LIKELY TO STAY** — This employee appears retained")
            st.metric("Attrition Probability", f"{probability*100:.1f}%")

        # Risk gauge
        st.progress(float(probability))

        # Key drivers
        st.markdown("---")
        st.subheader("📌 Key Risk Factors for This Employee")

        factors = {
            "Overtime": "High risk factor ⚠️" if overtime == "Yes" else "Low risk ✅",
            "Wellbeing Score": f"{input_dict['Wellbeing_Score']}/12 — {'Low ⚠️' if input_dict['Wellbeing_Score'] < 7 else 'Good ✅'}",
            "Promotion Deprivation": f"{input_dict['Promotion_Deprivation']:.0f} years — {'High ⚠️' if input_dict['Promotion_Deprivation'] > 5 else 'OK ✅'}",
            "Income per Tenure Year": f"£{input_dict['Income_per_Tenure_Year']:.0f} — {'Low ⚠️' if input_dict['Income_per_Tenure_Year'] < 1000 else 'OK ✅'}",
            "Stress Score": f"{input_dict['Stress_Score']:.2f} — {'High ⚠️' if input_dict['Stress_Score'] > 1 else 'Low ✅'}",
        }

        for factor, value in factors.items():
            st.write(f"**{factor}:** {value}")

    except ValueError as ve:
        st.error(f"❌ Invalid input: {str(ve)}")
        logger.error(f"Input validation error: {str(ve)}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        logger.error(f"Prediction error: {str(e)}")

st.markdown("---")
st.caption("Built by Deepthi | MSc Data Science & Analytics | University of Hertfordshire")
