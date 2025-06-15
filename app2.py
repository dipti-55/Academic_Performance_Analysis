# app.py
import streamlit as st
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# ----------------------------------------
# 🎨 Page Config and Style
# ----------------------------------------
st.set_page_config(page_title="📊 Academic Performance Predictor", layout="wide")

st.markdown("""
    <style>
    body {
        background-color: #F0F2F6;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 0.5em 2em;
        border-radius: 8px;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------
# 🎯 Title
# ----------------------------------------
st.title("🎓 Academic Performance Predictor")
st.markdown("Predict a student's **Overall Academic Performance** based on personal, academic, and lifestyle factors.")

# ----------------------------------------
# 📋 Input Form
# ----------------------------------------
with st.form("prediction_form"):
    st.markdown("### 📘 Academic Metrics")
    col1, col2 = st.columns(2)

    with col1:
        study_hours_per_day = st.slider("🕒 Study Hours/Day", 1, 12, 4)
        homework_completion_rate = st.slider("📄 Homework Completion Rate (%)", 40, 100, 80)
        tuition_hours_per_week = st.slider("📚 Tuition Hours/Week", 0, 12, 4)

    with col2:
        attendance_rate = st.slider("📆 Attendance Rate (%)", 60, 100, 85)
        sleep_hours_per_day = st.slider("🛏️ Sleep Hours/Day", 4, 12, 7)
        screen_time_non_academic = st.slider("📱 Non-Academic Screen Time (hrs)", 0, 10, 3)

    st.markdown("### 🏫 School & Environment")
    col3, col4 = st.columns(2)

    with col3:
        classroom_engagement_level = st.selectbox("🧑‍🏫 Classroom Engagement", ["low", "medium", "high"])
        parents_academic_involvement = st.selectbox("👨‍👩‍👧 Parents' Involvement", ["low", "medium", "high"])

    with col4:
        type_of_school = st.selectbox("🏫 Type of School", ["public", "private", "international"])
        household_income_bracket = st.selectbox("💰 Income Bracket", ["low", "mid", "high"])

    st.markdown("### 🧠 Lifestyle & Health")
    col5, col6 = st.columns(2)

    with col5:
        stress_levels_during_exams = st.selectbox("😰 Stress During Exams", ["low", "medium", "high"])
        extracurricular_involvement = st.selectbox("🎭 Extracurriculars", ["none", "low", "moderate", "high"])

    with col6:
        nutrition_level = st.selectbox("🥗 Nutrition Level", ["balanced", "junk-heavy", "mixed"])

    # Submit button
    submit = st.form_submit_button("🔍 Predict Performance")

# ----------------------------------------
# 🚀 Prediction
# ----------------------------------------
if submit:
    try:
        user_data = CustomData(
            study_hours_per_day=study_hours_per_day,
            attendance_rate=attendance_rate,
            homework_completion_rate=homework_completion_rate,
            classroom_engagement_level=classroom_engagement_level,
            parents_academic_involvement=parents_academic_involvement,
            type_of_school=type_of_school,
            tuition_hours_per_week=tuition_hours_per_week,
            sleep_hours_per_day=sleep_hours_per_day,
            screen_time_non_academic=screen_time_non_academic,
            household_income_bracket=household_income_bracket,
            stress_levels_during_exams=stress_levels_during_exams,
            extracurricular_involvement=extracurricular_involvement,
            nutrition_level=nutrition_level
        )

        input_df = user_data.get_data_as_data_frame()
        pipeline = PredictPipeline()
        prediction = pipeline.predict(input_df)

        # ✨ Display result
        st.success(f"🎯 Predicted Academic Performance: **{prediction[0]:.2f} / 100**")

    except Exception as e:
        st.error(f"❌ Error: {e}")
