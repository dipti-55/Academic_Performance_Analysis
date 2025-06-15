# app.py
import streamlit as st
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

st.set_page_config(page_title="📊 Academic Performance Predictor", layout="centered")

st.title("🎓 Academic Performance Predictor")
st.markdown("Fill in the student details below to estimate **Overall Academic Performance**")

# ----------------------------------------
# Form inputs
# ----------------------------------------
with st.form("prediction_form"):

    st.subheader("📘 Academic Metrics")
    study_hours_per_day = st.slider("Study Hours per Day", 1, 12, 4)
    attendance_rate = st.slider("Attendance Rate (%)", 60, 100, 85)
    homework_completion_rate = st.slider("Homework Completion Rate (%)", 40, 100, 80)
    tuition_hours_per_week = st.slider("Tuition/Coaching Hours per Week", 0, 12, 4)

    st.subheader("🏫 School & Environment")
    classroom_engagement_level = st.selectbox("Classroom Engagement Level", ["low", "medium", "high"])
    parents_academic_involvement = st.selectbox("Parents' Involvement in Academics", ["low", "medium", "high"])
    type_of_school = st.selectbox("Type of School", ["public", "private", "international"])
    household_income_bracket = st.selectbox("Household Income Bracket", ["low", "mid", "high"])

    st.subheader("🧠 Lifestyle & Health")
    stress_levels_during_exams = st.selectbox("Stress Levels During Exams", ["low", "medium", "high"])
    sleep_hours_per_day = st.slider("Sleep Hours per Day", 4, 12, 7)
    screen_time_non_academic = st.slider("Screen Time (Non-Academic, hrs/day)", 0, 10, 3)
    extracurricular_involvement = st.selectbox("Extracurricular Involvement", ["none", "low", "moderate", "high"])
    nutrition_level = st.selectbox("Nutrition Level", ["balanced", "junk-heavy", "mixed"])

    submit = st.form_submit_button("🔍 Predict Performance")

# ----------------------------------------
# Prediction logic
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

        st.success(f"🎯 Predicted Overall Academic Performance: **{prediction[0]:.2f} / 100**")

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
