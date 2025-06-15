import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'

            print("🔄 Loading preprocessor and model...")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            print("✅ Loaded. Transforming data and predicting...")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            # 🔒 Ensure predictions are within valid bounds
            preds = np.clip(preds, 25, 98)

            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        study_hours_per_day: int,
        attendance_rate: float,
        homework_completion_rate: float,
        classroom_engagement_level: str,
        parents_academic_involvement: str,
        type_of_school: str,
        tuition_hours_per_week: int,
        sleep_hours_per_day: float,
        screen_time_non_academic: float,
        household_income_bracket: str,
        stress_levels_during_exams: str,
        extracurricular_involvement: str,
        nutrition_level: str,
    ):
        self.study_hours_per_day = study_hours_per_day
        self.attendance_rate = attendance_rate
        self.homework_completion_rate = homework_completion_rate
        self.classroom_engagement_level = classroom_engagement_level
        self.parents_academic_involvement = parents_academic_involvement
        self.type_of_school = type_of_school
        self.tuition_hours_per_week = tuition_hours_per_week
        self.sleep_hours_per_day = sleep_hours_per_day
        self.screen_time_non_academic = screen_time_non_academic
        self.household_income_bracket = household_income_bracket
        self.stress_levels_during_exams = stress_levels_during_exams
        self.extracurricular_involvement = extracurricular_involvement
        self.nutrition_level = nutrition_level

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "study_hours_per_day": [self.study_hours_per_day],
                "attendance_rate": [self.attendance_rate],
                "homework_completion_rate": [self.homework_completion_rate],
                "classroom_engagement_level": [self.classroom_engagement_level],
                "parents_academic_involvement": [self.parents_academic_involvement],
                "type_of_school": [self.type_of_school],
                "tuition_hours_per_week": [self.tuition_hours_per_week],
                "sleep_hours_per_day": [self.sleep_hours_per_day],
                "screen_time_non_academic": [self.screen_time_non_academic],
                "household_income_bracket": [self.household_income_bracket],
                "stress_levels_during_exams": [self.stress_levels_during_exams],
                "extracurricular_involvement": [self.extracurricular_involvement],
                "nutrition_level": [self.nutrition_level],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
