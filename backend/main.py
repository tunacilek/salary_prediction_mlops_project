from fastapi import FastAPI, Depends
from sqlmodel import Session
from data import SalaryInput, PredictionOutput
from models import Salary, RequestSalary, SalaryDriftInput
from database import get_db, create_db_and_tables, engine
import mlflow.pyfunc
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency
from typing import List
import joblib
from fastapi import HTTPException
import traceback

# Register edilmiş modeli MLFLow'dan çağırıyoruz
# mlflow.set_tracking_uri("http://localhost:5001")
# model_uri = "models:/Best_Salary_Prediction/1"

model = joblib.load("saved_models/xgb_pipeline.pkl")

app = FastAPI()

def detect_drift(data1: pd.DataFrame, data2: pd.DataFrame, categorical_features: List[str], numerical_features: List[str]):
    drift_report = {}

    # Numerical features - KS Test
    for feature in numerical_features:
        stat, p_value = ks_2samp(data1[feature], data2[feature])
        drift_report[feature] = "Drift exists" if p_value < 0.05 else "No drift"

    # Categorical features - Chi-Square Test
    for feature in categorical_features:
        train_counts = data1[feature].value_counts()
        new_counts = data2[feature].value_counts()

        all_categories = list(set(train_counts.index).union(set(new_counts.index)))
        train_aligned = train_counts.reindex(all_categories, fill_value=0)
        new_aligned = new_counts.reindex(all_categories, fill_value=0)

        contingency = pd.DataFrame([train_aligned, new_aligned])
        _, p_value, _, _ = chi2_contingency(contingency)

        drift_report[feature] = "Drift exists" if p_value < 0.05 else "No drift"

    return drift_report
    
# Initialize the database and create tables on startup
@app.on_event("startup")
def on_startup():
    create_db_and_tables()

@app.post("/prediction/salary")
def predict_and_store(data: RequestSalary, db: Session = Depends(get_db)):
    try:
        print("✅ Received input:", data.dict())

        input_df = pd.DataFrame([{
            "Age": data.Age,
            "years_of_experience": data.years_of_experience,
            "Gender": data.Gender,
            "education_level": data.education_level,
            "job_title": data.job_title
        }])

        print("✅ DataFrame prepared for model:")
        print(input_df)

        # ✅ Pipeline handles preprocessing internally
        prediction = model.predict(input_df)[0]
        prediction = float(np.ceil(prediction))
        print(f"✅ Prediction successful: {prediction}")

        # Save to DB
        prediction_record = Salary(
            Age=data.Age,
            Gender=data.Gender,
            education_level=data.education_level,
            job_title=data.job_title,
            years_of_experience=data.years_of_experience,
            prediction=prediction
        )
        db.add(prediction_record)
        db.commit()
        print("✅ Data committed to DB")

        return {"predicted_salary": prediction}

    except Exception as e:
        print("❌ Error occurred during prediction or DB commit:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")



# Salary drift detection endpoint
@app.post("/drift/salary")
async def detect_salary_drift(request: SalaryDriftInput):
    # Load training data
    train_df = pd.read_sql("SELECT * FROM salarytrain", engine)

    # Load recent prediction data from the last n days
    prediction_df = pd.read_sql(f"""
        SELECT * FROM salary
        WHERE prediction_time > current_date - {request.n_days_before}
    """, engine)

    # Categorical features
    categorical_features = ['Gender', 'education_level', 'job_title']

    # Numerical features
    numerical_features = ['Age', 'years_of_experience']

    # Detect drift
    drift_result = detect_drift(train_df, prediction_df, categorical_features, numerical_features)

    return drift_result

@app.get("/test-db")
def test_db_connection():
    with engine.connect() as connection:
        result = connection.execute("SELECT 1")
        return {"db_test": [row for row in result]}
