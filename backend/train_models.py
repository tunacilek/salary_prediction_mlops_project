import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# MLflow ayarlama
mlflow.set_tracking_uri("http://localhost:5001/")
mlflow.set_experiment("SalaryPrediction")

df = pd.read_csv("Salary_Data.csv")
df.dropna(inplace=True)

categorical_cols = ["Gender", "education_level", "job_title"]
target_col = "Salary"

X = df.drop(columns=['Salary'])
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)

# Random Forest 
rf_preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)
    ],
    remainder="passthrough"
)

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

rf_pipeline = Pipeline(steps=[
    ("preprocessor", rf_preprocessor),
    ("regressor", rf_model)
])

with mlflow.start_run(run_name="Random_Forest_Model_Salary_v2"):
    rf_pipeline.fit(X_train, y_train)

    # Predictions
    y_train_pred = rf_pipeline.predict(X_train)
    y_test_pred = rf_pipeline.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)
    r2_train = r2_score(y_train, y_train_pred)

    # Log model
    mlflow.sklearn.log_model(rf_pipeline, "rf_model")

    # Log parameters
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", None)
    mlflow.log_param("min_samples_split", 2)
    mlflow.log_param("min_samples_leaf", 1)

    # Log metrics
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R2_test", r2_test)
    mlflow.log_metric("R2_train", r2_train)

    # Print
    print("Logged to MLflow at http://localhost:5001/")
    print("Mean Absolute Error (MAE):", mae)
    print("Test R^2 Score:", r2_test)
    print("Train R^2 Score:", r2_train)

#XGBoost
xgb_preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ],
    remainder="passthrough"
)

# XGB Model
xgb_model = XGBRegressor(
    objective="reg:squarederror",
    learning_rate=0.3,
    max_depth=7,
    n_estimators=200,
    subsample=0.8,
    random_state=42
)

# Create pipeline
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', xgb_preprocessor),
    ('regressor', xgb_model)
])



with mlflow.start_run(run_name="XGB_Model_Salary_v2"):
    # Fit model
    xgb_pipeline.fit(X_train, y_train)

    # Predict
    y_train_pred = xgb_pipeline.predict(X_train)
    y_pred = xgb_pipeline.predict(X_test)

    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    r2_test = r2_score(y_test, y_pred)
    r2_train = r2_score(y_train, y_train_pred)


    # Log model and parameters
    mlflow.log_param("learning_rate", 0.3)
    mlflow.log_param("max_depth", 7)
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("subsample", 0.8)

    mlflow.log_metric("mae", mae)
    mlflow.log_metric("Test r2_score", r2_test)
    mlflow.log_metric("Train r2_score", r2_train)


    # Log the full pipeline
    mlflow.sklearn.log_model(xgb_pipeline, "xgb_pipeline_model")

    print("MLflow run completed and model logged.")

# LightGBM
lgbm_preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ],
    remainder="passthrough"
)

lgbm = LGBMRegressor(
    learning_rate=0.1,
    max_depth=-1,
    n_estimators=200,
    num_leaves=50,
    subsample=0.8,
    random_state=42
)

lgbm_pipeline = Pipeline(steps=[
    ("preprocessor", lgbm_preprocessor),
    ("regressor", lgbm)
])

with mlflow.start_run(run_name="LGBM_Model_Salary_v2"):
    # Fit the pipeline
    lgbm_pipeline.fit(X_train, y_train)

    # Predict
    y_pred = lgbm_pipeline.predict(X_test)
    y_train_pred = lgbm_pipeline.predict(X_train)


    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    r2_train = r2_score(y_train, y_train_pred)


    # Log parameters
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("max_depth", -1)
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("num_leaves", 50)
    mlflow.log_param("subsample", 0.8)

    # Log metrics
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("Test r2_score", r2_test)
    mlflow.log_metric("Train r2_score", r2_train)

    # Log the complete pipeline
    mlflow.sklearn.log_model(lgbm_pipeline, "lgbm_pipeline_model")

    print("LightGBM model is logged to MLflow.")

# import joblib
# joblib.dump(xgb_model, "saved_models/xgboost_salary_prediction.pkl")