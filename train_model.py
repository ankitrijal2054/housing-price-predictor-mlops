import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from mlflow.models import infer_signature

# Define paths
PROCESSED_DATA_PATH = "data/processed/housing_processed.csv"
MODEL_PATH = "models/xgboost_model.json"

# Load processed data
df = pd.read_csv(PROCESSED_DATA_PATH)

# Define target and features
TARGET = "SalePrice"
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.astype('float64')
X_test = X_test.astype('float64')

# Initialize MLflow tracking
mlflow.set_experiment("Housing Price Prediction")

with mlflow.start_run():
    # Define model
    model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate performance
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Log parameters and metrics in MLflow
    mlflow.log_param("n_estimators", 500)
    mlflow.log_param("max_depth", 6)
    mlflow.log_param("learning_rate", 0.05)
    mlflow.log_param("subsample", 0.8)
    mlflow.log_param("colsample_bytree", 0.8)

    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2 Score", r2)

    # Save model
    os.makedirs("models", exist_ok=True)
    model.save_model(MODEL_PATH)
    
    # Infer model signature
    signature = infer_signature(X_train, model.predict(X_train))

    # Save & log model in MLflow
    os.makedirs("models", exist_ok=True)
    model.save_model(MODEL_PATH)
    mlflow.xgboost.log_model(model, artifact_path="xgboost_model", signature=signature, input_example=X_train.iloc[:5])

    print(f"Model saved at {MODEL_PATH}")
    print(f"MAE: {mae}, RMSE: {rmse}, R2: {r2}")
