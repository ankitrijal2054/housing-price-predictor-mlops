from fastapi import FastAPI
import xgboost as xgb
import pandas as pd
import uvicorn
import json

# Define paths
MODEL_PATH = "models/xgboost_model.json"

# Load the trained model
model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Housing Price Prediction API is running!"}

@app.post("/predict/")
def predict(features: dict):
    try:
        # Convert input JSON into a pandas DataFrame
        df = pd.DataFrame([features])
        
        # Make prediction
        prediction = model.predict(df)[0]

        return {"predicted_price": prediction}
    
    except Exception as e:
        return {"error": str(e)}

# Run API with uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
