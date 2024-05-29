from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel
import logging

# Initialize the app and load the model and scaler
app = FastAPI()

try:
    model = joblib.load('knn_model.joblib')
    scaler = joblib.load('scaler.joblib')
except Exception as e:
    logging.error(f"Error loading model or scaler: {e}")
    raise

@app.get("/")
def root():
    return "Welcome To Tuwaiq Academy"

# Define a Pydantic model for input data validation
class InputFeatures(BaseModel):
    appearance: int
    minutes_played: int
    highest_value: int
    award: int
    kmeans:int

def preprocessing(input_features: InputFeatures, scaler):
    dict_f = {
        'appearance': input_features.appearance,
        'minutes_played': input_features.minutes_played,
        'highest_value': input_features.highest_value,
        'award': input_features.award,
        'kmeans': input_features.kmeans
    }
    # Scale the input features using the provided scaler
    scaled_features = scaler.transform([list(dict_f.values())])
    return scaled_features

@app.post("/predict")
async def predict(input_features: InputFeatures):
    try:
        # Call preprocessing function with the scaler
        data = preprocessing(input_features, scaler)
        y_pred = model.predict(data)
        return {"pred": y_pred.tolist()[0]}
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
