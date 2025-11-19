from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
import numpy as np
import os

app = FastAPI(title="Wine Quality Prediction API")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        model_uri = "models:/wine_quality_model/Production"
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        print(f"Erreur de chargement du modèle: {e}")

class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

@app.get("/")
def read_root():
    return {"message": "Wine Quality Prediction API", "status": "running"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(features: WineFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        input_data = np.array([[
            features.fixed_acidity, features.volatile_acidity, 
            features.citric_acid, features.residual_sugar,
            features.chlorides, features.free_sulfur_dioxide,
            features.total_sulfur_dioxide, features.density,
            features.pH, features.sulphates, features.alcohol
        ]])
        
        prediction = model.predict(input_data)
        
        return {
            "quality_prediction": float(prediction[0]),
            "quality_class": int(round(prediction[0]))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))