from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.keras
from mlflow.tracking import MlflowClient
import numpy as np
import os
import time

app = FastAPI(title="Wine Quality Prediction API")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "wine-quality-model")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient()

model = None
current_model_version = None


def get_latest_model_version():
    """Récupère la dernière version du modèle depuis MLflow"""
    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        
        if not versions:
            return None
        
        # Trouver la version la plus récente
        latest = max(versions, key=lambda x: int(x.version))
        return latest.version
        
    except Exception as e:
        print(f"Erreur lors de la récupération de la version: {e}")
        return None


@app.on_event("startup")
async def load_model():
    """Charger le modèle au démarrage de l'API"""
    global model, current_model_version
    
    max_retries = 5
    retry_delay = 5  
    
    # Réessayer plusieurs fois (au cas où MLflow n'est pas encore prêt)
    for attempt in range(max_retries):
        try:
            print(f"Tentative {attempt + 1}/{max_retries} de connexion à MLflow...")
            
            latest_version = get_latest_model_version()
            
            if latest_version is None:
                print(f"⚠ Aucun modèle '{MODEL_NAME}' trouvé.")
                if attempt == max_retries - 1:
                    print("ℹ L'API démarrera sans modèle.")
                    return
                time.sleep(retry_delay)
                continue
            
            print(f"✓ Version la plus récente trouvée : {latest_version}")
            
            # Charger le modèle depuis MLflow
            model_uri = f"models:/{MODEL_NAME}/{latest_version}"
            model = mlflow.keras.load_model(model_uri)
            
            current_model_version = latest_version
            print(f"✓ Modèle chargé avec succès (version {latest_version})")
            return
            
        except Exception as e:
            print(f"✗ Erreur lors de la tentative {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                print(f"Nouvelle tentative dans {retry_delay} secondes...")
                time.sleep(retry_delay)
            else:
                print("⚠ L'API démarrera sans modèle.")


def check_for_model_update():
    """Vérifie si une nouvelle version du modèle est disponible et la charge automatiquement"""
    global model, current_model_version

    latest_version = get_latest_model_version()
    
    if latest_version is None:
        return False

    # Si nouvelle version détectée
    if current_model_version is None or latest_version != current_model_version:
        try:
            model_uri = f"models:/{MODEL_NAME}/{latest_version}"
            model = mlflow.keras.load_model(model_uri)
            current_model_version = latest_version
            print(f"✓ Modèle mis à jour vers la version {latest_version}")
            return True
        except Exception as e:
            print(f"✗ Erreur lors de la mise à jour du modèle: {e}")
            return False

    return False


class WineFeatures(BaseModel):
    type: str  # "red" ou "white"
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
    """Endpoint racine - informations de base sur l'API"""
    return {
        "message": "Wine Quality Prediction API",
        "status": "running",
        "model_name": MODEL_NAME
    }


@app.get("/health")
def health_check():
    """Health check - vérifier le statut de l'API et du modèle"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": MODEL_NAME,
        "model_version": current_model_version
    }


@app.get("/model/info")
def model_info():
    """Obtenir des informations détaillées sur le modèle actuellement chargé"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Récupérer les métadonnées du modèle depuis MLflow
        model_version_details = client.get_model_version(
            name=MODEL_NAME,
            version=current_model_version
        )
        
        return {
            "model_name": MODEL_NAME,
            "version": current_model_version,
            "run_id": model_version_details.run_id,
            "status": model_version_details.status,
            "creation_timestamp": model_version_details.creation_timestamp
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def predict(features: WineFeatures):
    """Prédire la qualité du vin à partir de ses caractéristiques"""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{MODEL_NAME}' not loaded. Please train and register a model first."
        )

    try:
        # Vérifier si une nouvelle version du modèle est disponible
        updated = check_for_model_update()

        # Encoder le type de vin : red=0, white=1
        wine_type = 0 if features.type.lower() == "red" else 1

        # Préparer les données d'entrée (ordre des features important!)
        input_data = np.array([[
            wine_type, 
            features.fixed_acidity, features.volatile_acidity,
            features.citric_acid, features.residual_sugar,
            features.chlorides, features.free_sulfur_dioxide,
            features.total_sulfur_dioxide, features.density,
            features.pH, features.sulphates, features.alcohol
        ]])

        # Faire la prédiction
        prediction = model.predict(input_data)

        return {
            "quality_prediction": float(prediction[0][0]),  # Score continu
            "quality_class": int(round(prediction[0][0])),  # Score arrondi (0-10)
            "model_updated": updated,
            "model_version": current_model_version,
            "model_name": MODEL_NAME,
            "wine_type": features.type
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/reload")
def reload_model():
    """Forcer le rechargement manuel du modèle depuis MLflow"""
    global model, current_model_version
    
    try:
        latest_version = get_latest_model_version()
        
        if latest_version is None:
            raise HTTPException(status_code=404, detail=f"No model '{MODEL_NAME}' found")
        
        # Charger la dernière version
        model_uri = f"models:/{MODEL_NAME}/{latest_version}"
        model = mlflow.keras.load_model(model_uri)
        current_model_version = latest_version
        
        return {
            "message": "Model reloaded successfully",
            "model_version": current_model_version
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))