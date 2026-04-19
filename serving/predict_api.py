from fastapi import FastAPI
import pandas as pd
import mlflow
import os
import yaml

## Chargement des paramètres
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
params_path = os.path.join(BASE_DIR, "params.yaml")

params = yaml.safe_load(open(params_path))
mlflow_params = params['mlflow']

## Définition de l'adresse du serveur MLflow
mlflow.set_tracking_uri(mlflow_params["MLFLOW_TRACKING_URI"])

MODEL_NAME = "Best_RandomForestClassifier"
MODEL_VERSION = "latest"

print("Chargement du modèle depuis MLflow Registry...")

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{MODEL_NAME}/{MODEL_VERSION}"
)

print("Modèle chargé avec succès")

app = FastAPI(title="API de prédiction ML")

@app.get("/")
def home():
    return {"message": "API MLflow opérationnelle"}
@app.get("/")
def health():
    return {"status": "ok"}
@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    prediction = model.predict(df)

    return {
        "prediction": int(prediction[0])
    }