import os

import mlflow
from fastapi import FastAPI

app = FastAPI(title="CTR Prediction Service")

MODEL_URI = os.environ.get("MODEL_URI", "models:/ctr-xgboost/Production")
_model = None


@app.on_event("startup")
def load_model() -> None:
    global _model
    _model = mlflow.pyfunc.load_model(MODEL_URI)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict")
def predict(features: dict) -> dict:
    prediction = _model.predict([features])
    return {"click_probability": float(prediction[0])}
