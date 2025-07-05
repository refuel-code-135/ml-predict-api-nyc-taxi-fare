import os

import joblib
import lightgbm as lgb
from fastapi import FastAPI, HTTPException
from features import generate_features
from pydantic import BaseModel

app = FastAPI()

model = None
pickup_kmeans = None
dropoff_kmeans = None


@app.on_event("startup")
def load_model_and_clusters():
    global model, pickup_kmeans, dropoff_kmeans

    model_path = os.path.join("data", "final_model_1percent.txt")
    pickup_cluster_path = os.path.join("data", "pickup_kmeans.pkl")
    dropoff_cluster_path = os.path.join("data", "dropoff_kmeans.pkl")

    model = lgb.Booster(model_file=model_path)
    pickup_kmeans = joblib.load(pickup_cluster_path)
    dropoff_kmeans = joblib.load(dropoff_cluster_path)


class TaxiFareRequest(BaseModel):
    pickup_datetime: str
    pickup_latitude: float
    pickup_longitude: float
    dropoff_latitude: float
    dropoff_longitude: float
    passenger_count: int


@app.post("/predict")
def predict_fare(request: TaxiFareRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model Loading Error")

    try:
        features = generate_features(request.dict(), pickup_kmeans, dropoff_kmeans)
        prediction = model.predict([features])[0]
        return {"predicted_fare": round(float(prediction), 2)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
