import os

import joblib
import lightgbm as lgb
from fastapi import FastAPI

app = FastAPI()

# グローバル変数としてモデルとクラスタ保持
model = None
pickup_kmeans = None
dropoff_kmeans = None


@app.on_event("startup")
def load_model_and_clusters():
    global model, pickup_kmeans, dropoff_kmeans

    model_path = os.path.join(os.path.dirname(__file__), "final_model_1percent.txt")
    pickup_cluster_path = os.path.join(os.path.dirname(__file__), "pickup_kmeans.pkl")
    dropoff_cluster_path = os.path.join(os.path.dirname(__file__), "dropoff_kmeans.pkl")

    model = lgb.Booster(model_file=model_path)
    pickup_kmeans = joblib.load(pickup_cluster_path)
    dropoff_kmeans = joblib.load(dropoff_cluster_path)

    print("✅ モデルとクラスタファイルをロードしました")


from datetime import datetime

import numpy as np


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # 地球半径 (km)
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


# ニューヨークの主要ランドマークの座標
landmarks = {
    "JFK": (40.6413, -73.7781),
    "LGA": (40.7769, -73.8740),
    "EWR": (40.6895, -74.1745),
    "Penn_Station": (40.7506, -73.9935),
    "Grand_Central": (40.7527, -73.9772),
    "Times_Square": (40.7580, -73.9855),
}


def generate_features(data: dict):
    pickup_dt = datetime.fromisoformat(data["pickup_datetime"])
    year = pickup_dt.year
    month = pickup_dt.month
    weekday = pickup_dt.weekday()
    hour = pickup_dt.hour

    delta_lat = abs(data["pickup_latitude"] - data["dropoff_latitude"])
    delta_lon = abs(data["pickup_longitude"] - data["dropoff_longitude"])

    distance_km = haversine_distance(
        data["pickup_latitude"],
        data["pickup_longitude"],
        data["dropoff_latitude"],
        data["dropoff_longitude"],
    )

    manhattan_km = haversine_distance(
        data["pickup_latitude"],
        data["pickup_longitude"],
        data["pickup_latitude"],
        data["dropoff_longitude"],
    ) + haversine_distance(
        data["pickup_latitude"],
        data["dropoff_longitude"],
        data["dropoff_latitude"],
        data["dropoff_longitude"],
    )

    landmark_features = {}
    for name, (lat, lon) in landmarks.items():
        landmark_features[f"pickup_distance_to_{name}"] = haversine_distance(
            data["pickup_latitude"], data["pickup_longitude"], lat, lon
        )
        landmark_features[f"dropoff_distance_to_{name}"] = haversine_distance(
            data["dropoff_latitude"], data["dropoff_longitude"], lat, lon
        )

    pickup_cluster = pickup_kmeans.predict(
        [[data["pickup_latitude"], data["pickup_longitude"]]]
    )[0]
    dropoff_cluster = dropoff_kmeans.predict(
        [[data["dropoff_latitude"], data["dropoff_longitude"]]]
    )[0]

    path_efficiency = distance_km / manhattan_km if manhattan_km != 0 else 1
    pickup_vs_dropoff_to_JFK = (
        landmark_features["pickup_distance_to_JFK"]
        - landmark_features["dropoff_distance_to_JFK"]
    )

    feature_vector = [
        data["pickup_longitude"],
        data["pickup_latitude"],
        data["dropoff_longitude"],
        data["dropoff_latitude"],
        data["passenger_count"],
        year,
        month,
        weekday,
        hour,
        distance_km,
        manhattan_km,
        landmark_features["pickup_distance_to_JFK"],
        landmark_features["dropoff_distance_to_JFK"],
        landmark_features["pickup_distance_to_LGA"],
        landmark_features["dropoff_distance_to_LGA"],
        landmark_features["pickup_distance_to_EWR"],
        landmark_features["dropoff_distance_to_EWR"],
        landmark_features["pickup_distance_to_Penn_Station"],
        landmark_features["dropoff_distance_to_Penn_Station"],
        landmark_features["pickup_distance_to_Grand_Central"],
        landmark_features["dropoff_distance_to_Grand_Central"],
        landmark_features["pickup_distance_to_Times_Square"],
        landmark_features["dropoff_distance_to_Times_Square"],
        0,  # holiday_flag
        pickup_cluster,
        dropoff_cluster,
        delta_lat,
        delta_lon,
        path_efficiency,
        pickup_vs_dropoff_to_JFK,
        0,  # pickup_cluster_avg_fare
        int(landmark_features["pickup_distance_to_JFK"] < 1.0),
        int(landmark_features["dropoff_distance_to_JFK"] < 1.0),
    ]

    return feature_vector


from fastapi import HTTPException
from pydantic import BaseModel


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
        raise HTTPException(status_code=500, detail="モデルが読み込まれていません")

    try:
        features = generate_features(request.dict())
        prediction = model.predict([features])[0]
        return {"predicted_fare": round(float(prediction), 2)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
