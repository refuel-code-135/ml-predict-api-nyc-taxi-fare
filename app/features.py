from datetime import datetime

import numpy as np


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius (km)
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


landmarks = {
    "JFK": (40.6413, -73.7781),
    "LGA": (40.7769, -73.8740),
    "EWR": (40.6895, -74.1745),
    "Penn_Station": (40.7506, -73.9935),
    "Grand_Central": (40.7527, -73.9772),
    "Times_Square": (40.7580, -73.9855),
}


def generate_features(data: dict, pickup_kmeans, dropoff_kmeans):
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
        0,
        pickup_cluster,
        dropoff_cluster,
        delta_lat,
        delta_lon,
        path_efficiency,
        pickup_vs_dropoff_to_JFK,
        0,
        int(landmark_features["pickup_distance_to_JFK"] < 1.0),
        int(landmark_features["dropoff_distance_to_JFK"] < 1.0),
    ]

    return feature_vector
