from datetime import datetime

import numpy as np

NEARBY_THRESHOLD_KM = 1.0

# Coordinates of major NYC landmarks
_landmarks = {
    "JFK": (40.6413, -73.7781),
    "LGA": (40.7769, -73.8740),
    "EWR": (40.6895, -74.1745),
    "Penn_Station": (40.7506, -73.9935),
    "Grand_Central": (40.7527, -73.9772),
    "Times_Square": (40.7580, -73.9855),
}


def _compute_haversine_distance(from_lat, from_lon, to_lat, to_lon):
    R = 6371  # Earth radius (km)
    from_lat, from_lon, to_lat, to_lon = map(
        np.radians, [from_lat, from_lon, to_lat, to_lon]
    )
    dlat = to_lat - from_lat
    dlon = to_lon - from_lon
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(from_lat) * np.cos(to_lat) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def generate_features(data: dict, pickup_kmeans, dropoff_kmeans):
    pickup_dt = datetime.fromisoformat(data["pickup_datetime"])
    year = pickup_dt.year
    month = pickup_dt.month
    weekday = pickup_dt.weekday()
    hour = pickup_dt.hour

    # Holiday flag: 1 if it's a major U.S. holiday, else 0
    holiday_flag = int(
        (month == 1 and pickup_dt.day == 1)  # New Year's Day
        or (month == 7 and pickup_dt.day == 4)  # Independence Day
        or (month == 12 and pickup_dt.day == 25)  # Christmas Day
        or (month == 12 and pickup_dt.day == 31)  # New Year's Eve
    )

    delta_lat = abs(data["pickup_latitude"] - data["dropoff_latitude"])
    delta_lon = abs(data["pickup_longitude"] - data["dropoff_longitude"])

    straight_km = _compute_haversine_distance(
        data["pickup_latitude"],
        data["pickup_longitude"],
        data["dropoff_latitude"],
        data["dropoff_longitude"],
    )

    manhattan_km = _compute_haversine_distance(
        data["pickup_latitude"],
        data["pickup_longitude"],
        data["pickup_latitude"],
        data["dropoff_longitude"],
    ) + _compute_haversine_distance(
        data["pickup_latitude"],
        data["dropoff_longitude"],
        data["dropoff_latitude"],
        data["dropoff_longitude"],
    )

    if manhattan_km != 0:
        path_efficiency = straight_km / manhattan_km
    else:
        # Avoid division by zero
        # This case is unlikely under normal conditions
        path_efficiency = 1.0

    landmark_features = {}
    for name, (lat, lon) in _landmarks.items():
        landmark_features[f"pickup_distance_to_{name}"] = _compute_haversine_distance(
            data["pickup_latitude"], data["pickup_longitude"], lat, lon
        )
        landmark_features[f"dropoff_distance_to_{name}"] = _compute_haversine_distance(
            data["dropoff_latitude"], data["dropoff_longitude"], lat, lon
        )

    pickup_cluster = pickup_kmeans.predict(
        [[data["pickup_latitude"], data["pickup_longitude"]]]
    )[0]
    dropoff_cluster = dropoff_kmeans.predict(
        [[data["dropoff_latitude"], data["dropoff_longitude"]]]
    )[0]

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
        straight_km,
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
        holiday_flag,
        pickup_cluster,
        dropoff_cluster,
        delta_lat,
        delta_lon,
        path_efficiency,
        pickup_vs_dropoff_to_JFK,
        0,
        int(landmark_features["pickup_distance_to_JFK"] < NEARBY_THRESHOLD_KM),
        int(landmark_features["dropoff_distance_to_JFK"] < NEARBY_THRESHOLD_KM),
    ]

    return feature_vector
