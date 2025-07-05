# ml-predict-api-nyc-taxi-fare

## Overview

A FastAPI-based machine learning inference API that predicts New York City taxi fares based on pickup/dropoff locations, time, and passenger count.  
The model was trained using LightGBM and deployed in a Dockerized environment with Nginx as a reverse proxy.


## Usage

```bash
docker compose up --build
```

## Example Prediction

```bash
# Request
curl -X POST http://localhost:9700/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pickup_datetime": "2025-08-01T14:30:00",
    "pickup_latitude": 40.7128,
    "pickup_longitude": -74.0060,
    "dropoff_latitude": 40.730610,
    "dropoff_longitude": -73.935242,
    "passenger_count": 2
  }'
```

```bash
# Response
{
  "predicted_fare": 3.13
}
```

## API document

http://localhost:9700/docs
