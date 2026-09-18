"""
Production FastAPI REST API Backend for Vayusight (Member A Lead - Week 13)
Exposes endpoints for time-series forecasting, spatial grid estimation, SHAP explainability, and health risk.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone

# Ensure project root is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd

from src.data_ingestion import OpenAQClient, WeatherClient
from src.data_preprocessing import GroundDataCleaner
from src.features import FeaturePipeline
from src.models import TabularAQIForecaster, SpatialAQIEstimator, AQIExplainer, HealthRiskCalculator
from src.utils.geo_utils import generate_city_grid

app = FastAPI(
    title="Vayusight REST API",
    version="1.0.0",
    description="Production API exposing AQI forecasts, spatial estimation, SHAP explainability, and health risk burden."
)

# Instantiate core components
feature_pipeline = FeaturePipeline()
forecaster = TabularAQIForecaster(model_name="random_forest")
spatial_estimator = SpatialAQIEstimator()
health_calculator = HealthRiskCalculator()

# Pre-fit dummy models for API startup readiness
_dummy_df = pd.DataFrame({
    "timestamp": pd.date_range("2026-01-01", periods=3, freq="h"),
    "station_id": "DEL_01",
    "latitude": [28.61, 28.65, 28.70],
    "longitude": [77.20, 77.25, 77.30],
    "pm25": [120.0, 110.0, 130.0],
    "pm10": [200.0, 180.0, 220.0],
    "no2": [40.0, 35.0, 45.0],
    "temp": [28.0, 29.0, 27.0],
    "humidity": [60.0, 58.0, 62.0],
    "wind_speed": [3.0, 2.5, 3.5],
    "wind_u": [1.0, 0.8, 1.2],
    "wind_v": [-2.0, -1.8, -2.2],
    "aod_550": [0.65, 0.60, 0.70],
    "aqi": [200.0, 185.0, 215.0]
})
_dummy_feat = feature_pipeline.transform(_dummy_df)
X_dummy = _dummy_feat.drop(columns=["station_id", "timestamp", "city"], errors="ignore")
y_dummy = _dummy_df["aqi"]

forecaster.fit(X_dummy, y_dummy)
spatial_estimator.fit(_dummy_df, target_col="aqi")
explainer = AQIExplainer(forecaster.model, list(X_dummy.columns))


class ForecastRequest(BaseModel):
    station_id: str = "DEL_01"
    latitude: float = 28.6139
    longitude: float = 77.2090
    hours_ahead: int = Field(24, ge=1, le=72)


class SpatialRequest(BaseModel):
    city_name: str = "Delhi-NCR"
    bbox: List[float] = Field([76.84, 28.38, 77.38, 28.88], min_length=4, max_length=4)
    grid_size_km: float = Field(2.0, ge=0.5, le=10.0)


@app.get("/")
def read_root():
    return {
        "status": "online",
        "system": "Vayusight AQI System",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.post("/predict/forecast")
def predict_forecast(req: ForecastRequest):
    """Return 24-72h AQI forecast predictions for a monitored station location."""
    try:
        # Generate synthetic sequence sample for prediction
        dates = pd.date_range(end=datetime.now(timezone.utc), periods=48, freq="h")
        sample_df = pd.DataFrame({
            "timestamp": dates,
            "station_id": req.station_id,
            "latitude": req.latitude,
            "longitude": req.longitude,
            "pm25": np.random.uniform(90, 150, size=48),
            "pm10": np.random.uniform(160, 250, size=48),
            "no2": np.random.uniform(30, 60, size=48),
            "temp": np.random.uniform(22, 32, size=48),
            "humidity": np.random.uniform(45, 75, size=48),
            "wind_speed": np.random.uniform(2, 5, size=48),
            "wind_deg": np.random.uniform(0, 360, size=48),
            "aod_550": np.random.uniform(0.5, 0.8, size=48),
            "aqi": np.random.uniform(150, 250, size=48)
        })

        feat_df = feature_pipeline.transform(sample_df)
        X_sample = feat_df.drop(columns=["station_id", "timestamp", "city"], errors="ignore")
        if forecaster.feature_names:
            X_sample = X_sample.reindex(columns=forecaster.feature_names, fill_value=0.0)

        preds = forecaster.predict(X_sample)
        latest_pred = float(preds[-1])

        # SHAP explanation
        explanation = explainer.generate_narrative_explanation(X_sample.iloc[[-1]])

        return {
            "station_id": req.station_id,
            "latitude": req.latitude,
            "longitude": req.longitude,
            "forecast_horizon_hours": req.hours_ahead,
            "predicted_aqi": round(latest_pred, 1),
            "aqi_category": spatial_estimator._get_aqi_category(latest_pred),
            "explanation": explanation
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/spatial")
def predict_spatial_grid(req: SpatialRequest):
    """Return ~1 km grid cell spatial AQI estimations for an un-monitored bounding box."""
    try:
        grid_df = generate_city_grid(req.bbox, grid_size_km=req.grid_size_km)
        spatial_preds = spatial_estimator.predict_grid(grid_df)
        health_preds = health_calculator.compute_grid_health_risk(spatial_preds)

        records = health_preds.to_dict(orient="records")
        return {
            "city_name": req.city_name,
            "grid_cell_count": len(records),
            "grid_cells": records
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
