"""
Unit & Integration Tests for Models, Evaluation Engine, and FastAPI App (Phases 4-6)
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from src.models import (
    TabularAQIForecaster,
    PyTorchAQIForecaster,
    SpatialAQIEstimator,
    AQIExplainer,
    HealthRiskCalculator
)
from src.evaluation import ModelEvaluator
from fastapi.testclient import TestClient
from app.main import app


def test_tabular_aqi_forecaster():
    X = pd.DataFrame({
        "pm25": np.random.uniform(50, 150, 100),
        "pm10": np.random.uniform(100, 250, 100),
        "temp": np.random.uniform(20, 35, 100)
    })
    y = pd.Series(X["pm25"] * 1.5 + np.random.normal(0, 5, 100))

    forecaster = TabularAQIForecaster(model_name="random_forest")
    forecaster.fit(X, y)
    preds = forecaster.predict(X)

    assert len(preds) == len(y)
    metrics = forecaster.evaluate(X, y)
    assert metrics["r2"] > 0.70
    assert "rmse" in metrics


def test_pytorch_aqi_forecaster():
    X = np.random.uniform(50, 150, size=(100, 5))
    y = X[:, 0] * 1.5

    py_forecaster = PyTorchAQIForecaster(seq_len=10, epochs=3, batch_size=16)
    py_forecaster.fit(X, y)
    preds = py_forecaster.predict(X)

    assert len(preds) == len(X)
    assert not np.isnan(preds).any()


def test_spatial_aqi_estimator():
    estimator = SpatialAQIEstimator()
    train_df = pd.DataFrame({
        "latitude": [28.61, 28.65, 28.70],
        "longitude": [77.20, 77.25, 77.30],
        "aod_550": [0.60, 0.70, 0.55],
        "temp": [28.0, 29.0, 27.0],
        "aqi": [180.0, 210.0, 160.0]
    })
    estimator.fit(train_df, target_col="aqi")

    grid_df = pd.DataFrame([
        {"cell_id": "CELL_1", "latitude": 28.62, "longitude": 77.22, "aod_550": 0.65},
        {"cell_id": "CELL_2", "latitude": 28.68, "longitude": 77.28, "aod_550": 0.58}
    ])
    grid_preds = estimator.predict_grid(grid_df)
    assert "estimated_aqi" in grid_preds.columns
    assert "aqi_category" in grid_preds.columns
    assert len(grid_preds) == 2


def test_aqi_explainer():
    X = pd.DataFrame({
        "pm25": [120.0, 110.0],
        "wind_speed": [2.5, 4.0],
        "aod_550": [0.70, 0.50]
    })
    y = pd.Series([200.0, 160.0])

    forecaster = TabularAQIForecaster(model_name="random_forest")
    forecaster.fit(X, y)

    explainer = AQIExplainer(forecaster.model, list(X.columns))
    explanation = explainer.generate_narrative_explanation(X.iloc[[0]])
    assert "summary_text" in explanation
    assert "top_features" in explanation


def test_health_risk_calculator():
    calculator = HealthRiskCalculator()
    grid_df = pd.DataFrame([
        {"cell_id": "C1", "pm25": 85.0},
        {"cell_id": "C2", "pm25": 25.0}
    ])

    health_df = calculator.compute_grid_health_risk(grid_df)
    assert "relative_risk_respiratory" in health_df.columns
    assert "estimated_excess_respiratory_events_per_100k" in health_df.columns
    assert health_df.iloc[0]["relative_risk_respiratory"] > health_df.iloc[1]["relative_risk_respiratory"]


def test_model_evaluator():
    evaluator = ModelEvaluator()
    y_true = np.array([100.0, 150.0, 200.0])
    y_pred = np.array([105.0, 145.0, 195.0])

    metrics = evaluator.compute_metrics(y_true, y_pred)
    assert metrics["r2"] > 0.90
    assert metrics["mae"] == 5.0

    ttest_res = evaluator.paired_ttest_models(y_true - y_pred, (y_true - y_pred) * 2)
    assert "p_value" in ttest_res


def test_fastapi_app_endpoints():
    client = TestClient(app)

    # Test Root
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert res_root.json()["system"] == "Vayusight AQI System"

    # Test Forecast Endpoint
    res_forecast = client.post("/predict/forecast", json={
        "station_id": "DEL_01",
        "latitude": 28.6139,
        "longitude": 77.2090,
        "hours_ahead": 24
    })
    assert res_forecast.status_code == 200
    assert "predicted_aqi" in res_forecast.json()

    # Test Spatial Endpoint
    res_spatial = client.post("/predict/spatial", json={
        "city_name": "Delhi-NCR",
        "bbox": [76.84, 28.38, 77.38, 28.88],
        "grid_size_km": 5.0
    })
    assert res_spatial.status_code == 200
    assert res_spatial.json()["grid_cell_count"] > 0
