"""
Unit Tests for Phase 3 Feature Engineering & EDA Modules (Weeks 6-7 Milestone)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from src.features import TimeSeriesFeatureExtractor, SpatialFeatureExtractor, FeaturePipeline
from src.eda import ExploratoryAnalyzer


def test_cyclical_time_features():
    extractor = TimeSeriesFeatureExtractor()
    dates = pd.date_range("2026-01-01 00:00", "2026-01-01 23:00", freq="h")
    df = pd.DataFrame({"timestamp": dates})
    df_cyc = extractor.add_cyclical_time_features(df)

    assert "hour_sin" in df_cyc.columns
    assert "hour_cos" in df_cyc.columns
    assert "day_sin" in df_cyc.columns
    assert "day_cos" in df_cyc.columns
    assert "month_sin" in df_cyc.columns
    assert "month_cos" in df_cyc.columns

    # Verify sine & cosine bounds [-1.0, 1.0]
    assert df_cyc["hour_sin"].min() >= -1.0 and df_cyc["hour_sin"].max() <= 1.0
    assert df_cyc["hour_cos"].min() >= -1.0 and df_cyc["hour_cos"].max() <= 1.0


def test_lag_features_creation():
    extractor = TimeSeriesFeatureExtractor(lags=[1, 3, 24])
    dates = pd.date_range("2026-01-01 00:00", "2026-01-02 12:00", freq="h")
    df = pd.DataFrame({
        "timestamp": dates,
        "station_id": "STN_01",
        "pm25": np.arange(100.0, 100.0 + len(dates))
    })

    df_lag = extractor.add_lag_features(df, target_cols=["pm25"])
    assert "pm25_lag_1h" in df_lag.columns
    assert "pm25_lag_3h" in df_lag.columns
    assert "pm25_lag_24h" in df_lag.columns

    # Check lag value shift
    assert pd.isna(df_lag.iloc[0]["pm25_lag_1h"])
    assert df_lag.iloc[1]["pm25_lag_1h"] == 100.0
    assert df_lag.iloc[3]["pm25_lag_3h"] == 100.0


def test_rolling_features_creation():
    extractor = TimeSeriesFeatureExtractor(rolling_windows=[6, 24])
    dates = pd.date_range("2026-01-01 00:00", "2026-01-02 12:00", freq="h")
    df = pd.DataFrame({
        "timestamp": dates,
        "station_id": "STN_01",
        "pm25": [100.0] * len(dates)
    })

    df_roll = extractor.add_rolling_features(df, target_cols=["pm25"])
    assert "pm25_roll_6h_mean" in df_roll.columns
    assert "pm25_roll_24h_std" in df_roll.columns
    assert "pm25_roll_6h_min" in df_roll.columns
    assert "pm25_roll_6h_max" in df_roll.columns

    # Mean of constant values should equal constant value
    assert df_roll.iloc[10]["pm25_roll_6h_mean"] == 100.0
    assert df_roll.iloc[10]["pm25_roll_6h_std"] == 0.0


def test_wind_vector_decomposition():
    extractor = SpatialFeatureExtractor()
    df = pd.DataFrame([
        {"wind_speed": 10.0, "wind_deg": 0.0},     # North wind -> V = -10, U = 0
        {"wind_speed": 10.0, "wind_deg": 90.0},    # East wind -> U = -10, V = 0
        {"wind_speed": 10.0, "wind_deg": 180.0},   # South wind -> V = 10, U = 0
        {"wind_speed": 10.0, "wind_deg": 270.0}    # West wind -> U = 10, V = 0
    ])

    df_wind = extractor.compute_wind_vectors(df)
    assert "wind_u" in df_wind.columns
    assert "wind_v" in df_wind.columns

    assert np.isclose(df_wind.iloc[0]["wind_u"], 0.0, atol=1e-3)
    assert np.isclose(df_wind.iloc[0]["wind_v"], -10.0, atol=1e-3)

    assert np.isclose(df_wind.iloc[1]["wind_u"], -10.0, atol=1e-3)
    assert np.isclose(df_wind.iloc[1]["wind_v"], 0.0, atol=1e-3)


def test_satellite_calibration_ratio():
    extractor = SpatialFeatureExtractor()
    df = pd.DataFrame([
        {"aod_550": 0.50, "pm25": 100.0},
        {"aod_550": 1.00, "pm25": 50.0}
    ])

    df_ratio = extractor.compute_satellite_calibration_ratio(df)
    assert "aod_pm25_ratio" in df_ratio.columns
    assert np.isclose(df_ratio.iloc[0]["aod_pm25_ratio"], 5.0, atol=1e-2)  # (0.50 * 1000) / 100 = 5.0
    assert np.isclose(df_ratio.iloc[1]["aod_pm25_ratio"], 20.0, atol=1e-2) # (1.00 * 1000) / 50 = 20.0


def test_feature_pipeline_integration():
    pipeline = FeaturePipeline()
    dates = pd.date_range("2026-01-01 00:00", "2026-01-02 23:00", freq="h")
    df = pd.DataFrame({
        "timestamp": dates,
        "station_id": "DEL_01",
        "latitude": 28.61,
        "longitude": 77.20,
        "pm25": np.random.uniform(50, 150, size=len(dates)),
        "pm10": np.random.uniform(100, 250, size=len(dates)),
        "no2": np.random.uniform(20, 60, size=len(dates)),
        "temp": np.random.uniform(20, 35, size=len(dates)),
        "humidity": np.random.uniform(40, 80, size=len(dates)),
        "wind_speed": np.random.uniform(1, 5, size=len(dates)),
        "wind_deg": np.random.uniform(0, 360, size=len(dates)),
        "aod_550": np.random.uniform(0.3, 0.9, size=len(dates)),
        "aqi": np.random.uniform(100, 300, size=len(dates))
    })

    feat_df = pipeline.transform(df)
    assert isinstance(feat_df, pd.DataFrame)
    assert len(feat_df) == len(df)
    assert "hour_sin" in feat_df.columns
    assert "pm25_lag_1h" in feat_df.columns
    assert "pm25_roll_6h_mean" in feat_df.columns
    assert "wind_u" in feat_df.columns
    assert "aod_pm25_ratio" in feat_df.columns
    assert "road_density_index" in feat_df.columns


def test_exploratory_analyzer():
    analyzer = ExploratoryAnalyzer()
    df = pd.DataFrame({
        "station_id": ["STN_A"] * 50 + ["STN_B"] * 50,
        "pm25": np.random.uniform(50, 150, 100),
        "pm10": np.random.uniform(100, 250, 100),
        "aod_550": np.random.uniform(0.3, 0.9, 100),
        "aqi": np.random.uniform(100, 300, 100)
    })

    corr_df = analyzer.compute_correlation_matrix(df)
    assert isinstance(corr_df, pd.DataFrame)
    assert "pm25" in corr_df.columns

    spatial_summary = analyzer.compute_spatial_summary(df)
    assert isinstance(spatial_summary, pd.DataFrame)
    assert len(spatial_summary) == 2  # STN_A and STN_B
