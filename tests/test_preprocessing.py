"""
Unit & Integration Tests for Phase 2 Preprocessing & ETL Pipeline (Weeks 3-5)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from src.data_preprocessing import GroundDataCleaner, SatelliteAODPreprocessor
from src.pipeline import MasterETLPipeline


def test_ground_cleaner_iqr_outlier_filtering():
    cleaner = GroundDataCleaner(iqr_multiplier=1.5)
    df = pd.DataFrame([
        {"station_id": "STN_1", "pm25": 50.0, "pm10": 100.0},
        {"station_id": "STN_1", "pm25": 55.0, "pm10": 105.0},
        {"station_id": "STN_1", "pm25": 52.0, "pm10": 102.0},
        {"station_id": "STN_1", "pm25": 9999.0, "pm10": 5000.0}  # Extreme anomaly
    ])
    clean_df = cleaner.filter_outliers_iqr(df)
    assert pd.isna(clean_df.loc[3, "pm25"])
    assert pd.isna(clean_df.loc[3, "pm10"])


def test_ground_cleaner_hourly_resampling():
    cleaner = GroundDataCleaner()
    dates = pd.date_range("2026-01-01 10:00", "2026-01-01 10:45", freq="15min")
    df = pd.DataFrame({
        "timestamp": dates,
        "station_id": "STN_1",
        "pm25": [100.0, 110.0, 120.0, 130.0],
        "latitude": 28.61,
        "longitude": 77.20
    })
    resampled = cleaner.resample_hourly(df)
    assert len(resampled) == 1
    assert resampled.iloc[0]["pm25"] == 115.0  # Mean of 100, 110, 120, 130


def test_ground_cleaner_knn_imputation():
    cleaner = GroundDataCleaner(knn_neighbors=2)
    df = pd.DataFrame([
        {"station_id": "STN_1", "pm25": 100.0, "pm10": 180.0, "temp": 25.0},
        {"station_id": "STN_1", "pm25": 110.0, "pm10": np.nan, "temp": 26.0},
        {"station_id": "STN_1", "pm25": 120.0, "pm10": 200.0, "temp": np.nan}
    ])
    imp_df = cleaner.impute_missing_values(df)
    assert not imp_df["pm10"].isna().any()
    assert not imp_df["temp"].isna().any()


def test_satellite_preprocessor_cloud_and_quality_filter():
    preprocessor = SatelliteAODPreprocessor(max_cloud_fraction=0.30, min_quality_flag=2)
    sat_df = pd.DataFrame([
        {"aod_550": 0.55, "cloud_fraction": 0.10, "quality_flag": 3},  # Keep
        {"aod_550": 0.85, "cloud_fraction": 0.50, "quality_flag": 3},  # Cloud filter drop
        {"aod_550": 0.40, "cloud_fraction": 0.05, "quality_flag": 1}   # Quality drop
    ])
    filtered = preprocessor.filter_quality_and_clouds(sat_df)
    assert len(filtered) == 1
    assert filtered.iloc[0]["aod_550"] == 0.55


def test_satellite_spatial_idw_interpolation():
    preprocessor = SatelliteAODPreprocessor()
    grid_df = pd.DataFrame([
        {"latitude": 28.61, "longitude": 77.20},
        {"latitude": 28.65, "longitude": 77.25}
    ])
    known_sat = pd.DataFrame([
        {"latitude": 28.60, "longitude": 77.20, "aod_550": 0.60},
        {"latitude": 28.70, "longitude": 77.30, "aod_550": 0.80}
    ])
    interp_grid = preprocessor.interpolate_spatial_aod_gaps(grid_df, known_sat)
    assert "aod_550" in interp_grid.columns
    assert 0.50 <= interp_grid.iloc[0]["aod_550"] <= 0.90


def test_master_etl_pipeline_execution(tmp_path):
    pipeline = MasterETLPipeline(output_dir=str(tmp_path))
    fused_df = pipeline.run_pipeline(city="Delhi", start_date="2026-01-01", end_date="2026-01-02")
    assert isinstance(fused_df, pd.DataFrame)
    assert len(fused_df) > 0
    assert "pm25" in fused_df.columns
    assert "aod_550" in fused_df.columns
    assert "aqi" in fused_df.columns

    # Verify Parquet & CSV file outputs
    assert (tmp_path / "fused_delhi_aqi.parquet").exists()
    assert (tmp_path / "fused_delhi_aqi.csv").exists()
