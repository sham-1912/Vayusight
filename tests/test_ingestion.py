"""
Unit Tests for Data Ingestion Modules (Member A & B Ingestion Verification)
"""

import pytest
import pandas as pd
from src.data_ingestion import (
    OpenAQClient,
    WAQIClient,
    CPCBClient,
    WeatherClient,
    KaggleDataLoader,
    EarthEngineAODClient
)
from config.schemas import GroundReading, SatelliteAODReading


def test_openaq_client_ingestion():
    client = OpenAQClient()
    readings = client.fetch_latest_measurements(city="Delhi")
    assert len(readings) > 0
    assert isinstance(readings[0], GroundReading)


def test_waqi_client_ingestion():
    client = WAQIClient()
    readings = client.fetch_city_feed(city="delhi")
    assert len(readings) > 0
    assert isinstance(readings[0], GroundReading)


def test_cpcb_client_ingestion():
    client = CPCBClient()
    readings = client.fetch_live_cpcb_data(city="Delhi")
    assert len(readings) > 0
    assert isinstance(readings[0], GroundReading)


def test_weather_client_ingestion():
    client = WeatherClient()
    weather = client.fetch_current_weather(28.6139, 77.2090)
    assert weather.temp is not None
    assert weather.humidity is not None


def test_kaggle_loader_synthetic_fallback():
    loader = KaggleDataLoader()
    df = loader.load_and_clean_csv("non_existent_file.csv")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "pm25" in df.columns
    assert "pm10" in df.columns


def test_gee_aod_client_simulation():
    client = EarthEngineAODClient(initialize_gee=False)
    delhi_bbox = [76.84, 28.38, 77.38, 28.88]
    aod_readings = client.fetch_aod_for_bbox(delhi_bbox, "2026-01-01", "2026-01-02")
    assert len(aod_readings) > 0
    assert isinstance(aod_readings[0], SatelliteAODReading)
    assert 0.0 <= aod_readings[0].aod_550 <= 5.0
