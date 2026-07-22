"""
Unit Tests for Pydantic Data Schemas (Week 1-2 Milestone Verification)
"""

import pytest
from datetime import datetime, timezone
import pandas as pd

from config.schemas import (
    GroundReading,
    WeatherReading,
    SatelliteAODReading,
    UnifiedAQIRecord,
    records_to_dataframe
)


def test_ground_reading_schema_valid():
    reading = GroundReading(
        timestamp=datetime.now(timezone.utc),
        station_id="STN_001",
        station_name="Central Monitoring Station",
        latitude=28.61,
        longitude=77.20,
        city="Delhi",
        pm25=120.5,
        pm10=200.0,
        no2=45.0
    )
    assert reading.station_id == "STN_001"
    assert reading.pm25 == 120.5
    assert reading.source == "CPCB"


def test_ground_reading_invalid_coords():
    with pytest.raises(ValueError):
        GroundReading(
            timestamp=datetime.now(timezone.utc),
            station_id="STN_INVALID",
            latitude=120.0,  # Invalid lat > 90
            longitude=77.20
        )


def test_weather_reading_schema_valid():
    weather = WeatherReading(
        timestamp=datetime.now(timezone.utc),
        latitude=28.61,
        longitude=77.20,
        temp=25.4,
        humidity=60.0,
        wind_speed=4.5,
        wind_deg=180.0,
        pressure=1013.25
    )
    assert weather.temp == 25.4
    assert weather.humidity == 60.0


def test_satellite_aod_schema_valid():
    aod = SatelliteAODReading(
        timestamp=datetime.now(timezone.utc),
        latitude=28.61,
        longitude=77.20,
        aod_550=0.65,
        satellite_name="MODIS",
        quality_flag=3
    )
    assert aod.aod_550 == 0.65


def test_records_to_dataframe_conversion():
    readings = [
        GroundReading(
            timestamp=datetime.now(timezone.utc),
            station_id="STN_A",
            latitude=28.61,
            longitude=77.20,
            pm25=100.0
        ),
        GroundReading(
            timestamp=datetime.now(timezone.utc),
            station_id="STN_B",
            latitude=28.65,
            longitude=77.25,
            pm25=110.0
        )
    ]
    df = records_to_dataframe(readings)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "pm25" in df.columns
    assert df.loc[0, "station_id"] == "STN_A"
