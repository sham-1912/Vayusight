"""
AQI-Fusion Data Schemas
Defines Pydantic v2 data models for uniform spatio-temporal alignment across
ground station readings, meteorological variables, and satellite AOD measurements.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
import pandas as pd


class GroundReading(BaseModel):
    """Ground station pollutant measurement model (CPCB / OpenAQ / WAQI)."""
    timestamp: datetime
    station_id: str
    station_name: Optional[str] = None
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    city: Optional[str] = None
    pm25: Optional[float] = Field(None, ge=0.0, description="PM2.5 concentration in ug/m3")
    pm10: Optional[float] = Field(None, ge=0.0, description="PM10 concentration in ug/m3")
    no2: Optional[float] = Field(None, ge=0.0, description="NO2 concentration in ug/m3")
    so2: Optional[float] = Field(None, ge=0.0, description="SO2 concentration in ug/m3")
    co: Optional[float] = Field(None, ge=0.0, description="CO concentration in mg/m3 or ug/m3")
    o3: Optional[float] = Field(None, ge=0.0, description="O3 concentration in ug/m3")
    source: str = "CPCB"


class WeatherReading(BaseModel):
    """Meteorological measurement model."""
    timestamp: datetime
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    temp: Optional[float] = Field(None, description="Temperature in Celsius")
    humidity: Optional[float] = Field(None, ge=0.0, le=100.0, description="Relative humidity %")
    wind_speed: Optional[float] = Field(None, ge=0.0, description="Wind speed in m/s")
    wind_deg: Optional[float] = Field(None, ge=0.0, le=360.0, description="Wind direction in degrees")
    pressure: Optional[float] = Field(None, ge=800.0, le=1100.0, description="Atmospheric pressure in hPa")
    rainfall: Optional[float] = Field(0.0, ge=0.0, description="Rainfall in mm")
    source: str = "OpenWeatherMap"


class SatelliteAODReading(BaseModel):
    """Satellite Aerosol Optical Depth (AOD) reading from MODIS / Sentinel-5P."""
    timestamp: datetime
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    aod_550: Optional[float] = Field(None, ge=0.0, le=5.0, description="Aerosol Optical Depth at 550nm")
    satellite_name: str = "MODIS"
    cloud_fraction: Optional[float] = Field(None, ge=0.0, le=1.0)
    quality_flag: int = 1


class UnifiedAQIRecord(BaseModel):
    """
    Fused spatio-temporal record combining ground pollutants, weather variables,
    and satellite AOD into a single feature vector.
    """
    timestamp: datetime
    station_id: str
    latitude: float
    longitude: float
    city: Optional[str] = "Delhi-NCR"
    
    # Pollutants
    pm25: Optional[float] = None
    pm10: Optional[float] = None
    no2: Optional[float] = None
    so2: Optional[float] = None
    co: Optional[float] = None
    o3: Optional[float] = None
    
    # Weather
    temp: Optional[float] = None
    humidity: Optional[float] = None
    wind_speed: Optional[float] = None
    wind_deg: Optional[float] = None
    pressure: Optional[float] = None
    
    # Satellite AOD
    aod_550: Optional[float] = None
    
    # Computed Target
    aqi: Optional[float] = None

    @field_validator('latitude', 'longitude')
    def validate_coords(cls, v):
        if pd.isna(v):
            raise ValueError("Coordinates cannot be NaN")
        return v


def records_to_dataframe(records: List[BaseModel]) -> pd.DataFrame:
    """Convert a list of Pydantic schema instances to a Pandas DataFrame."""
    if not records:
        return pd.DataFrame()
    return pd.DataFrame([r.model_dump() for r in records])
