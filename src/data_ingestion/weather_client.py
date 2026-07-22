"""
OpenWeatherMap Air Pollution & Weather Client (Member A Lead)
Retrieves temperature, humidity, wind speed/direction, pressure, and weather variables.
"""

import requests
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import pandas as pd

from config.schemas import WeatherReading, records_to_dataframe

logger = logging.getLogger(__name__)


class WeatherClient:
    """Client for OpenWeatherMap Weather & Air Pollution API."""

    BASE_URL = "https://api.openweathermap.org/data/2.5"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def fetch_current_weather(self, lat: float, lon: float) -> WeatherReading:
        """Fetch current weather variables for given lat/lon coordinates."""
        if not self.api_key:
            return self._generate_fallback(lat, lon)

        url = f"{self.BASE_URL}/weather"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "metric"
        }

        try:
            res = requests.get(url, params=params, timeout=10)
            if res.status_code == 200:
                data = res.json()
                main = data.get("main", {})
                wind = data.get("wind", {})
                rain = data.get("rain", {}).get("1h", 0.0)

                return WeatherReading(
                    timestamp=datetime.now(timezone.utc),
                    latitude=lat,
                    longitude=lon,
                    temp=main.get("temp"),
                    humidity=main.get("humidity"),
                    wind_speed=wind.get("speed"),
                    wind_deg=wind.get("deg"),
                    pressure=main.get("pressure"),
                    rainfall=rain,
                    source="OpenWeatherMap"
                )
            else:
                return self._generate_fallback(lat, lon)
        except Exception as e:
            logger.error(f"Weather API query failed: {e}. Returning fallback weather.")
            return self._generate_fallback(lat, lon)

    def _generate_fallback(self, lat: float, lon: float) -> WeatherReading:
        return WeatherReading(
            timestamp=datetime.now(timezone.utc),
            latitude=lat,
            longitude=lon,
            temp=28.5,
            humidity=55.0,
            wind_speed=3.2,
            wind_deg=220.0,
            pressure=1012.0,
            rainfall=0.0,
            source="OpenWeather_Fallback"
        )
