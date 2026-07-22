"""
OpenAQ REST API v2 Client (Member A Lead)
Fetches live and historical station air quality measurements across Indian cities,
normalizes parameters, and converts to GroundReading schema objects.
"""

import requests
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import pandas as pd

from config.schemas import GroundReading, records_to_dataframe

logger = logging.getLogger(__name__)

class OpenAQClient:
    """Client for OpenAQ REST API v2."""

    BASE_URL = "https://api.openaq.org/v2"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.headers = {}
        if api_key:
            self.headers["X-API-Key"] = api_key

    def fetch_latest_measurements(self, city: str = "Delhi", country: str = "IN", limit: int = 100) -> List[GroundReading]:
        """
        Fetch latest air quality measurements for a city.
        """
        url = f"{self.BASE_URL}/latest"
        params = {
            "country": country,
            "city": city,
            "limit": limit
        }

        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                return self._parse_results(results, city)
            else:
                logger.warning(f"OpenAQ API returned status code {response.status_code}. Using fallback generator.")
                return self._generate_fallback_data(city)
        except Exception as e:
            logger.error(f"Failed to query OpenAQ API: {e}. Generating fallback data for offline testing.")
            return self._generate_fallback_data(city)

    def _parse_results(self, results: List[Dict[str, Any]], city_name: str) -> List[GroundReading]:
        """Parse raw API JSON response into GroundReading objects."""
        readings = []
        for item in results:
            station_id = str(item.get("locationId", item.get("location", "UNKNOWN")))
            station_name = item.get("location", "Unknown Station")
            coords = item.get("coordinates", {})
            lat = coords.get("latitude")
            lon = coords.get("longitude")

            if lat is None or lon is None:
                continue

            measurements = {m["parameter"]: m["value"] for m in item.get("measurements", []) if "parameter" in m}

            # Parse last updated timestamp
            first_m = item.get("measurements", [{}])[0]
            last_updated_str = first_m.get("lastUpdated")
            if last_updated_str:
                ts = datetime.fromisoformat(last_updated_str.replace("Z", "+00:00"))
            else:
                ts = datetime.now(timezone.utc)

            reading = GroundReading(
                timestamp=ts,
                station_id=station_id,
                station_name=station_name,
                latitude=float(lat),
                longitude=float(lon),
                city=city_name,
                pm25=measurements.get("pm25"),
                pm10=measurements.get("pm10"),
                no2=measurements.get("no2"),
                so2=measurements.get("so2"),
                co=measurements.get("co"),
                o3=measurements.get("o3"),
                source="OpenAQ"
            )
            readings.append(reading)
        return readings

    def _generate_fallback_data(self, city_name: str) -> List[GroundReading]:
        """Generate realistic mock station measurements for offline testing."""
        stations = [
            {"id": "IND001", "name": "Anand Vihar", "lat": 28.647, "lon": 77.315, "pm25": 145.2, "pm10": 260.5, "no2": 45.0},
            {"id": "IND002", "name": "RK Puram", "lat": 28.563, "lon": 77.186, "pm25": 120.0, "pm10": 210.0, "no2": 38.5},
            {"id": "IND003", "name": "Punjabi Bagh", "lat": 28.674, "lon": 77.131, "pm25": 132.8, "pm10": 235.1, "no2": 42.1},
        ]

        now = datetime.now(timezone.utc)
        readings = []
        for s in stations:
            readings.append(GroundReading(
                timestamp=now,
                station_id=s["id"],
                station_name=s["name"],
                latitude=s["lat"],
                longitude=s["lon"],
                city=city_name,
                pm25=s["pm25"],
                pm10=s["pm10"],
                no2=s["no2"],
                so2=12.5,
                co=1.2,
                o3=28.4,
                source="OpenAQ_Fallback"
            ))
        return readings


if __name__ == "__main__":
    client = OpenAQClient()
    readings = client.fetch_latest_measurements(city="Delhi")
    df = records_to_dataframe(readings)
    print(f"Fetched {len(df)} OpenAQ records:\n{df[['station_id', 'station_name', 'pm25', 'pm10', 'no2']]}")
