"""
World Air Quality Index (WAQI / aqicn) API Client (Member A Lead)
Retrieves live AQI feed for stations worldwide using simple token authentication.
"""

import requests
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import pandas as pd

from config.schemas import GroundReading, records_to_dataframe

logger = logging.getLogger(__name__)


class WAQIClient:
    """Client for WAQI (World Air Quality Index / aqicn.org) API."""

    BASE_URL = "https://api.waqi.info"

    def __init__(self, token: str = "demo"):
        self.token = token

    def fetch_city_feed(self, city: str = "delhi") -> List[GroundReading]:
        """Fetch current feed for a specific city keyword."""
        url = f"{self.BASE_URL}/feed/{city}/?token={self.token}"

        try:
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                data = res.json()
                if data.get("status") == "ok":
                    return [self._parse_station_data(data.get("data", {}), city)]
            logger.warning(f"WAQI API feed request returned status {res.status_code}. Using fallback.")
            return self._generate_fallback(city)
        except Exception as e:
            logger.error(f"WAQI query error: {e}. Using fallback.")
            return self._generate_fallback(city)

    def _parse_station_data(self, data: Dict[str, Any], city: str) -> GroundReading:
        geo = data.get("city", {}).get("geo", [0.0, 0.0])
        lat, lon = (geo[0], geo[1]) if len(geo) >= 2 else (28.61, 77.20)
        station_name = data.get("city", {}).get("name", f"{city.capitalize()} Central")
        station_id = str(data.get("idx", "WAQI_001"))

        iaqi = data.get("iaqi", {})
        pm25 = iaqi.get("pm25", {}).get("v")
        pm10 = iaqi.get("pm10", {}).get("v")
        no2 = iaqi.get("no2", {}).get("v")
        so2 = iaqi.get("so2", {}).get("v")
        co = iaqi.get("co", {}).get("v")
        o3 = iaqi.get("o3", {}).get("v")

        return GroundReading(
            timestamp=datetime.now(timezone.utc),
            station_id=station_id,
            station_name=station_name,
            latitude=float(lat),
            longitude=float(lon),
            city=city,
            pm25=float(pm25) if pm25 is not None else None,
            pm10=float(pm10) if pm10 is not None else None,
            no2=float(no2) if no2 is not None else None,
            so2=float(so2) if so2 is not None else None,
            co=float(co) if co is not None else None,
            o3=float(o3) if o3 is not None else None,
            source="WAQI"
        )

    def _generate_fallback(self, city: str) -> List[GroundReading]:
        return [
            GroundReading(
                timestamp=datetime.now(timezone.utc),
                station_id="WAQI_DEMO_01",
                station_name=f"{city.capitalize()} Reference Monitor",
                latitude=28.6139 if "delhi" in city.lower() else 17.3850,
                longitude=77.2090 if "delhi" in city.lower() else 78.4867,
                city=city,
                pm25=115.0,
                pm10=195.0,
                no2=34.0,
                so2=9.0,
                co=0.9,
                o3=22.0,
                source="WAQI_Fallback"
            )
        ]
