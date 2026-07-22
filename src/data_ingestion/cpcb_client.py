"""
CPCB CAAQMS / data.gov.in Ingestion Module (Member A Lead)
Handles central air quality monitoring station data ingestion and normalization.
"""

import requests
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import pandas as pd

from config.schemas import GroundReading, records_to_dataframe

logger = logging.getLogger(__name__)


class CPCBClient:
    """Client for CPCB CAAQMS / data.gov.in Open Data portal API."""

    DATA_GOV_RESOURCE_URL = "https://api.data.gov.in/resource/3b47a077-380c-4204-b995-65210759f7d3"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def fetch_live_cpcb_data(self, city: str = "Delhi", limit: int = 50) -> List[GroundReading]:
        """
        Fetch station pollutant data from CPCB data portal endpoint.
        """
        if not self.api_key:
            logger.info("No data.gov.in API key provided. Using fallback CPCB dataset generator.")
            return self._generate_fallback(city)

        params = {
            "api-key": self.api_key,
            "format": "json",
            "filters[city]": city,
            "limit": limit
        }

        try:
            res = requests.get(self.DATA_GOV_RESOURCE_URL, params=params, timeout=10)
            if res.status_code == 200:
                records = res.json().get("records", [])
                return self._parse_records(records, city)
            else:
                return self._generate_fallback(city)
        except Exception as e:
            logger.error(f"CPCB API query error: {e}. Returning fallback data.")
            return self._generate_fallback(city)

    def _parse_records(self, records: List[Dict[str, Any]], city: str) -> List[GroundReading]:
        readings_map: Dict[str, Dict[str, Any]] = {}
        now = datetime.now(timezone.utc)

        for rec in records:
            station = rec.get("station", "CPCB_STATION")
            pollutant = rec.get("pollutant_id", "").upper().replace(".", "")
            val_str = rec.get("pollutant_avg", "NA")
            
            try:
                val = float(val_str)
            except (ValueError, TypeError):
                val = None

            if station not in readings_map:
                readings_map[station] = {
                    "station_id": station,
                    "latitude": float(rec.get("latitude", 28.61)),
                    "longitude": float(rec.get("longitude", 77.20)),
                    "pollutants": {}
                }
            if pollutant and val is not None:
                readings_map[station]["pollutants"][pollutant] = val

        readings = []
        for st_id, data in readings_map.items():
            pol = data["pollutants"]
            readings.append(GroundReading(
                timestamp=now,
                station_id=st_id,
                station_name=st_id,
                latitude=data["latitude"],
                longitude=data["longitude"],
                city=city,
                pm25=pol.get("PM25"),
                pm10=pol.get("PM10"),
                no2=pol.get("NO2"),
                so2=pol.get("SO2"),
                co=pol.get("CO"),
                o3=pol.get("O3"),
                source="CPCB_CAAQMS"
            ))
        return readings

    def _generate_fallback(self, city: str) -> List[GroundReading]:
        now = datetime.now(timezone.utc)
        return [
            GroundReading(
                timestamp=now,
                station_id="CPCB_DEL_01",
                station_name="CPCB ITO station",
                latitude=28.6289,
                longitude=77.2405,
                city=city,
                pm25=138.5,
                pm10=225.0,
                no2=41.2,
                so2=14.0,
                co=1.5,
                o3=31.0,
                source="CPCB_Fallback"
            ),
            GroundReading(
                timestamp=now,
                station_id="CPCB_DEL_02",
                station_name="CPCB Dwarka station",
                latitude=28.5921,
                longitude=77.0460,
                city=city,
                pm25=129.0,
                pm10=210.0,
                no2=37.5,
                so2=11.2,
                co=1.1,
                o3=29.0,
                source="CPCB_Fallback"
            )
        ]
