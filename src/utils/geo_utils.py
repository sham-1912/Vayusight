"""
Geospatial Utilities Module for AQI-Fusion (Member B Lead)
Provides bounding box filtering, spatial distance calculations, grid generation,
and spatial station matching utilities.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on the earth in kilometers.
    """
    R = 6371.0  # Earth radius in kilometers

    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2.0) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return float(R * c)


def filter_by_bbox(df: pd.DataFrame, bbox: List[float], lat_col: str = 'latitude', lon_col: str = 'longitude') -> pd.DataFrame:
    """
    Filter DataFrame records to those falling within a spatial bounding box.
    bbox format: [min_lon, min_lat, max_lon, max_lat]
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    mask = (
        (df[lat_col] >= min_lat) & (df[lat_col] <= max_lat) &
        (df[lon_col] >= min_lon) & (df[lon_col] <= max_lon)
    )
    return df[mask].copy()


def generate_city_grid(bbox: List[float], grid_size_km: float = 1.0) -> pd.DataFrame:
    """
    Generate a regular 2D spatial grid (~1 km grid cell centers) for an un-monitored area bounding box.
    bbox format: [min_lon, min_lat, max_lon, max_lat]
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    
    # 1 degree latitude ~ 111 km
    lat_step = grid_size_km / 111.0
    # 1 degree longitude ~ 111 * cos(mean_lat) km
    mean_lat = (min_lat + max_lat) / 2.0
    lon_step = grid_size_km / (111.0 * np.cos(np.radians(mean_lat)))

    lats = np.arange(min_lat, max_lat, lat_step)
    lons = np.arange(min_lon, max_lon, lon_step)

    grid_cells = []
    cell_id = 0
    for lat in lats:
        for lon in lons:
            grid_cells.append({
                'cell_id': f"GRID_{cell_id:05d}",
                'latitude': round(float(lat), 5),
                'longitude': round(float(lon), 5)
            })
            cell_id += 1

    return pd.DataFrame(grid_cells)


def match_nearest_station(lat: float, lon: float, stations_df: pd.DataFrame) -> Tuple[str, float]:
    """
    Find the nearest ground station ID and distance in km for a given (lat, lon) coordinate.
    """
    min_dist = float('inf')
    nearest_id = ""

    for idx, row in stations_df.iterrows():
        dist = haversine_distance(lat, lon, row['latitude'], row['longitude'])
        if dist < min_dist:
            min_dist = dist
            nearest_id = str(row['station_id'])

    return nearest_id, round(min_dist, 3)
