"""
Unit Tests for Geospatial Utilities (Member B Lead)
"""

import pytest
import pandas as pd
from src.utils.geo_utils import (
    haversine_distance,
    filter_by_bbox,
    generate_city_grid,
    match_nearest_station
)


def test_haversine_distance_known_points():
    # Distance between Connaught Place (28.6315, 77.2167) and India Gate (28.6129, 77.2295) ~ 2.4 km
    dist = haversine_distance(28.6315, 77.2167, 28.6129, 77.2295)
    assert 2.0 <= dist <= 3.0


def test_filter_by_bbox():
    df = pd.DataFrame([
        {"station_id": "IN_DELHI", "latitude": 28.61, "longitude": 77.20},
        {"station_id": "IN_MUMBAI", "latitude": 19.07, "longitude": 72.87},
        {"station_id": "IN_HYDERABAD", "latitude": 17.38, "longitude": 78.48}
    ])
    delhi_bbox = [76.84, 28.38, 77.38, 28.88]
    filtered = filter_by_bbox(df, delhi_bbox)
    assert len(filtered) == 1
    assert filtered.iloc[0]["station_id"] == "IN_DELHI"


def test_generate_city_grid():
    delhi_bbox = [77.10, 28.50, 77.20, 28.60]
    grid_df = generate_city_grid(delhi_bbox, grid_size_km=5.0)
    assert isinstance(grid_df, pd.DataFrame)
    assert len(grid_df) > 0
    assert "cell_id" in grid_df.columns
    assert "latitude" in grid_df.columns
    assert "longitude" in grid_df.columns


def test_match_nearest_station():
    stations_df = pd.DataFrame([
        {"station_id": "STN_NORTH", "latitude": 28.70, "longitude": 77.20},
        {"station_id": "STN_SOUTH", "latitude": 28.50, "longitude": 77.20}
    ])
    nearest_id, dist = match_nearest_station(28.71, 77.20, stations_df)
    assert nearest_id == "STN_NORTH"
    assert dist < 5.0
