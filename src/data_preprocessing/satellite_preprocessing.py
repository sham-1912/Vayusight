"""
Satellite AOD Preprocessing & Spatial Alignment Pipeline (Member B Lead - Week 4)
Handles quality flag filtering, cloud-masking, spatial nearest-neighbour matching
between satellite AOD grid pixels and ground stations, and spatial gap filling.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Tuple, Optional
from src.utils.geo_utils import haversine_distance

logger = logging.getLogger(__name__)


class SatelliteAODPreprocessor:
    """Preprocessor for MODIS and Sentinel-5P satellite AOD measurements."""

    def __init__(self, max_cloud_fraction: float = 0.30, min_quality_flag: int = 2):
        self.max_cloud_fraction = max_cloud_fraction
        self.min_quality_flag = min_quality_flag

    def filter_quality_and_clouds(self, df_sat: pd.DataFrame) -> pd.DataFrame:
        """
        Filter satellite records by quality flag and cloud fraction threshold.
        """
        if df_sat.empty:
            return df_sat

        df_filtered = df_sat.copy()
        
        # Cloud fraction filter
        if 'cloud_fraction' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['cloud_fraction'] <= self.max_cloud_fraction]

        # Quality flag filter
        if 'quality_flag' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['quality_flag'] >= self.min_quality_flag]

        # AOD physical range check (0.0 <= AOD <= 5.0)
        if 'aod_550' in df_filtered.columns:
            df_filtered = df_filtered[(df_filtered['aod_550'] >= 0.0) & (df_filtered['aod_550'] <= 5.0)]

        logger.info(f"Filtered satellite dataset from {len(df_sat)} to {len(df_filtered)} valid pixels")
        return df_filtered.reset_index(drop=True)

    def align_satellite_to_ground_stations(
        self,
        ground_df: pd.DataFrame,
        sat_df: pd.DataFrame,
        max_dist_km: float = 10.0
    ) -> pd.DataFrame:
        """
        Spatially match coarse satellite AOD grid points (1-10 km) to ground station coordinates.
        For each station reading at timestamp t, finds nearest satellite AOD reading within max_dist_km.
        """
        if ground_df.empty or sat_df.empty:
            aligned = ground_df.copy()
            aligned['aod_550'] = np.nan
            return aligned

        aligned_records = []
        sat_df = sat_df.copy()
        sat_df['timestamp_dt'] = pd.to_datetime(sat_df['timestamp'], utc=True)

        for idx, row in ground_df.iterrows():
            st_lat, st_lon = row['latitude'], row['longitude']
            st_ts = pd.to_datetime(row['timestamp'], utc=True)

            # Filter satellite data within +- 24 hours of station reading
            time_mask = (sat_df['timestamp_dt'] >= st_ts - pd.Timedelta(hours=24)) & \
                        (sat_df['timestamp_dt'] <= st_ts + pd.Timedelta(hours=24))
            sat_window = sat_df[time_mask]

            best_aod = np.nan
            min_dist = float('inf')

            if not sat_window.empty:
                for s_idx, s_row in sat_window.iterrows():
                    dist = haversine_distance(st_lat, st_lon, s_row['latitude'], s_row['longitude'])
                    if dist <= max_dist_km and dist < min_dist:
                        min_dist = dist
                        best_aod = s_row['aod_550']

            row_dict = row.to_dict()
            row_dict['aod_550'] = best_aod
            aligned_records.append(row_dict)

        return pd.DataFrame(aligned_records)

    def interpolate_spatial_aod_gaps(
        self,
        grid_df: pd.DataFrame,
        known_sat_df: pd.DataFrame,
        power: float = 2.0
    ) -> pd.DataFrame:
        """
        Perform Spatial Inverse Distance Weighting (IDW) interpolation
        to fill missing AOD values across un-monitored spatial grid cells.
        """
        if known_sat_df.empty or 'aod_550' not in known_sat_df.columns:
            grid_out = grid_df.copy()
            grid_out['aod_550'] = 0.50
            return grid_out

        valid_sat = known_sat_df.dropna(subset=['aod_550'])
        if valid_sat.empty:
            grid_out = grid_df.copy()
            grid_out['aod_550'] = 0.50
            return grid_out

        grid_out = grid_df.copy()
        interpolated_aod = []

        for idx, row in grid_out.iterrows():
            target_lat, target_lon = row['latitude'], row['longitude']
            
            dists = np.array([
                haversine_distance(target_lat, target_lon, s_lat, s_lon)
                for s_lat, s_lon in zip(valid_sat['latitude'], valid_sat['longitude'])
            ])

            # Exact location match
            if np.min(dists) < 1e-4:
                exact_val = valid_sat.iloc[np.argmin(dists)]['aod_550']
                interpolated_aod.append(exact_val)
            else:
                weights = 1.0 / (dists ** power)
                idw_val = np.sum(weights * valid_sat['aod_550'].values) / np.sum(weights)
                interpolated_aod.append(round(float(idw_val), 3))

        grid_out['aod_550'] = interpolated_aod
        return grid_out
