"""
Master Feature Pipeline Module for Vayusight
Integrates temporal lags, rolling statistics, cyclical encodings, wind vectors,
and spatial calibration ratios into a single production-ready feature transformer.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Optional, Tuple

from .time_features import TimeSeriesFeatureExtractor
from .spatial_features import SpatialFeatureExtractor

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Master pipeline executing full feature engineering transformation on clean fused data."""

    def __init__(
        self,
        lags: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
        target_cols: Optional[List[str]] = None
    ):
        self.time_extractor = TimeSeriesFeatureExtractor(lags=lags, rolling_windows=rolling_windows)
        self.spatial_extractor = SpatialFeatureExtractor()
        self.target_cols = target_cols if target_cols is not None else ["pm25", "pm10", "no2", "temp", "humidity", "aod_550"]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute full feature transformation pipeline:
        1. Add cyclical time features (hour_sin/cos, day_sin/cos, month_sin/cos)
        2. Generate temporal lag features per station
        3. Compute rolling window statistics (mean, std, min, max)
        4. Calculate wind vector components (U, V)
        5. Compute satellite AOD / PM2.5 calibration ratio
        6. Extract spatial land-use & road density covariates
        """
        if df.empty:
            logger.warning("Input DataFrame is empty. Returning empty DataFrame.")
            return df

        logger.info(f"Starting feature engineering on DataFrame with {len(df)} rows...")
        df_feat = df.copy()

        # Step 1: Cyclical Time Features
        df_feat = self.time_extractor.add_cyclical_time_features(df_feat)

        # Step 2: Temporal Lag Features
        df_feat = self.time_extractor.add_lag_features(df_feat, target_cols=self.target_cols)

        # Step 3: Rolling Window Statistics
        df_feat = self.time_extractor.add_rolling_features(df_feat, target_cols=self.target_cols)

        # Step 4: Wind Vector Decomposition
        df_feat = self.spatial_extractor.compute_wind_vectors(df_feat)

        # Step 5: Satellite AOD Calibration Ratio
        df_feat = self.spatial_extractor.compute_satellite_calibration_ratio(df_feat)

        # Step 6: OSM Spatial Land-Use Covariates
        df_feat = self.spatial_extractor.extract_osm_land_covariates(df_feat)

        # Fill any initial lag NaNs using forward-fill then 0 to ensure clean tabular input for ML
        feat_cols = [c for c in df_feat.columns if "_lag_" in c or "_roll_" in c]
        if "station_id" in df_feat.columns:
            df_feat[feat_cols] = df_feat.groupby("station_id")[feat_cols].bfill()
        df_feat[feat_cols] = df_feat[feat_cols].fillna(0.0)

        logger.info(f"Feature engineering complete! Generated {len(df_feat.columns)} columns.")
        return df_feat


if __name__ == "__main__":
    from src.pipeline.run_etl import MasterETLPipeline

    etl = MasterETLPipeline()
    fused_df = etl.run_pipeline(city="Delhi")

    pipeline = FeaturePipeline()
    feat_df = pipeline.transform(fused_df)

    print(f"Engineered Dataset Shape: {feat_df.shape}")
    print("New Feature Columns:", [c for c in feat_df.columns if "_lag_" in c or "_roll_" in c or "wind_" in c][:10])
