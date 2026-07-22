"""
Ground & Meteorological Data Cleaning Pipeline (Member A Lead - Week 3)
Provides IQR/Z-score outlier detection, hourly temporal resampling,
and multi-tier missing value imputation (Forward-fill + KNN / MICE Imputation).
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Optional
from sklearn.impute import KNNImputer

logger = logging.getLogger(__name__)


class GroundDataCleaner:
    """Cleaner for ground station pollutant and weather measurements."""

    POLLUTANT_BOUNDS = {
        "pm25": (0.0, 999.0),
        "pm10": (0.0, 1500.0),
        "no2": (0.0, 500.0),
        "so2": (0.0, 500.0),
        "co": (0.0, 50.0),
        "o3": (0.0, 500.0),
        "temp": (-10.0, 60.0),
        "humidity": (0.0, 100.0),
        "wind_speed": (0.0, 50.0),
        "pressure": (800.0, 1100.0)
    }

    def __init__(self, iqr_multiplier: float = 3.0, knn_neighbors: int = 5):
        self.iqr_multiplier = iqr_multiplier
        self.knn_neighbors = knn_neighbors

    def filter_outliers_iqr(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Filter out sensor anomalies using Interquartile Range (IQR) and physical domain bounds.
        """
        df_clean = df.copy()
        if columns is None:
            columns = [c for c in self.POLLUTANT_BOUNDS.keys() if c in df_clean.columns]

        for col in columns:
            if col not in df_clean.columns or df_clean[col].dropna().empty:
                continue

            # Physical domain threshold filtering
            min_val, max_val = self.POLLUTANT_BOUNDS.get(col, (0.0, 10000.0))
            df_clean.loc[(df_clean[col] < min_val) | (df_clean[col] > max_val), col] = np.nan

            # Statistical IQR filtering
            q25 = df_clean[col].quantile(0.25)
            q75 = df_clean[col].quantile(0.75)
            iqr = q75 - q25
            lower_bound = max(min_val, q25 - (self.iqr_multiplier * iqr))
            upper_bound = min(max_val, q75 + (self.iqr_multiplier * iqr))

            invalid_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
            num_outliers = invalid_mask.sum()
            if num_outliers > 0:
                logger.info(f"Filtered {num_outliers} outlier entries in column '{col}'")
                df_clean.loc[invalid_mask, col] = np.nan

        return df_clean

    def resample_hourly(self, df: pd.DataFrame, timestamp_col: str = 'timestamp', station_col: str = 'station_id') -> pd.DataFrame:
        """
        Resample irregular station sensor data into strict 1-hour temporal bins.
        """
        if timestamp_col not in df.columns or df.empty:
            return df

        df_sorted = df.copy()
        df_sorted[timestamp_col] = pd.to_datetime(df_sorted[timestamp_col], utc=True)
        
        numeric_cols = df_sorted.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = [c for c in df_sorted.columns if c not in numeric_cols and c != timestamp_col]

        resampled_groups = []
        if station_col in df_sorted.columns:
            for station_id, group in df_sorted.groupby(station_col):
                group_indexed = group.set_index(timestamp_col)
                # Resample numeric columns by hourly mean
                numeric_resampled = group_indexed[numeric_cols].resample('h').mean()
                # Forward fill station metadata
                for c in non_numeric_cols:
                    if c in group_indexed.columns:
                        numeric_resampled[c] = group_indexed[c].iloc[0]
                numeric_resampled[station_col] = station_id
                resampled_groups.append(numeric_resampled.reset_index())
            return pd.concat(resampled_groups, ignore_index=True)
        else:
            group_indexed = df_sorted.set_index(timestamp_col)
            numeric_resampled = group_indexed[numeric_cols].resample('h').mean()
            return numeric_resampled.reset_index()

    def impute_missing_values(self, df: pd.DataFrame, target_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Multi-tier imputation:
        1. Forward-fill for short gaps (up to 3 hours).
        2. KNN Imputation for remaining longer gaps.
        """
        df_imp = df.copy()
        if target_cols is None:
            target_cols = [c for c in ["pm25", "pm10", "no2", "so2", "co", "o3", "temp", "humidity", "wind_speed", "pressure"] if c in df_imp.columns]

        # Step 1: Forward-fill short gaps per station
        if "station_id" in df_imp.columns:
            df_imp[target_cols] = df_imp.groupby("station_id")[target_cols].ffill(limit=3)
            df_imp[target_cols] = df_imp.groupby("station_id")[target_cols].bfill(limit=1)

        # Step 2: KNN Imputation for remaining NaNs
        nans_remaining = df_imp[target_cols].isna().sum().sum()
        if nans_remaining > 0:
            logger.info(f"Performing KNN Imputation for {nans_remaining} remaining missing values...")
            imputer = KNNImputer(n_neighbors=self.knn_neighbors)
            imputed_vals = imputer.fit_transform(df_imp[target_cols])
            df_imp[target_cols] = imputed_vals

        return df_imp
