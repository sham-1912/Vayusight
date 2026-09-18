"""
Time-Series Feature Extraction Module for Vayusight (Member A Lead)
Generates temporal lags, rolling statistics, and cyclical sine/cosine time encodings
without causing temporal data leakage.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class TimeSeriesFeatureExtractor:
    """Extracts lag features, rolling statistics, and cyclical encodings for AQI time series."""

    DEFAULT_LAGS = [1, 3, 6, 12, 24, 48, 72]
    DEFAULT_ROLLING_WINDOWS = [6, 24, 168]  # 6h, 24h, 7 days (168h)

    def __init__(
        self,
        lags: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None
    ):
        self.lags = lags if lags is not None else self.DEFAULT_LAGS
        self.rolling_windows = rolling_windows if rolling_windows is not None else self.DEFAULT_ROLLING_WINDOWS

    def add_cyclical_time_features(self, df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
        """
        Encode cyclical time components (hour, day of week, month) into sine and cosine pairs.
        Prevents artificial discontinuity between 23:00 and 00:00 or Sunday and Monday.
        """
        df_out = df.copy()
        ts = pd.to_datetime(df_out[timestamp_col], utc=True)

        # Hour of day (0-23, period = 24)
        hour = ts.dt.hour
        df_out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        df_out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

        # Day of week (0-6, period = 7)
        dayofweek = ts.dt.dayofweek
        df_out["day_sin"] = np.sin(2 * np.pi * dayofweek / 7.0)
        df_out["day_cos"] = np.cos(2 * np.pi * dayofweek / 7.0)

        # Month of year (1-12, period = 12)
        month = ts.dt.month - 1
        df_out["month_sin"] = np.sin(2 * np.pi * month / 12.0)
        df_out["month_cos"] = np.cos(2 * np.pi * month / 12.0)

        return df_out

    def add_lag_features(
        self,
        df: pd.DataFrame,
        target_cols: List[str],
        station_col: str = "station_id",
        timestamp_col: str = "timestamp"
    ) -> pd.DataFrame:
        """
        Construct temporal lag features per station ID.
        For example, pm25_lag_1h, pm25_lag_24h, etc.
        """
        df_out = df.copy()
        df_out[timestamp_col] = pd.to_datetime(df_out[timestamp_col], utc=True)
        df_out = df_out.sort_values(by=[station_col, timestamp_col]).reset_index(drop=True)

        for col in target_cols:
            if col not in df_out.columns:
                continue
            for lag in self.lags:
                lag_col_name = f"{col}_lag_{lag}h"
                df_out[lag_col_name] = df_out.groupby(station_col)[col].shift(lag)

        return df_out

    def add_rolling_features(
        self,
        df: pd.DataFrame,
        target_cols: List[str],
        station_col: str = "station_id",
        timestamp_col: str = "timestamp"
    ) -> pd.DataFrame:
        """
        Construct rolling window statistics (mean, std, min, max) per station ID.
        Uses closed='left' logic to avoid current-timestep leakage when predicting.
        """
        df_out = df.copy()
        df_out[timestamp_col] = pd.to_datetime(df_out[timestamp_col], utc=True)
        df_out = df_out.sort_values(by=[station_col, timestamp_col]).reset_index(drop=True)

        for col in target_cols:
            if col not in df_out.columns:
                continue
            for window in self.rolling_windows:
                # Grouped rolling mean, std, min, max shifted by 1 to prevent data leakage
                grouped = df_out.groupby(station_col)[col].shift(1)
                
                df_out[f"{col}_roll_{window}h_mean"] = (
                    df_out.groupby(station_col)[col]
                    .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
                )
                df_out[f"{col}_roll_{window}h_std"] = (
                    df_out.groupby(station_col)[col]
                    .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).std().fillna(0.0))
                )
                df_out[f"{col}_roll_{window}h_min"] = (
                    df_out.groupby(station_col)[col]
                    .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).min())
                )
                df_out[f"{col}_roll_{window}h_max"] = (
                    df_out.groupby(station_col)[col]
                    .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).max())
                )

        return df_out
