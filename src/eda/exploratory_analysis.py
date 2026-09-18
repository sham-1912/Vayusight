"""
Exploratory Data Analysis Module for Vayusight (Member A & B Lead - Week 6)
Provides correlation analysis, seasonal decomposition, and spatial statistical summaries.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class ExploratoryAnalyzer:
    """Analyzer for statistical correlation, seasonality decomposition, and spatial distributions."""

    def compute_correlation_matrix(
        self,
        df: pd.DataFrame,
        cols: Optional[List[str]] = None,
        method: str = "pearson"
    ) -> pd.DataFrame:
        """
        Compute correlation matrix across pollutants, meteorological drivers, and satellite AOD.
        """
        if cols is None:
            cols = [c for c in ["pm25", "pm10", "no2", "so2", "co", "o3", "temp", "humidity", "wind_speed", "pressure", "aod_550", "aqi"] if c in df.columns]

        df_sub = df[cols].dropna()
        if df_sub.empty:
            return pd.DataFrame()

        return df_sub.corr(method=method).round(4)

    def compute_spatial_summary(
        self,
        df: pd.DataFrame,
        station_col: str = "station_id"
    ) -> pd.DataFrame:
        """
        Compute per-station spatial summary statistics (mean, std, p50, p95 for PM2.5 and AQI).
        """
        if station_col not in df.columns:
            return pd.DataFrame()

        numeric_cols = [c for c in ["pm25", "pm10", "no2", "aod_550", "aqi"] if c in df.columns]

        summary = df.groupby(station_col)[numeric_cols].agg([
            ("mean", "mean"),
            ("std", "std"),
            ("median", "median"),
            ("p95", lambda x: np.percentile(x.dropna(), 95) if len(x.dropna()) > 0 else np.nan)
        ]).round(2)

        return summary

    def seasonal_decomposition_stats(
        self,
        df: pd.DataFrame,
        target_col: str = "pm25",
        period: int = 24
    ) -> Dict[str, Any]:
        """
        Perform simple moving-average seasonal decomposition (Trend, Seasonality, Residual).
        """
        if target_col not in df.columns or len(df) < period * 2:
            return {"status": "insufficient_data"}

        series = df[target_col].ffill().bfill()
        
        # Trend using centered rolling mean
        trend = series.rolling(window=period, center=True).mean()
        detrended = series - trend
        
        # Seasonal component (hourly mean pattern)
        seasonality = detrended.groupby(df.index % period).transform("mean")
        residual = detrended - seasonality

        return {
            "mean_trend": float(trend.mean()),
            "std_trend": float(trend.std()),
            "seasonal_amplitude": float(seasonality.max() - seasonality.min()),
            "residual_variance": float(residual.var()),
            "status": "success"
        }
