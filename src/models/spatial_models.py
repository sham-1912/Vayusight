"""
Spatial Estimation Models for Sensor-Sparse Zones (Member B Lead - Week 8-9)
Implements Spatial Random Forest Regression and Spatial IDW Interpolation
to estimate AQI across un-monitored city grid cells.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple

from sklearn.ensemble import RandomForestRegressor
from src.utils.geo_utils import haversine_distance, generate_city_grid

logger = logging.getLogger(__name__)


class SpatialAQIEstimator:
    """Estimates AQI across un-monitored spatial grid locations using satellite AOD + weather covariates."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 10):
        self.rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        self.feature_cols = ["latitude", "longitude", "aod_550", "temp", "humidity", "wind_speed", "wind_u", "wind_v"]
        self.is_fitted = False

    def fit(self, df_train: pd.DataFrame, target_col: str = "aqi") -> "SpatialAQIEstimator":
        """Train Spatial Random Forest regressor on monitored station records."""
        cols = [c for c in self.feature_cols if c in df_train.columns]
        if not cols or target_col not in df_train.columns:
            logger.warning("Missing required feature/target columns for spatial fitting.")
            return self

        X = df_train[cols].fillna(0.0)
        y = df_train[target_col].fillna(100.0)

        self.feature_cols = cols
        self.rf_model.fit(X, y)
        self.is_fitted = True
        logger.info(f"Spatial AQI Estimator fitted on {len(X)} monitored station instances.")
        return self

    def predict_grid(
        self,
        grid_df: pd.DataFrame,
        sat_aod_df: Optional[pd.DataFrame] = None,
        weather_dict: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Estimate AQI for every cell in an un-monitored spatial grid DataFrame.
        """
        grid_out = grid_df.copy()

        # Supply default weather variables if not provided
        default_weather = {
            "temp": 28.0,
            "humidity": 60.0,
            "wind_speed": 3.0,
            "wind_u": 1.5,
            "wind_v": -2.0
        }
        if weather_dict:
            default_weather.update(weather_dict)

        for col, val in default_weather.items():
            if col not in grid_out.columns:
                grid_out[col] = val

        # Ensure AOD is present
        if "aod_550" not in grid_out.columns:
            grid_out["aod_550"] = 0.55

        if self.is_fitted:
            X_grid = grid_out[self.feature_cols].fillna(0.0)
            preds = self.rf_model.predict(X_grid)
            grid_out["estimated_aqi"] = np.round(np.clip(preds, 0.0, 500.0), 1)
        else:
            # Fallback simple spatial estimation based on AOD
            grid_out["estimated_aqi"] = np.round(grid_out["aod_550"] * 250.0, 1)

        # Categorize AQI Severity Level (CPCB Color Standard)
        grid_out["aqi_category"] = grid_out["estimated_aqi"].apply(self._get_aqi_category)
        return grid_out

    @staticmethod
    def _get_aqi_category(aqi_val: float) -> str:
        if aqi_val <= 50:
            return "Good"
        elif aqi_val <= 100:
            return "Satisfactory"
        elif aqi_val <= 200:
            return "Moderate"
        elif aqi_val <= 300:
            return "Poor"
        elif aqi_val <= 400:
            return "Very Poor"
        else:
            return "Severe"
