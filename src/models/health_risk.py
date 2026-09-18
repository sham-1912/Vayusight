"""
Hyperlocal Health Risk Translation Engine for Vayusight (Member B Lead - Week 10)
Translates forecasted PM2.5 concentrations on a ~1 km spatial grid into estimated health burden
(respiratory & cardiovascular risk events per 100,000 population) using WHO/GBD coefficients.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class HealthRiskCalculator:
    """Calculates estimated health incidence risk from PM2.5 levels using WHO concentration-response functions."""

    # WHO Global Air Quality Guideline baseline for PM2.5 (24-hour mean) is 15 ug/m3
    WHO_PM25_BASELINE = 15.0

    # Concentration-response coefficients (% increase in relative risk per 10 ug/m3 increase in PM2.5 above baseline)
    # Published GBD epidemiological coefficients for respiratory outcomes: ~1.08 Relative Risk per 10 ug/m3 (8% increase)
    BETA_RESPIRATORY_PER_10UG = 0.08
    BETA_CARDIOVASCULAR_PER_10UG = 0.06

    def compute_grid_health_risk(
        self,
        grid_df: pd.DataFrame,
        pm25_col: str = "pm25",
        population_baseline: int = 100000
    ) -> pd.DataFrame:
        """
        Calculate relative risk (RR) and excess estimated health risk events per 100,000 people for each grid cell.
        """
        grid_out = grid_df.copy()

        # If pm25 is missing, estimate from AQI if present
        if pm25_col not in grid_out.columns:
            if "estimated_aqi" in grid_out.columns:
                grid_out[pm25_col] = grid_out["estimated_aqi"] / 1.67
            else:
                grid_out[pm25_col] = 60.0

        pm25_vals = grid_out[pm25_col].fillna(60.0).values
        excess_pm25 = np.maximum(0.0, pm25_vals - self.WHO_PM25_BASELINE)

        # Relative Risk: RR = 1 + (BETA * excess_pm25 / 10)
        rr_respiratory = 1.0 + (self.BETA_RESPIRATORY_PER_10UG * (excess_pm25 / 10.0))
        rr_cardio = 1.0 + (self.BETA_CARDIOVASCULAR_PER_10UG * (excess_pm25 / 10.0))

        # Baseline incidence rate per 100,000 per week (~45 baseline respiratory events per 100k)
        baseline_incidence = 45.0
        excess_respiratory_events = np.round((rr_respiratory - 1.0) * baseline_incidence, 1)

        grid_out["relative_risk_respiratory"] = np.round(rr_respiratory, 3)
        grid_out["relative_risk_cardiovascular"] = np.round(rr_cardio, 3)
        grid_out["estimated_excess_respiratory_events_per_100k"] = excess_respiratory_events

        grid_out["health_risk_level"] = [self._get_risk_level(v) for v in excess_respiratory_events]

        return grid_out

    @staticmethod
    def _get_risk_level(events: float) -> str:
        if events <= 5.0:
            return "Low Risk"
        elif events <= 15.0:
            return "Moderate Risk"
        elif events <= 30.0:
            return "High Risk"
        else:
            return "Critical Risk"
