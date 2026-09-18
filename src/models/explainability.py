"""
Explainable AI (XAI) SHAP Module for Vayusight (Member B Lead - Week 10)
Computes SHAP feature attributions and generates natural language explanations for predictions.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple

# Try importing shap
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.info("SHAP library not installed. Operating in feature-importance fallback mode.")

logger = logging.getLogger(__name__)


class AQIExplainer:
    """Computes SHAP feature importance attributions and generates plain-language driver summaries."""

    FEATURE_FRIENDLY_NAMES = {
        "pm25": "PM2.5 fine dust concentration",
        "pm10": "PM10 coarse dust level",
        "no2": "Nitrogen Dioxide vehicle emissions",
        "so2": "Sulfur Dioxide industrial emissions",
        "temp": "Ambient Temperature",
        "humidity": "Relative Humidity",
        "wind_speed": "Wind Speed",
        "wind_u": "Zonal East-West Wind vector",
        "wind_v": "Meridional North-South Wind vector",
        "aod_550": "Satellite Aerosol Optical Depth",
        "aod_pm25_ratio": "Satellite-to-Ground Aerosol Ratio",
        "road_density_index": "Road Network Density"
    }

    def __init__(self, model: Any, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None

        if SHAP_AVAILABLE:
            try:
                self.explainer = shap.TreeExplainer(model)
            except Exception:
                try:
                    self.explainer = shap.Explainer(model)
                except Exception as e:
                    logger.warning(f"Could not initialize SHAP explainer: {e}")

    def compute_shap_values(self, X_sample: pd.DataFrame) -> np.ndarray:
        """Calculate SHAP matrix for input sample DataFrame."""
        if self.explainer is not None:
            try:
                shap_vals = self.explainer.shap_values(X_sample)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[0]
                return shap_vals
            except Exception as e:
                logger.warning(f"SHAP calculation failed: {e}. Falling back to weight approximation.")

        # Fallback feature weight approximation
        return self._fallback_feature_weights(X_sample)

    def _fallback_feature_weights(self, X_sample: pd.DataFrame) -> np.ndarray:
        """Construct fallback feature importance weights if SHAP library is unavailable."""
        if hasattr(self.model, "feature_importances_"):
            imp = self.model.feature_importances_
        else:
            imp = np.ones(X_sample.shape[1]) / X_sample.shape[1]

        # Multiply feature values by importances
        norm_x = (X_sample.values - np.mean(X_sample.values, axis=0)) / (np.std(X_sample.values, axis=0) + 1e-6)
        return norm_x * imp

    def generate_narrative_explanation(self, X_instance: pd.DataFrame, top_k: int = 3) -> Dict[str, Any]:
        """
        Generate plain-language explanation string summarizing top drivers for a specific prediction instance.
        """
        shap_vals = self.compute_shap_values(X_instance)
        if len(shap_vals.shape) > 1:
            instance_shap = shap_vals[0]
        else:
            instance_shap = shap_vals

        cols = list(X_instance.columns)
        shap_tuples = [(cols[i], instance_shap[i]) for i in range(min(len(cols), len(instance_shap)))]
        
        # Sort by magnitude of impact
        shap_tuples.sort(key=lambda x: abs(x[1]), reverse=True)
        top_drivers = shap_tuples[:top_k]

        positive_drivers = []
        negative_drivers = []

        for feat, val in top_drivers:
            friendly_name = self.FEATURE_FRIENDLY_NAMES.get(feat, feat.replace("_", " "))
            if val > 0:
                positive_drivers.append(f"{friendly_name} (+{val:.1f} AQI points)")
            else:
                negative_drivers.append(f"{friendly_name} ({val:.1f} AQI points)")

        narrative_parts = []
        if positive_drivers:
            narrative_parts.append("AQI is driven higher primarily by " + ", ".join(positive_drivers))
        if negative_drivers:
            narrative_parts.append("partially offset by " + ", ".join(negative_drivers))

        summary_text = ". ".join(narrative_parts) + "." if narrative_parts else "AQI levels are within expected baseline ranges."

        return {
            "summary_text": summary_text,
            "top_features": [{"feature": f, "shap_value": round(float(v), 3)} for f, v in top_drivers]
        }
