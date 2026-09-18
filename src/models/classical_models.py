"""
Classical Machine Learning Models for AQI Time-Series Forecasting (Member A Lead)
Implements XGBoost, LightGBM, CatBoost, and Random Forest regressors for tabular forecasting.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Try importing gradient boosting libraries
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import catboost as cb
    CB_AVAILABLE = True
except ImportError:
    CB_AVAILABLE = False

logger = logging.getLogger(__name__)


class TabularAQIForecaster:
    """Tabular machine learning regressor manager for multi-step AQI forecasting."""

    SUPPORTED_MODELS = ["xgboost", "lightgbm", "catboost", "random_forest"]

    def __init__(self, model_name: str = "xgboost", params: Optional[Dict[str, Any]] = None):
        self.model_name = model_name.lower()
        self.params = params or {}
        self.model = self._init_model()
        self.feature_names: List[str] = []

    def _init_model(self):
        """Instantiate the requested gradient boosting or forest regressor."""
        if self.model_name == "xgboost" and XGB_AVAILABLE:
            default_params = {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.05, "random_state": 42}
            default_params.update(self.params)
            return xgb.XGBRegressor(**default_params)

        elif self.model_name == "lightgbm" and LGB_AVAILABLE:
            default_params = {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.05, "random_state": 42, "verbose": -1}
            default_params.update(self.params)
            return lgb.LGBMRegressor(**default_params)

        elif self.model_name == "catboost" and CB_AVAILABLE:
            default_params = {"iterations": 100, "depth": 6, "learning_rate": 0.05, "random_seed": 42, "verbose": 0}
            default_params.update(self.params)
            return cb.CatBoostRegressor(**default_params)

        else:
            if self.model_name not in ["random_forest", "xgboost", "lightgbm", "catboost"]:
                logger.warning(f"Unknown model '{self.model_name}'. Defaulting to RandomForestRegressor.")
            default_params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
            default_params.update(self.params)
            return RandomForestRegressor(**default_params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TabularAQIForecaster":
        """Train tabular regressor on feature matrix X and target y."""
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for feature matrix X."""
        return self.model.predict(X)

    def evaluate(self, X: pd.DataFrame, y_true: pd.Series) -> Dict[str, float]:
        """Calculate RMSE, MAE, MAPE, and R2 evaluation metrics."""
        y_pred = self.predict(X)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        
        # Avoid zero division in MAPE
        eps = 1e-3
        mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0)
        r2 = float(r2_score(y_true, y_pred))

        return {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "mape": round(mape, 4),
            "r2": round(r2, 4)
        }

    def get_feature_importances(self) -> pd.DataFrame:
        """Return feature importance ranking DataFrame."""
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        else:
            importances = np.zeros(len(self.feature_names))

        df_imp = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)

        return df_imp
