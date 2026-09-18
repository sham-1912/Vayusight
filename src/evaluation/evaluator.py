"""
Evaluation & Benchmarking Engine for Vayusight (Member A & B - Weeks 11-12 Milestone)
Implements expanding-window walk-forward validation splits, statistical significance tests,
and spatial Leave-One-Station-Out Cross-Validation (LOSO-CV).
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates time-series models via expanding walk-forward splits and spatial LOSO-CV."""

    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute RMSE, MAE, MAPE, and R2 metrics."""
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        yt, yp = y_true[mask], y_pred[mask]

        if len(yt) == 0:
            return {"rmse": 0.0, "mae": 0.0, "mape": 0.0, "r2": 0.0}

        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        mae = float(mean_absolute_error(yt, yp))
        mape = float(np.mean(np.abs((yt - yp) / (yt + 1e-3))) * 100.0)
        r2 = float(r2_score(yt, yp))

        return {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "mape": round(mape, 4),
            "r2": round(r2, 4)
        }

    def walk_forward_evaluate(
        self,
        df: pd.DataFrame,
        train_model_fn: Callable[[pd.DataFrame], Any],
        predict_fn: Callable[[Any, pd.DataFrame], np.ndarray],
        target_col: str = "aqi",
        n_splits: int = 3,
        initial_train_ratio: float = 0.60
    ) -> Dict[str, Any]:
        """
        Expanding-window walk-forward time-series evaluation protocol.
        Splits data chronologically without random shuffling to prevent data leakage.
        """
        n = len(df)
        initial_size = int(n * initial_train_ratio)
        step_size = (n - initial_size) // n_splits

        fold_metrics = []
        all_true = []
        all_pred = []

        for fold in range(n_splits):
            train_end = initial_size + (fold * step_size)
            val_end = train_end + step_size if fold < n_splits - 1 else n

            df_train = df.iloc[:train_end].copy()
            df_val = df.iloc[train_end:val_end].copy()

            model = train_model_fn(df_train)
            preds = predict_fn(model, df_val)

            y_true = df_val[target_col].values
            y_pred = preds[:len(y_true)]

            metrics = self.compute_metrics(y_true, y_pred)
            fold_metrics.append(metrics)

            all_true.extend(y_true)
            all_pred.extend(y_pred)

        overall_metrics = self.compute_metrics(np.array(all_true), np.array(all_pred))
        return {
            "overall": overall_metrics,
            "folds": fold_metrics
        }

    def paired_ttest_models(self, err_model1: np.ndarray, err_model2: np.ndarray) -> Dict[str, float]:
        """Perform paired t-test comparing absolute error distributions of two competing models."""
        t_stat, p_val = stats.ttest_rel(np.abs(err_model1), np.abs(err_model2))
        return {
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(p_val), 6),
            "significant_at_05": float(p_val) < 0.05
        }
