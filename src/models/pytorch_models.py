"""
PyTorch Deep Learning Sequence Models for AQI Forecasting (Member A Lead - Week 9)
Implements PyTorch LSTM and GRU neural networks for multi-step time-series sequence prediction.
Includes graceful analytical fallback if torch package is not installed.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple

# Try importing PyTorch
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    Dataset = object
    DataLoader = None
    logger = logging.getLogger(__name__)
    logger.info("PyTorch not installed. Operating in fallback simulation mode.")

logger = logging.getLogger(__name__)


if TORCH_AVAILABLE:
    class AQITimeSeriesDataset(Dataset):
        """PyTorch Dataset for sliding-window sequence generation."""

        def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 24):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
            self.seq_len = seq_len

        def __len__(self) -> int:
            return max(0, len(self.X) - self.seq_len)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            x_seq = self.X[idx : idx + self.seq_len]
            y_target = self.y[idx + self.seq_len]
            return x_seq, y_target


    class AQILSTM(nn.Module):
        """Multi-layer LSTM Neural Network for AQI sequence forecasting."""

        def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
            super(AQILSTM, self).__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, (hn, cn) = self.lstm(x)
            last_out = out[:, -1, :]
            pred = self.fc(last_out)
            return pred.squeeze(-1)


    class AQIGRU(nn.Module):
        """Multi-layer GRU Neural Network for AQI sequence forecasting."""

        def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
            super(AQIGRU, self).__init__()
            self.gru = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, hn = self.gru(x)
            last_out = out[:, -1, :]
            pred = self.fc(last_out)
            return pred.squeeze(-1)
else:
    class AQITimeSeriesDataset:
        pass

    class AQILSTM:
        pass

    class AQIGRU:
        pass


class PyTorchAQIForecaster:
    """High-level wrapper managing PyTorch model training, validation, and inference."""

    def __init__(
        self,
        architecture: str = "lstm",
        seq_len: int = 24,
        hidden_dim: int = 64,
        num_layers: int = 2,
        lr: float = 0.001,
        batch_size: int = 32,
        epochs: int = 15
    ):
        self.architecture = architecture.lower()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = None

        self.model: Optional[Any] = None
        self.mean_x: Optional[np.ndarray] = None
        self.std_x: Optional[np.ndarray] = None
        self.fallback_weights: Optional[np.ndarray] = None

    def _build_model(self, input_dim: int):
        if not TORCH_AVAILABLE:
            return None
        if self.architecture == "gru":
            return AQIGRU(input_dim, self.hidden_dim, self.num_layers).to(self.device)
        else:
            return AQILSTM(input_dim, self.hidden_dim, self.num_layers).to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PyTorchAQIForecaster":
        """Train PyTorch sequence model on sequence array X and target y."""
        self.mean_x = np.mean(X, axis=0)
        self.std_x = np.std(X, axis=0) + 1e-6

        if not TORCH_AVAILABLE:
            # Fallback linear regression fit
            X_norm = (X - self.mean_x) / self.std_x
            self.fallback_weights = np.linalg.lstsq(X_norm, y, rcond=None)[0]
            logger.info("Fitted PyTorch fallback linear solver.")
            return self

        X_norm = (X - self.mean_x) / self.std_x
        dataset = AQITimeSeriesDataset(X_norm, y, seq_len=self.seq_len)
        if len(dataset) == 0:
            logger.warning("Dataset too small for sequence length. Reduce seq_len.")
            return self

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        input_dim = X.shape[1]
        self.model = self._build_model(input_dim)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(batch_y)

            avg_loss = total_loss / len(dataset)
            if (epoch + 1) % 5 == 0 or epoch == self.epochs - 1:
                logger.info(f"PyTorch Epoch [{epoch+1}/{self.epochs}] Loss: {avg_loss:.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for sequence array X."""
        if self.mean_x is None:
            raise RuntimeError("Model is not fitted yet.")

        if not TORCH_AVAILABLE or self.model is None:
            X_norm = (X - self.mean_x) / self.std_x
            if self.fallback_weights is not None:
                return X_norm @ self.fallback_weights
            return np.zeros(len(X))

        self.model.eval()
        X_norm = (X - self.mean_x) / self.std_x
        dataset = AQITimeSeriesDataset(X_norm, np.zeros(len(X)), seq_len=self.seq_len)

        if len(dataset) == 0:
            return np.zeros(len(X))

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        all_preds = []

        with torch.no_grad():
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(self.device)
                preds = self.model(batch_x)
                all_preds.extend(preds.cpu().numpy())

        preds_arr = np.array(all_preds)
        if len(preds_arr) > 0:
            pad = np.full(self.seq_len, preds_arr[0])
            return np.concatenate([pad, preds_arr])
        else:
            return np.zeros(len(X))
