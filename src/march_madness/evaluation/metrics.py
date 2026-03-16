from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss


def brier_score(y_true: pd.Series, y_prob: pd.Series) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def probability_metrics(y_true: pd.Series, y_prob: pd.Series) -> dict[str, float]:
    clipped = np.clip(y_prob.astype(float), 1e-6, 1 - 1e-6)
    return {
        "games": int(len(y_true)),
        "brier_score": brier_score(y_true, clipped),
        "log_loss": float(log_loss(y_true, clipped)),
        "accuracy": float(accuracy_score(y_true, clipped >= 0.5)),
    }
