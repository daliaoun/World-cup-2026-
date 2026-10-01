"""
Scoring rules for probabilistic 1X2 forecasts.

RPS (Ranked Probability Score) is the primary metric. Football outcomes are ORDERED
(home win - draw - away win): predicting a home win when the away side wins is a worse
error than predicting a draw. Log loss and Brier are blind to that ordering; RPS is not,
which is why it is the standard in the football-forecasting literature (Constantinou &
Fenton 2012). Lower is better.

    RPS = 1/(r-1) * sum_{i=1}^{r-1} ( sum_{j=1}^{i} (p_j - e_j) )^2

Reference points for 3-outcome football:
    0.333  a uniform 1/3-1/3-1/3 forecast is ~0.222 RPS in practice
    ~0.19-0.21   good bookmaker / strong model on club football
    higher on international football, which is noisier
"""
from __future__ import annotations
import numpy as np


def rps(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """probs: (n,3) in [home, draw, away] order. y: (n,) in {0,1,2}. Returns per-match RPS."""
    probs = np.asarray(probs, float)
    n, r = probs.shape
    E = np.zeros_like(probs)
    E[np.arange(n), np.asarray(y, int)] = 1.0
    cp = np.cumsum(probs, axis=1)
    ce = np.cumsum(E, axis=1)
    return ((cp[:, :r - 1] - ce[:, :r - 1]) ** 2).sum(axis=1) / (r - 1)


def log_loss(probs: np.ndarray, y: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    p = np.clip(np.asarray(probs, float), eps, 1)
    return -np.log(p[np.arange(len(y)), np.asarray(y, int)])


def brier(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, float)
    E = np.zeros_like(probs)
    E[np.arange(len(y)), np.asarray(y, int)] = 1.0
    return ((probs - E) ** 2).sum(axis=1)


def accuracy(probs: np.ndarray, y: np.ndarray) -> float:
    return float((np.asarray(probs).argmax(1) == np.asarray(y)).mean())


def summary(probs: np.ndarray, y: np.ndarray) -> dict:
    return {
        "n": int(len(y)),
        "RPS": float(rps(probs, y).mean()),
        "logloss": float(log_loss(probs, y).mean()),
        "brier": float(brier(probs, y).mean()),
        "accuracy": accuracy(probs, y),
    }


def calibration(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> "pd.DataFrame":
    """Reliability of every predicted probability, pooled across the 3 outcome classes."""
    import pandas as pd
    probs = np.asarray(probs, float).ravel()
    E = np.zeros((len(y), 3))
    E[np.arange(len(y)), np.asarray(y, int)] = 1.0
    E = E.ravel()
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(probs, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum() < 5:
            continue
        rows.append(dict(bin_lo=edges[b], bin_hi=edges[b + 1], n=int(m.sum()),
                         predicted=float(probs[m].mean()), observed=float(E[m].mean())))
    return pd.DataFrame(rows)


def american_to_prob(odds: int) -> float:
    """Convert American odds to an implied (vig-inclusive) probability."""
    return (-odds) / (-odds + 100) if odds < 0 else 100 / (odds + 100)


def devig(probs: dict) -> dict:
    """Remove the bookmaker's overround by proportional normalisation."""
    s = sum(probs.values())
    return {k: v / s for k, v in probs.items()}


def save_preds(path, P, e):
    """Persist a prediction matrix together with the match_ids it was built on."""
    np.savez(path, P=np.asarray(P, float), match_id=e.match_id.to_numpy())


def load_preds(path, e):
    """Load a prediction matrix and ASSERT it lines up with the eval frame given."""
    z = np.load(path, allow_pickle=True)
    if not np.array_equal(z["match_id"], e.match_id.to_numpy()):
        raise ValueError(f"{path}: match_id order does not match the eval frame")
    return z["P"]
