"""
Model validation metrics used in credit risk:
  - Gini coefficient (= 2*AUC - 1)
  - KS statistic (max separation between cumulative good/bad distributions)
  - Population Stability Index (PSI) for monitoring model drift
  - ROC curve data
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def gini_coefficient(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Gini = 2 * AUC - 1. Range [0, 1]. Higher = better discrimination."""
    auc = roc_auc_score(y_true, y_prob)
    return round(2 * auc - 1, 4)


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Kolmogorov-Smirnov statistic: max separation between cumulative
    distributions of good and bad borrowers.

    Returns KS value and the threshold where it occurs.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ks_values = tpr - fpr
    idx = np.argmax(ks_values)
    return {
        "ks": round(float(ks_values[idx]), 4),
        "threshold": round(float(thresholds[idx]), 4) if idx < len(thresholds) else None,
        "fpr_at_ks": round(float(fpr[idx]), 4),
        "tpr_at_ks": round(float(tpr[idx]), 4),
    }


def ks_curve_data(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 20) -> pd.DataFrame:
    """Compute cumulative good/bad distributions for KS chart."""
    df = pd.DataFrame({"prob": y_prob, "default": y_true})
    df = df.sort_values("prob")

    total_bad = df["default"].sum()
    total_good = len(df) - total_bad

    df["cum_bad"] = df["default"].cumsum() / total_bad
    df["cum_good"] = (1 - df["default"]).cumsum() / total_good
    df["ks_diff"] = (df["cum_good"] - df["cum_bad"]).abs()

    # Sample evenly for chart
    step = max(1, len(df) // n_bins)
    sampled = df.iloc[::step][["prob", "cum_bad", "cum_good", "ks_diff"]].copy()
    sampled.columns = ["probability", "cum_bad_pct", "cum_good_pct", "ks_gap"]
    return sampled.round(4).reset_index(drop=True)


def roc_data(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    """ROC curve points for charting."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    # Downsample to ~50 points for chart
    step = max(1, len(fpr) // 50)
    return pd.DataFrame({
        "fpr": np.round(fpr[::step], 4),
        "tpr": np.round(tpr[::step], 4),
    })


def population_stability_index(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    PSI measures how much the score distribution has shifted between
    development (expected) and validation (actual) samples.

    PSI < 0.10 : no significant shift
    0.10-0.25  : moderate shift, monitor
    > 0.25     : significant shift, re-develop model

    Returns PSI value and per-bin breakdown.
    """
    # Create bins from expected distribution
    _, bin_edges = np.histogram(expected, bins=n_bins)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    expected_counts = np.histogram(expected, bins=bin_edges)[0]
    actual_counts = np.histogram(actual, bins=bin_edges)[0]

    # Convert to proportions, add epsilon
    eps = 1e-4
    expected_pct = expected_counts / expected_counts.sum() + eps
    actual_pct = actual_counts / actual_counts.sum() + eps

    psi_components = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    total_psi = float(psi_components.sum())

    def _interpret(v: float) -> str:
        if v < 0.10:
            return "Stable"
        if v < 0.25:
            return "Moderate shift"
        return "Significant shift"

    bins = []
    for i in range(len(psi_components)):
        lo = bin_edges[i] if not np.isinf(bin_edges[i]) else None
        hi = bin_edges[i + 1] if not np.isinf(bin_edges[i + 1]) else None
        bins.append({
            "bin_low": round(lo, 2) if lo is not None else None,
            "bin_high": round(hi, 2) if hi is not None else None,
            "expected_pct": round(float(expected_pct[i]), 4),
            "actual_pct": round(float(actual_pct[i]), 4),
            "psi_component": round(float(psi_components[i]), 6),
        })

    return {
        "psi": round(total_psi, 4),
        "interpretation": _interpret(total_psi),
        "bins": bins,
    }
