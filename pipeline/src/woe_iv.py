"""
Weight of Evidence (WoE) and Information Value (IV) calculation.

WoE = ln(Distribution of Good / Distribution of Bad)
IV  = sum((Dist_Good - Dist_Bad) * WoE)

IV interpretation:
  < 0.02  : not predictive
  0.02-0.1: weak predictor
  0.1-0.3 : medium predictor
  0.3-0.5 : strong predictor
  > 0.5   : suspicious (over-predicting)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _optimal_bins(series: pd.Series, target: pd.Series, max_bins: int = 10) -> pd.Series:
    """Bin a numeric feature into quantile-based bins, merging tiny groups."""
    try:
        binned = pd.qcut(series, q=max_bins, duplicates="drop")
    except ValueError:
        binned = pd.cut(series, bins=min(max_bins, series.nunique()), duplicates="drop")
    return binned


def compute_woe_iv(
    df: pd.DataFrame,
    feature: str,
    target: str = "default",
    max_bins: int = 10,
) -> pd.DataFrame:
    """
    Compute WoE and IV for a single feature.

    Returns a DataFrame with columns:
      bin, count, events, non_events, event_rate,
      dist_events, dist_non_events, woe, iv_component
    """
    col = df[feature].copy()
    is_numeric = pd.api.types.is_numeric_dtype(col)

    if is_numeric and col.nunique() > max_bins:
        col = _optimal_bins(col, df[target], max_bins)

    grouped = (
        df.assign(_bin=col.astype(str))
        .groupby("_bin", observed=True)
        .agg(
            count=(target, "size"),
            events=(target, "sum"),
        )
    )
    grouped["non_events"] = grouped["count"] - grouped["events"]
    grouped["event_rate"] = grouped["events"] / grouped["count"]

    total_events = grouped["events"].sum()
    total_non_events = grouped["non_events"].sum()

    # Add small epsilon to avoid division by zero / log(0)
    eps = 0.5
    grouped["dist_events"] = (grouped["events"] + eps) / (total_events + eps * len(grouped))
    grouped["dist_non_events"] = (grouped["non_events"] + eps) / (
        total_non_events + eps * len(grouped)
    )

    grouped["woe"] = np.log(grouped["dist_non_events"] / grouped["dist_events"])
    grouped["iv_component"] = (grouped["dist_non_events"] - grouped["dist_events"]) * grouped["woe"]

    grouped = grouped.reset_index().rename(columns={"_bin": "bin"})
    grouped["feature"] = feature
    return grouped


def compute_all_woe_iv(
    df: pd.DataFrame,
    features: list[str],
    target: str = "default",
    max_bins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute WoE/IV for all features.

    Returns:
      woe_details: per-bin WoE table for every feature
      iv_summary:  one row per feature with total IV, sorted descending
    """
    all_details = []
    iv_rows = []

    for feat in features:
        detail = compute_woe_iv(df, feat, target, max_bins)
        all_details.append(detail)
        total_iv = detail["iv_component"].sum()
        iv_rows.append({"feature": feat, "iv": total_iv})

    woe_details = pd.concat(all_details, ignore_index=True)
    iv_summary = (
        pd.DataFrame(iv_rows)
        .sort_values("iv", ascending=False)
        .reset_index(drop=True)
    )

    # Interpret IV strength
    def _label(v: float) -> str:
        if v < 0.02:
            return "Not predictive"
        if v < 0.10:
            return "Weak"
        if v < 0.30:
            return "Medium"
        if v < 0.50:
            return "Strong"
        return "Suspicious"

    iv_summary["strength"] = iv_summary["iv"].apply(_label)
    return woe_details, iv_summary


def woe_transform(
    df: pd.DataFrame,
    woe_details: pd.DataFrame,
    features: list[str],
    max_bins: int = 10,
) -> pd.DataFrame:
    """Replace feature values with their WoE values."""
    result = df.copy()

    for feat in features:
        feat_woe = woe_details[woe_details["feature"] == feat].set_index("bin")["woe"]
        col = result[feat].copy()
        is_numeric = pd.api.types.is_numeric_dtype(col)

        if is_numeric and col.nunique() > max_bins:
            binned = _optimal_bins(col, result.get("default", pd.Series(dtype=int)), max_bins)
        else:
            binned = col

        mapped = binned.astype(str).map(feat_woe)
        # Fill unmapped with 0 (neutral WoE)
        result[f"{feat}_woe"] = mapped.fillna(0)

    return result
