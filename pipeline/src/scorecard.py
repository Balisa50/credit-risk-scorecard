"""
Logistic regression scorecard with Basel II-style points conversion.

Points formula:
  Factor = PDO / ln(2)
  Offset = Target_Score - Factor * ln(Target_Odds)
  Points_i = -(beta_i * WoE_ij * Factor) + Offset / n_features

Standard settings: target score 600 at 50:1 odds, PDO=20.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# Basel II scorecard parameters
TARGET_SCORE = 600
TARGET_ODDS = 50  # 50:1 good:bad at target score
PDO = 20  # points to double the odds


def build_scorecard(
    df: pd.DataFrame,
    woe_features: list[str],
    target: str = "default",
    test_size: float = 0.3,
    random_state: int = 42,
) -> dict:
    """
    Train logistic regression on WoE-transformed features and convert
    coefficients to a points-based scorecard.

    Returns dict with: model, scorecard_table, train/test sets, factor, offset.
    """
    X = df[woe_features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    # Points conversion
    factor = PDO / np.log(2)
    offset = TARGET_SCORE - factor * np.log(TARGET_ODDS)
    n_features = len(woe_features)

    coefs = model.coef_[0]
    intercept = model.intercept_[0]

    scorecard_rows = []
    for feat, beta in zip(woe_features, coefs):
        base_name = feat.replace("_woe", "")
        scorecard_rows.append({
            "feature": base_name,
            "woe_feature": feat,
            "coefficient": round(beta, 6),
            "base_points": round(-(beta * 0 * factor) + offset / n_features, 1),
        })

    scorecard_table = pd.DataFrame(scorecard_rows)

    return {
        "model": model,
        "scorecard_table": scorecard_table,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "factor": factor,
        "offset": offset,
        "intercept": intercept,
        "coefficients": dict(zip(woe_features, coefs)),
    }


def compute_scores(
    df: pd.DataFrame,
    woe_features: list[str],
    coefficients: dict[str, float],
    intercept: float,
    factor: float,
    offset: float,
) -> np.ndarray:
    """Compute credit scores for each row."""
    n = len(woe_features)
    scores = np.full(len(df), offset)

    for feat in woe_features:
        beta = coefficients[feat]
        woe_vals = df[feat].values
        scores += -(beta * woe_vals * factor)

    # Add intercept contribution
    scores += -(intercept * factor)

    return scores


def score_distribution(scores: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Bin scores and compute default rate per band."""
    score_df = pd.DataFrame({"score": scores, "default": labels})
    score_df["band"] = pd.cut(score_df["score"], bins=10)
    summary = (
        score_df.groupby("band", observed=True)
        .agg(count=("default", "size"), defaults=("default", "sum"))
        .reset_index()
    )
    summary["default_rate"] = summary["defaults"] / summary["count"]
    summary["band"] = summary["band"].astype(str)
    return summary
