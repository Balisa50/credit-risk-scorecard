"""
Stress testing: apply adverse economic scenarios to the scorecard
and estimate impact on default rates and expected losses.

Scenarios model real West African economic shocks:
  - Baseline: current conditions
  - Mild stress: moderate economic slowdown
  - Severe stress: regional economic crisis (currency devaluation, crop failure)
  - Extreme: COVID-scale disruption
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SCENARIOS = {
    "Baseline": {
        "income_shock": 0.0,
        "default_multiplier": 1.0,
        "description": "Current economic conditions",
    },
    "Mild Stress": {
        "income_shock": -0.10,
        "default_multiplier": 1.3,
        "description": "Moderate slowdown: 10% income drop, 30% rise in defaults",
    },
    "Severe Stress": {
        "income_shock": -0.25,
        "default_multiplier": 1.8,
        "description": "Regional crisis: 25% income drop, currency devaluation, crop failure",
    },
    "Extreme": {
        "income_shock": -0.40,
        "default_multiplier": 2.5,
        "description": "Pandemic-scale disruption: 40% income drop, supply chain collapse",
    },
}


def run_stress_tests(
    base_default_rate: float,
    total_portfolio_usd: float,
    loss_given_default: float = 0.55,
) -> list[dict]:
    """
    Apply stress scenarios to portfolio metrics.

    Args:
        base_default_rate: observed default rate under baseline
        total_portfolio_usd: total outstanding loan portfolio value
        loss_given_default: LGD (typical microfinance: 50-60%)

    Returns:
        List of scenario results with stressed PD, EL, and capital requirements.
    """
    results = []

    for name, scenario in SCENARIOS.items():
        stressed_pd = min(base_default_rate * scenario["default_multiplier"], 0.95)
        expected_loss = total_portfolio_usd * stressed_pd * loss_given_default
        # Basel II capital requirement (simplified): K = LGD * [N(G(PD)) - PD]
        # Using simplified version: capital ~= 8% * risk-weighted assets
        capital_requirement = expected_loss * 1.5  # simplified buffer

        results.append({
            "scenario": name,
            "description": scenario["description"],
            "income_shock_pct": scenario["income_shock"] * 100,
            "stressed_pd": round(stressed_pd, 4),
            "expected_loss_usd": round(expected_loss, 2),
            "capital_required_usd": round(capital_requirement, 2),
            "loss_rate_pct": round(
                (expected_loss / total_portfolio_usd) * 100, 2
            ),
        })

    return results


def stress_score_impact(
    scores: np.ndarray,
    scenario_shift: float,
) -> np.ndarray:
    """
    Shift credit scores downward under stress (lower score = higher risk).
    """
    return scores - scenario_shift


def stressed_default_rates_by_band(
    scores: np.ndarray,
    labels: np.ndarray,
    stress_multiplier: float,
    n_bands: int = 5,
) -> pd.DataFrame:
    """
    Compute default rates per score band under a stress scenario.
    """
    df = pd.DataFrame({"score": scores, "default": labels})
    df["band"] = pd.qcut(df["score"], q=n_bands, duplicates="drop")
    summary = (
        df.groupby("band", observed=True)
        .agg(count=("default", "size"), base_defaults=("default", "sum"))
        .reset_index()
    )
    summary["base_default_rate"] = summary["base_defaults"] / summary["count"]
    summary["stressed_default_rate"] = np.minimum(
        summary["base_default_rate"] * stress_multiplier, 0.95
    )
    summary["band"] = summary["band"].astype(str)
    return summary.round(4)
