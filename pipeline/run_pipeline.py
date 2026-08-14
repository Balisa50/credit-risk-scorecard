"""
Run the full credit risk scorecard pipeline and export results as JSON
for the Next.js dashboard.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from generate_data import generate
from sklearn.model_selection import train_test_split

from src.woe_iv import compute_all_woe_iv, woe_transform
from src.scorecard import build_scorecard, compute_scores, score_distribution
from src.validation import (
    gini_coefficient,
    ks_statistic,
    ks_curve_data,
    roc_data,
    population_stability_index,
)
from src.stress_test import run_stress_tests, stressed_default_rates_by_band, SCENARIOS

OUT_DIR = Path(__file__).parent.parent / "public" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("1/6 Generating loan data...")
    df = generate()
    print(f"    {len(df)} records, default rate: {df['default'].mean():.1%}")

    # Features for WoE/IV analysis
    features = [
        "age", "gender", "dependents", "monthly_income_usd",
        "years_in_business", "sector", "loan_purpose",
        "loan_amount_usd", "loan_term_months", "interest_rate_pct",
        "dti_ratio", "previous_loans", "previous_defaults",
        "dpd_history_days", "group_lending", "has_collateral", "country",
    ]

    print("2/6 Computing WoE and IV...")
    # WoE bins are derived from the target's event rate per bin, so fitting them
    # on the full frame lets a test row's own label shape its own encoding. The
    # split is therefore taken here, before any target-aware transform, and the
    # bins are learned on the training rows only.
    #
    # Measured on this data the leak was worth about 0.003 Gini and 0.012 KS,
    # which is small because 12,000 rows and coarse bins make the bin
    # statistics stable. It is fixed because a scorecard that an auditor can
    # read is the entire point of the project, not because the number moved.
    train_idx, test_idx = train_test_split(
        df.index, test_size=0.3, random_state=42, stratify=df["default"]
    )
    woe_details, iv_summary = compute_all_woe_iv(df.loc[train_idx], features)
    print(f"    Top features by IV:")
    for _, row in iv_summary.head(8).iterrows():
        print(f"      {row['feature']:25s} IV={row['iv']:.4f} ({row['strength']})")

    # Select features with IV >= 0.02 (at least weak predictive power)
    selected = iv_summary[iv_summary["iv"] >= 0.02]["feature"].tolist()
    print(f"    Selected {len(selected)} features (IV >= 0.02)")

    print("3/6 Building scorecard...")
    woe_features = [f"{f}_woe" for f in selected]
    df_woe = woe_transform(df, woe_details, selected)

    result = build_scorecard(df_woe, woe_features, train_idx=train_idx, test_idx=test_idx)
    model = result["model"]

    # Compute scores
    train_probs = model.predict_proba(result["X_train"])[:, 1]
    test_probs = model.predict_proba(result["X_test"])[:, 1]

    train_scores = compute_scores(
        result["X_train"], woe_features,
        result["coefficients"], result["intercept"],
        result["factor"], result["offset"],
    )
    test_scores = compute_scores(
        result["X_test"], woe_features,
        result["coefficients"], result["intercept"],
        result["factor"], result["offset"],
    )

    print("4/6 Validating model...")
    train_gini = gini_coefficient(result["y_train"].values, train_probs)
    test_gini = gini_coefficient(result["y_test"].values, test_probs)
    train_ks = ks_statistic(result["y_train"].values, train_probs)
    test_ks = ks_statistic(result["y_test"].values, test_probs)

    print(f"    Train Gini: {train_gini:.4f}, Test Gini: {test_gini:.4f}")
    print(f"    Train KS:   {train_ks['ks']:.4f}, Test KS:   {test_ks['ks']:.4f}")

    # PSI between train and test score distributions.
    #
    # Read this for what it is. PSI exists to detect population drift between
    # the data a model was built on and the data it is scoring later. Both
    # halves here come from one random split of one generated dataset, so a
    # near-zero value is arithmetic, not evidence that the scorecard is stable
    # in production. It is kept as a wiring check and labelled as such.
    #
    # A real stability figure needs an origination date on each loan and a
    # split by vintage. The generator has no time axis at all, so that test
    # cannot be run here yet; adding one is the next piece of work.
    psi_result = population_stability_index(train_scores, test_scores)
    psi_result["scope"] = "same-period random split, not temporal drift"
    print(f"    PSI: {psi_result['psi']:.4f} ({psi_result['scope']})")

    # ROC and KS curve data
    roc = roc_data(result["y_test"].values, test_probs)
    ks_curve = ks_curve_data(result["y_test"].values, test_probs)
    score_dist = score_distribution(test_scores, result["y_test"].values)

    print("5/6 Running stress tests...")
    total_portfolio = df["loan_amount_usd"].sum()
    base_default_rate = df["default"].mean()
    stress_results = run_stress_tests(base_default_rate, total_portfolio)

    for s in stress_results:
        print(f"    {s['scenario']:15s} PD={s['stressed_pd']:.1%} EL=${s['expected_loss_usd']:,.0f}")

    # Stressed default rates by score band
    stress_by_band = {}
    for name, scenario in SCENARIOS.items():
        band_data = stressed_default_rates_by_band(
            test_scores, result["y_test"].values, scenario["default_multiplier"]
        )
        stress_by_band[name] = band_data.to_dict(orient="records")

    print("6/6 Exporting JSON for dashboard...")

    # Dataset summary
    dataset_summary = {
        "total_records": len(df),
        "default_rate": round(df["default"].mean(), 4),
        "total_defaults": int(df["default"].sum()),
        "total_portfolio_usd": round(total_portfolio, 2),
        "avg_loan_usd": round(df["loan_amount_usd"].mean(), 2),
        "median_loan_usd": round(df["loan_amount_usd"].median(), 2),
        "countries": df["country"].nunique(),
        "sectors": df["sector"].nunique(),
        "train_size": len(result["X_train"]),
        "test_size": len(result["X_test"]),
    }

    # Country breakdown
    country_stats = (
        df.groupby("country")
        .agg(
            count=("default", "size"),
            defaults=("default", "sum"),
            avg_loan=("loan_amount_usd", "mean"),
        )
        .reset_index()
    )
    country_stats["default_rate"] = (country_stats["defaults"] / country_stats["count"]).round(4)
    country_stats["avg_loan"] = country_stats["avg_loan"].round(2)

    # Sector breakdown
    sector_stats = (
        df.groupby("sector")
        .agg(count=("default", "size"), defaults=("default", "sum"))
        .reset_index()
    )
    sector_stats["default_rate"] = (sector_stats["defaults"] / sector_stats["count"]).round(4)

    # IV summary for chart
    iv_chart = iv_summary[["feature", "iv", "strength"]].to_dict(orient="records")

    # WoE details for top features
    top_woe = {}
    for feat in selected[:8]:
        feat_data = woe_details[woe_details["feature"] == feat][
            ["bin", "count", "events", "non_events", "event_rate", "woe"]
        ].round(4)
        top_woe[feat] = feat_data.to_dict(orient="records")

    # Scorecard coefficients
    scorecard_info = {
        "target_score": 600,
        "target_odds": 50,
        "pdo": 20,
        "factor": round(result["factor"], 4),
        "offset": round(result["offset"], 4),
        "intercept": round(result["intercept"], 6),
        "features": [
            {
                "feature": f.replace("_woe", ""),
                "coefficient": round(c, 6),
            }
            for f, c in result["coefficients"].items()
        ],
    }

    # Validation metrics
    validation = {
        "train_gini": train_gini,
        "test_gini": test_gini,
        "train_ks": train_ks,
        "test_ks": test_ks,
        "psi": psi_result,
        "roc_curve": roc.to_dict(orient="records"),
        "ks_curve": ks_curve.to_dict(orient="records"),
        "score_distribution": score_dist.to_dict(orient="records"),
    }

    # Full export
    output = {
        "dataset": dataset_summary,
        "country_stats": country_stats.to_dict(orient="records"),
        "sector_stats": sector_stats.to_dict(orient="records"),
        "iv_summary": iv_chart,
        "woe_details": top_woe,
        "scorecard": scorecard_info,
        "validation": validation,
        "stress_tests": stress_results,
        "stress_by_band": stress_by_band,
    }

    out_path = OUT_DIR / "pipeline_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n    Exported to {out_path}")
    print("    Pipeline complete.")


if __name__ == "__main__":
    main()
