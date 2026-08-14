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

# Fraction of the book, by vintage, used for fitting.
TIME_SPLIT_QUANTILE = 0.7

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
    # Split by vintage, not at random. A random split puts loans from the same
    # months on both sides, so the holdout is drawn from the period the model
    # was fitted on and every stability measure looks perfect by construction.
    # Training on the earlier book and scoring the later one is the test a
    # scorecard actually faces in production.
    cutoff = df["vintage"].quantile(TIME_SPLIT_QUANTILE)
    train_idx = df.index[df["vintage"] <= cutoff]
    test_idx = df.index[df["vintage"] > cutoff]
    print(
        f"    Time split at vintage {int(cutoff)}: "
        f"train {len(train_idx)} loans (default {df.loc[train_idx, 'default'].mean():.1%}), "
        f"test {len(test_idx)} loans (default {df.loc[test_idx, 'default'].mean():.1%})"
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

    # PSI between the score distribution on the fitting book and on the later
    # book. This is now a real drift measure: the two populations are different
    # months, not two halves of one shuffle.
    #
    # It measures drift in THIS generator, whose deterioration is a stated
    # assumption (loosening leverage plus a macro shock in the back third), not
    # a measurement of any real portfolio.
    psi_result = population_stability_index(train_scores, test_scores)
    psi_result["scope"] = "earlier vintages vs later vintages"

    # Worth reporting next to the PSI, because on this book they disagree and
    # the disagreement is the point. PSI is computed on the SCORE distribution:
    # it answers "do the applicants look different", not "are they defaulting
    # more". Here the score distribution barely moves while realised defaults
    # rise by about a third, because the deterioration comes from a macro shock
    # that no feature in the scorecard observes. A monitoring setup watching
    # PSI alone would have seen nothing and called the model stable.
    train_dr = float(df.loc[train_idx, "default"].mean())
    test_dr = float(df.loc[test_idx, "default"].mean())
    psi_result["train_default_rate"] = round(train_dr, 4)
    psi_result["test_default_rate"] = round(test_dr, 4)
    psi_result["default_rate_shift"] = round(test_dr - train_dr, 4)

    print(f"    PSI: {psi_result['psi']:.4f} ({psi_result['interpretation']}, {psi_result['scope']})")
    print(
        f"    Realised default rate: {train_dr:.1%} -> {test_dr:.1%} "
        f"({test_dr - train_dr:+.1%}). PSI does not see this: it measures the "
        f"score distribution, not the outcome."
    )

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
