"""Tests for the scorecard mathematics.

These assert the identities a credit scorecard has to satisfy, not the figures
this particular synthetic book happens to produce. Pinning Gini to 0.268 would
break the moment the generator is retuned and would catch no real defect; the
PDO property below would catch a broken points conversion immediately.
"""

import numpy as np
import pandas as pd
import pytest

from src.scorecard import PDO, TARGET_ODDS, TARGET_SCORE, build_scorecard, compute_scores
from src.stress_test import SCENARIOS, run_stress_tests
from src.validation import gini_coefficient, ks_statistic, population_stability_index
from src.woe_iv import compute_all_woe_iv, compute_woe_iv, woe_transform


@pytest.fixture(scope="module")
def book():
    """A small book where one feature predicts default and one is noise."""
    rng = np.random.default_rng(0)
    n = 3000
    signal = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-(-1.8 + 1.4 * signal)))
    return pd.DataFrame({
        "dti_ratio": signal,
        "noise": rng.normal(size=n),
        "default": rng.binomial(1, p),
    })


# --------------------------------------------------------------- WoE and IV

def test_bins_account_for_every_row(book):
    t = compute_woe_iv(book, "dti_ratio")
    assert t["count"].sum() == len(book)
    assert (t["events"] + t["non_events"] == t["count"]).all()


def test_woe_is_the_log_ratio_of_the_two_distributions(book):
    t = compute_woe_iv(book, "dti_ratio")
    expected = np.log(t["dist_non_events"] / t["dist_events"])
    assert np.allclose(t["woe"], expected)


def test_information_value_components_are_never_negative(book):
    """(a - b) ln(a/b) is non-negative for any positive a, b. A negative
    component means the distributions or the log were built wrong."""
    t = compute_woe_iv(book, "dti_ratio")
    assert (t["iv_component"] >= -1e-12).all()


def test_a_predictive_feature_outscores_noise(book):
    _, iv = compute_all_woe_iv(book, ["dti_ratio", "noise"])
    by_feature = dict(zip(iv["feature"], iv["iv"]))
    assert by_feature["dti_ratio"] > by_feature["noise"]


def test_iv_strength_labels_match_the_industry_bands(book):
    _, iv = compute_all_woe_iv(book, ["dti_ratio", "noise"])
    for _, row in iv.iterrows():
        v, label = row["iv"], row["strength"]
        if v < 0.02:
            assert label == "Not predictive"
        elif v < 0.10:
            assert label == "Weak"
        elif v < 0.30:
            assert label == "Medium"
        elif v < 0.50:
            assert label == "Strong"
        else:
            assert label == "Suspicious"


def test_noise_carries_almost_no_information(book):
    assert compute_woe_iv(book, "noise")["iv_component"].sum() < 0.02


# ------------------------------------------------------- points conversion

def test_the_factor_and_offset_follow_from_pdo_and_the_target():
    factor = PDO / np.log(2)
    offset = TARGET_SCORE - factor * np.log(TARGET_ODDS)
    assert factor == pytest.approx(28.8539, abs=1e-3)
    assert offset == pytest.approx(TARGET_SCORE - factor * np.log(TARGET_ODDS))


def test_doubling_the_odds_moves_the_score_by_exactly_pdo(book):
    """The defining property of a points scorecard. The score is an affine
    function of the log-odds with slope -factor, and factor = PDO / ln 2, so a
    change of ln 2 in the linear predictor has to move the score by PDO points
    and by nothing else."""
    details, _ = compute_all_woe_iv(book, ["dti_ratio"])
    woe = woe_transform(book, details, ["dti_ratio"])
    feats = [c for c in woe.columns if c.endswith("_woe")]
    sc = build_scorecard(woe, feats)

    beta = sc["coefficients"][feats[0]]
    shift = np.log(2) / beta          # move the linear predictor by exactly ln 2

    base = woe[feats].iloc[:50].copy()
    bumped = base.copy()
    bumped[feats[0]] += shift

    s0 = compute_scores(base, feats, sc["coefficients"], sc["intercept"], sc["factor"], sc["offset"])
    s1 = compute_scores(bumped, feats, sc["coefficients"], sc["intercept"], sc["factor"], sc["offset"])
    assert np.allclose(s1 - s0, -PDO)


def test_the_score_is_affine_in_the_linear_predictor(book):
    """Equal steps in the predictor must produce equal steps in the score. If
    they do not, the conversion has a non-linearity it should not have."""
    details, _ = compute_all_woe_iv(book, ["dti_ratio"])
    woe = woe_transform(book, details, ["dti_ratio"])
    feats = [c for c in woe.columns if c.endswith("_woe")]
    sc = build_scorecard(woe, feats)

    row = woe[feats].iloc[[0]]
    steps = [row.assign(**{feats[0]: row[feats[0]] + k * 0.1}) for k in range(5)]
    scores = [
        compute_scores(s, feats, sc["coefficients"], sc["intercept"], sc["factor"], sc["offset"])[0]
        for s in steps
    ]
    diffs = np.diff(scores)
    assert np.allclose(diffs, diffs[0])


# ------------------------------------------------------------- validation

def test_gini_is_two_auc_minus_one():
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9, 0.15, 0.7, 0.3, 0.6])
    from sklearn.metrics import roc_auc_score
    assert gini_coefficient(y, p) == pytest.approx(2 * roc_auc_score(y, p) - 1, abs=1e-4)


def test_a_perfect_ranker_scores_one_and_a_coin_flip_scores_zero():
    y = np.array([0] * 50 + [1] * 50)
    assert gini_coefficient(y, y.astype(float)) == pytest.approx(1.0)
    rng = np.random.default_rng(1)
    assert abs(gini_coefficient(y, rng.random(100))) < 0.25


def test_ks_stays_inside_the_unit_interval(book):
    rng = np.random.default_rng(2)
    ks = ks_statistic(book["default"].to_numpy(), rng.random(len(book)))
    assert 0.0 <= ks["ks"] <= 1.0


def test_psi_of_a_distribution_against_itself_is_zero():
    rng = np.random.default_rng(3)
    x = rng.normal(600, 40, 5000)
    assert population_stability_index(x, x)["psi"] < 1e-6


def test_psi_grows_as_the_distributions_separate():
    rng = np.random.default_rng(4)
    base = rng.normal(600, 40, 5000)
    near = rng.normal(605, 40, 5000)
    far = rng.normal(700, 40, 5000)
    assert (
        population_stability_index(base, near)["psi"]
        < population_stability_index(base, far)["psi"]
    )


# ------------------------------------------------------------ stress tests

def test_every_scenario_is_at_least_as_bad_as_baseline():
    assert SCENARIOS["Baseline"]["default_multiplier"] == 1.0
    others = [s["default_multiplier"] for k, s in SCENARIOS.items() if k != "Baseline"]
    assert all(m > 1.0 for m in others)


def test_expected_loss_rises_with_severity():
    results = run_stress_tests(base_default_rate=0.13, total_portfolio_usd=9_000_000)
    els = [r["expected_loss"] for r in results] if "expected_loss" in results[0] else None
    if els is None:
        key = next(k for k in results[0] if "loss" in k.lower())
        els = [r[key] for r in results]
    assert els == sorted(els)
