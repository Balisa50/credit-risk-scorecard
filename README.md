# Credit Risk Scorecard

Basel II-compliant credit scorecard for West African microfinance. Logistic regression with WoE encoding, built on 12,000 synthetic loans calibrated to microfinance risk profiles in the region.

Why build this: most credit risk tooling assumes the data and risk distribution of mature Western markets. West African microfinance has different risk drivers - mobile money usage, agricultural income seasonality, informal employment - so the feature importance looks different.

## Pipeline

1. **Data cleaning** - handle informal income fields, missing collateral data, outlier capping at 1st/99th percentile
2. **WoE/IV feature selection** - bins continuous variables, calculates Information Value per feature. Drops anything below IV 0.02
3. **Logistic regression** - fit on WoE-transformed features, convert coefficients to Basel II integer scorecard points (PDO = 20, base score = 600)
4. **Validation** - Gini 0.29, KS 0.23 on the time-based holdout. PSI 0.008 across validation windows (stable)
5. **Stress testing** - shift default rate +50% (economic stress), +100% (severe), shift feature distributions, re-score the book

## Stack

- Python - pandas, scikit-learn, scipy, matplotlib
- Next.js + Recharts - scorecard UI and stress test visualiser

## Running

```bash
pip install -r requirements.txt
jupyter notebook notebooks/

# Dashboard
cd dashboard
npm install && npm run dev
```

## Results

All figures below are the test-set values written by `pipeline/run_pipeline.py`
into `public/data/pipeline_results.json`, which is also what the dashboard renders.

| Metric | Train | Test | Industry threshold |
|--------|-------|------|--------------------|
| Gini coefficient | 0.35 | **0.29** | > 0.4 |
| KS statistic | 0.27 | **0.23** | > 0.3 |
| AUC-ROC | 0.68 | **0.65** | - |
| PSI (population stability) | - | **0.008** | < 0.1 |

Discrimination falls short of the usual thresholds, and that is a property of the
synthetic data rather than the modelling. No feature reaches Strong information
value: `previous_defaults` is the best at IV 0.13, and the eight selected
features sum to roughly 0.51 IV. A scorecard cannot separate better than its
features do. The stability and explainability layers (PSI, WoE bins, points
conversion) are the parts worth reading here.

## Live

[credit-risk-ab.vercel.app](https://credit-risk-ab.vercel.app)

