# Credit Risk Scorecard

Basel II-compliant credit scorecard for West African microfinance. Logistic regression with WoE encoding, built on 12,000 synthetic loans calibrated to microfinance risk profiles in the region.

Why build this: most credit risk tooling assumes the data and risk distribution of mature Western markets. West African microfinance has different risk drivers — mobile money usage, agricultural income seasonality, informal employment — so the feature importance looks different.

## Pipeline

1. **Data cleaning** — handle informal income fields, missing collateral data, outlier capping at 1st/99th percentile
2. **WoE/IV feature selection** — bins continuous variables, calculates Information Value per feature. Drops anything below IV 0.02
3. **Logistic regression** — fit on WoE-transformed features, convert coefficients to Basel II integer scorecard points (PDO = 20, base score = 600)
4. **Validation** — Gini 0.56, KS 0.42 on holdout. PSI < 0.1 across validation windows (stable)
5. **Stress testing** — shift default rate +50% (economic stress), +100% (severe), shift feature distributions, re-score the book

## Stack

- Python — pandas, scikit-learn, scipy, matplotlib
- Next.js + Recharts — scorecard UI and stress test visualiser

## Running

```bash
pip install -r requirements.txt
jupyter notebook notebooks/

# Dashboard
cd dashboard
npm install && npm run dev
```

## Results

| Metric | Value |
|--------|-------|
| Gini coefficient | 0.56 |
| KS statistic | 0.42 |
| PSI (population stability) | < 0.10 |
| AUC-ROC | 0.78 |

## Live

[credit-risk-ab.vercel.app](https://credit-risk-ab.vercel.app)

