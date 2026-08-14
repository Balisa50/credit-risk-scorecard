"""
Generate synthetic but realistic West African microfinance loan data.
Distributions calibrated to typical microfinance portfolios in the region.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

N = 12_000  # total loan records

COUNTRIES = ["Gambia", "Senegal", "Ghana", "Nigeria", "Sierra Leone"]
COUNTRY_WEIGHTS = [0.30, 0.20, 0.20, 0.20, 0.10]

SECTORS = ["Agriculture", "Petty trade", "Services", "Manufacturing", "Livestock", "Fishing"]
SECTOR_WEIGHTS = [0.25, 0.30, 0.20, 0.10, 0.08, 0.07]

LOAN_PURPOSE = ["Working capital", "Equipment", "Inventory", "Expansion", "Emergency", "Education"]
PURPOSE_WEIGHTS = [0.35, 0.15, 0.20, 0.15, 0.10, 0.05]

GENDER = ["M", "F"]
GENDER_WEIGHTS = [0.45, 0.55]  # microfinance skews female


# Book history. Loans are spread across 36 monthly vintages so the portfolio
# has a time axis, which is what a scorecard actually gets validated against:
# you fit on the book you had and score the book that arrives next.
N_VINTAGES = 36
BOOK_START = "2022-01-01"


def generate() -> pd.DataFrame:
    # Vintage index 0..35, weighted so the book grows over time the way a
    # lending portfolio does rather than arriving all at once.
    vintage_weights = np.linspace(0.6, 1.4, N_VINTAGES)
    vintage_weights = vintage_weights / vintage_weights.sum()
    vintage = RNG.choice(N_VINTAGES, size=N, p=vintage_weights)
    origination_date = pd.to_datetime(BOOK_START) + pd.to_timedelta(
        vintage * 30, unit="D"
    )

    # Drift, stated plainly because it is an assumption of this generator and
    # not a measurement of any real portfolio. Two things move across the book,
    # both of which are ordinary in microfinance:
    #   1. underwriting loosens as the book grows, so later vintages carry more
    #      leverage (a higher loan-to-income multiple);
    #   2. a macro deterioration in the back third lifts default risk for
    #      everyone originated in that window.
    # Without something like this the portfolio is stationary, a PSI between
    # vintages is near zero by construction, and a time-based holdout tells you
    # nothing that a random split did not.
    vintage_frac = vintage / (N_VINTAGES - 1)
    leverage_drift = 1.0 + 0.35 * vintage_frac
    macro_stress = 0.45 * np.clip((vintage_frac - 0.65) / 0.35, 0, 1)

    age = RNG.integers(18, 65, size=N)
    gender = RNG.choice(GENDER, size=N, p=GENDER_WEIGHTS)
    country = RNG.choice(COUNTRIES, size=N, p=COUNTRY_WEIGHTS)
    sector = RNG.choice(SECTORS, size=N, p=SECTOR_WEIGHTS)
    purpose = RNG.choice(LOAN_PURPOSE, size=N, p=PURPOSE_WEIGHTS)

    # Monthly income in USD (typical microfinance borrower)
    monthly_income = np.clip(
        RNG.lognormal(mean=4.8, sigma=0.6, size=N), 40, 2000
    ).round(2)

    # Loan amount: 1x-8x monthly income
    loan_multiplier = RNG.uniform(1, 8, size=N) * leverage_drift
    loan_amount = (monthly_income * loan_multiplier).round(2)

    # Loan term in months
    loan_term = RNG.choice([3, 6, 9, 12, 18, 24], size=N, p=[0.10, 0.25, 0.15, 0.30, 0.12, 0.08])

    # Annual interest rate (microfinance rates are high: 15-45%)
    interest_rate = np.clip(RNG.normal(28, 8, size=N), 12, 50).round(1)

    # Number of previous loans with this MFI
    previous_loans = RNG.poisson(1.5, size=N)

    # Number of previous defaults
    previous_defaults = np.minimum(
        RNG.binomial(previous_loans, 0.12),
        previous_loans
    )

    # Days past due on worst previous loan
    dpd_history = np.where(
        previous_defaults > 0,
        RNG.integers(30, 180, size=N),
        RNG.choice([0, 0, 0, 0, 7, 14, 21], size=N)
    )

    # Group lending flag (common in microfinance)
    group_lending = RNG.choice([0, 1], size=N, p=[0.40, 0.60])

    # Has collateral
    has_collateral = RNG.choice([0, 1], size=N, p=[0.65, 0.35])

    # Debt-to-income ratio
    monthly_payment = (loan_amount * (interest_rate / 100 / 12)) / (
        1 - (1 + interest_rate / 100 / 12) ** (-loan_term)
    )
    dti = (monthly_payment / monthly_income).round(4)

    # Years at current business
    years_in_business = np.clip(RNG.exponential(4, size=N), 0, 30).round(1)

    # Dependents
    dependents = RNG.poisson(3, size=N)

    # --- Default probability model (latent) ---
    logit = (
        -2.5
        + 0.8 * (dti > 0.40).astype(float)
        + 0.6 * (previous_defaults > 0).astype(float)
        + 0.4 * (dpd_history > 30).astype(float)
        - 0.5 * (group_lending == 1).astype(float)
        - 0.4 * (has_collateral == 1).astype(float)
        + 0.3 * (years_in_business < 1).astype(float)
        - 0.3 * (previous_loans > 3).astype(float)
        + 0.2 * (age < 25).astype(float)
        + 0.15 * (loan_amount > 1000).astype(float)
        + 0.25 * (interest_rate > 35).astype(float)
        - 0.2 * (gender == "F").astype(float)
        + macro_stress
        + RNG.normal(0, 0.3, size=N)
    )
    prob_default = 1 / (1 + np.exp(-logit))
    default = (RNG.random(size=N) < prob_default).astype(int)

    df = pd.DataFrame({
        "loan_id": np.arange(1, N + 1),
        "origination_date": origination_date,
        "vintage": vintage,
        "country": country,
        "gender": gender,
        "age": age,
        "dependents": dependents,
        "monthly_income_usd": monthly_income,
        "years_in_business": years_in_business,
        "sector": sector,
        "loan_purpose": purpose,
        "loan_amount_usd": loan_amount,
        "loan_term_months": loan_term,
        "interest_rate_pct": interest_rate,
        "dti_ratio": dti,
        "previous_loans": previous_loans,
        "previous_defaults": previous_defaults,
        "dpd_history_days": dpd_history,
        "group_lending": group_lending,
        "has_collateral": has_collateral,
        "default": default,
    })
    return df


if __name__ == "__main__":
    df = generate()
    out = Path(__file__).parent / "loans.csv"
    df.to_csv(out, index=False)
    print(f"Generated {len(df)} records -> {out}")
    print(f"Default rate: {df['default'].mean():.1%}")
    by_v = df.groupby(df["origination_date"].dt.to_period("Q"))["default"].mean()
    print("Default rate by quarter:")
    print(by_v.round(3).to_string())
    print(f"Columns: {list(df.columns)}")
