export interface PipelineData {
  dataset: {
    total_records: number;
    default_rate: number;
    total_defaults: number;
    total_portfolio_usd: number;
    avg_loan_usd: number;
    median_loan_usd: number;
    countries: number;
    sectors: number;
    train_size: number;
    test_size: number;
  };
  country_stats: Array<{
    country: string;
    count: number;
    defaults: number;
    avg_loan: number;
    default_rate: number;
  }>;
  sector_stats: Array<{
    sector: string;
    count: number;
    defaults: number;
    default_rate: number;
  }>;
  iv_summary: Array<{
    feature: string;
    iv: number;
    strength: string;
  }>;
  woe_details: Record<
    string,
    Array<{
      bin: string;
      count: number;
      events: number;
      non_events: number;
      event_rate: number;
      woe: number;
    }>
  >;
  scorecard: {
    target_score: number;
    target_odds: number;
    pdo: number;
    factor: number;
    offset: number;
    intercept: number;
    features: Array<{ feature: string; coefficient: number }>;
  };
  validation: {
    train_gini: number;
    test_gini: number;
    train_ks: { ks: number; threshold: number | null; fpr_at_ks: number; tpr_at_ks: number };
    test_ks: { ks: number; threshold: number | null; fpr_at_ks: number; tpr_at_ks: number };
    psi: { psi: number; interpretation: string; bins: Array<Record<string, unknown>> };
    roc_curve: Array<{ fpr: number; tpr: number }>;
    ks_curve: Array<{ probability: number; cum_bad_pct: number; cum_good_pct: number; ks_gap: number }>;
    score_distribution: Array<{ band: string; count: number; defaults: number; default_rate: number }>;
  };
  stress_tests: Array<{
    scenario: string;
    description: string;
    income_shock_pct: number;
    stressed_pd: number;
    expected_loss_usd: number;
    capital_required_usd: number;
    loss_rate_pct: number;
  }>;
  stress_by_band: Record<
    string,
    Array<{
      band: string;
      count: number;
      base_defaults: number;
      base_default_rate: number;
      stressed_default_rate: number;
    }>
  >;
}
