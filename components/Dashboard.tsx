"use client";

import type { PipelineData } from "@/lib/types";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, AreaChart, Area, Legend, Cell,
} from "recharts";

const CYAN = "#00F0FF";
const PINK = "#ec4899";
const GREEN = "#22c55e";
const AMBER = "#eab308";
const RED = "#ef4444";
const MUTED = "#737373";

function Card({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={`rounded-2xl border border-white/10 bg-[#141414] p-6 ${className}`}>
      {children}
    </div>
  );
}

function Metric({ value, label, color = "text-white" }: { value: string; label: string; color?: string }) {
  return (
    <div className="flex flex-col gap-1">
      <span className={`font-mono text-2xl font-bold tabular-nums md:text-3xl ${color}`}>{value}</span>
      <span className="text-xs uppercase tracking-wider text-neutral-500">{label}</span>
    </div>
  );
}

function SectionTitle({ tag, title }: { tag: string; title: string }) {
  return (
    <div className="mb-8 flex flex-col gap-2">
      <span className="font-mono text-xs uppercase tracking-[0.2em] text-[#00F0FF]">{tag}</span>
      <h2 className="text-2xl font-semibold tracking-tight md:text-3xl">{title}</h2>
    </div>
  );
}

function strengthColor(s: string): string {
  if (s === "Strong") return GREEN;
  if (s === "Medium") return AMBER;
  if (s === "Weak") return MUTED;
  return RED;
}

export function Dashboard({ data }: { data: PipelineData }) {
  const d = data.dataset;

  return (
    <main className="mx-auto max-w-7xl space-y-16 px-4 py-12 md:px-8">
      {/* Header */}
      <header className="space-y-4">
        <span className="inline-block rounded-full border border-[#00F0FF]/30 bg-[#00F0FF]/10 px-3 py-1 font-mono text-xs uppercase tracking-wider text-[#00F0FF]">
          Credit Risk Modeling
        </span>
        <h1 className="text-4xl font-bold tracking-tight md:text-5xl">
          West Africa Microfinance
          <br />
          <span className="text-neutral-500">Credit Risk Scorecard</span>
        </h1>
        <p className="max-w-2xl text-neutral-400">
          Basel II-compliant scorecard built on {d.total_records.toLocaleString()} synthetic
          microfinance loans across {d.countries} West African countries. Weight of Evidence
          feature selection, logistic regression with points conversion, validated with Gini,
          KS statistic, and Population Stability Index.
        </p>
      </header>

      {/* Key Metrics */}
      <section>
        <Card>
          <div className="grid grid-cols-2 gap-6 md:grid-cols-4 lg:grid-cols-6">
            <Metric value={d.total_records.toLocaleString()} label="Loan Records" />
            <Metric value={`${(d.default_rate * 100).toFixed(1)}%`} label="Default Rate" color="text-red-400" />
            <Metric value={`$${(d.total_portfolio_usd / 1e6).toFixed(1)}M`} label="Portfolio Size" />
            <Metric value={`$${d.avg_loan_usd.toFixed(0)}`} label="Avg Loan" />
            <Metric value={data.validation.test_gini.toFixed(3)} label="Gini (Test)" color="text-[#00F0FF]" />
            <Metric value={data.validation.test_ks.ks.toFixed(3)} label="KS Stat (Test)" color="text-[#00F0FF]" />
          </div>
        </Card>
      </section>

      {/* Information Value */}
      <section>
        <SectionTitle tag="Feature Selection" title="Information Value (IV)" />
        <Card>
          <p className="mb-4 text-sm text-neutral-400">
            IV measures the predictive power of each feature. Features with IV &gt; 0.02 are
            selected for the scorecard. Strong predictors (IV &gt; 0.3) may indicate over-fitting.
          </p>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.iv_summary} layout="vertical" margin={{ left: 120 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
                <XAxis type="number" tick={{ fill: "#737373", fontSize: 12 }} />
                <YAxis
                  type="category"
                  dataKey="feature"
                  tick={{ fill: "#a3a3a3", fontSize: 12 }}
                  width={110}
                />
                <Tooltip
                  contentStyle={{ background: "#1a1a1a", border: "1px solid #333", borderRadius: 8 }}
                  labelStyle={{ color: "#e5e5e5" }}
                />
                <Bar dataKey="iv" radius={[0, 4, 4, 0]}>
                  {data.iv_summary.map((entry, i) => (
                    <Cell key={i} fill={strengthColor(entry.strength)} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-4 flex flex-wrap gap-4 text-xs text-neutral-500">
            <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-green-500" /> Strong (&gt;0.3)</span>
            <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-amber-500" /> Medium (0.1-0.3)</span>
            <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-neutral-500" /> Weak (0.02-0.1)</span>
          </div>
        </Card>
      </section>

      {/* Scorecard */}
      <section>
        <SectionTitle tag="Scorecard" title="Logistic Regression Coefficients" />
        <div className="grid gap-6 md:grid-cols-2">
          <Card>
            <h3 className="mb-4 text-sm font-semibold uppercase tracking-wider text-neutral-400">
              Scorecard Parameters
            </h3>
            <div className="space-y-3 font-mono text-sm">
              <div className="flex justify-between"><span className="text-neutral-500">Target Score</span><span>{data.scorecard.target_score}</span></div>
              <div className="flex justify-between"><span className="text-neutral-500">Target Odds</span><span>{data.scorecard.target_odds}:1</span></div>
              <div className="flex justify-between"><span className="text-neutral-500">PDO</span><span>{data.scorecard.pdo}</span></div>
              <div className="flex justify-between"><span className="text-neutral-500">Factor</span><span>{data.scorecard.factor.toFixed(4)}</span></div>
              <div className="flex justify-between"><span className="text-neutral-500">Offset</span><span>{data.scorecard.offset.toFixed(4)}</span></div>
              <div className="flex justify-between"><span className="text-neutral-500">Intercept</span><span>{data.scorecard.intercept.toFixed(6)}</span></div>
            </div>
          </Card>
          <Card>
            <h3 className="mb-4 text-sm font-semibold uppercase tracking-wider text-neutral-400">
              Feature Coefficients
            </h3>
            <div className="space-y-2">
              {data.scorecard.features.map((f) => (
                <div key={f.feature} className="flex items-center justify-between gap-2 text-sm">
                  <span className="truncate text-neutral-300">{f.feature}</span>
                  <span className={`font-mono tabular-nums ${f.coefficient > 0 ? "text-red-400" : "text-green-400"}`}>
                    {f.coefficient > 0 ? "+" : ""}{f.coefficient.toFixed(4)}
                  </span>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </section>

      {/* Validation */}
      <section>
        <SectionTitle tag="Model Validation" title="Gini, KS, ROC, PSI" />
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
          <Card>
            <Metric value={data.validation.train_gini.toFixed(3)} label="Train Gini" />
          </Card>
          <Card>
            <Metric value={data.validation.test_gini.toFixed(3)} label="Test Gini" color="text-[#00F0FF]" />
          </Card>
          <Card>
            <Metric value={data.validation.test_ks.ks.toFixed(3)} label="Test KS" color="text-[#00F0FF]" />
          </Card>
          <Card>
            <Metric
              value={data.validation.psi.psi.toFixed(4)}
              label={`PSI (${data.validation.psi.interpretation})`}
              color={data.validation.psi.psi < 0.1 ? "text-green-400" : "text-amber-400"}
            />
          </Card>
        </div>

        <div className="mt-6 grid gap-6 md:grid-cols-2">
          {/* ROC Curve */}
          <Card>
            <h3 className="mb-4 text-sm font-semibold uppercase tracking-wider text-neutral-400">
              ROC Curve
            </h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={data.validation.roc_curve}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
                  <XAxis dataKey="fpr" tick={{ fill: "#737373", fontSize: 11 }} label={{ value: "FPR", position: "bottom", fill: "#737373", fontSize: 11 }} />
                  <YAxis tick={{ fill: "#737373", fontSize: 11 }} label={{ value: "TPR", angle: -90, position: "insideLeft", fill: "#737373", fontSize: 11 }} />
                  <Tooltip contentStyle={{ background: "#1a1a1a", border: "1px solid #333", borderRadius: 8 }} />
                  <Area type="monotone" dataKey="tpr" stroke={CYAN} fill={CYAN} fillOpacity={0.15} strokeWidth={2} />
                  <Line type="linear" dataKey="fpr" stroke="#555" strokeDasharray="4 4" dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </Card>

          {/* KS Curve */}
          <Card>
            <h3 className="mb-4 text-sm font-semibold uppercase tracking-wider text-neutral-400">
              KS Curve
            </h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data.validation.ks_curve}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
                  <XAxis dataKey="probability" tick={{ fill: "#737373", fontSize: 11 }} />
                  <YAxis tick={{ fill: "#737373", fontSize: 11 }} />
                  <Tooltip contentStyle={{ background: "#1a1a1a", border: "1px solid #333", borderRadius: 8 }} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Line type="monotone" dataKey="cum_good_pct" name="Cum Good %" stroke={GREEN} strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="cum_bad_pct" name="Cum Bad %" stroke={RED} strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </div>
      </section>

      {/* Score Distribution */}
      <section>
        <SectionTitle tag="Score Analysis" title="Score Distribution by Band" />
        <Card>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.validation.score_distribution}>
                <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
                <XAxis dataKey="band" tick={{ fill: "#737373", fontSize: 10 }} angle={-20} textAnchor="end" height={60} />
                <YAxis tick={{ fill: "#737373", fontSize: 11 }} />
                <Tooltip contentStyle={{ background: "#1a1a1a", border: "1px solid #333", borderRadius: 8 }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="count" name="Loans" fill={CYAN} fillOpacity={0.7} radius={[4, 4, 0, 0]} />
                <Bar dataKey="defaults" name="Defaults" fill={RED} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </section>

      {/* Country and Sector breakdown */}
      <section>
        <SectionTitle tag="Portfolio" title="Country and Sector Analysis" />
        <div className="grid gap-6 md:grid-cols-2">
          <Card>
            <h3 className="mb-4 text-sm font-semibold uppercase tracking-wider text-neutral-400">
              Default Rate by Country
            </h3>
            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={data.country_stats}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
                  <XAxis dataKey="country" tick={{ fill: "#a3a3a3", fontSize: 12 }} />
                  <YAxis tick={{ fill: "#737373", fontSize: 11 }} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                  <Tooltip contentStyle={{ background: "#1a1a1a", border: "1px solid #333", borderRadius: 8 }} formatter={(v) => `${(Number(v) * 100).toFixed(1)}%`} />
                  <Bar dataKey="default_rate" fill={PINK} radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>
          <Card>
            <h3 className="mb-4 text-sm font-semibold uppercase tracking-wider text-neutral-400">
              Default Rate by Sector
            </h3>
            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={data.sector_stats} layout="vertical" margin={{ left: 80 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
                  <XAxis type="number" tick={{ fill: "#737373", fontSize: 11 }} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                  <YAxis type="category" dataKey="sector" tick={{ fill: "#a3a3a3", fontSize: 12 }} width={75} />
                  <Tooltip contentStyle={{ background: "#1a1a1a", border: "1px solid #333", borderRadius: 8 }} formatter={(v) => `${(Number(v) * 100).toFixed(1)}%`} />
                  <Bar dataKey="default_rate" fill={AMBER} radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </div>
      </section>

      {/* Stress Testing */}
      <section>
        <SectionTitle tag="Stress Testing" title="Economic Scenario Analysis" />
        <Card>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10 text-left text-xs uppercase tracking-wider text-neutral-500">
                  <th className="pb-3 pr-4">Scenario</th>
                  <th className="pb-3 pr-4">Income Shock</th>
                  <th className="pb-3 pr-4">Stressed PD</th>
                  <th className="pb-3 pr-4">Expected Loss</th>
                  <th className="pb-3 pr-4">Capital Required</th>
                  <th className="pb-3">Loss Rate</th>
                </tr>
              </thead>
              <tbody>
                {data.stress_tests.map((s, i) => {
                  const colors = ["text-green-400", "text-amber-400", "text-orange-400", "text-red-400"];
                  return (
                    <tr key={s.scenario} className="border-b border-white/5">
                      <td className={`py-3 pr-4 font-medium ${colors[i]}`}>{s.scenario}</td>
                      <td className="py-3 pr-4 font-mono tabular-nums">{s.income_shock_pct}%</td>
                      <td className="py-3 pr-4 font-mono tabular-nums">{(s.stressed_pd * 100).toFixed(1)}%</td>
                      <td className="py-3 pr-4 font-mono tabular-nums">${s.expected_loss_usd.toLocaleString()}</td>
                      <td className="py-3 pr-4 font-mono tabular-nums">${s.capital_required_usd.toLocaleString()}</td>
                      <td className="py-3 font-mono tabular-nums">{s.loss_rate_pct}%</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div className="mt-4 space-y-1 text-xs text-neutral-500">
            {data.stress_tests.map((s) => (
              <p key={s.scenario}><strong className="text-neutral-400">{s.scenario}:</strong> {s.description}</p>
            ))}
          </div>
        </Card>
      </section>

      {/* Methodology Footer */}
      <footer className="border-t border-white/10 pt-8 text-sm text-neutral-500">
        <div className="grid gap-6 md:grid-cols-3">
          <div>
            <h4 className="mb-2 font-semibold text-neutral-300">Methodology</h4>
            <p>Weight of Evidence (WoE) binning, Information Value (IV) feature selection, L2-regularized logistic regression, Basel II points-based scorecard conversion.</p>
          </div>
          <div>
            <h4 className="mb-2 font-semibold text-neutral-300">Validation</h4>
            <p>Gini coefficient, Kolmogorov-Smirnov statistic, Population Stability Index (PSI), train/test split with stratification.</p>
          </div>
          <div>
            <h4 className="mb-2 font-semibold text-neutral-300">Data</h4>
            <p>Synthetic dataset of {d.total_records.toLocaleString()} microfinance loans calibrated to West African lending patterns. Not real borrower data.</p>
          </div>
        </div>
        <p className="mt-6 text-neutral-600">Built by Abdoulie Balisa</p>
      </footer>
    </main>
  );
}
