"use client";

import { useEffect, useState } from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
  LineChart,
  Line,
  Legend,
} from "recharts";
import { formatPercent } from "@/lib/format";

interface ModelResult {
  rank: number;
  name: string;
  tier: "baseline" | "ml";
  sharpe: number;
  total_return_pct: number;
  max_drawdown_pct: number;
  win_rate: number;
  profit_factor: number;
  trades: number;
  avg_return_pct: number;
  equity_curve: number[];
  auc?: number;
  accuracy_pct?: number;
  precision_pct?: number;
  signal_rate_pct?: number;
  confidence_distribution?: { bucket: string; count: number; hit_rate: number }[];
}

interface TournamentData {
  generated_at: string;
  total_samples: number;
  total_tokens: number;
  period: string;
  fee_rate_pct: number;
  models: ModelResult[];
}

const MODEL_COLORS: Record<string, string> = {
  "Buy & Hold": "#6b7280",
  "SMA50 Filter": "#8b5cf6",
  "Momentum (42-bar)": "#f59e0b",
  XGBoost: "#3b82f6",
  LightGBM: "#22c55e",
};

function getColor(name: string) {
  return MODEL_COLORS[name] || "#94a3b8";
}

export default function TournamentPage() {
  const [data, setData] = useState<TournamentData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchTournament() {
      try {
        const res = await fetch("/api/tournament");
        const json = await res.json();
        if (json.error && !json.data) {
          setError(json.error);
        } else {
          setData(json.data);
        }
      } catch {
        setError("Failed to fetch tournament data");
      } finally {
        setLoading(false);
      }
    }
    fetchTournament();
  }, []);

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="card animate-pulse">
          <div className="h-8 w-64 rounded bg-slate-700" />
          <div className="mt-4 h-64 rounded bg-slate-700" />
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="card">
        <p className="text-center text-sm text-slate-500">
          {error || "No tournament data available."}
        </p>
        <p className="mt-2 text-center text-xs text-slate-600">
          Run: python services/python/scripts/run_tournament.py
        </p>
      </div>
    );
  }

  const { models } = data;

  // Prepare bar chart data
  const sharpeData = models.map((m) => ({
    name: m.name,
    value: m.sharpe,
    tier: m.tier,
  }));

  const returnData = models.map((m) => ({
    name: m.name,
    value: m.total_return_pct,
    tier: m.tier,
  }));

  // Equity curves (normalize all to same x-axis length)
  const maxLen = Math.max(...models.map((m) => m.equity_curve.length));
  const equityLines = models
    .filter((m) => m.equity_curve.length > 0)
    .map((m) => {
      const step = m.equity_curve.length / maxLen;
      const resampled = Array.from({ length: maxLen }, (_, i) => {
        const idx = Math.min(Math.floor(i * step), m.equity_curve.length - 1);
        return m.equity_curve[idx];
      });
      return { name: m.name, data: resampled };
    });

  const equityChartData = Array.from({ length: maxLen }, (_, i) => {
    const point: Record<string, number> = { idx: i };
    for (const line of equityLines) {
      point[line.name] = line.data[i];
    }
    return point;
  });

  const mlModels = models.filter((m) => m.tier === "ml");

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="card">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold text-slate-100">
              ML Tournament
            </h1>
            <p className="mt-1 text-sm text-slate-400">
              {data.total_tokens} tokens, {data.total_samples.toLocaleString()} samples
            </p>
          </div>
          <div className="text-right">
            <p className="text-xs text-slate-500">Period</p>
            <p className="text-sm text-slate-300">{data.period}</p>
            <p className="mt-1 text-xs text-slate-600">
              Generated: {data.generated_at}
            </p>
          </div>
        </div>
      </div>

      {/* Leaderboard */}
      <div className="card overflow-hidden p-0">
        <div className="px-6 pt-6">
          <h2 className="text-lg font-semibold text-slate-100">
            Leaderboard
          </h2>
          <p className="text-xs text-slate-500">Ranked by Sharpe Ratio</p>
        </div>
        <div className="mt-4 overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-[var(--border)]">
                <th className="table-header">#</th>
                <th className="table-header">Model</th>
                <th className="table-header">Tier</th>
                <th className="table-header text-right">Sharpe</th>
                <th className="table-header text-right">Return</th>
                <th className="table-header text-right">Max DD</th>
                <th className="table-header text-right">Win Rate</th>
                <th className="table-header text-right">PF</th>
                <th className="table-header text-right">Trades</th>
                <th className="table-header text-right">Avg Ret</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr
                  key={m.name}
                  className="border-b border-[var(--border)] transition-colors hover:bg-slate-800/50"
                >
                  <td className="table-cell font-bold text-slate-300">
                    {m.rank}
                  </td>
                  <td className="table-cell">
                    <span
                      className="font-medium"
                      style={{ color: getColor(m.name) }}
                    >
                      {m.name}
                    </span>
                  </td>
                  <td className="table-cell">
                    <span
                      className={`inline-block rounded px-2 py-0.5 text-xs font-medium ${
                        m.tier === "ml"
                          ? "bg-blue-500/20 text-blue-400"
                          : "bg-slate-500/20 text-slate-400"
                      }`}
                    >
                      {m.tier === "ml" ? "ML" : "Baseline"}
                    </span>
                  </td>
                  <td
                    className={`table-cell text-right font-mono ${
                      m.sharpe > 0 ? "text-green-400" : "text-red-400"
                    }`}
                  >
                    {m.sharpe.toFixed(3)}
                  </td>
                  <td
                    className={`table-cell text-right font-mono ${
                      m.total_return_pct > 0
                        ? "text-green-400"
                        : "text-red-400"
                    }`}
                  >
                    {formatPercent(m.total_return_pct)}
                  </td>
                  <td className="table-cell text-right font-mono text-red-400">
                    {formatPercent(m.max_drawdown_pct)}
                  </td>
                  <td className="table-cell text-right font-mono text-slate-300">
                    {m.win_rate.toFixed(1)}%
                  </td>
                  <td
                    className={`table-cell text-right font-mono ${
                      m.profit_factor >= 1
                        ? "text-green-400"
                        : "text-red-400"
                    }`}
                  >
                    {m.profit_factor.toFixed(3)}
                  </td>
                  <td className="table-cell text-right font-mono text-slate-400">
                    {m.trades.toLocaleString()}
                  </td>
                  <td
                    className={`table-cell text-right font-mono ${
                      m.avg_return_pct > 0
                        ? "text-green-400"
                        : "text-red-400"
                    }`}
                  >
                    {m.avg_return_pct.toFixed(3)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Sharpe Comparison Bar Chart */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="card">
          <h2 className="mb-4 text-lg font-semibold text-slate-100">
            Sharpe Ratio
          </h2>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={sharpeData} layout="vertical" margin={{ left: 100 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis
                type="number"
                tick={{ fill: "#94a3b8", fontSize: 12 }}
                axisLine={{ stroke: "#475569" }}
              />
              <YAxis
                type="category"
                dataKey="name"
                tick={{ fill: "#94a3b8", fontSize: 12 }}
                axisLine={{ stroke: "#475569" }}
                width={95}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1e293b",
                  border: "1px solid #334155",
                  borderRadius: "8px",
                  color: "#f1f5f9",
                }}
              />
              <Bar dataKey="value" name="Sharpe" radius={[0, 4, 4, 0]}>
                {sharpeData.map((entry, i) => (
                  <Cell
                    key={i}
                    fill={getColor(entry.name)}
                    opacity={0.85}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <h2 className="mb-4 text-lg font-semibold text-slate-100">
            Total Return (%)
          </h2>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={returnData} layout="vertical" margin={{ left: 100 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis
                type="number"
                tick={{ fill: "#94a3b8", fontSize: 12 }}
                axisLine={{ stroke: "#475569" }}
              />
              <YAxis
                type="category"
                dataKey="name"
                tick={{ fill: "#94a3b8", fontSize: 12 }}
                axisLine={{ stroke: "#475569" }}
                width={95}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1e293b",
                  border: "1px solid #334155",
                  borderRadius: "8px",
                  color: "#f1f5f9",
                }}
                formatter={(value: number) => [`${value.toFixed(2)}%`, "Return"]}
              />
              <Bar dataKey="value" name="Return %" radius={[0, 4, 4, 0]}>
                {returnData.map((entry, i) => (
                  <Cell
                    key={i}
                    fill={entry.value >= 0 ? getColor(entry.name) : "#ef4444"}
                    opacity={0.85}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Equity Curves */}
      <div className="card">
        <h2 className="mb-4 text-lg font-semibold text-slate-100">
          Equity Curves (Additive PnL)
        </h2>
        <p className="mb-2 text-xs text-slate-500">
          Starting capital = 1.0, additive PnL per period (non-compounding, accounts for overlapping 5-day returns)
        </p>
        <ResponsiveContainer width="100%" height={350}>
          <LineChart data={equityChartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis
              dataKey="idx"
              tick={false}
              axisLine={{ stroke: "#475569" }}
              label={{
                value: "Time",
                position: "insideBottom",
                fill: "#94a3b8",
                fontSize: 12,
              }}
            />
            <YAxis
              tick={{ fill: "#94a3b8", fontSize: 12 }}
              axisLine={{ stroke: "#475569" }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#1e293b",
                border: "1px solid #334155",
                borderRadius: "8px",
                color: "#f1f5f9",
              }}
              formatter={(value: number) => [value.toFixed(4), ""]}
            />
            <Legend />
            {equityLines.map((line) => (
              <Line
                key={line.name}
                type="monotone"
                dataKey={line.name}
                stroke={getColor(line.name)}
                dot={false}
                strokeWidth={2}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* ML Model Details */}
      {mlModels.length > 0 && (
        <div className="card">
          <h2 className="mb-4 text-lg font-semibold text-slate-100">
            ML Model Details
          </h2>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            {mlModels.map((m) => (
              <div
                key={m.name}
                className="rounded-lg border border-[var(--border)] bg-slate-800/50 p-4"
              >
                <h3
                  className="text-base font-semibold"
                  style={{ color: getColor(m.name) }}
                >
                  {m.name}
                </h3>
                <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
                  {m.auc !== undefined && (
                    <Stat label="AUC" value={m.auc.toFixed(4)} />
                  )}
                  {m.accuracy_pct !== undefined && (
                    <Stat label="Accuracy" value={`${m.accuracy_pct}%`} />
                  )}
                  {m.precision_pct !== undefined && (
                    <Stat label="Precision" value={`${m.precision_pct}%`} />
                  )}
                  {m.signal_rate_pct !== undefined && (
                    <Stat label="Signal Rate" value={`${m.signal_rate_pct}%`} />
                  )}
                </div>

                {m.confidence_distribution && (
                  <div className="mt-3">
                    <p className="text-xs text-slate-500">
                      Confidence vs Hit Rate
                    </p>
                    <div className="mt-1 space-y-1">
                      {m.confidence_distribution.map((b) => (
                        <div
                          key={b.bucket}
                          className="flex items-center text-xs"
                        >
                          <span className="w-16 text-slate-400">
                            {b.bucket}
                          </span>
                          <div className="mx-2 h-3 flex-1 overflow-hidden rounded bg-slate-700">
                            <div
                              className="h-full rounded"
                              style={{
                                width: `${Math.min(b.hit_rate, 100)}%`,
                                backgroundColor: getColor(m.name),
                                opacity: 0.7,
                              }}
                            />
                          </div>
                          <span className="w-12 text-right text-slate-300">
                            {b.hit_rate}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Footer Note */}
      <div className="text-center text-xs text-slate-600">
        Fee: {data.fee_rate_pct}% per trade | Returns are additive (non-compounding) |
        Forward returns capped at [-25%, +50%]
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-xs text-slate-500">{label}</p>
      <p className="font-mono text-slate-200">{value}</p>
    </div>
  );
}
