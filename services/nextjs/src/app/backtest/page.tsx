"use client";

import { useEffect, useState } from "react";
import { PerformanceChart } from "@/components/charts/performance-chart";
import {
  formatIDR,
  formatPercent,
  getPnlColor,
} from "@/lib/format";
import type { BacktestResult, PeriodBreakdown } from "@/types";
import { ProfitConsistencyChart } from "@/components/charts/profit-consistency-chart";
import { ConsistencyStats } from "@/components/charts/consistency-stats";

export default function BacktestPage() {
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchBacktest() {
      setLoading(true);
      try {
        const response = await fetch("/api/backtest");
        const data = await response.json();
        setResult(data.data || null);
      } catch (error) {
        console.error("Failed to fetch backtest:", error);
      } finally {
        setLoading(false);
      }
    }

    fetchBacktest();
  }, []);

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="card animate-pulse">
          <div className="h-8 w-48 rounded bg-slate-700" />
          <div className="mt-4 grid grid-cols-3 gap-4">
            {[1, 2, 3, 4, 5, 6].map((i) => (
              <div key={i} className="h-16 rounded bg-slate-700" />
            ))}
          </div>
        </div>
        <div className="card animate-pulse">
          <div className="h-64 rounded bg-slate-700" />
        </div>
      </div>
    );
  }

  if (!result || result.metrics.total_trades === 0) {
    return (
      <div className="card">
        <p className="text-center text-sm text-slate-500">
          No backtest data available. Results will appear here once closed trades exist in the database.
        </p>
      </div>
    );
  }

  const { metrics, monthly, equity_curve } = result;
  const { weekly } = result;

  const monthlyPeriods: PeriodBreakdown[] = monthly.map((m) => ({
    label: m.month,
    return_idr: m.return_idr,
    return_pct: m.return_pct,
    trades: m.trades,
    win_rate: m.win_rate,
  }));

  const weeklyPeriods: PeriodBreakdown[] = weekly.map((w) => ({
    label: w.week,
    return_idr: w.return_idr,
    return_pct: w.return_pct,
    trades: w.trades,
    win_rate: w.win_rate,
  }));

  return (
    <div className="space-y-6">
      {/* Metrics cards */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-3">
        <MetricCard
          title="Total Return"
          value={formatIDR(metrics.total_return_idr)}
          subtitle={formatPercent(metrics.total_return_pct)}
          valueClassName={getPnlColor(metrics.total_return_idr)}
        />
        <MetricCard
          title="Win Rate"
          value={formatPercent(metrics.win_rate, 1)}
          subtitle={`${metrics.winning_trades}W / ${metrics.losing_trades}L of ${metrics.total_trades} trades`}
        />
        <MetricCard
          title="Reward/Risk Ratio"
          value={metrics.reward_risk_ratio.toFixed(2)}
          subtitle="Average win / average loss"
        />
        <MetricCard
          title="Sharpe Ratio"
          value={metrics.sharpe_ratio.toFixed(2)}
          subtitle="Risk-adjusted return (annualized)"
        />
        <MetricCard
          title="Max Drawdown"
          value={formatPercent(-metrics.max_drawdown_pct, 1)}
          subtitle="Worst peak-to-trough drop"
          valueClassName="text-red-400"
        />
        <MetricCard
          title="Total Trades"
          value={metrics.total_trades.toString()}
          subtitle={`${metrics.winning_trades} winning, ${metrics.losing_trades} losing`}
        />
      </div>

      {/* Equity curve */}
      <div className="card">
        <h2 className="mb-4 text-lg font-semibold text-slate-100">
          Equity Curve
        </h2>
        <PerformanceChart data={equity_curve} startingValue={0} />
      </div>

      {/* Profit Consistency */}
      {monthlyPeriods.length > 0 && (
        <div className="card">
          <h2 className="mb-2 text-lg font-semibold text-slate-100">
            Profit Consistency
          </h2>
          <ProfitConsistencyChart
            monthly={monthlyPeriods}
            weekly={weeklyPeriods}
          />
        </div>
      )}

      {/* Consistency Stats */}
      {monthlyPeriods.length > 0 && (
        <ConsistencyStats
          data={monthlyPeriods}
          granularity="monthly"
        />
      )}

      {/* Monthly breakdown table */}
      {monthly.length > 0 && (
        <div className="card overflow-hidden p-0">
          <div className="px-6 pt-6">
            <h2 className="text-lg font-semibold text-slate-100">
              Monthly Breakdown
            </h2>
          </div>
          <div className="mt-4 overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-[var(--border)]">
                  <th className="table-header">Month</th>
                  <th className="table-header">Return (IDR)</th>
                  <th className="table-header">Return (%)</th>
                  <th className="table-header">Trades</th>
                  <th className="table-header">Win Rate</th>
                </tr>
              </thead>
              <tbody>
                {monthly.map((m) => (
                  <tr
                    key={m.month}
                    className="border-b border-[var(--border)] transition-colors hover:bg-slate-800/50"
                  >
                    <td className="table-cell font-medium text-slate-200">
                      {m.month}
                    </td>
                    <td className={`table-cell font-mono ${getPnlColor(m.return_idr)}`}>
                      {formatIDR(m.return_idr)}
                    </td>
                    <td className={`table-cell font-mono ${getPnlColor(m.return_pct)}`}>
                      {formatPercent(m.return_pct)}
                    </td>
                    <td className="table-cell">{m.trades}</td>
                    <td className="table-cell">{formatPercent(m.win_rate, 1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

function MetricCard({
  title,
  value,
  subtitle,
  valueClassName,
}: {
  title: string;
  value: string;
  subtitle: string;
  valueClassName?: string;
}) {
  return (
    <div className="card">
      <p className="text-sm text-slate-400">{title}</p>
      <p className={`mt-1 text-2xl font-bold ${valueClassName || "text-slate-100"}`}>
        {value}
      </p>
      <p className="mt-1 text-xs text-slate-500">{subtitle}</p>
    </div>
  );
}
