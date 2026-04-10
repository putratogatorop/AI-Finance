"use client";

import { useEffect, useState } from "react";
import { ProfitConsistencyChart } from "@/components/charts/profit-consistency-chart";
import type { BacktestResult, PeriodBreakdown } from "@/types";
import Link from "next/link";

export function ScannerConsistency() {
  const [data, setData] = useState<{
    monthly: PeriodBreakdown[];
    weekly: PeriodBreakdown[];
  } | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const res = await fetch("/api/backtest");
        const json = await res.json();
        const result: BacktestResult | null = json.data || null;
        if (!result) return;

        setData({
          monthly: result.monthly.map((m) => ({
            label: m.month,
            return_idr: m.return_idr,
            return_pct: m.return_pct,
            trades: m.trades,
            win_rate: m.win_rate,
          })),
          weekly: result.weekly.map((w) => ({
            label: w.week,
            return_idr: w.return_idr,
            return_pct: w.return_pct,
            trades: w.trades,
            win_rate: w.win_rate,
          })),
        });
      } catch {
        // silently fail — supplementary widget
      }
    }
    load();
  }, []);

  if (!data || data.monthly.length === 0) return null;

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-2">
        <h2 className="text-lg font-semibold text-slate-100">
          Profit Consistency
        </h2>
        <Link
          href="/backtest"
          className="text-xs text-brand-400 hover:text-brand-300 transition-colors"
        >
          Full view -&gt;
        </Link>
      </div>
      <ProfitConsistencyChart
        monthly={data.monthly}
        weekly={data.weekly}
        compact
      />
    </div>
  );
}
