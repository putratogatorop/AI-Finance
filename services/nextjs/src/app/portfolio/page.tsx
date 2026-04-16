"use client";

import { useEffect, useState } from "react";
import { PositionRow } from "@/components/portfolio/position-row";
import { formatIDR, getPnlColor } from "@/lib/format";
import type { PortfolioPosition } from "@/types";

interface PortfolioSummary {
  open_positions: number;
  total_invested_idr: number;
  total_unrealized_pnl_idr: number;
  total_realized_pnl_idr: number;
  portfolio_value_idr: number;
}

export default function PortfolioPage() {
  const [positions, setPositions] = useState<PortfolioPosition[]>([]);
  const [summary, setSummary] = useState<PortfolioSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [statusFilter, setStatusFilter] = useState<string>("open");

  useEffect(() => {
    async function fetchPortfolio() {
      setLoading(true);
      try {
        const params = new URLSearchParams();
        if (statusFilter) params.set("status", statusFilter);
        const response = await fetch(
          `/api/portfolio?${params.toString()}`
        );
        const result = await response.json();
        setPositions(result.data || []);
        setSummary(result.summary || null);
      } catch (error) {
        console.error("Failed to fetch portfolio:", error);
      } finally {
        setLoading(false);
      }
    }

    fetchPortfolio();
  }, [statusFilter]);

  return (
    <div className="space-y-6">
      {summary && (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <div className="card">
            <p className="text-sm text-slate-400">Portfolio Value</p>
            <p className="mt-1 text-2xl font-bold text-slate-100">
              {formatIDR(summary.portfolio_value_idr)}
            </p>
          </div>
          <div className="card">
            <p className="text-sm text-slate-400">Open Positions</p>
            <p className="mt-1 text-2xl font-bold text-slate-100">
              {summary.open_positions} / 8
            </p>
            <p className="mt-1 text-xs text-slate-500">
              Invested: {formatIDR(summary.total_invested_idr)}
            </p>
          </div>
          <div className="card">
            <p className="text-sm text-slate-400">Unrealized P&L</p>
            <p
              className={`mt-1 text-2xl font-bold ${getPnlColor(summary.total_unrealized_pnl_idr)}`}
            >
              {formatIDR(summary.total_unrealized_pnl_idr)}
            </p>
          </div>
          <div className="card">
            <p className="text-sm text-slate-400">Realized P&L</p>
            <p
              className={`mt-1 text-2xl font-bold ${getPnlColor(summary.total_realized_pnl_idr)}`}
            >
              {formatIDR(summary.total_realized_pnl_idr)}
            </p>
          </div>
        </div>
      )}

      <div className="flex gap-2">
        {["open", "closed", "stopped", ""].map((status) => (
          <button
            key={status || "all"}
            onClick={() => setStatusFilter(status)}
            className={`rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
              statusFilter === status
                ? "bg-brand-600 text-white"
                : "bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-slate-200"
            }`}
          >
            {status
              ? status.charAt(0).toUpperCase() + status.slice(1)
              : "All"}
          </button>
        ))}
      </div>

      <div className="card overflow-hidden p-0">
        {loading ? (
          <div className="p-6">
            <div className="space-y-3">
              {[1, 2, 3].map((i) => (
                <div
                  key={i}
                  className="h-12 animate-pulse rounded bg-slate-700"
                />
              ))}
            </div>
          </div>
        ) : positions.length === 0 ? (
          <div className="p-6">
            <p className="text-center text-sm text-slate-500">
              No {statusFilter || ""} positions found.
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-[var(--border)]">
                  <th className="table-header">Asset</th>
                  <th className="table-header">Status</th>
                  <th className="table-header">Entry Price</th>
                  <th className="table-header">Current Price</th>
                  <th className="table-header">Invested</th>
                  <th className="table-header">P&L (IDR)</th>
                  <th className="table-header">P&L (%)</th>
                  <th className="table-header">Stop Loss</th>
                  <th className="table-header">Opened</th>
                </tr>
              </thead>
              <tbody>
                {positions.map((position) => (
                  <PositionRow
                    key={position.id}
                    position={position}
                  />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
