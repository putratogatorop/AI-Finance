"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { PriceChart } from "@/components/charts/price-chart";
import {
  formatUSD,
  formatIDR,
  formatPercent,
  formatConfidence,
  formatDateTime,
  getPnlColor,
  getUrgencyLevel,
  getUrgencyBadgeClass,
} from "@/lib/format";
import type { AssetDetail } from "@/types";

export default function AssetDetailPage() {
  const params = useParams();
  const symbol = (params.symbol as string)?.toUpperCase() || "";
  const [asset, setAsset] = useState<AssetDetail | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!symbol) return;

    async function fetchAsset() {
      setLoading(true);
      try {
        const response = await fetch(`/api/assets/${symbol.toLowerCase()}`);
        const result = await response.json();
        setAsset(result.data || null);
      } catch (error) {
        console.error("Failed to fetch asset:", error);
      } finally {
        setLoading(false);
      }
    }

    fetchAsset();
  }, [symbol]);

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="card animate-pulse">
          <div className="h-8 w-32 rounded bg-slate-700" />
          <div className="mt-2 h-12 w-48 rounded bg-slate-700" />
        </div>
        <div className="card animate-pulse">
          <div className="h-96 rounded bg-slate-700" />
        </div>
      </div>
    );
  }

  if (!asset) {
    return (
      <div className="card">
        <p className="text-center text-sm text-slate-500">
          Asset &quot;{symbol}&quot; not found. No data available.
        </p>
        <div className="mt-4 text-center">
          <Link href="/" className="text-brand-400 hover:text-brand-300 text-sm">
            Back to Overview
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-3xl font-bold text-slate-100">{asset.symbol}</h1>
          <p className="mt-1 text-2xl font-mono text-slate-200">
            {formatUSD(asset.current_price)}
          </p>
          {asset.fundamentals?.category && (
            <p className="mt-1 text-sm text-slate-500">
              {asset.fundamentals.category}
              {asset.fundamentals.market_cap_rank && (
                <span className="ml-2">
                  Rank #{asset.fundamentals.market_cap_rank}
                </span>
              )}
            </p>
          )}
        </div>
        <Link
          href="/portfolio"
          className="btn-secondary text-sm"
        >
          View Portfolio
        </Link>
      </div>

      {/* Price chart */}
      <div className="card">
        <h2 className="mb-4 text-lg font-semibold text-slate-100">
          Price History (90 Days)
        </h2>
        <PriceChart data={asset.price_history} />
      </div>

      {/* Fundamentals + Model scores grid */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Fundamentals */}
        {asset.fundamentals && (
          <div className="card">
            <h2 className="mb-4 text-lg font-semibold text-slate-100">
              Fundamentals
            </h2>
            <dl className="space-y-3">
              <div className="flex justify-between">
                <dt className="text-sm text-slate-400">Market Cap</dt>
                <dd className="text-sm font-mono text-slate-200">
                  {asset.fundamentals.market_cap
                    ? `$${(asset.fundamentals.market_cap / 1e9).toFixed(1)}B`
                    : "N/A"}
                </dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-sm text-slate-400">Market Cap Rank</dt>
                <dd className="text-sm font-mono text-slate-200">
                  #{asset.fundamentals.market_cap_rank ?? "N/A"}
                </dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-sm text-slate-400">24h Volume</dt>
                <dd className="text-sm font-mono text-slate-200">
                  {asset.fundamentals.total_volume_24h
                    ? `$${(asset.fundamentals.total_volume_24h / 1e9).toFixed(2)}B`
                    : "N/A"}
                </dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-sm text-slate-400">Circulating Supply</dt>
                <dd className="text-sm font-mono text-slate-200">
                  {asset.fundamentals.circulating_supply
                    ? `${(asset.fundamentals.circulating_supply / 1e6).toFixed(2)}M`
                    : "N/A"}
                </dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-sm text-slate-400">Category</dt>
                <dd className="text-sm text-slate-200">
                  {asset.fundamentals.category ?? "N/A"}
                </dd>
              </div>
            </dl>
          </div>
        )}

        {/* Model scores */}
        <div className="card">
          <h2 className="mb-4 text-lg font-semibold text-slate-100">
            Model Scores
          </h2>
          <dl className="space-y-3">
            {(
              [
                ["XGBoost", asset.model_scores.xgboost],
                ["LightGBM", asset.model_scores.lightgbm],
                ["LSTM", asset.model_scores.lstm],
                ["Ensemble", asset.model_scores.ensemble],
              ] as [string, number | null][]
            ).map(([name, score]) => (
              <div key={name} className="flex items-center justify-between">
                <dt className="text-sm text-slate-400">{name}</dt>
                <dd className="flex items-center gap-3">
                  {score !== null ? (
                    <>
                      <div className="h-2 w-24 overflow-hidden rounded-full bg-slate-700">
                        <div
                          className="h-full rounded-full bg-brand-500"
                          style={{ width: `${Math.abs(score) * 100}%` }}
                        />
                      </div>
                      <span className="text-sm font-mono text-slate-200 w-12 text-right">
                        {formatConfidence(score)}
                      </span>
                    </>
                  ) : (
                    <span className="text-sm text-slate-500">No data</span>
                  )}
                </dd>
              </div>
            ))}
          </dl>
        </div>
      </div>

      {/* Feature values */}
      {Object.keys(asset.features).length > 0 && (
        <div className="card">
          <h2 className="mb-4 text-lg font-semibold text-slate-100">
            Feature Values
          </h2>
          <div className="grid grid-cols-2 gap-x-6 gap-y-2 sm:grid-cols-3 lg:grid-cols-4">
            {Object.entries(asset.features).map(([key, value]) => (
              <div key={key} className="flex justify-between border-b border-slate-800 py-1">
                <span className="text-xs text-slate-400 truncate mr-2">{key}</span>
                <span className="text-xs font-mono text-slate-300">
                  {typeof value === "number" ? value.toFixed(4) : String(value)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recent signals */}
      <div className="card overflow-hidden p-0">
        <div className="px-6 pt-6">
          <h2 className="text-lg font-semibold text-slate-100">
            Recent Signals
          </h2>
        </div>
        {asset.recent_signals.length === 0 ? (
          <div className="p-6">
            <p className="text-sm text-slate-500">No signals for this asset yet.</p>
          </div>
        ) : (
          <div className="mt-4 overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-[var(--border)]">
                  <th className="table-header">Date</th>
                  <th className="table-header">Action</th>
                  <th className="table-header">Confidence</th>
                  <th className="table-header">Expected Return</th>
                  <th className="table-header">Hold</th>
                  <th className="table-header">Agreement</th>
                  <th className="table-header">Urgency</th>
                </tr>
              </thead>
              <tbody>
                {asset.recent_signals.map((signal) => {
                  const urgency = getUrgencyLevel(signal);
                  return (
                    <tr
                      key={signal.id}
                      className="border-b border-[var(--border)] transition-colors hover:bg-slate-800/50"
                    >
                      <td className="table-cell text-xs">
                        {formatDateTime(signal.created_at)}
                      </td>
                      <td className="table-cell">
                        <span
                          className={
                            signal.action === "BUY"
                              ? "text-green-400 font-semibold"
                              : signal.action === "SELL" || signal.action === "EXIT"
                                ? "text-red-400 font-semibold"
                                : "text-yellow-400 font-semibold"
                          }
                        >
                          {signal.action}
                        </span>
                      </td>
                      <td className="table-cell font-mono">
                        {formatConfidence(signal.confidence)}
                      </td>
                      <td className={`table-cell font-mono ${getPnlColor(signal.expected_return_pct)}`}>
                        {formatPercent(signal.expected_return_pct)}
                      </td>
                      <td className="table-cell">{signal.suggested_hold_days}d</td>
                      <td className="table-cell">{signal.model_agreement}</td>
                      <td className="table-cell">
                        <span className={getUrgencyBadgeClass(urgency)}>
                          {urgency.toUpperCase()}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Trade history */}
      <div className="card overflow-hidden p-0">
        <div className="px-6 pt-6">
          <h2 className="text-lg font-semibold text-slate-100">
            Trade History
          </h2>
        </div>
        {asset.trade_history.length === 0 ? (
          <div className="p-6">
            <p className="text-sm text-slate-500">No trades for this asset yet.</p>
          </div>
        ) : (
          <div className="mt-4 overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-[var(--border)]">
                  <th className="table-header">Status</th>
                  <th className="table-header">Entry Price</th>
                  <th className="table-header">Exit Price</th>
                  <th className="table-header">Invested</th>
                  <th className="table-header">P&L (IDR)</th>
                  <th className="table-header">P&L (%)</th>
                  <th className="table-header">Opened</th>
                  <th className="table-header">Closed</th>
                </tr>
              </thead>
              <tbody>
                {asset.trade_history.map((trade) => (
                  <tr
                    key={trade.id}
                    className="border-b border-[var(--border)] transition-colors hover:bg-slate-800/50"
                  >
                    <td className="table-cell">
                      <span
                        className={`rounded-md px-2 py-0.5 text-xs font-semibold uppercase ${
                          trade.status === "open"
                            ? "bg-green-500/10 text-green-400"
                            : trade.status === "stopped"
                              ? "bg-red-500/10 text-red-400"
                              : "bg-slate-500/10 text-slate-400"
                        }`}
                      >
                        {trade.status}
                      </span>
                    </td>
                    <td className="table-cell font-mono">{formatUSD(trade.entry_price)}</td>
                    <td className="table-cell font-mono">
                      {trade.exit_price ? formatUSD(trade.exit_price) : "-"}
                    </td>
                    <td className="table-cell">{formatIDR(trade.entry_amount_idr)}</td>
                    <td className="table-cell">
                      {trade.pnl_idr !== null ? (
                        <span className={`font-mono ${getPnlColor(trade.pnl_idr)}`}>
                          {formatIDR(trade.pnl_idr)}
                        </span>
                      ) : (
                        <span className="text-slate-500">-</span>
                      )}
                    </td>
                    <td className="table-cell">
                      {trade.pnl_pct !== null ? (
                        <span className={`font-mono ${getPnlColor(trade.pnl_pct)}`}>
                          {formatPercent(trade.pnl_pct)}
                        </span>
                      ) : (
                        <span className="text-slate-500">-</span>
                      )}
                    </td>
                    <td className="table-cell text-xs">
                      {formatDateTime(trade.opened_at)}
                    </td>
                    <td className="table-cell text-xs">
                      {trade.closed_at ? formatDateTime(trade.closed_at) : "-"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
