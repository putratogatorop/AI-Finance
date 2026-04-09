"use client";

import Link from "next/link";

interface Signal {
  id: number;
  symbol: string;
  direction: number;
  ml_prob: number;
  vol_ratio: number;
  price_change: number;
  signal_time: Date;
  status: string;
  pnl_pct: number | null;
  exit_reason: string | null;
}

export default function ScannerTable({ signals }: { signals: Signal[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full">
        <thead>
          <tr className="border-b border-slate-700">
            <th className="table-header">Time</th>
            <th className="table-header">Coin</th>
            <th className="table-header">Direction</th>
            <th className="table-header">ML Prob</th>
            <th className="table-header">Vol Ratio</th>
            <th className="table-header">Price Move</th>
            <th className="table-header">Status</th>
            <th className="table-header">PnL</th>
          </tr>
        </thead>
        <tbody>
          {signals.map((s) => (
            <tr key={s.id} className="border-b border-slate-800 hover:bg-slate-800/50">
              <td className="table-cell text-slate-400 font-mono text-xs">
                {new Date(s.signal_time).toLocaleString("en-GB", {
                  month: "short", day: "2-digit",
                  hour: "2-digit", minute: "2-digit",
                })}
              </td>
              <td className="table-cell">
                <Link
                  href={`/scanner/${s.symbol.replace("USDT", "")}`}
                  className="text-brand-400 hover:text-brand-300 font-semibold"
                >
                  {s.symbol.replace("USDT", "")}
                </Link>
              </td>
              <td className="table-cell">
                <span className={s.direction === 1
                  ? "badge-green text-xs px-2 py-0.5 rounded"
                  : "badge-red text-xs px-2 py-0.5 rounded"
                }>
                  {s.direction === 1 ? "LONG" : "SHORT"}
                </span>
              </td>
              <td className="table-cell font-mono">
                <span className={s.ml_prob >= 0.70
                  ? "text-signal-green font-bold"
                  : s.ml_prob >= 0.60
                  ? "text-yellow-400"
                  : "text-slate-400"
                }>
                  {(s.ml_prob * 100).toFixed(0)}%
                </span>
              </td>
              <td className="table-cell font-mono text-slate-300">
                {s.vol_ratio.toFixed(1)}x
              </td>
              <td className="table-cell font-mono">
                <span className={s.price_change > 0 ? "text-signal-green" : "text-signal-red"}>
                  {(s.price_change * 100).toFixed(1)}%
                </span>
              </td>
              <td className="table-cell">
                <span className={
                  s.status === "active" ? "text-yellow-400" :
                  s.status === "traded" ? "text-brand-400" :
                  "text-slate-500"
                }>
                  {s.status}
                </span>
              </td>
              <td className="table-cell font-mono">
                {s.pnl_pct != null ? (
                  <span className={s.pnl_pct > 0 ? "text-signal-green" : "text-signal-red"}>
                    {(s.pnl_pct * 100).toFixed(2)}%
                  </span>
                ) : (
                  <span className="text-slate-500">--</span>
                )}
              </td>
            </tr>
          ))}
          {signals.length === 0 && (
            <tr>
              <td colSpan={8} className="table-cell text-center text-slate-500 py-8">
                No signals in the last 7 days
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
