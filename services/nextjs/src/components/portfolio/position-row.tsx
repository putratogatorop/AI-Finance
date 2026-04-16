"use client";

import Link from "next/link";
import { clsx } from "clsx";
import type { PortfolioPosition } from "@/types";
import {
  formatIDR,
  formatPercent,
  formatUSD,
  formatDateTime,
  getPnlColor,
} from "@/lib/format";

interface PositionRowProps {
  position: PortfolioPosition;
}

export function PositionRow({ position }: PositionRowProps) {
  const currentPrice = position.current_price ?? position.entry_price;
  const unrealizedPnlIdr =
    position.unrealized_pnl_idr ??
    (position.status === "open"
      ? Math.round(
          (currentPrice - position.entry_price) * position.quantity * 16000
        )
      : null);
  const unrealizedPnlPct =
    position.unrealized_pnl_pct ??
    (position.status === "open"
      ? ((currentPrice - position.entry_price) / position.entry_price) *
        100
      : null);

  const displayPnlIdr =
    position.status === "open" ? unrealizedPnlIdr : position.pnl_idr;
  const displayPnlPct =
    position.status === "open" ? unrealizedPnlPct : position.pnl_pct;

  const stopLossHitPct =
    ((currentPrice - position.stop_loss_price) /
      (position.entry_price - position.stop_loss_price)) *
    100;
  const stopLossProximity = Math.max(0, Math.min(100, 100 - stopLossHitPct));

  return (
    <tr className="border-b border-[var(--border)] transition-colors hover:bg-slate-800/50">
      <td className="table-cell">
        <Link
          href={`/assets/${position.asset.toLowerCase()}`}
          className="font-medium text-brand-400 hover:text-brand-300"
        >
          {position.asset}
        </Link>
      </td>
      <td className="table-cell">
        <span
          className={clsx(
            "rounded-md px-2 py-0.5 text-xs font-semibold uppercase",
            position.status === "open" &&
              "bg-green-500/10 text-green-400",
            position.status === "closed" &&
              "bg-slate-500/10 text-slate-400",
            position.status === "stopped" &&
              "bg-red-500/10 text-red-400"
          )}
        >
          {position.status}
        </span>
      </td>
      <td className="table-cell font-mono">
        {formatUSD(position.entry_price)}
      </td>
      <td className="table-cell font-mono">{formatUSD(currentPrice)}</td>
      <td className="table-cell">
        {formatIDR(position.entry_amount_idr)}
      </td>
      <td className="table-cell">
        {displayPnlIdr !== null && displayPnlIdr !== undefined ? (
          <span className={clsx("font-mono", getPnlColor(displayPnlIdr))}>
            {formatIDR(displayPnlIdr)}
          </span>
        ) : (
          <span className="text-slate-500">-</span>
        )}
      </td>
      <td className="table-cell">
        {displayPnlPct !== null && displayPnlPct !== undefined ? (
          <span className={clsx("font-mono", getPnlColor(displayPnlPct))}>
            {formatPercent(displayPnlPct)}
          </span>
        ) : (
          <span className="text-slate-500">-</span>
        )}
      </td>
      <td className="table-cell">
        {position.status === "open" ? (
          <div className="flex items-center gap-2">
            <div className="h-1.5 w-16 overflow-hidden rounded-full bg-slate-700">
              <div
                className={clsx(
                  "h-full rounded-full transition-all",
                  stopLossProximity > 60
                    ? "bg-red-500"
                    : stopLossProximity > 30
                      ? "bg-yellow-500"
                      : "bg-green-500"
                )}
                style={{ width: `${stopLossProximity}%` }}
              />
            </div>
            <span className="text-xs text-slate-500 font-mono">
              {formatUSD(position.stop_loss_price)}
            </span>
          </div>
        ) : (
          <span className="text-xs text-slate-500">-</span>
        )}
      </td>
      <td className="table-cell text-xs text-slate-500">
        {formatDateTime(position.opened_at)}
      </td>
    </tr>
  );
}
