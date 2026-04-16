"use client";

import { clsx } from "clsx";
import type { SignalRecord, AcknowledgeStatus } from "@/types";
import {
  formatConfidence,
  formatPercent,
  formatDateTime,
  getUrgencyLevel,
  getUrgencyBadgeClass,
} from "@/lib/format";

interface SignalCardProps {
  signal: SignalRecord;
  onAcknowledge: (id: number, status: AcknowledgeStatus) => void;
  isLoading?: boolean;
}

export function SignalCard({
  signal,
  onAcknowledge,
  isLoading = false,
}: SignalCardProps) {
  const urgency = signal.urgency ?? getUrgencyLevel(signal);

  const actionColors: Record<string, string> = {
    BUY: "text-green-400 bg-green-500/10 border-green-500/30",
    SELL: "text-red-400 bg-red-500/10 border-red-500/30",
    HOLD: "text-yellow-400 bg-yellow-500/10 border-yellow-500/30",
    EXIT: "text-red-400 bg-red-500/10 border-red-500/30",
  };

  return (
    <div
      className={clsx(
        "card relative overflow-hidden transition-shadow hover:shadow-md",
        signal.acknowledged && "opacity-60"
      )}
    >
      <div
        className={clsx(
          "absolute left-0 top-0 h-full w-1",
          urgency === "green" && "bg-green-500",
          urgency === "yellow" && "bg-yellow-500",
          urgency === "red" && "bg-red-500"
        )}
      />

      <div className="pl-4">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <span className="text-lg font-bold text-slate-100">
              {signal.asset}
            </span>
            <span
              className={clsx(
                "rounded-md border px-2 py-0.5 text-xs font-semibold uppercase",
                actionColors[signal.action] ||
                  "text-slate-400 bg-slate-500/10 border-slate-500/30"
              )}
            >
              {signal.action}
            </span>
            <span className={getUrgencyBadgeClass(urgency)}>
              {urgency.toUpperCase()}
            </span>
          </div>
          <span className="text-xs text-slate-500">
            {formatDateTime(signal.created_at)}
          </span>
        </div>

        <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
          <div>
            <p className="text-xs text-slate-500">Confidence</p>
            <p className="text-sm font-medium text-slate-200">
              {formatConfidence(signal.confidence)}
            </p>
          </div>
          <div>
            <p className="text-xs text-slate-500">Expected Return</p>
            <p
              className={clsx(
                "text-sm font-medium",
                signal.expected_return_pct > 0
                  ? "text-green-400"
                  : "text-red-400"
              )}
            >
              {formatPercent(signal.expected_return_pct)}
            </p>
          </div>
          <div>
            <p className="text-xs text-slate-500">Hold Duration</p>
            <p className="text-sm font-medium text-slate-200">
              {signal.suggested_hold_days} days
            </p>
          </div>
          <div>
            <p className="text-xs text-slate-500">Model Agreement</p>
            <p className="text-sm font-medium text-slate-200">
              {signal.model_agreement}
            </p>
          </div>
        </div>

        <div className="mt-3">
          <p className="text-xs text-slate-500">
            Stop Loss:{" "}
            <span className="text-red-400">
              {formatPercent(signal.stop_loss_pct)}
            </span>
          </p>
        </div>

        {!signal.acknowledged && (
          <div className="mt-4 flex gap-2">
            <button
              onClick={() => onAcknowledge(signal.id, "acted")}
              disabled={isLoading}
              className="btn-primary text-xs"
            >
              Act
            </button>
            <button
              onClick={() => onAcknowledge(signal.id, "seen")}
              disabled={isLoading}
              className="btn-secondary text-xs"
            >
              Seen
            </button>
            <button
              onClick={() => onAcknowledge(signal.id, "skipped")}
              disabled={isLoading}
              className="btn-secondary text-xs"
            >
              Skip
            </button>
          </div>
        )}

        {signal.acknowledged && (
          <div className="mt-4">
            <span className="text-xs text-slate-500">Acknowledged</span>
          </div>
        )}
      </div>
    </div>
  );
}
