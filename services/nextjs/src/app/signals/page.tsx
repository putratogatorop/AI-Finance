"use client";

import { useEffect, useState, useCallback } from "react";
import { SignalCard } from "@/components/signals/signal-card";
import type { SignalRecord, AcknowledgeStatus } from "@/types";

export default function SignalsPage() {
  const [signals, setSignals] = useState<SignalRecord[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [acknowledging, setAcknowledging] = useState<number | null>(null);
  const [filter, setFilter] = useState<{
    action: string;
    acknowledged: string;
    asset: string;
  }>({
    action: "",
    acknowledged: "false",
    asset: "",
  });

  const fetchSignals = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      if (filter.action) params.set("action", filter.action);
      if (filter.acknowledged)
        params.set("acknowledged", filter.acknowledged);
      if (filter.asset) params.set("asset", filter.asset);
      params.set("limit", "50");

      const response = await fetch(`/api/signals?${params.toString()}`);
      const result = await response.json();
      setSignals(result.data || []);
      setTotal(result.total || 0);
    } catch (error) {
      console.error("Failed to fetch signals:", error);
    } finally {
      setLoading(false);
    }
  }, [filter]);

  useEffect(() => {
    fetchSignals();
  }, [fetchSignals]);

  const handleAcknowledge = async (
    id: number,
    status: AcknowledgeStatus
  ) => {
    setAcknowledging(id);
    try {
      const response = await fetch(`/api/signals/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          acknowledged: true,
          acknowledge_status: status,
        }),
      });

      if (response.ok) {
        setSignals((prev) =>
          prev.map((s) =>
            s.id === id ? { ...s, acknowledged: true } : s
          )
        );
      }
    } catch (error) {
      console.error("Failed to acknowledge signal:", error);
    } finally {
      setAcknowledging(null);
    }
  };

  return (
    <div className="space-y-6">
      <div className="card">
        <div className="flex flex-wrap items-center gap-4">
          <div>
            <label className="block text-xs text-slate-400 mb-1">
              Action
            </label>
            <select
              value={filter.action}
              onChange={(e) =>
                setFilter({ ...filter, action: e.target.value })
              }
              className="input-field w-32"
            >
              <option value="">All</option>
              <option value="BUY">BUY</option>
              <option value="SELL">SELL</option>
              <option value="HOLD">HOLD</option>
              <option value="EXIT">EXIT</option>
            </select>
          </div>
          <div>
            <label className="block text-xs text-slate-400 mb-1">
              Status
            </label>
            <select
              value={filter.acknowledged}
              onChange={(e) =>
                setFilter({ ...filter, acknowledged: e.target.value })
              }
              className="input-field w-40"
            >
              <option value="">All</option>
              <option value="false">Unacknowledged</option>
              <option value="true">Acknowledged</option>
            </select>
          </div>
          <div>
            <label className="block text-xs text-slate-400 mb-1">
              Asset
            </label>
            <input
              type="text"
              value={filter.asset}
              onChange={(e) =>
                setFilter({
                  ...filter,
                  asset: e.target.value.toUpperCase(),
                })
              }
              placeholder="e.g. BTC"
              className="input-field w-28"
            />
          </div>
          <div className="flex items-end">
            <span className="text-sm text-slate-400">
              {total} signal{total !== 1 ? "s" : ""} found
            </span>
          </div>
        </div>
      </div>

      {loading ? (
        <div className="space-y-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="card animate-pulse">
              <div className="h-4 w-24 rounded bg-slate-700" />
              <div className="mt-4 grid grid-cols-4 gap-4">
                <div className="h-8 rounded bg-slate-700" />
                <div className="h-8 rounded bg-slate-700" />
                <div className="h-8 rounded bg-slate-700" />
                <div className="h-8 rounded bg-slate-700" />
              </div>
            </div>
          ))}
        </div>
      ) : signals.length === 0 ? (
        <div className="card">
          <p className="text-center text-sm text-slate-500">
            No signals match your filters. Try adjusting the criteria
            above.
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {signals.map((signal) => (
            <SignalCard
              key={signal.id}
              signal={signal}
              onAcknowledge={handleAcknowledge}
              isLoading={acknowledging === signal.id}
            />
          ))}
        </div>
      )}
    </div>
  );
}
