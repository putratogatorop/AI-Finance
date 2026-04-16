"use client";

import { useEffect, useState, useCallback } from "react";
import { formatDateTime } from "@/lib/format";
import type { AuditEntry } from "@/types";

export default function AuditPage() {
  const [entries, setEntries] = useState<AuditEntry[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [offset, setOffset] = useState(0);
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [filter, setFilter] = useState<{
    event_type: string;
    asset: string;
  }>({
    event_type: "",
    asset: "",
  });
  const limit = 50;

  const fetchAudit = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      params.set("limit", limit.toString());
      params.set("offset", offset.toString());
      if (filter.event_type) params.set("event_type", filter.event_type);
      if (filter.asset) params.set("asset", filter.asset);

      const response = await fetch(`/api/audit?${params.toString()}`);
      const result = await response.json();
      setEntries(result.data || []);
      setTotal(result.total || 0);
    } catch (error) {
      console.error("Failed to fetch audit log:", error);
    } finally {
      setLoading(false);
    }
  }, [offset, filter]);

  useEffect(() => {
    fetchAudit();
  }, [fetchAudit]);

  const totalPages = Math.ceil(total / limit);
  const currentPage = Math.floor(offset / limit) + 1;

  const eventTypeColors: Record<string, string> = {
    signal_generated: "text-blue-400 bg-blue-500/10",
    signal_acknowledged: "text-green-400 bg-green-500/10",
    position_opened: "text-emerald-400 bg-emerald-500/10",
    position_closed: "text-slate-400 bg-slate-500/10",
    stop_loss_triggered: "text-red-400 bg-red-500/10",
    model_retrained: "text-purple-400 bg-purple-500/10",
    data_fetched: "text-cyan-400 bg-cyan-500/10",
  };

  return (
    <div className="space-y-6">
      {/* Filters */}
      <div className="card">
        <div className="flex flex-wrap items-center gap-4">
          <div>
            <label className="block text-xs text-slate-400 mb-1">Event Type</label>
            <select
              value={filter.event_type}
              onChange={(e) => {
                setFilter({ ...filter, event_type: e.target.value });
                setOffset(0);
              }}
              className="input-field w-48"
            >
              <option value="">All Events</option>
              <option value="signal_generated">Signal Generated</option>
              <option value="signal_acknowledged">Signal Acknowledged</option>
              <option value="position_opened">Position Opened</option>
              <option value="position_closed">Position Closed</option>
              <option value="stop_loss_triggered">Stop Loss Triggered</option>
              <option value="model_retrained">Model Retrained</option>
              <option value="data_fetched">Data Fetched</option>
            </select>
          </div>
          <div>
            <label className="block text-xs text-slate-400 mb-1">Asset</label>
            <input
              type="text"
              value={filter.asset}
              onChange={(e) => {
                setFilter({ ...filter, asset: e.target.value.toUpperCase() });
                setOffset(0);
              }}
              placeholder="e.g. BTC"
              className="input-field w-28"
            />
          </div>
          <div className="flex items-end">
            <span className="text-sm text-slate-400">
              {total} event{total !== 1 ? "s" : ""} found
            </span>
          </div>
        </div>
      </div>

      {/* Audit log table */}
      <div className="card overflow-hidden p-0">
        {loading ? (
          <div className="p-6 space-y-3">
            {[1, 2, 3, 4, 5].map((i) => (
              <div key={i} className="h-10 animate-pulse rounded bg-slate-700" />
            ))}
          </div>
        ) : entries.length === 0 ? (
          <div className="p-6">
            <p className="text-center text-sm text-slate-500">
              No audit entries match your filters.
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-[var(--border)]">
                  <th className="table-header w-12">#</th>
                  <th className="table-header">Timestamp</th>
                  <th className="table-header">Event Type</th>
                  <th className="table-header">Asset</th>
                  <th className="table-header">Details</th>
                </tr>
              </thead>
              <tbody>
                {entries.map((entry) => {
                  const colorClass =
                    eventTypeColors[entry.event_type] ||
                    "text-slate-400 bg-slate-500/10";
                  const isExpanded = expandedId === entry.id;

                  return (
                    <tr
                      key={entry.id}
                      className="border-b border-[var(--border)] transition-colors hover:bg-slate-800/50 cursor-pointer"
                      onClick={() =>
                        setExpandedId(isExpanded ? null : entry.id)
                      }
                    >
                      <td className="table-cell text-xs text-slate-500 font-mono">
                        {entry.id}
                      </td>
                      <td className="table-cell text-xs">
                        {formatDateTime(entry.created_at)}
                      </td>
                      <td className="table-cell">
                        <span
                          className={`rounded-md px-2 py-0.5 text-xs font-medium ${colorClass}`}
                        >
                          {entry.event_type.replace(/_/g, " ")}
                        </span>
                      </td>
                      <td className="table-cell font-medium text-slate-200">
                        {entry.asset || "-"}
                      </td>
                      <td className="table-cell">
                        {isExpanded && entry.parsed_details ? (
                          <pre className="mt-1 max-w-md overflow-x-auto rounded bg-slate-800 p-2 text-xs text-slate-300 font-mono">
                            {JSON.stringify(entry.parsed_details, null, 2)}
                          </pre>
                        ) : entry.parsed_details ? (
                          <span className="text-xs text-slate-400 truncate block max-w-xs">
                            {JSON.stringify(entry.parsed_details).substring(0, 80)}
                            {JSON.stringify(entry.parsed_details).length > 80 ? "..." : ""}
                          </span>
                        ) : (
                          <span className="text-xs text-slate-500">
                            {entry.details
                              ? entry.details.substring(0, 80) +
                                (entry.details.length > 80 ? "..." : "")
                              : "-"}
                          </span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between border-t border-[var(--border)] px-6 py-3">
            <p className="text-sm text-slate-400">
              Page {currentPage} of {totalPages}
            </p>
            <div className="flex gap-2">
              <button
                onClick={() => setOffset(Math.max(0, offset - limit))}
                disabled={offset === 0}
                className="btn-secondary text-xs"
              >
                Previous
              </button>
              <button
                onClick={() => setOffset(offset + limit)}
                disabled={offset + limit >= total}
                className="btn-secondary text-xs"
              >
                Next
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
