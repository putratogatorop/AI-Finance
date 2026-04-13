"use client";

import { useState } from "react";
import {
  ResponsiveContainer,
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
  ReferenceLine,
  LabelList,
} from "recharts";
import { formatPercent } from "@/lib/format";
import type { PeriodBreakdown } from "@/types";

interface ProfitConsistencyChartProps {
  monthly: PeriodBreakdown[];
  weekly: PeriodBreakdown[];
  compact?: boolean;
}

export function ProfitConsistencyChart({
  monthly,
  weekly,
  compact = false,
}: ProfitConsistencyChartProps) {
  const [granularity, setGranularity] = useState<"monthly" | "weekly">(
    "monthly"
  );

  const allData = granularity === "monthly" ? monthly : weekly;
  const data = compact ? allData.slice(-6) : allData;
  const height = compact ? 200 : 350;

  if (data.length === 0) {
    return (
      <div
        className="flex items-center justify-center rounded-lg border border-[var(--border)] bg-slate-800/50 p-8"
        style={{ height }}
      >
        <p className="text-sm text-slate-500">No data available</p>
      </div>
    );
  }

  return (
    <div>
      {/* Toggle */}
      <div className="mb-4 flex items-center justify-between">
        {!compact && (
          <p className="text-xs text-slate-500">
            {data.filter((d) => d.return_pct > 0).length}/{data.length}{" "}
            periods profitable
          </p>
        )}
        <div className="ml-auto flex rounded-lg border border-[var(--border)] bg-slate-800/50 p-0.5">
          <button
            onClick={() => setGranularity("monthly")}
            className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
              granularity === "monthly"
                ? "bg-brand-600 text-white"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Monthly
          </button>
          <button
            onClick={() => setGranularity("weekly")}
            className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
              granularity === "weekly"
                ? "bg-brand-600 text-white"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Weekly
          </button>
        </div>
      </div>

      {/* Chart */}
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart
          data={data}
          margin={{ top: 20, right: 20, bottom: 5, left: 10 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis
            dataKey="label"
            tick={{ fill: "#94a3b8", fontSize: compact ? 10 : 12 }}
            tickLine={{ stroke: "#475569" }}
            axisLine={{ stroke: "#475569" }}
          />
          <YAxis
            yAxisId="left"
            tick={{ fill: "#94a3b8", fontSize: 12 }}
            tickLine={{ stroke: "#475569" }}
            axisLine={{ stroke: "#475569" }}
            tickFormatter={(v: number) => `${v.toFixed(1)}%`}
          />
          <YAxis
            yAxisId="right"
            orientation="right"
            domain={[0, 100]}
            tick={{ fill: "#94a3b8", fontSize: 12 }}
            tickLine={{ stroke: "#475569" }}
            axisLine={{ stroke: "#475569" }}
            tickFormatter={(v: number) => `${v}%`}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#1e293b",
              border: "1px solid #334155",
              borderRadius: "8px",
              color: "#f1f5f9",
            }}
            formatter={(value: number, name: string) => {
              if (name === "Return %") return [formatPercent(value), name];
              if (name === "Win Rate") return [`${value.toFixed(1)}%`, name];
              return [value, name];
            }}
            labelFormatter={(label: string) => label}
          />
          <ReferenceLine yAxisId="left" y={0} stroke="#475569" />
          <Bar
            yAxisId="left"
            dataKey="return_pct"
            name="Return %"
            radius={[4, 4, 0, 0]}
          >
            {data.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={entry.return_pct >= 0 ? "#22c55e" : "#ef4444"}
                opacity={0.8}
              />
            ))}
            {!compact && (
              <LabelList
                dataKey="trades"
                position="top"
                fill="#64748b"
                fontSize={10}
                formatter={(v: number) => `${v}t`}
              />
            )}
          </Bar>
          <Line
            yAxisId="right"
            type="monotone"
            dataKey="win_rate"
            name="Win Rate"
            stroke="#facc15"
            strokeWidth={2}
            dot={{ r: 3, fill: "#facc15" }}
            activeDot={{ r: 5 }}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
