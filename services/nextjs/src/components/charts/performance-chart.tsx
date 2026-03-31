"use client";

import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from "recharts";
import { formatIDRCompact } from "@/lib/format";

interface PerformanceChartProps {
  data: { date: string; value: number }[];
  height?: number;
  startingValue?: number;
}

export function PerformanceChart({
  data,
  height = 350,
  startingValue,
}: PerformanceChartProps) {
  if (data.length === 0) {
    return (
      <div
        className="flex items-center justify-center rounded-lg border border-[var(--border)] bg-slate-800/50 p-8"
        style={{ height }}
      >
        <p className="text-sm text-slate-500">
          No performance data available
        </p>
      </div>
    );
  }

  const chartData = data.map((d) => ({
    date: new Date(d.date).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
    }),
    value: d.value,
  }));

  const lastValue = chartData[chartData.length - 1]?.value ?? 0;
  const firstValue = startingValue ?? chartData[0]?.value ?? 0;
  const isPositive = lastValue >= firstValue;

  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart
        data={chartData}
        margin={{ top: 5, right: 20, bottom: 5, left: 10 }}
      >
        <defs>
          <linearGradient
            id="performanceGradient"
            x1="0"
            y1="0"
            x2="0"
            y2="1"
          >
            <stop
              offset="5%"
              stopColor={isPositive ? "#22c55e" : "#ef4444"}
              stopOpacity={0.3}
            />
            <stop
              offset="95%"
              stopColor={isPositive ? "#22c55e" : "#ef4444"}
              stopOpacity={0}
            />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
        <XAxis
          dataKey="date"
          tick={{ fill: "#94a3b8", fontSize: 12 }}
          tickLine={{ stroke: "#475569" }}
          axisLine={{ stroke: "#475569" }}
        />
        <YAxis
          tick={{ fill: "#94a3b8", fontSize: 12 }}
          tickLine={{ stroke: "#475569" }}
          axisLine={{ stroke: "#475569" }}
          tickFormatter={(value: number) => formatIDRCompact(value)}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: "#1e293b",
            border: "1px solid #334155",
            borderRadius: "8px",
            color: "#f1f5f9",
          }}
          labelStyle={{ color: "#94a3b8" }}
          formatter={(value: number) => [
            formatIDRCompact(value),
            "Value",
          ]}
        />
        {startingValue && (
          <ReferenceLine
            y={startingValue}
            stroke="#475569"
            strokeDasharray="5 5"
            label={{
              value: "Start",
              fill: "#94a3b8",
              fontSize: 11,
              position: "left",
            }}
          />
        )}
        <Area
          type="monotone"
          dataKey="value"
          stroke={isPositive ? "#22c55e" : "#ef4444"}
          strokeWidth={2}
          fill="url(#performanceGradient)"
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
