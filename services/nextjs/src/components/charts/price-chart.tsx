"use client";

import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";
import type { DailyPrice } from "@/types";
import { formatUSD } from "@/lib/format";

interface PriceChartProps {
  data: DailyPrice[];
  height?: number;
  showVolume?: boolean;
}

export function PriceChart({
  data,
  height = 400,
  showVolume = true,
}: PriceChartProps) {
  if (data.length === 0) {
    return (
      <div
        className="flex items-center justify-center rounded-lg border border-[var(--border)] bg-slate-800/50 p-8"
        style={{ height }}
      >
        <p className="text-sm text-slate-500">No price data available</p>
      </div>
    );
  }

  const chartData = data.map((d) => ({
    date: new Date(d.date).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      timeZone: "Asia/Jakarta",
    }),
    close: d.close,
    high: d.high,
    low: d.low,
    volume: d.volume,
  }));

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart
        data={chartData}
        margin={{ top: 5, right: 20, bottom: 5, left: 10 }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
        <XAxis
          dataKey="date"
          tick={{ fill: "#94a3b8", fontSize: 12 }}
          tickLine={{ stroke: "#475569" }}
          axisLine={{ stroke: "#475569" }}
        />
        <YAxis
          yAxisId="price"
          orientation="right"
          tick={{ fill: "#94a3b8", fontSize: 12 }}
          tickLine={{ stroke: "#475569" }}
          axisLine={{ stroke: "#475569" }}
          tickFormatter={(value: number) => formatUSD(value)}
          domain={["auto", "auto"]}
        />
        {showVolume && (
          <YAxis
            yAxisId="volume"
            orientation="left"
            tick={{ fill: "#94a3b8", fontSize: 10 }}
            tickLine={{ stroke: "#475569" }}
            axisLine={{ stroke: "#475569" }}
            tickFormatter={(value: number) => {
              if (value >= 1_000_000)
                return `${(value / 1_000_000).toFixed(1)}M`;
              if (value >= 1_000) return `${(value / 1_000).toFixed(0)}K`;
              return value.toString();
            }}
          />
        )}
        <Tooltip
          contentStyle={{
            backgroundColor: "#1e293b",
            border: "1px solid #334155",
            borderRadius: "8px",
            color: "#f1f5f9",
          }}
          labelStyle={{ color: "#94a3b8" }}
          formatter={(value: number, name: string) => {
            if (name === "volume")
              return [value.toLocaleString(), "Volume"];
            return [
              formatUSD(value),
              name.charAt(0).toUpperCase() + name.slice(1),
            ];
          }}
        />
        <Legend
          wrapperStyle={{ color: "#94a3b8", fontSize: 12, paddingTop: 8 }}
        />
        {showVolume && (
          <Bar
            yAxisId="volume"
            dataKey="volume"
            fill="#3b82f6"
            opacity={0.15}
            name="volume"
          />
        )}
        <Line
          yAxisId="price"
          type="monotone"
          dataKey="close"
          stroke="#3b82f6"
          strokeWidth={2}
          dot={false}
          name="close"
        />
        <Line
          yAxisId="price"
          type="monotone"
          dataKey="high"
          stroke="#22c55e"
          strokeWidth={1}
          strokeDasharray="3 3"
          dot={false}
          name="high"
        />
        <Line
          yAxisId="price"
          type="monotone"
          dataKey="low"
          stroke="#ef4444"
          strokeWidth={1}
          strokeDasharray="3 3"
          dot={false}
          name="low"
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
