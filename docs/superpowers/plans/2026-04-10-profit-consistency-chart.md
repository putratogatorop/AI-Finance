# Profit Consistency Chart Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Profit Consistency" section with weekly/monthly toggle showing return bars + win rate line + trade counts, on both the backtest page (full) and scanner page (compact summary).

**Architecture:** Extend the existing `/api/backtest` endpoint to return weekly breakdowns alongside monthly. Build a shared `ProfitConsistencyChart` component (Recharts ComposedChart) used in both pages. Add consistency stats (streaks, profitable periods, std dev) computed client-side from the period data.

**Tech Stack:** Next.js, Recharts (already installed), Prisma, TypeScript, Tailwind CSS

---

### Task 1: Add `WeeklyBreakdown` type and extend `BacktestResult`

**Files:**
- Modify: `services/nextjs/src/types/index.ts:82-94`

- [ ] **Step 1: Add WeeklyBreakdown type and update BacktestResult**

In `services/nextjs/src/types/index.ts`, add `WeeklyBreakdown` and a `weekly` field to `BacktestResult`:

```typescript
export interface WeeklyBreakdown {
  week: string;        // e.g. "2024-W03"
  return_idr: number;
  return_pct: number;
  trades: number;
  win_rate: number;
}

// Also add a unified type for the chart component:
export interface PeriodBreakdown {
  label: string;       // "2024-01" or "2024-W03"
  return_idr: number;
  return_pct: number;
  trades: number;
  win_rate: number;
}
```

Update `BacktestResult`:
```typescript
export interface BacktestResult {
  metrics: BacktestMetrics;
  monthly: MonthlyBreakdown[];
  weekly: WeeklyBreakdown[];
  equity_curve: { date: string; value: number }[];
}
```

- [ ] **Step 2: Commit**

```bash
git add services/nextjs/src/types/index.ts
git commit -m "feat: add WeeklyBreakdown and PeriodBreakdown types"
```

---

### Task 2: Extend `/api/backtest` to return weekly data

**Files:**
- Modify: `services/nextjs/src/app/api/backtest/route.ts:104-136`

- [ ] **Step 1: Add weekly aggregation logic**

After the existing `monthlyMap` block (line 104-122), add a parallel `weeklyMap`:

```typescript
const weeklyMap = new Map<
  string,
  { return_idr: number; trades: number; wins: number }
>();
for (const p of closedPositions) {
  const date = p.closed_at ?? p.opened_at;
  // ISO week: get Monday-based week number
  const d = new Date(date);
  const jan1 = new Date(d.getFullYear(), 0, 1);
  const dayOfYear = Math.ceil(
    (d.getTime() - jan1.getTime()) / 86_400_000
  );
  const weekNum = Math.ceil((dayOfYear + jan1.getDay()) / 7);
  const weekKey = `${d.getFullYear()}-W${String(weekNum).padStart(2, "0")}`;
  const existing = weeklyMap.get(weekKey) || {
    return_idr: 0,
    trades: 0,
    wins: 0,
  };
  existing.return_idr += p.pnl_idr ?? 0;
  existing.trades += 1;
  if ((p.pnl_idr ?? 0) > 0) existing.wins += 1;
  weeklyMap.set(weekKey, existing);
}

const weekly = Array.from(weeklyMap.entries())
  .sort(([a], [b]) => a.localeCompare(b))
  .map(([week, data]) => ({
    week,
    return_idr: data.return_idr,
    return_pct:
      data.trades > 0
        ? (data.return_idr / (data.trades * 1_000_000)) * 100
        : 0,
    trades: data.trades,
    win_rate:
      data.trades > 0 ? (data.wins / data.trades) * 100 : 0,
  }));
```

- [ ] **Step 2: Add `weekly` to both response paths**

In the empty-data response (line 12-28), add `weekly: []` alongside `monthly: []`.

In the main response (line 138-154), add `weekly` to the returned data object:
```typescript
return NextResponse.json({
  data: {
    metrics: { ... },
    monthly,
    weekly,
    equity_curve: equityCurve,
  },
});
```

- [ ] **Step 3: Commit**

```bash
git add services/nextjs/src/app/api/backtest/route.ts
git commit -m "feat: add weekly breakdown to /api/backtest endpoint"
```

---

### Task 3: Build `ProfitConsistencyChart` component

**Files:**
- Create: `services/nextjs/src/components/charts/profit-consistency-chart.tsx`

- [ ] **Step 1: Create the chart component**

```tsx
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
} from "recharts";
import { formatPercent } from "@/lib/format";
import type { PeriodBreakdown } from "@/types";

interface ProfitConsistencyChartProps {
  monthly: PeriodBreakdown[];
  weekly: PeriodBreakdown[];
  compact?: boolean; // true = scanner page (last 6 periods, shorter)
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
          margin={{ top: 5, right: 20, bottom: 5, left: 10 }}
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
            label={
              compact
                ? false
                : {
                    position: "top",
                    fill: "#64748b",
                    fontSize: 10,
                    formatter: (_: number, entry: unknown) => {
                      const item = entry as { trades?: number };
                      return item?.trades != null ? `${item.trades}t` : "";
                    },
                  }
            }
          >
            {data.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={entry.return_pct >= 0 ? "#22c55e" : "#ef4444"}
                opacity={0.8}
              />
            ))}
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
```

- [ ] **Step 2: Commit**

```bash
git add services/nextjs/src/components/charts/profit-consistency-chart.tsx
git commit -m "feat: add ProfitConsistencyChart component with weekly/monthly toggle"
```

---

### Task 4: Add consistency stats component

**Files:**
- Create: `services/nextjs/src/components/charts/consistency-stats.tsx`

- [ ] **Step 1: Create the consistency stats component**

```tsx
import type { PeriodBreakdown } from "@/types";
import { formatPercent } from "@/lib/format";

interface ConsistencyStatsProps {
  data: PeriodBreakdown[];
  granularity: string;
}

function computeStreaks(data: PeriodBreakdown[]) {
  let maxWin = 0;
  let maxLoss = 0;
  let curWin = 0;
  let curLoss = 0;

  for (const d of data) {
    if (d.return_pct > 0) {
      curWin += 1;
      curLoss = 0;
      if (curWin > maxWin) maxWin = curWin;
    } else {
      curLoss += 1;
      curWin = 0;
      if (curLoss > maxLoss) maxLoss = curLoss;
    }
  }

  return { maxWin, maxLoss };
}

export function ConsistencyStats({ data, granularity }: ConsistencyStatsProps) {
  if (data.length === 0) return null;

  const profitable = data.filter((d) => d.return_pct > 0).length;
  const profitableRate = (profitable / data.length) * 100;
  const { maxWin, maxLoss } = computeStreaks(data);

  const returns = data.map((d) => d.return_pct);
  const mean = returns.reduce((s, r) => s + r, 0) / returns.length;
  const stdDev = Math.sqrt(
    returns.reduce((s, r) => s + (r - mean) ** 2, 0) / returns.length
  );

  const worst = data.reduce((w, d) =>
    d.return_pct < w.return_pct ? d : w
  );
  const best = data.reduce((b, d) =>
    d.return_pct > b.return_pct ? d : b
  );

  const label = granularity === "monthly" ? "months" : "weeks";

  return (
    <div className="grid grid-cols-2 gap-4 lg:grid-cols-5">
      <StatCard
        title={`Profitable ${label}`}
        value={`${profitable}/${data.length}`}
        subtitle={formatPercent(profitableRate, 0)}
        positive={profitableRate >= 50}
      />
      <StatCard
        title="Best streak"
        value={`${maxWin} ${label}`}
        subtitle="Consecutive wins"
        positive
      />
      <StatCard
        title="Worst streak"
        value={`${maxLoss} ${label}`}
        subtitle="Consecutive losses"
        positive={maxLoss <= 2}
      />
      <StatCard
        title="Return Std Dev"
        value={`${stdDev.toFixed(2)}%`}
        subtitle="Lower = more consistent"
        positive={stdDev < 5}
      />
      <StatCard
        title="Best / Worst"
        value={formatPercent(best.return_pct, 1)}
        subtitle={`Worst: ${formatPercent(worst.return_pct, 1)} (${worst.label})`}
        positive={best.return_pct > Math.abs(worst.return_pct)}
      />
    </div>
  );
}

function StatCard({
  title,
  value,
  subtitle,
  positive,
}: {
  title: string;
  value: string;
  subtitle: string;
  positive: boolean;
}) {
  return (
    <div className="card p-4">
      <p className="text-xs text-slate-400 uppercase tracking-wide">{title}</p>
      <p
        className={`mt-1 text-xl font-bold ${
          positive ? "text-green-400" : "text-red-400"
        }`}
      >
        {value}
      </p>
      <p className="mt-1 text-xs text-slate-500">{subtitle}</p>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add services/nextjs/src/components/charts/consistency-stats.tsx
git commit -m "feat: add ConsistencyStats component for streak and variance analysis"
```

---

### Task 5: Integrate into backtest page (full version)

**Files:**
- Modify: `services/nextjs/src/app/backtest/page.tsx:120-165`

- [ ] **Step 1: Replace the existing monthly bar chart with ProfitConsistencyChart**

Add imports at top of file:
```typescript
import { ProfitConsistencyChart } from "@/components/charts/profit-consistency-chart";
import { ConsistencyStats } from "@/components/charts/consistency-stats";
import type { PeriodBreakdown } from "@/types";
```

Remove the existing Recharts imports (`ResponsiveContainer`, `BarChart`, `Bar`, `XAxis`, `YAxis`, `CartesianGrid`, `Tooltip`, `Cell`) since they'll no longer be used directly in this file.

After `const { metrics, monthly, equity_curve } = result;` (line 72), add:
```typescript
const { weekly } = result;

const monthlyPeriods: PeriodBreakdown[] = monthly.map((m) => ({
  label: m.month,
  return_idr: m.return_idr,
  return_pct: m.return_pct,
  trades: m.trades,
  win_rate: m.win_rate,
}));

const weeklyPeriods: PeriodBreakdown[] = weekly.map((w) => ({
  label: w.week,
  return_idr: w.return_idr,
  return_pct: w.return_pct,
  trades: w.trades,
  win_rate: w.win_rate,
}));
```

- [ ] **Step 2: Replace the "Monthly Returns" chart section (lines 120-165)**

Replace the entire `{/* Monthly breakdown chart */}` section with:

```tsx
{/* Profit Consistency */}
{monthlyPeriods.length > 0 && (
  <div className="card">
    <h2 className="mb-2 text-lg font-semibold text-slate-100">
      Profit Consistency
    </h2>
    <ProfitConsistencyChart
      monthly={monthlyPeriods}
      weekly={weeklyPeriods}
    />
  </div>
)}

{/* Consistency Stats */}
{monthlyPeriods.length > 0 && (
  <ConsistencyStats
    data={monthlyPeriods}
    granularity="monthly"
  />
)}
```

Note: The `ConsistencyStats` here defaults to monthly view. For simplicity, this shows monthly stats since the chart component internally manages its own toggle state. If the user wants stats to sync with the chart toggle, that can be a follow-up enhancement.

- [ ] **Step 3: Commit**

```bash
git add services/nextjs/src/app/backtest/page.tsx
git commit -m "feat: replace monthly bar chart with ProfitConsistencyChart on backtest page"
```

---

### Task 6: Add compact version to scanner page

**Files:**
- Modify: `services/nextjs/src/app/scanner/page.tsx`

Since the scanner page is a server component and the chart needs client interactivity, we need a small client wrapper.

- [ ] **Step 1: Create scanner consistency wrapper**

Create `services/nextjs/src/components/scanner/scanner-consistency.tsx`:

```tsx
"use client";

import { useEffect, useState } from "react";
import { ProfitConsistencyChart } from "@/components/charts/profit-consistency-chart";
import type { BacktestResult, PeriodBreakdown } from "@/types";
import Link from "next/link";

export function ScannerConsistency() {
  const [data, setData] = useState<{
    monthly: PeriodBreakdown[];
    weekly: PeriodBreakdown[];
  } | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const res = await fetch("/api/backtest");
        const json = await res.json();
        const result: BacktestResult | null = json.data || null;
        if (!result) return;

        setData({
          monthly: result.monthly.map((m) => ({
            label: m.month,
            return_idr: m.return_idr,
            return_pct: m.return_pct,
            trades: m.trades,
            win_rate: m.win_rate,
          })),
          weekly: result.weekly.map((w) => ({
            label: w.week,
            return_idr: w.return_idr,
            return_pct: w.return_pct,
            trades: w.trades,
            win_rate: w.win_rate,
          })),
        });
      } catch {
        // silently fail — this is a supplementary widget
      }
    }
    load();
  }, []);

  if (!data || data.monthly.length === 0) return null;

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-2">
        <h2 className="text-lg font-semibold text-slate-100">
          Profit Consistency
        </h2>
        <Link
          href="/backtest"
          className="text-xs text-brand-400 hover:text-brand-300 transition-colors"
        >
          Full view ->
        </Link>
      </div>
      <ProfitConsistencyChart
        monthly={data.monthly}
        weekly={data.weekly}
        compact
      />
    </div>
  );
}
```

- [ ] **Step 2: Add to scanner page**

In `services/nextjs/src/app/scanner/page.tsx`, add import:
```typescript
import { ScannerConsistency } from "@/components/scanner/scanner-consistency";
```

Add the component between the stats cards and the signals table (after line 112, before line 114):
```tsx
      {/* Profit Consistency (compact) */}
      <ScannerConsistency />
```

- [ ] **Step 3: Commit**

```bash
git add services/nextjs/src/components/scanner/scanner-consistency.tsx services/nextjs/src/app/scanner/page.tsx
git commit -m "feat: add compact profit consistency chart to scanner page"
```

---

### Task 7: Fix trade count label on bars

The Recharts `Bar` `label` prop doesn't receive the full data entry by default. We need a custom label component.

**Files:**
- Modify: `services/nextjs/src/components/charts/profit-consistency-chart.tsx`

- [ ] **Step 1: Replace the Bar label prop with a custom label renderer**

Replace the `label` prop on the `<Bar>` component with a custom render function. Remove the inline `label` object and instead use:

```tsx
<Bar
  yAxisId="left"
  dataKey="return_pct"
  name="Return %"
  radius={[4, 4, 0, 0]}
>
  {!compact &&
    data.map((entry, index) => (
      <Cell
        key={`cell-${index}`}
        fill={entry.return_pct >= 0 ? "#22c55e" : "#ef4444"}
        opacity={0.8}
      />
    ))}
  {compact &&
    data.map((entry, index) => (
      <Cell
        key={`cell-${index}`}
        fill={entry.return_pct >= 0 ? "#22c55e" : "#ef4444"}
        opacity={0.8}
      />
    ))}
</Bar>
```

For trade count labels, add a second hidden bar that only renders labels:

Actually, the simpler approach: use Recharts `<LabelList>` component inside the `<Bar>`:

```tsx
import { LabelList } from "recharts";

// Inside the <Bar>:
{!compact && (
  <LabelList
    dataKey="trades"
    position="top"
    fill="#64748b"
    fontSize={10}
    formatter={(v: number) => `${v}t`}
  />
)}
```

- [ ] **Step 2: Commit**

```bash
git add services/nextjs/src/components/charts/profit-consistency-chart.tsx
git commit -m "fix: use LabelList for trade count labels on consistency bars"
```

---

### Task 8: Verify and test

- [ ] **Step 1: Run TypeScript check**

```bash
cd services/nextjs && npx tsc --noEmit
```

Expected: No type errors.

- [ ] **Step 2: Run dev server and visually verify**

```bash
cd services/nextjs && npm run dev
```

Check:
- `/backtest` — "Profit Consistency" section with toggle, bar+line chart, consistency stats, then the existing monthly table
- `/scanner` — compact version between stats cards and signals table
- Toggle switches between weekly and monthly on both pages
- Green/red bars, yellow win rate line, trade count labels on full view

- [ ] **Step 3: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix: polish profit consistency chart rendering"
```
