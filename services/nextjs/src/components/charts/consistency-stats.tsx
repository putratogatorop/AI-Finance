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
