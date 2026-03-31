import Link from "next/link";
import { prisma } from "@/lib/prisma";
import {
  formatIDR,
  formatPercent,
  getPnlColor,
  getUrgencyLevel,
} from "@/lib/format";
import { PerformanceChart } from "@/components/charts/performance-chart";

async function getOverviewData() {
  const today = new Date();
  today.setHours(0, 0, 0, 0);

  const [openPositions, closedPositions, todaysSignals] = await Promise.all([
    prisma.portfolio.findMany({ where: { status: "open" } }),
    prisma.portfolio.findMany({
      where: { status: { in: ["closed", "stopped"] } },
      orderBy: { closed_at: "asc" },
    }),
    prisma.signal.findMany({
      where: { created_at: { gte: today } },
      orderBy: { created_at: "desc" },
    }),
  ]);

  let totalInvested = 0;
  let totalUnrealizedPnl = 0;
  for (const pos of openPositions) {
    totalInvested += pos.entry_amount_idr;
    const latestPrice = await prisma.assetPriceHourly.findFirst({
      where: { asset: pos.asset },
      orderBy: { timestamp: "desc" },
      select: { close: true },
    });
    if (latestPrice) {
      totalUnrealizedPnl +=
        (latestPrice.close - pos.entry_price) * pos.quantity * 16000;
    }
  }

  const totalRealizedPnl = closedPositions.reduce(
    (sum, p) => sum + (p.pnl_idr ?? 0),
    0
  );
  const portfolioValue = totalInvested + totalUnrealizedPnl;
  const totalPnl = totalRealizedPnl + totalUnrealizedPnl;
  const totalPnlPct =
    totalInvested > 0 ? (totalPnl / totalInvested) * 100 : 0;

  let cumulative = 0;
  const equityCurve = closedPositions.map((p) => {
    cumulative += p.pnl_idr ?? 0;
    return {
      date: (p.closed_at ?? p.opened_at).toISOString().split("T")[0],
      value: cumulative,
    };
  });

  return {
    portfolioValue,
    totalPnl,
    totalPnlPct,
    openPositionsCount: openPositions.length,
    todaysSignals: todaysSignals.map((s) => ({
      ...s,
      created_at: s.created_at.toISOString(),
      urgency: getUrgencyLevel(s),
    })),
    equityCurve,
  };
}

export default async function OverviewPage() {
  const data = await getOverviewData();

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          title="Portfolio Value"
          value={formatIDR(data.portfolioValue)}
          subtitle="Total invested + unrealized P&L"
        />
        <StatCard
          title="Total P&L"
          value={formatIDR(data.totalPnl)}
          subtitle={formatPercent(data.totalPnlPct)}
          valueClassName={getPnlColor(data.totalPnl)}
        />
        <StatCard
          title="Open Positions"
          value={`${data.openPositionsCount} / 8`}
          subtitle="Active trades"
        />
        <StatCard
          title="Today's Signals"
          value={data.todaysSignals.length.toString()}
          subtitle={
            data.todaysSignals.length > 0
              ? `${data.todaysSignals.filter((s) => !s.acknowledged).length} unacknowledged`
              : "No new signals"
          }
        />
      </div>

      <div className="card">
        <h2 className="mb-4 text-lg font-semibold text-slate-100">
          Portfolio Performance
        </h2>
        {data.equityCurve.length > 0 ? (
          <PerformanceChart data={data.equityCurve} startingValue={0} />
        ) : (
          <div className="flex h-64 items-center justify-center">
            <p className="text-sm text-slate-500">
              No closed trades yet. Performance chart will appear after
              your first trade closes.
            </p>
          </div>
        )}
      </div>

      <div className="card">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-slate-100">
            Today&apos;s Signals
          </h2>
          <Link
            href="/signals"
            className="text-sm text-brand-400 hover:text-brand-300"
          >
            View all signals
          </Link>
        </div>
        {data.todaysSignals.length === 0 ? (
          <p className="text-sm text-slate-500">
            No signals generated today. Check back after the daily signal
            generation runs.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-[var(--border)]">
                  <th className="table-header">Asset</th>
                  <th className="table-header">Action</th>
                  <th className="table-header">Confidence</th>
                  <th className="table-header">Expected Return</th>
                  <th className="table-header">Hold</th>
                  <th className="table-header">Agreement</th>
                  <th className="table-header">Urgency</th>
                </tr>
              </thead>
              <tbody>
                {data.todaysSignals.map((signal) => (
                  <tr
                    key={signal.id}
                    className="border-b border-[var(--border)] transition-colors hover:bg-slate-800/50"
                  >
                    <td className="table-cell">
                      <Link
                        href={`/assets/${signal.asset.toLowerCase()}`}
                        className="font-medium text-brand-400 hover:text-brand-300"
                      >
                        {signal.asset}
                      </Link>
                    </td>
                    <td className="table-cell">
                      <span
                        className={
                          signal.action === "BUY"
                            ? "text-green-400 font-semibold"
                            : signal.action === "SELL" ||
                                signal.action === "EXIT"
                              ? "text-red-400 font-semibold"
                              : "text-yellow-400 font-semibold"
                        }
                      >
                        {signal.action}
                      </span>
                    </td>
                    <td className="table-cell font-mono">
                      {Math.round(signal.confidence * 100)}%
                    </td>
                    <td className="table-cell">
                      <span
                        className={getPnlColor(signal.expected_return_pct)}
                      >
                        {formatPercent(signal.expected_return_pct)}
                      </span>
                    </td>
                    <td className="table-cell">
                      {signal.suggested_hold_days}d
                    </td>
                    <td className="table-cell">
                      {signal.model_agreement}
                    </td>
                    <td className="table-cell">
                      <span
                        className={
                          signal.urgency === "green"
                            ? "badge-green"
                            : signal.urgency === "yellow"
                              ? "badge-yellow"
                              : "badge-red"
                        }
                      >
                        {signal.urgency?.toUpperCase()}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

function StatCard({
  title,
  value,
  subtitle,
  valueClassName,
}: {
  title: string;
  value: string;
  subtitle: string;
  valueClassName?: string;
}) {
  return (
    <div className="card">
      <p className="text-sm text-slate-400">{title}</p>
      <p
        className={`mt-1 text-2xl font-bold ${valueClassName || "text-slate-100"}`}
      >
        {value}
      </p>
      <p className="mt-1 text-xs text-slate-500">{subtitle}</p>
    </div>
  );
}
