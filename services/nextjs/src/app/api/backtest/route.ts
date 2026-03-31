import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";

export async function GET(_request: NextRequest) {
  try {
    const closedPositions = await prisma.portfolio.findMany({
      where: { status: { in: ["closed", "stopped"] } },
      orderBy: { closed_at: "asc" },
    });

    if (closedPositions.length === 0) {
      return NextResponse.json({
        data: {
          metrics: {
            total_return_idr: 0,
            total_return_pct: 0,
            win_rate: 0,
            reward_risk_ratio: 0,
            sharpe_ratio: 0,
            max_drawdown_pct: 0,
            total_trades: 0,
            winning_trades: 0,
            losing_trades: 0,
          },
          monthly: [],
          equity_curve: [],
        },
      });
    }

    const totalTrades = closedPositions.length;
    const winningTrades = closedPositions.filter(
      (p) => (p.pnl_idr ?? 0) > 0
    );
    const losingTrades = closedPositions.filter(
      (p) => (p.pnl_idr ?? 0) <= 0
    );

    const totalReturnIdr = closedPositions.reduce(
      (sum, p) => sum + (p.pnl_idr ?? 0),
      0
    );
    const totalInvested = closedPositions.reduce(
      (sum, p) => sum + p.entry_amount_idr,
      0
    );
    const totalReturnPct =
      totalInvested > 0 ? (totalReturnIdr / totalInvested) * 100 : 0;
    const winRate =
      totalTrades > 0 ? (winningTrades.length / totalTrades) * 100 : 0;

    const avgWin =
      winningTrades.length > 0
        ? winningTrades.reduce((s, p) => s + (p.pnl_idr ?? 0), 0) /
          winningTrades.length
        : 0;
    const avgLoss =
      losingTrades.length > 0
        ? Math.abs(
            losingTrades.reduce((s, p) => s + (p.pnl_idr ?? 0), 0) /
              losingTrades.length
          )
        : 1;
    const rewardRiskRatio = avgLoss > 0 ? avgWin / avgLoss : 0;

    let cumulative = 0;
    const equityCurve = closedPositions.map((p) => {
      cumulative += p.pnl_idr ?? 0;
      return {
        date: (p.closed_at ?? p.opened_at).toISOString().split("T")[0],
        value: cumulative,
      };
    });

    let peak = 0;
    let maxDrawdown = 0;
    for (const point of equityCurve) {
      if (point.value > peak) peak = point.value;
      const drawdown =
        peak > 0 ? ((peak - point.value) / peak) * 100 : 0;
      if (drawdown > maxDrawdown) maxDrawdown = drawdown;
    }

    const returns = closedPositions.map(
      (p) => ((p.pnl_idr ?? 0) / p.entry_amount_idr) * 100
    );
    const avgReturn =
      returns.length > 0
        ? returns.reduce((s, r) => s + r, 0) / returns.length
        : 0;
    const stdDev =
      returns.length > 1
        ? Math.sqrt(
            returns.reduce(
              (s, r) => s + Math.pow(r - avgReturn, 2),
              0
            ) /
              (returns.length - 1)
          )
        : 1;
    const sharpeRatio =
      stdDev > 0 ? (avgReturn / stdDev) * Math.sqrt(52) : 0;

    const monthlyMap = new Map<
      string,
      { return_idr: number; trades: number; wins: number }
    >();
    for (const p of closedPositions) {
      const date = p.closed_at ?? p.opened_at;
      const monthKey = `${date.getFullYear()}-${String(
        date.getMonth() + 1
      ).padStart(2, "0")}`;
      const existing = monthlyMap.get(monthKey) || {
        return_idr: 0,
        trades: 0,
        wins: 0,
      };
      existing.return_idr += p.pnl_idr ?? 0;
      existing.trades += 1;
      if ((p.pnl_idr ?? 0) > 0) existing.wins += 1;
      monthlyMap.set(monthKey, existing);
    }

    const monthly = Array.from(monthlyMap.entries())
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([month, data]) => ({
        month,
        return_idr: data.return_idr,
        return_pct:
          data.trades > 0
            ? (data.return_idr / (data.trades * 1_000_000)) * 100
            : 0,
        trades: data.trades,
        win_rate:
          data.trades > 0 ? (data.wins / data.trades) * 100 : 0,
      }));

    return NextResponse.json({
      data: {
        metrics: {
          total_return_idr: totalReturnIdr,
          total_return_pct: totalReturnPct,
          win_rate: winRate,
          reward_risk_ratio: rewardRiskRatio,
          sharpe_ratio: sharpeRatio,
          max_drawdown_pct: maxDrawdown,
          total_trades: totalTrades,
          winning_trades: winningTrades.length,
          losing_trades: losingTrades.length,
        },
        monthly,
        equity_curve: equityCurve,
      },
    });
  } catch (error) {
    console.error("GET /api/backtest error:", error);
    return NextResponse.json(
      { error: "Failed to fetch backtest results" },
      { status: 500 }
    );
  }
}
