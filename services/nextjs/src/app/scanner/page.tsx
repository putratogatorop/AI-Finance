import { prisma } from "@/lib/prisma";
import ScannerTable from "@/components/scanner/scanner-table";
import { ScannerConsistency } from "@/components/scanner/scanner-consistency";
import { Prisma } from "@prisma/client";
import { readFileSync, existsSync } from "fs";

const PER_PAGE = 30;
const SORTABLE_FIELDS = ["signal_time", "ml_prob", "vol_ratio", "price_change", "pnl_pct"] as const;
type SortField = (typeof SORTABLE_FIELDS)[number];

interface Props {
  searchParams: Promise<{
    status?: string;
    direction?: string;
    symbol?: string;
    sortBy?: string;
    sortDir?: string;
    page?: string;
  }>;
}

export default async function ScannerPage({ searchParams }: Props) {
  const params = await searchParams;

  // Build where clause
  const where: Prisma.ScannerSignalWhereInput = {};
  if (params.status && params.status !== "all") where.status = params.status;
  if (params.direction && params.direction !== "all") where.direction = parseInt(params.direction);
  if (params.symbol) where.symbol = { contains: params.symbol.toUpperCase() };

  // Sorting
  const sortBy: SortField = SORTABLE_FIELDS.includes(params.sortBy as SortField)
    ? (params.sortBy as SortField)
    : "signal_time";
  const sortDir: "asc" | "desc" = params.sortDir === "asc" ? "asc" : "desc";

  // Pagination
  const page = Math.max(1, parseInt(params.page || "1") || 1);
  const skip = (page - 1) * PER_PAGE;

  // Queries
  const [signals, totalFiltered] = await Promise.all([
    prisma.scannerSignal.findMany({
      where,
      orderBy: { [sortBy]: sortDir },
      skip,
      take: PER_PAGE,
    }),
    prisma.scannerSignal.count({ where }),
  ]);

  // Load KPI stats from tournament summary (validated backtest results)
  const TOURNAMENT_PATH =
    "C:/Users/togat/Desktop/AI-Finance/models/results/tournament_summary.json";
  let tournamentStats = { trades: 0, winRate: 0, profitFactor: 0, totalReturn: 0, sharpe: 0 };
  if (existsSync(TOURNAMENT_PATH)) {
    const raw = JSON.parse(readFileSync(TOURNAMENT_PATH, "utf-8"));
    const v64 = raw.models?.find((m: any) => m.name.includes("v6.4"));
    if (v64) {
      tournamentStats = {
        trades: v64.trades,
        winRate: v64.win_rate,
        profitFactor: v64.profit_factor,
        totalReturn: v64.total_return_pct,
        sharpe: v64.sharpe,
      };
    }
  }
  const totalPages = Math.max(1, Math.ceil(totalFiltered / PER_PAGE));

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">
          Momentum Scanner{" "}
          <span className="text-sm font-normal text-brand-400">v6.4 Scanner+ML+ATR Trail</span>
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          ML-filtered breakout signals across 200+ coins
        </p>
      </div>

      {/* Stats Cards — from tournament backtest (validated results) */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <div className="card p-4">
          <p className="text-xs text-slate-400 uppercase tracking-wide">Trades</p>
          <p className="text-2xl font-bold text-white mt-1">{tournamentStats.trades}</p>
        </div>
        <div className="card p-4">
          <p className="text-xs text-slate-400 uppercase tracking-wide">Win Rate</p>
          <p className="text-2xl font-bold text-signal-green mt-1">
            {tournamentStats.winRate}%
          </p>
        </div>
        <div className="card p-4">
          <p className="text-xs text-slate-400 uppercase tracking-wide">Profit Factor</p>
          <p className="text-2xl font-bold text-signal-green mt-1">
            {tournamentStats.profitFactor.toFixed(2)}
          </p>
        </div>
        <div className="card p-4">
          <p className="text-xs text-slate-400 uppercase tracking-wide">Sharpe</p>
          <p className="text-2xl font-bold text-brand-400 mt-1">
            {tournamentStats.sharpe.toFixed(1)}
          </p>
        </div>
        <div className="card p-4">
          <p className="text-xs text-slate-400 uppercase tracking-wide">Total Return</p>
          <p className="text-2xl font-bold text-signal-green mt-1">
            +{tournamentStats.totalReturn}%
          </p>
        </div>
      </div>

      {/* Profit Consistency (compact) */}
      <ScannerConsistency />

      {/* Signals Table */}
      <div className="card">
        <ScannerTable
          signals={signals}
          totalFiltered={totalFiltered}
          page={page}
          totalPages={totalPages}
          sortBy={sortBy}
          sortDir={sortDir}
          filters={{
            status: params.status || "all",
            direction: params.direction || "all",
            symbol: params.symbol || "",
          }}
        />
      </div>
    </div>
  );
}
