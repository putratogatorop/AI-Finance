import { prisma } from "@/lib/prisma";

const PER_PAGE = 10;

const STRATEGY_LABELS: Record<string, string> = {
  A: "Vol+Momentum",
  B: "PriceScan",
  C: "VB Piggyback",
};

interface Props {
  searchParams: Promise<{
    symbol?: string;
    strategy?: string;
    exitReason?: string;
    page?: string;
    monthPage?: string;
    coinPage?: string;
  }>;
}

export default async function ScannerLongPage({ searchParams }: Props) {
  const params = await searchParams;
  const page = Math.max(1, parseInt(params.page || "1") || 1);
  const monthPage = Math.max(1, parseInt(params.monthPage || "1") || 1);
  const coinPage = Math.max(1, parseInt(params.coinPage || "1") || 1);
  const strategyFilter = params.strategy || "all";
  const exitReasonFilter = params.exitReason || "all";
  const symbolFilter = params.symbol || "";

  // Per-strategy KPI stats
  type StrategyStats = {
    trades: number; wr: number; pf: number; avgPnl: number;
    tpPct: number; slPct: number; avgBars: number;
  };
  const strategyStats: Record<string, StrategyStats> = {
    A: { trades: 0, wr: 0, pf: 0, avgPnl: 0, tpPct: 0, slPct: 0, avgBars: 0 },
    B: { trades: 0, wr: 0, pf: 0, avgPnl: 0, tpPct: 0, slPct: 0, avgBars: 0 },
    C: { trades: 0, wr: 0, pf: 0, avgPnl: 0, tpPct: 0, slPct: 0, avgBars: 0 },
  };

  for (const strat of ["A", "B", "C"]) {
    try {
      const r: any[] = await prisma.$queryRawUnsafe(`
        SELECT COUNT(*)::int as trades,
          ROUND(COUNT(*) FILTER (WHERE pnl_pct > 0)::numeric/GREATEST(COUNT(*),1)*100,1) as wr,
          ROUND(NULLIF(SUM(pnl_pct) FILTER (WHERE pnl_pct>0),0)::numeric/ABS(NULLIF(SUM(pnl_pct) FILTER (WHERE pnl_pct<=0),0))::numeric,2) as pf,
          ROUND(AVG(pnl_pct)::numeric*100,2) as avg_pnl,
          ROUND(COUNT(*) FILTER (WHERE exit_reason='take_profit')::numeric/GREATEST(COUNT(*),1)*100,1) as tp_pct,
          ROUND(COUNT(*) FILTER (WHERE exit_reason='stop_loss')::numeric/GREATEST(COUNT(*),1)*100,1) as sl_pct,
          ROUND(AVG(bars_held)::numeric, 0) as avg_bars
        FROM scanner_long_backtest WHERE strategy = '${strat}'
      `);
      if (r[0]) {
        strategyStats[strat] = {
          trades: r[0].trades,
          wr: Number(r[0].wr) || 0,
          pf: Number(r[0].pf) || 0,
          avgPnl: Number(r[0].avg_pnl) || 0,
          tpPct: Number(r[0].tp_pct) || 0,
          slPct: Number(r[0].sl_pct) || 0,
          avgBars: Number(r[0].avg_bars) || 0,
        };
      }
    } catch {}
  }

  // Determine winner (highest PF)
  const winner = (["A", "B", "C"] as const).reduce((best, s) =>
    strategyStats[s].pf > strategyStats[best].pf ? s : best, "A" as string);

  // Monthly performance for winning strategy (paginated)
  let monthly: any[] = [];
  let monthlyTotal = 0;
  try {
    const mc: any[] = await prisma.$queryRawUnsafe(`
      SELECT COUNT(DISTINCT date_trunc('month', signal_time::timestamptz))::int as n
      FROM scanner_long_backtest WHERE strategy = '${winner}'
    `);
    monthlyTotal = mc[0]?.n || 0;
    monthly = await prisma.$queryRawUnsafe(`
      SELECT date_trunc('month', signal_time::timestamptz) as month,
        COUNT(*)::int as trades,
        COUNT(*) FILTER (WHERE pnl_pct > 0)::int as wins,
        COUNT(*) FILTER (WHERE pnl_pct <= 0)::int as losses,
        ROUND(COUNT(*) FILTER (WHERE pnl_pct > 0)::numeric/GREATEST(COUNT(*),1)*100,1) as wr,
        ROUND(SUM(pnl_pct)::numeric*100,1) as total_pnl,
        ROUND(AVG(pnl_pct)::numeric*100,2) as avg_pnl
      FROM scanner_long_backtest WHERE strategy = '${winner}'
      GROUP BY 1 ORDER BY 1 DESC
      LIMIT ${PER_PAGE} OFFSET ${(monthPage - 1) * PER_PAGE}
    `);
  } catch {}

  // Top coins for winning strategy (paginated)
  let topCoins: any[] = [];
  let coinsTotal = 0;
  try {
    const cc: any[] = await prisma.$queryRawUnsafe(`
      SELECT COUNT(DISTINCT symbol)::int as n
      FROM scanner_long_backtest WHERE strategy = '${winner}'
    `);
    coinsTotal = cc[0]?.n || 0;
    topCoins = await prisma.$queryRawUnsafe(`
      SELECT symbol, COUNT(*)::int as trades,
        COUNT(*) FILTER (WHERE pnl_pct > 0)::int as wins,
        ROUND(COUNT(*) FILTER (WHERE pnl_pct > 0)::numeric/GREATEST(COUNT(*),1)*100,1) as wr,
        ROUND(SUM(pnl_pct)::numeric*100,1) as total_pnl,
        ROUND(AVG(pnl_pct)::numeric*100,2) as avg_pnl
      FROM scanner_long_backtest WHERE strategy = '${winner}'
      GROUP BY symbol ORDER BY SUM(pnl_pct) DESC
      LIMIT ${PER_PAGE} OFFSET ${(coinPage - 1) * PER_PAGE}
    `);
  } catch {}

  // All trades (paginated, filtered)
  let trades: any[] = [];
  let totalTrades = 0;
  try {
    let wh = "WHERE 1=1";
    if (strategyFilter !== "all") wh += ` AND strategy = '${strategyFilter}'`;
    if (exitReasonFilter !== "all") wh += ` AND exit_reason = '${exitReasonFilter}'`;
    if (symbolFilter) wh += ` AND symbol ILIKE '%${symbolFilter.toUpperCase()}%'`;

    const cr: any[] = await prisma.$queryRawUnsafe(
      `SELECT COUNT(*)::int as n FROM scanner_long_backtest ${wh}`
    );
    totalTrades = cr[0]?.n || 0;
    trades = await prisma.$queryRawUnsafe(`
      SELECT symbol, strategy, signal_time, entry_price, exit_price,
        pnl_pct, exit_reason, bars_held
      FROM scanner_long_backtest ${wh}
      ORDER BY signal_time DESC LIMIT ${PER_PAGE} OFFSET ${(page - 1) * PER_PAGE}
    `);
  } catch {}

  const monthPages = Math.max(1, Math.ceil(monthlyTotal / PER_PAGE));
  const coinPages = Math.max(1, Math.ceil(coinsTotal / PER_PAGE));
  const tradePages = Math.max(1, Math.ceil(totalTrades / PER_PAGE));

  function pg(key: string, val: number) {
    const p = new URLSearchParams();
    if (key !== "page") p.set("page", String(page));
    if (key !== "monthPage") p.set("monthPage", String(monthPage));
    if (key !== "coinPage") p.set("coinPage", String(coinPage));
    if (strategyFilter !== "all") p.set("strategy", strategyFilter);
    if (exitReasonFilter !== "all") p.set("exitReason", exitReasonFilter);
    if (symbolFilter) p.set("symbol", symbolFilter);
    p.set(key, String(val));
    return `?${p.toString()}`;
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white">
          Scanner Long{" "}
          <span className="text-sm font-normal text-green-400">3-Strategy Backtest</span>
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          SL=5% | TP=15% | R:R=3:1 | 198 coins
        </p>
      </div>

      {/* KPI Cards — 3 columns, one per strategy */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {(["A", "B", "C"] as const).map((s) => {
          const st = strategyStats[s];
          const isWinner = s === winner;
          return (
            <div
              key={s}
              className={`card p-4 ${isWinner ? "border-l-4 border-green-500" : ""}`}
            >
              <div className="flex items-center gap-2 mb-3">
                <span className="text-xs font-bold text-slate-400 bg-slate-800 px-2 py-0.5 rounded">
                  {s}
                </span>
                <span className="text-sm font-semibold text-white">
                  {STRATEGY_LABELS[s]}
                </span>
                {isWinner && (
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-green-900/50 text-green-400 ml-auto">
                    WINNER
                  </span>
                )}
              </div>
              <div className="grid grid-cols-2 gap-2">
                <MiniKPI label="Trades" value={st.trades.toLocaleString()} />
                <MiniKPI
                  label="Win Rate"
                  value={`${st.wr}%`}
                  color={st.wr > 50 ? "green" : "red"}
                />
                <MiniKPI
                  label="Profit Factor"
                  value={st.pf.toFixed(2)}
                  color={st.pf > 1 ? "green" : "red"}
                />
                <MiniKPI
                  label="Avg PnL"
                  value={`${st.avgPnl > 0 ? "+" : ""}${st.avgPnl}%`}
                  color={st.avgPnl > 0 ? "green" : "red"}
                />
              </div>
            </div>
          );
        })}
      </div>

      {/* Strategy Comparison Table */}
      <div className="card overflow-hidden p-0">
        <div className="px-4 py-3 border-b border-[var(--border)]">
          <h2 className="text-sm font-semibold text-slate-200">Strategy Comparison</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-[var(--border)]">
                <th className="px-3 py-2 text-left text-slate-500">Strategy</th>
                <th className="px-3 py-2 text-right text-slate-500">Trades</th>
                <th className="px-3 py-2 text-right text-slate-500">WR%</th>
                <th className="px-3 py-2 text-right text-slate-500">PF</th>
                <th className="px-3 py-2 text-right text-slate-500">Avg PnL</th>
                <th className="px-3 py-2 text-right text-slate-500">TP%</th>
                <th className="px-3 py-2 text-right text-slate-500">SL%</th>
                <th className="px-3 py-2 text-right text-slate-500">Timeout%</th>
                <th className="px-3 py-2 text-right text-slate-500">Avg Bars</th>
              </tr>
            </thead>
            <tbody>
              {(["A", "B", "C"] as const).map((s) => {
                const st = strategyStats[s];
                const isWinner = s === winner;
                const timeoutPct = Math.max(
                  0,
                  Math.round((100 - st.tpPct - st.slPct) * 10) / 10
                );
                return (
                  <tr
                    key={s}
                    className={`border-b border-[var(--border)] hover:bg-slate-800/50 ${
                      isWinner ? "bg-green-950/20" : ""
                    }`}
                  >
                    <td className="px-3 py-1.5">
                      <span className="text-white font-medium">
                        {s}: {STRATEGY_LABELS[s]}
                      </span>
                      {isWinner && (
                        <span className="text-[10px] ml-2 px-1.5 py-0.5 rounded bg-green-900/50 text-green-400">
                          BEST
                        </span>
                      )}
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono text-slate-300">
                      {st.trades}
                    </td>
                    <td
                      className={`px-3 py-1.5 text-right font-mono ${
                        st.wr >= 50 ? "text-green-400" : "text-red-400"
                      }`}
                    >
                      {st.wr}%
                    </td>
                    <td
                      className={`px-3 py-1.5 text-right font-mono font-bold ${
                        st.pf >= 1 ? "text-green-400" : "text-red-400"
                      }`}
                    >
                      {st.pf.toFixed(2)}
                    </td>
                    <td
                      className={`px-3 py-1.5 text-right font-mono ${
                        st.avgPnl >= 0 ? "text-green-400" : "text-red-400"
                      }`}
                    >
                      {st.avgPnl > 0 ? "+" : ""}
                      {st.avgPnl}%
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono text-green-400">
                      {st.tpPct}%
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono text-red-400">
                      {st.slPct}%
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono text-slate-400">
                      {timeoutPct}%
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono text-slate-300">
                      {st.avgBars}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Monthly + Top Coins side by side */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Monthly Performance */}
        <div className="card overflow-hidden p-0">
          <div className="px-4 py-3 border-b border-[var(--border)] flex justify-between items-center">
            <div>
              <h2 className="text-sm font-semibold text-slate-200">
                Monthly Performance
              </h2>
              <p className="text-[10px] text-slate-500">
                Strategy {winner} ({STRATEGY_LABELS[winner]})
              </p>
            </div>
            <Pager
              current={monthPage}
              total={monthPages}
              paramKey="monthPage"
              buildHref={pg}
            />
          </div>
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-[var(--border)]">
                <th className="px-3 py-2 text-left text-slate-500">Month</th>
                <th className="px-3 py-2 text-right text-slate-500">Trades</th>
                <th className="px-3 py-2 text-right text-slate-500">W/L</th>
                <th className="px-3 py-2 text-right text-slate-500">WR</th>
                <th className="px-3 py-2 text-right text-slate-500">Total PnL</th>
                <th className="px-3 py-2 text-right text-slate-500">Avg</th>
              </tr>
            </thead>
            <tbody>
              {monthly.map((r: any, i: number) => (
                <tr
                  key={i}
                  className="border-b border-[var(--border)] hover:bg-slate-800/50"
                >
                  <td className="px-3 py-1.5 text-slate-300 font-mono">
                    {new Date(r.month).toLocaleDateString("en-CA", {
                      year: "numeric",
                      month: "short",
                    })}
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-400">
                    {r.trades}
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-400">
                    {r.wins}/{r.losses}
                  </td>
                  <td
                    className={`px-3 py-1.5 text-right font-mono ${
                      Number(r.wr) >= 50 ? "text-green-400" : "text-red-400"
                    }`}
                  >
                    {Number(r.wr)}%
                  </td>
                  <td
                    className={`px-3 py-1.5 text-right font-mono ${
                      Number(r.total_pnl) >= 0 ? "text-green-400" : "text-red-400"
                    }`}
                  >
                    {Number(r.total_pnl) > 0 ? "+" : ""}
                    {Number(r.total_pnl)}%
                  </td>
                  <td
                    className={`px-3 py-1.5 text-right font-mono ${
                      Number(r.avg_pnl) >= 0 ? "text-green-400" : "text-red-400"
                    }`}
                  >
                    {Number(r.avg_pnl) > 0 ? "+" : ""}
                    {Number(r.avg_pnl)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Top Coins */}
        <div className="card overflow-hidden p-0">
          <div className="px-4 py-3 border-b border-[var(--border)] flex justify-between items-center">
            <div>
              <h2 className="text-sm font-semibold text-slate-200">Top Coins</h2>
              <p className="text-[10px] text-slate-500">
                Strategy {winner} ({STRATEGY_LABELS[winner]})
              </p>
            </div>
            <Pager
              current={coinPage}
              total={coinPages}
              paramKey="coinPage"
              buildHref={pg}
            />
          </div>
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-[var(--border)]">
                <th className="px-3 py-2 text-left text-slate-500">Symbol</th>
                <th className="px-3 py-2 text-right text-slate-500">Trades</th>
                <th className="px-3 py-2 text-right text-slate-500">Wins</th>
                <th className="px-3 py-2 text-right text-slate-500">WR</th>
                <th className="px-3 py-2 text-right text-slate-500">Total PnL</th>
                <th className="px-3 py-2 text-right text-slate-500">Avg</th>
              </tr>
            </thead>
            <tbody>
              {topCoins.map((r: any, i: number) => (
                <tr
                  key={i}
                  className="border-b border-[var(--border)] hover:bg-slate-800/50"
                >
                  <td className="px-3 py-1.5 text-white font-medium">
                    {r.symbol.replace("USDT", "")}
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-400">
                    {r.trades}
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-green-400">
                    {r.wins}
                  </td>
                  <td
                    className={`px-3 py-1.5 text-right font-mono ${
                      Number(r.wr) >= 50 ? "text-green-400" : "text-slate-300"
                    }`}
                  >
                    {Number(r.wr)}%
                  </td>
                  <td
                    className={`px-3 py-1.5 text-right font-mono ${
                      Number(r.total_pnl) >= 0 ? "text-green-400" : "text-red-400"
                    }`}
                  >
                    {Number(r.total_pnl) > 0 ? "+" : ""}
                    {Number(r.total_pnl)}%
                  </td>
                  <td
                    className={`px-3 py-1.5 text-right font-mono ${
                      Number(r.avg_pnl) >= 0 ? "text-green-400" : "text-red-400"
                    }`}
                  >
                    {Number(r.avg_pnl) > 0 ? "+" : ""}
                    {Number(r.avg_pnl)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Filters */}
      <div className="card p-4">
        <form className="flex gap-4 items-end flex-wrap">
          <div>
            <label className="text-xs text-slate-500 block mb-1">Symbol</label>
            <input
              name="symbol"
              defaultValue={symbolFilter}
              placeholder="e.g. ENJ"
              className="bg-slate-800 border border-[var(--border)] rounded px-3 py-1.5 text-sm text-white w-28"
            />
          </div>
          <div>
            <label className="text-xs text-slate-500 block mb-1">Strategy</label>
            <select
              name="strategy"
              defaultValue={strategyFilter}
              className="bg-slate-800 border border-[var(--border)] rounded px-3 py-1.5 text-sm text-white"
            >
              <option value="all">All</option>
              <option value="A">A: Vol+Momentum</option>
              <option value="B">B: PriceScan</option>
              <option value="C">C: VB Piggyback</option>
            </select>
          </div>
          <div>
            <label className="text-xs text-slate-500 block mb-1">Exit Reason</label>
            <select
              name="exitReason"
              defaultValue={exitReasonFilter}
              className="bg-slate-800 border border-[var(--border)] rounded px-3 py-1.5 text-sm text-white"
            >
              <option value="all">All</option>
              <option value="take_profit">Take Profit</option>
              <option value="stop_loss">Stop Loss</option>
              <option value="timeout">Timeout</option>
            </select>
          </div>
          <button
            type="submit"
            className="bg-brand-600 hover:bg-brand-500 text-white px-4 py-1.5 rounded text-sm"
          >
            Filter
          </button>
        </form>
      </div>

      {/* All Trades */}
      <div className="card overflow-hidden p-0">
        <div className="px-4 py-3 border-b border-[var(--border)] flex justify-between items-center">
          <h2 className="text-sm font-semibold text-slate-200">
            All Trades ({totalTrades.toLocaleString()})
          </h2>
          <Pager current={page} total={tradePages} paramKey="page" buildHref={pg} />
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-[var(--border)]">
                <th className="px-3 py-2 text-left text-slate-500">Date</th>
                <th className="px-3 py-2 text-left text-slate-500">Symbol</th>
                <th className="px-3 py-2 text-center text-slate-500">Strategy</th>
                <th className="px-3 py-2 text-right text-slate-500">Entry</th>
                <th className="px-3 py-2 text-right text-slate-500">Exit</th>
                <th className="px-3 py-2 text-right text-slate-500">PnL%</th>
                <th className="px-3 py-2 text-center text-slate-500">Exit Reason</th>
                <th className="px-3 py-2 text-right text-slate-500">Bars Held</th>
              </tr>
            </thead>
            <tbody>
              {trades.map((t: any, i: number) => (
                <tr
                  key={i}
                  className="border-b border-[var(--border)] hover:bg-slate-800/50"
                >
                  <td className="px-3 py-1.5 text-slate-400 font-mono whitespace-nowrap">
                    {new Date(t.signal_time).toLocaleDateString("en-CA")}
                  </td>
                  <td className="px-3 py-1.5 text-white font-medium">
                    {t.symbol.replace("USDT", "")}
                  </td>
                  <td className="px-3 py-1.5 text-center">
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-700 text-slate-300">
                      {t.strategy}: {STRATEGY_LABELS[t.strategy] || t.strategy}
                    </span>
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-300">
                    ${Number(t.entry_price).toPrecision(4)}
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-300">
                    ${Number(t.exit_price).toPrecision(4)}
                  </td>
                  <td
                    className={`px-3 py-1.5 text-right font-mono font-bold ${
                      Number(t.pnl_pct) > 0 ? "text-green-400" : "text-red-400"
                    }`}
                  >
                    {Number(t.pnl_pct) > 0 ? "+" : ""}
                    {(Number(t.pnl_pct) * 100).toFixed(1)}%
                  </td>
                  <td className="px-3 py-1.5 text-center">
                    <span
                      className={`text-[10px] px-1.5 py-0.5 rounded ${
                        t.exit_reason === "take_profit"
                          ? "bg-green-900/50 text-green-400"
                          : t.exit_reason === "stop_loss"
                          ? "bg-red-900/50 text-red-400"
                          : "bg-slate-700 text-slate-400"
                      }`}
                    >
                      {t.exit_reason}
                    </span>
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-300">
                    {t.bars_held}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function MiniKPI({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color?: string;
}) {
  const c =
    color === "green"
      ? "text-green-400"
      : color === "red"
      ? "text-red-400"
      : "text-white";
  return (
    <div>
      <p className="text-[10px] text-slate-500 uppercase tracking-wide">{label}</p>
      <p className={`text-sm font-bold mt-0.5 font-mono ${c}`}>{value}</p>
    </div>
  );
}

function Pager({
  current,
  total,
  paramKey,
  buildHref,
}: {
  current: number;
  total: number;
  paramKey: string;
  buildHref: (key: string, val: number) => string;
}) {
  return (
    <div className="flex items-center gap-2 text-xs">
      {current > 1 && (
        <a
          href={buildHref(paramKey, current - 1)}
          className="px-2 py-1 rounded bg-slate-800 text-slate-300 hover:bg-slate-700"
        >
          Prev
        </a>
      )}
      <span className="text-slate-500">
        {current}/{total}
      </span>
      {current < total && (
        <a
          href={buildHref(paramKey, current + 1)}
          className="px-2 py-1 rounded bg-slate-800 text-slate-300 hover:bg-slate-700"
        >
          Next
        </a>
      )}
    </div>
  );
}
