import { prisma } from "@/lib/prisma";
import { readFileSync, existsSync } from "fs";
import { SystemStatus } from "@/components/system-status";

const PER_PAGE = 10;

interface Props {
  searchParams: Promise<{
    symbol?: string;
    timeframe?: string;
    status?: string;
    page?: string;
    simPage?: string;
    monthPage?: string;
    coinPage?: string;
  }>;
}

export default async function ScannerPage({ searchParams }: Props) {
  const params = await searchParams;
  const page = Math.max(1, parseInt(params.page || "1") || 1);
  const simPage = Math.max(1, parseInt(params.simPage || "1") || 1);
  const monthPage = Math.max(1, parseInt(params.monthPage || "1") || 1);
  const coinPage = Math.max(1, parseInt(params.coinPage || "1") || 1);
  const tfFilter = params.timeframe || "all";
  const statusFilter = params.status || "all";
  const symbolFilter = params.symbol || "";

  // Load research stats + tournament models (dual-direction)
  const TOURNAMENT_PATH = "C:/Users/togat/Desktop/AI-Finance/models/results/tournament_summary.json";
  let rs: any = {
    volume_breakouts_total: 0, big_movers_total: 0, big_mover_rate: 0,
    best_exit_strategy: "", profitable_months_pct: 0, coins_in_whitelist: 0,
    long_ml_threshold: 0, short_ml_threshold: 0, direction: "",
  };
  let tournamentModels: any[] = [];
  if (existsSync(TOURNAMENT_PATH)) {
    const raw = JSON.parse(readFileSync(TOURNAMENT_PATH, "utf-8"));
    if (raw.research_stats) rs = raw.research_stats;
    if (raw.models) tournamentModels = raw.models;
  }

  // Live dual-direction signals (short + long scanners)
  let liveShort: any[] = [];
  let liveLong: any[] = [];
  try {
    liveShort = await prisma.$queryRawUnsafe(`
      SELECT symbol, ml_prob, vol_ratio, price_drop, bounce_pct, signal_time, entry_price, status, pnl_pct, exit_reason
      FROM scanner_signals_v2 ORDER BY signal_time DESC LIMIT 10
    `);
  } catch {}
  try {
    liveLong = await prisma.$queryRawUnsafe(`
      SELECT symbol, ml_prob, vol_ratio, price_rise, pullback_pct, signal_time, entry_price, status, pnl_pct, exit_reason
      FROM scanner_signals_long_v2 ORDER BY signal_time DESC LIMIT 10
    `);
  } catch {}

  // Unified dual-direction view: long v2 ML + short v2 ML filtered OOS trades
  const DUAL_CTE = `
    WITH dual AS (
      SELECT symbol, signal_time, entry_price, exit_price, pnl_pct, exit_reason, bars_held, ml_prob,
        'long' as direction,
        CASE WHEN pnl_pct > 0 THEN 'won' ELSE 'lost' END as status
      FROM scanner_long_v2_ml_filtered
      UNION ALL
      SELECT symbol, signal_time, entry_price, exit_price, pnl_pct, exit_reason, bars_held, ml_prob,
        'short' as direction,
        CASE WHEN pnl_pct > 0 THEN 'won' ELSE 'lost' END as status
      FROM scanner_short_v2_ml_filtered
    )
  `;

  // Overall stats
  let stats = { trades: 0, wins: 0, losses: 0, wr: 0, pf: 0, avgPnl: 0, totalPnl: 0, precision: 0 };
  try {
    const r: any[] = await prisma.$queryRawUnsafe(`
      ${DUAL_CTE}
      SELECT COUNT(*)::int as trades, COUNT(*) FILTER (WHERE status='won')::int as wins,
        COUNT(*) FILTER (WHERE status='lost')::int as losses,
        ROUND(COUNT(*) FILTER (WHERE status='won')::numeric/GREATEST(COUNT(*),1)*100,1) as wr,
        ROUND(NULLIF(SUM(pnl_pct) FILTER (WHERE pnl_pct>0),0)::numeric/ABS(NULLIF(SUM(pnl_pct) FILTER (WHERE pnl_pct<=0),0))::numeric,2) as pf,
        ROUND(AVG(pnl_pct)::numeric*100,2) as avg_pnl,
        ROUND(SUM(pnl_pct)::numeric*100,1) as total_pnl,
        0 as precision
      FROM dual
    `);
    if (r[0]) stats = { trades: r[0].trades, wins: r[0].wins, losses: r[0].losses, wr: Number(r[0].wr)||0, pf: Number(r[0].pf)||0, avgPnl: Number(r[0].avg_pnl)||0, totalPnl: Number(r[0].total_pnl)||0, precision: Number(r[0].precision)||0 };
  } catch {}

  // Realistic simulation: 1 trade/day (best ML prob per day), with running equity
  let simTrades: any[] = [];
  let simTotal = 0;
  let simStats = { trades: 0, wins: 0, wr: 0, pf: 0, avgPnl: 0, finalEquity: 0 };
  try {
    const simCountRows: any[] = await prisma.$queryRawUnsafe(`
      ${DUAL_CTE}
      SELECT COUNT(*)::int as n FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY signal_time::date ORDER BY ml_prob DESC) as rn
        FROM dual
      ) t WHERE rn = 1
    `);
    simTotal = simCountRows[0]?.n || 0;

    const simStatsRows: any[] = await prisma.$queryRawUnsafe(`
      ${DUAL_CTE}, daily AS (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY signal_time::date ORDER BY ml_prob DESC) as rn
        FROM dual
      )
      SELECT COUNT(*)::int as trades, COUNT(*) FILTER (WHERE status='won')::int as wins,
        ROUND(COUNT(*) FILTER (WHERE status='won')::numeric/GREATEST(COUNT(*),1)*100,1) as wr,
        ROUND(NULLIF(SUM(pnl_pct) FILTER (WHERE pnl_pct>0),0)::numeric/ABS(NULLIF(SUM(pnl_pct) FILTER (WHERE pnl_pct<=0),0))::numeric,2) as pf,
        ROUND(AVG(pnl_pct)::numeric*100,2) as avg_pnl
      FROM daily WHERE rn = 1
    `);
    if (simStatsRows[0]) simStats = { trades: simStatsRows[0].trades, wins: simStatsRows[0].wins, wr: Number(simStatsRows[0].wr)||0, pf: Number(simStatsRows[0].pf)||0, avgPnl: Number(simStatsRows[0].avg_pnl)||0, finalEquity: 0 };

    simTrades = await prisma.$queryRawUnsafe(`
      ${DUAL_CTE}, daily AS (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY signal_time::date ORDER BY ml_prob DESC) as rn
        FROM dual
      ),
      ordered AS (
        SELECT *, ROW_NUMBER() OVER (ORDER BY signal_time) as trade_num
        FROM daily WHERE rn = 1
      )
      SELECT trade_num, symbol, direction, signal_time, ml_prob, entry_price, exit_price,
        pnl_pct, exit_reason, status, bars_held,
        ROUND((SUM(pnl_pct) OVER (ORDER BY signal_time) * 100)::numeric, 1) as cum_pnl_pct,
        ROUND((600.0 + 600.0 * SUM(pnl_pct) OVER (ORDER BY signal_time))::numeric, 0) as equity
      FROM ordered
      ORDER BY signal_time DESC
      LIMIT ${PER_PAGE} OFFSET ${(simPage - 1) * PER_PAGE}
    `);
  } catch {}

  // Monthly performance (paginated)
  let monthly: any[] = [];
  let monthlyTotal = 0;
  try {
    const mc: any[] = await prisma.$queryRawUnsafe(`${DUAL_CTE} SELECT COUNT(DISTINCT date_trunc('month', signal_time))::int as n FROM dual`);
    monthlyTotal = mc[0]?.n || 0;
    monthly = await prisma.$queryRawUnsafe(`
      ${DUAL_CTE}, daily AS (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY signal_time::date ORDER BY ml_prob DESC) as rn
        FROM dual
      )
      SELECT date_trunc('month', signal_time) as month,
        COUNT(*)::int as trades, COUNT(*) FILTER (WHERE status='won')::int as wins,
        COUNT(*) FILTER (WHERE status='lost')::int as losses,
        ROUND(COUNT(*) FILTER (WHERE status='won')::numeric/GREATEST(COUNT(*),1)*100,1) as wr,
        ROUND(SUM(pnl_pct)::numeric*100,1) as total_pnl,
        ROUND(AVG(pnl_pct)::numeric*100,2) as avg_pnl
      FROM daily WHERE rn = 1
      GROUP BY 1 ORDER BY 1 DESC
      LIMIT ${PER_PAGE} OFFSET ${(monthPage - 1) * PER_PAGE}
    `);
  } catch {}

  // Top coins (paginated)
  let topCoins: any[] = [];
  let coinsTotal = 0;
  try {
    const cc: any[] = await prisma.$queryRawUnsafe(`${DUAL_CTE} SELECT COUNT(DISTINCT symbol)::int as n FROM dual`);
    coinsTotal = cc[0]?.n || 0;
    topCoins = await prisma.$queryRawUnsafe(`
      ${DUAL_CTE}
      SELECT symbol, COUNT(*)::int as trades, COUNT(*) FILTER (WHERE status='won')::int as wins,
        ROUND(COUNT(*) FILTER (WHERE status='won')::numeric/GREATEST(COUNT(*),1)*100,1) as wr,
        ROUND(SUM(pnl_pct)::numeric*100,1) as total_pnl,
        ROUND(AVG(pnl_pct)::numeric*100,2) as avg_pnl
      FROM dual GROUP BY symbol
      ORDER BY SUM(pnl_pct) DESC
      LIMIT ${PER_PAGE} OFFSET ${(coinPage - 1) * PER_PAGE}
    `);
  } catch {}

  // ML Backtest trades (paginated, filtered)
  let trades: any[] = [];
  let totalTrades = 0;
  try {
    let wh = "WHERE 1=1";
    if (tfFilter !== "all") wh += ` AND direction = '${tfFilter === "long" || tfFilter === "short" ? tfFilter : tfFilter}'`;
    if (statusFilter !== "all") wh += ` AND status = '${statusFilter}'`;
    if (symbolFilter) wh += ` AND symbol ILIKE '%${symbolFilter.toUpperCase()}%'`;

    const cr: any[] = await prisma.$queryRawUnsafe(`${DUAL_CTE} SELECT COUNT(*)::int as n FROM dual ${wh}`);
    totalTrades = cr[0]?.n || 0;
    trades = await prisma.$queryRawUnsafe(`
      ${DUAL_CTE}
      SELECT symbol, direction, signal_time, ml_prob, entry_price, exit_price,
        pnl_pct, exit_reason, status, bars_held
      FROM dual ${wh}
      ORDER BY signal_time DESC LIMIT ${PER_PAGE} OFFSET ${(page - 1) * PER_PAGE}
    `);
  } catch {}

  const simPages = Math.max(1, Math.ceil(simTotal / PER_PAGE));
  const monthPages = Math.max(1, Math.ceil(monthlyTotal / PER_PAGE));
  const coinPages = Math.max(1, Math.ceil(coinsTotal / PER_PAGE));
  const tradePages = Math.max(1, Math.ceil(totalTrades / PER_PAGE));

  // Helper to build href preserving other params
  function pg(key: string, val: number) {
    const p = new URLSearchParams();
    if (key !== "page") p.set("page", String(page));
    if (key !== "simPage") p.set("simPage", String(simPage));
    if (key !== "monthPage") p.set("monthPage", String(monthPage));
    if (key !== "coinPage") p.set("coinPage", String(coinPage));
    if (tfFilter !== "all") p.set("timeframe", tfFilter);
    if (statusFilter !== "all") p.set("status", statusFilter);
    if (symbolFilter) p.set("symbol", symbolFilter);
    p.set(key, String(val));
    return `?${p.toString()}`;
  }

  return (
    <div className="space-y-6">
      <SystemStatus />
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white">
          Volume Breakout Scanner{" "}
          <span className="text-sm font-normal"><span className="text-green-400">LONG</span> + <span className="text-red-400">SHORT</span> dual-direction</span>
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          Long ML {">="} {rs.long_ml_threshold ?? "—"} | Short ML {">="} {rs.short_ml_threshold ?? "—"} | {rs.coins_in_whitelist} coins | {rs.best_exit_strategy} | {rs.profitable_months_pct}% profitable months
        </p>
      </div>

      {/* Tournament Leaderboard */}
      {tournamentModels.length > 0 && (
        <div className="card overflow-hidden p-0">
          <div className="px-4 py-3 border-b border-[var(--border)]">
            <h2 className="text-sm font-semibold text-slate-200">Tournament Leaderboard (Backtest OOS)</h2>
          </div>
          <table className="w-full text-xs">
            <thead><tr className="border-b border-[var(--border)]">
              <th className="px-3 py-2 text-center text-slate-500">Rank</th>
              <th className="px-3 py-2 text-left text-slate-500">Model</th>
              <th className="px-3 py-2 text-center text-slate-500">Dir</th>
              <th className="px-3 py-2 text-right text-slate-500">Trades</th>
              <th className="px-3 py-2 text-right text-slate-500">WR</th>
              <th className="px-3 py-2 text-right text-slate-500">PF</th>
              <th className="px-3 py-2 text-right text-slate-500">Avg</th>
              <th className="px-3 py-2 text-right text-slate-500">Total Return</th>
            </tr></thead>
            <tbody>
              {tournamentModels.map((m: any, i: number) => (
                <tr key={i} className="border-b border-[var(--border)] hover:bg-slate-800/50">
                  <td className="px-3 py-1.5 text-center font-mono text-slate-400">{m.rank}</td>
                  <td className="px-3 py-1.5 text-white">{m.name}</td>
                  <td className={`px-3 py-1.5 text-center font-semibold ${m.direction === "long" ? "text-green-400" : "text-red-400"}`}>{(m.direction || "").toUpperCase()}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-300">{m.trades?.toLocaleString()}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-green-400">{m.win_rate}%</td>
                  <td className="px-3 py-1.5 text-right font-mono text-brand-400">{m.profit_factor?.toFixed(2)}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-green-400">+{m.avg_return_pct}%</td>
                  <td className="px-3 py-1.5 text-right font-mono text-green-400">+{m.total_return_pct?.toLocaleString()}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Live Signals: Long + Short side by side */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <LiveSignalCard title="LONG Scanner (Live)" direction="long" signals={liveLong} />
        <LiveSignalCard title="SHORT Scanner (Live)" direction="short" signals={liveShort} />
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-3">
        <KPI label="All Trades" value={stats.trades.toLocaleString()} />
        <KPI label="Win Rate" value={`${stats.wr}%`} color={stats.wr > 50 ? "green" : "red"} />
        <KPI label="Profit Factor" value={stats.pf.toFixed(2)} color={stats.pf > 1.5 ? "green" : "red"} />
        <KPI label="Avg/Trade" value={`+${stats.avgPnl}%`} color="green" />
        <KPI label="1/Day Trades" value={simStats.trades.toLocaleString()} />
        <KPI label="1/Day WR" value={`${simStats.wr}%`} color={simStats.wr > 50 ? "green" : "red"} />
        <KPI label="1/Day PF" value={simStats.pf.toFixed(2)} color={simStats.pf > 1.5 ? "green" : "red"} />
        <KPI label="1/Day Avg" value={`+${simStats.avgPnl}%`} color="green" />
      </div>

      {/* Realistic Simulation: 1 trade/day */}
      <div className="card overflow-hidden p-0">
        <div className="px-4 py-3 border-b border-[var(--border)] flex justify-between items-center">
          <div>
            <h2 className="text-sm font-semibold text-slate-200">Realistic Simulation — 1 Best Trade per Day</h2>
            <p className="text-[10px] text-slate-500">Picks highest ML probability per day. Equity starts at $600, fixed position size (not compounded).</p>
          </div>
          <Pager current={simPage} total={simPages} paramKey="simPage" buildHref={pg} />
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-[var(--border)]">
                <th className="px-3 py-2 text-center text-slate-500">#</th>
                <th className="px-3 py-2 text-left text-slate-500">Date</th>
                <th className="px-3 py-2 text-left text-slate-500">Symbol</th>
                <th className="px-3 py-2 text-center text-slate-500">Dir</th>
                <th className="px-3 py-2 text-right text-slate-500">ML Prob</th>
                <th className="px-3 py-2 text-right text-slate-500">Entry</th>
                <th className="px-3 py-2 text-right text-slate-500">Exit</th>
                <th className="px-3 py-2 text-right text-slate-500">Bars</th>
                <th className="px-3 py-2 text-right text-slate-500">PnL</th>
                <th className="px-3 py-2 text-center text-slate-500">Reason</th>
                <th className="px-3 py-2 text-right text-slate-500">Cum PnL</th>
                <th className="px-3 py-2 text-right text-slate-500">Equity</th>
              </tr>
            </thead>
            <tbody>
              {simTrades.map((t: any, i: number) => (
                <tr key={i} className="border-b border-[var(--border)] hover:bg-slate-800/50">
                  <td className="px-3 py-1.5 text-center text-slate-500 font-mono">{Number(t.trade_num)}</td>
                  <td className="px-3 py-1.5 text-slate-300 font-mono whitespace-nowrap">
                    {new Date(t.signal_time).toLocaleDateString("en-CA")}
                  </td>
                  <td className="px-3 py-1.5 text-white font-medium">{t.symbol.replace("USDT", "")}</td>
                  <td className={`px-3 py-1.5 text-center font-semibold ${t.direction === "long" ? "text-green-400" : "text-red-400"}`}>{(t.direction || "").toUpperCase()}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-brand-400">{Number(t.ml_prob).toFixed(2)}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-300">${Number(t.entry_price).toPrecision(4)}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-300">${Number(t.exit_price).toPrecision(4)}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-400">{t.bars_held}</td>
                  <td className={`px-3 py-1.5 text-right font-mono font-bold ${Number(t.pnl_pct) > 0 ? "text-green-400" : "text-red-400"}`}>
                    {Number(t.pnl_pct) > 0 ? "+" : ""}{(Number(t.pnl_pct) * 100).toFixed(1)}%
                  </td>
                  <td className="px-3 py-1.5 text-center">
                    <span className={`text-[10px] px-1.5 py-0.5 rounded ${
                      t.exit_reason === "take_profit" ? "bg-green-900/50 text-green-400" :
                      t.exit_reason === "stop_loss" ? "bg-red-900/50 text-red-400" :
                      "bg-slate-700 text-slate-400"
                    }`}>{t.exit_reason}</span>
                  </td>
                  <td className={`px-3 py-1.5 text-right font-mono ${Number(t.cum_pnl_pct) >= 0 ? "text-green-400" : "text-red-400"}`}>
                    {Number(t.cum_pnl_pct) > 0 ? "+" : ""}{Number(t.cum_pnl_pct)}%
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-white">
                    ${Number(t.equity).toLocaleString("en-US", { maximumFractionDigits: 0 })}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Monthly + Top Coins side by side */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Monthly (1 trade/day) */}
        <div className="card overflow-hidden p-0">
          <div className="px-4 py-3 border-b border-[var(--border)] flex justify-between items-center">
            <h2 className="text-sm font-semibold text-slate-200">Monthly Performance (1/day)</h2>
            <Pager current={monthPage} total={monthPages} paramKey="monthPage" buildHref={pg} />
          </div>
          <table className="w-full text-xs">
            <thead><tr className="border-b border-[var(--border)]">
              <th className="px-3 py-2 text-left text-slate-500">Month</th>
              <th className="px-3 py-2 text-right text-slate-500">Trades</th>
              <th className="px-3 py-2 text-right text-slate-500">W/L</th>
              <th className="px-3 py-2 text-right text-slate-500">WR</th>
              <th className="px-3 py-2 text-right text-slate-500">PnL</th>
            </tr></thead>
            <tbody>
              {monthly.map((r: any, i: number) => (
                <tr key={i} className="border-b border-[var(--border)] hover:bg-slate-800/50">
                  <td className="px-3 py-1.5 text-slate-300 font-mono">{new Date(r.month).toLocaleDateString("en-CA", { year: "numeric", month: "short" })}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-400">{r.trades}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-400">{r.wins}/{r.losses}</td>
                  <td className={`px-3 py-1.5 text-right font-mono ${Number(r.wr) >= 50 ? "text-green-400" : "text-red-400"}`}>{Number(r.wr)}%</td>
                  <td className={`px-3 py-1.5 text-right font-mono ${Number(r.total_pnl) >= 0 ? "text-green-400" : "text-red-400"}`}>{Number(r.total_pnl) > 0 ? "+" : ""}{Number(r.total_pnl)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Top Coins */}
        <div className="card overflow-hidden p-0">
          <div className="px-4 py-3 border-b border-[var(--border)] flex justify-between items-center">
            <h2 className="text-sm font-semibold text-slate-200">Coin Performance</h2>
            <Pager current={coinPage} total={coinPages} paramKey="coinPage" buildHref={pg} />
          </div>
          <table className="w-full text-xs">
            <thead><tr className="border-b border-[var(--border)]">
              <th className="px-3 py-2 text-left text-slate-500">Symbol</th>
              <th className="px-3 py-2 text-right text-slate-500">Trades</th>
              <th className="px-3 py-2 text-right text-slate-500">Wins</th>
              <th className="px-3 py-2 text-right text-slate-500">WR</th>
              <th className="px-3 py-2 text-right text-slate-500">Total PnL</th>
              <th className="px-3 py-2 text-right text-slate-500">Avg</th>
            </tr></thead>
            <tbody>
              {topCoins.map((r: any, i: number) => (
                <tr key={i} className="border-b border-[var(--border)] hover:bg-slate-800/50">
                  <td className="px-3 py-1.5 text-white font-medium">{r.symbol.replace("USDT", "")}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-400">{r.trades}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-green-400">{r.wins}</td>
                  <td className={`px-3 py-1.5 text-right font-mono ${Number(r.wr) >= 50 ? "text-green-400" : "text-slate-300"}`}>{Number(r.wr)}%</td>
                  <td className={`px-3 py-1.5 text-right font-mono ${Number(r.total_pnl) >= 0 ? "text-green-400" : "text-red-400"}`}>{Number(r.total_pnl) > 0 ? "+" : ""}{Number(r.total_pnl)}%</td>
                  <td className={`px-3 py-1.5 text-right font-mono ${Number(r.avg_pnl) >= 0 ? "text-green-400" : "text-red-400"}`}>{Number(r.avg_pnl) > 0 ? "+" : ""}{Number(r.avg_pnl)}%</td>
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
            <input name="symbol" defaultValue={symbolFilter} placeholder="e.g. ENJ" className="bg-slate-800 border border-[var(--border)] rounded px-3 py-1.5 text-sm text-white w-28" />
          </div>
          <div>
            <label className="text-xs text-slate-500 block mb-1">Direction</label>
            <select name="timeframe" defaultValue={tfFilter} className="bg-slate-800 border border-[var(--border)] rounded px-3 py-1.5 text-sm text-white">
              <option value="all">All</option>
              <option value="long">Long</option>
              <option value="short">Short</option>
            </select>
          </div>
          <div>
            <label className="text-xs text-slate-500 block mb-1">Status</label>
            <select name="status" defaultValue={statusFilter} className="bg-slate-800 border border-[var(--border)] rounded px-3 py-1.5 text-sm text-white">
              <option value="all">All</option>
              <option value="won">Won</option>
              <option value="lost">Lost</option>
            </select>
          </div>
          <button type="submit" className="bg-brand-600 hover:bg-brand-500 text-white px-4 py-1.5 rounded text-sm">Filter</button>
        </form>
      </div>

      {/* All Dual-Direction ML Trades */}
      <div className="card overflow-hidden p-0">
        <div className="px-4 py-3 border-b border-[var(--border)] flex justify-between items-center">
          <h2 className="text-sm font-semibold text-slate-200">All ML-Filtered Trades ({totalTrades.toLocaleString()})</h2>
          <Pager current={page} total={tradePages} paramKey="page" buildHref={pg} />
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-[var(--border)]">
                <th className="px-3 py-2 text-left text-slate-500">Time</th>
                <th className="px-3 py-2 text-left text-slate-500">Symbol</th>
                <th className="px-3 py-2 text-center text-slate-500">Dir</th>
                <th className="px-3 py-2 text-right text-slate-500">ML</th>
                <th className="px-3 py-2 text-right text-slate-500">Entry</th>
                <th className="px-3 py-2 text-right text-slate-500">Exit</th>
                <th className="px-3 py-2 text-right text-slate-500">Bars</th>
                <th className="px-3 py-2 text-right text-slate-500">PnL</th>
                <th className="px-3 py-2 text-center text-slate-500">Reason</th>
              </tr>
            </thead>
            <tbody>
              {trades.map((t: any, i: number) => (
                <tr key={i} className="border-b border-[var(--border)] hover:bg-slate-800/50">
                  <td className="px-3 py-1.5 text-slate-400 font-mono whitespace-nowrap">
                    {new Date(t.signal_time).toLocaleDateString("en-CA")}
                  </td>
                  <td className="px-3 py-1.5 text-white font-medium">{t.symbol.replace("USDT", "")}</td>
                  <td className={`px-3 py-1.5 text-center font-semibold ${t.direction === "long" ? "text-green-400" : "text-red-400"}`}>{(t.direction || "").toUpperCase()}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-brand-400">{Number(t.ml_prob).toFixed(2)}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-300">${Number(t.entry_price).toPrecision(4)}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-300">${Number(t.exit_price).toPrecision(4)}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-400">{t.bars_held}</td>
                  <td className={`px-3 py-1.5 text-right font-mono font-bold ${Number(t.pnl_pct) > 0 ? "text-green-400" : "text-red-400"}`}>
                    {Number(t.pnl_pct) > 0 ? "+" : ""}{(Number(t.pnl_pct)*100).toFixed(1)}%
                  </td>
                  <td className="px-3 py-1.5 text-center">
                    <span className={`text-[10px] px-1.5 py-0.5 rounded ${
                      t.exit_reason === "take_profit" ? "bg-green-900/50 text-green-400" :
                      t.exit_reason === "stop_loss" ? "bg-red-900/50 text-red-400" :
                      "bg-slate-700 text-slate-400"
                    }`}>{t.exit_reason}</span>
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

function KPI({ label, value, color }: { label: string; value: string; color?: string }) {
  const c = color === "green" ? "text-green-400" : color === "red" ? "text-red-400" : color === "brand" ? "text-brand-400" : "text-white";
  return (
    <div className="card p-3">
      <p className="text-[10px] text-slate-500 uppercase tracking-wide">{label}</p>
      <p className={`text-lg font-bold mt-0.5 ${c}`}>{value}</p>
    </div>
  );
}

function LiveSignalCard({ title, direction, signals }: { title: string; direction: "long" | "short"; signals: any[] }) {
  const accent = direction === "long" ? "text-green-400" : "text-red-400";
  const moveKey = direction === "long" ? "price_rise" : "price_drop";
  const retraceKey = direction === "long" ? "pullback_pct" : "bounce_pct";
  const retraceLabel = direction === "long" ? "Pullback" : "Bounce";
  return (
    <div className="card overflow-hidden p-0">
      <div className="px-4 py-3 border-b border-[var(--border)] flex justify-between items-center">
        <h2 className={`text-sm font-semibold ${accent}`}>{title}</h2>
        <span className="text-[10px] text-slate-500">{signals.length} recent</span>
      </div>
      {signals.length === 0 ? (
        <div className="px-4 py-6 text-xs text-slate-500 text-center">No live signals yet. Scanner may be idle or regime blocked.</div>
      ) : (
        <table className="w-full text-xs">
          <thead><tr className="border-b border-[var(--border)]">
            <th className="px-3 py-2 text-left text-slate-500">Time</th>
            <th className="px-3 py-2 text-left text-slate-500">Symbol</th>
            <th className="px-3 py-2 text-right text-slate-500">ML</th>
            <th className="px-3 py-2 text-right text-slate-500">Vol</th>
            <th className="px-3 py-2 text-right text-slate-500">Move</th>
            <th className="px-3 py-2 text-right text-slate-500">{retraceLabel}</th>
            <th className="px-3 py-2 text-right text-slate-500">Entry</th>
            <th className="px-3 py-2 text-center text-slate-500">Status</th>
            <th className="px-3 py-2 text-right text-slate-500">PnL</th>
          </tr></thead>
          <tbody>
            {signals.map((s: any, i: number) => (
              <tr key={i} className="border-b border-[var(--border)] hover:bg-slate-800/50">
                <td className="px-3 py-1.5 text-slate-400 font-mono whitespace-nowrap">
                  {new Date(s.signal_time).toLocaleString("en-CA", { month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit" })}
                </td>
                <td className="px-3 py-1.5 text-white font-medium">{s.symbol.replace("USDT", "")}</td>
                <td className="px-3 py-1.5 text-right font-mono text-brand-400">{Number(s.ml_prob ?? 0).toFixed(2)}</td>
                <td className="px-3 py-1.5 text-right font-mono text-slate-300">{Number(s.vol_ratio ?? 0).toFixed(1)}x</td>
                <td className={`px-3 py-1.5 text-right font-mono ${accent}`}>{(Number(s[moveKey] ?? 0) * 100).toFixed(1)}%</td>
                <td className="px-3 py-1.5 text-right font-mono text-slate-300">{(Number(s[retraceKey] ?? 0) * 100).toFixed(1)}%</td>
                <td className="px-3 py-1.5 text-right font-mono text-slate-300">${Number(s.entry_price ?? 0).toPrecision(4)}</td>
                <td className="px-3 py-1.5 text-center">
                  <span className={`text-[10px] px-1.5 py-0.5 rounded ${
                    s.status === "active" ? "bg-blue-900/50 text-blue-400" :
                    s.status === "won" ? "bg-green-900/50 text-green-400" :
                    s.status === "lost" ? "bg-red-900/50 text-red-400" :
                    "bg-slate-700 text-slate-400"
                  }`}>{s.status ?? "—"}</span>
                </td>
                <td className={`px-3 py-1.5 text-right font-mono font-bold ${s.pnl_pct == null ? "text-slate-500" : Number(s.pnl_pct) > 0 ? "text-green-400" : "text-red-400"}`}>
                  {s.pnl_pct == null ? "—" : `${Number(s.pnl_pct) > 0 ? "+" : ""}${(Number(s.pnl_pct) * 100).toFixed(1)}%`}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function Pager({ current, total, paramKey, buildHref }: { current: number; total: number; paramKey: string; buildHref: (key: string, val: number) => string }) {
  return (
    <div className="flex items-center gap-2 text-xs">
      {current > 1 && <a href={buildHref(paramKey, current - 1)} className="px-2 py-1 rounded bg-slate-800 text-slate-300 hover:bg-slate-700">Prev</a>}
      <span className="text-slate-500">{current}/{total}</span>
      {current < total && <a href={buildHref(paramKey, current + 1)} className="px-2 py-1 rounded bg-slate-800 text-slate-300 hover:bg-slate-700">Next</a>}
    </div>
  );
}
