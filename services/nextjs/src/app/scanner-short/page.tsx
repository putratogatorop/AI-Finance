import { prisma } from "@/lib/prisma";

const PER_PAGE = 10;
const V2_TABLE = "scanner_short_v2";
const V2_WHERE = `WHERE variant = 'A'`;
const V2_ML_TABLE = "scanner_short_v2_ml_filtered";

interface Props {
  searchParams: Promise<{
    symbol?: string;
    exitReason?: string;
    page?: string;
    simPage?: string;
    monthPage?: string;
    coinPage?: string;
  }>;
}

export default async function ScannerShortPage({ searchParams }: Props) {
  const params = await searchParams;
  const page = Math.max(1, parseInt(params.page || "1") || 1);
  const simPage = Math.max(1, parseInt(params.simPage || "1") || 1);
  const monthPage = Math.max(1, parseInt(params.monthPage || "1") || 1);
  const coinPage = Math.max(1, parseInt(params.coinPage || "1") || 1);
  const exitReasonFilter = params.exitReason || "all";
  const symbolFilter = params.symbol || "";

  // ── KPI: v2 Raw + v2 ML stats ──
  type KpiStats = {
    trades: number; wr: number; pf: number; avgPnl: number;
    tpPct: number; slPct: number; avgBars: number;
  };
  const emptyStats: KpiStats = { trades: 0, wr: 0, pf: 0, avgPnl: 0, tpPct: 0, slPct: 0, avgBars: 0 };
  let rawStats = { ...emptyStats };
  let mlStats = { ...emptyStats };

  const kpiSqlV2 = `
    SELECT COUNT(*)::int as trades,
      ROUND(COUNT(*) FILTER (WHERE pnl_pct > 0)::numeric/GREATEST(COUNT(*),1)*100,1) as wr,
      ROUND(NULLIF(SUM(pnl_pct) FILTER (WHERE pnl_pct>0),0)::numeric/ABS(NULLIF(SUM(pnl_pct) FILTER (WHERE pnl_pct<=0),0))::numeric,2) as pf,
      ROUND(AVG(pnl_pct)::numeric*100,2) as avg_pnl,
      ROUND(COUNT(*) FILTER (WHERE exit_reason='take_profit')::numeric/GREATEST(COUNT(*),1)*100,1) as tp_pct,
      ROUND(COUNT(*) FILTER (WHERE exit_reason='stop_loss')::numeric/GREATEST(COUNT(*),1)*100,1) as sl_pct,
      ROUND(AVG(bars_held)::numeric, 0) as avg_bars
    FROM ${V2_TABLE} ${V2_WHERE}
  `;

  const kpiSqlMl = `
    SELECT COUNT(*)::int as trades,
      ROUND(COUNT(*) FILTER (WHERE pnl_pct > 0)::numeric/GREATEST(COUNT(*),1)*100,1) as wr,
      ROUND(NULLIF(SUM(pnl_pct) FILTER (WHERE pnl_pct>0),0)::numeric/ABS(NULLIF(SUM(pnl_pct) FILTER (WHERE pnl_pct<=0),0))::numeric,2) as pf,
      ROUND(AVG(pnl_pct)::numeric*100,2) as avg_pnl,
      ROUND(COUNT(*) FILTER (WHERE exit_reason='take_profit')::numeric/GREATEST(COUNT(*),1)*100,1) as tp_pct,
      ROUND(COUNT(*) FILTER (WHERE exit_reason='stop_loss')::numeric/GREATEST(COUNT(*),1)*100,1) as sl_pct,
      ROUND(AVG(bars_held)::numeric, 0) as avg_bars
    FROM ${V2_ML_TABLE}
  `;

  try {
    const r: any[] = await prisma.$queryRawUnsafe(kpiSqlV2);
    if (r[0]) rawStats = {
      trades: r[0].trades, wr: Number(r[0].wr) || 0, pf: Number(r[0].pf) || 0,
      avgPnl: Number(r[0].avg_pnl) || 0, tpPct: Number(r[0].tp_pct) || 0,
      slPct: Number(r[0].sl_pct) || 0, avgBars: Number(r[0].avg_bars) || 0,
    };
  } catch {}

  try {
    const r: any[] = await prisma.$queryRawUnsafe(kpiSqlMl);
    if (r[0]) mlStats = {
      trades: r[0].trades, wr: Number(r[0].wr) || 0, pf: Number(r[0].pf) || 0,
      avgPnl: Number(r[0].avg_pnl) || 0, tpPct: Number(r[0].tp_pct) || 0,
      slPct: Number(r[0].sl_pct) || 0, avgBars: Number(r[0].avg_bars) || 0,
    };
  } catch {}

  // ── Realistic Simulation (v2 + ML): $600, 10% pos, max 3/day ──
  const SIM_CAPITAL = 600;
  const SIM_POSITION_PCT = 0.10;
  const SIM_MAX_TRADES_DAY = 999; // no limit — 10% position sizing protects the portfolio
  const SIM_TABLE = V2_ML_TABLE;
  let simTrades: any[] = [];
  let simTotal = 0;
  const simStats = { trades: 0, wins: 0, wr: 0, pf: 0, finalEquity: 0, totalReturn: 0, months: 0, monthlyAvg: 0 };
  try {
    const simCountRows: any[] = await prisma.$queryRawUnsafe(`
      SELECT COUNT(*)::int as n FROM (
        SELECT *, ROW_NUMBER() OVER (
          PARTITION BY signal_time::date ORDER BY signal_time ASC
        ) as rn
        FROM ${SIM_TABLE}
      ) t WHERE rn <= ${SIM_MAX_TRADES_DAY}
    `);
    simTotal = simCountRows[0]?.n || 0;

    const simStatsRows: any[] = await prisma.$queryRawUnsafe(`
      WITH daily AS (
        SELECT *, ROW_NUMBER() OVER (
          PARTITION BY signal_time::date ORDER BY signal_time ASC
        ) as rn
        FROM ${SIM_TABLE}
      ),
      filtered AS (SELECT * FROM daily WHERE rn <= ${SIM_MAX_TRADES_DAY})
      SELECT COUNT(*)::int as trades,
        COUNT(*) FILTER (WHERE pnl_pct > 0)::int as wins,
        ROUND(COUNT(*) FILTER (WHERE pnl_pct > 0)::numeric/GREATEST(COUNT(*),1)*100,1) as wr,
        ROUND(NULLIF(SUM(pnl_pct) FILTER (WHERE pnl_pct>0),0)::numeric/
          ABS(NULLIF(SUM(pnl_pct) FILTER (WHERE pnl_pct<=0),0))::numeric,2) as pf,
        COUNT(DISTINCT date_trunc('month', signal_time))::int as months
      FROM filtered
    `);
    if (simStatsRows[0]) {
      const s = simStatsRows[0];
      simStats.trades = s.trades;
      simStats.wins = s.wins;
      simStats.wr = Number(s.wr) || 0;
      simStats.pf = Number(s.pf) || 0;
      simStats.months = s.months || 1;
    }

    // Fetch all sim trades (for compounding calculation)
    const allSimTrades: any[] = await prisma.$queryRawUnsafe(`
      WITH daily AS (
        SELECT *, ROW_NUMBER() OVER (
          PARTITION BY signal_time::date ORDER BY signal_time ASC
        ) as rn
        FROM ${SIM_TABLE}
      ),
      filtered AS (
        SELECT *, ROW_NUMBER() OVER (ORDER BY signal_time, symbol) as trade_num
        FROM daily WHERE rn <= ${SIM_MAX_TRADES_DAY}
      )
      SELECT trade_num, symbol, signal_time, entry_price, exit_price,
        pnl_pct, exit_reason, bars_held, ml_prob
      FROM filtered
      ORDER BY signal_time ASC, symbol
    `);

    // Compute compounding equity in JS (SQL can't do this properly)
    let equity = SIM_CAPITAL;
    const enriched = allSimTrades.map((t: any) => {
      const pnl = Number(t.pnl_pct);
      const posSize = equity * SIM_POSITION_PCT;
      const tradePnlUsd = posSize * pnl;
      equity += tradePnlUsd;
      return {
        ...t,
        trade_pnl_usd: Math.round(tradePnlUsd * 100) / 100,
        equity: Math.round(equity),
        cum_return_pct: Math.round((equity / SIM_CAPITAL - 1) * 1000) / 10,
      };
    });

    simStats.finalEquity = Math.round(equity);
    simStats.totalReturn = Math.round((equity / SIM_CAPITAL - 1) * 1000) / 10;
    simStats.monthlyAvg = simStats.months > 0
      ? Math.round(simStats.totalReturn / simStats.months * 10) / 10
      : 0;

    // Paginate (reverse for display — newest first)
    const reversed = [...enriched].reverse();
    const start = (simPage - 1) * PER_PAGE;
    simTrades = reversed.slice(start, start + PER_PAGE).map((t: any) => ({
      ...t,
      trade_pnl_usd: t.trade_pnl_usd,
      cum_pnl_pct: t.cum_return_pct,
      equity: t.equity,
    }));
  } catch {}

  // ── Monthly Performance (v2 + ML, paginated) ──
  let monthly: any[] = [];
  let monthlyTotal = 0;
  try {
    const mc: any[] = await prisma.$queryRawUnsafe(`
      SELECT COUNT(DISTINCT date_trunc('month', signal_time))::int as n
      FROM ${V2_ML_TABLE}
    `);
    monthlyTotal = mc[0]?.n || 0;
    monthly = await prisma.$queryRawUnsafe(`
      SELECT date_trunc('month', signal_time) as month,
        COUNT(*)::int as trades,
        COUNT(*) FILTER (WHERE pnl_pct > 0)::int as wins,
        COUNT(*) FILTER (WHERE pnl_pct <= 0)::int as losses,
        ROUND(COUNT(*) FILTER (WHERE pnl_pct > 0)::numeric/GREATEST(COUNT(*),1)*100,1) as wr,
        ROUND(SUM(pnl_pct)::numeric*100,1) as total_pnl,
        ROUND(AVG(pnl_pct)::numeric*100,2) as avg_pnl
      FROM ${V2_ML_TABLE}
      GROUP BY 1 ORDER BY 1 DESC
      LIMIT ${PER_PAGE} OFFSET ${(monthPage - 1) * PER_PAGE}
    `);
  } catch {}

  // ── Top Coins (v2 + ML, paginated) ──
  let topCoins: any[] = [];
  let coinsTotal = 0;
  try {
    const cc: any[] = await prisma.$queryRawUnsafe(`
      SELECT COUNT(DISTINCT symbol)::int as n FROM ${V2_ML_TABLE}
    `);
    coinsTotal = cc[0]?.n || 0;
    topCoins = await prisma.$queryRawUnsafe(`
      SELECT symbol, COUNT(*)::int as trades,
        COUNT(*) FILTER (WHERE pnl_pct > 0)::int as wins,
        ROUND(COUNT(*) FILTER (WHERE pnl_pct > 0)::numeric/GREATEST(COUNT(*),1)*100,1) as wr,
        ROUND(SUM(pnl_pct)::numeric*100,1) as total_pnl,
        ROUND(AVG(pnl_pct)::numeric*100,2) as avg_pnl
      FROM ${V2_ML_TABLE}
      GROUP BY symbol ORDER BY SUM(pnl_pct) DESC
      LIMIT ${PER_PAGE} OFFSET ${(coinPage - 1) * PER_PAGE}
    `);
  } catch {}

  // ── All v2 Trades (paginated, filterable) ──
  let trades: any[] = [];
  let totalTrades = 0;
  try {
    let wh = `${V2_WHERE}`;
    if (exitReasonFilter !== "all") wh += ` AND exit_reason = '${exitReasonFilter}'`;
    if (symbolFilter) wh += ` AND symbol ILIKE '%${symbolFilter.toUpperCase()}%'`;

    const cr: any[] = await prisma.$queryRawUnsafe(
      `SELECT COUNT(*)::int as n FROM ${V2_TABLE} ${wh}`
    );
    totalTrades = cr[0]?.n || 0;
    trades = await prisma.$queryRawUnsafe(`
      SELECT symbol, signal_time, entry_price, exit_price,
        pnl_pct, exit_reason, bars_held, bounce_pct, rejection_speed
      FROM ${V2_TABLE} ${wh}
      ORDER BY signal_time DESC
      LIMIT ${PER_PAGE} OFFSET ${(page - 1) * PER_PAGE}
    `);
  } catch {}

  const simPages = Math.max(1, Math.ceil(simTotal / PER_PAGE));
  const monthPages = Math.max(1, Math.ceil(monthlyTotal / PER_PAGE));
  const coinPages = Math.max(1, Math.ceil(coinsTotal / PER_PAGE));
  const tradePages = Math.max(1, Math.ceil(totalTrades / PER_PAGE));

  function pg(key: string, val: number) {
    const p = new URLSearchParams();
    if (key !== "page") p.set("page", String(page));
    if (key !== "simPage") p.set("simPage", String(simPage));
    if (key !== "monthPage") p.set("monthPage", String(monthPage));
    if (key !== "coinPage") p.set("coinPage", String(coinPage));
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
          Scanner Short v2{" "}
          <span className="text-sm font-normal text-red-400">
            Delayed Entry + Regime Filter
          </span>
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          Bounce rejection entry | 5% SL / 5% TP | BTC daily EMA + breadth filter | 197 coins
        </p>
      </div>

      {/* KPI Comparison: v2 Raw vs v2 + ML */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* v2 Raw */}
        <div className="card p-4">
          <div className="flex items-center gap-2 mb-3">
            <span className="text-xs font-bold text-slate-400 bg-slate-800 px-2 py-0.5 rounded">
              v2 RAW
            </span>
            <span className="text-sm font-semibold text-white">
              Delayed Entry (Variant A)
            </span>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <MiniKPI label="Trades" value={rawStats.trades.toLocaleString()} />
            <MiniKPI label="Win Rate" value={`${rawStats.wr}%`} color={rawStats.wr > 50 ? "green" : "red"} />
            <MiniKPI label="Profit Factor" value={rawStats.pf.toFixed(2)} color={rawStats.pf > 1 ? "green" : "red"} />
            <MiniKPI label="Avg PnL" value={`${rawStats.avgPnl > 0 ? "+" : ""}${rawStats.avgPnl}%`} color={rawStats.avgPnl > 0 ? "green" : "red"} />
          </div>
        </div>

        {/* v2 + ML */}
        <div className="card p-4 border-l-4 border-red-500">
          <div className="flex items-center gap-2 mb-3">
            <span className="text-xs font-bold text-red-400 bg-red-900/30 px-2 py-0.5 rounded">
              v2 + ML
            </span>
            <span className="text-sm font-semibold text-white">
              ML-Filtered
            </span>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <MiniKPI label="Trades" value={mlStats.trades.toLocaleString()} />
            <MiniKPI label="Win Rate" value={`${mlStats.wr}%`} color={mlStats.wr > 50 ? "green" : "red"} />
            <MiniKPI label="Profit Factor" value={mlStats.pf.toFixed(2)} color={mlStats.pf > 1 ? "green" : "red"} />
            <MiniKPI label="Avg PnL" value={`${mlStats.avgPnl > 0 ? "+" : ""}${mlStats.avgPnl}%`} color={mlStats.avgPnl > 0 ? "green" : "red"} />
          </div>
        </div>
      </div>

      {/* Realistic Simulation (v2 + ML) */}
      <div className="card overflow-hidden p-0">
        <div className="px-4 py-3 border-b border-[var(--border)]">
          <div className="flex justify-between items-start">
            <div>
              <h2 className="text-sm font-semibold text-slate-200">
                v2 + ML Simulation — $600 Capital, 10% Position, No Daily Limit
              </h2>
              <p className="text-[10px] text-slate-500 mt-0.5">
                ML @0.70 filtered OOS trades. Bounce rejection + regime. 10% position, compounding.
              </p>
            </div>
            <Pager current={simPage} total={simPages} paramKey="simPage" buildHref={pg} />
          </div>
          {/* Sim KPIs */}
          <div className="grid grid-cols-2 md:grid-cols-6 gap-3 mt-3">
            <MiniKPI label="Sim Trades" value={simStats.trades.toLocaleString()} />
            <MiniKPI label="Win Rate" value={`${simStats.wr}%`} color={simStats.wr > 50 ? "green" : "red"} />
            <MiniKPI label="Profit Factor" value={simStats.pf.toFixed(2)} color={simStats.pf > 1 ? "green" : "red"} />
            <MiniKPI label="Final Equity" value={`$${simStats.finalEquity.toLocaleString()}`} color={simStats.finalEquity > SIM_CAPITAL ? "green" : "red"} />
            <MiniKPI label="Total Return" value={`${simStats.totalReturn > 0 ? "+" : ""}${simStats.totalReturn}%`} color={simStats.totalReturn > 0 ? "green" : "red"} />
            <MiniKPI label="Avg/Month" value={`${simStats.monthlyAvg > 0 ? "+" : ""}${simStats.monthlyAvg}%`} color={simStats.monthlyAvg > 0 ? "green" : "red"} />
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-[var(--border)]">
                <th className="px-3 py-2 text-center text-slate-500">#</th>
                <th className="px-3 py-2 text-left text-slate-500">Date</th>
                <th className="px-3 py-2 text-left text-slate-500">Symbol</th>
                <th className="px-3 py-2 text-right text-slate-500">ML</th>
                <th className="px-3 py-2 text-right text-slate-500">Entry</th>
                <th className="px-3 py-2 text-right text-slate-500">Exit</th>
                <th className="px-3 py-2 text-right text-slate-500">PnL%</th>
                <th className="px-3 py-2 text-right text-slate-500">PnL $</th>
                <th className="px-3 py-2 text-center text-slate-500">Exit</th>
                <th className="px-3 py-2 text-right text-slate-500">Bars</th>
                <th className="px-3 py-2 text-right text-slate-500">Cum PnL</th>
                <th className="px-3 py-2 text-right text-slate-500">Equity</th>
              </tr>
            </thead>
            <tbody>
              {simTrades.map((t: any, i: number) => (
                <tr key={i} className="border-b border-[var(--border)] hover:bg-slate-800/50">
                  <td className="px-3 py-1.5 text-center text-slate-500 font-mono">{Number(t.trade_num)}</td>
                  <td className="px-3 py-1.5 text-slate-300 font-mono whitespace-nowrap">
                    {new Date(t.signal_time).toLocaleDateString("en-CA", { timeZone: "Asia/Jakarta" })}
                  </td>
                  <td className="px-3 py-1.5 text-white font-medium">{t.symbol.replace("USDT", "")}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-brand-400">{t.ml_prob ? Number(t.ml_prob).toFixed(2) : "—"}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-300">${Number(t.entry_price).toPrecision(4)}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-300">${Number(t.exit_price).toPrecision(4)}</td>
                  <td className={`px-3 py-1.5 text-right font-mono font-bold ${Number(t.pnl_pct) > 0 ? "text-green-400" : "text-red-400"}`}>
                    {Number(t.pnl_pct) > 0 ? "+" : ""}{(Number(t.pnl_pct) * 100).toFixed(1)}%
                  </td>
                  <td className={`px-3 py-1.5 text-right font-mono ${Number(t.trade_pnl_usd) > 0 ? "text-green-400" : "text-red-400"}`}>
                    {Number(t.trade_pnl_usd) > 0 ? "+" : ""}${Number(t.trade_pnl_usd).toFixed(2)}
                  </td>
                  <td className="px-3 py-1.5 text-center">
                    <span className={`text-[10px] px-1.5 py-0.5 rounded ${
                      t.exit_reason === "take_profit" ? "bg-green-900/50 text-green-400" :
                      t.exit_reason === "stop_loss" ? "bg-red-900/50 text-red-400" :
                      "bg-slate-700 text-slate-400"
                    }`}>{t.exit_reason}</span>
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-400">{t.bars_held}</td>
                  <td className={`px-3 py-1.5 text-right font-mono ${Number(t.cum_pnl_pct) >= 0 ? "text-green-400" : "text-red-400"}`}>
                    {Number(t.cum_pnl_pct) > 0 ? "+" : ""}{Number(t.cum_pnl_pct)}%
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-white font-bold">
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
        {/* Monthly Performance (v2 Raw) */}
        <div className="card overflow-hidden p-0">
          <div className="px-4 py-3 border-b border-[var(--border)] flex justify-between items-center">
            <div>
              <h2 className="text-sm font-semibold text-slate-200">Monthly Performance (v2 Raw)</h2>
              <p className="text-[10px] text-slate-500">Variant A — delayed entry trades</p>
            </div>
            <Pager current={monthPage} total={monthPages} paramKey="monthPage" buildHref={pg} />
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
                <tr key={i} className="border-b border-[var(--border)] hover:bg-slate-800/50">
                  <td className="px-3 py-1.5 text-slate-300 font-mono">
                    {new Date(r.month).toLocaleDateString("en-CA", { year: "numeric", month: "short", timeZone: "Asia/Jakarta" })}
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-400">{r.trades}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-400">{r.wins}/{r.losses}</td>
                  <td className={`px-3 py-1.5 text-right font-mono ${Number(r.wr) >= 50 ? "text-green-400" : "text-red-400"}`}>
                    {Number(r.wr)}%
                  </td>
                  <td className={`px-3 py-1.5 text-right font-mono ${Number(r.total_pnl) >= 0 ? "text-green-400" : "text-red-400"}`}>
                    {Number(r.total_pnl) > 0 ? "+" : ""}{Number(r.total_pnl)}%
                  </td>
                  <td className={`px-3 py-1.5 text-right font-mono ${Number(r.avg_pnl) >= 0 ? "text-green-400" : "text-red-400"}`}>
                    {Number(r.avg_pnl) > 0 ? "+" : ""}{Number(r.avg_pnl)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Top Coins (v2 Raw) */}
        <div className="card overflow-hidden p-0">
          <div className="px-4 py-3 border-b border-[var(--border)] flex justify-between items-center">
            <div>
              <h2 className="text-sm font-semibold text-slate-200">Top Coins (v2 Raw)</h2>
              <p className="text-[10px] text-slate-500">Variant A — delayed entry trades</p>
            </div>
            <Pager current={coinPage} total={coinPages} paramKey="coinPage" buildHref={pg} />
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
                <tr key={i} className="border-b border-[var(--border)] hover:bg-slate-800/50">
                  <td className="px-3 py-1.5 text-white font-medium">{r.symbol.replace("USDT", "")}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-400">{r.trades}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-green-400">{r.wins}</td>
                  <td className={`px-3 py-1.5 text-right font-mono ${Number(r.wr) >= 50 ? "text-green-400" : "text-slate-300"}`}>
                    {Number(r.wr)}%
                  </td>
                  <td className={`px-3 py-1.5 text-right font-mono ${Number(r.total_pnl) >= 0 ? "text-green-400" : "text-red-400"}`}>
                    {Number(r.total_pnl) > 0 ? "+" : ""}{Number(r.total_pnl)}%
                  </td>
                  <td className={`px-3 py-1.5 text-right font-mono ${Number(r.avg_pnl) >= 0 ? "text-green-400" : "text-red-400"}`}>
                    {Number(r.avg_pnl) > 0 ? "+" : ""}{Number(r.avg_pnl)}%
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

      {/* All v2 Trades */}
      <div className="card overflow-hidden p-0">
        <div className="px-4 py-3 border-b border-[var(--border)] flex justify-between items-center">
          <h2 className="text-sm font-semibold text-slate-200">
            All v2 Trades ({totalTrades.toLocaleString()})
          </h2>
          <Pager current={page} total={tradePages} paramKey="page" buildHref={pg} />
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-[var(--border)]">
                <th className="px-3 py-2 text-left text-slate-500">Date</th>
                <th className="px-3 py-2 text-left text-slate-500">Symbol</th>
                <th className="px-3 py-2 text-right text-slate-500">Entry</th>
                <th className="px-3 py-2 text-right text-slate-500">Exit</th>
                <th className="px-3 py-2 text-right text-slate-500">PnL%</th>
                <th className="px-3 py-2 text-center text-slate-500">Exit Reason</th>
                <th className="px-3 py-2 text-right text-slate-500">Bars</th>
                <th className="px-3 py-2 text-right text-slate-500">Bounce%</th>
                <th className="px-3 py-2 text-right text-slate-500">Rej Speed</th>
              </tr>
            </thead>
            <tbody>
              {trades.map((t: any, i: number) => (
                <tr key={i} className="border-b border-[var(--border)] hover:bg-slate-800/50">
                  <td className="px-3 py-1.5 text-slate-400 font-mono whitespace-nowrap">
                    {new Date(t.signal_time).toLocaleDateString("en-CA", { timeZone: "Asia/Jakarta" })}
                  </td>
                  <td className="px-3 py-1.5 text-white font-medium">{t.symbol.replace("USDT", "")}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-300">
                    ${Number(t.entry_price).toPrecision(4)}
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-300">
                    ${Number(t.exit_price).toPrecision(4)}
                  </td>
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
                  <td className="px-3 py-1.5 text-right font-mono text-slate-300">{t.bars_held}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-400">
                    {t.bounce_pct != null ? `${(Number(t.bounce_pct) * 100).toFixed(1)}%` : "-"}
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-400">
                    {t.rejection_speed != null ? Number(t.rejection_speed).toFixed(2) : "-"}
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

function MiniKPI({ label, value, color }: { label: string; value: string; color?: string }) {
  const c = color === "green" ? "text-green-400" : color === "red" ? "text-red-400" : "text-white";
  return (
    <div>
      <p className="text-[10px] text-slate-500 uppercase tracking-wide">{label}</p>
      <p className={`text-sm font-bold mt-0.5 font-mono ${c}`}>{value}</p>
    </div>
  );
}

function Pager({ current, total, paramKey, buildHref }: {
  current: number; total: number; paramKey: string;
  buildHref: (key: string, val: number) => string;
}) {
  return (
    <div className="flex items-center gap-2 text-xs">
      {current > 1 && (
        <a href={buildHref(paramKey, current - 1)} className="px-2 py-1 rounded bg-slate-800 text-slate-300 hover:bg-slate-700">Prev</a>
      )}
      <span className="text-slate-500">{current}/{total}</span>
      {current < total && (
        <a href={buildHref(paramKey, current + 1)} className="px-2 py-1 rounded bg-slate-800 text-slate-300 hover:bg-slate-700">Next</a>
      )}
    </div>
  );
}
