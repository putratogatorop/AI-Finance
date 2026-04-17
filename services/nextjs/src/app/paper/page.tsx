import Link from "next/link";
import { prisma } from "@/lib/prisma";

const PER_PAGE = 20;
const STARTING_EQUITY = 100.0;

const THRESHOLDS = [0.80, 0.75, 0.70, 0.65, 0.60];

interface Props {
  searchParams: Promise<{ page?: string; status?: string; direction?: string; threshold?: string }>;
}

export default async function PaperTradesPage({ searchParams }: Props) {
  const params = await searchParams;
  const page = Math.max(1, parseInt(params.page || "1") || 1);
  const statusFilter = params.status || "all";
  const dirFilter = params.direction || "all";
  const threshold = parseFloat(params.threshold || "0.80");

  // Aggregate stats
  let stats = {
    totalTrades: 0, open: 0, won: 0, lost: 0, timeout: 0,
    wr: 0, pf: 0, avgPnl: 0, totalPnl: 0, currentEquity: STARTING_EQUITY,
    totalFees: 0,
  };
  try {
    const r: any[] = await prisma.$queryRawUnsafe(`
      SELECT
        COUNT(*)::int as total,
        COUNT(*) FILTER (WHERE status='open')::int as open_n,
        COUNT(*) FILTER (WHERE status='won')::int as won_n,
        COUNT(*) FILTER (WHERE status='lost')::int as lost_n,
        COUNT(*) FILTER (WHERE status='timeout')::int as to_n,
        COALESCE(ROUND(COUNT(*) FILTER (WHERE status='won')::numeric/GREATEST(COUNT(*) FILTER (WHERE status IN ('won','lost','timeout')),1)*100,1),0) as wr,
        COALESCE(ROUND(NULLIF(SUM(pnl_usd) FILTER (WHERE pnl_usd>0),0)::numeric/ABS(NULLIF(SUM(pnl_usd) FILTER (WHERE pnl_usd<=0),0))::numeric,2),0) as pf,
        COALESCE(ROUND(AVG(pnl_pct) FILTER (WHERE pnl_pct IS NOT NULL)::numeric*100,2),0) as avg_pnl,
        COALESCE(ROUND(SUM(pnl_usd)::numeric,2),0) as total_pnl_usd,
        COALESCE(ROUND(SUM(fees_usd)::numeric,2),0) as total_fees
      FROM paper_trades WHERE threshold = ${threshold}
    `);
    if (r[0]) {
      stats = {
        totalTrades: r[0].total, open: r[0].open_n,
        won: r[0].won_n, lost: r[0].lost_n, timeout: r[0].to_n,
        wr: Number(r[0].wr)||0, pf: Number(r[0].pf)||0, avgPnl: Number(r[0].avg_pnl)||0,
        totalPnl: Number(r[0].total_pnl_usd)||0,
        currentEquity: STARTING_EQUITY + (Number(r[0].total_pnl_usd)||0),
        totalFees: Number(r[0].total_fees)||0,
      };
    }
  } catch {}

  // Open positions
  let openTrades: any[] = [];
  try {
    openTrades = await prisma.$queryRawUnsafe(`
      SELECT id, symbol, direction, ml_prob, entry_time, entry_price,
        position_usd, tp_price, sl_price
      FROM paper_trades WHERE status='open' AND threshold = ${threshold}
      ORDER BY entry_time DESC
    `);
  } catch {}

  // Equity curve (last 100 closed trades)
  let equityCurve: any[] = [];
  try {
    equityCurve = await prisma.$queryRawUnsafe(`
      SELECT exit_time, equity_after, pnl_usd, direction
      FROM paper_trades
      WHERE status IN ('won','lost','timeout') AND threshold = ${threshold}
      ORDER BY exit_time DESC LIMIT 100
    `);
    equityCurve = [...equityCurve].reverse();
  } catch {}

  // Closed trades (paginated)
  let closedTrades: any[] = [];
  let totalClosed = 0;
  try {
    let wh = `WHERE threshold = ${threshold} AND status IN ('won','lost','timeout')`;
    if (statusFilter !== "all") wh = `WHERE threshold = ${threshold} AND status = '${statusFilter}'`;
    if (dirFilter !== "all") wh += ` AND direction = '${dirFilter}'`;

    const cr: any[] = await prisma.$queryRawUnsafe(`SELECT COUNT(*)::int as n FROM paper_trades ${wh}`);
    totalClosed = cr[0]?.n || 0;
    closedTrades = await prisma.$queryRawUnsafe(`
      SELECT id, symbol, direction, ml_prob, entry_time, exit_time,
        entry_price, exit_price, position_usd, pnl_pct, pnl_usd,
        exit_reason, equity_after, fees_usd
      FROM paper_trades ${wh}
      ORDER BY exit_time DESC
      LIMIT ${PER_PAGE} OFFSET ${(page - 1) * PER_PAGE}
    `);
  } catch {}

  const totalPages = Math.max(1, Math.ceil(totalClosed / PER_PAGE));
  const totalReturn = (stats.currentEquity / STARTING_EQUITY - 1) * 100;
  const maxEquity = Math.max(STARTING_EQUITY, ...equityCurve.map((e: any) => Number(e.equity_after) || 0));
  const minEquity = Math.min(STARTING_EQUITY, ...equityCurve.map((e: any) => Number(e.equity_after) || STARTING_EQUITY));

  // Comparison summary across all thresholds
  let thresholdSummary: any[] = [];
  try {
    thresholdSummary = await prisma.$queryRawUnsafe(`
      SELECT threshold,
        COUNT(*)::int as trades,
        COUNT(*) FILTER (WHERE status='open')::int as open_n,
        COALESCE(ROUND(COUNT(*) FILTER (WHERE status='won')::numeric
          /GREATEST(COUNT(*) FILTER (WHERE status IN ('won','lost','timeout')),1)*100,1),0) as wr,
        COALESCE(ROUND(NULLIF(SUM(pnl_usd) FILTER (WHERE pnl_usd>0),0)::numeric
          /ABS(NULLIF(SUM(pnl_usd) FILTER (WHERE pnl_usd<=0),0))::numeric,2),0) as pf,
        COALESCE(ROUND(SUM(pnl_usd)::numeric,2),0) as total_pnl,
        ROUND((100.0 + COALESCE(SUM(pnl_usd),0))::numeric,2) as equity
      FROM paper_trades GROUP BY threshold ORDER BY threshold DESC
    `);
  } catch {}

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white">
            Paper Trading{" "}
            <span className="text-sm font-normal text-yellow-400">DRY-RUN · No real orders</span>
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Multi-threshold comparison. Starting equity ${STARTING_EQUITY.toFixed(2)} per threshold
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-sm text-slate-400">ML Threshold:</span>
          <div className="flex gap-1">
            {THRESHOLDS.map(th => (
              <Link key={th} href={`/paper?threshold=${th}`}
                className={`px-3 py-1.5 rounded text-sm font-mono transition-colors ${
                  th === threshold
                    ? "bg-brand-500 text-white"
                    : "bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-white"
                }`}>
                {th.toFixed(2)}
              </Link>
            ))}
          </div>
        </div>
      </div>

      {/* Threshold comparison table */}
      {thresholdSummary.length > 0 && (
        <div className="card overflow-hidden p-0">
          <div className="px-4 py-3 border-b border-[var(--border)]">
            <h2 className="text-sm font-semibold text-slate-200">Threshold Comparison</h2>
          </div>
          <table className="w-full text-xs">
            <thead><tr className="border-b border-[var(--border)]">
              <th className="px-3 py-2 text-left text-slate-500">Threshold</th>
              <th className="px-3 py-2 text-right text-slate-500">Trades</th>
              <th className="px-3 py-2 text-right text-slate-500">Open</th>
              <th className="px-3 py-2 text-right text-slate-500">Win Rate</th>
              <th className="px-3 py-2 text-right text-slate-500">Profit Factor</th>
              <th className="px-3 py-2 text-right text-slate-500">PnL</th>
              <th className="px-3 py-2 text-right text-slate-500">Equity</th>
            </tr></thead>
            <tbody>
              {thresholdSummary.map((row: any) => {
                const isCurrent = Number(row.threshold) === threshold;
                return (
                  <tr key={row.threshold} className={`border-b border-[var(--border)] ${isCurrent ? "bg-brand-500/10" : "hover:bg-slate-800/50"}`}>
                    <td className="px-3 py-1.5">
                      <Link href={`/paper?threshold=${row.threshold}`} className={`font-mono ${isCurrent ? "text-brand-400 font-bold" : "text-slate-300 hover:text-white"}`}>
                        {Number(row.threshold).toFixed(2)} {isCurrent ? "◄" : ""}
                      </Link>
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono text-slate-300">{row.trades}</td>
                    <td className="px-3 py-1.5 text-right font-mono text-blue-400">{row.open_n}</td>
                    <td className="px-3 py-1.5 text-right font-mono text-slate-300">{Number(row.wr).toFixed(1)}%</td>
                    <td className={`px-3 py-1.5 text-right font-mono ${Number(row.pf) >= 1.5 ? "text-green-400" : Number(row.pf) >= 1 ? "text-yellow-400" : "text-red-400"}`}>{Number(row.pf).toFixed(2)}</td>
                    <td className={`px-3 py-1.5 text-right font-mono ${Number(row.total_pnl) >= 0 ? "text-green-400" : "text-red-400"}`}>${Number(row.total_pnl).toFixed(2)}</td>
                    <td className={`px-3 py-1.5 text-right font-mono ${Number(row.equity) >= 100 ? "text-green-400" : "text-red-400"}`}>${Number(row.equity).toFixed(2)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Headline KPIs */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
        <KPI label="Current Equity" value={`$${stats.currentEquity.toFixed(2)}`} color={stats.currentEquity >= STARTING_EQUITY ? "green" : "red"} />
        <KPI label="Total Return" value={`${totalReturn >= 0 ? "+" : ""}${totalReturn.toFixed(1)}%`} color={totalReturn >= 0 ? "green" : "red"} />
        <KPI label="Trades" value={`${stats.totalTrades}`} />
        <KPI label="Open" value={`${stats.open}`} color="brand" />
        <KPI label="Win Rate" value={`${stats.wr}%`} color={stats.wr > 50 ? "green" : "red"} />
        <KPI label="Profit Factor" value={stats.pf.toFixed(2)} color={stats.pf > 1.5 ? "green" : "red"} />
        <KPI label="Fees Paid" value={`$${stats.totalFees.toFixed(2)}`} color="slate" />
      </div>

      {/* Equity Curve (simple SVG) */}
      {equityCurve.length > 1 && (
        <div className="card p-4">
          <h2 className="text-sm font-semibold text-slate-200 mb-3">Equity Curve (last {equityCurve.length} closed trades)</h2>
          <svg viewBox={`0 0 ${Math.max(equityCurve.length * 6, 100)} 100`} className="w-full h-40" preserveAspectRatio="none">
            {/* Starting equity line */}
            <line x1="0" y1="50" x2={Math.max(equityCurve.length * 6, 100)} y2="50" stroke="#334155" strokeWidth="0.3" strokeDasharray="2" />
            <polyline
              fill="none"
              stroke={totalReturn >= 0 ? "#4ade80" : "#f87171"}
              strokeWidth="1"
              points={equityCurve.map((e: any, i: number) => {
                const y = 100 - ((Number(e.equity_after) - minEquity) / Math.max(maxEquity - minEquity, 1)) * 100;
                return `${i * 6},${y.toFixed(1)}`;
              }).join(" ")}
            />
          </svg>
          <div className="flex justify-between text-[10px] text-slate-500 mt-1">
            <span>Min: ${minEquity.toFixed(2)}</span>
            <span>Start: ${STARTING_EQUITY.toFixed(2)}</span>
            <span>Max: ${maxEquity.toFixed(2)}</span>
            <span>Now: ${stats.currentEquity.toFixed(2)}</span>
          </div>
        </div>
      )}

      {/* Open Positions */}
      {openTrades.length > 0 && (
        <div className="card overflow-hidden p-0">
          <div className="px-4 py-3 border-b border-[var(--border)]">
            <h2 className="text-sm font-semibold text-slate-200">Open Positions ({openTrades.length})</h2>
          </div>
          <table className="w-full text-xs">
            <thead><tr className="border-b border-[var(--border)]">
              <th className="px-3 py-2 text-left text-slate-500">Entry Time</th>
              <th className="px-3 py-2 text-left text-slate-500">Symbol</th>
              <th className="px-3 py-2 text-center text-slate-500">Dir</th>
              <th className="px-3 py-2 text-right text-slate-500">ML</th>
              <th className="px-3 py-2 text-right text-slate-500">Entry</th>
              <th className="px-3 py-2 text-right text-slate-500">Size</th>
              <th className="px-3 py-2 text-right text-slate-500">TP</th>
              <th className="px-3 py-2 text-right text-slate-500">SL</th>
            </tr></thead>
            <tbody>
              {openTrades.map((t: any) => (
                <tr key={t.id} className="border-b border-[var(--border)] hover:bg-slate-800/50">
                  <td className="px-3 py-1.5 text-slate-400 font-mono whitespace-nowrap">
                    {new Date(t.entry_time).toLocaleString("en-CA", { month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit" })}
                  </td>
                  <td className="px-3 py-1.5 text-white font-medium">{t.symbol.replace("USDT", "")}</td>
                  <td className={`px-3 py-1.5 text-center font-semibold ${t.direction === "long" ? "text-green-400" : "text-red-400"}`}>{t.direction.toUpperCase()}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-brand-400">{Number(t.ml_prob).toFixed(2)}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-300">${Number(t.entry_price).toPrecision(4)}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-300">${Number(t.position_usd).toFixed(2)}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-green-400">${Number(t.tp_price).toPrecision(4)}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-red-400">${Number(t.sl_price).toPrecision(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Filters */}
      <div className="card p-4">
        <form className="flex gap-4 items-end flex-wrap">
          <div>
            <label className="text-xs text-slate-500 block mb-1">Direction</label>
            <select name="direction" defaultValue={dirFilter} className="bg-slate-800 border border-[var(--border)] rounded px-3 py-1.5 text-sm text-white">
              <option value="all">All</option>
              <option value="long">Long</option>
              <option value="short">Short</option>
            </select>
          </div>
          <div>
            <label className="text-xs text-slate-500 block mb-1">Status</label>
            <select name="status" defaultValue={statusFilter} className="bg-slate-800 border border-[var(--border)] rounded px-3 py-1.5 text-sm text-white">
              <option value="all">All Closed</option>
              <option value="won">Won</option>
              <option value="lost">Lost</option>
              <option value="timeout">Timeout</option>
            </select>
          </div>
          <button type="submit" className="bg-brand-600 hover:bg-brand-500 text-white px-4 py-1.5 rounded text-sm">Filter</button>
        </form>
      </div>

      {/* Closed Trades */}
      <div className="card overflow-hidden p-0">
        <div className="px-4 py-3 border-b border-[var(--border)] flex justify-between items-center">
          <h2 className="text-sm font-semibold text-slate-200">Closed Trades ({totalClosed})</h2>
          <div className="flex items-center gap-2 text-xs">
            {page > 1 && <Link href={`?page=${page-1}&status=${statusFilter}&direction=${dirFilter}&threshold=${threshold}`} className="px-2 py-1 rounded bg-slate-800 text-slate-300 hover:bg-slate-700">Prev</Link>}
            <span className="text-slate-500">{page}/{totalPages}</span>
            {page < totalPages && <Link href={`?page=${page+1}&status=${statusFilter}&direction=${dirFilter}&threshold=${threshold}`} className="px-2 py-1 rounded bg-slate-800 text-slate-300 hover:bg-slate-700">Next</Link>}
          </div>
        </div>
        {closedTrades.length === 0 ? (
          <div className="px-4 py-8 text-xs text-slate-500 text-center">No closed trades yet. Scanner or paper executor may not be running.</div>
        ) : (
          <table className="w-full text-xs">
            <thead><tr className="border-b border-[var(--border)]">
              <th className="px-3 py-2 text-left text-slate-500">Entry</th>
              <th className="px-3 py-2 text-left text-slate-500">Exit</th>
              <th className="px-3 py-2 text-left text-slate-500">Symbol</th>
              <th className="px-3 py-2 text-center text-slate-500">Dir</th>
              <th className="px-3 py-2 text-right text-slate-500">ML</th>
              <th className="px-3 py-2 text-right text-slate-500">Entry $</th>
              <th className="px-3 py-2 text-right text-slate-500">Exit $</th>
              <th className="px-3 py-2 text-right text-slate-500">Size</th>
              <th className="px-3 py-2 text-right text-slate-500">PnL%</th>
              <th className="px-3 py-2 text-right text-slate-500">PnL $</th>
              <th className="px-3 py-2 text-center text-slate-500">Reason</th>
              <th className="px-3 py-2 text-right text-slate-500">Equity</th>
            </tr></thead>
            <tbody>
              {closedTrades.map((t: any) => (
                <tr key={t.id} className="border-b border-[var(--border)] hover:bg-slate-800/50">
                  <td className="px-3 py-1.5 text-slate-400 font-mono whitespace-nowrap">
                    {new Date(t.entry_time).toLocaleString("en-CA", { month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit" })}
                  </td>
                  <td className="px-3 py-1.5 text-slate-400 font-mono whitespace-nowrap">
                    {new Date(t.exit_time).toLocaleString("en-CA", { month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit" })}
                  </td>
                  <td className="px-3 py-1.5 text-white font-medium">{t.symbol.replace("USDT", "")}</td>
                  <td className={`px-3 py-1.5 text-center font-semibold ${t.direction === "long" ? "text-green-400" : "text-red-400"}`}>{t.direction.toUpperCase()}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-brand-400">{Number(t.ml_prob).toFixed(2)}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-300">${Number(t.entry_price).toPrecision(4)}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-300">${Number(t.exit_price).toPrecision(4)}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-400">${Number(t.position_usd).toFixed(2)}</td>
                  <td className={`px-3 py-1.5 text-right font-mono font-bold ${Number(t.pnl_pct) > 0 ? "text-green-400" : "text-red-400"}`}>
                    {Number(t.pnl_pct) > 0 ? "+" : ""}{(Number(t.pnl_pct) * 100).toFixed(1)}%
                  </td>
                  <td className={`px-3 py-1.5 text-right font-mono ${Number(t.pnl_usd) > 0 ? "text-green-400" : "text-red-400"}`}>
                    {Number(t.pnl_usd) > 0 ? "+" : ""}${Number(t.pnl_usd).toFixed(2)}
                  </td>
                  <td className="px-3 py-1.5 text-center">
                    <span className={`text-[10px] px-1.5 py-0.5 rounded ${
                      t.exit_reason === "take_profit" ? "bg-green-900/50 text-green-400" :
                      t.exit_reason === "stop_loss" ? "bg-red-900/50 text-red-400" :
                      "bg-slate-700 text-slate-400"
                    }`}>{t.exit_reason}</span>
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-white">
                    ${Number(t.equity_after ?? STARTING_EQUITY).toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

function KPI({ label, value, color }: { label: string; value: string; color?: string }) {
  const c = color === "green" ? "text-green-400" : color === "red" ? "text-red-400" :
            color === "brand" ? "text-brand-400" : color === "slate" ? "text-slate-400" : "text-white";
  return (
    <div className="card p-3">
      <p className="text-[10px] text-slate-500 uppercase tracking-wide">{label}</p>
      <p className={`text-lg font-bold mt-0.5 ${c}`}>{value}</p>
    </div>
  );
}
