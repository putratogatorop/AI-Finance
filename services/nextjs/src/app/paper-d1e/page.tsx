import Link from "next/link";
import { prisma } from "@/lib/prisma";
import { SystemStatus } from "@/components/system-status";

const STARTING_EQUITY = 100.0;
const D1E_SENTINEL_THRESHOLD = 2.0;
const PER_PAGE = 30;

// Locked Phase-17 account specs — must match paper_executor.py D1E_ACCOUNTS.
const D1E_STRATEGIES = [
  { key: "d1e_s1_e1", label: "S1 × E1", sizing: "S1 (BTC trend)", exit: "E1 (fixed 5%/15%)" },
  { key: "d1e_s1_e2", label: "S1 × E2", sizing: "S1 (BTC trend)", exit: "E2 (ATR 2x/6x)" },
  { key: "d1e_s4_e1", label: "S4 × E1", sizing: "S4 (classifier)", exit: "E1 (fixed 5%/15%)" },
  { key: "d1e_s4_e2", label: "S4 × E2", sizing: "S4 (classifier)", exit: "E2 (ATR 2x/6x)" },
];

interface Props {
  searchParams: Promise<{ strategy?: string; page?: string }>;
}

export default async function PaperD1ePage({ searchParams }: Props) {
  const params = await searchParams;
  const selectedStrategy = params.strategy || "all";
  const page = Math.max(1, parseInt(params.page || "1") || 1);

  // Per-strategy summary across all 4 D1e accounts.
  let strategySummary: any[] = [];
  try {
    strategySummary = await prisma.$queryRawUnsafe(`
      SELECT
        strategy,
        COUNT(*)::int AS trades,
        COUNT(*) FILTER (WHERE status='open')::int AS open_n,
        COUNT(*) FILTER (WHERE status='won')::int AS won_n,
        COUNT(*) FILTER (WHERE status IN ('lost','timeout'))::int AS loss_n,
        ROUND((100.0 * COUNT(*) FILTER (WHERE status='won') /
               NULLIF(COUNT(*) FILTER (WHERE status IN ('won','lost','timeout')), 0))::numeric, 1) AS wr,
        ROUND(COALESCE(
          SUM(pnl_usd) FILTER (WHERE pnl_usd>0)
            / ABS(NULLIF(SUM(pnl_usd) FILTER (WHERE pnl_usd<=0), 0)),
          0)::numeric, 2) AS pf,
        ROUND(COALESCE(SUM(pnl_usd), 0)::numeric, 2) AS total_pnl,
        ROUND((${STARTING_EQUITY} + COALESCE(SUM(pnl_usd), 0))::numeric, 2) AS equity,
        MAX(entry_time) AS latest_entry
      FROM paper_trades
      WHERE strategy LIKE 'd1e_%' AND threshold = ${D1E_SENTINEL_THRESHOLD}
      GROUP BY strategy
      ORDER BY strategy
    `);
  } catch {}
  const summaryByKey: Record<string, any> = Object.fromEntries(
    strategySummary.map((r: any) => [r.strategy, r])
  );

  // Signal pipeline funnel: how many D1e signals fired, how many passed v4 gate.
  let signalFunnel = { total: 0, kept: 0, latest: null as Date | null };
  try {
    const r: any[] = await prisma.$queryRawUnsafe(`
      SELECT
        COUNT(*)::int AS total,
        COUNT(*) FILTER (WHERE cls_kept)::int AS kept,
        MAX(signal_time) AS latest
      FROM scanner_signals_d1e
    `);
    if (r[0]) signalFunnel = { total: r[0].total, kept: r[0].kept, latest: r[0].latest };
  } catch {}

  // Total trade count opened across all D1e accounts (for funnel comparison).
  const totalD1eTrades = strategySummary.reduce((acc: number, r: any) => acc + Number(r.trades || 0), 0);

  // Open positions across selected strategy (or all D1e).
  const stratClause =
    selectedStrategy !== "all"
      ? `AND strategy = '${selectedStrategy.replace(/[^a-z0-9_]/gi, "")}'`
      : `AND strategy LIKE 'd1e_%'`;
  let openTrades: any[] = [];
  try {
    openTrades = await prisma.$queryRawUnsafe(`
      SELECT id, strategy, symbol, direction, entry_time, entry_price, position_usd,
             tp_price, sl_price, atr14_at_entry, btc_score, cls_score, pos_scale, exit_kind
      FROM paper_trades
      WHERE status='open' AND threshold = ${D1E_SENTINEL_THRESHOLD} ${stratClause}
      ORDER BY entry_time DESC
      LIMIT 50
    `);
  } catch {}

  // Closed trades, paginated.
  let closedTrades: any[] = [];
  let totalClosed = 0;
  try {
    const cr: any[] = await prisma.$queryRawUnsafe(`
      SELECT COUNT(*)::int AS n FROM paper_trades
      WHERE status IN ('won','lost','timeout')
        AND threshold = ${D1E_SENTINEL_THRESHOLD} ${stratClause}
    `);
    totalClosed = cr[0]?.n || 0;
    closedTrades = await prisma.$queryRawUnsafe(`
      SELECT id, strategy, symbol, direction, entry_time, exit_time, entry_price, exit_price,
             position_usd, pnl_pct, pnl_usd, exit_reason, equity_after, pos_scale, exit_kind
      FROM paper_trades
      WHERE status IN ('won','lost','timeout')
        AND threshold = ${D1E_SENTINEL_THRESHOLD} ${stratClause}
      ORDER BY exit_time DESC NULLS LAST
      LIMIT ${PER_PAGE} OFFSET ${(page - 1) * PER_PAGE}
    `);
  } catch {}
  const totalPages = Math.max(1, Math.ceil(totalClosed / PER_PAGE));

  return (
    <div className="space-y-6">
      <SystemStatus />
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white">
            Paper Trading — D1e (Phase 17){" "}
            <span className="text-sm font-normal text-yellow-400">DRY-RUN · No real orders</span>
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            4 SHORT-only accounts running side-by-side from <code className="text-slate-300">scanner_signals_d1e</code>.
            Starting equity ${STARTING_EQUITY.toFixed(2)} per account · 5% × 5x × pos_scale (cap 1.5).
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Link href="/paper" className="text-xs text-slate-400 hover:text-white px-3 py-1.5 rounded bg-slate-800 hover:bg-slate-700">
            ← v2 multi-threshold
          </Link>
        </div>
      </div>

      {/* Signal pipeline funnel */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <KPI
          label="D1e Signals Fired"
          value={`${signalFunnel.total}`}
          hint={signalFunnel.latest ? `latest ${new Date(signalFunnel.latest).toLocaleString()}` : "no signals yet"}
        />
        <KPI
          label="Kept by v4 (≥ 0.5010)"
          value={`${signalFunnel.kept}`}
          hint={signalFunnel.total > 0 ? `${((signalFunnel.kept / signalFunnel.total) * 100).toFixed(1)}% pass rate` : ""}
          color="brand"
        />
        <KPI label="Trades Opened" value={`${totalD1eTrades}`} color={totalD1eTrades > 0 ? "green" : "slate"} />
        <KPI
          label="BTC Trend Sign-Lock"
          value={signalFunnel.kept > 0 ? "active" : "—"}
          hint="S1 sizing zeros out when btc_score ≥ 0"
          color="slate"
        />
      </div>

      {/* Per-strategy comparison */}
      <div className="card overflow-hidden p-0">
        <div className="px-4 py-3 border-b border-[var(--border)]">
          <h2 className="text-sm font-semibold text-slate-200">Strategy Comparison (4 D1e accounts)</h2>
        </div>
        <table className="w-full text-xs">
          <thead><tr className="border-b border-[var(--border)]">
            <th className="px-3 py-2 text-left text-slate-500">Strategy</th>
            <th className="px-3 py-2 text-left text-slate-500">Sizing</th>
            <th className="px-3 py-2 text-left text-slate-500">Exit</th>
            <th className="px-3 py-2 text-right text-slate-500">Trades</th>
            <th className="px-3 py-2 text-right text-slate-500">Open</th>
            <th className="px-3 py-2 text-right text-slate-500">WR</th>
            <th className="px-3 py-2 text-right text-slate-500">PF</th>
            <th className="px-3 py-2 text-right text-slate-500">PnL</th>
            <th className="px-3 py-2 text-right text-slate-500">Equity</th>
          </tr></thead>
          <tbody>
            {D1E_STRATEGIES.map(spec => {
              const r = summaryByKey[spec.key];
              const trades = Number(r?.trades || 0);
              const isSelected = selectedStrategy === spec.key;
              return (
                <tr key={spec.key} className={`border-b border-[var(--border)] ${isSelected ? "bg-brand-500/10" : "hover:bg-slate-800/50"}`}>
                  <td className="px-3 py-1.5">
                    <Link
                      href={`/paper-d1e?strategy=${selectedStrategy === spec.key ? "all" : spec.key}`}
                      className={`font-mono ${isSelected ? "text-brand-400 font-bold" : "text-slate-300 hover:text-white"}`}
                    >
                      {spec.label} {isSelected ? "◄" : ""}
                    </Link>
                  </td>
                  <td className="px-3 py-1.5 text-slate-400">{spec.sizing}</td>
                  <td className="px-3 py-1.5 text-slate-400">{spec.exit}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-300">{trades}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-blue-400">{Number(r?.open_n || 0)}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-300">
                    {trades > 0 ? `${Number(r?.wr || 0).toFixed(1)}%` : "—"}
                  </td>
                  <td className={`px-3 py-1.5 text-right font-mono ${
                    Number(r?.pf || 0) >= 1.5 ? "text-green-400"
                      : Number(r?.pf || 0) >= 1 ? "text-yellow-400"
                      : trades > 0 ? "text-red-400" : "text-slate-500"}`}>
                    {trades > 0 ? Number(r?.pf || 0).toFixed(2) : "—"}
                  </td>
                  <td className={`px-3 py-1.5 text-right font-mono ${
                    Number(r?.total_pnl || 0) >= 0 ? "text-green-400" : "text-red-400"}`}>
                    ${Number(r?.total_pnl || 0).toFixed(2)}
                  </td>
                  <td className={`px-3 py-1.5 text-right font-mono ${
                    Number(r?.equity || STARTING_EQUITY) >= STARTING_EQUITY ? "text-green-400" : "text-red-400"}`}>
                    ${Number(r?.equity || STARTING_EQUITY).toFixed(2)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {selectedStrategy !== "all" && (
          <div className="px-4 py-2 border-t border-[var(--border)] text-xs text-slate-500">
            Filtered to <span className="text-brand-400 font-mono">{selectedStrategy}</span>.{" "}
            <Link href="/paper-d1e" className="hover:text-white underline">clear filter</Link>
          </div>
        )}
      </div>

      {/* Open positions */}
      {openTrades.length > 0 && (
        <div className="card overflow-hidden p-0">
          <div className="px-4 py-3 border-b border-[var(--border)]">
            <h2 className="text-sm font-semibold text-slate-200">Open Positions ({openTrades.length})</h2>
          </div>
          <table className="w-full text-xs">
            <thead><tr className="border-b border-[var(--border)]">
              <th className="px-3 py-2 text-left text-slate-500">Entry</th>
              <th className="px-3 py-2 text-left text-slate-500">Strategy</th>
              <th className="px-3 py-2 text-left text-slate-500">Symbol</th>
              <th className="px-3 py-2 text-right text-slate-500">Cls</th>
              <th className="px-3 py-2 text-right text-slate-500">BTC</th>
              <th className="px-3 py-2 text-right text-slate-500">Scale</th>
              <th className="px-3 py-2 text-right text-slate-500">Entry $</th>
              <th className="px-3 py-2 text-right text-slate-500">Pos $</th>
              <th className="px-3 py-2 text-right text-slate-500">SL</th>
              <th className="px-3 py-2 text-right text-slate-500">TP</th>
              <th className="px-3 py-2 text-center text-slate-500">Exit</th>
            </tr></thead>
            <tbody>
              {openTrades.map((t: any) => (
                <tr key={t.id} className="border-b border-[var(--border)] hover:bg-slate-800/50">
                  <td className="px-3 py-1.5 text-slate-400 font-mono whitespace-nowrap">
                    {new Date(t.entry_time).toLocaleString("en-CA", { month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit" })}
                  </td>
                  <td className="px-3 py-1.5 font-mono text-slate-300">{(t.strategy || "").replace("d1e_", "")}</td>
                  <td className="px-3 py-1.5 text-white font-medium">{t.symbol.replace("USDT", "")}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-brand-400">
                    {t.cls_score != null ? Number(t.cls_score).toFixed(3) : "—"}
                  </td>
                  <td className={`px-3 py-1.5 text-right font-mono ${Number(t.btc_score) < 0 ? "text-red-400" : "text-slate-500"}`}>
                    {t.btc_score != null ? Number(t.btc_score).toFixed(2) : "—"}
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-300">
                    {t.pos_scale != null ? Number(t.pos_scale).toFixed(2) : "—"}
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-300">${Number(t.entry_price).toPrecision(4)}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-300">${Number(t.position_usd).toFixed(2)}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-red-400">${Number(t.sl_price).toPrecision(4)}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-green-400">${Number(t.tp_price).toPrecision(4)}</td>
                  <td className="px-3 py-1.5 text-center font-mono text-slate-400 uppercase text-[10px]">{t.exit_kind || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Closed trades */}
      <div className="card overflow-hidden p-0">
        <div className="px-4 py-3 border-b border-[var(--border)] flex justify-between items-center">
          <h2 className="text-sm font-semibold text-slate-200">Closed Trades ({totalClosed})</h2>
          <div className="flex items-center gap-2 text-xs">
            {page > 1 && <Link href={`?page=${page - 1}${selectedStrategy !== "all" ? `&strategy=${selectedStrategy}` : ""}`} className="px-2 py-1 rounded bg-slate-800 text-slate-300 hover:bg-slate-700">Prev</Link>}
            <span className="text-slate-500">{page}/{totalPages}</span>
            {page < totalPages && <Link href={`?page=${page + 1}${selectedStrategy !== "all" ? `&strategy=${selectedStrategy}` : ""}`} className="px-2 py-1 rounded bg-slate-800 text-slate-300 hover:bg-slate-700">Next</Link>}
          </div>
        </div>
        {closedTrades.length === 0 ? (
          <div className="px-4 py-8 text-xs text-slate-500 text-center">
            No closed D1e trades yet.{" "}
            {signalFunnel.total === 0
              ? "Scanner has not produced a D1e signal — gates may not have triggered yet."
              : "Signals fired but no trades closed yet."}
          </div>
        ) : (
          <table className="w-full text-xs">
            <thead><tr className="border-b border-[var(--border)]">
              <th className="px-3 py-2 text-left text-slate-500">Exit</th>
              <th className="px-3 py-2 text-left text-slate-500">Strategy</th>
              <th className="px-3 py-2 text-left text-slate-500">Symbol</th>
              <th className="px-3 py-2 text-right text-slate-500">Scale</th>
              <th className="px-3 py-2 text-right text-slate-500">Entry $</th>
              <th className="px-3 py-2 text-right text-slate-500">Exit $</th>
              <th className="px-3 py-2 text-right text-slate-500">PnL %</th>
              <th className="px-3 py-2 text-right text-slate-500">PnL $</th>
              <th className="px-3 py-2 text-center text-slate-500">Reason</th>
              <th className="px-3 py-2 text-right text-slate-500">Equity</th>
            </tr></thead>
            <tbody>
              {closedTrades.map((t: any) => (
                <tr key={t.id} className="border-b border-[var(--border)] hover:bg-slate-800/50">
                  <td className="px-3 py-1.5 text-slate-400 font-mono whitespace-nowrap">
                    {t.exit_time ? new Date(t.exit_time).toLocaleString("en-CA", { month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit" }) : "—"}
                  </td>
                  <td className="px-3 py-1.5 font-mono text-slate-300">{(t.strategy || "").replace("d1e_", "")}</td>
                  <td className="px-3 py-1.5 text-white font-medium">{t.symbol.replace("USDT", "")}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-400">
                    {t.pos_scale != null ? Number(t.pos_scale).toFixed(2) : "—"}
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-300">${Number(t.entry_price).toPrecision(4)}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-300">${Number(t.exit_price).toPrecision(4)}</td>
                  <td className={`px-3 py-1.5 text-right font-mono ${Number(t.pnl_pct) >= 0 ? "text-green-400" : "text-red-400"}`}>
                    {Number(t.pnl_pct) >= 0 ? "+" : ""}{(Number(t.pnl_pct) * 100).toFixed(2)}%
                  </td>
                  <td className={`px-3 py-1.5 text-right font-mono ${Number(t.pnl_usd) >= 0 ? "text-green-400" : "text-red-400"}`}>
                    {Number(t.pnl_usd) > 0 ? "+" : ""}${Number(t.pnl_usd).toFixed(2)}
                  </td>
                  <td className="px-3 py-1.5 text-center">
                    <span className={`text-[10px] px-1.5 py-0.5 rounded ${
                      t.exit_reason === "take_profit" ? "bg-green-900/50 text-green-400" :
                      t.exit_reason === "stop_loss" ? "bg-red-900/50 text-red-400" :
                      t.exit_reason === "rapid_rally" ? "bg-yellow-900/50 text-yellow-400" :
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

function KPI({ label, value, hint, color }: { label: string; value: string; hint?: string; color?: string }) {
  const c = color === "green" ? "text-green-400"
    : color === "red" ? "text-red-400"
    : color === "brand" ? "text-brand-400"
    : color === "slate" ? "text-slate-400"
    : "text-white";
  return (
    <div className="card p-3">
      <p className="text-[10px] text-slate-500 uppercase tracking-wide">{label}</p>
      <p className={`text-lg font-bold mt-0.5 ${c}`}>{value}</p>
      {hint && <p className="text-[10px] text-slate-500 mt-0.5">{hint}</p>}
    </div>
  );
}
