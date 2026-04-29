import Link from "next/link";
import { prisma } from "@/lib/prisma";
import { SystemStatus } from "@/components/system-status";

const STARTING_EQUITY = 200.0;
const V8_SENTINEL_THRESHOLD = 3.0;
const PER_PAGE = 30;

// BALANCED 4-detector portfolio — locked Week 8, 2026-04-29.
// Backtest: portfolio wf_p5 1.486, holdout PF 1.960 (run 92373355).
const V8_STRATEGIES = [
  {
    key: "v8_macd_pullback_short_e2",
    label: "Pullback Short",
    dir: "SHORT",
    desc: "daily MACD bear + 4h cross-down ≥2 prior above",
    exit: "E2 (2× ATR SL / 6× ATR TP)",
    wfp5: "2.10",
    signal_table: "scanner_signals_macd_pullback_short",
  },
  {
    key: "v8_macd_early_trend_short_e2",
    label: "Early Trend Short",
    dir: "SHORT",
    desc: "fresh bear ≤10d since bear-flip + 4h cross-down",
    exit: "E2 (2× ATR SL / 6× ATR TP)",
    wfp5: "2.05",
    signal_table: "scanner_signals_macd_early_trend_short",
  },
  {
    key: "v8_macd_pullback_long_e2",
    label: "Pullback Long",
    dir: "LONG",
    desc: "daily MACD bull + 4h cross-up ≥2 prior below",
    exit: "E2 (2× ATR SL / 6× ATR TP)",
    wfp5: "1.10",
    signal_table: "scanner_signals_macd_pullback_long",
  },
  {
    key: "v8_rsi_recovery_long_e2",
    label: "RSI Recovery Long",
    dir: "LONG",
    desc: "daily RSI(14) cross-up through 35 (oversold recovery)",
    exit: "E2 (2× ATR SL / 6× ATR TP)",
    wfp5: "1.81",
    signal_table: "scanner_signals_rsi_recovery_long",
  },
];

interface Props {
  searchParams: Promise<{ strategy?: string; page?: string }>;
}

export default async function PaperV8Page({ searchParams }: Props) {
  const params = await searchParams;
  const selectedStrategy = params.strategy || "all";
  const page = Math.max(1, parseInt(params.page || "1") || 1);

  // Per-strategy summary
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
      WHERE strategy LIKE 'v8_%' AND threshold = ${V8_SENTINEL_THRESHOLD}
      GROUP BY strategy
      ORDER BY strategy
    `);
  } catch {}
  const summaryByKey: Record<string, any> = Object.fromEntries(
    strategySummary.map((r: any) => [r.strategy, r])
  );

  // Portfolio totals
  const totalTrades = strategySummary.reduce((acc: number, r: any) => acc + Number(r.trades || 0), 0);
  const totalPnl = strategySummary.reduce((acc: number, r: any) => acc + Number(r.total_pnl || 0), 0);
  const portfolioEquity = STARTING_EQUITY * V8_STRATEGIES.length + totalPnl;

  // Per-scanner signal counts (one query per table, all try/catched)
  const signalCounts: Record<string, { total: number; latest: Date | null }> = {};
  for (const spec of V8_STRATEGIES) {
    try {
      const rows: any[] = await prisma.$queryRawUnsafe(`
        SELECT COUNT(*)::int AS total, MAX(signal_time) AS latest
        FROM ${spec.signal_table}
      `);
      signalCounts[spec.key] = { total: rows[0]?.total || 0, latest: rows[0]?.latest || null };
    } catch {
      signalCounts[spec.key] = { total: 0, latest: null };
    }
  }
  const totalSignals = Object.values(signalCounts).reduce((a, b) => a + b.total, 0);

  // Open positions
  const stratClause =
    selectedStrategy !== "all"
      ? `AND strategy = '${selectedStrategy.replace(/[^a-z0-9_]/gi, "")}'`
      : `AND strategy LIKE 'v8_%'`;
  let openTrades: any[] = [];
  try {
    openTrades = await prisma.$queryRawUnsafe(`
      SELECT id, strategy, symbol, direction, entry_time, entry_price, position_usd,
             tp_price, sl_price, atr14_at_entry, btc_score, pos_scale, exit_kind
      FROM paper_trades
      WHERE status='open' AND threshold = ${V8_SENTINEL_THRESHOLD} ${stratClause}
      ORDER BY entry_time DESC
      LIMIT 50
    `);
  } catch {}

  // Closed trades, paginated
  let closedTrades: any[] = [];
  let totalClosed = 0;
  try {
    const cr: any[] = await prisma.$queryRawUnsafe(`
      SELECT COUNT(*)::int AS n FROM paper_trades
      WHERE status IN ('won','lost','timeout')
        AND threshold = ${V8_SENTINEL_THRESHOLD} ${stratClause}
    `);
    totalClosed = cr[0]?.n || 0;
    closedTrades = await prisma.$queryRawUnsafe(`
      SELECT id, strategy, symbol, direction, entry_time, exit_time, entry_price, exit_price,
             position_usd, pnl_pct, pnl_usd, exit_reason, equity_after, pos_scale, exit_kind
      FROM paper_trades
      WHERE status IN ('won','lost','timeout')
        AND threshold = ${V8_SENTINEL_THRESHOLD} ${stratClause}
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
            Paper Trading — v8 BALANCED{" "}
            <span className="text-sm font-normal text-yellow-400">DRY-RUN · No real orders</span>
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            4-detector portfolio · portfolio wf_p5&nbsp;1.486 · holdout PF&nbsp;1.96.
            Starting equity ${STARTING_EQUITY.toFixed(2)} per account · phase-switch ATR sizing (v6).
            Source:{" "}
            <code className="text-slate-300">
              scanner_signals_macd_pullback_short + _early_trend_short + _pullback_long + _rsi_recovery_long
            </code>
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Link href="/paper-d1e" className="text-xs text-slate-400 hover:text-white px-3 py-1.5 rounded bg-slate-800 hover:bg-slate-700">
            ← D1e (Phase 17)
          </Link>
          <Link href="/paper" className="text-xs text-slate-400 hover:text-white px-3 py-1.5 rounded bg-slate-800 hover:bg-slate-700">
            v2 multi-threshold
          </Link>
        </div>
      </div>

      {/* Portfolio KPIs */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <KPI
          label="Total Signals (4 scanners)"
          value={`${totalSignals}`}
          hint="cumulative across all v8 scanner tables"
        />
        <KPI
          label="Total Trades Opened"
          value={`${totalTrades}`}
          color={totalTrades > 0 ? "brand" : "slate"}
          hint={`${totalClosed} closed`}
        />
        <KPI
          label="Portfolio PnL"
          value={`$${totalPnl >= 0 ? "+" : ""}${totalPnl.toFixed(2)}`}
          color={totalPnl >= 0 ? "green" : "red"}
          hint={`combined across all 4 accounts`}
        />
        <KPI
          label="Portfolio Equity"
          value={`$${portfolioEquity.toFixed(2)}`}
          color={portfolioEquity >= STARTING_EQUITY * V8_STRATEGIES.length ? "green" : "red"}
          hint={`seed $${(STARTING_EQUITY * V8_STRATEGIES.length).toFixed(0)} (4 × $${STARTING_EQUITY})`}
        />
      </div>

      {/* Per-strategy comparison */}
      <div className="card overflow-hidden p-0">
        <div className="px-4 py-3 border-b border-[var(--border)]">
          <h2 className="text-sm font-semibold text-slate-200">
            Strategy Comparison (4 v8 BALANCED detectors)
          </h2>
        </div>
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-[var(--border)]">
              <th className="px-3 py-2 text-left text-slate-500">Strategy</th>
              <th className="px-3 py-2 text-left text-slate-500">Dir</th>
              <th className="px-3 py-2 text-left text-slate-500">Detector</th>
              <th className="px-3 py-2 text-right text-slate-500">wf_p5</th>
              <th className="px-3 py-2 text-right text-slate-500">Signals</th>
              <th className="px-3 py-2 text-right text-slate-500">Trades</th>
              <th className="px-3 py-2 text-right text-slate-500">Open</th>
              <th className="px-3 py-2 text-right text-slate-500">WR</th>
              <th className="px-3 py-2 text-right text-slate-500">PF</th>
              <th className="px-3 py-2 text-right text-slate-500">PnL</th>
              <th className="px-3 py-2 text-right text-slate-500">Equity</th>
            </tr>
          </thead>
          <tbody>
            {V8_STRATEGIES.map((spec) => {
              const r = summaryByKey[spec.key];
              const trades = Number(r?.trades || 0);
              const isSelected = selectedStrategy === spec.key;
              const sig = signalCounts[spec.key] || { total: 0, latest: null };
              return (
                <tr
                  key={spec.key}
                  className={`border-b border-[var(--border)] ${isSelected ? "bg-brand-500/10" : "hover:bg-slate-800/50"}`}
                >
                  <td className="px-3 py-1.5">
                    <Link
                      href={`/paper-v8?strategy=${selectedStrategy === spec.key ? "all" : spec.key}`}
                      className={`font-mono ${isSelected ? "text-brand-400 font-bold" : "text-slate-300 hover:text-white"}`}
                    >
                      {spec.label} {isSelected ? "◄" : ""}
                    </Link>
                  </td>
                  <td className="px-3 py-1.5">
                    <span className={`text-[10px] px-1.5 py-0.5 rounded font-mono ${
                      spec.dir === "SHORT" ? "bg-red-900/50 text-red-400" : "bg-green-900/50 text-green-400"
                    }`}>
                      {spec.dir}
                    </span>
                  </td>
                  <td className="px-3 py-1.5 text-slate-400 text-[10px]">{spec.desc}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-brand-400">{spec.wfp5}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-400">{sig.total}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-300">{trades}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-blue-400">{Number(r?.open_n || 0)}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-300">
                    {trades > 0 ? `${Number(r?.wr || 0).toFixed(1)}%` : "—"}
                  </td>
                  <td className={`px-3 py-1.5 text-right font-mono ${
                    Number(r?.pf || 0) >= 1.5 ? "text-green-400"
                      : Number(r?.pf || 0) >= 1 ? "text-yellow-400"
                      : trades > 0 ? "text-red-400" : "text-slate-500"
                  }`}>
                    {trades > 0 ? Number(r?.pf || 0).toFixed(2) : "—"}
                  </td>
                  <td className={`px-3 py-1.5 text-right font-mono ${
                    Number(r?.total_pnl || 0) >= 0 ? "text-green-400" : "text-red-400"
                  }`}>
                    ${Number(r?.total_pnl || 0).toFixed(2)}
                  </td>
                  <td className={`px-3 py-1.5 text-right font-mono ${
                    Number(r?.equity || STARTING_EQUITY) >= STARTING_EQUITY ? "text-green-400" : "text-red-400"
                  }`}>
                    ${Number(r?.equity || STARTING_EQUITY).toFixed(2)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {selectedStrategy !== "all" && (
          <div className="px-4 py-2 border-t border-[var(--border)] text-xs text-slate-500">
            Filtered to{" "}
            <span className="text-brand-400 font-mono">{selectedStrategy}</span>.{" "}
            <Link href="/paper-v8" className="hover:text-white underline">clear filter</Link>
          </div>
        )}
      </div>

      {/* Open positions */}
      {openTrades.length > 0 && (
        <div className="card overflow-hidden p-0">
          <div className="px-4 py-3 border-b border-[var(--border)]">
            <h2 className="text-sm font-semibold text-slate-200">
              Open Positions ({openTrades.length})
            </h2>
          </div>
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-[var(--border)]">
                <th className="px-3 py-2 text-left text-slate-500">Entry</th>
                <th className="px-3 py-2 text-left text-slate-500">Strategy</th>
                <th className="px-3 py-2 text-left text-slate-500">Symbol</th>
                <th className="px-3 py-2 text-center text-slate-500">Dir</th>
                <th className="px-3 py-2 text-right text-slate-500">BTC</th>
                <th className="px-3 py-2 text-right text-slate-500">Scale</th>
                <th className="px-3 py-2 text-right text-slate-500">Entry $</th>
                <th className="px-3 py-2 text-right text-slate-500">Pos $</th>
                <th className="px-3 py-2 text-right text-slate-500">SL</th>
                <th className="px-3 py-2 text-right text-slate-500">TP</th>
                <th className="px-3 py-2 text-center text-slate-500">Exit</th>
              </tr>
            </thead>
            <tbody>
              {openTrades.map((t: any) => (
                <tr key={t.id} className="border-b border-[var(--border)] hover:bg-slate-800/50">
                  <td className="px-3 py-1.5 text-slate-400 font-mono whitespace-nowrap">
                    {new Date(t.entry_time).toLocaleString("en-CA", {
                      month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit",
                    })}
                  </td>
                  <td className="px-3 py-1.5 font-mono text-slate-300 text-[10px]">
                    {(t.strategy || "").replace("v8_", "").replace(/_e2$/, "")}
                  </td>
                  <td className="px-3 py-1.5 text-white font-medium">{t.symbol.replace("USDT", "")}</td>
                  <td className="px-3 py-1.5 text-center">
                    <span className={`text-[10px] px-1 rounded ${
                      t.direction === "short" ? "text-red-400" : "text-green-400"
                    }`}>
                      {t.direction?.toUpperCase()}
                    </span>
                  </td>
                  <td className={`px-3 py-1.5 text-right font-mono ${
                    Number(t.btc_score) < 0 ? "text-red-400" : "text-slate-500"
                  }`}>
                    {t.btc_score != null ? Number(t.btc_score).toFixed(2) : "—"}
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-300">
                    {t.pos_scale != null ? `${(Number(t.pos_scale) * 100).toFixed(1)}%` : "—"}
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-300">
                    ${Number(t.entry_price).toPrecision(4)}
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-300">
                    ${Number(t.position_usd).toFixed(2)}
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-red-400">
                    ${Number(t.sl_price).toPrecision(4)}
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-green-400">
                    ${Number(t.tp_price).toPrecision(4)}
                  </td>
                  <td className="px-3 py-1.5 text-center font-mono text-slate-400 uppercase text-[10px]">
                    {t.exit_kind || "—"}
                  </td>
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
            {page > 1 && (
              <Link
                href={`?page=${page - 1}${selectedStrategy !== "all" ? `&strategy=${selectedStrategy}` : ""}`}
                className="px-2 py-1 rounded bg-slate-800 text-slate-300 hover:bg-slate-700"
              >
                Prev
              </Link>
            )}
            <span className="text-slate-500">{page}/{totalPages}</span>
            {page < totalPages && (
              <Link
                href={`?page=${page + 1}${selectedStrategy !== "all" ? `&strategy=${selectedStrategy}` : ""}`}
                className="px-2 py-1 rounded bg-slate-800 text-slate-300 hover:bg-slate-700"
              >
                Next
              </Link>
            )}
          </div>
        </div>
        {closedTrades.length === 0 ? (
          <div className="px-4 py-8 text-xs text-slate-500 text-center">
            No closed v8 trades yet.{" "}
            {totalSignals === 0
              ? "Scanners have not produced any signals — detectors may not have triggered yet."
              : `${totalSignals} signal(s) fired, trades not closed yet.`}
          </div>
        ) : (
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-[var(--border)]">
                <th className="px-3 py-2 text-left text-slate-500">Exit</th>
                <th className="px-3 py-2 text-left text-slate-500">Strategy</th>
                <th className="px-3 py-2 text-left text-slate-500">Symbol</th>
                <th className="px-3 py-2 text-center text-slate-500">Dir</th>
                <th className="px-3 py-2 text-right text-slate-500">Scale</th>
                <th className="px-3 py-2 text-right text-slate-500">Entry $</th>
                <th className="px-3 py-2 text-right text-slate-500">Exit $</th>
                <th className="px-3 py-2 text-right text-slate-500">PnL %</th>
                <th className="px-3 py-2 text-right text-slate-500">PnL $</th>
                <th className="px-3 py-2 text-center text-slate-500">Reason</th>
                <th className="px-3 py-2 text-right text-slate-500">Equity</th>
              </tr>
            </thead>
            <tbody>
              {closedTrades.map((t: any) => (
                <tr key={t.id} className="border-b border-[var(--border)] hover:bg-slate-800/50">
                  <td className="px-3 py-1.5 text-slate-400 font-mono whitespace-nowrap">
                    {t.exit_time
                      ? new Date(t.exit_time).toLocaleString("en-CA", {
                          month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit",
                        })
                      : "—"}
                  </td>
                  <td className="px-3 py-1.5 font-mono text-slate-300 text-[10px]">
                    {(t.strategy || "").replace("v8_", "").replace(/_e2$/, "")}
                  </td>
                  <td className="px-3 py-1.5 text-white font-medium">{t.symbol.replace("USDT", "")}</td>
                  <td className="px-3 py-1.5 text-center">
                    <span className={`text-[10px] px-1 rounded ${
                      t.direction === "short" ? "text-red-400" : "text-green-400"
                    }`}>
                      {t.direction?.toUpperCase()}
                    </span>
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-400">
                    {t.pos_scale != null ? `${(Number(t.pos_scale) * 100).toFixed(1)}%` : "—"}
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-300">
                    ${Number(t.entry_price).toPrecision(4)}
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-300">
                    ${Number(t.exit_price).toPrecision(4)}
                  </td>
                  <td className={`px-3 py-1.5 text-right font-mono ${
                    Number(t.pnl_pct) >= 0 ? "text-green-400" : "text-red-400"
                  }`}>
                    {Number(t.pnl_pct) >= 0 ? "+" : ""}
                    {(Number(t.pnl_pct) * 100).toFixed(2)}%
                  </td>
                  <td className={`px-3 py-1.5 text-right font-mono ${
                    Number(t.pnl_usd) >= 0 ? "text-green-400" : "text-red-400"
                  }`}>
                    {Number(t.pnl_usd) > 0 ? "+" : ""}${Number(t.pnl_usd).toFixed(2)}
                  </td>
                  <td className="px-3 py-1.5 text-center">
                    <span className={`text-[10px] px-1.5 py-0.5 rounded ${
                      t.exit_reason === "take_profit"
                        ? "bg-green-900/50 text-green-400"
                        : t.exit_reason === "stop_loss"
                        ? "bg-red-900/50 text-red-400"
                        : "bg-slate-700 text-slate-400"
                    }`}>
                      {t.exit_reason}
                    </span>
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

function KPI({
  label,
  value,
  hint,
  color,
}: {
  label: string;
  value: string;
  hint?: string;
  color?: string;
}) {
  const c =
    color === "green"
      ? "text-green-400"
      : color === "red"
      ? "text-red-400"
      : color === "brand"
      ? "text-brand-400"
      : color === "slate"
      ? "text-slate-400"
      : "text-white";
  return (
    <div className="card p-3">
      <p className="text-[10px] text-slate-500 uppercase tracking-wide">{label}</p>
      <p className={`text-lg font-bold mt-0.5 ${c}`}>{value}</p>
      {hint && <p className="text-[10px] text-slate-500 mt-0.5">{hint}</p>}
    </div>
  );
}
