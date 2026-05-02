import type { ReactNode } from "react";
import Link from "next/link";
import { prisma } from "@/lib/prisma";
import { SystemStatus } from "@/components/system-status";

const STARTING_EQUITY = 200.0;
const V8_SENTINEL_THRESHOLD = 3.0;
const PER_PAGE = 30;

// Filter-era cutoff. Paper trades whose entry_time predates this are hidden
// from this dashboard so the view reflects the system as it actually runs
// today: BGM classifier scoring + per-detector cls_score floor (0.40 on
// macd_pullback_long, see paper_executor.py V8_CLS_SCORE_FLOOR). Pre-cutoff
// trades are still in the DB and visible on /paper-v8 (the baseline page).
//
// Ratchet this date forward when meaningful policy changes ship (new floors,
// new sizing) so the live numbers you read here reflect the current rules.
const FILTER_ERA_CUTOFF = "2026-05-01T03:11:00Z";

// Trade-quality classifiers (sizing-mode) — production HistGradientBoosting models.
// AUC + n_train values come from the 3-year backtest validation
// (see services/python/models/v8_classifier_3y/*_meta.json).
const V8_ML_STRATEGIES = [
  {
    key: "v8_macd_pullback_long_e2",
    label: "Pullback Long",
    dir: "LONG",
    desc: "daily MACD bull + 4h cross-up ≥2 prior below",
    auc_holdout: 0.660,
    n_train: 9102,
    sizing_lift_pct: 1.34, // % equity lift on 2026Q1 holdout vs baseline
    classifier_status: "ENABLED",
  },
  {
    key: "v8_macd_early_trend_short_e2",
    label: "Early Trend Short",
    dir: "SHORT",
    desc: "fresh bear ≤10d since bear-flip + 4h cross-down",
    auc_holdout: 0.671,
    n_train: 1391,
    sizing_lift_pct: 0.03,
    classifier_status: "ENABLED",
  },
  {
    key: "v8_macd_pullback_short_e2",
    label: "Pullback Short",
    dir: "SHORT",
    desc: "daily MACD bear + 4h cross-down ≥2 prior above",
    auc_holdout: 0.584,
    n_train: 2480,
    sizing_lift_pct: -0.34,
    classifier_status: "ENABLED",
  },
  {
    key: "v8_rsi_recovery_long_e2",
    label: "RSI Recovery Long",
    dir: "LONG",
    desc: "daily RSI(14) cross-up through 35 (oversold recovery)",
    auc_holdout: null,
    n_train: 0,
    sizing_lift_pct: null,
    classifier_status: "RULE-BASED (only 3 trades in 3y data)",
  },
];

interface Props {
  searchParams: Promise<{ strategy?: string; page?: string }>;
}

export default async function PaperV8MLPage({ searchParams }: Props) {
  const params = await searchParams;
  const selectedStrategy = params.strategy || "all";
  const page = Math.max(1, parseInt(params.page || "1") || 1);

  // Per-strategy summary, scoped to v8 trades. The classifier writes its
  // probability score to ml_prob (replacing the old sentinel 1.0). We
  // distinguish "scored" trades (ml_prob != 1.0) from pre-classifier
  // legacy rows (ml_prob = 1.0). total_pnl is LIVE (classifier-sized).
  // total_pnl_baseline is the no-classifier counterfactual, computed from
  // pnl_usd_unscaled (NULL → COALESCE to pnl_usd, which equals the live
  // value for pre-baseline-column rows). The gap = BGM contribution.
  let strategySummary: any[] = [];
  try {
    strategySummary = await prisma.$queryRawUnsafe(`
      SELECT
        strategy,
        COUNT(*)::int AS trades_total,
        COUNT(*) FILTER (WHERE ml_prob != 1.0)::int AS trades_scored,
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
        ROUND(COALESCE(SUM(COALESCE(pnl_usd_unscaled, pnl_usd)), 0)::numeric, 2) AS total_pnl_baseline,
        ROUND((${STARTING_EQUITY} + COALESCE(SUM(pnl_usd), 0))::numeric, 2) AS equity,
        ROUND(AVG(ml_prob) FILTER (WHERE ml_prob != 1.0)::numeric, 3) AS avg_score,
        ROUND(AVG(pos_scale) FILTER (WHERE ml_prob != 1.0)::numeric, 3) AS avg_size_pct,
        MAX(entry_time) AS latest_entry
      FROM paper_trades
      WHERE strategy LIKE 'v8_%' AND threshold = ${V8_SENTINEL_THRESHOLD}
        AND entry_time >= '${FILTER_ERA_CUTOFF}'::timestamptz
      GROUP BY strategy
      ORDER BY strategy
    `);
  } catch {}
  const summaryByKey: Record<string, any> = Object.fromEntries(
    strategySummary.map((r: any) => [r.strategy, r])
  );

  // Portfolio totals (across the 4 v8 accounts)
  const totalTrades = strategySummary.reduce((acc: number, r: any) => acc + Number(r.trades_total || 0), 0);
  const totalScored = strategySummary.reduce((acc: number, r: any) => acc + Number(r.trades_scored || 0), 0);
  const totalPnl = strategySummary.reduce((acc: number, r: any) => acc + Number(r.total_pnl || 0), 0);
  const totalPnlBaseline = strategySummary.reduce(
    (acc: number, r: any) => acc + Number(r.total_pnl_baseline || 0),
    0,
  );
  // BGM contribution = live PnL minus the no-classifier counterfactual.
  // Positive = sizing helped; negative = sizing hurt.
  const bgmLift = totalPnl - totalPnlBaseline;
  const seedTotal = STARTING_EQUITY * V8_ML_STRATEGIES.length;
  const bgmLiftPctOfSeed = seedTotal > 0 ? (bgmLift / seedTotal) * 100 : 0;
  const portfolioEquity = seedTotal + totalPnl;

  // Open positions — scoped to the selected strategy (or all v8)
  const stratClause =
    selectedStrategy !== "all"
      ? `AND strategy = '${selectedStrategy.replace(/[^a-z0-9_]/gi, "")}'`
      : `AND strategy LIKE 'v8_%'`;
  let openTrades: any[] = [];
  try {
    openTrades = await prisma.$queryRawUnsafe(`
      SELECT id, strategy, symbol, direction, entry_time, entry_price, position_usd,
             tp_price, sl_price, ml_prob, pos_scale, btc_score, exit_kind
      FROM paper_trades
      WHERE status='open' AND threshold = ${V8_SENTINEL_THRESHOLD} ${stratClause}
        AND entry_time >= '${FILTER_ERA_CUTOFF}'::timestamptz
      ORDER BY entry_time DESC
      LIMIT 50
    `);
  } catch {}

  // Closed trades, paginated (only show scored trades by default — toggle later if needed)
  let closedTrades: any[] = [];
  let totalClosed = 0;
  try {
    const cr: any[] = await prisma.$queryRawUnsafe(`
      SELECT COUNT(*)::int AS n FROM paper_trades
      WHERE status IN ('won','lost','timeout')
        AND threshold = ${V8_SENTINEL_THRESHOLD} ${stratClause}
        AND entry_time >= '${FILTER_ERA_CUTOFF}'::timestamptz
    `);
    totalClosed = cr[0]?.n || 0;
    closedTrades = await prisma.$queryRawUnsafe(`
      SELECT id, strategy, symbol, direction, entry_time, exit_time, entry_price, exit_price,
             position_usd, pnl_pct, pnl_usd, exit_reason, equity_after,
             ml_prob, pos_scale, exit_kind
      FROM paper_trades
      WHERE status IN ('won','lost','timeout')
        AND threshold = ${V8_SENTINEL_THRESHOLD} ${stratClause}
        AND entry_time >= '${FILTER_ERA_CUTOFF}'::timestamptz
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
            Paper Trading — v8 BALANCED with ML Classifier{" "}
            <span className="text-sm font-normal text-yellow-400">DRY-RUN · sizing-mode</span>
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            <span className="text-brand-400 font-semibold">Filter-era only:</span>{" "}
            showing trades opened on or after{" "}
            <code className="text-slate-300">{FILTER_ERA_CUTOFF}</code> — the
            point at which the cls_score floor (0.40 on{" "}
            <code className="text-slate-300">macd_pullback_long</code>) went
            live. Pre-cutoff trades (pre-classifier deploys, plus the noisy 70-
            trade unfiltered window) are hidden here; they remain on{" "}
            <Link href="/paper-v8" className="underline hover:text-white">
              /paper-v8
            </Link>{" "}
            (the baseline page) and in the raw{" "}
            <code className="text-slate-300">paper_trades</code> table.{" "}
            Per-detector HistGradientBoosting classifier sizes each entry by a
            multiplier in <span className="text-slate-300">[0.50, 1.50]</span>{" "}
            (capital-neutral on the train distribution). Live classifier score
            is stored in <code className="text-slate-300">ml_prob</code>;
            adjusted size in <code className="text-slate-300">pos_scale</code>.
            Models trained on 3-year snapshot 2026-04-01, validated on 2026Q1
            holdout.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Link href="/paper-v8" className="text-xs text-slate-400 hover:text-white px-3 py-1.5 rounded bg-slate-800 hover:bg-slate-700">
            ← v8 baseline
          </Link>
          <Link href="/paper-d1e" className="text-xs text-slate-400 hover:text-white px-3 py-1.5 rounded bg-slate-800 hover:bg-slate-700">
            D1e (Phase 17)
          </Link>
        </div>
      </div>

      {/* Portfolio KPIs */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <KPI
          label="Total v8 Trades"
          value={`${totalTrades}`}
          color={totalTrades > 0 ? "brand" : "slate"}
          hint={`${totalScored} scored by classifier`}
        />
        <KPI
          label="Portfolio PnL (Live)"
          value={`$${totalPnl >= 0 ? "+" : ""}${totalPnl.toFixed(2)}`}
          color={totalPnl >= 0 ? "green" : "red"}
          hint="with BGM sizing"
        />
        <KPI
          label="Portfolio PnL (Baseline)"
          value={`$${totalPnlBaseline >= 0 ? "+" : ""}${totalPnlBaseline.toFixed(2)}`}
          color={totalPnlBaseline >= 0 ? "green" : "red"}
          hint={
            <>
              cls_multiplier = 1.0 ·{" "}
              <Link href="/paper-v8" className="underline hover:text-white">/paper-v8</Link>
            </>
          }
        />
        <KPI
          label="BGM Lift"
          value={`${bgmLift >= 0 ? "+" : ""}$${bgmLift.toFixed(2)}`}
          color={bgmLift >= 0 ? "green" : "red"}
          hint={`${bgmLift >= 0 ? "+" : ""}${bgmLiftPctOfSeed.toFixed(2)}% of seed (live − baseline)`}
        />
        <KPI
          label="Portfolio Equity"
          value={`$${portfolioEquity.toFixed(2)}`}
          color={portfolioEquity >= seedTotal ? "green" : "red"}
          hint={`seed $${seedTotal.toFixed(0)} (4 × $${STARTING_EQUITY})`}
        />
      </div>

      {/* Per-strategy comparison with classifier status */}
      <div className="card overflow-hidden p-0">
        <div className="px-4 py-3 border-b border-[var(--border)]">
          <h2 className="text-sm font-semibold text-slate-200">
            Strategy Comparison — Classifier Status & Live Performance
          </h2>
          <p className="text-xs text-slate-500 mt-1">
            AUC + sizing lift values are from the 2026Q1 backtest holdout.
            Live values are running averages since the classifier was deployed.
          </p>
        </div>
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-[var(--border)]">
              <th className="px-3 py-2 text-left text-slate-500">Strategy</th>
              <th className="px-3 py-2 text-left text-slate-500">Dir</th>
              <th className="px-3 py-2 text-left text-slate-500">Detector</th>
              <th className="px-3 py-2 text-right text-slate-500">Classifier</th>
              <th className="px-3 py-2 text-right text-slate-500">AUC</th>
              <th className="px-3 py-2 text-right text-slate-500">n train</th>
              <th className="px-3 py-2 text-right text-slate-500">Live n</th>
              <th className="px-3 py-2 text-right text-slate-500">Scored</th>
              <th className="px-3 py-2 text-right text-slate-500">Avg Score</th>
              <th className="px-3 py-2 text-right text-slate-500">Avg Size%</th>
              <th className="px-3 py-2 text-right text-slate-500">WR</th>
              <th className="px-3 py-2 text-right text-slate-500">PF</th>
              <th className="px-3 py-2 text-right text-slate-500">PnL</th>
            </tr>
          </thead>
          <tbody>
            {V8_ML_STRATEGIES.map((spec) => {
              const r = summaryByKey[spec.key];
              const trades_total = Number(r?.trades_total || 0);
              const trades_scored = Number(r?.trades_scored || 0);
              const isSelected = selectedStrategy === spec.key;
              const cls_on = spec.classifier_status === "ENABLED";
              return (
                <tr
                  key={spec.key}
                  className={`border-b border-[var(--border)] ${isSelected ? "bg-brand-500/10" : "hover:bg-slate-800/50"}`}
                >
                  <td className="px-3 py-1.5">
                    <Link
                      href={`/paper-v8-ml?strategy=${selectedStrategy === spec.key ? "all" : spec.key}`}
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
                  <td className="px-3 py-1.5 text-right">
                    <span className={`text-[10px] px-1.5 py-0.5 rounded font-mono ${
                      cls_on ? "bg-brand-900/50 text-brand-400" : "bg-slate-700 text-slate-400"
                    }`}>
                      {cls_on ? "ML" : "RULES"}
                    </span>
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-400">
                    {spec.auc_holdout != null ? spec.auc_holdout.toFixed(3) : "—"}
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-500">
                    {spec.n_train > 0 ? spec.n_train.toLocaleString() : "—"}
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-300">{trades_total}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-brand-400">{trades_scored}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-300">
                    {trades_scored > 0 ? Number(r?.avg_score || 0).toFixed(3) : "—"}
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-300">
                    {trades_scored > 0 ? `${(Number(r?.avg_size_pct || 0) * 100).toFixed(2)}%` : "—"}
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-300">
                    {trades_total > 0 ? `${Number(r?.wr || 0).toFixed(1)}%` : "—"}
                  </td>
                  <td className={`px-3 py-1.5 text-right font-mono ${
                    Number(r?.pf || 0) >= 1.5 ? "text-green-400"
                      : Number(r?.pf || 0) >= 1 ? "text-yellow-400"
                      : trades_total > 0 ? "text-red-400" : "text-slate-500"
                  }`}>
                    {trades_total > 0 ? Number(r?.pf || 0).toFixed(2) : "—"}
                  </td>
                  <td className={`px-3 py-1.5 text-right font-mono ${
                    Number(r?.total_pnl || 0) >= 0 ? "text-green-400" : "text-red-400"
                  }`}>
                    ${Number(r?.total_pnl || 0).toFixed(2)}
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
            <Link href="/paper-v8-ml" className="hover:text-white underline">clear filter</Link>
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
                <th className="px-3 py-2 text-right text-slate-500">Score</th>
                <th className="px-3 py-2 text-right text-slate-500">Size%</th>
                <th className="px-3 py-2 text-right text-slate-500">BTC</th>
                <th className="px-3 py-2 text-right text-slate-500">Entry $</th>
                <th className="px-3 py-2 text-right text-slate-500">Pos $</th>
                <th className="px-3 py-2 text-right text-slate-500">SL</th>
                <th className="px-3 py-2 text-right text-slate-500">TP</th>
              </tr>
            </thead>
            <tbody>
              {openTrades.map((t: any) => {
                const score = Number(t.ml_prob);
                const isScored = score !== 1.0;
                return (
                  <tr key={t.id} className="border-b border-[var(--border)] hover:bg-slate-800/50">
                    <td className="px-3 py-1.5 text-slate-400 font-mono whitespace-nowrap">
                      {new Date(t.entry_time).toLocaleString("en-CA", {
                        month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit", timeZone: "Asia/Jakarta",
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
                      !isScored ? "text-slate-500"
                        : score >= 0.5 ? "text-green-400"
                        : score >= 0.3 ? "text-yellow-400" : "text-orange-400"
                    }`}>
                      {isScored ? score.toFixed(3) : "1.000⚠"}
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono text-slate-300">
                      {t.pos_scale != null ? `${(Number(t.pos_scale) * 100).toFixed(1)}%` : "—"}
                    </td>
                    <td className={`px-3 py-1.5 text-right font-mono ${
                      Number(t.btc_score) < 0 ? "text-red-400" : "text-slate-500"
                    }`}>
                      {t.btc_score != null ? Number(t.btc_score).toFixed(2) : "—"}
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
                  </tr>
                );
              })}
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
            No closed v8 trades yet.
          </div>
        ) : (
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-[var(--border)]">
                <th className="px-3 py-2 text-left text-slate-500">Exit</th>
                <th className="px-3 py-2 text-left text-slate-500">Strategy</th>
                <th className="px-3 py-2 text-left text-slate-500">Symbol</th>
                <th className="px-3 py-2 text-center text-slate-500">Dir</th>
                <th className="px-3 py-2 text-right text-slate-500">Score</th>
                <th className="px-3 py-2 text-right text-slate-500">Size%</th>
                <th className="px-3 py-2 text-right text-slate-500">Entry $</th>
                <th className="px-3 py-2 text-right text-slate-500">Exit $</th>
                <th className="px-3 py-2 text-right text-slate-500">PnL %</th>
                <th className="px-3 py-2 text-right text-slate-500">PnL $</th>
                <th className="px-3 py-2 text-center text-slate-500">Reason</th>
                <th className="px-3 py-2 text-right text-slate-500">Equity</th>
              </tr>
            </thead>
            <tbody>
              {closedTrades.map((t: any) => {
                const score = Number(t.ml_prob);
                const isScored = score !== 1.0;
                return (
                  <tr key={t.id} className="border-b border-[var(--border)] hover:bg-slate-800/50">
                    <td className="px-3 py-1.5 text-slate-400 font-mono whitespace-nowrap">
                      {t.exit_time
                        ? new Date(t.exit_time).toLocaleString("en-CA", {
                            month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit", timeZone: "Asia/Jakarta",
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
                    <td className={`px-3 py-1.5 text-right font-mono ${
                      !isScored ? "text-slate-500"
                        : score >= 0.5 ? "text-green-400"
                        : score >= 0.3 ? "text-yellow-400" : "text-orange-400"
                    }`}>
                      {isScored ? score.toFixed(3) : "1.000⚠"}
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
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      {/* Footer note explaining the score color scheme */}
      <div className="card p-4 text-xs text-slate-400 space-y-1">
        <p>
          <span className="text-slate-300 font-semibold">Score color legend:</span>{" "}
          <span className="text-green-400 font-mono">≥ 0.50</span> high-confidence trade (size multiplier &gt; 1.0),{" "}
          <span className="text-yellow-400 font-mono">0.30–0.50</span> medium,{" "}
          <span className="text-orange-400 font-mono">&lt; 0.30</span> low-confidence (size scaled down toward 0.5×),{" "}
          <span className="text-slate-500 font-mono">1.000⚠</span> pre-classifier deploy (sentinel — no scoring applied).
        </p>
        <p>
          The classifier was trained to be capital-neutral: average size multiplier across the 3-year train distribution
          is exactly 1.0. Above-average scores get larger sizes; below-average get smaller. Total capital deployed
          on average matches the baseline v6 phase-switch sizing.
        </p>
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
  hint?: ReactNode;
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
