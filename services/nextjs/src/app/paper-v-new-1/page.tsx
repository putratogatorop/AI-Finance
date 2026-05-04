import type { ReactNode } from "react";
import Link from "next/link";
import { prisma } from "@/lib/prisma";

const PER_PAGE = 50;
const SEVEN_DAYS_AGO_SQL = "NOW() - INTERVAL '7 days'";
const ONE_DAY_AGO_SQL = "NOW() - INTERVAL '24 hours'";

interface Props {
  searchParams: Promise<{ page?: string; direction?: string }>;
}

export default async function PaperVNew1Page({ searchParams }: Props) {
  const params = await searchParams;
  const page = Math.max(1, parseInt(params.page || "1") || 1);
  const dirFilter = params.direction || "all";

  // ── Live status: latest run + 24h signal count + latest CFGI ─────────────
  let statusRow: any = null;
  try {
    const rows: any[] = await prisma.$queryRawUnsafe(`
      SELECT
        MAX(timestamp) AS last_run,
        COUNT(*) FILTER (WHERE timestamp >= ${ONE_DAY_AGO_SQL})::int AS signals_24h,
        COUNT(*) FILTER (WHERE action = 'taken' AND timestamp >= ${ONE_DAY_AGO_SQL})::int AS taken_24h,
        (SELECT cfgi_value FROM v_new_1_shadow_signals ORDER BY timestamp DESC LIMIT 1) AS latest_cfgi
      FROM v_new_1_shadow_signals
    `);
    statusRow = rows[0] ?? null;
  } catch {}

  // ── 7-day stats summary (by direction) ───────────────────────────────────
  let statsRows: any[] = [];
  try {
    statsRows = await prisma.$queryRawUnsafe(`
      SELECT
        direction,
        COUNT(*) FILTER (WHERE action = 'taken')::int AS taken_n,
        COUNT(*) FILTER (WHERE action != 'taken')::int AS skipped_n,
        ROUND(AVG(score_platt) FILTER (WHERE action = 'taken')::numeric, 4) AS avg_score_taken,
        ROUND(AVG(score_platt)::numeric, 4) AS avg_score_all,
        ROUND(AVG(rank) FILTER (WHERE action = 'taken')::numeric, 1) AS avg_rank_taken
      FROM v_new_1_shadow_signals
      WHERE timestamp >= ${SEVEN_DAYS_AGO_SQL}
      GROUP BY direction
      ORDER BY direction
    `);
  } catch {}
  const statsByDir: Record<string, any> = Object.fromEntries(
    statsRows.map((r: any) => [r.direction, r])
  );

  // ── Skip-reason breakdown ─────────────────────────────────────────────────
  let skipRows: any[] = [];
  try {
    skipRows = await prisma.$queryRawUnsafe(`
      SELECT action, COUNT(*)::int AS n
      FROM v_new_1_shadow_signals
      WHERE action != 'taken' AND timestamp >= ${SEVEN_DAYS_AGO_SQL}
      GROUP BY action
      ORDER BY n DESC
    `);
  } catch {}

  // ── CFGI regime distribution (7d) ─────────────────────────────────────────
  let cfgiRows: any[] = [];
  try {
    cfgiRows = await prisma.$queryRawUnsafe(`
      SELECT
        COUNT(*) FILTER (WHERE cfgi_value < 40)::int AS fear_n,
        COUNT(*) FILTER (WHERE cfgi_value >= 40 AND cfgi_value < 60)::int AS neutral_n,
        COUNT(*) FILTER (WHERE cfgi_value >= 60)::int AS greed_n,
        COUNT(DISTINCT timestamp)::int AS total_runs
      FROM v_new_1_shadow_signals
      WHERE timestamp >= ${SEVEN_DAYS_AGO_SQL}
    `);
  } catch {}
  const cfgiStat = cfgiRows[0] ?? null;
  const cfgiTotal = cfgiStat
    ? (Number(cfgiStat.fear_n) + Number(cfgiStat.neutral_n) + Number(cfgiStat.greed_n))
    : 0;

  // ── Top symbols by taken signal frequency (7d) ───────────────────────────
  let topSymbols: any[] = [];
  try {
    topSymbols = await prisma.$queryRawUnsafe(`
      SELECT
        symbol,
        COUNT(*) FILTER (WHERE action = 'taken')::int AS taken_n,
        COUNT(*) FILTER (WHERE action = 'taken' AND direction = 'long')::int AS long_n,
        COUNT(*) FILTER (WHERE action = 'taken' AND direction = 'short')::int AS short_n,
        ROUND(AVG(score_platt) FILTER (WHERE action = 'taken')::numeric, 3) AS avg_score,
        ROUND(AVG(rank) FILTER (WHERE action = 'taken')::numeric, 1) AS avg_rank
      FROM v_new_1_shadow_signals
      WHERE timestamp >= ${SEVEN_DAYS_AGO_SQL}
      GROUP BY symbol
      HAVING COUNT(*) FILTER (WHERE action = 'taken') > 0
      ORDER BY taken_n DESC, avg_score DESC
      LIMIT 20
    `);
  } catch {}

  // ── Latest signals table (last 7 days, paginated) ─────────────────────────
  const dirClause =
    dirFilter !== "all"
      ? `AND direction = '${dirFilter === "long" ? "long" : "short"}'`
      : "";

  let totalSignals = 0;
  let signals: any[] = [];
  try {
    const cr: any[] = await prisma.$queryRawUnsafe(`
      SELECT COUNT(*)::int AS n
      FROM v_new_1_shadow_signals
      WHERE timestamp >= ${SEVEN_DAYS_AGO_SQL} ${dirClause}
    `);
    totalSignals = cr[0]?.n || 0;

    signals = await prisma.$queryRawUnsafe(`
      SELECT id, timestamp, symbol, direction, score_raw, score_platt,
             rank, cfgi_value, action, market_close
      FROM v_new_1_shadow_signals
      WHERE timestamp >= ${SEVEN_DAYS_AGO_SQL} ${dirClause}
      ORDER BY timestamp DESC, score_platt DESC
      LIMIT ${PER_PAGE} OFFSET ${(page - 1) * PER_PAGE}
    `);
  } catch {}
  const totalPages = Math.max(1, Math.ceil(totalSignals / PER_PAGE));

  // ── Totals for KPIs ───────────────────────────────────────────────────────
  const totalTaken7d = statsRows.reduce(
    (acc: number, r: any) => acc + Number(r.taken_n || 0), 0
  );
  const longStats = statsByDir["long"];
  const shortStats = statsByDir["short"];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white">
            v_new_1 Shadow Mode{" "}
            <span className="text-sm font-normal text-yellow-400">
              OBSERVATION ONLY · no trading
            </span>
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Big-movers detection — long+short pair scored every 4h via cron.{" "}
            <span className="text-brand-400 font-semibold">No actual trades are placed.</span>{" "}
            Signals are logged to DB for review. CFGI gates: longs only when{" "}
            <code className="text-slate-300">CFGI &gt; 60</code>, shorts only when{" "}
            <code className="text-slate-300">CFGI &lt; 50</code>.
            Top-K policy: 13 longs, 14 shorts per run. Score thresholds: long ≥ 0.317, short ≥ 0.468.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Link
            href="/paper-v8-ml"
            className="text-xs text-slate-400 hover:text-white px-3 py-1.5 rounded bg-slate-800 hover:bg-slate-700"
          >
            ← Paper v8 ML
          </Link>
        </div>
      </div>

      {/* Live status banner */}
      <div className="card p-4">
        <div className="flex flex-wrap items-center gap-x-6 gap-y-3 text-xs">
          <StatusItem
            label="Last cron run"
            value={
              statusRow?.last_run
                ? relativeTime(new Date(statusRow.last_run))
                : "never"
            }
            tone={
              statusRow?.last_run
                ? minutesAgo(new Date(statusRow.last_run)) < 300
                  ? "green"
                  : minutesAgo(new Date(statusRow.last_run)) < 600
                  ? "yellow"
                  : "red"
                : "gray"
            }
          />
          <StatusItem
            label="Signals (24h)"
            value={String(statusRow?.signals_24h ?? 0)}
            tone={statusRow?.signals_24h > 0 ? "green" : "gray"}
          />
          <StatusItem
            label="Taken (24h)"
            value={String(statusRow?.taken_24h ?? 0)}
            tone={statusRow?.taken_24h > 0 ? "green" : "gray"}
          />
          <StatusItem
            label="Latest CFGI"
            value={
              statusRow?.latest_cfgi != null
                ? `${statusRow.latest_cfgi} — ${cfgiLabel(Number(statusRow.latest_cfgi))}`
                : "—"
            }
            tone={
              statusRow?.latest_cfgi != null
                ? Number(statusRow.latest_cfgi) > 60
                  ? "green"
                  : Number(statusRow.latest_cfgi) < 40
                  ? "red"
                  : "yellow"
                : "gray"
            }
          />
        </div>
        {!statusRow?.last_run && (
          <p className="text-xs text-slate-500 mt-3">
            No shadow signals in DB yet. The cron runs every 4h; the first batch
            will appear after the migration is deployed and the shadow scorer runs
            at least once.{" "}
            <code className="text-slate-300 font-mono">
              v_new_1_shadow_signals
            </code>{" "}
            table must exist on VPS Postgres first.
          </p>
        )}
      </div>

      {/* KPI strip */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <KPI
          label="Total Taken (7d)"
          value={`${totalTaken7d}`}
          color={totalTaken7d > 0 ? "brand" : "slate"}
          hint="across long + short"
        />
        <KPI
          label="Long Taken (7d)"
          value={`${longStats?.taken_n ?? 0}`}
          color="green"
          hint={
            longStats?.avg_score_taken != null
              ? `avg score ${Number(longStats.avg_score_taken).toFixed(3)}`
              : "no data"
          }
        />
        <KPI
          label="Short Taken (7d)"
          value={`${shortStats?.taken_n ?? 0}`}
          color="red"
          hint={
            shortStats?.avg_score_taken != null
              ? `avg score ${Number(shortStats.avg_score_taken).toFixed(3)}`
              : "no data"
          }
        />
        <KPI
          label="Cron Runs (7d)"
          value={`${cfgiStat?.total_runs ?? 0}`}
          color="slate"
          hint="distinct 4h timestamps"
        />
      </div>

      {/* Stats + CFGI distribution */}
      <div className="grid md:grid-cols-2 gap-4">
        {/* Direction stats */}
        <div className="card overflow-hidden p-0">
          <div className="px-4 py-3 border-b border-[var(--border)]">
            <h2 className="text-sm font-semibold text-slate-200">7-Day Stats by Direction</h2>
          </div>
          {statsRows.length === 0 ? (
            <div className="px-4 py-6 text-xs text-slate-500 text-center">No data yet.</div>
          ) : (
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-[var(--border)]">
                  <th className="px-3 py-2 text-left text-slate-500">Direction</th>
                  <th className="px-3 py-2 text-right text-slate-500">Taken</th>
                  <th className="px-3 py-2 text-right text-slate-500">Skipped</th>
                  <th className="px-3 py-2 text-right text-slate-500">Avg Score (taken)</th>
                  <th className="px-3 py-2 text-right text-slate-500">Avg Rank</th>
                </tr>
              </thead>
              <tbody>
                {statsRows.map((r: any) => (
                  <tr key={r.direction} className="border-b border-[var(--border)]">
                    <td className="px-3 py-1.5">
                      <span
                        className={`text-[10px] px-1.5 py-0.5 rounded font-mono ${
                          r.direction === "short"
                            ? "bg-red-900/50 text-red-400"
                            : "bg-green-900/50 text-green-400"
                        }`}
                      >
                        {r.direction.toUpperCase()}
                      </span>
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono text-slate-200">
                      {r.taken_n}
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono text-slate-500">
                      {r.skipped_n}
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono text-brand-400">
                      {r.avg_score_taken != null
                        ? Number(r.avg_score_taken).toFixed(3)
                        : "—"}
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono text-slate-400">
                      {r.avg_rank_taken != null
                        ? Number(r.avg_rank_taken).toFixed(1)
                        : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {/* CFGI regime distribution + skip reasons */}
        <div className="space-y-4">
          {/* CFGI regime */}
          <div className="card overflow-hidden p-0">
            <div className="px-4 py-3 border-b border-[var(--border)]">
              <h2 className="text-sm font-semibold text-slate-200">CFGI Regime (7d)</h2>
              <p className="text-xs text-slate-500 mt-0.5">
                Distribution of CFGI values across all scoring runs
              </p>
            </div>
            <div className="px-4 py-3 flex gap-6 text-xs">
              {cfgiTotal > 0 ? (
                <>
                  <div>
                    <p className="text-slate-500">Fear (&lt;40)</p>
                    <p className="font-mono text-red-400 text-base font-bold">
                      {cfgiStat ? pct(Number(cfgiStat.fear_n), cfgiTotal) : "—"}
                    </p>
                  </div>
                  <div>
                    <p className="text-slate-500">Neutral (40–60)</p>
                    <p className="font-mono text-yellow-400 text-base font-bold">
                      {cfgiStat ? pct(Number(cfgiStat.neutral_n), cfgiTotal) : "—"}
                    </p>
                  </div>
                  <div>
                    <p className="text-slate-500">Greed (&gt;60)</p>
                    <p className="font-mono text-green-400 text-base font-bold">
                      {cfgiStat ? pct(Number(cfgiStat.greed_n), cfgiTotal) : "—"}
                    </p>
                  </div>
                </>
              ) : (
                <p className="text-slate-500">No data yet.</p>
              )}
            </div>
          </div>

          {/* Skip reasons */}
          <div className="card overflow-hidden p-0">
            <div className="px-4 py-3 border-b border-[var(--border)]">
              <h2 className="text-sm font-semibold text-slate-200">Skip Reasons (7d)</h2>
            </div>
            {skipRows.length === 0 ? (
              <div className="px-4 py-4 text-xs text-slate-500">No data yet.</div>
            ) : (
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-[var(--border)]">
                    <th className="px-3 py-2 text-left text-slate-500">Reason</th>
                    <th className="px-3 py-2 text-right text-slate-500">Count</th>
                  </tr>
                </thead>
                <tbody>
                  {skipRows.map((r: any) => (
                    <tr key={r.action} className="border-b border-[var(--border)]">
                      <td className="px-3 py-1.5">
                        <span className="font-mono text-slate-400 text-[10px]">
                          {r.action}
                        </span>
                      </td>
                      <td className="px-3 py-1.5 text-right font-mono text-slate-300">
                        {r.n}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>

      {/* Top symbols */}
      {topSymbols.length > 0 && (
        <div className="card overflow-hidden p-0">
          <div className="px-4 py-3 border-b border-[var(--border)]">
            <h2 className="text-sm font-semibold text-slate-200">
              Top Symbols by Signal Frequency (7d — taken only)
            </h2>
          </div>
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-[var(--border)]">
                <th className="px-3 py-2 text-left text-slate-500">Symbol</th>
                <th className="px-3 py-2 text-right text-slate-500">Taken</th>
                <th className="px-3 py-2 text-right text-slate-500">Long</th>
                <th className="px-3 py-2 text-right text-slate-500">Short</th>
                <th className="px-3 py-2 text-right text-slate-500">Avg Score</th>
                <th className="px-3 py-2 text-right text-slate-500">Avg Rank</th>
              </tr>
            </thead>
            <tbody>
              {topSymbols.map((r: any) => (
                <tr
                  key={r.symbol}
                  className="border-b border-[var(--border)] hover:bg-slate-800/50"
                >
                  <td className="px-3 py-1.5 text-white font-medium">
                    {r.symbol.replace("USDT", "")}
                    <span className="text-slate-500 text-[10px] ml-1 font-normal">USDT</span>
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-brand-400">
                    {r.taken_n}
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-green-400">
                    {r.long_n > 0 ? r.long_n : "—"}
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-red-400">
                    {r.short_n > 0 ? r.short_n : "—"}
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-300">
                    {Number(r.avg_score).toFixed(3)}
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-400">
                    {Number(r.avg_rank).toFixed(1)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Latest signals (last 7 days) */}
      <div className="card overflow-hidden p-0">
        <div className="px-4 py-3 border-b border-[var(--border)] flex items-center justify-between flex-wrap gap-2">
          <div className="flex items-center gap-3">
            <h2 className="text-sm font-semibold text-slate-200">
              Latest Signals — last 7 days ({totalSignals})
            </h2>
            <div className="flex gap-1 text-xs">
              {["all", "long", "short"].map((d) => (
                <Link
                  key={d}
                  href={`?direction=${d}`}
                  className={`px-2 py-0.5 rounded ${
                    dirFilter === d
                      ? "bg-brand-600/20 text-brand-400"
                      : "text-slate-500 hover:text-slate-300"
                  }`}
                >
                  {d}
                </Link>
              ))}
            </div>
          </div>
          <div className="flex items-center gap-2 text-xs">
            {page > 1 && (
              <Link
                href={`?page=${page - 1}${dirFilter !== "all" ? `&direction=${dirFilter}` : ""}`}
                className="px-2 py-1 rounded bg-slate-800 text-slate-300 hover:bg-slate-700"
              >
                Prev
              </Link>
            )}
            <span className="text-slate-500">{page}/{totalPages}</span>
            {page < totalPages && (
              <Link
                href={`?page=${page + 1}${dirFilter !== "all" ? `&direction=${dirFilter}` : ""}`}
                className="px-2 py-1 rounded bg-slate-800 text-slate-300 hover:bg-slate-700"
              >
                Next
              </Link>
            )}
          </div>
        </div>

        {signals.length === 0 ? (
          <div className="px-4 py-8 text-xs text-slate-500 text-center">
            No signals in the last 7 days. Waiting for first cron run after migration is deployed.
          </div>
        ) : (
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-[var(--border)]">
                <th className="px-3 py-2 text-left text-slate-500">Time</th>
                <th className="px-3 py-2 text-left text-slate-500">Symbol</th>
                <th className="px-3 py-2 text-center text-slate-500">Dir</th>
                <th className="px-3 py-2 text-right text-slate-500">Platt Score</th>
                <th className="px-3 py-2 text-right text-slate-500">Rank</th>
                <th className="px-3 py-2 text-right text-slate-500">CFGI</th>
                <th className="px-3 py-2 text-right text-slate-500">Market $</th>
                <th className="px-3 py-2 text-center text-slate-500">Action</th>
              </tr>
            </thead>
            <tbody>
              {signals.map((s: any) => {
                const score = Number(s.score_platt);
                const isTaken = s.action === "taken";
                return (
                  <tr
                    key={s.id}
                    className={`border-b border-[var(--border)] ${
                      isTaken ? "hover:bg-green-950/20" : "hover:bg-slate-800/40"
                    }`}
                  >
                    <td className="px-3 py-1.5 text-slate-400 font-mono whitespace-nowrap">
                      {relativeTime(new Date(s.timestamp))}
                    </td>
                    <td className="px-3 py-1.5 text-white font-medium">
                      {String(s.symbol).replace("USDT", "")}
                      <span className="text-slate-500 text-[10px] ml-0.5">USDT</span>
                    </td>
                    <td className="px-3 py-1.5 text-center">
                      <span
                        className={`text-[10px] px-1 rounded ${
                          s.direction === "short"
                            ? "text-red-400"
                            : "text-green-400"
                        }`}
                      >
                        {s.direction?.toUpperCase()}
                      </span>
                    </td>
                    <td
                      className={`px-3 py-1.5 text-right font-mono ${
                        score >= 0.6
                          ? "text-green-400"
                          : score >= 0.45
                          ? "text-yellow-400"
                          : "text-slate-400"
                      }`}
                    >
                      {score.toFixed(3)}
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono text-slate-400">
                      #{Number(s.rank)}
                    </td>
                    <td
                      className={`px-3 py-1.5 text-right font-mono ${
                        Number(s.cfgi_value) > 60
                          ? "text-green-400"
                          : Number(s.cfgi_value) < 40
                          ? "text-red-400"
                          : "text-yellow-400"
                      }`}
                    >
                      {Number(s.cfgi_value)}
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono text-slate-300">
                      {Number(s.market_close) > 0
                        ? `$${Number(s.market_close).toPrecision(4)}`
                        : "—"}
                    </td>
                    <td className="px-3 py-1.5 text-center">
                      <ActionBadge action={String(s.action)} />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      {/* Footer note */}
      <div className="card p-4 text-xs text-slate-400 space-y-1">
        <p>
          <span className="text-slate-300 font-semibold">Score color legend:</span>{" "}
          <span className="text-green-400 font-mono">≥ 0.60</span> high confidence,{" "}
          <span className="text-yellow-400 font-mono">0.45–0.60</span> medium,{" "}
          <span className="text-slate-400 font-mono">&lt; 0.45</span> below threshold.
        </p>
        <p>
          <span className="text-slate-300 font-semibold">CFGI gates:</span>{" "}
          <span className="text-green-400 font-mono">longs</span> only fire when CFGI &gt; 60 (greed),{" "}
          <span className="text-red-400 font-mono">shorts</span> only fire when CFGI &lt; 50 (fear/neutral).
          <span className="text-slate-500"> Signals with </span>
          <code className="text-slate-300">skipped_cfgi_*</code>
          <span className="text-slate-500"> were blocked by this gate.</span>
        </p>
        <p>
          This page is read-only. No trades are placed. Shadow scoring runs every 4h via cron on the
          VPS and writes to <code className="text-slate-300">v_new_1_shadow_signals</code> in Postgres.
          Promote to paper trading after 2–4 weeks of satisfactory shadow performance.
        </p>
      </div>
    </div>
  );
}

// ── Helper components ─────────────────────────────────────────────────────────

function ActionBadge({ action }: { action: string }) {
  if (action === "taken") {
    return (
      <span className="text-[10px] px-1.5 py-0.5 rounded bg-green-900/50 text-green-400 font-mono">
        taken
      </span>
    );
  }
  if (action.startsWith("skipped_cfgi")) {
    return (
      <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-700 text-slate-400 font-mono">
        cfgi
      </span>
    );
  }
  if (action === "skipped_score") {
    return (
      <span className="text-[10px] px-1.5 py-0.5 rounded bg-yellow-900/30 text-yellow-600 font-mono">
        score
      </span>
    );
  }
  if (action === "skipped_topk") {
    return (
      <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-500 font-mono">
        topk
      </span>
    );
  }
  return (
    <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-700 text-slate-400 font-mono">
      {action}
    </span>
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

function StatusItem({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone: "green" | "yellow" | "red" | "gray";
}) {
  const dot = {
    green: "bg-green-400",
    yellow: "bg-yellow-400",
    red: "bg-red-400",
    gray: "bg-slate-500",
  }[tone];
  return (
    <div className="flex items-center gap-1.5">
      <span className={`inline-block h-2 w-2 rounded-full ${dot}`} />
      <span className="text-slate-500">{label}:</span>
      <span className="font-mono text-slate-200">{value}</span>
    </div>
  );
}

// ── Pure helpers ──────────────────────────────────────────────────────────────

function minutesAgo(d: Date): number {
  return (Date.now() - d.getTime()) / 60000;
}

function relativeTime(d: Date): string {
  const mins = minutesAgo(d);
  if (mins < 1) return "just now";
  if (mins < 60) return `${Math.floor(mins)}m ago`;
  const hrs = mins / 60;
  if (hrs < 24) return `${Math.floor(hrs)}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

function cfgiLabel(v: number): string {
  if (v < 25) return "extreme fear";
  if (v < 40) return "fear";
  if (v < 60) return "neutral";
  if (v < 75) return "greed";
  return "extreme greed";
}

function pct(n: number, total: number): string {
  if (total === 0) return "—";
  return `${((n / total) * 100).toFixed(0)}%`;
}
