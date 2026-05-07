import { prisma } from "@/lib/prisma";

export const dynamic = "force-dynamic";
export const revalidate = 60;

const INIT_BALANCE = 20.0;

type Kpi = {
  equity: number;
  pnlUsd: number;
  pnlPct: number;
  openCount: number;
  realizedTrades: number;
  winRate: number | null;
  profitFactor: number | null;
  avgWinPct: number | null;
  avgLossPct: number | null;
};

function pct(n: number | null, dp = 2): string {
  if (n === null || !isFinite(n)) return "—";
  return `${n >= 0 ? "+" : ""}${n.toFixed(dp)}%`;
}

function usd(n: number, dp = 2): string {
  return `${n >= 0 ? "" : "-"}$${Math.abs(n).toFixed(dp)}`;
}

function classifyPnl(p: number | null): string {
  if (p === null) return "text-slate-400";
  return p > 0 ? "text-green-400" : p < 0 ? "text-red-400" : "text-slate-300";
}

export default async function PaperSoloPage() {
  // ── KPIs ────────────────────────────────────────────────────────────────
  const kpi: Kpi = {
    equity: INIT_BALANCE,
    pnlUsd: 0,
    pnlPct: 0,
    openCount: 0,
    realizedTrades: 0,
    winRate: null,
    profitFactor: null,
    avgWinPct: null,
    avgLossPct: null,
  };
  try {
    const eqRow: any[] = await prisma.$queryRawUnsafe(`
      SELECT
        COALESCE(SUM(pnl_usd), 0)::float AS realized_pnl,
        COUNT(*)::int AS n_trades,
        COUNT(*) FILTER (WHERE pnl_pct > 0)::int AS n_wins,
        COUNT(*) FILTER (WHERE pnl_pct <= 0)::int AS n_losses,
        AVG(pnl_pct) FILTER (WHERE pnl_pct > 0)::float AS avg_win,
        AVG(pnl_pct) FILTER (WHERE pnl_pct <= 0)::float AS avg_loss,
        COALESCE(SUM(pnl_usd) FILTER (WHERE pnl_usd > 0), 0)::float AS gross_win,
        COALESCE(SUM(pnl_usd) FILTER (WHERE pnl_usd <= 0), 0)::float AS gross_loss
      FROM v_spot_paper_trades
    `);
    const opRow: any[] = await prisma.$queryRawUnsafe(`
      SELECT COUNT(*)::int AS n FROM v_spot_paper_positions WHERE status = 'open'
    `);
    const r = eqRow[0] ?? {};
    kpi.realizedTrades = Number(r.n_trades || 0);
    kpi.pnlUsd = Number(r.realized_pnl || 0);
    kpi.equity = INIT_BALANCE + kpi.pnlUsd;
    kpi.pnlPct = (kpi.pnlUsd / INIT_BALANCE) * 100;
    if (kpi.realizedTrades > 0) {
      kpi.winRate = (Number(r.n_wins) / kpi.realizedTrades) * 100;
      kpi.avgWinPct = r.avg_win !== null ? Number(r.avg_win) : null;
      kpi.avgLossPct = r.avg_loss !== null ? Number(r.avg_loss) : null;
      const grossLoss = Math.abs(Number(r.gross_loss || 0));
      kpi.profitFactor = grossLoss > 0 ? Number(r.gross_win) / grossLoss : null;
    }
    kpi.openCount = Number(opRow[0]?.n || 0);
  } catch {}

  // ── Open positions (max 2) ──────────────────────────────────────────────
  let openPositions: any[] = [];
  try {
    openPositions = await prisma.$queryRawUnsafe(`
      SELECT p.id, p.symbol, p.side, p.is_meme, p.tier, p.entry_time, p.entry_price,
             p.notional_usd, p.position_size_pct, p.bgm_score,
             p.running_extreme, p.profit_lock_active, p.lstm_regime,
             EXTRACT(EPOCH FROM (NOW() - p.entry_time)) / 86400.0 AS days_held
      FROM v_spot_paper_positions p
      WHERE p.status = 'open'
      ORDER BY p.entry_time DESC
    `);
  } catch {}

  // ── Last 25 closed trades ──────────────────────────────────────────────
  let closedTrades: any[] = [];
  try {
    closedTrades = await prisma.$queryRawUnsafe(`
      SELECT id, symbol, side, is_meme, tier, entry_time, exit_time,
             entry_price, exit_price, bars_held, exit_reason,
             pnl_pct, pnl_usd, bgm_score, position_size_pct
      FROM v_spot_paper_trades
      ORDER BY exit_time DESC
      LIMIT 25
    `);
  } catch {}

  // ── Eligible signals (long, bgm>=0.65, last 24h) ───────────────────────
  let eligibleSignals: any[] = [];
  try {
    eligibleSignals = await prisma.$queryRawUnsafe(`
      SELECT s.id, s.signal_time, s.symbol, s.side, s.tier, s.is_meme, s.bgm_score,
             EXISTS (
               SELECT 1 FROM v_spot_paper_positions p WHERE p.signal_id = s.id
             ) AS taken_by_spot
      FROM v_new_2_signals s
      WHERE s.side = 'long'
        AND s.bgm_score >= 0.65
        AND s.signal_time >= NOW() - INTERVAL '24 hours'
      ORDER BY s.signal_time DESC, s.bgm_score DESC
      LIMIT 15
    `);
  } catch {}

  return (
    <main className="min-h-screen bg-[var(--bg)] text-slate-200">
      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Header */}
        <header className="mb-8">
          <h1 className="text-2xl font-bold text-slate-100">Paper — Solo (Spot)</h1>
          <p className="text-xs text-slate-500 mt-1">
            1–2 position spot portfolio. Long-only, no leverage, 50% equity per slot, BGM ≥ 0.65.
            Cross-OOS backtest (thr_select + cal + holdout): +6077% vs BTC HODL −37%.
          </p>
          <p className="text-[10px] text-slate-600 mt-1">
            Reads <span className="font-mono">v_new_2_signals</span> read-only — independent of
            the main futures executor. Same scanner, stricter filter (long + BGM ≥ 0.65).
          </p>
        </header>

        {/* KPI strip */}
        <section className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          <div className="card p-4">
            <div className="text-[11px] uppercase tracking-wide text-slate-500">Equity</div>
            <div className={`mt-1 text-2xl font-mono font-bold ${classifyPnl(kpi.pnlUsd)}`}>
              {usd(kpi.equity)}
            </div>
            <div className={`text-xs font-mono mt-0.5 ${classifyPnl(kpi.pnlUsd)}`}>
              {pct(kpi.pnlPct)} ({usd(kpi.pnlUsd)})
            </div>
          </div>
          <div className="card p-4">
            <div className="text-[11px] uppercase tracking-wide text-slate-500">Open / Realized</div>
            <div className="mt-1 text-2xl font-mono font-bold text-slate-100">
              {kpi.openCount}<span className="text-slate-500 text-base">/2</span>
              <span className="text-slate-500 text-base"> · {kpi.realizedTrades}</span>
            </div>
            <div className="text-xs text-slate-500 mt-0.5">slot used / closed trades</div>
          </div>
          <div className="card p-4">
            <div className="text-[11px] uppercase tracking-wide text-slate-500">Win Rate</div>
            <div className="mt-1 text-2xl font-mono font-bold text-slate-100">
              {kpi.winRate !== null ? `${kpi.winRate.toFixed(1)}%` : "—"}
            </div>
            <div className="text-xs text-slate-500 mt-0.5">
              {kpi.avgWinPct !== null && kpi.avgLossPct !== null
                ? `+${kpi.avgWinPct.toFixed(1)}% / ${kpi.avgLossPct.toFixed(1)}%`
                : "no trades yet"}
            </div>
          </div>
          <div className="card p-4">
            <div className="text-[11px] uppercase tracking-wide text-slate-500">Profit Factor</div>
            <div className="mt-1 text-2xl font-mono font-bold text-slate-100">
              {kpi.profitFactor !== null ? kpi.profitFactor.toFixed(2) : "—"}
            </div>
            <div className="text-xs text-slate-500 mt-0.5">gross W / gross L</div>
          </div>
        </section>

        {/* Open positions */}
        <section className="mb-8">
          <h2 className="text-sm font-semibold text-slate-200 mb-3">
            Open positions <span className="text-slate-500">({openPositions.length}/2)</span>
          </h2>
          <div className="card overflow-hidden p-0">
            {openPositions.length === 0 ? (
              <div className="px-4 py-6 text-xs text-slate-500 text-center">
                No open positions — waiting for the next eligible long signal (BGM ≥ 0.65).
              </div>
            ) : (
              <table className="w-full text-xs">
                <thead className="bg-[var(--bg-subtle)]">
                  <tr className="border-b border-[var(--border)] text-slate-500">
                    <th className="px-3 py-2 text-left">Symbol</th>
                    <th className="px-3 py-2 text-left">Tier</th>
                    <th className="px-3 py-2 text-right">Entry</th>
                    <th className="px-3 py-2 text-right">Notional</th>
                    <th className="px-3 py-2 text-right">Size %</th>
                    <th className="px-3 py-2 text-right">BGM</th>
                    <th className="px-3 py-2 text-left">Regime</th>
                    <th className="px-3 py-2 text-right">Held</th>
                    <th className="px-3 py-2 text-right">PL Lock</th>
                  </tr>
                </thead>
                <tbody>
                  {openPositions.map((p: any) => (
                    <tr key={p.id} className="border-b border-[var(--border)] hover:bg-[var(--bg-subtle)]">
                      <td className="px-3 py-2 font-mono text-slate-200">
                        {p.symbol}
                        {p.is_meme && <span className="ml-1 text-[9px] text-pink-400">MEME</span>}
                      </td>
                      <td className="px-3 py-2 text-slate-400 font-mono text-[10px]">{p.tier}</td>
                      <td className="px-3 py-2 text-right font-mono text-slate-300">
                        {Number(p.entry_price).toFixed(6)}
                      </td>
                      <td className="px-3 py-2 text-right font-mono text-slate-300">
                        ${Number(p.notional_usd).toFixed(2)}
                      </td>
                      <td className="px-3 py-2 text-right font-mono text-slate-400">
                        {Number(p.position_size_pct).toFixed(0)}%
                      </td>
                      <td className="px-3 py-2 text-right font-mono text-brand-400">
                        {p.bgm_score !== null ? Number(p.bgm_score).toFixed(3) : "—"}
                      </td>
                      <td className="px-3 py-2 text-slate-400 font-mono text-[10px]">
                        {p.lstm_regime || "—"}
                      </td>
                      <td className="px-3 py-2 text-right font-mono text-slate-400">
                        {Number(p.days_held).toFixed(1)}d
                      </td>
                      <td className="px-3 py-2 text-right font-mono">
                        {p.profit_lock_active ? (
                          <span className="text-yellow-400">ARMED</span>
                        ) : (
                          <span className="text-slate-600">—</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </section>

        {/* Closed trades */}
        <section className="mb-8">
          <h2 className="text-sm font-semibold text-slate-200 mb-3">
            Last {Math.min(25, closedTrades.length)} closed
          </h2>
          <div className="card overflow-hidden p-0">
            {closedTrades.length === 0 ? (
              <div className="px-4 py-6 text-xs text-slate-500 text-center">
                No closed trades yet.
              </div>
            ) : (
              <table className="w-full text-xs">
                <thead className="bg-[var(--bg-subtle)]">
                  <tr className="border-b border-[var(--border)] text-slate-500">
                    <th className="px-3 py-2 text-left">Exit Time</th>
                    <th className="px-3 py-2 text-left">Symbol</th>
                    <th className="px-3 py-2 text-right">PnL %</th>
                    <th className="px-3 py-2 text-right">PnL $</th>
                    <th className="px-3 py-2 text-right">Bars</th>
                    <th className="px-3 py-2 text-right">BGM</th>
                    <th className="px-3 py-2 text-left">Reason</th>
                  </tr>
                </thead>
                <tbody>
                  {closedTrades.map((t: any) => (
                    <tr key={t.id} className="border-b border-[var(--border)] hover:bg-[var(--bg-subtle)]">
                      <td className="px-3 py-2 text-slate-400 font-mono text-[10px]">
                        {new Date(t.exit_time).toISOString().slice(0, 16).replace("T", " ")}
                      </td>
                      <td className="px-3 py-2 font-mono text-slate-200">
                        {t.symbol}
                        {t.is_meme && <span className="ml-1 text-[9px] text-pink-400">MEME</span>}
                      </td>
                      <td className={`px-3 py-2 text-right font-mono font-semibold ${classifyPnl(Number(t.pnl_pct))}`}>
                        {pct(Number(t.pnl_pct))}
                      </td>
                      <td className={`px-3 py-2 text-right font-mono ${classifyPnl(Number(t.pnl_usd))}`}>
                        {usd(Number(t.pnl_usd))}
                      </td>
                      <td className="px-3 py-2 text-right font-mono text-slate-400">{t.bars_held}</td>
                      <td className="px-3 py-2 text-right font-mono text-brand-400">
                        {t.bgm_score !== null ? Number(t.bgm_score).toFixed(3) : "—"}
                      </td>
                      <td className="px-3 py-2 text-slate-400 font-mono text-[10px]">{t.exit_reason}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </section>

        {/* Eligible signals (decision context) */}
        <section className="mb-8">
          <h2 className="text-sm font-semibold text-slate-200 mb-3">
            Eligible signals (24h) <span className="text-slate-500">— long, BGM ≥ 0.65</span>
          </h2>
          <div className="card overflow-hidden p-0">
            {eligibleSignals.length === 0 ? (
              <div className="px-4 py-6 text-xs text-slate-500 text-center">
                No qualifying signals fired in the last 24h.
              </div>
            ) : (
              <table className="w-full text-xs">
                <thead className="bg-[var(--bg-subtle)]">
                  <tr className="border-b border-[var(--border)] text-slate-500">
                    <th className="px-3 py-2 text-left">Time</th>
                    <th className="px-3 py-2 text-left">Symbol</th>
                    <th className="px-3 py-2 text-left">Tier</th>
                    <th className="px-3 py-2 text-right">BGM</th>
                    <th className="px-3 py-2 text-center">Spot</th>
                  </tr>
                </thead>
                <tbody>
                  {eligibleSignals.map((s: any) => (
                    <tr key={s.id} className="border-b border-[var(--border)]">
                      <td className="px-3 py-2 text-slate-400 font-mono text-[10px]">
                        {new Date(s.signal_time).toISOString().slice(0, 16).replace("T", " ")}
                      </td>
                      <td className="px-3 py-2 font-mono text-slate-200">
                        {s.symbol}
                        {s.is_meme && <span className="ml-1 text-[9px] text-pink-400">MEME</span>}
                      </td>
                      <td className="px-3 py-2 text-slate-400 font-mono text-[10px]">{s.tier}</td>
                      <td className="px-3 py-2 text-right font-mono text-brand-400">
                        {Number(s.bgm_score).toFixed(3)}
                      </td>
                      <td className="px-3 py-2 text-center">
                        {s.taken_by_spot ? (
                          <span className="text-[10px] text-green-400">TAKEN</span>
                        ) : (
                          <span className="text-[10px] text-slate-500">queued</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </section>

        <footer className="text-[10px] text-slate-600 text-center pt-6">
          spot paper · cap=2 · 50% equity/slot · no leverage · BGM ≥ 0.65 · ATR-trail exits ·
          v_spot_paper_executor every 5min on VPS.
        </footer>
      </div>
    </main>
  );
}
