import { prisma } from "@/lib/prisma";

type Tone = "green" | "yellow" | "red" | "gray";

async function fetchOne(sql: string): Promise<any[]> {
  try {
    return (await prisma.$queryRawUnsafe(sql)) as any[];
  } catch {
    return [];
  }
}

async function fetchHeartbeats() {
  const [ingest, btc, sigShort, sigLong, paper] = await Promise.all([
    fetchOne(`SELECT MAX(timestamp) AS ts FROM asset_prices_15m`),
    fetchOne(`
      WITH recent AS (
        SELECT timestamp, close
        FROM asset_prices_15m WHERE asset = 'BTCUSDT'
        ORDER BY timestamp DESC LIMIT 20
      )
      SELECT
        (SELECT close FROM recent ORDER BY timestamp DESC LIMIT 1) AS price,
        AVG(close) AS sma20
      FROM recent
    `),
    fetchOne(`SELECT MAX(signal_time) AS ts FROM scanner_signals_v2`),
    fetchOne(`SELECT MAX(signal_time) AS ts FROM scanner_signals_long_v2`),
    fetchOne(`SELECT MAX(entry_time) AS ts FROM paper_trades`),
  ]);
  return {
    ingestTs: ingest[0]?.ts ?? null,
    btcPrice: btc[0]?.price != null ? Number(btc[0].price) : null,
    btcSma20: btc[0]?.sma20 != null ? Number(btc[0].sma20) : null,
    lastShortSignal: sigShort[0]?.ts ?? null,
    lastLongSignal: sigLong[0]?.ts ?? null,
    lastPaperTrade: paper[0]?.ts ?? null,
  };
}

function ago(ts: Date | string | null): { label: string; tone: Tone } {
  if (!ts) return { label: "never", tone: "gray" };
  const mins = (Date.now() - new Date(ts).getTime()) / 60000;
  if (mins < 1) return { label: "just now", tone: "green" };
  if (mins < 60) return { label: `${Math.floor(mins)}m ago`, tone: mins < 20 ? "green" : "yellow" };
  const hrs = mins / 60;
  if (hrs < 24) return { label: `${Math.floor(hrs)}h ago`, tone: "yellow" };
  return { label: `${Math.floor(hrs / 24)}d ago`, tone: "red" };
}

export async function SystemStatus() {
  const s = await fetchHeartbeats();

  // Ingest tone: fresh = green, stale > 20 min = red
  const ingestMins = s.ingestTs ? (Date.now() - new Date(s.ingestTs).getTime()) / 60000 : Infinity;
  const ingestLabel = ago(s.ingestTs).label;
  const ingestTone: Tone = ingestMins > 20 ? "red" : "green";

  const btcBullish = s.btcPrice != null && s.btcSma20 != null && s.btcPrice > s.btcSma20;
  const btcPct = s.btcPrice && s.btcSma20 ? (s.btcPrice / s.btcSma20 - 1) * 100 : null;

  const lastShort = ago(s.lastShortSignal);
  const lastLong = ago(s.lastLongSignal);
  const lastPaper = ago(s.lastPaperTrade);

  const allQuiet = !s.lastShortSignal && !s.lastLongSignal && !s.lastPaperTrade;

  return (
    <div className="card p-3">
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-xs">
        <Item tone={ingestTone} label="Ingest" value={ingestLabel} />
        <Item
          tone={btcBullish ? "green" : "red"}
          label="BTC regime"
          value={
            s.btcPrice != null && s.btcSma20 != null ? (
              <>
                {btcBullish ? "bullish" : "bearish"}
                {btcPct != null && (
                  <span className={btcBullish ? "text-green-400" : "text-red-400"}>
                    {" "}
                    ({btcPct > 0 ? "+" : ""}
                    {btcPct.toFixed(2)}% vs SMA20)
                  </span>
                )}
              </>
            ) : (
              "—"
            )
          }
        />
        <Item tone={lastShort.tone} label="Last SHORT signal" value={lastShort.label} />
        <Item tone={lastLong.tone} label="Last LONG signal" value={lastLong.label} />
        <Item tone={lastPaper.tone} label="Last paper trade" value={lastPaper.label} />
      </div>
      {allQuiet && ingestTone === "green" && (
        <p className="text-[10px] text-slate-500 mt-2">
          Workers are running — fresh candles are landing in the DB. Zero signals means the market hasn&rsquo;t produced a setup yet;
          {btcBullish ? " the SHORT scanner auto-gates while BTC > SMA20." : " the LONG scanner waits for a vol-spike + pullback + reclaim sequence."}
        </p>
      )}
      {ingestTone === "red" && (
        <p className="text-[10px] text-red-400 mt-2">
          Ingest appears stalled ({ingestLabel}). Check the <code className="font-mono">ingest</code> container on the VPS.
        </p>
      )}
    </div>
  );
}

function Item({ tone, label, value }: { tone: Tone; label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-center gap-1.5">
      <Dot tone={tone} />
      <span className="text-slate-500">{label}:</span>
      <span className="font-mono text-slate-200">{value}</span>
    </div>
  );
}

function Dot({ tone }: { tone: Tone }) {
  const cls = {
    green: "bg-green-400",
    yellow: "bg-yellow-400",
    red: "bg-red-400",
    gray: "bg-slate-500",
  }[tone];
  return <span className={`inline-block h-2 w-2 rounded-full ${cls}`} />;
}
