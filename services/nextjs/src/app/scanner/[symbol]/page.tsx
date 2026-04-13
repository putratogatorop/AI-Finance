import { prisma } from "@/lib/prisma";
import TokenChart from "@/components/scanner/token-chart";

export default async function TokenDetailPage({
  params,
}: {
  params: { symbol: string };
}) {
  const symbol = params.symbol.toUpperCase();
  const symbolUsdt = `${symbol}USDT`;

  // Get recent 15m price data (last 48h = 192 bars)
  const priceData = await prisma.assetPrice15m.findMany({
    where: {
      asset: symbol,
      timestamp: {
        gte: new Date(Date.now() - 48 * 60 * 60 * 1000),
      },
    },
    orderBy: { timestamp: "asc" },
  });

  // Get signals for this coin
  const signals = await prisma.scannerSignal.findMany({
    where: { symbol: symbolUsdt },
    orderBy: { signal_time: "desc" },
    take: 20,
  });

  // Stats for this coin
  const totalTrades = signals.filter((s) => s.status === "traded").length;
  const wins = signals.filter((s) => s.pnl_pct != null && s.pnl_pct > 0).length;
  const winRate = totalTrades > 0 ? wins / totalTrades : 0;
  const avgPnl = totalTrades > 0
    ? signals.filter((s) => s.pnl_pct != null).reduce((sum, s) => sum + (s.pnl_pct || 0), 0) / totalTrades
    : 0;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">{symbol}/USDT</h1>
        <div className="flex gap-4 mt-2 text-sm">
          <span className="text-slate-400">
            Signals: <span className="text-white font-semibold">{signals.length}</span>
          </span>
          <span className="text-slate-400">
            Traded: <span className="text-white font-semibold">{totalTrades}</span>
          </span>
          <span className="text-slate-400">
            Win Rate: <span className="text-signal-green font-semibold">{(winRate * 100).toFixed(0)}%</span>
          </span>
          <span className="text-slate-400">
            Avg PnL: <span className={avgPnl > 0 ? "text-signal-green font-semibold" : "text-signal-red font-semibold"}>
              {(avgPnl * 100).toFixed(2)}%
            </span>
          </span>
        </div>
      </div>

      {/* TradingView Chart Embed */}
      <div className="card p-0 overflow-hidden" style={{ height: "500px" }}>
        <TokenChart symbol={symbolUsdt} />
      </div>

      {/* Price Data from DB */}
      {priceData.length > 0 && (
        <div className="card p-4">
          <h2 className="text-lg font-semibold text-white mb-2">Last 48h (from DB)</h2>
          <div className="grid grid-cols-4 gap-4 text-sm">
            <div>
              <p className="text-slate-400">Current</p>
              <p className="text-white font-mono">${priceData[priceData.length - 1].close.toFixed(6)}</p>
            </div>
            <div>
              <p className="text-slate-400">48h High</p>
              <p className="text-signal-green font-mono">${Math.max(...priceData.map(p => p.high)).toFixed(6)}</p>
            </div>
            <div>
              <p className="text-slate-400">48h Low</p>
              <p className="text-signal-red font-mono">${Math.min(...priceData.map(p => p.low)).toFixed(6)}</p>
            </div>
            <div>
              <p className="text-slate-400">48h Volume</p>
              <p className="text-white font-mono">{priceData.reduce((s, p) => s + p.volume, 0).toLocaleString()}</p>
            </div>
          </div>
        </div>
      )}

      {/* Signal History */}
      <div className="card">
        <div className="p-4 border-b border-slate-700">
          <h2 className="text-lg font-semibold text-white">Signal History</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-slate-700">
                <th className="table-header">Time</th>
                <th className="table-header">Direction</th>
                <th className="table-header">ML Prob</th>
                <th className="table-header">Vol Ratio</th>
                <th className="table-header">Status</th>
                <th className="table-header">PnL</th>
                <th className="table-header">Exit</th>
              </tr>
            </thead>
            <tbody>
              {signals.map((s) => (
                <tr key={s.id} className="border-b border-slate-800">
                  <td className="table-cell text-slate-400 font-mono text-xs">
                    {new Date(s.signal_time).toLocaleString("en-GB", {
                      month: "short", day: "2-digit",
                      hour: "2-digit", minute: "2-digit",
                    })}
                  </td>
                  <td className="table-cell">
                    <span className={s.direction === 1 ? "badge-green text-xs px-2 py-0.5 rounded" : "badge-red text-xs px-2 py-0.5 rounded"}>
                      {s.direction === 1 ? "LONG" : "SHORT"}
                    </span>
                  </td>
                  <td className="table-cell font-mono">{(s.ml_prob * 100).toFixed(0)}%</td>
                  <td className="table-cell font-mono">{s.vol_ratio.toFixed(1)}x</td>
                  <td className="table-cell text-slate-400">{s.status}</td>
                  <td className="table-cell font-mono">
                    {s.pnl_pct != null ? (
                      <span className={s.pnl_pct > 0 ? "text-signal-green" : "text-signal-red"}>
                        {(s.pnl_pct * 100).toFixed(2)}%
                      </span>
                    ) : "--"}
                  </td>
                  <td className="table-cell text-slate-500 text-xs">{s.exit_reason || "--"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
