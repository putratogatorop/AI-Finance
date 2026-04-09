import { prisma } from "@/lib/prisma";
import ScannerTable from "@/components/scanner/scanner-table";

export default async function ScannerPage() {
  // Get recent signals (last 7 days)
  const signals = await prisma.scannerSignal.findMany({
    where: {
      signal_time: {
        gte: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
      },
    },
    orderBy: { signal_time: "desc" },
    take: 100,
  });

  // Get summary stats
  const totalSignals = await prisma.scannerSignal.count();
  const activeSignals = await prisma.scannerSignal.count({
    where: { status: "active" },
  });
  const tradedSignals = await prisma.scannerSignal.count({
    where: { status: "traded" },
  });

  // Win rate from traded signals
  const wins = await prisma.scannerSignal.count({
    where: { status: "traded", pnl_pct: { gt: 0 } },
  });
  const winRate = tradedSignals > 0 ? wins / tradedSignals : 0;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Momentum Scanner</h1>
        <p className="text-sm text-slate-400 mt-1">
          ML-filtered breakout signals across 200+ coins
        </p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="card p-4">
          <p className="text-xs text-slate-400 uppercase tracking-wide">Total Signals</p>
          <p className="text-2xl font-bold text-white mt-1">{totalSignals.toLocaleString()}</p>
        </div>
        <div className="card p-4">
          <p className="text-xs text-slate-400 uppercase tracking-wide">Active</p>
          <p className="text-2xl font-bold text-signal-green mt-1">{activeSignals}</p>
        </div>
        <div className="card p-4">
          <p className="text-xs text-slate-400 uppercase tracking-wide">Traded</p>
          <p className="text-2xl font-bold text-white mt-1">{tradedSignals}</p>
        </div>
        <div className="card p-4">
          <p className="text-xs text-slate-400 uppercase tracking-wide">Win Rate</p>
          <p className="text-2xl font-bold text-signal-green mt-1">
            {(winRate * 100).toFixed(1)}%
          </p>
        </div>
      </div>

      {/* Signals Table */}
      <div className="card">
        <div className="p-4 border-b border-slate-700">
          <h2 className="text-lg font-semibold text-white">Recent Signals</h2>
        </div>
        <ScannerTable signals={signals} />
      </div>
    </div>
  );
}
