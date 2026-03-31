import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";

export async function GET(
  _request: NextRequest,
  { params }: { params: { symbol: string } }
) {
  try {
    const symbol = params.symbol.toUpperCase();

    const [
      priceHistory,
      fundamentals,
      recentSignals,
      tradeHistory,
      latestPrice,
    ] = await Promise.all([
      prisma.assetPriceDaily.findMany({
        where: { asset: symbol },
        orderBy: { date: "desc" },
        take: 90,
      }),
      prisma.assetFundamental.findFirst({
        where: { asset: symbol },
        orderBy: { fetched_at: "desc" },
      }),
      prisma.signal.findMany({
        where: { asset: symbol },
        orderBy: { created_at: "desc" },
        take: 20,
      }),
      prisma.portfolio.findMany({
        where: { asset: symbol },
        orderBy: { opened_at: "desc" },
      }),
      prisma.assetPriceHourly.findFirst({
        where: { asset: symbol },
        orderBy: { timestamp: "desc" },
        select: { close: true },
      }),
    ]);

    const latestSignal = recentSignals[0];
    const modelScores = {
      xgboost: null as number | null,
      lightgbm: null as number | null,
      lstm: null as number | null,
      ensemble: latestSignal ? latestSignal.confidence : null,
    };
    const features: Record<string, number> = {};

    return NextResponse.json({
      data: {
        symbol,
        current_price: latestPrice?.close ?? 0,
        price_history: priceHistory.reverse().map((p) => ({
          date: p.date.toISOString().split("T")[0],
          open: p.open,
          high: p.high,
          low: p.low,
          close: p.close,
          volume: p.volume,
        })),
        fundamentals: fundamentals
          ? {
              market_cap: fundamentals.market_cap,
              market_cap_rank: fundamentals.market_cap_rank,
              total_volume_24h: fundamentals.total_volume_24h,
              circulating_supply: fundamentals.circulating_supply,
              category: fundamentals.category,
            }
          : null,
        recent_signals: recentSignals.map((s) => ({
          ...s,
          created_at: s.created_at.toISOString(),
        })),
        trade_history: tradeHistory.map((p) => ({
          ...p,
          opened_at: p.opened_at.toISOString(),
          closed_at: p.closed_at?.toISOString() ?? null,
        })),
        model_scores: modelScores,
        features,
      },
    });
  } catch (error) {
    console.error("GET /api/assets/[symbol] error:", error);
    return NextResponse.json(
      { error: "Failed to fetch asset details" },
      { status: 500 }
    );
  }
}
