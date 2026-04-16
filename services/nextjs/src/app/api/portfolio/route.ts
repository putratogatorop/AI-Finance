import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const status = searchParams.get("status");
    const asset = searchParams.get("asset");

    const where: Record<string, unknown> = {};
    if (status) where.status = status;
    if (asset) where.asset = asset.toUpperCase();

    const positions = await prisma.portfolio.findMany({
      where,
      orderBy: { opened_at: "desc" },
    });

    const openPositions = positions.filter((p) => p.status === "open");
    const currentPrices: Record<string, number> = {};

    if (openPositions.length > 0) {
      const assets = [...new Set(openPositions.map((p) => p.asset))];
      for (const a of assets) {
        const latestPrice = await prisma.assetPriceHourly.findFirst({
          where: { asset: a },
          orderBy: { timestamp: "desc" },
          select: { close: true },
        });
        if (latestPrice) {
          currentPrices[a] = latestPrice.close;
        }
      }
    }

    const enrichedPositions = positions.map((p) => {
      const currentPrice = currentPrices[p.asset] ?? p.entry_price;
      const isOpen = p.status === "open";
      return {
        ...p,
        opened_at: p.opened_at.toISOString(),
        closed_at: p.closed_at?.toISOString() ?? null,
        current_price: isOpen ? currentPrice : undefined,
        unrealized_pnl_idr: isOpen
          ? Math.round(
              (currentPrice - p.entry_price) * p.quantity * 16000
            )
          : undefined,
        unrealized_pnl_pct: isOpen
          ? ((currentPrice - p.entry_price) / p.entry_price) * 100
          : undefined,
      };
    });

    const openCount = enrichedPositions.filter(
      (p) => p.status === "open"
    ).length;
    const totalInvested = enrichedPositions
      .filter((p) => p.status === "open")
      .reduce((sum, p) => sum + p.entry_amount_idr, 0);
    const totalUnrealizedPnl = enrichedPositions
      .filter((p) => p.status === "open")
      .reduce((sum, p) => sum + (p.unrealized_pnl_idr ?? 0), 0);
    const totalRealizedPnl = enrichedPositions
      .filter((p) => p.status !== "open")
      .reduce((sum, p) => sum + (p.pnl_idr ?? 0), 0);

    return NextResponse.json({
      data: enrichedPositions,
      summary: {
        open_positions: openCount,
        total_invested_idr: totalInvested,
        total_unrealized_pnl_idr: totalUnrealizedPnl,
        total_realized_pnl_idr: totalRealizedPnl,
        portfolio_value_idr: totalInvested + totalUnrealizedPnl,
      },
    });
  } catch (error) {
    console.error("GET /api/portfolio error:", error);
    return NextResponse.json(
      { error: "Failed to fetch portfolio" },
      { status: 500 }
    );
  }
}
