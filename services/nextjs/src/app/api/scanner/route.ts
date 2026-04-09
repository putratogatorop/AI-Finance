import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const status = searchParams.get("status");
    const symbol = searchParams.get("symbol");
    const limit = parseInt(searchParams.get("limit") || "50");

    const where: any = {};
    if (status) where.status = status;
    if (symbol) where.symbol = symbol;

    const signals = await prisma.scannerSignal.findMany({
      where,
      orderBy: { signal_time: "desc" },
      take: limit,
    });

    const stats = {
      total: await prisma.scannerSignal.count(),
      active: await prisma.scannerSignal.count({ where: { status: "active" } }),
      traded: await prisma.scannerSignal.count({ where: { status: "traded" } }),
    };

    return NextResponse.json({ signals, stats });
  } catch (error) {
    console.error("Scanner API error:", error);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
