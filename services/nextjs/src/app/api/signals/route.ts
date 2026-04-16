import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const limit = parseInt(searchParams.get("limit") || "50", 10);
    const offset = parseInt(searchParams.get("offset") || "0", 10);
    const asset = searchParams.get("asset");
    const action = searchParams.get("action");
    const acknowledged = searchParams.get("acknowledged");

    const where: Record<string, unknown> = {};
    if (asset) where.asset = asset.toUpperCase();
    if (action) where.action = action.toUpperCase();
    if (acknowledged !== null && acknowledged !== undefined && acknowledged !== "") {
      where.acknowledged = acknowledged === "true";
    }

    const [signals, total] = await Promise.all([
      prisma.signal.findMany({
        where,
        orderBy: { created_at: "desc" },
        take: limit,
        skip: offset,
      }),
      prisma.signal.count({ where }),
    ]);

    return NextResponse.json({
      data: signals.map((s) => ({
        ...s,
        created_at: s.created_at.toISOString(),
      })),
      total,
      limit,
      offset,
    });
  } catch (error) {
    console.error("GET /api/signals error:", error);
    return NextResponse.json(
      { error: "Failed to fetch signals" },
      { status: 500 }
    );
  }
}
