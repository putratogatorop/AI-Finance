import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const limit = parseInt(searchParams.get("limit") || "100", 10);
    const offset = parseInt(searchParams.get("offset") || "0", 10);
    const eventType = searchParams.get("event_type");
    const asset = searchParams.get("asset");

    const where: Record<string, unknown> = {};
    if (eventType) where.event_type = eventType;
    if (asset) where.asset = asset.toUpperCase();

    const [entries, total] = await Promise.all([
      prisma.auditLog.findMany({
        where,
        orderBy: { created_at: "desc" },
        take: limit,
        skip: offset,
      }),
      prisma.auditLog.count({ where }),
    ]);

    return NextResponse.json({
      data: entries.map((e) => {
        let parsedDetails: Record<string, unknown> | undefined;
        if (e.details) {
          try {
            parsedDetails = JSON.parse(e.details);
          } catch {
            parsedDetails = undefined;
          }
        }
        return {
          ...e,
          created_at: e.created_at.toISOString(),
          parsed_details: parsedDetails,
        };
      }),
      total,
      limit,
      offset,
    });
  } catch (error) {
    console.error("GET /api/audit error:", error);
    return NextResponse.json(
      { error: "Failed to fetch audit log" },
      { status: 500 }
    );
  }
}
