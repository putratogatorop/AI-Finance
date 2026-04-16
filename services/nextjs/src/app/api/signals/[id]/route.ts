import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";

export async function PATCH(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const id = parseInt(params.id, 10);
    if (isNaN(id)) {
      return NextResponse.json({ error: "Invalid signal ID" }, { status: 400 });
    }

    const body = await request.json();
    const { acknowledged, acknowledge_status } = body;

    const signal = await prisma.signal.update({
      where: { id },
      data: { acknowledged: acknowledged ?? true },
    });

    await prisma.auditLog.create({
      data: {
        event_type: "signal_acknowledged",
        asset: signal.asset,
        details: JSON.stringify({
          signal_id: id,
          action: signal.action,
          acknowledge_status: acknowledge_status || "seen",
          confidence: signal.confidence,
        }),
        created_at: new Date(),
      },
    });

    return NextResponse.json({
      data: {
        ...signal,
        created_at: signal.created_at.toISOString(),
      },
    });
  } catch (error) {
    console.error("PATCH /api/signals/[id] error:", error);
    return NextResponse.json(
      { error: "Failed to update signal" },
      { status: 500 }
    );
  }
}

export async function GET(
  _request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const id = parseInt(params.id, 10);
    if (isNaN(id)) {
      return NextResponse.json({ error: "Invalid signal ID" }, { status: 400 });
    }

    const signal = await prisma.signal.findUnique({ where: { id } });
    if (!signal) {
      return NextResponse.json({ error: "Signal not found" }, { status: 404 });
    }

    return NextResponse.json({
      data: {
        ...signal,
        created_at: signal.created_at.toISOString(),
      },
    });
  } catch (error) {
    console.error("GET /api/signals/[id] error:", error);
    return NextResponse.json(
      { error: "Failed to fetch signal" },
      { status: 500 }
    );
  }
}
