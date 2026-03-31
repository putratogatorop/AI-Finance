import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import type { RiskSettings } from "@/types";

const DEFAULT_SETTINGS: RiskSettings = {
  stop_loss_pct: -8,
  max_positions: 8,
  confidence_threshold: 0.7,
  position_size_idr: 1_000_000,
  max_single_asset_exposure_pct: 25,
  portfolio_drawdown_pause_pct: 20,
  retrain_schedule: "weekly",
};

export async function GET(_request: NextRequest) {
  try {
    const settingsEntry = await prisma.auditLog.findFirst({
      where: { event_type: "settings_updated" },
      orderBy: { created_at: "desc" },
    });

    if (settingsEntry?.details) {
      try {
        const settings = JSON.parse(
          settingsEntry.details
        ) as RiskSettings;
        return NextResponse.json({ data: settings });
      } catch {
        // Fall back to defaults
      }
    }

    return NextResponse.json({ data: DEFAULT_SETTINGS });
  } catch (error) {
    console.error("GET /api/settings error:", error);
    return NextResponse.json(
      { error: "Failed to fetch settings" },
      { status: 500 }
    );
  }
}

export async function PUT(request: NextRequest) {
  try {
    const body = await request.json();

    const settings: RiskSettings = {
      stop_loss_pct: parseFloat(body.stop_loss_pct),
      max_positions: parseInt(body.max_positions, 10),
      confidence_threshold: parseFloat(body.confidence_threshold),
      position_size_idr: parseInt(body.position_size_idr, 10),
      max_single_asset_exposure_pct: parseFloat(
        body.max_single_asset_exposure_pct
      ),
      portfolio_drawdown_pause_pct: parseFloat(
        body.portfolio_drawdown_pause_pct
      ),
      retrain_schedule: body.retrain_schedule,
    };

    if (settings.stop_loss_pct > 0 || settings.stop_loss_pct < -50) {
      return NextResponse.json(
        { error: "Stop loss must be between -50% and 0%" },
        { status: 400 }
      );
    }
    if (settings.max_positions < 1 || settings.max_positions > 20) {
      return NextResponse.json(
        { error: "Max positions must be between 1 and 20" },
        { status: 400 }
      );
    }
    if (
      settings.confidence_threshold < 0.1 ||
      settings.confidence_threshold > 1.0
    ) {
      return NextResponse.json(
        {
          error: "Confidence threshold must be between 0.1 and 1.0",
        },
        { status: 400 }
      );
    }
    if (
      settings.position_size_idr < 100_000 ||
      settings.position_size_idr > 100_000_000
    ) {
      return NextResponse.json(
        {
          error:
            "Position size must be between Rp 100.000 and Rp 100.000.000",
        },
        { status: 400 }
      );
    }

    await prisma.auditLog.create({
      data: {
        event_type: "settings_updated",
        details: JSON.stringify(settings),
        created_at: new Date(),
      },
    });

    return NextResponse.json({ data: settings });
  } catch (error) {
    console.error("PUT /api/settings error:", error);
    return NextResponse.json(
      { error: "Failed to update settings" },
      { status: 500 }
    );
  }
}
