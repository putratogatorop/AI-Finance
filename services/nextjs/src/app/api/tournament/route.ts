import { NextResponse } from "next/server";
import { readFileSync, existsSync } from "fs";

const TOURNAMENT_PATH =
  "C:/Users/togat/Desktop/AI-Finance/models/results/tournament_summary.json";

export async function GET() {
  try {
    if (!existsSync(TOURNAMENT_PATH)) {
      return NextResponse.json(
        { data: null, error: "Tournament results not found. Run run_tournament.py first." },
        { status: 404 }
      );
    }

    const raw = readFileSync(TOURNAMENT_PATH, "utf-8");
    const data = JSON.parse(raw);

    return NextResponse.json({ data });
  } catch (error) {
    console.error("GET /api/tournament error:", error);
    return NextResponse.json(
      { error: "Failed to load tournament results" },
      { status: 500 }
    );
  }
}
