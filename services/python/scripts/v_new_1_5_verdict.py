"""v_new_1.5 verdict — side-by-side comparison vs v_new_1 baseline.

Loads meta JSONs for both directions of both v_new_1 and v_new_1_v1_5,
computes per-year OOS metric deltas, prints a verdict table, and writes
docs/V_NEW_1_5_VERDICT-{date}.md.

USAGE:
    services/python/.venv/bin/python3 \\
        services/python/scripts/v_new_1_5_verdict.py
"""

from __future__ import annotations

import json
import pathlib
import sys
from datetime import UTC, datetime

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
PYTHON_ROOT = REPO_ROOT / "services" / "python"
sys.path.insert(0, str(PYTHON_ROOT))

MODELS = PYTHON_ROOT / "models"

LIFT_THRESHOLDS = {
    "PASS": 0.05,        # +5% relative on median OOS PR-AUC
    "MARGINAL": 0.01,    # +1% to +5%
}


def _load_meta(direction: str, suffix: str) -> dict:
    name = f"v_new_1_{direction}{suffix}"
    p = MODELS / name / f"{name}_meta.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing meta: {p}")
    with open(p) as f:
        return json.load(f)


def _row(name: str, base: float, new: float) -> str:
    delta = new - base
    pct = (delta / base * 100.0) if base else float("nan")
    return f"| {name} | {base:.4f} | {new:.4f} | {delta:+.4f} ({pct:+.1f}%) |"


def _verdict_label(base: float, new: float) -> str:
    if base <= 0:
        return "UNDEFINED (baseline = 0)"
    rel = (new - base) / base
    if rel >= LIFT_THRESHOLDS["PASS"]:
        return "🟢 PASS"
    if rel >= LIFT_THRESHOLDS["MARGINAL"]:
        return "🟡 MARGINAL"
    if rel >= -LIFT_THRESHOLDS["MARGINAL"]:
        return "⚪ FLAT"
    return "🔴 FAIL"


def main() -> None:
    print("Loading meta JSONs...")
    base_long = _load_meta("long", "")
    base_short = _load_meta("short", "")
    new_long = _load_meta("long", "_v1_5")
    new_short = _load_meta("short", "_v1_5")

    base_long_pr = base_long["aggregate_oos_metrics"]["median_pr_auc"]
    base_short_pr = base_short["aggregate_oos_metrics"]["median_pr_auc"]
    new_long_pr = new_long["aggregate_oos_metrics"]["median_pr_auc"]
    new_short_pr = new_short["aggregate_oos_metrics"]["median_pr_auc"]

    long_verdict = _verdict_label(base_long_pr, new_long_pr)
    short_verdict = _verdict_label(base_short_pr, new_short_pr)

    # Per-year tables
    years = sorted(base_long["per_year_oos_metrics"].keys())
    long_lines = []
    short_lines = []
    for y in years:
        bl = base_long["per_year_oos_metrics"][y].get("pr_auc", float("nan"))
        nl = new_long["per_year_oos_metrics"][y].get("pr_auc", float("nan"))
        bs = base_short["per_year_oos_metrics"][y].get("pr_auc", float("nan"))
        ns = new_short["per_year_oos_metrics"][y].get("pr_auc", float("nan"))
        long_lines.append(_row(f"{y} PR-AUC", bl, nl))
        short_lines.append(_row(f"{y} PR-AUC", bs, ns))

    base_long_top = {
        y: base_long["per_year_oos_metrics"][y].get("top_decile_precision", float("nan"))
        for y in years
    }
    new_long_top = {
        y: new_long["per_year_oos_metrics"][y].get("top_decile_precision", float("nan"))
        for y in years
    }
    base_short_top = {
        y: base_short["per_year_oos_metrics"][y].get("top_decile_precision", float("nan"))
        for y in years
    }
    new_short_top = {
        y: new_short["per_year_oos_metrics"][y].get("top_decile_precision", float("nan"))
        for y in years
    }

    long_top_lines = [
        _row(f"{y} top-decile precision", base_long_top[y], new_long_top[y])
        for y in years
    ]
    short_top_lines = [
        _row(f"{y} top-decile precision", base_short_top[y], new_short_top[y])
        for y in years
    ]

    # SHAP — see if any sequence features cracked the top 20
    sequence_feature_prefix = ("score_long_", "score_short_")
    shap_long = new_long.get("shap_importance_top20", {})
    shap_short = new_short.get("shap_importance_top20", {})
    seq_in_long = {
        f: v for f, v in shap_long.items() if f.startswith(sequence_feature_prefix)
    }
    seq_in_short = {
        f: v for f, v in shap_short.items() if f.startswith(sequence_feature_prefix)
    }

    today = datetime.now(UTC).strftime("%Y-%m-%d")
    md_path = REPO_ROOT / "docs" / f"V_NEW_1_5_VERDICT-{today}.md"

    body = []
    body.append(f"# v_new_1.5 — Sequence-feature BGM Verdict\n")
    body.append(f"**Date:** {today}\n")
    body.append(f"**Baseline:** v_new_1 (44 features, Phase 3 retrained with proper temporal CV + OOF capture)\n")
    body.append(f"**Candidate:** v_new_1.5 (44 base + 12 sequence features = 56)\n")
    body.append("\n---\n\n")
    body.append("## Headline\n\n")
    body.append(f"- **LONG:** {long_verdict}  (baseline {base_long_pr:.4f} → v1.5 {new_long_pr:.4f})\n")
    body.append(f"- **SHORT:** {short_verdict}  (baseline {base_short_pr:.4f} → v1.5 {new_short_pr:.4f})\n")
    body.append("\n## Per-year PR-AUC (LONG)\n\n")
    body.append("| year | v_new_1 | v_new_1.5 | delta (relative) |\n|---|---|---|---|\n")
    body.extend([line + "\n" for line in long_lines])
    body.append("\n## Per-year top-decile precision (LONG)\n\n")
    body.append("| year | v_new_1 | v_new_1.5 | delta (relative) |\n|---|---|---|---|\n")
    body.extend([line + "\n" for line in long_top_lines])
    body.append("\n## Per-year PR-AUC (SHORT)\n\n")
    body.append("| year | v_new_1 | v_new_1.5 | delta (relative) |\n|---|---|---|---|\n")
    body.extend([line + "\n" for line in short_lines])
    body.append("\n## Per-year top-decile precision (SHORT)\n\n")
    body.append("| year | v_new_1 | v_new_1.5 | delta (relative) |\n|---|---|---|---|\n")
    body.extend([line + "\n" for line in short_top_lines])

    body.append("\n## Sequence features in SHAP top-20\n\n")
    body.append(f"- **LONG:** {len(seq_in_long)} of 12 sequence features in top-20\n")
    if seq_in_long:
        body.append("```\n")
        for f, v in seq_in_long.items():
            body.append(f"  {f}: {v}\n")
        body.append("```\n")
    body.append(f"- **SHORT:** {len(seq_in_short)} of 12 sequence features in top-20\n")
    if seq_in_short:
        body.append("```\n")
        for f, v in seq_in_short.items():
            body.append(f"  {f}: {v}\n")
        body.append("```\n")

    body.append("\n## Decision\n\n")
    if "PASS" in long_verdict or "PASS" in short_verdict:
        body.append("- At least one direction passed the +5% lift gate. **Proceed to Phase 4 with v_new_1.5 scores.**\n")
        body.append("- After Phase 4, if 4-year compounded return ≥ 220%, hot-promote v_new_1.5 to shadow scoring.\n")
        body.append("- Then proceed to v_new_2 (CPU LSTM stacker) per V_NEW_2_LSTM_RL_REPLAN-2026-05-04.md.\n")
    elif "MARGINAL" in long_verdict or "MARGINAL" in short_verdict:
        body.append("- Marginal lift on at least one direction. **Treat v_new_1.5 as a side experiment.**\n")
        body.append("- Spend the next compute budget on MACRO Track B (top-200 universe, concentrated sizing).\n")
        body.append("- Do NOT proceed to v_new_2 LSTM yet — sequence signal is too weak to expect LSTM to help.\n")
    else:
        body.append("- No meaningful lift. **Do NOT proceed to v_new_2 LSTM.**\n")
        body.append("- Sequence patterns aren't there at the BGM level; an LSTM probably won't extract them either on this data.\n")
        body.append("- Pivot fully to MACRO Track B.\n")

    md_path.write_text("".join(body))
    print(f"\nWrote: {md_path}")
    print()
    print(f"LONG:  {long_verdict}  ({base_long_pr:.4f} → {new_long_pr:.4f})")
    print(f"SHORT: {short_verdict}  ({base_short_pr:.4f} → {new_short_pr:.4f})")


if __name__ == "__main__":
    main()
