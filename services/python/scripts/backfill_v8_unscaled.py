"""Backfill notional_usd_unscaled / pnl_usd_unscaled / fees_usd_unscaled / cls_multiplier
for v8 paper trades that were closed before the BGM baseline column shipped.

For each v8 closed trade with `pnl_usd_unscaled IS NULL`:
  - If `ml_prob = 1.0` (sentinel — classifier never ran): set cls_multiplier = 1.0 and
    unscaled values equal to the live values (no-op for COALESCE, but explicit).
  - Otherwise: recompute cls_multiplier from `ml_prob` using the per-detector
    quantile mapping in `models/v8_classifier_3y/{stem}_meta.json` (same math as
    `V8Classifier._score_to_size` in src/ml/v8/classifier.py), then back out the
    pre-classifier base notional and recompute pnl_usd_unscaled / fees_usd_unscaled
    using the same FEE_RATE the executor uses.

Idempotent: only updates rows where `pnl_usd_unscaled IS NULL`.

Run:
    docker exec ai-finance-paper-1 python /app/scripts/backfill_v8_unscaled.py
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine, text

FEE_RATE = 0.0006  # mirrors paper_executor.FEE_RATE
V8_SENTINEL_THRESHOLD = 3.0
DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:MySQL100%25@localhost:5432/market",
)

MODELS_DIR = Path(__file__).resolve().parents[1] / "models" / "v8_classifier_3y"

DETECTOR_MODEL_STEMS: dict[str, str] = {
    "v8_macd_pullback_long_e2": "macd_pullback_long_histgbm_3y_v1",
    "v8_macd_pullback_short_e2": "macd_pullback_short_histgbm_3y_v1",
    "v8_macd_early_trend_short_e2": "macd_early_trend_short_histgbm_3y_v1",
    # v8_rsi_recovery_long_e2: not trained (only 3 trades in 3y data) → uses 1.0
}


def _score_to_size(meta: dict, score: float) -> float:
    """Same math as V8Classifier._score_to_size — quantile rank → linear in [low, high]."""
    quantiles = np.asarray(meta["train_score_quantiles"], dtype=float)
    n_q = len(quantiles)
    pos = int(np.searchsorted(quantiles, score, side="right"))
    rank = max(0.0, min(1.0, pos / n_q))
    size_low = float(meta.get("size_low", 0.5))
    size_high = float(meta.get("size_high", 1.5))
    return size_low + (size_high - size_low) * rank


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("backfill")

    metas: dict[str, dict] = {}
    for det, stem in DETECTOR_MODEL_STEMS.items():
        meta_path = MODELS_DIR / f"{stem}_meta.json"
        if not meta_path.exists():
            log.warning("meta missing for %s at %s — those rows will be skipped", det, meta_path)
            continue
        with open(meta_path) as f:
            metas[det] = json.load(f)
    log.info("loaded %d detector metas: %s", len(metas), sorted(metas.keys()))

    engine = create_engine(DB_URL)

    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT id, strategy, ml_prob, position_usd, pnl_pct, pnl_usd, fees_usd
            FROM paper_trades
            WHERE strategy LIKE 'v8\\_%' ESCAPE '\\'
              AND threshold = :th
              AND status IN ('won', 'lost', 'timeout')
              AND pnl_usd_unscaled IS NULL
            ORDER BY id
        """), {"th": V8_SENTINEL_THRESHOLD}).fetchall()

    log.info("found %d candidate rows to backfill", len(rows))
    sentinel_n = 0
    scored_n = 0
    skipped_n = 0

    for row in rows:
        trade_id, strategy, ml_prob, pos_usd, pnl_pct, pnl_usd, fees_usd = row
        ml_prob = float(ml_prob) if ml_prob is not None else 1.0
        pos_usd = float(pos_usd) if pos_usd is not None else 0.0
        pnl_pct = float(pnl_pct) if pnl_pct is not None else 0.0
        pnl_usd_live = float(pnl_usd) if pnl_usd is not None else 0.0
        fees_usd_live = float(fees_usd) if fees_usd is not None else 0.0

        # Sentinel: classifier never ran on this trade. cls_multiplier was 1.0,
        # so notional_usd_unscaled = position_usd and pnl_usd_unscaled = pnl_usd.
        if abs(ml_prob - 1.0) < 1e-9:
            cls_multiplier = 1.0
            notional_usd_unscaled = pos_usd
            pnl_usd_unscaled = pnl_usd_live
            fees_usd_unscaled = fees_usd_live
            sentinel_n += 1
        else:
            meta = metas.get(strategy)
            if meta is None or strategy not in DETECTOR_MODEL_STEMS:
                skipped_n += 1
                continue
            cls_multiplier = _score_to_size(meta, ml_prob)
            if cls_multiplier <= 0:
                skipped_n += 1
                continue
            notional_usd_unscaled = pos_usd / cls_multiplier
            fees_usd_unscaled = notional_usd_unscaled * FEE_RATE * 2
            pnl_usd_unscaled = notional_usd_unscaled * pnl_pct - fees_usd_unscaled
            scored_n += 1

        with engine.begin() as conn:
            conn.execute(text("""
                UPDATE paper_trades
                SET notional_usd_unscaled = :nuu,
                    pnl_usd_unscaled      = :puu,
                    fees_usd_unscaled     = :fuu,
                    cls_multiplier        = :cm
                WHERE id = :id
            """), {
                "nuu": notional_usd_unscaled,
                "puu": pnl_usd_unscaled,
                "fuu": fees_usd_unscaled,
                "cm": cls_multiplier,
                "id": trade_id,
            })

    log.info(
        "backfill done — sentinel=%d, scored=%d, skipped=%d (no_meta or invalid_multiplier)",
        sentinel_n, scored_n, skipped_n,
    )

    # Quick sanity: show the new live-vs-baseline gap per detector
    with engine.connect() as conn:
        diag = conn.execute(text("""
            SELECT strategy,
                   COUNT(*)                                                   AS n,
                   COUNT(*) FILTER (WHERE ml_prob = 1.0)                      AS sentinel_n,
                   ROUND(SUM(pnl_usd)::numeric, 2)                            AS live_pnl,
                   ROUND(SUM(COALESCE(pnl_usd_unscaled, pnl_usd))::numeric, 2) AS baseline_pnl,
                   ROUND((SUM(pnl_usd) - SUM(COALESCE(pnl_usd_unscaled, pnl_usd)))::numeric, 2) AS bgm_lift
            FROM paper_trades
            WHERE strategy LIKE 'v8\\_%' ESCAPE '\\'
              AND threshold = :th
              AND status IN ('won', 'lost', 'timeout')
            GROUP BY strategy
            ORDER BY strategy
        """), {"th": V8_SENTINEL_THRESHOLD}).fetchall()
    log.info("post-backfill per-strategy live vs baseline:")
    for r in diag:
        log.info("  %-32s n=%-4d sentinel=%-4d live=%-8s baseline=%-8s bgm_lift=%s",
                 r[0], r[1], r[2], f"${r[3]}", f"${r[4]}", f"${r[5]}")


if __name__ == "__main__":
    main()
