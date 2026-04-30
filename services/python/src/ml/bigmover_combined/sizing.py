"""Continuous BTC-trend-score position sizing.

Sign-locked: a SHORT trade only takes a non-zero size when the BTC trend
score is negative; a LONG trade only when the score is positive. The
magnitude of the score scales the position. Out-of-direction trades are
zeroed (the explicit "no trades against BTC trend" rule).

Designed for the Candidate-B trend score (range [-1, +1], from
src/ml/bigmover_combined/features.py:btc_trend_score). Other scores in
the same range work with the same sizing.

The sizing curve and its knobs are written to a JSON manifest at training
time so the backtest reproduces exactly. See `default_sizing_config()`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SizingConfig:
    """Knobs for the continuous sizing curve.

    Curve formula:
      direction_ok = (score > 0 and direction == "long")
                  or (score < 0 and direction == "short")
      if not direction_ok:
          scale = 0.0
      else:
          mag = min(abs(score), 1.0)
          scale = clip(slope * mag + intercept, floor, cap)

    Defaults from the Phase-3 BTC-trend specialist's recommendation:
      slope=1.5, intercept=0.0, floor=0.0, cap=1.5  → linear in |score|, capped 1.5x.
    """
    slope: float = 1.5
    intercept: float = 0.0
    floor: float = 0.0
    cap: float = 1.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def position_scale(
    btc_trend_score: float,
    direction: str,
    *,
    config: SizingConfig | None = None,
) -> float:
    """Continuous sizing scale.

    Args:
      btc_trend_score: signed score in [-1, +1]. Positive = bull, negative = bear.
      direction: 'short' or 'long'.
      config: SizingConfig, or None for defaults.

    Returns:
      Position scale (multiplier on base position size). Always in [floor, cap]
      when direction matches the sign of the score; 0.0 otherwise.
      Float('nan') for the score is treated as no information → 0.0 (don't trade).
    """
    cfg = config or SizingConfig()
    if btc_trend_score is None:
        return 0.0
    # NaN check without importing math/numpy (this fn is hot).
    if btc_trend_score != btc_trend_score:
        return 0.0
    direction_ok = (
        (btc_trend_score > 0.0 and direction == "long")
        or (btc_trend_score < 0.0 and direction == "short")
    )
    if not direction_ok:
        return 0.0
    mag = abs(btc_trend_score)
    if mag > 1.0:
        mag = 1.0
    raw = cfg.slope * mag + cfg.intercept
    if raw < cfg.floor:
        return cfg.floor
    if raw > cfg.cap:
        return cfg.cap
    return raw


def default_sizing_config() -> SizingConfig:
    return SizingConfig(slope=1.5, intercept=0.0, floor=0.0, cap=1.5)
