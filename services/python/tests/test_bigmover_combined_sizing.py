"""Sizing function tests."""
from __future__ import annotations

import math

from src.ml.bigmover_combined.sizing import (
    SizingConfig,
    default_sizing_config,
    position_scale,
)


def test_short_with_bear_score_scales_by_magnitude():
    cfg = default_sizing_config()
    # full-bear score = -1.0, mag 1.0, slope 1.5, intercept 0 → 1.5
    assert position_scale(-1.0, "short", config=cfg) == 1.5
    # half-bear → 0.75
    assert math.isclose(position_scale(-0.5, "short", config=cfg), 0.75, abs_tol=1e-12)


def test_long_with_bull_score_scales_by_magnitude():
    cfg = default_sizing_config()
    assert position_scale(1.0, "long", config=cfg) == 1.5
    assert math.isclose(position_scale(0.4, "long", config=cfg), 0.6, abs_tol=1e-12)


def test_sign_mismatch_zeros_position():
    cfg = default_sizing_config()
    assert position_scale(0.5, "short", config=cfg) == 0.0
    assert position_scale(-0.5, "long", config=cfg) == 0.0


def test_zero_score_zeros_position_either_side():
    cfg = default_sizing_config()
    assert position_scale(0.0, "short", config=cfg) == 0.0
    assert position_scale(0.0, "long", config=cfg) == 0.0


def test_nan_score_zeros_position():
    cfg = default_sizing_config()
    assert position_scale(float("nan"), "short", config=cfg) == 0.0
    assert position_scale(float("nan"), "long", config=cfg) == 0.0


def test_cap_clipping():
    cfg = SizingConfig(slope=10.0, intercept=0.0, floor=0.0, cap=2.0)
    # mag=1.0, slope=10 → raw 10, clipped to cap 2.0
    assert position_scale(1.0, "long", config=cfg) == 2.0


def test_floor_clipping():
    cfg = SizingConfig(slope=0.1, intercept=0.0, floor=0.5, cap=1.0)
    # mag=0.1, slope=0.1, intercept=0 → raw 0.01, floored to 0.5
    assert position_scale(0.1, "long", config=cfg) == 0.5


def test_score_clamped_to_unit_magnitude():
    """Scores beyond ±1 are clamped (defensive — true scores stay in [-1,1])."""
    cfg = default_sizing_config()
    assert position_scale(2.5, "long", config=cfg) == position_scale(1.0, "long", config=cfg)
    assert position_scale(-2.5, "short", config=cfg) == position_scale(-1.0, "short", config=cfg)
