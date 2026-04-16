"""Layered exit backtest: SHORT-only, bar-by-bar sim with trailing stops."""
import sys

sys.path.insert(0, ".")

import glob
import logging
from dataclasses import dataclass
from itertools import product

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

import os
DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:MySQL100%25@localhost:5432/market")
DATA_DIR = "../../data/raw/15m"
THRESHOLD = 0.90
BARS_24H = 96  # 24h of 15m candles


# ---------------------------------------------------------------------------
# Task 1 — CSV Loader
# ---------------------------------------------------------------------------


def load_symbol_bars(symbol: str) -> pd.DataFrame:
    """Load all 15m CSVs for *symbol*, concat, convert open_time, sort."""
    pattern = f"{DATA_DIR}/{symbol}/{symbol}-15m-*.csv"
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files found for {symbol} at {pattern}")

    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    # Older CSVs use milliseconds (13 digits), newer ones microseconds (16 digits).
    # Normalise everything to microseconds before converting.
    ts = df["open_time"].values.copy()
    mask_ms = ts < 1_000_000_000_000_000  # < 1e15 → milliseconds
    ts[mask_ms] *= 1000  # ms → us
    df["open_time"] = pd.to_datetime(ts, unit="us", utc=True)
    df.sort_values("open_time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def get_bars_after_signal(
    bars_df: pd.DataFrame, signal_time, n_bars: int = 96
) -> np.ndarray | None:
    """Return (n_bars, 3) array of [high, low, close] starting AFTER signal_time.

    Uses searchsorted for O(log n) lookup. Returns None if not enough bars remain.
    """
    idx = bars_df["open_time"].searchsorted(signal_time, side="right")
    if idx + n_bars > len(bars_df):
        return None
    chunk = bars_df.iloc[idx : idx + n_bars]
    return chunk[["high", "low", "close"]].to_numpy(dtype=np.float64)


# ---------------------------------------------------------------------------
# Task 2 — Single-Trade Layered Sim
# ---------------------------------------------------------------------------


@dataclass
class LayerConfig:
    weight: float
    sl: float
    tp: float
    trail_act: float
    trail_dist: float


def sim_trade(
    entry: float, bars: np.ndarray, layers: list[LayerConfig]
) -> tuple[float, str, str]:
    """Simulate a SHORT trade with layered exits.

    Parameters
    ----------
    entry : float
        Entry price.
    bars : np.ndarray
        Shape (n_bars, 3) with columns [high, low, close].
    layers : list[LayerConfig]
        Exit layers with weight, SL, TP, trailing activation & distance.

    Returns
    -------
    total_pnl : float
        Weighted sum of per-layer PnL (as fraction, e.g. 0.05 = +5 %).
    exit_reason : str
        Dominant exit type by weight.
    exit_detail : str
        Per-layer breakdown like "L1:sl_hit-1.5 L2:tp_hit+6.0 L3:trail_hit+4.5".
    """
    n_layers = len(layers)
    closed = [False] * n_layers
    layer_pnl = [0.0] * n_layers
    layer_reason = ["time_exit"] * n_layers
    best_fav = [0.0] * n_layers  # best favorable move seen per layer

    for bar in bars:
        bar_high, bar_low, bar_close = bar[0], bar[1], bar[2]
        adverse = (bar_high - entry) / entry  # price UP → bad for short
        favorable = (entry - bar_low) / entry  # price DOWN → good for short

        for i, layer in enumerate(layers):
            if closed[i]:
                continue

            # 1. SL check (conservative — SL first within same bar)
            if adverse >= layer.sl:
                layer_pnl[i] = -layer.sl
                layer_reason[i] = "sl_hit"
                closed[i] = True
                continue

            # 2. Trailing stop check
            if best_fav[i] >= layer.trail_act:
                # Trailing is active — check retracement from best
                current_fav_from_high = (entry - bar_high) / entry
                retracement = best_fav[i] - current_fav_from_high
                if retracement >= layer.trail_dist:
                    layer_pnl[i] = best_fav[i] - layer.trail_dist
                    layer_reason[i] = "trail_hit"
                    closed[i] = True
                    continue

            # 3. TP check
            if favorable >= layer.tp:
                layer_pnl[i] = layer.tp
                layer_reason[i] = "tp_hit"
                closed[i] = True
                continue

            # 4. Update best favorable
            if favorable > best_fav[i]:
                best_fav[i] = favorable

        # Early exit if all layers closed
        if all(closed):
            break

    # Time-exit remaining layers at last close
    last_close = bars[-1, 2]
    for i in range(n_layers):
        if not closed[i]:
            layer_pnl[i] = (entry - last_close) / entry
            layer_reason[i] = "time_exit"
            closed[i] = True

    # Aggregate
    total_pnl = sum(layers[i].weight * layer_pnl[i] for i in range(n_layers))

    # Dominant exit reason by weight
    reason_weights: dict[str, float] = {}
    for i in range(n_layers):
        reason_weights[layer_reason[i]] = (
            reason_weights.get(layer_reason[i], 0.0) + layers[i].weight
        )
    exit_reason = max(reason_weights, key=reason_weights.get)  # type: ignore[arg-type]

    # Detail string
    parts = []
    for i in range(n_layers):
        parts.append(f"L{i+1}:{layer_reason[i]}{layer_pnl[i]*100:+.1f}")
    exit_detail = " ".join(parts)

    return total_pnl, exit_reason, exit_detail


# ---------------------------------------------------------------------------
# Task 3 — Signal Loading + Bar Preloading
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    'vol_ratio', 'buy_ratio', 'price_change_1bar', 'price_change_2bar',
    'bar_range_pct', 'upper_wick_pct', 'lower_wick_pct', 'body_pct',
    'atr_14', 'rsi_14', 'volatility_20',
    'dist_from_20_high', 'dist_from_20_low', 'price_vs_ema_50',
    'btc_ret_24bar', 'btc_ret_96bar', 'btc_vol_ratio',
    'direction', 'tf_numeric',
    'avg_trade_size', 'vol_x_buyrat', 'candle_quality', 'wick_ratio',
    'dollar_imbalance', 'exec_vs_close', 'vol_ratio_adj',
    'trend_aligned', 'btc_aligned', 'dist_range_pos',
    'coin_gainer_ratio',
]


def load_signals(engine, model) -> pd.DataFrame:
    """Load volume breakouts, engineer features, run ML, filter SHORT picks."""
    # Whitelist: coins with >= 20 prior backtest trades
    # Fall back to volume_breakouts if ml_backtest_trades is empty/missing
    with engine.connect() as conn:
        try:
            wl_rows = conn.execute(text("""
                SELECT symbol FROM ml_backtest_trades
                GROUP BY symbol HAVING COUNT(*) >= 20
            """)).fetchall()
        except Exception:
            wl_rows = []
        if not wl_rows:
            logger.warning("ml_backtest_trades empty/missing, building whitelist from "
                           "volume_breakouts (direction=-1, >= 20 signals)")
            wl_rows = conn.execute(text("""
                SELECT symbol FROM volume_breakouts
                WHERE direction = -1
                GROUP BY symbol HAVING COUNT(*) >= 20
            """)).fetchall()
    whitelist = {r[0] for r in wl_rows}
    logger.info(f"Whitelist: {len(whitelist)} coins")

    # Load data
    logger.info("Loading volume_breakouts...")
    df = pd.read_sql(
        "SELECT * FROM volume_breakouts "
        "WHERE fwd_max_gain_24h IS NOT NULL AND fwd_max_loss_24h IS NOT NULL",
        engine,
    )
    logger.info(f"Loaded {len(df)} rows")

    # Feature engineering (exact same as backtest_short_only.py)
    df['avg_trade_size'] = df['quote_volume'] / df['trades'].clip(lower=1)
    df['vol_x_buyrat'] = df['vol_ratio'] * df['buy_ratio']
    df['candle_quality'] = df['body_pct'] - df['upper_wick_pct']
    df['wick_ratio'] = df['upper_wick_pct'] / (df['lower_wick_pct'] + 0.001)
    df['dollar_imbalance'] = (
        (2 * df['taker_buy_quote'] - df['quote_volume']) / df['quote_volume'].clip(lower=1)
    )
    df['exec_vs_close'] = (
        (df['quote_volume'] / df['volume'].clip(lower=1e-10)) / df['close'].clip(lower=1e-10)
    )
    df['vol_ratio_adj'] = df['vol_ratio'] / (df['volatility_20'] + 0.001)
    df['trend_aligned'] = (df['direction'] * df['price_vs_ema_50']).clip(lower=0)
    df['btc_aligned'] = (df['direction'] * df['btc_ret_24bar']).clip(lower=0)
    df['dist_range_pos'] = df['dist_from_20_high'] / (
        df['dist_from_20_high'].abs() + df['dist_from_20_low'].abs() + 0.001
    )
    df['tf_numeric'] = df['timeframe'].map({'15m': 0, '1h': 1, '4h': 2})
    df['signal_time'] = pd.to_datetime(df['signal_time'])

    coin_stats = df[df['is_big_mover'] == True].groupby('symbol').agg(  # noqa: E712
        gainers=('move_type', lambda x: (x == 'GAINER').sum()),
        losers=('move_type', lambda x: (x == 'LOSER').sum()),
    ).reset_index()
    coin_stats['coin_gainer_ratio'] = coin_stats['gainers'] / (coin_stats['losers'] + 1)
    df['coin_gainer_ratio'] = df['symbol'].map(
        dict(zip(coin_stats['symbol'], coin_stats['coin_gainer_ratio']))
    ).fillna(1.5)

    # ML prediction
    X = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
    proba = model.predict_proba(X)[:, 1]
    df['ml_proba'] = proba

    # Filter: SHORT only + ML >= 0.90 + whitelist
    picks = df[
        (df['ml_proba'] >= THRESHOLD) & (df['direction'] == -1) & (df['symbol'].isin(whitelist))
    ].copy()
    logger.info(f"SHORT-only picks: {len(picks)}")

    # Convert signal_time to UTC
    picks['signal_time'] = pd.to_datetime(picks['signal_time'], utc=True)
    return picks


def preload_bars(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Load 15m bar data for all symbols. Return dict {symbol: DataFrame}."""
    bar_cache: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            bar_cache[sym] = load_symbol_bars(sym)
        except FileNotFoundError:
            logger.warning(f"No CSV data for {sym}, skipping")
    logger.info(f"Bar cache: {len(bar_cache)} symbols loaded")
    return bar_cache


def attach_bar_arrays(picks: pd.DataFrame, bar_cache: dict[str, pd.DataFrame]) -> list[dict]:
    """Attach bar arrays to each signal row. Skip if not enough data."""
    trades = []
    skipped = 0
    for _, r in picks.iterrows():
        sym = r['symbol']
        if sym not in bar_cache:
            skipped += 1
            continue
        bars = get_bars_after_signal(bar_cache[sym], r['signal_time'], BARS_24H)
        if bars is None:
            skipped += 1
            continue
        trades.append({
            'symbol': sym,
            'timeframe': r['timeframe'],
            'signal_time': r['signal_time'],
            'ml_proba': float(r['ml_proba']),
            'close': float(r['close']),
            'vol_ratio': float(r['vol_ratio']),
            'buy_ratio': float(r['buy_ratio']),
            'direction': int(r['direction']),
            'is_big_mover': bool(r['is_big_mover']),
            'move_type': r['move_type'],
            'fwd_max_gain_24h': float(r['fwd_max_gain_24h']),
            'fwd_max_loss_24h': float(r['fwd_max_loss_24h']),
            'fwd_ret_24h': float(r['fwd_ret_24h']) if pd.notna(r['fwd_ret_24h']) else None,
            'fwd_ret_48h': float(r['fwd_ret_48h']) if pd.notna(r['fwd_ret_48h']) else None,
            'fwd_ret_1w': float(r['fwd_ret_1w']) if pd.notna(r['fwd_ret_1w']) else None,
            'rsi_14': float(r['rsi_14']) if pd.notna(r['rsi_14']) else None,
            'atr_14': float(r['atr_14']) if pd.notna(r['atr_14']) else None,
            'btc_ret_24bar': float(r['btc_ret_24bar']) if pd.notna(r['btc_ret_24bar']) else None,
            'coin_gainer_ratio': float(r['coin_gainer_ratio']),
            'bars': bars,
        })
    logger.info(f"Attached bars: {len(trades)} trades, {skipped} skipped")
    return trades


# ---------------------------------------------------------------------------
# Task 4 — Grid Sweep Phase 1 (Per-Layer)
# ---------------------------------------------------------------------------

LAYER_GRIDS = [
    {
        'sl': [0.01, 0.015, 0.02],
        'tp': [0.02, 0.03, 0.04],
        'trail_act': [0.015, 0.02, 0.03],
        'trail_dist': [0.005, 0.01, 0.015],
    },
    {
        'sl': [0.02, 0.03, 0.04],
        'tp': [0.05, 0.06, 0.08],
        'trail_act': [0.03, 0.04, 0.05],
        'trail_dist': [0.01, 0.015, 0.02],
    },
    {
        'sl': [0.04, 0.05, 0.07],
        'tp': [0.08, 0.10, 0.13],
        'trail_act': [0.05, 0.07, 0.09],
        'trail_dist': [0.02, 0.03, 0.04],
    },
]
LAYER_WEIGHTS = [0.25, 0.50, 0.25]

DEFAULT_LAYERS = [
    LayerConfig(weight=0.25, sl=0.015, tp=0.03, trail_act=0.02, trail_dist=0.01),
    LayerConfig(weight=0.50, sl=0.03, tp=0.06, trail_act=0.04, trail_dist=0.015),
    LayerConfig(weight=0.25, sl=0.05, tp=0.10, trail_act=0.07, trail_dist=0.03),
]


def eval_config(trades: list[dict], layers: list[LayerConfig]) -> dict:
    """Run sim_trade on all trades, then filter to 1-trade/day. Return stats."""
    results = []
    for t in trades:
        pnl, reason, detail = sim_trade(t['close'], t['bars'], layers)
        results.append({
            'signal_time': t['signal_time'],
            'ml_proba': t['ml_proba'],
            'pnl': pnl,
            'reason': reason,
        })

    # 1-trade/day: pick highest ml_proba per day
    df = pd.DataFrame(results)
    df['date'] = pd.to_datetime(df['signal_time']).dt.date
    df = df.sort_values('ml_proba', ascending=False).drop_duplicates('date', keep='first')

    n = len(df)
    if n == 0:
        return {'n': 0, 'wr': 0.0, 'pf': 0.0, 'total_pnl': 0.0, 'avg_pnl': 0.0}

    wins = (df['pnl'] > 0).sum()
    wr = wins / n
    gross_profit = df.loc[df['pnl'] > 0, 'pnl'].sum()
    gross_loss = abs(df.loc[df['pnl'] <= 0, 'pnl'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else 99.0
    total_pnl = df['pnl'].sum()
    avg_pnl = df['pnl'].mean()

    return {'n': n, 'wr': wr, 'pf': pf, 'total_pnl': total_pnl, 'avg_pnl': avg_pnl}


def sweep_phase1(trades: list[dict]) -> list[list[LayerConfig]]:
    """Sweep each layer independently (81 combos each). Return top 5 per layer."""
    top_per_layer: list[list[LayerConfig]] = []

    for layer_idx in range(3):
        grid = LAYER_GRIDS[layer_idx]
        combos = list(product(grid['sl'], grid['tp'], grid['trail_act'], grid['trail_dist']))
        logger.info(f"Phase 1 — Layer {layer_idx + 1}: sweeping {len(combos)} combos")

        results = []
        for sl, tp, trail_act, trail_dist in combos:
            layers = list(DEFAULT_LAYERS)  # shallow copy
            layers[layer_idx] = LayerConfig(
                weight=LAYER_WEIGHTS[layer_idx],
                sl=sl, tp=tp, trail_act=trail_act, trail_dist=trail_dist,
            )
            stats = eval_config(trades, layers)
            results.append((stats, LayerConfig(
                weight=LAYER_WEIGHTS[layer_idx],
                sl=sl, tp=tp, trail_act=trail_act, trail_dist=trail_dist,
            )))

        # Sort by (PF, total_pnl) descending
        results.sort(key=lambda x: (x[0]['pf'], x[0]['total_pnl']), reverse=True)
        top5 = [r[1] for r in results[:5]]
        top_per_layer.append(top5)

        logger.info(f"  Layer {layer_idx + 1} top 5:")
        for i, (stats, cfg) in enumerate(results[:5]):
            logger.info(
                f"    #{i+1} SL={cfg.sl} TP={cfg.tp} TR={cfg.trail_act}/{cfg.trail_dist}"
                f"  PF={stats['pf']:.2f} PnL={stats['total_pnl']*100:+.1f}%"
                f" WR={stats['wr']*100:.1f}% n={stats['n']}"
            )

    return top_per_layer


# ---------------------------------------------------------------------------
# Task 5 — Grid Sweep Phase 2 (Cross-Layer)
# ---------------------------------------------------------------------------


def sweep_phase2(
    trades: list[dict], top_per_layer: list[list[LayerConfig]]
) -> tuple[list[LayerConfig], dict]:
    """Cross-product of top 5 x 5 x 5 = 125 combos. Return best config + stats."""
    combos = list(product(top_per_layer[0], top_per_layer[1], top_per_layer[2]))
    logger.info(f"Phase 2 — sweeping {len(combos)} cross-layer combos")

    best_layers: list[LayerConfig] = []
    best_stats: dict = {'pf': 0.0, 'total_pnl': -999.0}

    for i, (l1, l2, l3) in enumerate(combos):
        layers = [l1, l2, l3]
        stats = eval_config(trades, layers)

        if (stats['pf'], stats['total_pnl']) > (best_stats['pf'], best_stats['total_pnl']):
            best_stats = stats
            best_layers = layers

        if (i + 1) % 25 == 0:
            logger.info(
                f"  Phase 2 progress: {i+1}/{len(combos)}"
                f"  best PF={best_stats['pf']:.2f} PnL={best_stats['total_pnl']*100:+.1f}%"
            )

    logger.info(f"\n{'='*60}")
    logger.info("BEST CONFIG:")
    for i, l in enumerate(best_layers):
        logger.info(
            f"  Layer {i+1}: w={l.weight} SL={l.sl} TP={l.tp}"
            f" trail_act={l.trail_act} trail_dist={l.trail_dist}"
        )
    logger.info(
        f"  Stats: n={best_stats['n']} WR={best_stats['wr']*100:.1f}%"
        f" PF={best_stats['pf']:.2f} PnL={best_stats['total_pnl']*100:+.1f}%"
        f" Avg={best_stats['avg_pnl']*100:+.2f}%"
    )
    logger.info(f"{'='*60}")

    return best_layers, best_stats


# ---------------------------------------------------------------------------
# Task 6 — Save Results to ml_backtest_trades
# ---------------------------------------------------------------------------


def run_and_save(trades: list[dict], layers: list[LayerConfig], engine):
    """Run sim on all trades with best layers, DROP+recreate table, insert, print stats."""
    # Run sim on all trades
    results = []
    for t in trades:
        pnl, reason, detail = sim_trade(t['close'], t['bars'], layers)
        results.append({**t, 'pnl_pct': pnl, 'exit_reason': reason, 'exit_detail': detail})

    # DROP and recreate table
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS ml_backtest_trades"))
        conn.execute(text("""
            CREATE TABLE ml_backtest_trades (
                id SERIAL PRIMARY KEY, symbol VARCHAR(20), timeframe VARCHAR(4),
                signal_time TIMESTAMPTZ, ml_proba DOUBLE PRECISION,
                close_price DOUBLE PRECISION,
                vol_ratio DOUBLE PRECISION, buy_ratio DOUBLE PRECISION,
                direction SMALLINT,
                is_big_mover BOOLEAN, move_type VARCHAR(10),
                pnl_pct DOUBLE PRECISION,
                exit_reason VARCHAR(20), status VARCHAR(10),
                exit_detail VARCHAR(100),
                fwd_max_gain_24h DOUBLE PRECISION, fwd_max_loss_24h DOUBLE PRECISION,
                fwd_ret_24h DOUBLE PRECISION, fwd_ret_48h DOUBLE PRECISION,
                fwd_ret_1w DOUBLE PRECISION,
                rsi_14 DOUBLE PRECISION, atr_14 DOUBLE PRECISION,
                btc_ret_24bar DOUBLE PRECISION,
                coin_gainer_ratio DOUBLE PRECISION
            )
        """))
        conn.commit()

    # Build rows for insert — ensure all values are native Python types
    def _f(v):
        """Convert numpy/pandas scalar to Python float or None."""
        if v is None:
            return None
        try:
            fv = float(v)
            return None if np.isnan(fv) else fv
        except (TypeError, ValueError):
            return None

    rows = []
    for r in results:
        rows.append({
            'symbol': r['symbol'],
            'timeframe': r['timeframe'],
            'signal_time': r['signal_time'],
            'ml_proba': float(r['ml_proba']),
            'close_price': float(r['close']),
            'vol_ratio': float(r['vol_ratio']),
            'buy_ratio': _f(r['buy_ratio']),
            'direction': int(r['direction']),
            'is_big_mover': bool(r['is_big_mover']),
            'move_type': r['move_type'] if r.get('move_type') not in (None, 'NONE') else None,
            'pnl_pct': float(r['pnl_pct']) if pd.notna(r['pnl_pct']) else 0.0,
            'exit_reason': r['exit_reason'],
            'status': 'won' if r['pnl_pct'] > 0 else 'lost',
            'exit_detail': r['exit_detail'],
            'fwd_max_gain_24h': _f(r['fwd_max_gain_24h']),
            'fwd_max_loss_24h': _f(r['fwd_max_loss_24h']),
            'fwd_ret_24h': _f(r['fwd_ret_24h']),
            'fwd_ret_48h': _f(r['fwd_ret_48h']),
            'fwd_ret_1w': _f(r['fwd_ret_1w']),
            'rsi_14': _f(r['rsi_14']),
            'atr_14': _f(r['atr_14']),
            'btc_ret_24bar': _f(r['btc_ret_24bar']),
            'coin_gainer_ratio': float(r['coin_gainer_ratio']),
        })

    # Insert in batches of 1000
    Session = sessionmaker(bind=engine)
    session = Session()
    for i in range(0, len(rows), 1000):
        session.execute(text("""
            INSERT INTO ml_backtest_trades
                (symbol, timeframe, signal_time, ml_proba, close_price, vol_ratio, buy_ratio,
                 direction, is_big_mover, move_type, pnl_pct, exit_reason, status, exit_detail,
                 fwd_max_gain_24h, fwd_max_loss_24h, fwd_ret_24h, fwd_ret_48h, fwd_ret_1w,
                 rsi_14, atr_14, btc_ret_24bar, coin_gainer_ratio)
            VALUES
                (:symbol, :timeframe, :signal_time, :ml_proba, :close_price, :vol_ratio,
                 :buy_ratio, :direction, :is_big_mover, :move_type, :pnl_pct, :exit_reason,
                 :status, :exit_detail, :fwd_max_gain_24h, :fwd_max_loss_24h, :fwd_ret_24h,
                 :fwd_ret_48h, :fwd_ret_1w, :rsi_14, :atr_14, :btc_ret_24bar,
                 :coin_gainer_ratio)
        """), rows[i:i + 1000])
        session.commit()
    session.close()
    logger.info(f"Saved {len(rows)} trades to ml_backtest_trades")

    # Print final stats
    with engine.connect() as conn:
        s = conn.execute(text("""
            SELECT COUNT(*)::int,
                COUNT(*) FILTER (WHERE status='won')::int,
                ROUND(COUNT(*) FILTER (WHERE status='won')::numeric
                      / GREATEST(COUNT(*),1) * 100, 1),
                ROUND(NULLIF(SUM(pnl_pct) FILTER (WHERE pnl_pct>0), 0)::numeric
                      / ABS(NULLIF(SUM(pnl_pct) FILTER (WHERE pnl_pct<=0), 0))::numeric, 2),
                ROUND(AVG(pnl_pct)::numeric * 100, 2),
                ROUND(SUM(pnl_pct)::numeric * 100, 0),
                COUNT(*) FILTER (WHERE exit_reason='tp_hit')::int,
                COUNT(*) FILTER (WHERE exit_reason='sl_hit')::int,
                COUNT(*) FILTER (WHERE exit_reason='time_exit')::int,
                COUNT(*) FILTER (WHERE exit_reason='trail_hit')::int
            FROM ml_backtest_trades
        """)).fetchone()
        logger.info(f"\n{'='*60}")
        logger.info("SHORT-ONLY LAYERED EXIT: ML>=0.90 (all trades)")
        logger.info(f"{'='*60}")
        logger.info(f"Trades: {s[0]}")
        logger.info(f"W/L: {s[1]}W / {s[0]-s[1]}L")
        logger.info(f"Win Rate: {s[2]}%")
        logger.info(f"Profit Factor: {float(s[3]):.2f}" if s[3] else "Profit Factor: N/A")
        logger.info(f"Avg PnL: {float(s[4]):+.2f}%")
        logger.info(f"Total PnL: {float(s[5]):+.0f}%")
        logger.info(f"Exits: TP={s[6]}, SL={s[7]}, Trail={s[9]}, Time={s[8]}")

        # 1 trade/day stats
        d = conn.execute(text("""
            WITH daily AS (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY signal_time::date ORDER BY ml_proba DESC
                    ) as rn
                FROM ml_backtest_trades
            )
            SELECT COUNT(*)::int,
                ROUND(COUNT(*) FILTER (WHERE status='won')::numeric
                      / GREATEST(COUNT(*),1) * 100, 1),
                ROUND(NULLIF(SUM(pnl_pct) FILTER (WHERE pnl_pct>0), 0)::numeric
                      / ABS(NULLIF(SUM(pnl_pct) FILTER (WHERE pnl_pct<=0), 0))::numeric, 2),
                ROUND(AVG(pnl_pct)::numeric * 100, 2),
                ROUND(SUM(pnl_pct)::numeric * 100, 0),
                COUNT(*) FILTER (WHERE exit_reason='tp_hit')::int,
                COUNT(*) FILTER (WHERE exit_reason='sl_hit')::int,
                COUNT(*) FILTER (WHERE exit_reason='time_exit')::int,
                COUNT(*) FILTER (WHERE exit_reason='trail_hit')::int
            FROM daily WHERE rn = 1
        """)).fetchone()
        logger.info(f"\n1 TRADE/DAY:")
        logger.info(f"Trades: {d[0]}")
        logger.info(f"Win Rate: {d[1]}%")
        logger.info(f"Profit Factor: {float(d[2]):.2f}" if d[2] else "Profit Factor: N/A")
        logger.info(f"Avg PnL: {float(d[3]):+.2f}%")
        logger.info(f"Total PnL: {float(d[4]):+.0f}%")
        logger.info(f"Exits: TP={d[5]}, SL={d[6]}, Trail={d[8]}, Time={d[7]}")
        if d[0] > 0 and d[4]:
            logger.info(f"Monthly: {float(d[4])/37:+.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    engine = create_engine(DB_URL)
    model = joblib.load("models/bigmover_classifier.joblib")

    picks = load_signals(engine, model)
    symbols = picks['symbol'].unique().tolist()
    bar_cache = preload_bars(symbols)
    trades = attach_bar_arrays(picks, bar_cache)
    logger.info(f"Ready: {len(trades)} trades")

    top_per_layer = sweep_phase1(trades)
    best_layers, best_stats = sweep_phase2(trades, top_per_layer)
    run_and_save(trades, best_layers, engine)
