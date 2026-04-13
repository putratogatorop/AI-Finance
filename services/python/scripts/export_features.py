"""Export features + targets to Parquet for Colab training.

Outputs: data/features/tournament_data.parquet
Contains: 200 tokens × 4 years × 28 features + target + metadata
Upload this single file to Google Colab for ML tournament.
"""

import glob
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("C:/Users/togat/Desktop/AI-Finance/data/raw/4h")
OUT_DIR = Path("C:/Users/togat/Desktop/AI-Finance/data/features")
TARGET_HORIZON = 30  # 5 days in 4h candles
TARGET_THRESHOLD = 0.02  # 2%


def build_features_fast(df):
    """Vectorized feature engineering — no loops."""
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    out = pd.DataFrame(index=df.index)

    for p in [6, 12, 24, 42, 84, 168]:
        out[f"ret_{p}"] = np.log(close / close.shift(p))

    for p in [20, 50, 100, 200]:
        out[f"p2sma_{p}"] = close / close.rolling(p).mean() - 1

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["ema_ratio"] = ema12 / ema26 - 1

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta).where(delta < 0, 0.0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi"] = (100 - (100 / (1 + rs)) - 50) / 50

    out["macd_norm"] = ((ema12 - ema26) - (ema12 - ema26).ewm(9).mean()) / close

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    out["bb_width"] = (2 * bb_std) / bb_mid
    out["bb_pct"] = (close - (bb_mid - 2 * bb_std)) / (4 * bb_std)

    ret = close.pct_change()
    out["vol_6"] = ret.rolling(6).std()
    out["vol_24"] = ret.rolling(24).std()
    out["vol_72"] = ret.rolling(72).std()
    out["vol_ratio"] = out["vol_6"] / out["vol_72"].replace(0, np.nan)

    prev_c = close.shift(1)
    tr = pd.concat([high - low, (high - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
    out["atr_norm"] = tr.rolling(14).mean() / close

    out["vol_ratio_v"] = volume / volume.rolling(20).mean().replace(0, np.nan)
    out["clv"] = (2 * close - high - low) / (high - low).replace(0, np.nan)
    out["skew"] = ret.rolling(24).skew()
    out["drawdown"] = close / close.rolling(168).max() - 1
    out["above_sma50"] = (close > close.rolling(50).mean()).astype(float)
    out["above_sma200"] = (close > close.rolling(200).mean()).astype(float)

    if "open_time" in df.columns:
        ts = pd.to_datetime(df["open_time"], unit="ms")
        out["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
        out["dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)

    # Meta columns
    out["_close"] = close
    out["_atr"] = tr.rolling(14).mean()

    # Target
    forward_ret = close.shift(-TARGET_HORIZON) / close - 1
    out["_target"] = (forward_ret > TARGET_THRESHOLD).astype(float)
    out.loc[forward_ret.isna(), "_target"] = np.nan
    out["_forward_ret"] = forward_ret

    return out


def main():
    manifest = pd.read_csv(RAW_DIR / "manifest.csv")
    manifest = manifest.sort_values("months", ascending=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_data = []
    feature_cols = None

    t0 = time.time()
    for idx, (_, row) in enumerate(manifest.iterrows()):
        symbol = row["symbol"]
        asset = symbol.replace("USDT", "")

        files = sorted(glob.glob(str(RAW_DIR / symbol / "*.csv")))
        if not files:
            continue

        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        df = df[df["open_time"].between(1_600_000_000_000, 2_000_000_000_000)]
        df = df.drop_duplicates(subset="open_time").sort_values("open_time").reset_index(drop=True)

        if len(df) < 500:
            continue

        feat = build_features_fast(df)
        feat["_asset"] = asset
        feat["_timestamp"] = pd.to_datetime(df["open_time"], unit="ms")

        if feature_cols is None:
            feature_cols = [c for c in feat.columns if not c.startswith("_")]

        all_data.append(feat)

        if (idx + 1) % 50 == 0:
            logger.info(f"  Processed {idx + 1}/{len(manifest)} tokens...")

    combined = pd.concat(all_data, ignore_index=True)

    # Drop rows missing features or target
    valid = combined[feature_cols + ["_target"]].notna().all(axis=1)
    combined = combined[valid].reset_index(drop=True)

    logger.info(f"\nTotal: {len(combined):,} rows, {combined['_asset'].nunique()} tokens, {len(feature_cols)} features")
    logger.info(f"Target positive rate: {combined['_target'].mean():.1%}")

    # Save
    out_path = OUT_DIR / "tournament_data.parquet"
    combined.to_parquet(out_path, index=False)
    size_mb = out_path.stat().st_size / 1024 / 1024
    logger.info(f"\nSaved to {out_path} ({size_mb:.0f} MB)")
    logger.info(f"Features: {feature_cols}")
    logger.info(f"Time: {time.time() - t0:.0f}s")

    # Also save as CSV for Colab upload (Colab handles parquet too, but just in case)
    csv_path = OUT_DIR / "tournament_data.csv.gz"
    combined.to_csv(csv_path, index=False, compression="gzip")
    csv_size = csv_path.stat().st_size / 1024 / 1024
    logger.info(f"Also saved CSV.gz: {csv_path} ({csv_size:.0f} MB)")


if __name__ == "__main__":
    main()
