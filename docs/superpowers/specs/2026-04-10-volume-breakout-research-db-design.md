# Volume Breakout Research Database — Design Spec

Date: 2026-04-10
Status: Draft

## Purpose

Build a research dataset that captures every significant volume spike (2-4x) across 200+ coins on 3 timeframes, and records what actually happens to price afterward. This is pure observation — no trading logic, no strategy assumptions. The dataset becomes the foundation for ML to find real patterns.

## 1. Database Table

```sql
CREATE TABLE volume_breakouts (
    id              SERIAL PRIMARY KEY,

    -- Event identification
    symbol          VARCHAR(20) NOT NULL,       -- e.g. "ETHUSDT"
    timeframe       VARCHAR(4) NOT NULL,        -- "15m", "1h", "4h"
    signal_time     TIMESTAMPTZ NOT NULL,       -- when the volume spike happened

    -- Breakout bar data
    open            DOUBLE PRECISION NOT NULL,
    high            DOUBLE PRECISION NOT NULL,
    low             DOUBLE PRECISION NOT NULL,
    close           DOUBLE PRECISION NOT NULL,
    volume          DOUBLE PRECISION NOT NULL,
    quote_volume    DOUBLE PRECISION NOT NULL,
    trades          INTEGER,

    -- Volume context
    vol_ratio       DOUBLE PRECISION NOT NULL,  -- volume / 20-period avg (the 2-4x spike)
    vol_avg_20      DOUBLE PRECISION NOT NULL,  -- 20-period avg volume for reference

    -- Price context at signal time
    price_change_1bar   DOUBLE PRECISION,       -- single bar return (close-to-close)
    price_change_2bar   DOUBLE PRECISION,       -- 2-bar return
    atr_14              DOUBLE PRECISION,       -- 14-period ATR (as fraction of price)
    bar_range_pct       DOUBLE PRECISION,       -- (high - low) / close
    upper_wick_pct      DOUBLE PRECISION,       -- upper wick as % of bar range
    lower_wick_pct      DOUBLE PRECISION,       -- lower wick as % of bar range
    body_pct            DOUBLE PRECISION,       -- body as % of bar range
    direction           SMALLINT NOT NULL,      -- +1 = up bar, -1 = down bar

    -- Wider context at signal time
    dist_from_20_high   DOUBLE PRECISION,       -- % distance from 20-period high
    dist_from_20_low    DOUBLE PRECISION,       -- % distance from 20-period low
    price_vs_ema_50     DOUBLE PRECISION,       -- close / 50-period EMA - 1
    rsi_14              DOUBLE PRECISION,       -- 14-period RSI
    volatility_20       DOUBLE PRECISION,       -- 20-period stdev of returns

    -- BTC context at signal time
    btc_price           DOUBLE PRECISION,
    btc_ret_24bar       DOUBLE PRECISION,       -- BTC return over 24 bars
    btc_ret_96bar       DOUBLE PRECISION,       -- BTC return over 96 bars (~daily on 15m)
    btc_vol_ratio       DOUBLE PRECISION,       -- BTC volume ratio at same time

    -- Forward returns (what actually happened)
    fwd_ret_12h         DOUBLE PRECISION,       -- return 12h after signal
    fwd_ret_24h         DOUBLE PRECISION,       -- return 24h after signal
    fwd_ret_48h         DOUBLE PRECISION,       -- return 48h after signal
    fwd_ret_1w          DOUBLE PRECISION,       -- return 1 week after signal
    fwd_ret_2w          DOUBLE PRECISION,       -- return 2 weeks after signal
    fwd_ret_1m          DOUBLE PRECISION,       -- return 1 month after signal

    -- Forward price extremes (how far did it go before reversing?)
    fwd_max_gain_12h    DOUBLE PRECISION,       -- max favorable excursion in 12h
    fwd_max_gain_24h    DOUBLE PRECISION,
    fwd_max_gain_48h    DOUBLE PRECISION,
    fwd_max_gain_1w     DOUBLE PRECISION,
    fwd_max_loss_12h    DOUBLE PRECISION,       -- max adverse excursion in 12h
    fwd_max_loss_24h    DOUBLE PRECISION,
    fwd_max_loss_48h    DOUBLE PRECISION,
    fwd_max_loss_1w     DOUBLE PRECISION,

    created_at      TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(symbol, timeframe, signal_time)
);

CREATE INDEX idx_vb_symbol ON volume_breakouts(symbol);
CREATE INDEX idx_vb_timeframe ON volume_breakouts(timeframe);
CREATE INDEX idx_vb_signal_time ON volume_breakouts(signal_time);
CREATE INDEX idx_vb_vol_ratio ON volume_breakouts(vol_ratio);
```

### Why These Columns

**Breakout bar data**: The raw candle. Lets ML learn candle shape patterns (big body = conviction, big wick = rejection).

**Volume context**: `vol_ratio` is the core filter (2-4x). `vol_avg_20` lets us normalize across coins.

**Price context**: Where in its range is this coin? Near highs = continuation potential. Near lows = possible reversal trap. ATR tells us if this move is big relative to normal volatility.

**BTC context**: Crypto is correlated. BTC trend at the moment of breakout matters.

**Forward returns**: The labels. Close-to-close at each horizon. No strategy assumptions — just "what happened."

**Forward extremes (MFE/MAE)**: Critical for trade management research. If a breakout typically hits +8% before pulling back to +3% at 48h, we know the optimal exit is somewhere in between. This is data the current scanner never captured.

## 2. Timeframe Derivation

We only have 15m bars in CSVs. Derive 1h and 4h by resampling:

```python
# 1h bars from 15m
df_1h = df_15m.resample("1h", on="open_time").agg({
    "open": "first", "high": "max", "low": "min", "close": "last",
    "volume": "sum", "quote_volume": "sum", "trades": "sum"
}).dropna()

# 4h bars from 15m
df_4h = df_15m.resample("4h", on="open_time").agg({...same...}).dropna()
```

## 3. Backfill Script

`services/python/scripts/backfill_volume_breakouts.py`

### Logic

```
For each of 198 coins:
    Load all 15m CSVs → DataFrame
    Resample to 1h, 4h
    For each timeframe (15m, 1h, 4h):
        Compute 20-period rolling volume average
        Compute ATR(14), EMA(50), RSI(14), volatility(20)
        For each bar where volume >= 2.0 * vol_avg_20:
            Record breakout bar data + context features
            Look ahead to compute forward returns at each horizon
            Look ahead to compute MFE/MAE at each horizon
            Insert into volume_breakouts table
    Load BTC data for same period → join btc context by nearest timestamp
```

### Forward Return Horizon Mapping (in bars)

| Horizon | 15m bars | 1h bars | 4h bars |
|---------|----------|---------|---------|
| 12h     | 48       | 12      | 3       |
| 24h     | 96       | 24      | 6       |
| 48h     | 192      | 48      | 12      |
| 1 week  | 672      | 168     | 42      |
| 2 weeks | 1344     | 336     | 84      |
| 1 month | 2880     | 720     | 180     |

For forward extremes (MFE/MAE), scan all bars within the horizon window:
```python
# MFE (max favorable excursion) for a long signal
fwd_max_gain_24h = max((high[i+1:i+97] - close[i]) / close[i])
# MAE (max adverse excursion) for a long signal
fwd_max_loss_24h = min((low[i+1:i+97] - close[i]) / close[i])
```

For short-direction signals, flip the signs.

### Volume Ratio Threshold

Use `vol_ratio >= 2.0` (not 4.0 like the old scanner). This captures a wider range:
- 2-3x: moderate spikes — potentially early signals before the big move
- 3-4x: strong spikes — the current scanner's territory
- 4x+: extreme spikes — possibly climax/exhaustion bars

Let the data show which range leads to follow-through. Don't filter by assumption.

### Estimated Row Count

With 198 coins, ~3 years of data, vol_ratio >= 2.0:
- 15m: ~100k-200k events (many moderate spikes)
- 1h: ~30k-60k events
- 4h: ~10k-20k events
- Total: ~150k-280k rows

This is a rich dataset for ML.

### Performance

Processing 198 coins x 3 timeframes sequentially could take 30-60 minutes. Acceptable for a one-time backfill. Use batch inserts (1000 rows at a time) to avoid DB bottleneck.

### Cooldown

Apply a cooldown per (symbol, timeframe): skip if another breakout was detected within the last N bars:
- 15m: 8 bars (2h cooldown)
- 1h: 4 bars (4h cooldown)  
- 4h: 2 bars (8h cooldown)

This prevents counting the same move multiple times.

## 4. Prisma Schema Addition

```prisma
model VolumeBreakout {
  id              Int       @id @default(autoincrement())
  symbol          String    @db.VarChar(20)
  timeframe       String    @db.VarChar(4)
  signal_time     DateTime  @db.Timestamptz()

  open            Float
  high            Float
  low             Float
  close           Float
  volume          Float
  quote_volume    Float
  trades          Int?

  vol_ratio       Float
  vol_avg_20      Float

  price_change_1bar   Float?
  price_change_2bar   Float?
  atr_14              Float?
  bar_range_pct       Float?
  upper_wick_pct      Float?
  lower_wick_pct      Float?
  body_pct            Float?
  direction           Int     @db.SmallInt

  dist_from_20_high   Float?
  dist_from_20_low    Float?
  price_vs_ema_50     Float?
  rsi_14              Float?
  volatility_20       Float?

  btc_price           Float?
  btc_ret_24bar       Float?
  btc_ret_96bar       Float?
  btc_vol_ratio       Float?

  fwd_ret_12h         Float?
  fwd_ret_24h         Float?
  fwd_ret_48h         Float?
  fwd_ret_1w          Float?
  fwd_ret_2w          Float?
  fwd_ret_1m          Float?

  fwd_max_gain_12h    Float?
  fwd_max_gain_24h    Float?
  fwd_max_gain_48h    Float?
  fwd_max_gain_1w     Float?
  fwd_max_loss_12h    Float?
  fwd_max_loss_24h    Float?
  fwd_max_loss_48h    Float?
  fwd_max_loss_1w     Float?

  created_at      DateTime  @default(now()) @db.Timestamptz()

  @@unique([symbol, timeframe, signal_time])
  @@index([symbol])
  @@index([timeframe])
  @@index([signal_time])
  @@index([vol_ratio])
  @@map("volume_breakouts")
}
```

## 5. SQLAlchemy Model

Add to `services/python/src/db/models.py`:

```python
class VolumeBreakout(Base):
    __tablename__ = "volume_breakouts"
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(4), nullable=False)
    signal_time = Column(DateTime(timezone=True), nullable=False)
    # ... all columns matching Prisma schema
    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "signal_time"),
    )
```

## 6. What This Enables (Next Phase — Not In Scope)

Once the table is populated:

1. **Exploratory analysis**: "What % of 3x volume spikes on 4h lead to +10% in 48h?" Pure SQL queries.
2. **Feature-rich ML**: Train on 150k+ labeled events instead of 1673 signals. The model has real data about outcomes.
3. **Optimal horizon discovery**: Maybe 48h trades are wrong. Maybe 1-week holds after 4h breakouts are where the edge is.
4. **MFE/MAE trade management**: If typical MFE at 24h is +8% but close-to-close return is +3%, we know there's +5% being left on the table with wrong exits.
5. **Cross-timeframe confirmation**: Does a 15m spike that's also a 1h spike have better follow-through?
6. **Coin clustering**: Group coins by breakout behavior — some always follow through, some always fade.

## 7. Success Criteria

- Table populated with 100k+ rows across all 3 timeframes
- Every row has forward returns at all 6 horizons (12h through 1 month)
- Every row has MFE/MAE at 4 horizons (12h through 1 week)
- Backfill completes in under 60 minutes
- Basic sanity check: avg forward return distribution looks reasonable (centered near 0, fat tails)
