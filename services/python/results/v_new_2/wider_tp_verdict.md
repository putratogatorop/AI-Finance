# v_new_2 — Wider TP Sweep (2026-05-05)

U4 universe (memes + AI + top-50). 5× leverage, max_concurrent=5, 1% risk.
BGM threshold 0.6. All variants use honest funding+slippage.

| variant | exit | pl% | filtered_n | H2 compd | H2 DD | Q1 compd | Q1 DD | Q1 CAGR |
|---|---|---|---|---|---|---|---|---|
| V0_baseline | trail_default | — | 12499 | +16.0% | -7.9% | +12.8% | -5.0% | +63%/yr |
| V1_pl_10 | trail_default | 10% | 4509 | +17.0% | -7.2% | -6.9% | -7.3% | -25%/yr |
| V2_pl_15 | trail_default | 15% | 6056 | +25.1% | -13.6% | -2.9% | -4.5% | -11%/yr |
| V3_pl_20 | trail_default | 20% | 7432 | +15.1% | -5.3% | +3.0% | -4.2% | +13%/yr |
| V4_pl_30 | trail_default | 30% | 9255 | +13.8% | -7.8% | +1.2% | -5.3% | +5%/yr |
| V5_trend_only | trend_only | — | 1927 | +29.1% | -4.6% | -7.6% | -7.0% | -28%/yr |
| V6_trail_5x | trail_5 | 30% | 5428 | +34.7% | -5.1% | -5.5% | -7.4% | -20%/yr |
| V7_trail_8x | trail_8 | 30% | 3704 | +13.5% | -9.1% | -11.4% | -11.1% | -39%/yr |
| V8_5x_pl_15 | trail_5 | 15% | 3720 | +33.0% | -15.0% | -14.2% | -14.1% | -46%/yr |

## Best by Q1 CAGR
- **V0_baseline**: Q1 CAGR +63%/yr, DD -5.0%
- **V3_pl_20**: Q1 CAGR +13%/yr, DD -4.2%
- **V4_pl_30**: Q1 CAGR +5%/yr, DD -5.3%
