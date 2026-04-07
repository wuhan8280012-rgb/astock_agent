# Trend Initiation Strategy (趋势启动策略)

## Thesis

Buy stocks emerging from a **consolidation / shakeout phase** (振仓) where
the transition coefficient is **accelerating** and momentum is picking up —
but the stock has **not yet topped**.

The core insight: the best entry is NOT when a stock is already in a strong
trend (that's where the old RS+Angle strategy bought and failed), but at the
**initiation point** — when the trend is just starting after a quiet period.

## Why the Previous Strategy (RS+Angle v2) Failed

| Problem | Detail |
|---------|--------|
| 3 stacked hard filters | coef≥0 AND RS>1.0 AND angle>0 shrank pool to ~30 stocks |
| Emergency exit whipsaw | coef<-0.5 triggered 176 times (18.3% of trades) — systematic buy-high sell-low |
| Correlated factors | All 3 filters measured the same dimension (trend) |
| 10d rebalance + 1.3× buffer | Extremely high churn, ~78% cumulative trading cost |
| Result | -64.86% total, Sharpe -0.74, identical on static & PIT data |

## New Design: Trend Initiation

### Key Changes from RS+Angle v2

| Aspect | RS+Angle v2 (failed) | Trend Initiation |
|--------|----------------------|------------------|
| Hard filters | 3 correlated trend filters | **None** on signal dimensions |
| Emergency exit | coef < -0.5 daily check | **Removed** |
| Core signal | RS + coef level (already in trend) | **coef delta** (trend starting) |
| Factor diversity | All trend-correlated | Mixed: trend + volatility + headroom |
| Holdings | 10 concentrated | 15 diversified |
| Rebalance | 10d (biweekly) | 20d (monthly) |
| Buffer | 1.3× (Top 13) | 1.5× (Top 22) |
| Max single | 12% | 8% |
| RS dependency | Yes (industry calc) | **No** — simpler, larger pool |

### Factor Design

| Factor | Weight | Signal | Direction | Captures |
|--------|--------|--------|-----------|----------|
| **coef_delta_5d** | 30% | `coef_today - coef_5d_ago` | Higher = better | coef 加速 — trend initiation |
| **vol_contraction** | 25% | `vol_10d / vol_60d` | Lower = better | 振仓后压缩 — prior consolidation |
| **mom_accel** | 25% | `ret_20d - ret_prev_20d` | Higher = better | 动量加速 — momentum speeding up |
| **headroom** | 20% | `close / max_60d` | Lower = better | 未到顶 — room to run |

### How the Factors Work Together

```
1. vol_contraction < 1   →  Stock was volatile, now quiet (consolidation complete)
2. coef_delta > 0        →  MA20 angle is accelerating (trend starting)
3. mom_accel > 0         →  Recent returns > prior returns (momentum building)
4. headroom < 1          →  Below 60d high (hasn't topped yet)

Together: "quiet stock waking up with accelerating trend, still early"
```

### Transition Coefficient Recap

```python
base = tanh(a0 / 10.0)      # Current angle strength
turn = tanh((a0 - a1) / 5.0) # Angle acceleration
coef = 0.7 * base + 0.3 * turn
```

The **delta** (coef_today - coef_5d_ago) captures stocks where this
composite is **rising** — the trend is initiating or accelerating.

## Portfolio Rules

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Holdings | 15 | More diversified — reduce single-stock risk |
| Max single weight | 8% | Less concentrated than v2's 12% |
| Rebalance | Monthly (20d) | Half the trading cost of v2's 10d |
| Buffer band | 1.5× (Top 22) | Less churn than v2's 1.3× |
| Trend filter | CSI1000 > 200d MA | Reduce equity in bear markets |
| Range position | 50% max | Less aggressive than v2's 30% |
| Bear position | 0% | Full cash in deep bear |
| Stop loss | -20% individual | Wider — give trend time to develop |
| Slippage | 0.2% | Lower — less concentrated buying |
| Min 20d avg amount | 2亿 | Slightly lower bar than v2's 3亿 |

## Entry / Hold / Exit

```
Entry:  Top 15 composite rank (no hard signal filters)
Hold:   Stay in Top 22 buffer
Exit:   Rank drops below buffer (Top 22)
        OR  individual stop loss -20%
        OR  trend filter off (reduce to 50% → 0%)
```

No emergency exit — the old mechanism was the #1 source of losses.

## Files

| File | Purpose |
|------|---------|
| `run_backtest.py` | Backtest entry point; `TrendInitBacktest._score_universe()` |
| `README.md` | This file |

## Usage

```bash
# Static sample (development only — has survivorship bias)
python strategies/rs_angle_strategy/run_backtest.py --dataset csi1000_5y

# PIT sample (ground truth)
python strategies/rs_angle_strategy/run_backtest.py --dataset csi1000_5y_pit
```
