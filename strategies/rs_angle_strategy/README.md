# RS + MA20 Angle Strategy (龙头策略 v2)

## Thesis

Combine **industry-relative strength** (RS) with **MA20 angle transition
coefficient** to identify leaders that are *accelerating* their outperformance.

The two core indicators answer complementary questions:
- **RS vs industry**: "Who is the leader?" (cross-sectional ranking)
- **Transition coefficient**: "Is the leadership strengthening?" (time-series signal)

Their intersection — stocks with RS > 1.0 **and** rising transition_coef — captures
leaders entering their acceleration phase, the most valuable signal for entry.

## Difference from Leader Strategy v1

| Aspect | Leader v1 | RS + Angle v2 |
|--------|-----------|---------------|
| Primary signal | 5-factor blend (mom/RS/absorption/vol) | RS + transition_coef (70% combined) |
| Hard filters | Basic filters only | transition_coef ≥ 0, RS > 1.0, MA20 angle > 0 |
| Emergency exit | None | coef < -0.5 = immediate sell |
| Rebalance | Monthly (20d) | Biweekly (10d) |
| Buffer band | 1.5× | 1.3× (tighter) |
| Absorption factor | 20% weight | Removed — RS + angle captures the same signal |

## Factor Design

### Hard Filters (all must pass)

| Filter | Threshold | Purpose |
|--------|-----------|---------|
| transition_coef ≥ 0 | 0.0 | Only stocks with strengthening MA20 trends |
| RS vs industry > 1.0 | 1.0 | Must outperform its sector |
| ma20_angle_deg > 0 | 0° | MA20 must be sloping upward |

### Ranking Factors

| Factor | Weight | Direction | Why |
|--------|--------|-----------|-----|
| **RS vs industry** | 40% | Higher = better | Core leader definition |
| **transition_coef** | 30% | Higher = better | Trend acceleration — the angle is *getting steeper* |
| **RS new-high (120d)** | 15% | 1/0 binary | Breakout confirmation |
| **60d momentum** | 15% | Higher = better | Absolute return baseline |

### Transition Coefficient Deep Dive

```python
base = tanh(a0 / 10.0)      # Current angle strength
turn = tanh((a0 - a1) / 5.0) # Angle acceleration
coef = 0.7 * base + 0.3 * turn
```

| State | base | turn | coef | Trading meaning |
|-------|------|------|------|-----------------|
| **Launch acceleration** | small+ / 0 | large+ | 0.1~0.5 | Best entry — angle flipping from 0° to positive |
| **Sustained strength** | large+ | ~0 | 0.5~1.0 | Hold position |
| **Topping deceleration** | large+ | negative | Declining | Angle still positive but shrinking — caution |
| **Breakdown** | negative | large- | -0.5~-1.0 | Emergency exit trigger |

## Portfolio Rules

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Holdings | 10 | Leaders are few; concentrate |
| Max single weight | 12% | Give true leaders room |
| Rebalance | Biweekly (10d) | Leader rotation is faster than sector rotation |
| Buffer band | 1.3× (Top 13) | Tighter than v1; reduce stale holdings |
| Trend filter | CSI1000 > 200d MA | No leaders in bear markets |
| Bear position | 30% max | Aggressive reduction |
| Stop loss | -15% individual | Hard backstop |
| Emergency exit | coef < -0.5 | Trend collapse — sell immediately, don't wait for rebalance |
| Slippage | 0.3% | Concentrated buying premium |
| Min 20d avg amount | 3亿 | Leaders must be liquid |

## Entry / Hold / Exit

```
Entry:  RS > 1.0  AND  transition_coef >= 0  AND  ma20_angle > 0°  AND  Top 10 rank
Hold:   Stay in Top 13 buffer  AND  coef >= -0.5
Exit:   Rank drops below buffer (Top 13)
        OR  transition_coef < -0.5 (emergency — don't wait for rebalance)
        OR  individual stop loss -15%
        OR  trend filter off (reduce to 30%)
```

## Risk Notes

1. **Must use PIT data** — F strategy lesson: static +269% → PIT +16.57% (F strategy results)
2. **transition_coef look-ahead** — MA20 uses 20 days of data; ensure coef
   only uses T-1 closes in backtest
3. **Threshold sensitivity** — Recommend testing coef ≥ {-0.2, -0.1, 0, 0.1, 0.2}
4. **10d rebalance cost** — More trades than 20d; 0.3% slippage accounts for this

## Files

| File | Purpose |
|------|---------|
| `run_backtest.py` | Backtest entry point; `RSAngleBacktest._score_universe()` + emergency exit `run()` |
| `README.md` | This file |

## Usage

```bash
# Static sample (development only — has survivorship bias)
python strategies/rs_angle_strategy/run_backtest.py --dataset csi1000_5y

# PIT sample (ground truth)
python strategies/rs_angle_strategy/run_backtest.py --dataset csi1000_5y_pit
```
