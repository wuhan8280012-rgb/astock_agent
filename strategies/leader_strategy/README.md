# Leader Strategy (龙头策略)

## Thesis

A-shares price on **expectations** ("炒预期, 卖事实"). By the time quarterly revenue
growth is published, the move is priced in. This strategy replaces lagging
fundamental screens with real-time capital-flow proxies to detect institutional
accumulation *before* earnings confirm.

## Factor Design

| Factor | Weight | Signal | Data Source |
|--------|--------|--------|-------------|
| **60d Momentum** | 30% | Trend continuation | daily close (existing) |
| **RS vs Industry** | 25% | Sector-relative strength >1.0 = true leader | sw_l1 industry index (existing) |
| **Absorption Score** | 20% | High turnover + flat/rising price = institutional accumulation | daily amount + pct_chg (existing) |
| **RS New High** | 10% | 60d RS at 120d high = accelerating leadership | computed from above |
| **Low Volatility** | 15% | Quality filter, reduces noise | daily returns (existing) |

### Why RS vs Industry, not absolute momentum?

Absolute 60d momentum selects "hot sector + any stock". Industry-relative RS selects
"strongest stock within each sector" — the actual definition of a leader. Combined with
60d absolute momentum, this captures leaders in *rising* sectors.

### Why absorption score?

`turnover_surge + price_not_falling` is the cheapest proxy for "someone is accumulating".
No new API calls needed — it uses existing daily `amount` and `pct_chg` fields.

## Portfolio Rules

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Holdings | 10 | Concentrated; leaders are few |
| Max single weight | 12% | Give true leaders room |
| Max industry | not constrained | Leaders cluster by nature |
| Rebalance | Monthly (20d) | Leader cycles are monthly, not weekly |
| Trend filter | CSI1000 > 200d MA | No leaders in bear markets |
| Bear position | 30% max | More aggressive reduction than F's 50% |
| Stop loss | -15% individual | Hard backstop |
| Slippage | 0.3% | Higher than F's 0.2% for concentrated buying |
| Min 20d avg amount | 3亿 | Tighter than F's 1亿; leaders must be liquid |

## Entry vs Hold vs Exit

```
Entry:  RS vs industry > 1.0  AND  absorption_score > 0  AND  60d momentum positive
Hold:   Revenue confirms (future: check quarterly TTM growth after earnings)
Exit:   RS rank drops below buffer  OR  individual stop loss -15%  OR  trend filter off
```

## Experiment Plan

1. **Phase 0** (this PR): Pure price-volume leader factors on existing data bundle.
   No new API calls. Validate whether RS + absorption has IC > 0 on PIT sample.
2. **Phase 1**: Add `moneyflow` API (already defined in `BATCH_CAPABLE_APIS`) for
   main-player net inflow — a richer accumulation signal.
3. **Phase 2**: Add `margin_detail` API for leverage-money tracking.
4. **Phase 3**: Quarterly revenue TTM growth as hold validation filter.

All experiments run on PIT constituent data — the non-negotiable lesson from
F strategy failure (static +269% → PIT +16.57%).

## Files

| File | Purpose |
|------|---------|
| `leader_factors.py` | Factor calculation functions (absorption, RS, composite) |
| `run_backtest.py` | Backtest entry point; `LeaderBacktest._score_universe()` override |

## Usage

```bash
# Static sample (has survivorship bias — use for development only)
python strategies/leader_strategy/run_backtest.py --dataset csi1000_5y

# PIT sample (ground truth)
python strategies/leader_strategy/run_backtest.py --dataset csi1000_5y_pit
```
