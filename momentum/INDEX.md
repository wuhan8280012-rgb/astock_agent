# Momentum Rotation Strategy - Quick Index

## Getting Started

1. **First Time?** → Start with [README.md](README.md)
2. **Ready to Deploy?** → Follow [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)
3. **Want Examples?** → Run [example_usage.py](example_usage.py)
4. **Need Details?** → Check [MANIFEST.md](MANIFEST.md)

## Core Components

| File | Purpose | Size | Key Class/Function |
|------|---------|------|-------------------|
| [config.py](config.py) | Strategy configuration | 6.8 KB | `MomentumConfig` |
| [universe.py](universe.py) | Stock filtering | 9.4 KB | `filter_universe()` |
| [calculator.py](calculator.py) | Momentum scoring | 7.7 KB | `rank_by_momentum()` |
| [rebalancer.py](rebalancer.py) | Trade computation | 7.5 KB | `compute_rebalance()` |
| [risk_control.py](risk_control.py) | Risk management | 8.5 KB | `check_regime_filter()` |
| [engine.py](engine.py) | Main orchestrator | 16.9 KB | `MomentumEngine` |

## Key Classes

### MomentumConfig (config.py)
The "genome" for strategy evolution. Contains all 25+ tunable parameters.

```python
config = MomentumConfig()
evolved = config.mutate({"top_n": 15})
config.save(Path("config_v1.0.0.json"))
```

### MomentumEngine (engine.py)
Main orchestrator for weekly and daily operations.

```python
engine = MomentumEngine(config, client)
weekly = engine.run_weekly_rebalance(as_of, conn)
daily = engine.run_daily_monitor(as_of, conn)
```

## Key Functions

| Function | Module | Purpose |
|----------|--------|---------|
| `filter_universe()` | universe.py | Filter A-shares by market cap, turnover, ST status |
| `rank_by_momentum()` | calculator.py | Calculate and rank by momentum |
| `compute_rebalance()` | rebalancer.py | Determine trades (buy/sell/hold) |
| `check_regime_filter()` | risk_control.py | Market regime classification |
| `apply_stop_loss()` | risk_control.py | Check trailing stop loss |
| `check_drawdown()` | risk_control.py | Circuit breaker check |

## Strategy Pipeline

### Weekly (Friday 3:00 PM)
```
Filter Universe (500+ stocks)
  ↓
Calculate Momentum (multi-horizon)
  ↓
Rank by Score
  ↓
Check Regime (HALT/DEFENSIVE/RUN)
  ↓
Compute Rebalance (identify trades)
  ↓
Apply Position Limits
  ↓
Execute Trades & Record Audit Trail
```

### Daily (4:00 PM)
```
Check Regime Status
  ↓
Check Stop Losses (-8%)
  ↓
Calculate Drawdown
  ↓
Circuit Breaker Check (-20%)
  ↓
Record Monitoring Data
```

## Configuration Parameters

**Quick Reference**

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `top_n` | 10 | 1-50 | Holdings target |
| `momentum_type` | composite | {simple, risk_adjusted, composite} | Calculation method |
| `volatility_penalty` | 0.3 | 0.0-1.0 | High-vol stock penalty |
| `stop_loss_pct` | -0.08 | -0.2 to 0 | Trailing stop loss |
| `universe_min_market_cap` | 5.0 | 1.0+ | Billions CNY |
| `universe_min_avg_turnover` | 20.0 | 5.0+ | Millions CNY |
| `rebalance_threshold` | 0.3 | 0.0-1.0 | Rank change to trigger |

See [config.py](config.py) for all 25+ parameters.

## Database Tables

Automatically created by engine:

```
momentum_rebalance_log
  - as_of, config_version, universe_size, regime
  - buys_count, sells_count, holds_count
  - turnover_pct, estimated_cost
  - rebalance_required, top_5_codes

momentum_daily_monitor
  - as_of, config_version
  - portfolio_value, peak_value, drawdown_pct
  - stopped_out, regime
```

## Quick Examples

### 1. Create Config
```python
from momentum import MomentumConfig
config = MomentumConfig(top_n=10, momentum_type="composite")
config.save(Path("config_v1.0.0.json"))
```

### 2. Mutate for Evolution
```python
evolved = config.mutate({
    "top_n": 15,
    "volatility_penalty": 0.5,
    "description": "Evolved variant A"
})
```

### 3. Run Strategy
```python
from momentum import MomentumEngine
engine = MomentumEngine(config, client)
result = engine.run_weekly_rebalance("20240315", conn)
print(f"Buys: {len(result.buys)}, Sells: {len(result.sells)}")
```

### 4. Query Results
```python
# Last 10 rebalances
cursor = conn.execute("""
    SELECT as_of, regime, buys_count, sells_count, turnover_pct
    FROM momentum_rebalance_log
    ORDER BY created_at DESC LIMIT 10
""")
```

## Common Tasks

### Check Universe Size
```python
from momentum.universe import filter_universe
universe = filter_universe(client, "20240315", config)
print(f"Investable: {len(universe)} stocks")
```

### Check Momentum Scores
```python
from momentum.calculator import rank_by_momentum
ranked = rank_by_momentum(universe, client, "20240315", config)
print(ranked.head(10)[["ts_code", "momentum_score"]])
```

### Check Market Regime
```python
from momentum.risk_control import check_regime_filter
regime = check_regime_filter(client, "20240315", config)
print(f"Regime: {regime.regime}, Cash%: {regime.cash_pct*100:.0f}%")
```

### Schedule Tasks
```python
from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()
scheduler.add_job(weekly_rebalance_task, "cron", day_of_week=4, hour=15)
scheduler.add_job(daily_monitor_task, "cron", hour=16)
scheduler.start()
```

## Files Overview

```
momentum/
├── Core Modules (7 files, ~65 KB)
│   ├── __init__.py              Package init
│   ├── config.py                Configuration genome
│   ├── universe.py              Stock filtering
│   ├── calculator.py            Momentum scoring
│   ├── rebalancer.py            Trade computation
│   ├── risk_control.py          Risk management
│   └── engine.py                Main orchestrator
│
├── Documentation (3 files, ~40 KB)
│   ├── README.md                Full documentation
│   ├── INTEGRATION_GUIDE.md      Integration instructions
│   └── MANIFEST.md              File inventory
│
├── Examples
│   └── example_usage.py          Working examples
│
└── This Index
    └── INDEX.md                 Quick reference
```

## Deployment Checklist

- [ ] Copy momentum/ directory
- [ ] Create momentum_configs/ directory
- [ ] Save baseline config
- [ ] Add scheduler jobs
- [ ] Run example_usage.py
- [ ] Run backtest
- [ ] Integrate with execution system
- [ ] Set up monitoring queries
- [ ] Validate risk controls

## Documentation Map

```
You are here: INDEX.md (Quick reference)
    ↓
For setup/usage: README.md (Complete API docs)
    ↓
For deployment: INTEGRATION_GUIDE.md (Deploy instructions)
    ↓
For details: MANIFEST.md (File inventory)
    ↓
For code: Read *.py files directly (full type hints + docstrings)
```

## Support

- **API Questions** → See [README.md](README.md) API sections
- **Integration Help** → See [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)
- **Code Examples** → Run [example_usage.py](example_usage.py) or see [README.md](README.md)
- **File Details** → See [MANIFEST.md](MANIFEST.md)
- **Troubleshooting** → See [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) Troubleshooting

## Key Statistics

- **Production Code**: ~65 KB (7 files)
- **Documentation**: ~40 KB (3 files)
- **Examples**: ~8 KB (1 file)
- **Total**: ~113 KB
- **All Syntax Valid**: ✓
- **All Fully Type Hinted**: ✓
- **All Fully Documented**: ✓
- **Production Ready**: ✓

## Next Step

1. Read [README.md](README.md) for complete documentation
2. Run `python example_usage.py` to see it in action
3. Follow [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) to deploy

---

**Status**: ✓ Complete and Production Ready
**Version**: 1.0.0
**Last Updated**: 2026-03-21
