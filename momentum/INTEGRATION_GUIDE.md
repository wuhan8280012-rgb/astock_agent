# Momentum Rotation Strategy - Integration Guide

This document explains how to integrate the momentum rotation strategy into the existing trading system.

## Quick Start

### 1. Import the Engine

```python
from momentum.engine import MomentumEngine
from momentum.config import MomentumConfig
from data_pipeline.tushare_client import TushareClient
import sqlite3

# Load or create config
config = MomentumConfig.load(Path("momentum_configs/active.json"))

# Create engine
client = TushareClient()
engine = MomentumEngine(config, client)
```

### 2. Schedule Weekly Rebalance (Friday 3:00 PM)

```python
from apscheduler.schedulers.background import BackgroundScheduler

def weekly_rebalance_task():
    config = MomentumConfig.load(Path("momentum_configs/active.json"))
    client = TushareClient()
    engine = MomentumEngine(config, client)

    from db.repository import get_db
    with get_db() as conn:
        result = engine.run_weekly_rebalance(datetime.now().strftime("%Y%m%d"), conn)

        if result.success:
            logger.info(f"Rebalance successful: {result.buys_count} buys, {result.sells_count} sells")
            # Execute trades based on result.buys, result.sells
        else:
            logger.error(f"Rebalance failed: {result.error_message}")

scheduler = BackgroundScheduler()
scheduler.add_job(
    weekly_rebalance_task,
    "cron",
    day_of_week=4,  # Friday
    hour=15,        # 3:00 PM
    minute=0,
    id="momentum_weekly_rebalance"
)
scheduler.start()
```

### 3. Schedule Daily Monitor (4:00 PM)

```python
def daily_monitor_task():
    config = MomentumConfig.load(Path("momentum_configs/active.json"))
    client = TushareClient()
    engine = MomentumEngine(config, client)

    from db.repository import get_db
    with get_db() as conn:
        result = engine.run_daily_monitor(datetime.now().strftime("%Y%m%d"), conn)

        if result.success:
            logger.info(f"Daily monitor: portfolio={result.portfolio_value:,.0f} CNY, drawdown={result.drawdown_pct:.2f}%")

            if result.stopped_out:
                logger.warning(f"Stopped out: {result.stopped_out}")
                # Execute stop-loss sells

            if result.circuit_breaker_triggered:
                logger.critical("Circuit breaker triggered, halting all trading")
                # Freeze all new positions
        else:
            logger.error(f"Daily monitor failed: {result.error_message}")

scheduler.add_job(
    daily_monitor_task,
    "cron",
    hour=16,        # 4:00 PM
    minute=0,
    id="momentum_daily_monitor"
)
```

## Architecture Integration

### 1. Data Flow

```
TushareClient (singleton)
    ↓
MomentumEngine
    ├─ filter_universe()       → Universe module
    ├─ rank_by_momentum()      → Calculator module
    ├─ check_regime_filter()   → Risk control module
    ├─ compute_rebalance()     → Rebalancer module
    └─ Database audit trail    → SQLite
```

### 2. Database Tables Created

The engine automatically creates these audit tables:

```sql
-- Weekly rebalance log
CREATE TABLE momentum_rebalance_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    as_of TEXT NOT NULL,
    config_version TEXT NOT NULL,
    universe_size INTEGER,
    top_n_count INTEGER,
    regime TEXT,
    buys_count INTEGER,
    sells_count INTEGER,
    holds_count INTEGER,
    turnover_pct REAL,
    estimated_cost REAL,
    rebalance_required BOOLEAN,
    top_5_codes TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Daily monitoring log
CREATE TABLE momentum_daily_monitor (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    as_of TEXT NOT NULL,
    config_version TEXT NOT NULL,
    portfolio_value REAL,
    peak_value REAL,
    drawdown_pct REAL,
    stopped_out TEXT,
    regime TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
```

### 3. Execution Flow

```
Weekly Rebalance (Friday 3:00 PM):
┌─────────────────────────────────────────────────────┐
│ 1. Filter universe                                  │
│    - Remove ST stocks                               │
│    - Remove new listings (<60 days)                │
│    - Remove low market cap (<5B CNY)               │
│    - Remove low turnover (<20M CNY)                │
│    Result: Investable universe (e.g., 500 stocks)  │
├─────────────────────────────────────────────────────┤
│ 2. Calculate momentum                               │
│    - Fetch 120 days of price history               │
│    - Calc simple momentum (20, 60, 120 day)        │
│    - Apply volatility penalty                      │
│    - Weight and combine                            │
│    Result: Momentum score per stock                │
├─────────────────────────────────────────────────────┤
│ 3. Rank & filter                                   │
│    - Rank by momentum score                        │
│    - Select top 10                                 │
│    Result: Target portfolio                        │
├─────────────────────────────────────────────────────┤
│ 4. Check regime                                    │
│    - Fetch CSI 300 breadth                         │
│    - Classify: HALT, DEFENSIVE, NEUTRAL, RUN       │
│    - Adjust position limits                        │
├─────────────────────────────────────────────────────┤
│ 5. Compute rebalance                               │
│    - Compare current holdings with target          │
│    - Identify: buys, sells, holds                  │
│    - Calculate turnover & estimated costs          │
│    Result: Trade list with confidence              │
├─────────────────────────────────────────────────────┤
│ 6. Apply position limits                           │
│    - Cap single position (max 15%)                 │
│    - Add cash if regime requires                   │
│    - Normalize weights                             │
│    Result: Final position weights                  │
├─────────────────────────────────────────────────────┤
│ 7. Execute trades                                  │
│    - (Integration with execution system)           │
│    - Track fills in database                       │
│    Result: Updated holdings                        │
└─────────────────────────────────────────────────────┘

Daily Monitor (4:00 PM):
┌─────────────────────────────────────────────────────┐
│ 1. Check regime                                    │
│    - Market regime status                          │
│    - Log for analysis                              │
├─────────────────────────────────────────────────────┤
│ 2. Apply stop losses                               │
│    - Check -8% trailing stop                       │
│    - Identify stopped-out positions                │
├─────────────────────────────────────────────────────┤
│ 3. Check drawdown                                  │
│    - Compare current vs. peak value                │
│    - Trigger circuit breaker if >20%               │
├─────────────────────────────────────────────────────┤
│ 4. Execute stop sales                              │
│    - (Integration with execution system)           │
│    Result: Liquidate losing positions              │
└─────────────────────────────────────────────────────┘
```

## Integration with Execution System

The momentum strategy produces trade lists but doesn't execute directly. Integration with your execution system:

```python
# After rebalance calculation
rebalance_result = engine.run_weekly_rebalance(as_of, conn)

# Execute buys
for buy in rebalance_result.buys:
    ts_code = buy["ts_code"]
    target_weight = buy["target_weight"]
    amount_to_buy = total_capital * target_weight
    execute_buy(ts_code, amount_to_buy)

# Execute sells
for sell in rebalance_result.sells:
    ts_code = sell["ts_code"]
    execute_sell(ts_code, quantity="all")  # Liquidate position

# Track fills
update_holdings_in_database(conn, ts_code, shares, entry_price, entry_date)
```

## Config Management

### Initial Setup

```python
# 1. Create baseline config
config = MomentumConfig(
    lookback_days=[20, 60, 120],
    lookback_weights=[0.5, 0.3, 0.2],
    universe_min_market_cap=5.0,
    universe_min_avg_turnover=20.0,
    top_n=10,
    momentum_type="composite",
    description="Initial baseline momentum rotation"
)

# 2. Save as baseline
config.save(Path("momentum_configs/baseline_v1.0.0.json"))

# 3. Create symlink for "active" config
# ln -s baseline_v1.0.0.json active.json

# 4. Run scheduler with active config
```

### Evolution Loop (for Opus AI)

```python
# 1. Load baseline
baseline = MomentumConfig.load(Path("momentum_configs/active.json"))

# 2. Generate mutations (Opus does this)
mutations = {
    "top_n": 12,
    "volatility_penalty": 0.4,
    "lookback_days": [25, 65, 125],
    "description": "Variant A: Adjusted horizons and vol penalty"
}

# 3. Create mutated config
variant = baseline.mutate(mutations)
variant.save(Path(f"momentum_configs/v{variant.version}.json"))

# 4. Backtest variant...
# If performance improves:
#   - Promote variant to active
#   - ln -sf v1.0.1.json active.json
#   - Restart scheduler
```

### Config Versioning

All configs are versioned and tracked:

```
momentum_configs/
├── baseline_v1.0.0.json         # Original baseline
├── v1.0.1.json                  # First evolution
├── v1.0.2.json                  # Second evolution
├── v1.1.0.json                  # Major revision
└── active.json -> v1.0.2.json   # Currently active

Promotion strategy:
baseline_v1.0.0 (baseline)
    ↓
v1.0.1 (Sharpe +2%, test period 4 weeks)
    ↓
PROMOTE if confirmed → active.json
```

## Monitoring & Analytics

### Query Recent Rebalances

```python
import sqlite3

conn = sqlite3.connect("investment.db")
conn.row_factory = sqlite3.Row

# Get last 10 rebalances
cursor = conn.execute("""
    SELECT as_of, config_version, universe_size, regime,
           buys_count, sells_count, turnover_pct
    FROM momentum_rebalance_log
    ORDER BY created_at DESC
    LIMIT 10
""")

for row in cursor:
    print(f"{row['as_of']}: {row['buys_count']} buys, {row['sells_count']} sells, "
          f"turnover={row['turnover_pct']:.1f}%, regime={row['regime']}")
```

### Track Performance by Config Version

```python
# Group by config version
cursor = conn.execute("""
    SELECT config_version, COUNT(*) as rebalance_count,
           AVG(turnover_pct) as avg_turnover,
           AVG(buys_count + sells_count) as avg_trades
    FROM momentum_rebalance_log
    GROUP BY config_version
    ORDER BY config_version DESC
""")

for row in cursor:
    print(f"Config {row['config_version']}: "
          f"{row['rebalance_count']} runs, "
          f"avg_turnover={row['avg_turnover']:.1f}%, "
          f"avg_trades={row['avg_trades']:.1f}")
```

### Drawdown Tracking

```python
# Get daily portfolio metrics
cursor = conn.execute("""
    SELECT as_of, portfolio_value, peak_value, drawdown_pct,
           CASE WHEN stopped_out != '' THEN 1 ELSE 0 END as had_stops
    FROM momentum_daily_monitor
    ORDER BY as_of DESC
    LIMIT 20
""")

for row in cursor:
    print(f"{row['as_of']}: value={row['portfolio_value']:,.0f}, "
          f"drawdown={row['drawdown_pct']:.1f}%, stops={row['had_stops']}")
```

## Error Handling

The engine is designed to fail gracefully:

```python
result = engine.run_weekly_rebalance(as_of, conn)

if not result.success:
    logger.error(f"Rebalance failed: {result.error_message}")
    # Don't execute trades
    # Alert team
    # Retry next week
else:
    # Proceed with execution
    for buy in result.buys:
        execute_trade(buy)
```

Common errors and handling:

| Error | Cause | Action |
|-------|-------|--------|
| No stocks passed universe filter | Market corruption or data issue | Log, alert, skip week |
| Failed to fetch prices | Tushare API down | Retry with backoff |
| No regime data | CSI 300 suspended | Default to NEUTRAL regime |
| Empty ranked universe | All stocks filtered out | Skip rebalance |

## Performance Optimization

### Data Caching

The system is designed to minimize API calls:

```python
# TushareClient caches:
# - Trade calendar (cached in memory)
# - Batch queries (30 stocks per call)
# - Rate limiting (singleton pattern)

# Momentum calculation:
# - Prices fetched once per run
# - Volatility computed once
# - Shared across all indicators
```

### Database Indexing

Create indexes for common queries:

```sql
CREATE INDEX idx_momentum_rebalance_date ON momentum_rebalance_log(as_of);
CREATE INDEX idx_momentum_rebalance_config ON momentum_rebalance_log(config_version);
CREATE INDEX idx_momentum_monitor_date ON momentum_daily_monitor(as_of);
```

## Troubleshooting

### Strategy not rebalancing

```python
# Check if scheduler is running
from apscheduler.schedulers.background import BackgroundScheduler
scheduler = BackgroundScheduler()
scheduler.print_jobs()  # See all registered jobs

# Check active config
active_config = MomentumConfig.load(Path("momentum_configs/active.json"))
print(active_config.summary())

# Check recent logs
tail -f logs/momentum.log
```

### Low momentum scores

```python
# Check universe size
ranked = rank_by_momentum(universe, client, as_of, config)
print(f"Ranked {len(ranked)} stocks")

# Examine top stocks
print(ranked.head(10)[["ts_code", "momentum_score"]])

# Verify price data
daily = client.daily(ts_code="000001.SZ", end_date=as_of)
print(f"Prices for 000001.SZ: {len(daily)} days")
```

### Too much turnover

```python
# Increase rebalance_threshold
config.rebalance_threshold = 0.5  # 50% rank change required

# Or reduce volatility penalty
config.volatility_penalty = 0.1

# Or use longer lookback periods
config.lookback_days = [30, 90, 180]  # Less reactive
```

## Next Steps

1. **Configure**: Set up initial config in `momentum_configs/`
2. **Schedule**: Add weekly rebalance and daily monitor jobs
3. **Backtest**: Run historical backtest to validate performance
4. **Execute**: Integrate with execution system to place actual trades
5. **Monitor**: Track performance metrics in database
6. **Evolve**: Use Opus AI to generate and test config mutations
7. **Promote**: Promote winning variants to active config

## Questions?

Refer to:
- `momentum/README.md` - Detailed module documentation
- `momentum/example_usage.py` - Code examples
- `momentum/config.py` - Config schema and validation
- Audit tables in database - Historical performance data
