# Momentum Rotation Strategy

A production-ready A-share individual stock momentum rotation system with weekly rebalancing. This module implements a systematic approach to identifying and rotating into the best-performing stocks while managing risk through regime filters and position limits.

## Architecture Overview

The momentum strategy is organized into modular components that can be evolved and optimized:

```
┌─────────────────────────────────────────────────────────────┐
│                    MomentumEngine (Orchestrator)            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Weekly Rebalance Pipeline:                                │
│  1. filter_universe()      → Filter by market cap/turnover  │
│  2. rank_by_momentum()     → Calculate momentum scores      │
│  3. check_regime_filter()  → Determine regime & limits     │
│  4. compute_rebalance()    → Calculate trades              │
│  5. apply_position_limits()→ Enforce constraints           │
│  6. Record audit trail     → Log to database               │
│                                                              │
│  Daily Monitor Pipeline:                                    │
│  1. check_regime_filter()  → Market regime status          │
│  2. apply_stop_loss()      → Liquidate losers              │
│  3. check_drawdown()       → Circuit breaker check         │
│  4. Record audit trail     → Log to database               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Core Modules

### 1. `config.py` - MomentumConfig (The "Genome")

Strategy configuration as a JSON-serializable dataclass. This is the "genome" that Opus AI will evolve for strategy optimization.

**Key Parameters:**

```python
config = MomentumConfig(
    # Momentum calculation
    lookback_days=[20, 60, 120],           # Multi-horizon lookback
    lookback_weights=[0.5, 0.3, 0.2],      # Weighted combination
    momentum_type="composite",              # simple|risk_adjusted|composite
    volatility_penalty=0.3,                 # Penalize volatile stocks

    # Universe filtering
    universe_min_market_cap=5.0,            # Billions CNY
    universe_min_avg_turnover=20.0,         # Millions CNY daily
    universe_exclude_st=True,               # Exclude ST stocks
    universe_exclude_new_days=60,           # Min listing age

    # Portfolio construction
    top_n=10,                               # Holdings target
    max_single_weight=0.15,                 # 15% max per stock
    rebalance_weekday=4,                    # Friday (0=Mon, 4=Fri)
    rebalance_threshold=0.3,                # 30% rank change to trigger

    # Risk management
    stop_loss_pct=-0.08,                   # -8% trailing stop
    regime_filter_enabled=True,
    regime_halt_cash_pct=1.0,              # 100% cash in HALT
    regime_defensive_cash_pct=0.5,         # 50% cash in DEFENSIVE
)
```

**API:**

```python
# Create and validate
config = MomentumConfig()
print(config.summary())

# Evolution: mutate config with changes
evolved = config.mutate({
    "top_n": 15,
    "volatility_penalty": 0.5,
    "description": "Higher risk tolerance variant"
})

# Persistence
config.save(Path("config_v1.0.0.json"))
loaded = MomentumConfig.load(Path("config_v1.0.0.json"))

# Serialization
dict_data = config.to_dict()
config2 = MomentumConfig.from_dict(dict_data)
```

### 2. `universe.py` - Universe Filtering

Filters A-share stocks based on market cap, turnover, ST status, and suspension status.

**Key Function:**

```python
def filter_universe(
    client: TushareClient,
    as_of: str,           # YYYYMMDD format
    config: MomentumConfig,
) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
    - ts_code: Stock code
    - name: Company name
    - industry: Industry classification
    - market_cap: Market cap in billions CNY
    - avg_turnover: Average daily turnover in millions CNY
    """
    pass
```

**Filters Applied:**

1. Exchange: SSE (Shanghai) or SZSE (Shenzhen)
2. Status: Listed (not delisted)
3. ST stocks: Excluded if `universe_exclude_st=True`
4. New listings: Excluded if listing_age < `universe_exclude_new_days`
5. Market cap: >= `universe_min_market_cap` billion CNY
6. Turnover: >= `universe_min_avg_turnover` million CNY
7. Suspended: Excluded if suspended on date

**Example:**

```python
from data_pipeline.tushare_client import TushareClient
from momentum.universe import filter_universe
from momentum.config import MomentumConfig

client = TushareClient()
config = MomentumConfig()
universe = filter_universe(client, "20240315", config)
print(f"Filtered universe: {len(universe)} stocks")
print(universe.head(10))
```

### 3. `calculator.py` - Momentum Scoring

Calculates momentum scores using multiple methods and time horizons.

**Available Methods:**

```python
# Simple momentum: pure return
momentum = calc_simple_momentum(prices, lookback=60)

# Risk-adjusted: return / volatility
momentum = calc_risk_adjusted_momentum(prices, lookback=60)

# Composite: weighted combination across multiple horizons
momentum = calc_composite_momentum(prices_dict, config, ts_code="000001.SZ")
```

**Ranking Function:**

```python
def rank_by_momentum(
    universe: pd.DataFrame,
    client: TushareClient,
    as_of: str,
    config: MomentumConfig,
) -> pd.DataFrame:
    """
    Returns DataFrame with:
    - Original universe columns (ts_code, name, etc.)
    - momentum_score: Numeric score
    - momentum_rank: 1=best, N=worst

    Sorted by momentum_score descending
    """
    pass
```

**Example:**

```python
ranked = rank_by_momentum(universe, client, "20240315", config)
print(ranked.head()[["ts_code", "momentum_score", "momentum_rank"]])
# ts_code      momentum_score  momentum_rank
# 000858.SZ         0.2531         1.0
# 000002.SZ         0.2412         2.0
# ...
```

### 4. `rebalancer.py` - Rebalancing Logic

Determines what trades to make based on current holdings and target ranking.

**Core Function:**

```python
def compute_rebalance(
    current_holdings: List[dict],      # Current positions
    target_ranking: pd.DataFrame,      # Ranked universe
    config: MomentumConfig,
    total_capital: float,
    current_prices: dict,              # ts_code -> price
) -> RebalanceResult:
    """
    Returns RebalanceResult with:
    - buys: List[Trade] - New positions to enter
    - sells: List[Trade] - Positions to exit
    - holds: List[Trade] - Positions to maintain
    - turnover_pct: Portfolio turnover percentage
    - estimated_cost: Transaction costs
    - rebalance_required: Whether rebalance is needed
    """
    pass
```

**Example:**

```python
rebalance = compute_rebalance(
    current_holdings=[
        {"ts_code": "000001.SZ", "shares": 1000, ...},
        {"ts_code": "000858.SZ", "shares": 800, ...},
    ],
    target_ranking=ranked,
    config=config,
    total_capital=1_000_000.0,
    current_prices={"000001.SZ": 16.0, "000858.SZ": 125.0},
)

print(f"Buys: {len(rebalance.buys)}")
for trade in rebalance.buys:
    print(f"  {trade.ts_code}: {trade.reason}")

print(f"Sells: {len(rebalance.sells)}")
print(f"Turnover: {rebalance.turnover_pct:.1f}%")
```

### 5. `risk_control.py` - Risk Management

Implements regime filters, stop losses, position limits, and circuit breakers.

**Key Functions:**

```python
# Regime filter
regime = check_regime_filter(client, "20240315", config)
# Returns: RegimeInfo(regime="HALT"|"DEFENSIVE"|"NEUTRAL"|"RUN"|"STRONG_RUN",
#                     cash_pct=0.0-1.0, max_positions=int, signal_strength=0-100)

# Stop loss check
remaining, stopped_out = apply_stop_loss(holdings, current_prices, config)
# Returns: (remaining_holdings, list_of_stopped_out_codes)

# Position limits
adjusted_weights = apply_position_limits(target_weights, config, regime_info)

# Circuit breaker
is_ok, drawdown = check_drawdown(portfolio_value=950000, peak_value=1000000)
# Returns: (is_ok_bool, drawdown_pct)
```

**Regime Classification:**

| Regime | Signal | Cash % | Max Positions | Use Case |
|--------|--------|--------|---------------|----------|
| HALT | Market down >-2% | 100% | 0 | Market crash |
| DEFENSIVE | Market down -0.5% to -2% | 50% | Top N/2 | Correction |
| NEUTRAL | No strong signal | 0% | N | Normal |
| RUN | Market up 0.5% to 2% | 0% | N | Bull market |
| STRONG_RUN | Market up >2% | 0% | N | Strong bull |

### 6. `engine.py` - Main Orchestrator

The MomentumEngine ties all components together and manages the full trading pipeline.

**Weekly Rebalance:**

```python
from momentum.engine import MomentumEngine
import sqlite3

config = MomentumConfig()
client = TushareClient()
engine = MomentumEngine(config, client)

conn = sqlite3.connect("investment.db")
result = engine.run_weekly_rebalance("20240315", conn)

print(f"Success: {result.success}")
print(f"Universe size: {result.universe_size}")
print(f"Buys: {len(result.buys)}, Sells: {len(result.sells)}")
print(f"Turnover: {result.turnover_pct:.1f}%")
print(f"Regime: {result.regime}")
```

**Daily Monitor:**

```python
daily_result = engine.run_daily_monitor("20240315", conn)

print(f"Portfolio value: {daily_result.portfolio_value:,.2f} CNY")
print(f"Drawdown: {daily_result.drawdown_pct:.2f}%")
print(f"Stopped out: {daily_result.stopped_out}")
print(f"Circuit breaker: {daily_result.circuit_breaker_triggered}")
```

## Usage Examples

### Example 1: Basic Setup

```python
from momentum.config import MomentumConfig
from momentum.engine import MomentumEngine
from data_pipeline.tushare_client import TushareClient

# Create config
config = MomentumConfig(
    top_n=10,
    momentum_type="composite",
    regime_filter_enabled=True,
)

# Create engine
client = TushareClient()
engine = MomentumEngine(config, client)

# Save config for auditing
config.save(Path("momentum_configs/v1.0.0.json"))
```

### Example 2: Evolution Loop (for Opus AI)

```python
# Load baseline
baseline = MomentumConfig.load(Path("momentum_configs/baseline.json"))

# Mutate for experiment
mutations = {
    "top_n": 15,
    "lookback_days": [30, 75, 150],
    "volatility_penalty": 0.5,
    "description": "Evolved variant A",
}
evolved = baseline.mutate(mutations)

# Save evolved variant
evolved.save(Path(f"momentum_configs/{evolved.version}.json"))

# Test in backtest...
# If performance improves, promote evolved to new baseline
```

### Example 3: Integration with Scheduler

```python
from apscheduler.schedulers.background import BackgroundScheduler
import sqlite3

def weekly_rebalance_job():
    config = MomentumConfig.load(Path("momentum_configs/active.json"))
    client = TushareClient()
    engine = MomentumEngine(config, client)

    with sqlite3.connect("investment.db") as conn:
        result = engine.run_weekly_rebalance("20240315", conn)
        if result.success:
            logger.info(f"Rebalance successful: {result.buys_count} buys, {result.sells_count} sells")
        else:
            logger.error(f"Rebalance failed: {result.error_message}")

def daily_monitor_job():
    config = MomentumConfig.load(Path("momentum_configs/active.json"))
    client = TushareClient()
    engine = MomentumEngine(config, client)

    with sqlite3.connect("investment.db") as conn:
        result = engine.run_daily_monitor("20240315", conn)
        if result.circuit_breaker_triggered:
            logger.warning("Circuit breaker triggered, halting trading")

scheduler = BackgroundScheduler()
scheduler.add_job(weekly_rebalance_job, "cron", day_of_week="fri", hour=15)
scheduler.add_job(daily_monitor_job, "cron", hour=16)
scheduler.start()
```

## Database Schema

The engine creates and uses the following tables:

```sql
-- Rebalance audit trail
CREATE TABLE momentum_rebalance_log (
    id INTEGER PRIMARY KEY,
    as_of TEXT,
    config_version TEXT,
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
    created_at TEXT
);

-- Daily monitoring
CREATE TABLE momentum_daily_monitor (
    id INTEGER PRIMARY KEY,
    as_of TEXT,
    config_version TEXT,
    portfolio_value REAL,
    peak_value REAL,
    drawdown_pct REAL,
    stopped_out TEXT,
    regime TEXT,
    created_at TEXT
);
```

## Performance Considerations

### Data Fetching
- Batch API calls where possible (uses TushareClient's batching)
- Cache trade calendar in TushareClient
- Reuse price data across multiple calculations

### Momentum Calculation
- Pre-compute volatility once per run
- Vectorize returns calculation
- Cache price series by stock

### Database
- Indexed queries by (as_of, config_version)
- WAL mode enabled for concurrent reads
- Summarize old audit records periodically

## Evolution & Optimization

The MomentumConfig is specifically designed for Opus AI evolution:

1. **All parameters are JSON-serializable** - Easy to mutate and test
2. **Version tracking** - parent_version and version track lineage
3. **Audit trail** - All runs recorded with config version
4. **Backtest ready** - All calculations are deterministic

**Typical Evolution Loop:**

```
Baseline Config (v1.0.0)
    ↓
Mutate: top_n, lookback_days, volatility_penalty (→ v1.0.1)
    ↓
Backtest: Compare performance
    ↓
If better: Promote to baseline
If worse: Archive, try different mutations
```

## Logging

All modules use loguru for structured logging:

```python
from loguru import logger

logger.info(f"Filtered universe: {len(universe)} stocks")
logger.debug(f"Top 5: {universe.head(5)}")
logger.warning(f"No price data for {ts_code}")
logger.error(f"Failed to fetch universe: {e}")
```

## Type Hints

All functions are fully type-hinted for IDE support and runtime validation:

```python
def rank_by_momentum(
    universe: pd.DataFrame,
    client: TushareClient,
    as_of: str,
    config: MomentumConfig,
) -> pd.DataFrame:
    pass
```

## Testing

Run the example usage script:

```bash
cd /sessions/festive-jolly-meitner/mnt/stock_agent
python momentum/example_usage.py
```

## Integration Notes

- All data fetching goes through TushareClient (singleton pattern)
- All persistence goes through db/repository.py
- Config is agnostic to portfolio construction (allows for easy customization)
- Risk control is modular and can be enhanced independently

## Future Enhancements

- [ ] Add mean-reversion score calculations
- [ ] Implement sector-aware momentum (don't short sectors)
- [ ] Add machine learning momentum predictor
- [ ] Implement portfolio-level optimization (mean-variance)
- [ ] Add slippage/commission models for more accurate backtesting
- [ ] Multi-timeframe momentum (daily + weekly + monthly)
