# Evolution Engine Integration Guide

## Overview

The Opus AI Self-Evolution Engine automatically optimizes the momentum rotation strategy through:

1. **Performance Evaluation** - Analyzes historical rebalancing data
2. **AI Reflection** - Opus API analyzes performance and proposes improvements
3. **Sandbox Testing** - Backtests mutations without live trading
4. **Safe Promotion** - Statistically validates improvements before deployment

## Architecture

```
EvolutionEngine (orchestrator)
├── PerformanceEvaluator
│   ├── evaluate_performance() - Calculate metrics from history
│   └── compare_versions() - A/B comparison
├── ReflectorEngine (Opus API)
│   ├── reflect_on_performance() - AI analysis
│   └── generate_mutation_proposals() - Parameter mutations
├── SandboxRunner
│   ├── run_backtest() - Simulate strategy
│   └── compare_with_baseline() - Promotion decision
└── EvolutionRegistry (SQLite)
    ├── register_version() - Store config
    ├── get_version() - Retrieve config
    ├── get_active_version() - Current production version
    └── promote_version() - Mark as active
```

## Usage

### Running an Evolution Cycle

```python
import sqlite3
from evolution.engine import EvolutionEngine
from data_pipeline.tushare_client import TushareClient

# Setup
conn = sqlite3.connect("db/investment.db")
client = TushareClient(token="YOUR_TOKEN")

# Run evolution
result = EvolutionEngine.run_evolution_cycle(
    conn=conn,
    as_of="2025-03-21",
    tushare_client=client,
    backtest_start_offset_days=90,
    backtest_end_offset_days=7,
)

# Check results
print(result.summary())
print(f"Promoted: {result.best_version}")
print(f"Improvement: {result.best_improvement_pct:.1f}%")
```

### Manual Evaluation

```python
from evolution.evaluator import PerformanceEvaluator

# Evaluate a specific version
perf = PerformanceEvaluator.evaluate_performance(
    conn=conn,
    strategy_version="1.2.0",
    window_days=90,
)

print(perf.summary())
# Returns: PerformanceReport with sharpe_ratio, max_drawdown, etc.
```

### AI Reflection

```python
from evolution.reflector import ReflectorEngine
from momentum.config import MomentumConfig

config = MomentumConfig.load(Path("momentum/config.json"))

# Get Opus analysis
reflection = ReflectorEngine.reflect_on_performance(
    perf=perf,
    config=config,
    recent_trades=[],  # Optional
)

print(reflection.analysis)
print(reflection.market_regime)
for proposal in reflection.proposals:
    print(f"- {proposal.rationale}")
```

### Sandbox Testing

```python
from evolution.sandbox import SandboxRunner
from evolution.registry import EvolutionRegistry

# Create a mutation
mutated = config.mutate({"top_n": 12, "volatility_penalty": 0.25})

# Backtest it
result = SandboxRunner.run_backtest(
    config=mutated,
    start_date="2024-12-01",
    end_date="2025-03-07",
    client=client,
)

print(f"Return: {result.total_return:.2f}%")
print(f"Sharpe: {result.sharpe_ratio:.3f}")
```

### Registry Management

```python
from evolution.registry import EvolutionRegistry

# Register a new version
EvolutionRegistry.register_version(
    conn=conn,
    config=mutated,
    parent_version="1.2.0",
    mutation_reason="Increased position count based on Opus analysis",
)

# Get version history
lineage = EvolutionRegistry.get_lineage(conn, "1.2.1")
for entry in lineage:
    print(f"{entry['version']}: {entry['description']}")

# Promote a version
EvolutionRegistry.promote_version(
    conn=conn,
    version="1.2.1",
    compared_with="1.2.0",
    confidence=0.85,
    reason="5.2% improvement in Sharpe ratio",
)

# Get active version
active = EvolutionRegistry.get_active_version(conn)
print(f"Production version: {active.version}")
```

## Database Schema

### evolution_versions
```sql
CREATE TABLE evolution_versions (
    id INTEGER PRIMARY KEY,
    version TEXT UNIQUE,
    parent_version TEXT,
    config_json TEXT,
    description TEXT,
    mutation_reason TEXT,
    created_at TEXT,
    promoted_at TEXT  -- NULL if not promoted
);
```

### evolution_evaluations
```sql
CREATE TABLE evolution_evaluations (
    id INTEGER PRIMARY KEY,
    version TEXT,
    evaluation_date TEXT,
    window_days INTEGER,
    total_return REAL,
    annualized_return REAL,
    sharpe_ratio REAL,
    max_drawdown REAL,
    calmar_ratio REAL,
    win_rate REAL,
    avg_turnover REAL,
    total_trades INTEGER,
    information_ratio REAL,
    evaluation_json TEXT,
    created_at TEXT
);
```

### evolution_promotions
```sql
CREATE TABLE evolution_promotions (
    id INTEGER PRIMARY KEY,
    version TEXT,
    promoted_at TEXT,
    reason TEXT,
    compared_with TEXT,
    confidence REAL,
    created_at TEXT
);
```

## Safety Constraints

The engine enforces strict safety gates:

```python
# From EvolutionEngine class
MAX_DRAWDOWN_THRESHOLD = -25.0  # Never promote worse than -25% max DD
MIN_SHARPE_THRESHOLD = 0.3      # Minimum Sharpe ratio for promotion
MIN_IMPROVEMENT_PCT = 5.0       # Require 5% improvement to promote
```

A mutation is only promoted if:
1. ✓ Backtest completes successfully
2. ✓ Sharpe ratio ≥ 0.3
3. ✓ Max drawdown > -25%
4. ✓ Overall score improves by ≥ 5%
5. ✓ Confidence ≥ 50%

## Scheduler Integration

Add to your weekly/monthly scheduler:

```python
from evolution.engine import EvolutionEngine

# Weekly evolution cycle
def evolution_job(as_of_date: str):
    with get_db() as conn:
        result = EvolutionEngine.run_full_iteration(
            conn=conn,
            as_of=as_of_date,
            tushare_client=client,
        )

    if result["success"]:
        logger.info(f"Evolution cycle complete: {result['message']}")
        if result["best_version"]:
            notify_team(f"New version promoted: {result['best_version']}")
    else:
        logger.error(f"Evolution cycle failed: {result['error']}")

# Schedule with APScheduler, Celery, or cron
# APScheduler example:
# scheduler.add_job(
#     evolution_job,
#     "cron",
#     day_of_week="fri",
#     hour=18,
#     args=["2025-03-21"],
#     id="weekly_evolution",
# )
```

## Performance Metrics

The engine tracks:

| Metric | Calculation | Interpretation |
|--------|-----------|-----------------|
| **Sharpe Ratio** | (annual_return - rf) / volatility | Risk-adjusted returns; >1 is excellent |
| **Max Drawdown** | worst peak-to-trough decline | Maximum loss from peak |
| **Calmar Ratio** | annual_return / abs(max_drawdown) | Return per unit of drawdown risk |
| **Win Rate** | profitable_trades / total_trades × 100 | % of winning transactions |
| **Turnover** | sum(abs(weight_changes)) / 2 | Portfolio activity cost |
| **Info Ratio** | excess_return / tracking_error | Alpha per unit of risk |

## Troubleshooting

### "No rebalance logs found"
- Evolution engine expects momentum_rebalance_log table
- Ensure momentum engine is running and logging rebalances
- Table structure depends on momentum/engine.py

### "No promoted version found"
- Database is new; create baseline with:
```python
from momentum.config import MomentumConfig
from evolution.registry import EvolutionRegistry

config = MomentumConfig()
EvolutionRegistry.register_version(conn, config, mutation_reason="Baseline")
EvolutionRegistry.promote_version(conn, config.version, reason="Initial baseline")
```

### Backtest failures
- Check TushareClient connection and data availability
- Verify date range is not in future
- Ensure universe has sufficient stocks
- Check data_pipeline logs

### Opus API errors
- Verify OPENROUTER_API_KEY environment variable
- Check OPENROUTER_MODEL_PRIMARY is valid (default: "anthropic/claude-3.5-sonnet")
- Ensure network connectivity to openrouter.ai

## Customization

### Adjusting Safety Thresholds

```python
# In your code before running
EvolutionEngine.MAX_DRAWDOWN_THRESHOLD = -30.0  # More aggressive
EvolutionEngine.MIN_SHARPE_THRESHOLD = 0.2      # More relaxed
EvolutionEngine.MIN_IMPROVEMENT_PCT = 3.0       # Lower bar
```

### Custom Mutation Strategy

```python
# Extend ReflectorEngine for custom logic
class CustomReflector(ReflectorEngine):
    SYSTEM_PROMPT = "Custom instructions..."

    @staticmethod
    def reflect_on_performance(perf, config, trades):
        # Custom implementation
        pass
```

### Custom Backtest Logic

```python
# Extend SandboxRunner for custom simulation
class CustomSandbox(SandboxRunner):
    @staticmethod
    def run_backtest(config, start_date, end_date, client):
        # Custom backtest implementation
        pass
```

## Monitoring & Observability

All components log to loguru logger with component tag "evolution_*":
- evolution_reflector
- evolution_evaluator
- evolution_sandbox
- evolution_registry

View logs:
```bash
tail -f logs/latest.log | grep evolution_
```

Query database for evolution history:
```python
history = EvolutionRegistry.get_version_history(conn, limit=20)
for v in history:
    print(f"{v['version']}: {v['description']}")
```

## References

- MomentumConfig: `/momentum/config.py`
- Momentum Calculator: `/momentum/calculator.py`
- Data Pipeline: `/data_pipeline/tushare_client.py`
- LLM Client: `/config/settings.py`
