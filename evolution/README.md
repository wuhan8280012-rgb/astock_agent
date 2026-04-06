# Opus AI Self-Evolution Engine

A production-ready strategy optimization system that uses Claude Opus API to automatically reflect on momentum rotation performance, propose parameter improvements, test them safely in sandbox, and promote winners.

## Features

✓ **Automatic Performance Analysis** - Evaluates historical rebalancing data with comprehensive metrics
✓ **AI-Powered Reflection** - Opus API analyzes strategy behavior and proposes mutations
✓ **Safe Sandbox Testing** - Backtests mutations before any deployment
✓ **Intelligent Promotion** - Statistical validation prevents overfitting and ensures real improvements
✓ **Complete History Tracking** - Full evolution lineage with parent-child relationships
✓ **Production-Grade Logging** - Loguru integration for monitoring and debugging

## Architecture

### Core Components

#### 1. `evaluator.py` - Performance Evaluation
Calculates strategy metrics from historical trading data:
- Total return, annualized return, Sharpe ratio
- Max drawdown, Calmar ratio
- Win rate, average turnover, trade count
- Information ratio

```python
perf = PerformanceEvaluator.evaluate_performance(
    conn, "1.0.0", window_days=90
)
print(f"Sharpe: {perf.sharpe_ratio:.3f}")
```

#### 2. `reflector.py` - AI Reflection Engine
Uses Opus API to analyze performance and propose mutations:
- Market regime detection (trending/mean-reverting)
- Parameter sensitivity analysis
- Mutation proposal generation with confidence scores
- Constrained to incremental changes (±10-25% per parameter)

```python
reflection = ReflectorEngine.reflect_on_performance(perf, config)
for proposal in reflection.proposals:
    print(f"{proposal.rationale} -> {proposal.parameter_changes}")
```

#### 3. `sandbox.py` - Sandbox Backtest Runner
Simulates strategy performance safely:
- Simplified momentum backtest (rule-based, no LLM)
- Weekly rebalancing simulation
- Performance metrics calculation
- Statistical comparison with baseline

```python
result = SandboxRunner.run_backtest(
    config, "2024-12-01", "2025-03-07", client
)
decision = SandboxRunner.compare_with_baseline(result, baseline)
```

#### 4. `registry.py` - Version Registry
SQLite-based version management:
- Stores configurations with parent-child relationships
- Tracks promotions and evaluations
- Maintains full evolution lineage
- Provides active version lookup

```python
EvolutionRegistry.register_version(conn, config, parent="1.0.0")
active = EvolutionRegistry.get_active_version(conn)
history = EvolutionRegistry.get_version_history(conn, limit=20)
```

#### 5. `engine.py` - Evolution Orchestrator
Coordinates complete evolution cycles:
1. Load active configuration
2. Evaluate recent performance
3. Call Opus for analysis and proposals
4. Test each mutation in sandbox
5. Promote best candidate if better than baseline

```python
result = EvolutionEngine.run_evolution_cycle(
    conn, "2025-03-21", tushare_client
)
print(result.summary())
```

## Safety Constraints

The engine enforces strict gates to prevent degradation:

```python
MAX_DRAWDOWN_THRESHOLD = -25.0  # Never promote worse drawdown
MIN_SHARPE_THRESHOLD = 0.3      # Minimum risk-adjusted return
MIN_IMPROVEMENT_PCT = 5.0       # Require 5% improvement
```

A mutation is promoted only if:
1. ✓ Backtest succeeds
2. ✓ Sharpe ratio ≥ 0.3
3. ✓ Max drawdown > -25%
4. ✓ Overall score improves ≥ 5%
5. ✓ Statistical confidence ≥ 50%

## Quick Start

### Basic Usage

```python
from evolution.engine import EvolutionEngine
from db.repository import get_db
from data_pipeline.tushare_client import TushareClient

with get_db() as conn:
    client = TushareClient(token="YOUR_TOKEN")

    # Run weekly evolution
    result = EvolutionEngine.run_evolution_cycle(
        conn=conn,
        as_of="2025-03-21",
        tushare_client=client,
    )

    print(result.summary())
    if result.best_version:
        print(f"Promoted: {result.best_version}")
        print(f"Improvement: {result.best_improvement_pct:.1f}%")
```

### Scheduler Integration

```python
from apscheduler.schedulers.background import BackgroundScheduler

def evolution_job():
    with get_db() as conn:
        result = EvolutionEngine.run_full_iteration(
            conn, "2025-03-21", client
        )

    if result["best_version"]:
        notify_team(f"Promoted: {result['best_version']}")

scheduler = BackgroundScheduler()
scheduler.add_job(
    evolution_job,
    'cron',
    day_of_week='fri',
    hour=18,
    id='weekly_evolution',
)
scheduler.start()
```

## Configuration Parameters (Tunable)

The engine evolves these `MomentumConfig` parameters:

| Parameter | Default | Range | Mutation Sensitivity |
|-----------|---------|-------|---------------------|
| `top_n` | 10 | 5-20 | High |
| `lookback_days` | [20, 60, 120] | adaptive | High |
| `volatility_penalty` | 0.3 | 0.0-0.5 | Medium |
| `rebalance_threshold` | 0.3 | 0.1-0.5 | Low |
| `stop_loss_pct` | -0.08 | -0.05 to -0.15 | Medium |
| `regime_defensive_cash_pct` | 0.5 | 0.3-0.8 | Medium |
| `max_single_weight` | 0.15 | 0.1-0.25 | Low |

Opus AI proposes mutations using domain knowledge:
- Trending markets → longer lookback periods
- High volatility → increase volatility penalty
- Low diversification → increase top_n
- High drawdown → increase defensive cash

## Database Schema

Three core tables track evolution state:

### evolution_versions
```sql
SELECT version, parent_version, created_at, promoted_at
FROM evolution_versions
WHERE promoted_at IS NOT NULL;  -- Active versions
```

### evolution_evaluations
```sql
SELECT version, sharpe_ratio, max_drawdown, total_return
FROM evolution_evaluations
WHERE evaluation_date > date('now', '-90 days');
```

### evolution_promotions
```sql
SELECT version, compared_with, improvement_pct, confidence
FROM evolution_promotions
ORDER BY promoted_at DESC;
```

## Performance Metrics

### Calculated Metrics

| Metric | Interpretation |
|--------|-----------------|
| **Sharpe Ratio** | Risk-adjusted returns (>1.0 excellent, >0.5 good) |
| **Calmar Ratio** | Return per unit of max drawdown |
| **Max Drawdown** | Worst peak-to-trough decline (-20% to -50% typical) |
| **Win Rate** | % of profitable trades (40-60% typical) |
| **Information Ratio** | Alpha per unit of tracking error |
| **Turnover** | Portfolio rebalancing costs (10-30% typical) |

### Scoring Formula

```python
# Composite score used for promotion decisions
score = (
    sharpe_ratio * 0.4 +           # 40% weight: risk-adjusted return
    (total_return / 100) * 0.3 +   # 30% weight: absolute return
    (win_rate / 100) * 0.2 +       # 20% weight: hit ratio
    (abs(max_drawdown) / 100) * 0.1  # 10% weight: drawdown control
)
```

## Example Workflows

### 1. Evaluate Current Strategy
```python
perf = PerformanceEvaluator.evaluate_performance(
    conn, "1.0.0", window_days=90
)
print(perf.summary())
```

### 2. Get Opus Analysis
```python
reflection = ReflectorEngine.reflect_on_performance(perf, config)
print(f"Issues: {reflection.key_issues}")
print(f"Opportunities: {reflection.opportunities}")
```

### 3. Test a Specific Mutation
```python
mutated = config.mutate({"top_n": 12})
result = SandboxRunner.run_backtest(mutated, start, end, client)
decision = SandboxRunner.compare_with_baseline(result, baseline)
if decision.promote:
    EvolutionRegistry.promote_version(conn, mutated.version)
```

### 4. View Evolution History
```python
lineage = EvolutionRegistry.get_lineage(conn, "1.5.2")
for entry in lineage:
    print(f"{entry['version']}: {entry['description']}")
```

## Troubleshooting

### Issue: "No rebalance logs found"
**Cause**: momentum_rebalance_log table is empty
**Solution**: Ensure momentum engine is running and logging rebalances
```python
# Check if table exists and has data
conn.execute("SELECT COUNT(*) FROM momentum_rebalance_log").fetchone()
```

### Issue: Backtest returns 0% return
**Cause**: Data pipeline issue or empty universe
**Solution**: Verify TushareClient connectivity and universe filtering
```python
from momentum.universe import filter_universe
universe = filter_universe(client, "2025-03-21")
print(f"Universe size: {len(universe)}")
```

### Issue: Opus API timeouts
**Cause**: Network latency or API rate limits
**Solution**: Check environment variables and retry with backoff
```python
# In config/.env
OPENROUTER_API_KEY=sk-...
OPENROUTER_MODEL_PRIMARY=anthropic/claude-3.5-sonnet
```

## Advanced Usage

### Custom Safety Thresholds
```python
# More aggressive evolution
EvolutionEngine.MAX_DRAWDOWN_THRESHOLD = -30.0
EvolutionEngine.MIN_SHARPE_THRESHOLD = 0.2
EvolutionEngine.MIN_IMPROVEMENT_PCT = 2.0
```

### Custom Reflector
```python
class CustomReflector(ReflectorEngine):
    SYSTEM_PROMPT = "Your custom instructions..."

    @staticmethod
    def reflect_on_performance(perf, config, trades):
        # Custom logic here
        pass
```

### Batch Evolution
```python
# Test multiple versions in parallel
configs = [config.mutate({...}) for _ in range(10)]
for cfg in configs:
    result = SandboxRunner.run_backtest(cfg, start, end, client)
```

## Performance Expectations

Typical evolution cycle execution times:

| Step | Time | Notes |
|------|------|-------|
| **Evaluation** | <5s | Database query |
| **Reflection** | 10-30s | Opus API call |
| **Single Backtest** | 30-120s | Data fetch + simulation |
| **Full Cycle (5 mutations)** | 2-10 min | Sequential backtests |

## Monitoring

Enable debug logging:
```python
from loguru import logger
logger.enable("evolution")
```

View logs:
```bash
tail -f logs/latest.log | grep evolution_
```

## Dependencies

```
anthropic>=0.7.0         # Claude API
openai>=1.0.0            # OpenRouter proxy
pandas>=1.3.0            # Data manipulation
numpy>=1.21.0            # Numerical computing
loguru>=0.6.0            # Structured logging
sqlite3                  # Standard library
```

## Files

```
evolution/
├── __init__.py           # Package initialization
├── evaluator.py          # Performance metrics (240 lines)
├── reflector.py          # Opus AI analysis (270 lines)
├── sandbox.py            # Backtest engine (310 lines)
├── registry.py           # Version management (300 lines)
├── engine.py             # Orchestrator (380 lines)
├── example_usage.py      # Usage examples (350 lines)
├── README.md             # This file
└── INTEGRATION_GUIDE.md   # Detailed guide
```

Total: ~1,850 lines of production code

## Contributing

When extending the evolution engine:

1. **Add new metrics** → Update `PerformanceReport` dataclass
2. **Custom evolution logic** → Extend `EvolutionEngine` class
3. **New backtest features** → Extend `SandboxRunner.run_backtest()`
4. **Versioning changes** → Update `EvolutionRegistry` tables

## License

Part of the quantitative trading system. See main project LICENSE.

## Support

For issues or questions:
1. Check INTEGRATION_GUIDE.md for detailed reference
2. Review example_usage.py for working examples
3. Check database schema in registry.py
4. Enable debug logging to see detailed execution flow
