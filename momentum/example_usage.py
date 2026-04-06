"""
Example usage of the momentum rotation strategy module.
Demonstrates how to use all core components.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from data_pipeline.tushare_client import TushareClient
from momentum.calculator import rank_by_momentum
from momentum.config import MomentumConfig
from momentum.engine import MomentumEngine
from momentum.rebalancer import compute_rebalance
from momentum.risk_control import (
    apply_position_limits,
    apply_stop_loss,
    check_drawdown,
    check_regime_filter,
)
from momentum.universe import filter_universe


def example_1_config_creation():
    """Example 1: Create and manipulate configurations."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Configuration Management")
    print("=" * 60)

    # Create baseline config
    config = MomentumConfig(
        lookback_days=[20, 60, 120],
        lookback_weights=[0.5, 0.3, 0.2],
        universe_min_market_cap=5.0,
        universe_min_avg_turnover=20.0,
        top_n=10,
        rebalance_threshold=0.3,
        momentum_type="composite",
    )

    print("Baseline Config:")
    print(config.summary())

    # Mutate config (for evolution)
    mutated = config.mutate({
        "top_n": 15,
        "lookback_days": [30, 60, 120],
        "volatility_penalty": 0.5,
        "description": "Evolved config with more holdings and higher vol penalty",
    })

    print("\n\nMutated Config:")
    print(mutated.summary())

    # Save and load
    config_path = Path("/tmp/momentum_config_v1.0.0.json")
    config.save(config_path)
    loaded = MomentumConfig.load(config_path)

    print(f"\n\nLoaded config version: {loaded.version}")
    print(f"Description: {loaded.description}")


def example_2_universe_filtering():
    """Example 2: Filter universe by criteria."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Universe Filtering")
    print("=" * 60)

    client = TushareClient()
    config = MomentumConfig()

    as_of = "20240315"  # Example date

    try:
        universe = filter_universe(client, as_of, config)
        print(f"Filtered universe: {len(universe)} stocks")
        print("\nTop 10 by market cap:")
        print(universe.head(10)[["ts_code", "name", "market_cap", "avg_turnover"]])
    except Exception as e:
        print(f"Error fetching universe (expected if running without real Tushare): {e}")


def example_3_momentum_calculation():
    """Example 3: Calculate momentum scores."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Momentum Calculation")
    print("=" * 60)

    client = TushareClient()
    config = MomentumConfig()

    # Create sample universe
    universe = pd.DataFrame({
        "ts_code": ["000001.SZ", "000002.SZ", "000858.SZ"],
        "name": ["平安银行", "万科", "五粮液"],
        "industry": ["银行", "房地产", "食品"],
        "market_cap": [100.0, 150.0, 120.0],
        "avg_turnover": [50.0, 60.0, 55.0],
    })

    as_of = "20240315"

    try:
        ranked = rank_by_momentum(universe, client, as_of, config)
        print(f"Ranked {len(ranked)} stocks by momentum")
        print("\nTop 5 by momentum score:")
        print(ranked.head(5)[["ts_code", "name", "momentum_score", "momentum_rank"]])
    except Exception as e:
        print(f"Error calculating momentum (expected if running without real Tushare): {e}")


def example_4_rebalancing():
    """Example 4: Compute rebalance trades."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Rebalancing")
    print("=" * 60)

    config = MomentumConfig(top_n=5)

    # Sample current holdings
    current_holdings = [
        {
            "ts_code": "000001.SZ",
            "shares": 1000,
            "entry_price": 15.0,
            "entry_date": "20240101",
            "momentum_rank": 8,
        },
        {
            "ts_code": "000858.SZ",
            "shares": 800,
            "entry_price": 120.0,
            "entry_date": "20240101",
            "momentum_rank": 3,
        },
    ]

    # Sample target ranking
    target_ranking = pd.DataFrame({
        "ts_code": ["000858.SZ", "000002.SZ", "000001.SZ", "000009.SZ", "000333.SZ"],
        "momentum_score": [0.25, 0.22, 0.18, 0.15, 0.12],
        "momentum_rank": [1.0, 2.0, 3.0, 4.0, 5.0],
    })

    current_prices = {
        "000001.SZ": 16.0,
        "000858.SZ": 125.0,
        "000002.SZ": 25.0,
        "000009.SZ": 8.5,
        "000333.SZ": 18.0,
    }

    rebalance = compute_rebalance(
        current_holdings,
        target_ranking,
        config,
        total_capital=1_000_000.0,
        current_prices=current_prices,
    )

    print(f"Rebalance Result:")
    print(f"  Buys: {len(rebalance.buys)}")
    for trade in rebalance.buys:
        print(f"    - {trade.ts_code}: {trade.reason}")

    print(f"  Sells: {len(rebalance.sells)}")
    for trade in rebalance.sells:
        print(f"    - {trade.ts_code}: {trade.reason}")

    print(f"  Holds: {len(rebalance.holds)}")
    for trade in rebalance.holds:
        print(f"    - {trade.ts_code}")

    print(f"  Turnover: {rebalance.turnover_pct:.2f}%")
    print(f"  Estimated Cost: {rebalance.estimated_cost:.2f} CNY")
    print(f"  Rebalance Required: {rebalance.rebalance_required}")


def example_5_risk_management():
    """Example 5: Risk management and regime filters."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Risk Management")
    print("=" * 60)

    client = TushareClient()
    config = MomentumConfig()

    # Check regime
    print("1. Regime Filter:")
    try:
        regime = check_regime_filter(client, "20240315", config)
        print(f"   Regime: {regime.regime}")
        print(f"   Cash %: {regime.cash_pct*100:.1f}%")
        print(f"   Signal Strength: {regime.signal_strength:.1f}")
    except Exception as e:
        print(f"   Error: {e}")

    # Check stop loss
    print("\n2. Stop Loss Check:")
    holdings = [
        {"ts_code": "000001.SZ", "shares": 1000, "entry_price": 15.0},
        {"ts_code": "000858.SZ", "shares": 800, "entry_price": 120.0},
    ]
    current_prices = {
        "000001.SZ": 13.5,  # -10% loss, should trigger stop
        "000858.SZ": 125.0,  # +4% gain, should hold
    }

    remaining, stopped = apply_stop_loss(holdings, current_prices, config)
    print(f"   Remaining: {len(remaining)}")
    print(f"   Stopped out: {stopped}")

    # Check drawdown
    print("\n3. Drawdown Check:")
    is_ok, drawdown = check_drawdown(
        portfolio_value=950_000.0,
        peak_value=1_000_000.0,
        max_drawdown_threshold=-0.20,
    )
    print(f"   Portfolio value: 950,000 CNY")
    print(f"   Peak value: 1,000,000 CNY")
    print(f"   Drawdown: {drawdown*100:.2f}%")
    print(f"   Circuit breaker OK: {is_ok}")


def example_6_full_engine():
    """Example 6: Use the full MomentumEngine."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Full Engine Integration")
    print("=" * 60)

    config = MomentumConfig(
        top_n=5,
        rebalance_weekday=4,  # Friday
        momentum_type="composite",
    )

    client = TushareClient()
    engine = MomentumEngine(config, client)

    print(f"Created MomentumEngine:")
    print(f"  Config version: {config.version}")
    print(f"  Top N holdings: {config.top_n}")
    print(f"  Rebalance day: {config._weekday_name(config.rebalance_weekday)}")

    print("\nEngine is ready for:")
    print("  - run_weekly_rebalance(as_of, conn)")
    print("  - run_daily_monitor(as_of, conn)")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MOMENTUM ROTATION STRATEGY - USAGE EXAMPLES")
    print("=" * 60)

    example_1_config_creation()
    example_4_rebalancing()
    example_5_risk_management()
    example_6_full_engine()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
