"""
Example usage of the Opus AI Self-Evolution Engine.
Demonstrates each component and how to use the evolution system in practice.
"""

import sqlite3
from datetime import datetime
from pathlib import Path

from config.settings import DB_PATH
from data_pipeline.tushare_client import TushareClient
from db.repository import get_db
from evolution.engine import EvolutionEngine
from evolution.evaluator import PerformanceEvaluator
from evolution.reflector import ReflectorEngine
from evolution.registry import EvolutionRegistry
from evolution.sandbox import SandboxRunner
from loguru import logger
from momentum.config import MomentumConfig


def example_1_basic_evaluation():
    """Example 1: Evaluate performance of current strategy version."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 1: Basic Performance Evaluation")
    logger.info("=" * 60)

    with get_db() as conn:
        # Evaluate the active version over last 90 days
        active_config = EvolutionRegistry.get_active_version(conn)
        if not active_config:
            logger.warning("No active version found, skipping example")
            return

        perf = PerformanceEvaluator.evaluate_performance(
            conn=conn,
            strategy_version=active_config.version,
            window_days=90,
        )

        print(perf.summary())
        print(f"\nKey metrics:")
        print(f"  Return: {perf.total_return:.2f}%")
        print(f"  Sharpe: {perf.sharpe_ratio:.3f}")
        print(f"  Max DD: {perf.max_drawdown:.2f}%")
        print(f"  Trades: {perf.total_trades}")


def example_2_version_comparison():
    """Example 2: Compare two strategy versions."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 2: Compare Strategy Versions")
    logger.info("=" * 60)

    with get_db() as conn:
        # Get two versions to compare (from history)
        history = EvolutionRegistry.get_version_history(conn, limit=2)
        if len(history) < 2:
            logger.warning("Need at least 2 versions for comparison")
            return

        v1 = history[0]["version"]
        v2 = history[1]["version"]

        comparison = PerformanceEvaluator.compare_versions(
            conn=conn,
            version_a=v1,
            version_b=v2,
            window_days=90,
        )

        print(comparison.summary)
        print(f"\nBetter version: {comparison.better_version}")
        print(f"Confidence: {comparison.confidence:.1%}")


def example_3_ai_reflection():
    """Example 3: Get Opus AI analysis of strategy performance."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 3: Opus AI Reflection")
    logger.info("=" * 60)

    with get_db() as conn:
        active_config = EvolutionRegistry.get_active_version(conn)
        if not active_config:
            logger.warning("No active version found")
            return

        # Get performance report
        perf = PerformanceEvaluator.evaluate_performance(
            conn=conn,
            strategy_version=active_config.version,
            window_days=90,
        )

        # Request AI reflection
        print("Requesting Opus AI analysis...")
        reflection = ReflectorEngine.reflect_on_performance(
            perf=perf,
            config=active_config,
            recent_trades=[],
        )

        print(f"\nMarket Regime: {reflection.market_regime}")
        print(f"\nAnalysis:\n{reflection.analysis}")

        print(f"\nKey Issues ({len(reflection.key_issues)}):")
        for issue in reflection.key_issues:
            print(f"  - {issue}")

        print(f"\nOpportunities ({len(reflection.opportunities)}):")
        for opp in reflection.opportunities:
            print(f"  - {opp}")

        print(f"\nProposals ({len(reflection.proposals)}):")
        for i, prop in enumerate(reflection.proposals, 1):
            print(f"\n  Proposal {i}:")
            print(f"    Rationale: {prop.rationale}")
            print(f"    Changes: {prop.parameter_changes}")
            print(f"    Confidence: {prop.confidence:.0%}")
            print(f"    Risk Level: {prop.risk_level}")


def example_4_sandbox_backtest():
    """Example 4: Run sandbox backtest of a mutated strategy."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 4: Sandbox Backtest")
    logger.info("=" * 60)

    with get_db() as conn:
        # Get current config
        active_config = EvolutionRegistry.get_active_version(conn)
        if not active_config:
            logger.warning("No active version found")
            return

        # Create a mutation (increase position count)
        mutated = active_config.mutate({
            "top_n": 12,
            "volatility_penalty": 0.25,
        })
        print(f"Created mutation: {mutated.version}")

        # Register it
        EvolutionRegistry.register_version(
            conn=conn,
            config=mutated,
            parent_version=active_config.version,
            mutation_reason="Test: increase positions and volatility penalty",
        )

        # Run backtest
        print("Running sandbox backtest (this may take a minute)...")
        client = TushareClient(token="demo")  # Use your actual token

        result = SandboxRunner.run_backtest(
            config=mutated,
            start_date="2024-12-01",
            end_date="2025-03-07",
            client=client,
        )

        if result.success:
            print(f"\nBacktest Results:")
            print(f"  Return: {result.total_return:.2f}%")
            print(f"  Sharpe: {result.sharpe_ratio:.3f}")
            print(f"  Max DD: {result.max_drawdown:.2f}%")
            print(f"  Turnover: {result.turnover:.2f}%")
            print(f"  Trades: {result.trade_count}")
            print(f"  Rebalances: {result.num_rebalances}")
        else:
            print(f"Backtest failed: {result.error_message}")


def example_5_version_registry():
    """Example 5: Explore version history and registry."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 5: Version Registry & History")
    logger.info("=" * 60)

    with get_db() as conn:
        # Get recent version history
        history = EvolutionRegistry.get_version_history(conn, limit=10)
        print(f"Recent versions ({len(history)}):")
        for v in history:
            promoted = "✓" if v["promoted"] else " "
            print(f"  [{promoted}] {v['version']}: {v['description']}")

        if history:
            # Get full lineage of latest version
            latest = history[0]["version"]
            lineage = EvolutionRegistry.get_lineage(conn, latest)

            print(f"\nLineage for {latest} ({len(lineage)} generations):")
            for entry in lineage:
                print(f"  {entry['version']}")
                if entry["description"]:
                    print(f"    └─ {entry['description']}")

        # Get active version
        active = EvolutionRegistry.get_active_version(conn)
        if active:
            print(f"\nActive (Production) Version: {active.version}")
            print(active.summary())


def example_6_full_evolution_cycle():
    """Example 6: Run a complete evolution cycle (main use case)."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 6: Full Evolution Cycle")
    logger.info("=" * 60)

    with get_db() as conn:
        client = TushareClient(token="demo")  # Use your actual token

        # Run evolution
        print("Starting evolution cycle...")
        result = EvolutionEngine.run_evolution_cycle(
            conn=conn,
            as_of="2025-03-21",
            tushare_client=client,
            backtest_start_offset_days=90,
            backtest_end_offset_days=7,
        )

        print(result.summary())

        if result.test_results:
            print(f"\nDetailed Test Results:")
            for tr in result.test_results:
                status = "✓ PROMOTE" if tr.promotion_decision.promote else "✗ REJECT"
                print(f"\n  {status}")
                print(f"    Mutation: {tr.mutated_version}")
                print(f"    Confidence: {tr.promotion_decision.confidence:.0%}")
                print(f"    Reason: {tr.promotion_decision.reason[:80]}...")


def example_7_scheduler_integration():
    """Example 7: How to integrate with a task scheduler."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 7: Scheduler Integration (APScheduler)")
    logger.info("=" * 60)

    # This is example code for integration
    print("""
# Example: Weekly evolution with APScheduler

from apscheduler.schedulers.background import BackgroundScheduler
from evolution.engine import EvolutionEngine
from db.repository import get_db
from data_pipeline.tushare_client import TushareClient

def evolution_job(as_of_date: str):
    \"\"\"Run weekly evolution cycle.\"\"\"
    try:
        with get_db() as conn:
            client = TushareClient(token=TUSHARE_TOKEN)
            result = EvolutionEngine.run_full_iteration(
                conn=conn,
                as_of=as_of_date,
                tushare_client=client,
            )

        if result["success"]:
            logger.info(f"Evolution success: {result['message']}")

            # Notify team if promoted
            if result["best_version"]:
                send_slack_notification(
                    f"New version promoted: {result['best_version']}\\n"
                    f"Improvement: {result['improvement_pct']:.1f}%"
                )
        else:
            logger.error(f"Evolution failed: {result['error']}")
            send_slack_notification(f"Evolution cycle failed: {result['error']}")

    except Exception as e:
        logger.error(f"Scheduler job failed: {e}")

# Schedule weekly evolution on Friday at 6 PM
scheduler = BackgroundScheduler()
scheduler.add_job(
    evolution_job,
    'cron',
    day_of_week='fri',
    hour=18,
    minute=0,
    args=['2025-03-21'],
    id='weekly_evolution',
)
scheduler.start()
    """)


def example_8_custom_metrics():
    """Example 8: Access detailed metrics from evaluations."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 8: Custom Metrics Analysis")
    logger.info("=" * 60)

    with get_db() as conn:
        active_config = EvolutionRegistry.get_active_version(conn)
        if not active_config:
            logger.warning("No active version found")
            return

        perf = PerformanceEvaluator.evaluate_performance(
            conn=conn,
            strategy_version=active_config.version,
            window_days=90,
        )

        print(f"Version {perf.version} Metrics:")
        print(f"\nReturns:")
        print(f"  Total: {perf.total_return:.2f}%")
        print(f"  Annualized: {perf.annualized_return:.2f}%")

        print(f"\nRisk-Adjusted:")
        print(f"  Sharpe Ratio: {perf.sharpe_ratio:.4f}")
        print(f"  Calmar Ratio: {perf.calmar_ratio:.4f}")
        print(f"  Info Ratio: {perf.information_ratio:.4f}")

        print(f"\nDrawdown & Risk:")
        print(f"  Max Drawdown: {perf.max_drawdown:.2f}%")

        print(f"\nTrading Activity:")
        print(f"  Win Rate: {perf.win_rate:.1f}%")
        print(f"  Total Trades: {perf.total_trades}")
        print(f"  Avg Turnover: {perf.avg_turnover:.2f}%")
        print(f"  Rebalances: {perf.num_rebalances}")


def main():
    """Run all examples."""
    logger.info("\n\n")
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║" + " OPUS AI SELF-EVOLUTION ENGINE - USAGE EXAMPLES ".center(58) + "║")
    logger.info("╚" + "═" * 58 + "╝")
    logger.info("\n")

    try:
        example_1_basic_evaluation()
        print("\n")

        example_5_version_registry()
        print("\n")

        example_2_version_comparison()
        print("\n")

        # example_3_ai_reflection()  # Requires API key
        # print("\n")

        # example_4_sandbox_backtest()  # Requires real data
        # print("\n")

        # example_6_full_evolution_cycle()  # Full cycle - may be slow
        # print("\n")

        example_7_scheduler_integration()
        print("\n")

        example_8_custom_metrics()
        print("\n")

    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
