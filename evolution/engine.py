"""
Main evolution orchestrator - coordinates the complete evolution cycle.
Evaluates -> Reflects -> Proposes -> Tests -> Promotes.
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

from loguru import logger

from data_pipeline.tushare_client import TushareClient
from momentum.config import MomentumConfig

from .evaluator import PerformanceEvaluator
from .reflector import ReflectorEngine
from .registry import EvolutionRegistry
from .sandbox import PromotionDecision, SandboxRunner


@dataclass
class MutationTestResult:
    """Result of testing a single mutation."""

    proposal_index: int
    baseline_version: str
    mutated_version: str
    promotion_decision: PromotionDecision
    success: bool
    error_message: str = ""


@dataclass
class EvolutionCycleResult:
    """Results of one complete evolution cycle."""

    timestamp: str
    active_version: str
    cycle_success: bool
    mutations_tested: int
    mutations_promoted: int
    best_version: Optional[str]
    best_improvement_pct: float
    test_results: List[MutationTestResult]
    error_message: str = ""

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            f"Evolution Cycle {self.timestamp}",
            f"  Active: {self.active_version}",
            f"  Mutations tested: {self.mutations_tested}",
            f"  Mutations promoted: {self.mutations_promoted}",
            f"  Best improvement: {self.best_improvement_pct:.1f}%",
        ]
        if self.best_version and self.best_version != self.active_version:
            lines.append(f"  Promoted: {self.best_version}")
        if not self.cycle_success:
            lines.append(f"  ERROR: {self.error_message}")
        return "\n".join(lines)


class EvolutionEngine:
    """Orchestrates strategy evolution cycles."""

    # Safety parameters
    MAX_DRAWDOWN_THRESHOLD = -25.0  # Don't promote anything worse than this
    MIN_SHARPE_THRESHOLD = 0.3  # Minimum acceptable risk-adjusted return
    MIN_IMPROVEMENT_PCT = 5.0  # Require 5% improvement to promote

    @staticmethod
    def run_evolution_cycle(
        conn: sqlite3.Connection,
        as_of: str,
        tushare_client: TushareClient,
        backtest_start_offset_days: int = 90,
        backtest_end_offset_days: int = 7,
    ) -> EvolutionCycleResult:
        """
        Run one complete evolution cycle.

        Steps:
        1. Load active configuration
        2. Evaluate recent performance
        3. Call Opus reflection
        4. Test mutation proposals in sandbox
        5. Promote best candidate if better than baseline

        Args:
            conn: Database connection
            as_of: Reference date (YYYY-MM-DD)
            tushare_client: TushareClient instance
            backtest_start_offset_days: Days back for backtest window
            backtest_end_offset_days: Days back for backtest end

        Returns:
            EvolutionCycleResult summarizing the cycle
        """
        timestamp = datetime.now().isoformat()
        logger.info(f"Starting evolution cycle at {timestamp}")

        try:
            # Step 1: Load active configuration
            active_config = EvolutionRegistry.get_active_version(conn)
            if not active_config:
                logger.info("No promoted version, creating baseline")
                active_config = MomentumConfig()
                EvolutionRegistry.register_version(conn, active_config, mutation_reason="Baseline initialization")
                EvolutionRegistry.promote_version(conn, active_config.version, reason="Initial baseline")

            logger.info(f"Active version: {active_config.version}")

            # Step 2: Evaluate recent performance
            perf_report = PerformanceEvaluator.evaluate_performance(
                conn, active_config.version, window_days=90
            )
            logger.info(f"Performance report:\n{perf_report.summary()}")

            # Step 3: Reflect using Opus
            recent_trades = []  # TODO: Fetch from trading log if available
            reflection = ReflectorEngine.reflect_on_performance(
                perf_report, active_config, recent_trades
            )
            logger.info(
                f"Reflection: {len(reflection.proposals)} proposals, "
                f"{len(reflection.key_issues)} issues"
            )

            # Step 4: Test mutations
            test_results: List[MutationTestResult] = []
            promoted_version = None
            best_improvement = 0.0

            # Get backtest date range
            as_of_date = datetime.strptime(as_of, "%Y-%m-%d")
            backtest_end = as_of_date - timedelta(days=backtest_end_offset_days)
            backtest_start = backtest_end - timedelta(days=backtest_start_offset_days)

            backtest_start_str = backtest_start.strftime("%Y-%m-%d")
            backtest_end_str = backtest_end.strftime("%Y-%m-%d")

            logger.info(
                f"Backtest window: {backtest_start_str} to {backtest_end_str}"
            )

            # Baseline backtest
            baseline_result = SandboxRunner.run_backtest(
                active_config,
                backtest_start_str,
                backtest_end_str,
                tushare_client,
            )
            logger.info(
                f"Baseline result: return={baseline_result.total_return:.2f}%, "
                f"sharpe={baseline_result.sharpe_ratio:.3f}"
            )

            # Test each proposal
            for i, proposal in enumerate(reflection.proposals):
                logger.info(f"Testing proposal {i+1}/{len(reflection.proposals)}")

                try:
                    # Create mutated config
                    mutated_config = active_config.mutate(
                        proposal.parameter_changes
                    )
                    mutated_version = mutated_config.version

                    # Register mutated version
                    EvolutionRegistry.register_version(
                        conn,
                        mutated_config,
                        parent_version=active_config.version,
                        mutation_reason=f"Proposal {i+1}: {proposal.rationale[:100]}",
                    )
                    logger.info(f"Created mutation {mutated_version}")

                    # Run sandbox backtest
                    candidate_result = SandboxRunner.run_backtest(
                        mutated_config,
                        backtest_start_str,
                        backtest_end_str,
                        tushare_client,
                    )

                    # Compare with baseline
                    promotion_decision = SandboxRunner.compare_with_baseline(
                        candidate_result,
                        baseline_result,
                        min_improvement_pct=EvolutionEngine.MIN_IMPROVEMENT_PCT,
                        min_sharpe=EvolutionEngine.MIN_SHARPE_THRESHOLD,
                        max_drawdown=EvolutionEngine.MAX_DRAWDOWN_THRESHOLD,
                    )

                    test_result = MutationTestResult(
                        proposal_index=i,
                        baseline_version=active_config.version,
                        mutated_version=mutated_version,
                        promotion_decision=promotion_decision,
                        success=True,
                    )
                    test_results.append(test_result)

                    logger.info(
                        f"Proposal {i+1} result: promote={promotion_decision.promote}, "
                        f"confidence={promotion_decision.confidence:.1%}\n"
                        f"{promotion_decision.reason}"
                    )

                    # Check if this is best improvement
                    if (
                        promotion_decision.promote
                        and promotion_decision.confidence > 0.5
                    ):
                        improvement = (
                            promotion_decision.metrics_comparison.get(
                                "overall_improvement_pct", 0
                            )
                        )
                        if improvement > best_improvement:
                            best_improvement = improvement
                            promoted_version = mutated_version

                except Exception as e:
                    logger.error(f"Proposal {i+1} test failed: {e}")
                    test_result = MutationTestResult(
                        proposal_index=i,
                        baseline_version=active_config.version,
                        mutated_version="",
                        promotion_decision=PromotionDecision(
                            promote=False,
                            confidence=0.0,
                            reason=f"Test failed: {e}",
                            metrics_comparison={},
                        ),
                        success=False,
                        error_message=str(e),
                    )
                    test_results.append(test_result)

            # Step 5: Promote best candidate
            mutations_promoted = 0
            if promoted_version and best_improvement > 0:
                try:
                    best_config = EvolutionRegistry.get_version(conn, promoted_version)
                    if best_config:
                        EvolutionRegistry.promote_version(
                            conn,
                            promoted_version,
                            compared_with=active_config.version,
                            confidence=0.8,
                            reason=f"Evolution cycle: {best_improvement:.1f}% improvement",
                        )
                        mutations_promoted = 1
                        logger.info(f"Promoted {promoted_version}")
                except Exception as e:
                    logger.error(f"Failed to promote {promoted_version}: {e}")
                    promoted_version = None

            # Summary
            mutations_tested = len(test_results)
            result = EvolutionCycleResult(
                timestamp=timestamp,
                active_version=active_config.version,
                cycle_success=True,
                mutations_tested=mutations_tested,
                mutations_promoted=mutations_promoted,
                best_version=promoted_version,
                best_improvement_pct=best_improvement,
                test_results=test_results,
            )

            logger.info(result.summary())
            return result

        except Exception as e:
            logger.error(f"Evolution cycle failed: {e}")
            return EvolutionCycleResult(
                timestamp=timestamp,
                active_version="unknown",
                cycle_success=False,
                mutations_tested=0,
                mutations_promoted=0,
                best_version=None,
                best_improvement_pct=0.0,
                test_results=[],
                error_message=str(e),
            )

    @staticmethod
    def run_full_iteration(
        conn: sqlite3.Connection,
        as_of: str,
        tushare_client: TushareClient,
    ) -> dict:
        """
        Run one complete evolution iteration (for scheduler).

        Args:
            conn: Database connection
            as_of: Reference date
            tushare_client: TushareClient instance

        Returns:
            Dict with iteration results
        """
        logger.info("Running full evolution iteration")

        try:
            result = EvolutionEngine.run_evolution_cycle(
                conn, as_of, tushare_client
            )

            return {
                "success": result.cycle_success,
                "active_version": result.active_version,
                "mutations_tested": result.mutations_tested,
                "mutations_promoted": result.mutations_promoted,
                "best_version": result.best_version,
                "improvement_pct": result.best_improvement_pct,
                "timestamp": result.timestamp,
                "message": result.summary(),
            }

        except Exception as e:
            logger.error(f"Full iteration failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
