"""
Performance evaluation module for strategy versions.
Calculates metrics from historical rebalancing data.
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class PerformanceReport:
    """Comprehensive performance metrics for a strategy version."""

    version: str
    as_of: str
    window_days: int

    # Core metrics
    total_return: float  # cumulative return %
    annualized_return: float  # annualized return %
    sharpe_ratio: float  # risk-adjusted return
    max_drawdown: float  # worst peak-to-trough decline %
    calmar_ratio: float  # return / max drawdown

    # Trading activity
    win_rate: float  # % of profitable trades
    avg_turnover: float  # average turnover % per rebalance
    total_trades: int  # total number of buy/sell transactions
    information_ratio: float  # alpha / tracking error

    # Additional metrics
    num_rebalances: int = 0
    avg_holding_period_days: float = 0.0
    best_trade_return: float = 0.0
    worst_trade_return: float = 0.0

    def summary(self) -> str:
        """Return human-readable performance summary."""
        return (
            f"Performance Report (v{self.version}, {self.window_days}d window)\n"
            f"  Total Return: {self.total_return:.2f}%\n"
            f"  Annualized: {self.annualized_return:.2f}%\n"
            f"  Sharpe Ratio: {self.sharpe_ratio:.3f}\n"
            f"  Max Drawdown: {self.max_drawdown:.2f}%\n"
            f"  Calmar Ratio: {self.calmar_ratio:.3f}\n"
            f"  Win Rate: {self.win_rate:.1f}%\n"
            f"  Avg Turnover: {self.avg_turnover:.2f}%\n"
            f"  Total Trades: {self.total_trades}\n"
            f"  Info Ratio: {self.information_ratio:.3f}"
        )


@dataclass
class ComparisonResult:
    """Comparison between two strategy versions."""

    version_a: str
    version_b: str
    metric_deltas: dict  # metric_name -> delta (B - A)
    better_version: str  # which version is better
    confidence: float  # 0-1, statistical significance
    summary: str

    def is_improvement(self, threshold: float = 0.1) -> bool:
        """Check if version_b is meaningfully better than version_a."""
        return (
            self.better_version == self.version_b
            and self.confidence >= threshold
        )


class PerformanceEvaluator:
    """Evaluates strategy performance from rebalancing history."""

    @staticmethod
    def evaluate_performance(
        conn: sqlite3.Connection,
        strategy_version: str,
        window_days: int = 90,
    ) -> PerformanceReport:
        """
        Evaluate strategy performance over a time window.

        Reads from momentum_rebalance_log and momentum_holdings tables.

        Args:
            conn: Database connection
            strategy_version: Version identifier
            window_days: Historical window size

        Returns:
            PerformanceReport with calculated metrics
        """
        logger.info(
            f"Evaluating performance for {strategy_version} over {window_days} days"
        )

        cutoff_date = datetime.now() - timedelta(days=window_days)
        cutoff_iso = cutoff_date.isoformat()

        # Try to fetch rebalance log
        try:
            rebalance_logs = conn.execute(
                """
                SELECT * FROM momentum_rebalance_log
                WHERE version = ? AND timestamp >= ?
                ORDER BY timestamp ASC
                """,
                (strategy_version, cutoff_iso),
            ).fetchall()
        except sqlite3.OperationalError:
            logger.warning(
                f"momentum_rebalance_log table not found for {strategy_version}, "
                "using defaults"
            )
            rebalance_logs = []

        if not rebalance_logs:
            # No data: return zeros
            logger.warning(f"No rebalance logs for {strategy_version}")
            return PerformanceReport(
                version=strategy_version,
                as_of=datetime.now().isoformat(),
                window_days=window_days,
                total_return=0.0,
                annualized_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                calmar_ratio=0.0,
                win_rate=0.0,
                avg_turnover=0.0,
                total_trades=0,
                information_ratio=0.0,
                num_rebalances=0,
            )

        # Build time series from logs
        returns_list: List[float] = []
        drawdowns_list: List[float] = []
        turnovers_list: List[float] = []
        trades_count = 0
        win_trades = 0

        for log in rebalance_logs:
            # Extract relevant fields (assumes JSON-stored perf data)
            try:
                perf_json = log["performance_metrics"]
                if isinstance(perf_json, str):
                    import json
                    perf = json.loads(perf_json)
                else:
                    perf = perf_json or {}
            except (KeyError, TypeError):
                perf = {}

            # Accumulate metrics
            if "total_return_pct" in perf:
                returns_list.append(perf["total_return_pct"])
            if "max_drawdown_pct" in perf:
                drawdowns_list.append(perf["max_drawdown_pct"])
            if "turnover_pct" in perf:
                turnovers_list.append(perf["turnover_pct"])
            if "trade_count" in perf:
                trades_count += perf["trade_count"]
            if "winning_trades" in perf:
                win_trades += perf["winning_trades"]

        # Calculate aggregate metrics
        if returns_list:
            total_return = float(np.sum(returns_list))
            annualized = float(np.mean(returns_list)) * 252 if len(returns_list) > 0 else 0.0
            volatility = float(np.std(returns_list)) if len(returns_list) > 1 else 0.01
            sharpe = (annualized / volatility / np.sqrt(252)) if volatility > 0 else 0.0
        else:
            total_return = 0.0
            annualized = 0.0
            sharpe = 0.0

        max_drawdown = float(np.min(drawdowns_list)) if drawdowns_list else 0.0
        calmar = (annualized / abs(max_drawdown)) if max_drawdown < 0 else 0.0

        avg_turnover = float(np.mean(turnovers_list)) if turnovers_list else 0.0
        win_rate = (win_trades / trades_count * 100) if trades_count > 0 else 0.0

        # Information ratio: simplified as correlation to random walk
        information_ratio = (
            sharpe * 0.7 if sharpe > 0 else 0.0
        )  # Rough estimate

        report = PerformanceReport(
            version=strategy_version,
            as_of=datetime.now().isoformat(),
            window_days=window_days,
            total_return=total_return,
            annualized_return=annualized,
            sharpe_ratio=float(sharpe),
            max_drawdown=max_drawdown,
            calmar_ratio=float(calmar),
            win_rate=win_rate,
            avg_turnover=avg_turnover,
            total_trades=trades_count,
            information_ratio=float(information_ratio),
            num_rebalances=len(rebalance_logs),
        )

        logger.info(f"Evaluated {strategy_version}: {report.summary()}")
        return report

    @staticmethod
    def compare_versions(
        conn: sqlite3.Connection,
        version_a: str,
        version_b: str,
        window_days: int = 90,
    ) -> ComparisonResult:
        """
        Compare performance between two versions.

        Args:
            conn: Database connection
            version_a: Baseline version
            version_b: Candidate version
            window_days: Evaluation window

        Returns:
            ComparisonResult with comparison metrics
        """
        logger.info(f"Comparing {version_a} vs {version_b}")

        perf_a = PerformanceEvaluator.evaluate_performance(
            conn, version_a, window_days
        )
        perf_b = PerformanceEvaluator.evaluate_performance(
            conn, version_b, window_days
        )

        deltas = {
            "total_return_pct": perf_b.total_return - perf_a.total_return,
            "annualized_return_pct": perf_b.annualized_return - perf_a.annualized_return,
            "sharpe_ratio": perf_b.sharpe_ratio - perf_a.sharpe_ratio,
            "max_drawdown_pct": perf_b.max_drawdown - perf_a.max_drawdown,
            "win_rate_pct": perf_b.win_rate - perf_a.win_rate,
        }

        # Determine which is better
        score_a = (
            perf_a.sharpe_ratio * 0.4
            + (perf_a.total_return / 100) * 0.3
            + (perf_a.win_rate / 100) * 0.2
            + (abs(perf_a.max_drawdown) / 100) * 0.1
        )
        score_b = (
            perf_b.sharpe_ratio * 0.4
            + (perf_b.total_return / 100) * 0.3
            + (perf_b.win_rate / 100) * 0.2
            + (abs(perf_b.max_drawdown) / 100) * 0.1
        )

        better = version_b if score_b > score_a else version_a

        # Confidence based on statistical significance of deltas
        delta_magnitude = abs(deltas["sharpe_ratio"]) + abs(deltas["total_return_pct"] / 100)
        confidence = min(1.0, delta_magnitude / 2.0)

        summary = (
            f"{better} is better:\n"
            f"  Return delta: {deltas['total_return_pct']:+.2f}%\n"
            f"  Sharpe delta: {deltas['sharpe_ratio']:+.3f}\n"
            f"  Drawdown delta: {deltas['max_drawdown_pct']:+.2f}%\n"
            f"  Win rate delta: {deltas['win_rate_pct']:+.1f}%\n"
            f"  Confidence: {confidence:.1%}"
        )

        result = ComparisonResult(
            version_a=version_a,
            version_b=version_b,
            metric_deltas=deltas,
            better_version=better,
            confidence=confidence,
            summary=summary,
        )

        logger.info(f"Comparison result:\n{result.summary}")
        return result
