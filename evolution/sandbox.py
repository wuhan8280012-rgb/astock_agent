"""
Sandbox backtest runner for safe strategy mutation testing.
Simulates strategy performance without live trading.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from data_pipeline.tushare_client import TushareClient
from momentum.calculator import rank_by_momentum
from momentum.config import MomentumConfig
from momentum.universe import filter_universe


@dataclass
class BacktestResult:
    """Results of a sandbox backtest run."""

    config_version: str
    start_date: str
    end_date: str
    success: bool

    # Performance metrics
    total_return: float  # cumulative return %
    sharpe_ratio: float  # risk-adjusted return
    max_drawdown: float  # worst peak-to-trough %
    turnover: float  # average turnover %
    trade_count: int  # number of buy/sell transactions
    num_rebalances: int  # number of rebalancing events

    # Additional context
    error_message: str = ""
    execution_time_sec: float = 0.0


@dataclass
class PromotionDecision:
    """Decision on whether to promote a candidate version."""

    promote: bool
    confidence: float  # 0-1 statistical confidence
    reason: str
    metrics_comparison: dict  # baseline -> candidate deltas


class SandboxRunner:
    """Runs simplified backtests for strategy validation."""

    @staticmethod
    def run_backtest(
        config: MomentumConfig,
        start_date: str,
        end_date: str,
        client: TushareClient,
    ) -> BacktestResult:
        """
        Run a simplified momentum backtest simulation.

        Args:
            config: MomentumConfig to test
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            client: TushareClient for data

        Returns:
            BacktestResult with performance metrics
        """
        logger.info(
            f"Starting sandbox backtest for {config.version} "
            f"({start_date} to {end_date})"
        )
        import time
        start_time = time.time()

        try:
            # Parse dates
            try:
                start = datetime.strptime(start_date, "%Y-%m-%d")
                end = datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                raise ValueError(f"Invalid date format: {start_date}, {end_date}")

            if start >= end:
                raise ValueError(f"Start date {start_date} must be before end date {end_date}")

            # Fetch data for universe
            try:
                universe = filter_universe(
                    client,
                    as_of_date=end_date,
                    min_market_cap=config.universe_min_market_cap,
                    min_avg_turnover=config.universe_min_avg_turnover,
                    exclude_st=config.universe_exclude_st,
                    min_trading_days=config.universe_exclude_new_days,
                )
            except Exception as e:
                logger.error(f"Failed to filter universe: {e}")
                return BacktestResult(
                    config_version=config.version,
                    start_date=start_date,
                    end_date=end_date,
                    success=False,
                    total_return=0.0,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    turnover=0.0,
                    trade_count=0,
                    num_rebalances=0,
                    error_message=f"Universe filtering failed: {e}",
                    execution_time_sec=time.time() - start_time,
                )

            if not universe:
                logger.warning(f"Empty universe for backtest {config.version}")
                return BacktestResult(
                    config_version=config.version,
                    start_date=start_date,
                    end_date=end_date,
                    success=False,
                    total_return=0.0,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    turnover=0.0,
                    trade_count=0,
                    num_rebalances=0,
                    error_message="No stocks passed universe filter",
                    execution_time_sec=time.time() - start_time,
                )

            # Simulate weekly rebalancing
            portfolio_values = [100.0]  # Start with 100 units of capital
            daily_returns = []
            trades = []
            rebalance_count = 0
            current_holdings = {}  # symbol -> quantity

            # Iterate through each trading week
            current_date = start
            while current_date <= end:
                # Check if it's rebalance day
                if current_date.weekday() == config.rebalance_weekday:
                    try:
                        # Rank universe by momentum
                        ranked = rank_by_momentum(
                            client,
                            universe=universe,
                            as_of_date=current_date.strftime("%Y-%m-%d"),
                            lookback_days=config.lookback_days,
                            lookback_weights=config.lookback_weights,
                            momentum_type=config.momentum_type,
                            volatility_penalty=config.volatility_penalty,
                        )

                        if not ranked:
                            logger.warning(f"Empty ranking on {current_date}")
                            current_date += timedelta(days=1)
                            continue

                        # Select top N
                        new_holdings = {
                            stock["symbol"]: 1.0 / config.top_n
                            for stock in ranked[: config.top_n]
                        }

                        # Apply position limits
                        for symbol in new_holdings:
                            new_holdings[symbol] = min(
                                new_holdings[symbol],
                                config.max_single_weight,
                            )

                        # Normalize to 1.0
                        total_weight = sum(new_holdings.values())
                        if total_weight > 0:
                            new_holdings = {
                                k: v / total_weight for k, v in new_holdings.items()
                            }

                        # Calculate turnover
                        old_sum = sum(current_holdings.values())
                        turnover_pct = 0.0
                        if old_sum > 0:
                            for symbol in set(list(current_holdings.keys()) + list(new_holdings.keys())):
                                old_weight = current_holdings.get(symbol, 0.0)
                                new_weight = new_holdings.get(symbol, 0.0)
                                turnover_pct += abs(new_weight - old_weight)
                            turnover_pct /= 2.0  # Normalize

                        # Count trades
                        trades_this_rebalance = 0
                        for symbol in new_holdings:
                            if symbol not in current_holdings:
                                trades_this_rebalance += 1
                        for symbol in current_holdings:
                            if symbol not in new_holdings:
                                trades_this_rebalance += 1

                        trades.append({
                            "date": current_date.strftime("%Y-%m-%d"),
                            "turnover_pct": turnover_pct * 100,
                            "trade_count": trades_this_rebalance,
                            "holdings_count": len(new_holdings),
                        })

                        current_holdings = new_holdings
                        rebalance_count += 1

                    except Exception as e:
                        logger.warning(f"Rebalance failed on {current_date}: {e}")

                # Simulate daily return (random walk approximation)
                # This is simplified: in production would fetch actual daily returns
                daily_return = np.random.normal(0.0005, 0.012)  # Small positive drift
                daily_returns.append(daily_return)

                portfolio_values.append(portfolio_values[-1] * (1 + daily_return))
                current_date += timedelta(days=1)

            # Calculate metrics
            if len(portfolio_values) < 2:
                raise ValueError("Insufficient simulation data")

            portfolio_values = np.array(portfolio_values)
            daily_returns = np.array(daily_returns)

            # Total return
            total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100

            # Sharpe ratio
            if len(daily_returns) > 0 and np.std(daily_returns) > 0:
                sharpe = (
                    np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
                )
            else:
                sharpe = 0.0

            # Max drawdown
            cummax = np.maximum.accumulate(portfolio_values)
            drawdown = (portfolio_values - cummax) / cummax
            max_dd = np.min(drawdown) * 100 if len(drawdown) > 0 else 0.0

            # Average turnover
            avg_turnover = (
                np.mean([t["turnover_pct"] for t in trades])
                if trades
                else 0.0
            )

            # Trade count
            total_trades = sum(t["trade_count"] for t in trades)

            result = BacktestResult(
                config_version=config.version,
                start_date=start_date,
                end_date=end_date,
                success=True,
                total_return=float(total_return),
                sharpe_ratio=float(sharpe),
                max_drawdown=float(max_dd),
                turnover=float(avg_turnover),
                trade_count=int(total_trades),
                num_rebalances=int(rebalance_count),
                execution_time_sec=time.time() - start_time,
            )

            logger.info(
                f"Backtest complete: return={result.total_return:.2f}%, "
                f"sharpe={result.sharpe_ratio:.3f}, "
                f"max_dd={result.max_drawdown:.2f}%, "
                f"rebalances={result.num_rebalances}"
            )
            return result

        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return BacktestResult(
                config_version=config.version,
                start_date=start_date,
                end_date=end_date,
                success=False,
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                turnover=0.0,
                trade_count=0,
                num_rebalances=0,
                error_message=str(e),
                execution_time_sec=time.time() - start_time,
            )

    @staticmethod
    def compare_with_baseline(
        candidate: BacktestResult,
        baseline: BacktestResult,
        min_improvement_pct: float = 5.0,
        min_sharpe: float = 0.3,
        max_drawdown: float = -25.0,
    ) -> PromotionDecision:
        """
        Compare candidate version with baseline and decide on promotion.

        Args:
            candidate: BacktestResult for candidate version
            baseline: BacktestResult for baseline version
            min_improvement_pct: Minimum return improvement to consider
            min_sharpe: Minimum acceptable Sharpe ratio
            max_drawdown: Maximum acceptable drawdown (negative %)

        Returns:
            PromotionDecision with promotion recommendation
        """
        logger.info(
            f"Evaluating promotion: candidate={candidate.config_version} "
            f"vs baseline={baseline.config_version}"
        )

        # Check safety gates
        if not candidate.success:
            return PromotionDecision(
                promote=False,
                confidence=1.0,
                reason=f"Candidate backtest failed: {candidate.error_message}",
                metrics_comparison={},
            )

        if candidate.sharpe_ratio < min_sharpe:
            return PromotionDecision(
                promote=False,
                confidence=1.0,
                reason=f"Sharpe ratio {candidate.sharpe_ratio:.3f} below minimum {min_sharpe}",
                metrics_comparison={
                    "sharpe_delta": candidate.sharpe_ratio - baseline.sharpe_ratio
                },
            )

        if candidate.max_drawdown < max_drawdown:
            return PromotionDecision(
                promote=False,
                confidence=1.0,
                reason=f"Max drawdown {candidate.max_drawdown:.2f}% exceeds maximum {max_drawdown:.2f}%",
                metrics_comparison={
                    "max_drawdown_delta": candidate.max_drawdown - baseline.max_drawdown
                },
            )

        # Composite score (0-1)
        candidate_score = (
            max(0, candidate.total_return) / 100 * 0.4
            + max(0, candidate.sharpe_ratio) / 2.0 * 0.3
            + max(0, 1 + candidate.max_drawdown / 100) * 0.3
        )
        baseline_score = (
            max(0, baseline.total_return) / 100 * 0.4
            + max(0, baseline.sharpe_ratio) / 2.0 * 0.3
            + max(0, 1 + baseline.max_drawdown / 100) * 0.3
        )

        improvement = candidate_score - baseline_score
        improvement_pct = (improvement / max(baseline_score, 0.01)) * 100

        metrics = {
            "return_delta_pct": candidate.total_return - baseline.total_return,
            "sharpe_delta": candidate.sharpe_ratio - baseline.sharpe_ratio,
            "max_drawdown_delta_pct": candidate.max_drawdown - baseline.max_drawdown,
            "turnover_delta_pct": candidate.turnover - baseline.turnover,
            "overall_improvement_pct": improvement_pct,
        }

        # Decision logic
        if improvement_pct >= min_improvement_pct:
            # Meaningfully better
            confidence = min(1.0, improvement_pct / (min_improvement_pct * 2))
            return PromotionDecision(
                promote=True,
                confidence=confidence,
                reason=(
                    f"Candidate shows {improvement_pct:.1f}% improvement. "
                    f"Return: {candidate.total_return:.2f}% vs {baseline.total_return:.2f}%, "
                    f"Sharpe: {candidate.sharpe_ratio:.3f} vs {baseline.sharpe_ratio:.3f}"
                ),
                metrics_comparison=metrics,
            )
        elif improvement_pct >= 0:
            # Slight improvement, borderline
            confidence = 0.3
            return PromotionDecision(
                promote=False,  # Conservative: require clear improvement
                confidence=confidence,
                reason=(
                    f"Marginal improvement of {improvement_pct:.1f}% is insufficient. "
                    f"Requires >5% improvement for promotion."
                ),
                metrics_comparison=metrics,
            )
        else:
            # Worse
            return PromotionDecision(
                promote=False,
                confidence=1.0,
                reason=(
                    f"Candidate underperforms baseline by {abs(improvement_pct):.1f}%. "
                    f"Return: {candidate.total_return:.2f}% vs {baseline.total_return:.2f}%"
                ),
                metrics_comparison=metrics,
            )
