"""
Momentum rotation strategy backtest engine.

Simulates weekly rebalancing through historical trading days.
Uses the same momentum calculation and ranking logic as live trading.

Key features:
- T+1 execution (signals generated at close, executed at next open)
- Daily stop-loss checks between rebalance days
- Portfolio tracking, trade logging, position history
- Comprehensive metrics: returns, risk, Sharpe, max drawdown, Calmar, win rate, turnover
- Benchmark comparison (CSI300)
- Local price data caching to avoid repeated API calls

Usage:
    python -m backtest.momentum_backtest --start 20240101 --end 20251231
    python -m backtest.momentum_backtest --start 20240101 --end 20251231 --config momentum_v2.json
    python -m backtest.momentum_backtest --start 20240101 --end 20251231 --initial-capital 500000
"""

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.tushare_client import TushareClient
from momentum.config import MomentumConfig
from momentum.calculator import rank_by_momentum
from momentum.rebalancer import compute_rebalance
from momentum.risk_control import check_regime_filter, apply_stop_loss
from momentum.universe import filter_universe


@dataclass
class TradeRecord:
    """Single trade execution record."""
    date: str
    ts_code: str
    action: str  # "buy" or "sell"
    price: float
    shares: int
    value: float
    reason: str


@dataclass
class Position:
    """Current or historical position."""
    ts_code: str
    entry_date: str
    entry_price: float
    shares: int
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    momentum_rank: Optional[int] = None


@dataclass
class DailySnapshot:
    """Daily portfolio state for metrics calculation."""
    date: str
    total_value: float
    cash: float
    equity_value: float
    positions: List[Position] = field(default_factory=list)
    daily_return: float = 0.0


@dataclass
class BacktestReport:
    """Comprehensive backtest results."""
    summary: Dict = field(default_factory=dict)
    daily_values: List[DailySnapshot] = field(default_factory=list)
    trades: List[TradeRecord] = field(default_factory=list)
    monthly_returns: Dict[str, float] = field(default_factory=dict)
    position_history: List[Position] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)


class MomentumBacktester:
    """Momentum rotation strategy backtest engine."""

    def __init__(
        self,
        config: MomentumConfig,
        initial_capital: float = 1_000_000,
    ):
        """
        Initialize backtest engine.

        Args:
            config: MomentumConfig instance
            initial_capital: Starting portfolio value in CNY
        """
        self.config = config
        self.initial_capital = initial_capital

        # Portfolio state
        self.cash = initial_capital
        self.equity_value = 0.0
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []

        # Historical tracking
        self.daily_snapshots: List[DailySnapshot] = []
        self.trades: List[TradeRecord] = []
        self.position_history: List[Position] = []

        # Data caching
        self.price_cache: Dict[str, pd.DataFrame] = {}
        self.trade_calendar: List[str] = []

        # Client
        self.client = TushareClient()

        logger.info(
            f"Initialized MomentumBacktester: config_v{config.version}, "
            f"capital={initial_capital:,.0f}"
        )

    def run(self, start_date: str, end_date: str) -> BacktestReport:
        """
        Run backtest simulation.

        Args:
            start_date: Start date (YYYYMMDD format)
            end_date: End date (YYYYMMDD format)

        Returns:
            BacktestReport with results
        """
        logger.info(f"Starting backtest: {start_date} -> {end_date}")

        # Get trade calendar
        cal_start = f"{int(start_date[:4]) - 1}0101"
        cal_end = f"{int(end_date[:4]) + 1}1231"
        all_trade_dates = self.client.get_trade_calendar(cal_start, cal_end)
        self.trade_calendar = [d for d in all_trade_dates if start_date <= d <= end_date]

        if not self.trade_calendar:
            logger.error(f"No trade dates found in range {start_date}-{end_date}")
            return self._create_empty_report()

        logger.info(f"Trading days in range: {len(self.trade_calendar)}")

        # Main backtest loop
        last_rebalance_date = None
        peak_value = self.initial_capital

        for idx, trade_date in enumerate(self.trade_calendar):
            try:
                # Check if rebalance day (weekday + weekly frequency)
                should_rebalance = self._should_rebalance(trade_date, last_rebalance_date)

                if should_rebalance:
                    logger.debug(f"Rebalance signal on {trade_date}")
                    self._execute_rebalance(trade_date)
                    last_rebalance_date = trade_date
                else:
                    # Daily stop-loss checks
                    self._apply_daily_stop_loss(trade_date)

                # Update daily value and snapshot
                total_value = self._calculate_portfolio_value(trade_date)
                peak_value = max(peak_value, total_value)

                daily_return = 0.0
                if self.daily_snapshots:
                    prev_value = self.daily_snapshots[-1].total_value
                    daily_return = (total_value - prev_value) / prev_value if prev_value > 0 else 0.0

                snapshot = DailySnapshot(
                    date=trade_date,
                    total_value=total_value,
                    cash=self.cash,
                    equity_value=total_value - self.cash,
                    positions=list(self.positions.values()),
                    daily_return=daily_return,
                )
                self.daily_snapshots.append(snapshot)

            except Exception as e:
                logger.error(f"Error on {trade_date}: {e}")
                continue

        logger.info(f"Backtest completed. Total trades: {len(self.trades)}")

        # Generate report
        return self._generate_report()

    def _should_rebalance(self, current_date: str, last_rebalance_date: Optional[str]) -> bool:
        """Check if rebalance should occur on current_date."""
        # Parse dates
        curr_dt = datetime.strptime(current_date, "%Y%m%d")

        # Check weekday (0=Mon, 4=Fri)
        if curr_dt.weekday() != self.config.rebalance_weekday:
            return False

        # If first rebalance or enough time passed since last rebalance (7 days)
        if last_rebalance_date is None:
            return True

        last_dt = datetime.strptime(last_rebalance_date, "%Y%m%d")
        days_since = (curr_dt - last_dt).days

        return days_since >= 7

    def _execute_rebalance(self, signal_date: str) -> None:
        """
        Execute rebalancing on signal_date, trades execute on next trading day (T+1).

        Args:
            signal_date: Date when signal is generated (YYYYMMDD)
        """
        logger.info(f"=== Rebalance on {signal_date} ===")

        # Find next trading day for execution
        curr_idx = self.trade_calendar.index(signal_date)
        if curr_idx >= len(self.trade_calendar) - 1:
            logger.warning(f"No next trading day after {signal_date}")
            return

        execution_date = self.trade_calendar[curr_idx + 1]

        try:
            # Step 1: Filter universe
            universe = filter_universe(self.client, signal_date, self.config)
            if universe.empty:
                logger.warning(f"Empty universe on {signal_date}")
                return

            # Step 2: Rank by momentum
            ranked = rank_by_momentum(universe, self.client, signal_date, self.config)
            if ranked.empty:
                logger.warning(f"No momentum scores on {signal_date}")
                return

            # Step 3: Check market regime
            regime_info = check_regime_filter(self.client, signal_date, self.config)
            logger.debug(f"Market regime: {regime_info.regime}")

            # Step 4: Compute rebalance (what trades to make)
            current_holdings = [
                {
                    "ts_code": ts_code,
                    "shares": pos.shares,
                    "entry_price": pos.entry_price,
                    "momentum_rank": pos.momentum_rank,
                }
                for ts_code, pos in self.positions.items()
            ]

            # Get current prices for rebalance calculation
            current_prices = self._get_prices_for_date(signal_date, ranked["ts_code"].tolist())

            total_capital = self.cash + sum(
                self.positions[ts].shares * current_prices.get(ts, 0)
                for ts in self.positions
                if ts in current_prices
            )

            rebalance_result = compute_rebalance(
                current_holdings,
                ranked,
                self.config,
                total_capital,
                current_prices,
            )

            logger.info(
                f"Rebalance plan: {len(rebalance_result.buys)} buys, "
                f"{len(rebalance_result.sells)} sells, "
                f"turnover={rebalance_result.turnover_pct:.1f}%"
            )

            # Step 5: Execute trades on T+1 (execution_date)
            self._execute_trades(
                execution_date,
                rebalance_result.sells,
                rebalance_result.buys,
                ranked,
            )

        except Exception as e:
            logger.error(f"Rebalance failed on {signal_date}: {e}", exc_info=True)

    def _execute_trades(
        self,
        execution_date: str,
        sells: List,
        buys: List,
        ranked: pd.DataFrame,
    ) -> None:
        """
        Execute sell and buy trades on execution_date.

        Args:
            execution_date: Date to execute trades (YYYYMMDD)
            sells: List of Trade objects to sell
            buys: List of Trade objects to buy
            ranked: DataFrame with momentum ranks
        """
        # Get prices for execution date
        all_codes = set()
        for pos in self.positions.keys():
            all_codes.add(pos)
        for buy in buys:
            all_codes.add(buy.ts_code)

        prices = self._get_prices_for_date(execution_date, list(all_codes))

        if not prices:
            logger.warning(f"No price data available for {execution_date}")
            return

        # Execute sells first
        for sell in sells:
            ts_code = sell.ts_code
            if ts_code in self.positions:
                pos = self.positions[ts_code]
                price = prices.get(ts_code)

                if price is None or price <= 0:
                    logger.warning(f"Invalid price for {ts_code} on {execution_date}: {price}")
                    continue

                # Check limit-down (can't sell at limit-down)
                if self._is_limit_down(ts_code, execution_date):
                    logger.debug(f"Skip sell {ts_code}: limit-down on {execution_date}")
                    continue

                value = pos.shares * price
                commission = value * 0.0025  # 0.25%
                stamp_tax = value * 0.001  # 0.1%
                net_proceeds = value - commission - stamp_tax

                self.cash += net_proceeds

                # Record trade
                self.trades.append(TradeRecord(
                    date=execution_date,
                    ts_code=ts_code,
                    action="sell",
                    price=price,
                    shares=pos.shares,
                    value=value,
                    reason=sell.reason,
                ))

                # Close position
                pos.exit_date = execution_date
                pos.exit_price = price
                self.closed_positions.append(pos)
                del self.positions[ts_code]

                logger.info(
                    f"SELL {ts_code}: {pos.shares} @ {price:.2f} = {value:,.0f} CNY "
                    f"(net: {net_proceeds:,.0f})"
                )

        # Execute buys
        for buy in buys:
            ts_code = buy.ts_code
            price = prices.get(ts_code)

            if price is None or price <= 0:
                logger.warning(f"Invalid price for {ts_code} on {execution_date}: {price}")
                continue

            # Check limit-up (can't buy at limit-up)
            if self._is_limit_up(ts_code, execution_date):
                logger.debug(f"Skip buy {ts_code}: limit-up on {execution_date}")
                continue

            # Calculate shares to buy (equal weight strategy)
            target_value = (self.cash + self._calculate_equity_value()) / self.config.top_n
            target_value = min(target_value, self.cash * 0.95)  # Don't deploy all cash
            shares = int(target_value / price)

            if shares <= 0:
                logger.debug(f"Insufficient cash for {ts_code}: need {target_value:,.0f}, have {self.cash:,.0f}")
                continue

            cost = shares * price
            commission = cost * 0.0025  # 0.25%
            total_cost = cost + commission

            if total_cost > self.cash:
                logger.debug(f"Insufficient cash for {ts_code}: need {total_cost:,.0f}, have {self.cash:,.0f}")
                continue

            self.cash -= total_cost

            # Find momentum rank for this stock
            rank_row = ranked[ranked["ts_code"] == ts_code]
            momentum_rank = int(rank_row["momentum_rank"].iloc[0]) if not rank_row.empty else None

            # Create or update position
            if ts_code in self.positions:
                pos = self.positions[ts_code]
                pos.shares += shares
                pos.entry_price = (pos.entry_price * pos.shares + cost) / (pos.shares + shares)
                pos.momentum_rank = momentum_rank
            else:
                pos = Position(
                    ts_code=ts_code,
                    entry_date=execution_date,
                    entry_price=price,
                    shares=shares,
                    momentum_rank=momentum_rank,
                )
                self.positions[ts_code] = pos

            # Record trade
            self.trades.append(TradeRecord(
                date=execution_date,
                ts_code=ts_code,
                action="buy",
                price=price,
                shares=shares,
                value=cost,
                reason=buy.reason,
            ))

            logger.info(
                f"BUY {ts_code}: {shares} @ {price:.2f} = {cost:,.0f} CNY "
                f"(rank={momentum_rank})"
            )

    def _apply_daily_stop_loss(self, trade_date: str) -> None:
        """
        Check and apply stop losses on non-rebalance days.

        Args:
            trade_date: Current trading date (YYYYMMDD)
        """
        if not self.positions:
            return

        ts_codes = list(self.positions.keys())
        prices = self._get_prices_for_date(trade_date, ts_codes)

        if not prices:
            return

        holdings = [
            {
                "ts_code": ts,
                "shares": self.positions[ts].shares,
                "entry_price": self.positions[ts].entry_price,
            }
            for ts in ts_codes
        ]

        remaining, stopped_out = apply_stop_loss(holdings, prices, self.config)

        # Liquidate stopped-out positions
        for ts_code in stopped_out:
            if ts_code not in self.positions:
                continue

            pos = self.positions[ts_code]
            price = prices[ts_code]
            value = pos.shares * price
            commission = value * 0.0025
            net_proceeds = value - commission

            self.cash += net_proceeds

            self.trades.append(TradeRecord(
                date=trade_date,
                ts_code=ts_code,
                action="sell",
                price=price,
                shares=pos.shares,
                value=value,
                reason=f"Stop loss: {self.config.stop_loss_pct*100:.1f}%",
            ))

            pos.exit_date = trade_date
            pos.exit_price = price
            self.closed_positions.append(pos)
            del self.positions[ts_code]

            logger.info(f"STOP LOSS {ts_code} @ {price:.2f}")

    def _calculate_portfolio_value(self, trade_date: str) -> float:
        """Calculate total portfolio value on trade_date."""
        if not self.positions:
            return self.cash

        ts_codes = list(self.positions.keys())
        prices = self._get_prices_for_date(trade_date, ts_codes)

        if not prices:
            return self.cash

        equity_value = sum(
            self.positions[ts].shares * prices.get(ts, 0)
            for ts in ts_codes
            if ts in prices
        )

        return self.cash + equity_value

    def _calculate_equity_value(self) -> float:
        """Calculate current equity value across all positions."""
        equity_value = 0.0
        for ts, pos in self.positions.items():
            cache_df = self.price_cache.get(ts)
            if cache_df is None or len(cache_df) == 0:
                continue
            close_series = pd.to_numeric(cache_df["close"], errors="coerce").dropna()
            if close_series.empty:
                continue
            equity_value += pos.shares * float(close_series.iloc[-1])
        return equity_value

    def _get_prices_for_date(self, trade_date: str, ts_codes: List[str]) -> Dict[str, float]:
        """
        Get closing prices for stocks on trade_date.

        Uses cache to avoid repeated API calls. Returns dict of ts_code -> close_price.

        Args:
            trade_date: Date in YYYYMMDD format
            ts_codes: List of ts_code strings

        Returns:
            Dict mapping ts_code to closing price
        """
        if not ts_codes:
            return {}

        prices = {}
        uncached_codes = []

        # Check cache
        for ts_code in ts_codes:
            if ts_code in self.price_cache:
                cache_df = self.price_cache[ts_code]
                matching = cache_df[cache_df["trade_date"] == trade_date]
                if not matching.empty:
                    prices[ts_code] = float(matching.iloc[0]["close"])
                else:
                    uncached_codes.append(ts_code)
            else:
                uncached_codes.append(ts_code)

        # Fetch uncached
        if uncached_codes:
            try:
                # Fetch daily data for uncached codes
                daily_data = self.client.daily(
                    ts_code=",".join(uncached_codes),
                    start_date=self._get_start_date_for_fetch(trade_date),
                    end_date=trade_date,
                    fields=["ts_code", "trade_date", "close"],
                )

                if not daily_data.empty:
                    daily_data["close"] = pd.to_numeric(daily_data["close"], errors="coerce")

                    # Cache and extract prices for requested date
                    for ts_code in uncached_codes:
                        ts_data = daily_data[daily_data["ts_code"] == ts_code].copy()
                        if ts_data.empty:
                            continue

                        # Store in cache
                        self.price_cache[ts_code] = ts_data

                        # Extract price for requested date
                        matching = ts_data[ts_data["trade_date"] == trade_date]
                        if not matching.empty:
                            prices[ts_code] = float(matching.iloc[0]["close"])

            except Exception as e:
                logger.debug(f"Failed to fetch prices for {len(uncached_codes)} codes on {trade_date}: {e}")

        return prices

    def _get_start_date_for_fetch(self, end_date: str, lookback_days: int = 30) -> str:
        """Calculate start date for price fetching with buffer."""
        end_dt = datetime.strptime(end_date, "%Y%m%d")
        start_dt = end_dt - timedelta(days=lookback_days)
        return start_dt.strftime("%Y%m%d")

    def _is_limit_up(self, ts_code: str, trade_date: str) -> bool:
        """Check if stock hit limit-up on trade_date."""
        # Simplified: check if daily return >= 9.8%
        prices = self._get_prices_for_date(trade_date, [ts_code])
        if ts_code not in prices:
            return False

        current_price = prices[ts_code]
        prev_date = self._get_previous_trading_date(trade_date)

        if prev_date:
            prev_prices = self._get_prices_for_date(prev_date, [ts_code])
            if ts_code in prev_prices:
                prev_price = prev_prices[ts_code]
                if prev_price > 0:
                    ret = (current_price - prev_price) / prev_price
                    return ret >= 0.098  # 9.8% threshold for limit-up

        return False

    def _is_limit_down(self, ts_code: str, trade_date: str) -> bool:
        """Check if stock hit limit-down on trade_date."""
        # Simplified: check if daily return <= -9.8%
        prices = self._get_prices_for_date(trade_date, [ts_code])
        if ts_code not in prices:
            return False

        current_price = prices[ts_code]
        prev_date = self._get_previous_trading_date(trade_date)

        if prev_date:
            prev_prices = self._get_prices_for_date(prev_date, [ts_code])
            if ts_code in prev_prices:
                prev_price = prev_prices[ts_code]
                if prev_price > 0:
                    ret = (current_price - prev_price) / prev_price
                    return ret <= -0.098  # -9.8% threshold for limit-down

        return False

    def _get_previous_trading_date(self, current_date: str) -> Optional[str]:
        """Get previous trading date from trade calendar."""
        if current_date not in self.trade_calendar:
            return None

        idx = self.trade_calendar.index(current_date)
        if idx > 0:
            return self.trade_calendar[idx - 1]

        return None

    def _calculate_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not self.daily_snapshots:
            return {}

        snaps = self.daily_snapshots
        initial = self.initial_capital
        final = snaps[-1].total_value

        # Returns
        total_return = (final - initial) / initial
        trading_days = len(snaps)
        annualized_return = (1 + total_return) ** (252 / trading_days) - 1 if trading_days > 1 else 0.0

        # Risk metrics
        daily_returns = [s.daily_return for s in snaps]
        if len(daily_returns) > 1:
            mean_ret = sum(daily_returns) / len(daily_returns)
            variance = sum((r - mean_ret) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
            volatility = variance ** 0.5
            annualized_vol = volatility * (252 ** 0.5)
        else:
            annualized_vol = 0.0

        # Max drawdown
        peak = initial
        max_dd = 0.0
        for snap in snaps:
            if snap.total_value > peak:
                peak = snap.total_value
            dd = (peak - snap.total_value) / peak
            max_dd = max(max_dd, dd)

        # Sharpe ratio
        risk_free_rate = 0.025
        sharpe = (annualized_return - risk_free_rate) / annualized_vol if annualized_vol > 0 else 0.0

        # Calmar ratio
        calmar = annualized_return / max_dd if max_dd > 0 else 0.0

        # Trade stats
        wins = [t for t in self.closed_positions if t.exit_price and t.entry_price > 0
                and (t.exit_price - t.entry_price) / t.entry_price > 0]
        win_rate = len(wins) / len(self.closed_positions) if self.closed_positions else 0.0

        avg_turnover = len(self.trades) / trading_days if trading_days > 0 else 0.0

        # Fetch benchmark (CSI300)
        benchmark_return = self._get_benchmark_return()

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
            "avg_turnover_per_day": avg_turnover,
            "total_trades": len(self.trades),
            "total_positions_closed": len(self.closed_positions),
            "benchmark_return": benchmark_return,
        }

    def _get_benchmark_return(self) -> float:
        """Fetch CSI300 benchmark return for backtest period."""
        if not self.daily_snapshots:
            return 0.0

        start_date = self.daily_snapshots[0].date
        end_date = self.daily_snapshots[-1].date

        try:
            bench_data = self.client.index_daily(
                ts_code="000300.SH",
                start_date=start_date,
                end_date=end_date,
                fields=["trade_date", "close"],
            )

            if not bench_data.empty and len(bench_data) > 1:
                bench_data["close"] = pd.to_numeric(bench_data["close"], errors="coerce")
                bench_data = bench_data.sort_values("trade_date")
                start_price = bench_data.iloc[0]["close"]
                end_price = bench_data.iloc[-1]["close"]
                if start_price > 0:
                    return (end_price - start_price) / start_price

        except Exception as e:
            logger.warning(f"Failed to fetch benchmark: {e}")

        return 0.0

    def _calculate_monthly_returns(self) -> Dict[str, float]:
        """Calculate monthly returns."""
        monthly = {}

        for snap in self.daily_snapshots:
            month_key = snap.date[:6]  # YYYYMM

            if month_key not in monthly:
                monthly[month_key] = {
                    "start_value": snap.total_value,
                    "end_value": snap.total_value,
                }
            else:
                monthly[month_key]["end_value"] = snap.total_value

        monthly_returns = {}
        for month_key, values in monthly.items():
            start_val = values["start_value"]
            end_val = values["end_value"]
            if start_val > 0:
                monthly_returns[month_key] = (end_val - start_val) / start_val

        return monthly_returns

    def _generate_report(self) -> BacktestReport:
        """Generate final backtest report."""
        metrics = self._calculate_metrics()
        monthly_returns = self._calculate_monthly_returns()

        summary = {
            "config_version": self.config.version,
            "start_date": self.daily_snapshots[0].date if self.daily_snapshots else "N/A",
            "end_date": self.daily_snapshots[-1].date if self.daily_snapshots else "N/A",
            "initial_capital": self.initial_capital,
            "final_value": self.daily_snapshots[-1].total_value if self.daily_snapshots else self.initial_capital,
            "trading_days": len(self.daily_snapshots),
            **metrics,
        }

        return BacktestReport(
            summary=summary,
            daily_values=self.daily_snapshots,
            trades=self.trades,
            monthly_returns=monthly_returns,
            position_history=self.closed_positions,
            metrics=metrics,
        )

    def _create_empty_report(self) -> BacktestReport:
        """Create an empty report for error cases."""
        return BacktestReport(
            summary={
                "error": "No trade dates found",
                "initial_capital": self.initial_capital,
            },
            daily_values=[],
            trades=[],
            monthly_returns={},
            position_history=[],
            metrics={},
        )


def print_report(report: BacktestReport) -> None:
    """Print backtest report to console."""
    summary = report.summary

    print("\n" + "="*80)
    print("MOMENTUM BACKTEST REPORT")
    print("="*80)

    print(f"\nConfig Version: {summary.get('config_version', 'N/A')}")
    print(f"Period: {summary.get('start_date')} -> {summary.get('end_date')}")
    print(f"Trading Days: {summary.get('trading_days')}")
    print(f"\nCapital: {summary.get('initial_capital'):,.0f} CNY")
    print(f"Final Value: {summary.get('final_value'):,.0f} CNY")

    print(f"\n--- Returns ---")
    print(f"Total Return: {summary.get('total_return', 0)*100:.2f}%")
    print(f"Annualized Return: {summary.get('annualized_return', 0)*100:.2f}%")
    print(f"Benchmark (CSI300) Return: {summary.get('benchmark_return', 0)*100:.2f}%")

    print(f"\n--- Risk ---")
    print(f"Annualized Volatility: {summary.get('annualized_volatility', 0)*100:.2f}%")
    print(f"Max Drawdown: {summary.get('max_drawdown', 0)*100:.2f}%")

    print(f"\n--- Risk-Adjusted ---")
    print(f"Sharpe Ratio: {summary.get('sharpe_ratio', 0):.2f}")
    print(f"Calmar Ratio: {summary.get('calmar_ratio', 0):.2f}")

    print(f"\n--- Trading ---")
    print(f"Total Trades: {summary.get('total_trades')}")
    print(f"Positions Closed: {summary.get('total_positions_closed')}")
    print(f"Win Rate: {summary.get('win_rate', 0)*100:.2f}%")
    print(f"Avg Turnover per Day: {summary.get('avg_turnover_per_day', 0):.2f}")

    if report.monthly_returns:
        print(f"\n--- Monthly Returns ---")
        for month, ret in sorted(report.monthly_returns.items()):
            print(f"  {month}: {ret*100:+.2f}%")

    print("\n" + "="*80 + "\n")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Momentum rotation strategy backtest"
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start date (YYYYMMDD)",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End date (YYYYMMDD)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to momentum config JSON (default: use built-in config)",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=1_000_000,
        help="Initial capital in CNY (default: 1,000,000)",
    )

    args = parser.parse_args()

    # Load config
    if args.config:
        logger.info(f"Loading config from {args.config}")
        config = MomentumConfig.load(Path(args.config))
    else:
        logger.info("Using default MomentumConfig")
        config = MomentumConfig()

    # Run backtest
    backtest = MomentumBacktester(config, args.initial_capital)
    report = backtest.run(args.start, args.end)

    # Print report
    print_report(report)

    # Return exit code
    if report.summary.get("error"):
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
