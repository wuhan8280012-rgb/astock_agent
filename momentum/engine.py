"""
Main momentum rotation engine that orchestrates the strategy.
Ties together universe filtering, momentum calculation, ranking, risk control, and rebalancing.
"""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from data_pipeline.tushare_client import TushareClient
from momentum.calculator import rank_by_momentum
from momentum.config import MomentumConfig
from momentum.rebalancer import RebalanceResult, compute_rebalance
from momentum.risk_control import (
    RegimeInfo,
    apply_position_limits,
    apply_stop_loss,
    check_drawdown,
    check_regime_filter,
)
from momentum.universe import filter_universe


@dataclass
class WeeklyRebalanceResult:
    """Results of a weekly rebalance run."""

    success: bool
    as_of: str
    universe_size: int
    current_holdings_count: int
    target_holdings_count: int
    buys: List[dict]
    sells: List[dict]
    holds: List[dict]
    turnover_pct: float
    estimated_cost: float
    regime: str
    rebalance_required: bool
    error_message: str = ""
    timestamp: str = ""


@dataclass
class DailyMonitorResult:
    """Results of a daily monitoring run."""

    success: bool
    as_of: str
    portfolio_value: float
    peak_value: float
    drawdown_pct: float
    stopped_out: List[str]
    regime: str
    circuit_breaker_triggered: bool
    error_message: str = ""
    timestamp: str = ""


class MomentumEngine:
    """Main orchestrator for momentum rotation strategy."""

    def __init__(self, config: MomentumConfig, client: TushareClient):
        """
        Initialize momentum engine.

        Args:
            config: MomentumConfig instance
            client: TushareClient instance
        """
        self.config = config
        self.client = client
        logger.info(f"Initialized MomentumEngine with config v{config.version}")

    def run_weekly_rebalance(
        self,
        as_of: str,
        conn: sqlite3.Connection,
    ) -> WeeklyRebalanceResult:
        """
        Run the weekly rebalance pipeline.

        Pipeline:
        1. Filter universe by market cap, turnover, ST status
        2. Fetch historical prices
        3. Calculate momentum scores
        4. Rank stocks
        5. Check regime filter
        6. Determine rebalance trades
        7. Apply position limits
        8. Record audit trail

        Args:
            as_of: reference date (YYYYMMDD format)
            conn: database connection

        Returns:
            WeeklyRebalanceResult with full audit trail
        """
        result = WeeklyRebalanceResult(
            success=False,
            as_of=as_of,
            universe_size=0,
            current_holdings_count=0,
            target_holdings_count=0,
            buys=[],
            sells=[],
            holds=[],
            turnover_pct=0.0,
            estimated_cost=0.0,
            regime="UNKNOWN",
            rebalance_required=False,
            timestamp=datetime.now().isoformat(),
        )

        try:
            # Step 1: Filter universe
            logger.info(f"Step 1: Filtering universe for {as_of}")
            universe = filter_universe(self.client, as_of, self.config)
            result.universe_size = len(universe)

            if universe.empty:
                result.error_message = "No stocks passed universe filter"
                logger.warning(result.error_message)
                return result

            logger.info(f"Filtered universe: {len(universe)} stocks")

            # Step 2: Rank by momentum
            logger.info("Step 2: Calculating momentum and ranking")
            ranked = rank_by_momentum(universe, self.client, as_of, self.config)

            if ranked.empty:
                result.error_message = "Failed to rank universe by momentum"
                logger.error(result.error_message)
                return result

            logger.info(f"Ranked {len(ranked)} stocks, top 5: {ranked.head(5)['ts_code'].tolist()}")

            # Step 3: Check regime filter
            logger.info("Step 3: Checking market regime")
            regime_info = check_regime_filter(self.client, as_of, self.config)
            result.regime = regime_info.regime

            # Step 4: Get current holdings from database
            logger.info("Step 4: Fetching current holdings")
            current_holdings = self._fetch_current_holdings(conn)
            result.current_holdings_count = len(current_holdings)

            logger.info(f"Current holdings: {len(current_holdings)} positions")

            # Step 5: Fetch current prices for calculations
            logger.info("Step 5: Fetching current prices")
            current_prices = self._fetch_current_prices(conn, as_of)

            # Step 6: Compute rebalance
            logger.info("Step 6: Computing rebalance")
            rebalance_result = compute_rebalance(
                current_holdings,
                ranked,
                self.config,
                total_capital=self._get_total_capital(conn),
                current_prices=current_prices,
            )

            # Step 7: Apply position limits based on regime
            logger.info("Step 7: Applying position limits")
            target_weights = rebalance_result.new_weights
            adjusted_weights = apply_position_limits(target_weights, self.config, regime_info)

            # Step 8: Record audit trail
            logger.info("Step 8: Recording audit trail")
            self._record_rebalance_audit(
                conn,
                as_of,
                universe,
                ranked,
                rebalance_result,
                regime_info,
                adjusted_weights,
            )

            # Populate result
            result.target_holdings_count = self.config.top_n
            result.buys = [
                {
                    "ts_code": t.ts_code,
                    "reason": t.reason,
                    "target_weight": t.target_weight,
                }
                for t in rebalance_result.buys
            ]
            result.sells = [
                {"ts_code": t.ts_code, "reason": t.reason}
                for t in rebalance_result.sells
            ]
            result.holds = [
                {
                    "ts_code": t.ts_code,
                    "reason": t.reason,
                    "target_weight": t.target_weight,
                }
                for t in rebalance_result.holds
            ]
            result.turnover_pct = rebalance_result.turnover_pct
            result.estimated_cost = rebalance_result.estimated_cost
            result.rebalance_required = rebalance_result.rebalance_required
            result.success = True

            logger.info(
                f"Weekly rebalance complete: buys={len(result.buys)}, "
                f"sells={len(result.sells)}, holds={len(result.holds)}, "
                f"turnover={result.turnover_pct:.1f}%"
            )

        except Exception as e:
            logger.exception(f"Exception in weekly rebalance: {e}")
            result.error_message = str(e)
            result.success = False

        return result

    def run_daily_monitor(
        self,
        as_of: str,
        conn: sqlite3.Connection,
    ) -> DailyMonitorResult:
        """
        Run daily monitoring for stop losses and circuit breakers.

        Args:
            as_of: reference date (YYYYMMDD format)
            conn: database connection

        Returns:
            DailyMonitorResult with stop-loss triggers and other alerts
        """
        result = DailyMonitorResult(
            success=False,
            as_of=as_of,
            portfolio_value=0.0,
            peak_value=0.0,
            drawdown_pct=0.0,
            stopped_out=[],
            regime="UNKNOWN",
            circuit_breaker_triggered=False,
            timestamp=datetime.now().isoformat(),
        )

        try:
            # Check regime
            logger.info(f"Daily monitor: Checking regime for {as_of}")
            regime_info = check_regime_filter(self.client, as_of, self.config)
            result.regime = regime_info.regime

            # Fetch current holdings
            logger.info("Daily monitor: Fetching current holdings")
            current_holdings = self._fetch_current_holdings(conn)

            if not current_holdings:
                logger.info("No current holdings, skipping daily monitor")
                result.success = True
                return result

            # Fetch current prices
            logger.info("Daily monitor: Fetching current prices")
            current_prices = self._fetch_current_prices(conn, as_of)

            # Check stop losses
            logger.info("Daily monitor: Checking stop losses")
            remaining_holdings, stopped_out = apply_stop_loss(
                current_holdings, current_prices, self.config
            )

            # Calculate portfolio metrics
            portfolio_value = sum(
                h.get("shares", 0) * current_prices.get(h["ts_code"], 0)
                for h in remaining_holdings
            )
            peak_value = self._get_peak_portfolio_value(conn)
            is_ok, drawdown = check_drawdown(portfolio_value, peak_value)

            # Record audit trail
            self._record_daily_monitor_audit(
                conn,
                as_of,
                portfolio_value,
                peak_value,
                stopped_out,
                regime_info,
            )

            # Populate result
            result.stopped_out = stopped_out
            result.portfolio_value = portfolio_value
            result.peak_value = peak_value
            result.drawdown_pct = drawdown * 100
            result.circuit_breaker_triggered = not is_ok
            result.success = True

            logger.info(
                f"Daily monitor complete: portfolio={portfolio_value:.2f}, "
                f"drawdown={drawdown*100:.2f}%, stopped_out={len(stopped_out)}"
            )

        except Exception as e:
            logger.exception(f"Exception in daily monitor: {e}")
            result.error_message = str(e)
            result.success = False

        return result

    def _fetch_current_holdings(self, conn: sqlite3.Connection) -> List[dict]:
        """Fetch current holdings from database."""
        try:
            cursor = conn.execute(
                """
                SELECT ts_code, shares, entry_price, entry_date, momentum_rank
                FROM momentum_holdings
                WHERE status = 'active'
                ORDER BY entry_date ASC
                """
            )
            holdings = []
            for row in cursor.fetchall():
                holdings.append({
                    "ts_code": row[0],
                    "shares": row[1],
                    "entry_price": row[2],
                    "entry_date": row[3],
                    "momentum_rank": row[4],
                })
            return holdings
        except Exception as e:
            logger.debug(f"Failed to fetch holdings from DB: {e}, returning empty list")
            return []

    def _fetch_current_prices(self, conn: sqlite3.Connection, as_of: str) -> dict:
        """Fetch current prices for holdings."""
        try:
            holdings = self._fetch_current_holdings(conn)
            if not holdings:
                return {}

            ts_codes = [h["ts_code"] for h in holdings]
            daily_data = self.client.daily(
                ts_code=",".join(ts_codes),
                end_date=as_of,
                fields=["ts_code", "close"],
            )

            if daily_data.empty:
                return {}

            # Get latest close for each stock
            latest = daily_data.sort_values("trade_date").drop_duplicates("ts_code", keep="last")
            prices = {
                row["ts_code"]: float(row["close"])
                for _, row in latest.iterrows()
            }
            return prices
        except Exception as e:
            logger.debug(f"Failed to fetch prices: {e}")
            return {}

    def _get_total_capital(self, conn: sqlite3.Connection) -> float:
        """Get total capital for portfolio."""
        try:
            cursor = conn.execute("SELECT total_capital FROM momentum_portfolio LIMIT 1")
            row = cursor.fetchone()
            return float(row[0]) if row else 1_000_000.0
        except Exception:
            return 1_000_000.0

    def _get_peak_portfolio_value(self, conn: sqlite3.Connection) -> float:
        """Get peak portfolio value from historical records."""
        try:
            cursor = conn.execute(
                "SELECT MAX(portfolio_value) FROM momentum_daily_monitor WHERE portfolio_value > 0"
            )
            row = cursor.fetchone()
            return float(row[0]) if row and row[0] else 1_000_000.0
        except Exception:
            return 1_000_000.0

    def _record_rebalance_audit(
        self,
        conn: sqlite3.Connection,
        as_of: str,
        universe: pd.DataFrame,
        ranked: pd.DataFrame,
        rebalance: RebalanceResult,
        regime: RegimeInfo,
        weights: dict,
    ) -> None:
        """Record rebalance audit trail to database."""
        try:
            # Ensure table exists
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS momentum_rebalance_log (
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
                )
                """
            )

            top_5 = ",".join(ranked.head(5)["ts_code"].tolist())
            conn.execute(
                """
                INSERT INTO momentum_rebalance_log
                (as_of, config_version, universe_size, top_n_count, regime,
                 buys_count, sells_count, holds_count, turnover_pct, estimated_cost,
                 rebalance_required, top_5_codes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    as_of,
                    self.config.version,
                    len(universe),
                    self.config.top_n,
                    regime.regime,
                    len(rebalance.buys),
                    len(rebalance.sells),
                    len(rebalance.holds),
                    rebalance.turnover_pct,
                    rebalance.estimated_cost,
                    rebalance.rebalance_required,
                    top_5,
                ),
            )
            conn.commit()
            logger.debug(f"Recorded rebalance audit for {as_of}")
        except Exception as e:
            logger.warning(f"Failed to record rebalance audit: {e}")

    def _record_daily_monitor_audit(
        self,
        conn: sqlite3.Connection,
        as_of: str,
        portfolio_value: float,
        peak_value: float,
        stopped_out: List[str],
        regime: RegimeInfo,
    ) -> None:
        """Record daily monitor audit trail to database."""
        try:
            # Ensure table exists
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS momentum_daily_monitor (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    as_of TEXT NOT NULL,
                    config_version TEXT NOT NULL,
                    portfolio_value REAL,
                    peak_value REAL,
                    drawdown_pct REAL,
                    stopped_out TEXT,
                    regime TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                )
                """
            )

            drawdown = (portfolio_value - peak_value) / peak_value * 100 if peak_value > 0 else 0
            stopped_out_str = ",".join(stopped_out) if stopped_out else ""

            conn.execute(
                """
                INSERT INTO momentum_daily_monitor
                (as_of, config_version, portfolio_value, peak_value, drawdown_pct,
                 stopped_out, regime)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    as_of,
                    self.config.version,
                    portfolio_value,
                    peak_value,
                    drawdown,
                    stopped_out_str,
                    regime.regime,
                ),
            )
            conn.commit()
            logger.debug(f"Recorded daily monitor for {as_of}")
        except Exception as e:
            logger.warning(f"Failed to record daily monitor: {e}")


# Required import at end to avoid circular dependency
from dataclasses import dataclass
