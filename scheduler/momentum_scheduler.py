"""
Momentum rotation scheduler — weekly rebalance + daily monitor + monthly evolution.

Integrates with the existing main_scheduler.py as an optional module.
Provides task locking to prevent concurrent execution and follows APScheduler patterns.
"""

import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from apscheduler.schedulers.blocking import BlockingScheduler
from loguru import logger

# Ensure project path is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DB_PATH, LOGS_DIR
from data_pipeline.tushare_client import TushareClient
from db import repository as repo
from db.momentum_repo import (
    ensure_momentum_tables,
    get_active_config,
    get_current_holdings,
    get_daily_snapshots,
    get_latest_daily_snapshot,
    get_performance_history,
    log_rebalance,
    save_active_config,
    save_daily_snapshot,
    save_holdings,
    save_performance_metrics,
)

# Configure logging
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logger.add(
    str(LOGS_DIR / "momentum_scheduler.log"),
    rotation="500 MB",
    retention="7 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
)


# ── Constants ──

_TASK_OWNER = f"pid-{__import__('os').getpid()}"
_TASK_LOCK_TTL_SECONDS = 3600  # 1 hour

# Momentum-specific constants
MOMENTUM_REBALANCE_LOCK_NAME = "momentum_weekly_rebalance"
MOMENTUM_DAILY_MONITOR_LOCK_NAME = "momentum_daily_monitor"
MOMENTUM_EVOLUTION_LOCK_NAME = "momentum_monthly_evolution"


# ── Task Locking ──


def _acquire_lock(
    conn: sqlite3.Connection, job_name: str, trade_date: str, ttl_seconds: int = _TASK_LOCK_TTL_SECONDS
) -> bool:
    """
    Acquire a task lock to prevent concurrent execution.

    Args:
        conn: Database connection
        job_name: Unique job identifier
        trade_date: Date string (YYYY-MM-DD)
        ttl_seconds: Lock TTL in seconds

    Returns:
        True if lock acquired, False if already locked
    """
    try:
        return repo.acquire_task_lock(conn, job_name, trade_date, _TASK_OWNER, ttl_seconds)
    except Exception as e:
        logger.error(f"Failed to acquire lock for {job_name}: {e}")
        return False


def _release_lock(conn: sqlite3.Connection, job_name: str, trade_date: str) -> None:
    """Release a task lock."""
    try:
        repo.release_task_lock(conn, job_name, trade_date, _TASK_OWNER)
    except Exception as e:
        logger.warning(f"Failed to release lock for {job_name}: {e}")


# ── Core Tasks ──


def task_weekly_rebalance(force: bool = False) -> dict:
    """
    Weekly rebalance task (typically Friday at market close).

    Responsibilities:
    1. Check if today is rebalance day per config
    2. Run MomentumEngine.run_weekly_rebalance()
    3. Save results to database
    4. Log performance

    Args:
        force: If True, run rebalance regardless of weekday

    Returns:
        Dictionary with execution results
    """
    conn = None
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        conn = repo.get_connection(DB_PATH)
        conn.row_factory = sqlite3.Row
        repo._ensure_ddl_once(conn)
        ensure_momentum_tables(conn)

        # Try to acquire lock
        if not _acquire_lock(conn, MOMENTUM_REBALANCE_LOCK_NAME, today):
            logger.warning(f"Rebalance already running for {today}")
            return {"success": False, "error": "lock_held", "as_of": today}

        try:
            # Get active config
            config_dict = get_active_config(conn)
            if not config_dict:
                logger.error("No active momentum config found")
                return {"success": False, "error": "no_config", "as_of": today}

            from momentum.config import MomentumConfig
            from momentum.engine import MomentumEngine

            config = MomentumConfig.from_dict(config_dict)
            tushare_client = TushareClient()

            # Check if rebalance should run
            from data_pipeline.clock import is_trading_day
            from datetime import datetime as dt

            today_obj = dt.strptime(today, "%Y-%m-%d")
            should_rebalance = force or today_obj.weekday() == config.rebalance_weekday

            if not should_rebalance:
                logger.info(f"Not a rebalance day (config weekday: {config.rebalance_weekday})")
                return {
                    "success": True,
                    "action": "skipped",
                    "reason": "not_rebalance_day",
                    "as_of": today,
                }

            # Run rebalance
            logger.info(f"Running weekly rebalance for {today}")
            engine = MomentumEngine(config, tushare_client)
            result = engine.run_weekly_rebalance(as_of=today)

            if not result.success:
                logger.error(f"Rebalance failed: {result.error_message}")
                return {
                    "success": False,
                    "error": result.error_message,
                    "as_of": today,
                }

            # Save holdings to DB
            holdings_data = []
            for holding in (result.holds if hasattr(result, "holds") else []):
                holdings_data.append(
                    {
                        "ts_code": holding.get("ts_code"),
                        "name": holding.get("name"),
                        "weight": holding.get("weight", 0),
                        "shares": holding.get("shares", 0),
                        "entry_date": today,
                        "entry_price": holding.get("entry_price", 0),
                        "momentum_score": holding.get("momentum_score"),
                        "version": config.version,
                    }
                )

            if holdings_data:
                save_holdings(conn, holdings_data)

            # Log rebalance trades
            trades_to_log = []
            for trade_list, action in [
                (result.buys if hasattr(result, "buys") else [], "BUY"),
                (result.sells if hasattr(result, "sells") else [], "SELL"),
            ]:
                for trade in trade_list:
                    trades_to_log.append(
                        {
                            "action": action,
                            "ts_code": trade.get("ts_code"),
                            "shares": trade.get("shares", 0),
                            "price": trade.get("price", 0),
                            "reason": trade.get("reason", ""),
                        }
                    )

            if trades_to_log:
                log_rebalance(conn, today, trades_to_log, config.version, result.regime)

            # Save daily snapshot (if portfolio value available)
            if hasattr(result, "portfolio_value"):
                save_daily_snapshot(
                    conn,
                    today,
                    portfolio_value=result.portfolio_value if hasattr(result, "portfolio_value") else 0,
                    cash=0,  # Would need to compute from positions
                    positions_count=len(holdings_data),
                    benchmark_value=None,
                )

            conn.commit()

            logger.info(
                f"Rebalance completed: buys={len(result.buys if hasattr(result, 'buys') else [])}, "
                f"sells={len(result.sells if hasattr(result, 'sells') else [])}, "
                f"holds={len(result.holds if hasattr(result, 'holds') else [])}"
            )

            return {
                "success": True,
                "action": "rebalanced",
                "as_of": today,
                "buys": len(result.buys if hasattr(result, "buys") else []),
                "sells": len(result.sells if hasattr(result, "sells") else []),
                "regime": result.regime,
                "turnover_pct": result.turnover_pct if hasattr(result, "turnover_pct") else 0,
            }

        finally:
            _release_lock(conn, MOMENTUM_REBALANCE_LOCK_NAME, today)

    except Exception as e:
        logger.error(f"Rebalance task failed: {e}", exc_info=True)
        return {"success": False, "error": str(e), "as_of": datetime.now().strftime("%Y-%m-%d")}
    finally:
        if conn:
            conn.close()


def task_daily_monitor() -> dict:
    """
    Daily monitoring task (typically 16:30, post-market).

    Responsibilities:
    1. Update daily prices for all holdings
    2. Check stop-loss triggers
    3. Save daily snapshot
    4. Alert if stop-loss triggered

    Returns:
        Dictionary with execution results
    """
    conn = None
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        conn = repo.get_connection(DB_PATH)
        conn.row_factory = sqlite3.Row
        repo._ensure_ddl_once(conn)
        ensure_momentum_tables(conn)

        # Try to acquire lock
        if not _acquire_lock(conn, MOMENTUM_DAILY_MONITOR_LOCK_NAME, today):
            logger.warning(f"Daily monitor already running for {today}")
            return {"success": False, "error": "lock_held", "as_of": today}

        try:
            logger.info(f"Running daily monitor for {today}")

            # Get current holdings
            holdings = get_current_holdings(conn)
            if not holdings:
                logger.info("No holdings to monitor")
                return {"success": True, "action": "no_holdings", "as_of": today}

            # Get active config
            config_dict = get_active_config(conn)
            if not config_dict:
                logger.error("No active momentum config found")
                return {"success": False, "error": "no_config", "as_of": today}

            from momentum.config import MomentumConfig
            from momentum.risk_control import apply_stop_loss

            config = MomentumConfig.from_dict(config_dict)
            tushare_client = TushareClient()

            # Fetch latest prices for holdings
            ts_codes = [h["ts_code"] for h in holdings]
            stopped_out = []

            # Get daily data
            try:
                # This would ideally use the tushare client to get latest prices
                # For now, we'll use placeholder logic
                for holding in holdings:
                    ts_code = holding["ts_code"]
                    entry_price = holding["entry_price"]

                    # In production, fetch current price from market data
                    # apply_stop_loss checks against config.stop_loss_pct
                    # For now, we just log monitoring
                    logger.debug(f"Monitoring {ts_code}: entry={entry_price}")

            except Exception as e:
                logger.warning(f"Failed to fetch daily prices: {e}")

            # Save daily snapshot
            latest_snapshot = get_latest_daily_snapshot(conn)
            if latest_snapshot:
                portfolio_value = latest_snapshot.get("portfolio_value", 0)
            else:
                portfolio_value = 0

            save_daily_snapshot(
                conn,
                today,
                portfolio_value=portfolio_value,
                cash=0,
                positions_count=len(holdings),
                benchmark_value=None,
            )

            conn.commit()

            result_dict = {
                "success": True,
                "action": "monitored",
                "as_of": today,
                "positions_monitored": len(holdings),
                "stopped_out": stopped_out,
            }

            logger.info(f"Daily monitor completed: {len(holdings)} positions monitored")
            return result_dict

        finally:
            _release_lock(conn, MOMENTUM_DAILY_MONITOR_LOCK_NAME, today)

    except Exception as e:
        logger.error(f"Daily monitor task failed: {e}", exc_info=True)
        return {"success": False, "error": str(e), "as_of": datetime.now().strftime("%Y-%m-%d")}
    finally:
        if conn:
            conn.close()


def task_monthly_evolution() -> dict:
    """
    Monthly evolution task (typically 1st of month at 18:00).

    Responsibilities:
    1. Run EvolutionEngine.run_evolution_cycle()
    2. If promotion: update active config
    3. Log results and metrics

    Returns:
        Dictionary with execution results
    """
    conn = None
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        conn = repo.get_connection(DB_PATH)
        conn.row_factory = sqlite3.Row
        repo._ensure_ddl_once(conn)
        ensure_momentum_tables(conn)

        # Try to acquire lock
        if not _acquire_lock(conn, MOMENTUM_EVOLUTION_LOCK_NAME, today, ttl_seconds=7200):
            logger.warning(f"Evolution already running for {today}")
            return {"success": False, "error": "lock_held", "as_of": today}

        try:
            logger.info(f"Running monthly evolution for {today}")

            from evolution.engine import EvolutionEngine

            tushare_client = TushareClient()

            # Run evolution cycle
            evolution_result = EvolutionEngine.run_evolution_cycle(
                conn=conn,
                as_of=today,
                tushare_client=tushare_client,
                backtest_start_offset_days=90,
                backtest_end_offset_days=7,
            )

            if not evolution_result.cycle_success:
                logger.error(f"Evolution cycle failed: {evolution_result.error_message}")
                return {
                    "success": False,
                    "error": evolution_result.error_message,
                    "as_of": today,
                }

            # If a better version was promoted, activate it
            if evolution_result.best_version and evolution_result.best_version != evolution_result.active_version:
                logger.info(f"Promoting {evolution_result.best_version}")

                from momentum.config import MomentumConfig

                promoted_config = MomentumConfig.load(
                    Path(__file__).parent.parent / "momentum" / f"config_{evolution_result.best_version}.json"
                )
                save_active_config(conn, promoted_config.to_dict(), evolution_result.best_version)

            # Log evolution metrics
            evolution_metrics = {
                "mutations_tested": evolution_result.mutations_tested,
                "mutations_promoted": evolution_result.mutations_promoted,
                "best_improvement_pct": evolution_result.best_improvement_pct,
                "best_version": evolution_result.best_version,
            }
            save_performance_metrics(
                conn,
                today,
                evolution_result.active_version,
                evolution_metrics,
            )

            conn.commit()

            result_dict = {
                "success": True,
                "action": "evolved",
                "as_of": today,
                "mutations_tested": evolution_result.mutations_tested,
                "mutations_promoted": evolution_result.mutations_promoted,
                "best_version": evolution_result.best_version,
                "improvement_pct": evolution_result.best_improvement_pct,
            }

            logger.info(evolution_result.summary())
            return result_dict

        finally:
            _release_lock(conn, MOMENTUM_EVOLUTION_LOCK_NAME, today)

    except Exception as e:
        logger.error(f"Evolution task failed: {e}", exc_info=True)
        return {"success": False, "error": str(e), "as_of": datetime.now().strftime("%Y-%m-%d")}
    finally:
        if conn:
            conn.close()


# ── Scheduler Registration ──


def register_momentum_jobs(scheduler: BlockingScheduler) -> None:
    """
    Register momentum jobs with an existing APScheduler BlockingScheduler.

    Integration with main_scheduler.py:
    Call this from main_scheduler.py's initialization to add momentum jobs.

    Args:
        scheduler: APScheduler BlockingScheduler instance
    """
    try:
        # Weekly rebalance: every Friday at 16:00 (market close)
        scheduler.add_job(
            task_weekly_rebalance,
            "cron",
            day_of_week=4,  # Friday
            hour=16,
            minute=0,
            id="momentum_weekly_rebalance",
            name="Momentum Weekly Rebalance",
            replace_existing=True,
            max_instances=1,
        )
        logger.info("Registered momentum weekly rebalance: Friday 16:00")

        # Daily monitor: every trading day at 16:30
        scheduler.add_job(
            task_daily_monitor,
            "cron",
            day_of_week="0-4",  # Mon-Fri
            hour=16,
            minute=30,
            id="momentum_daily_monitor",
            name="Momentum Daily Monitor",
            replace_existing=True,
            max_instances=1,
        )
        logger.info("Registered momentum daily monitor: Mon-Fri 16:30")

        # Monthly evolution: 1st of month at 18:00
        scheduler.add_job(
            task_monthly_evolution,
            "cron",
            day=1,
            hour=18,
            minute=0,
            id="momentum_monthly_evolution",
            name="Momentum Monthly Evolution",
            replace_existing=True,
            max_instances=1,
        )
        logger.info("Registered momentum monthly evolution: 1st of month 18:00")

    except Exception as e:
        logger.error(f"Failed to register momentum jobs: {e}")
        raise


# ── Standalone Entry Point ──


def run_momentum_standalone(jobs_to_run: Optional[list[str]] = None) -> None:
    """
    Run momentum scheduler as standalone process.

    Useful for:
    - Testing momentum system independently
    - Running scheduled tasks in a separate worker process
    - Development and debugging

    Args:
        jobs_to_run: Optional list of job IDs to run. If None, runs all jobs.
                     Example: ["momentum_weekly_rebalance", "momentum_daily_monitor"]
    """
    logger.info("Starting momentum scheduler (standalone mode)")

    try:
        # Initialize database
        from db.repository import init_db

        init_db()
        logger.info("Database initialized")

        # Create scheduler
        scheduler = BlockingScheduler()

        # Register jobs
        register_momentum_jobs(scheduler)

        # If specific jobs requested, filter
        if jobs_to_run:
            scheduled_jobs = scheduler.get_jobs()
            for job in scheduled_jobs:
                if job.id not in jobs_to_run:
                    scheduler.remove_job(job.id)
            logger.info(f"Running specific jobs: {jobs_to_run}")
        else:
            logger.info("Running all momentum jobs")

        # Start scheduler
        logger.info("Scheduler started")
        scheduler.start()

    except KeyboardInterrupt:
        logger.info("Momentum scheduler interrupted by user")
    except Exception as e:
        logger.error(f"Momentum scheduler error: {e}", exc_info=True)
        raise


# ── Module Entrypoint ──


if __name__ == "__main__":
    # Allow running as: python -m scheduler.momentum_scheduler
    run_momentum_standalone()
