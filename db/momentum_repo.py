"""
Database repository for momentum rotation strategy.
Manages tables: momentum_config_active, momentum_holdings, momentum_rebalance_log,
momentum_daily_snapshot, momentum_performance_log.

All functions follow the sqlite3.Row factory pattern from repository.py.
"""

import json
import sqlite3
from datetime import datetime
from typing import Any, Optional

from loguru import logger

from config.settings import DB_PATH


# ── DDL Initialization ──


def ensure_momentum_tables(conn: sqlite3.Connection) -> None:
    """Create all momentum-related tables if they don't exist."""
    _ensure_momentum_config_active_table(conn)
    _ensure_momentum_holdings_table(conn)
    _ensure_momentum_rebalance_log_table(conn)
    _ensure_momentum_daily_snapshot_table(conn)
    _ensure_momentum_performance_log_table(conn)
    logger.info("Momentum tables ensured")


def _ensure_momentum_config_active_table(conn: sqlite3.Connection) -> None:
    """Create momentum_config_active table."""
    conn.execute(
        """CREATE TABLE IF NOT EXISTS momentum_config_active (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            config_json TEXT NOT NULL,
            version TEXT NOT NULL UNIQUE,
            activated_at TEXT DEFAULT (datetime('now')),
            description TEXT
        )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_momentum_config_version ON momentum_config_active(version)"
    )


def _ensure_momentum_holdings_table(conn: sqlite3.Connection) -> None:
    """Create momentum_holdings table for current holdings."""
    conn.execute(
        """CREATE TABLE IF NOT EXISTS momentum_holdings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_code TEXT NOT NULL,
            name TEXT,
            weight REAL NOT NULL,
            shares INTEGER NOT NULL,
            entry_date TEXT NOT NULL,
            entry_price REAL NOT NULL,
            momentum_score REAL,
            version TEXT,
            updated_at TEXT DEFAULT (datetime('now')),
            UNIQUE(ts_code, version)
        )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_momentum_holdings_code ON momentum_holdings(ts_code)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_momentum_holdings_version ON momentum_holdings(version)"
    )


def _ensure_momentum_rebalance_log_table(conn: sqlite3.Connection) -> None:
    """Create momentum_rebalance_log table for rebalance events."""
    conn.execute(
        """CREATE TABLE IF NOT EXISTS momentum_rebalance_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            version TEXT NOT NULL,
            action TEXT NOT NULL,
            ts_code TEXT NOT NULL,
            shares INTEGER NOT NULL,
            price REAL NOT NULL,
            reason TEXT,
            regime TEXT,
            logged_at TEXT DEFAULT (datetime('now')),
            UNIQUE(date, version, action, ts_code)
        )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_momentum_rebalance_date ON momentum_rebalance_log(date)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_momentum_rebalance_version ON momentum_rebalance_log(version)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_momentum_rebalance_code ON momentum_rebalance_log(ts_code)"
    )


def _ensure_momentum_daily_snapshot_table(conn: sqlite3.Connection) -> None:
    """Create momentum_daily_snapshot table for daily portfolio state."""
    conn.execute(
        """CREATE TABLE IF NOT EXISTS momentum_daily_snapshot (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL UNIQUE,
            portfolio_value REAL NOT NULL,
            cash REAL NOT NULL,
            positions_count INTEGER NOT NULL,
            benchmark_value REAL,
            recorded_at TEXT DEFAULT (datetime('now'))
        )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_momentum_daily_snapshot_date ON momentum_daily_snapshot(date)"
    )


def _ensure_momentum_performance_log_table(conn: sqlite3.Connection) -> None:
    """Create momentum_performance_log table for periodic metrics."""
    conn.execute(
        """CREATE TABLE IF NOT EXISTS momentum_performance_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            version TEXT NOT NULL,
            total_return REAL,
            sharpe REAL,
            max_drawdown REAL,
            win_rate REAL,
            avg_win REAL,
            avg_loss REAL,
            trades_count INTEGER,
            metrics_json TEXT,
            recorded_at TEXT DEFAULT (datetime('now')),
            UNIQUE(date, version)
        )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_momentum_performance_date ON momentum_performance_log(date)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_momentum_performance_version ON momentum_performance_log(version)"
    )


# ── Config Functions ──


def save_active_config(conn: sqlite3.Connection, config_dict: dict, version: str) -> None:
    """
    Save or update the active momentum configuration.

    Args:
        conn: Database connection
        config_dict: Configuration as dictionary
        version: Version string (e.g., "1.0.0")
    """
    try:
        config_json = json.dumps(config_dict, ensure_ascii=False)
        description = config_dict.get("description", "")

        conn.execute(
            """INSERT INTO momentum_config_active (config_json, version, description)
               VALUES (?, ?, ?)
               ON CONFLICT(version) DO UPDATE SET
                   config_json = excluded.config_json,
                   activated_at = datetime('now')""",
            (config_json, version, description),
        )
        logger.info(f"Saved momentum config v{version}")
    except Exception as e:
        logger.error(f"Failed to save momentum config: {e}")
        raise


def get_active_config(conn: sqlite3.Connection) -> Optional[dict]:
    """
    Get the most recently activated momentum configuration.

    Returns:
        Dictionary containing config, or None if not found
    """
    try:
        row = conn.execute(
            """SELECT config_json, version FROM momentum_config_active
               ORDER BY activated_at DESC LIMIT 1"""
        ).fetchone()

        if not row:
            return None

        config_dict = json.loads(row["config_json"])
        config_dict["_version"] = row["version"]
        config_dict["_activated_at"] = row["activated_at"] if "activated_at" in dict(row) else None
        return config_dict
    except Exception as e:
        logger.error(f"Failed to get active momentum config: {e}")
        return None


def get_config_by_version(conn: sqlite3.Connection, version: str) -> Optional[dict]:
    """Get momentum configuration by specific version."""
    try:
        row = conn.execute(
            "SELECT config_json FROM momentum_config_active WHERE version = ?",
            (version,),
        ).fetchone()

        if not row:
            return None

        return json.loads(row["config_json"])
    except Exception as e:
        logger.error(f"Failed to get momentum config v{version}: {e}")
        return None


# ── Holdings Functions ──


def save_holdings(conn: sqlite3.Connection, holdings: list[dict]) -> None:
    """
    Save current holdings (replaces previous holdings for this version).

    Args:
        conn: Database connection
        holdings: List of holdings, each with ts_code, name, weight, shares, entry_date, entry_price, momentum_score, version
    """
    try:
        # Delete old holdings for this version
        if holdings and "version" in holdings[0]:
            version = holdings[0]["version"]
            conn.execute("DELETE FROM momentum_holdings WHERE version = ?", (version,))

        # Insert new holdings
        for h in holdings:
            conn.execute(
                """INSERT INTO momentum_holdings
                   (ts_code, name, weight, shares, entry_date, entry_price, momentum_score, version)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    h["ts_code"],
                    h.get("name", ""),
                    h["weight"],
                    h["shares"],
                    h["entry_date"],
                    h["entry_price"],
                    h.get("momentum_score"),
                    h.get("version"),
                ),
            )
        logger.info(f"Saved {len(holdings)} holdings")
    except Exception as e:
        logger.error(f"Failed to save holdings: {e}")
        raise


def get_current_holdings(conn: sqlite3.Connection, version: Optional[str] = None) -> list[dict]:
    """
    Get current holdings, optionally filtered by version.

    Args:
        conn: Database connection
        version: Optional version filter

    Returns:
        List of holding dictionaries
    """
    try:
        if version:
            rows = conn.execute(
                "SELECT * FROM momentum_holdings WHERE version = ? ORDER BY weight DESC",
                (version,),
            ).fetchall()
        else:
            # Get latest version's holdings
            rows = conn.execute(
                """SELECT * FROM momentum_holdings
                   WHERE version = (SELECT version FROM momentum_config_active ORDER BY activated_at DESC LIMIT 1)
                   ORDER BY weight DESC"""
            ).fetchall()

        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"Failed to get current holdings: {e}")
        return []


# ── Rebalance Log Functions ──


def log_rebalance(
    conn: sqlite3.Connection,
    date: str,
    trades: list[dict],
    version: str,
    regime: str,
) -> None:
    """
    Log rebalance actions (BUY/SELL/HOLD).

    Args:
        conn: Database connection
        date: Trade date (YYYY-MM-DD)
        trades: List of trades, each with action, ts_code, shares, price, reason
        version: Configuration version
        regime: Market regime (e.g., "RUN", "HALT")
    """
    try:
        for trade in trades:
            conn.execute(
                """INSERT INTO momentum_rebalance_log
                   (date, version, action, ts_code, shares, price, reason, regime)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    date,
                    version,
                    trade["action"],
                    trade["ts_code"],
                    trade["shares"],
                    trade["price"],
                    trade.get("reason", ""),
                    regime,
                ),
            )
        logger.info(f"Logged {len(trades)} rebalance trades for {date}")
    except Exception as e:
        logger.error(f"Failed to log rebalance: {e}")
        raise


def get_rebalance_history(
    conn: sqlite3.Connection, start_date: str, end_date: str, ts_code: Optional[str] = None
) -> list[dict]:
    """
    Get rebalance history for a date range.

    Args:
        conn: Database connection
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        ts_code: Optional filter by stock code

    Returns:
        List of rebalance records
    """
    try:
        if ts_code:
            rows = conn.execute(
                """SELECT * FROM momentum_rebalance_log
                   WHERE date BETWEEN ? AND ? AND ts_code = ?
                   ORDER BY date DESC, action""",
                (start_date, end_date, ts_code),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM momentum_rebalance_log
                   WHERE date BETWEEN ? AND ?
                   ORDER BY date DESC, action""",
                (start_date, end_date),
            ).fetchall()

        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"Failed to get rebalance history: {e}")
        return []


# ── Daily Snapshot Functions ──


def save_daily_snapshot(
    conn: sqlite3.Connection,
    date: str,
    portfolio_value: float,
    cash: float,
    positions_count: int,
    benchmark_value: Optional[float] = None,
) -> None:
    """
    Save daily portfolio snapshot.

    Args:
        conn: Database connection
        date: Date (YYYY-MM-DD)
        portfolio_value: Total portfolio value
        cash: Cash balance
        positions_count: Number of open positions
        benchmark_value: Optional benchmark portfolio value for comparison
    """
    try:
        conn.execute(
            """INSERT INTO momentum_daily_snapshot
               (date, portfolio_value, cash, positions_count, benchmark_value)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(date) DO UPDATE SET
                   portfolio_value = excluded.portfolio_value,
                   cash = excluded.cash,
                   positions_count = excluded.positions_count,
                   benchmark_value = excluded.benchmark_value,
                   recorded_at = datetime('now')""",
            (date, portfolio_value, cash, positions_count, benchmark_value),
        )
        logger.info(f"Saved daily snapshot for {date}: value={portfolio_value:.2f}, cash={cash:.2f}")
    except Exception as e:
        logger.error(f"Failed to save daily snapshot: {e}")
        raise


def get_daily_snapshots(conn: sqlite3.Connection, start_date: str, end_date: str) -> list[dict]:
    """
    Get daily snapshots for a date range.

    Args:
        conn: Database connection
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        List of daily snapshot records
    """
    try:
        rows = conn.execute(
            """SELECT * FROM momentum_daily_snapshot
               WHERE date BETWEEN ? AND ?
               ORDER BY date DESC""",
            (start_date, end_date),
        ).fetchall()

        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"Failed to get daily snapshots: {e}")
        return []


def get_latest_daily_snapshot(conn: sqlite3.Connection) -> Optional[dict]:
    """Get the most recent daily snapshot."""
    try:
        row = conn.execute(
            "SELECT * FROM momentum_daily_snapshot ORDER BY date DESC LIMIT 1"
        ).fetchone()

        return dict(row) if row else None
    except Exception as e:
        logger.error(f"Failed to get latest daily snapshot: {e}")
        return None


# ── Performance Log Functions ──


def save_performance_metrics(
    conn: sqlite3.Connection,
    date: str,
    version: str,
    metrics: dict,
) -> None:
    """
    Save performance metrics for a period.

    Args:
        conn: Database connection
        date: Period end date (YYYY-MM-DD)
        version: Configuration version
        metrics: Dictionary containing performance metrics
                 (total_return, sharpe, max_drawdown, win_rate, avg_win, avg_loss, trades_count, etc.)
    """
    try:
        conn.execute(
            """INSERT INTO momentum_performance_log
               (date, version, total_return, sharpe, max_drawdown, win_rate,
                avg_win, avg_loss, trades_count, metrics_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(date, version) DO UPDATE SET
                   total_return = excluded.total_return,
                   sharpe = excluded.sharpe,
                   max_drawdown = excluded.max_drawdown,
                   win_rate = excluded.win_rate,
                   avg_win = excluded.avg_win,
                   avg_loss = excluded.avg_loss,
                   trades_count = excluded.trades_count,
                   metrics_json = excluded.metrics_json,
                   recorded_at = datetime('now')""",
            (
                date,
                version,
                metrics.get("total_return"),
                metrics.get("sharpe"),
                metrics.get("max_drawdown"),
                metrics.get("win_rate"),
                metrics.get("avg_win"),
                metrics.get("avg_loss"),
                metrics.get("trades_count"),
                json.dumps(metrics, ensure_ascii=False),
            ),
        )
        logger.info(
            f"Saved performance metrics for {date} v{version}: "
            f"return={metrics.get('total_return', 0):.2f}%, sharpe={metrics.get('sharpe', 0):.2f}"
        )
    except Exception as e:
        logger.error(f"Failed to save performance metrics: {e}")
        raise


def get_performance_history(
    conn: sqlite3.Connection, version: Optional[str] = None, limit: int = 30
) -> list[dict]:
    """
    Get performance history.

    Args:
        conn: Database connection
        version: Optional filter by configuration version
        limit: Maximum number of records to return

    Returns:
        List of performance records
    """
    try:
        if version:
            rows = conn.execute(
                """SELECT * FROM momentum_performance_log
                   WHERE version = ?
                   ORDER BY date DESC LIMIT ?""",
                (version, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM momentum_performance_log
                   ORDER BY date DESC LIMIT ?""",
                (limit,),
            ).fetchall()

        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"Failed to get performance history: {e}")
        return []


def get_performance_stats(
    conn: sqlite3.Connection, version: Optional[str] = None, period_days: int = 30
) -> Optional[dict]:
    """
    Get aggregated performance statistics over a period.

    Args:
        conn: Database connection
        version: Optional filter by version
        period_days: Look back this many days

    Returns:
        Aggregated stats dictionary or None
    """
    try:
        if version:
            row = conn.execute(
                """SELECT
                    COUNT(*) as records,
                    AVG(total_return) as avg_return,
                    AVG(sharpe) as avg_sharpe,
                    MIN(max_drawdown) as min_drawdown,
                    AVG(win_rate) as avg_win_rate,
                    SUM(trades_count) as total_trades
                   FROM momentum_performance_log
                   WHERE version = ?
                   AND date >= datetime('now', '-' || ? || ' days')""",
                (version, period_days),
            ).fetchone()
        else:
            row = conn.execute(
                """SELECT
                    COUNT(*) as records,
                    AVG(total_return) as avg_return,
                    AVG(sharpe) as avg_sharpe,
                    MIN(max_drawdown) as min_drawdown,
                    AVG(win_rate) as avg_win_rate,
                    SUM(trades_count) as total_trades
                   FROM momentum_performance_log
                   WHERE date >= datetime('now', '-' || ? || ' days')""",
                (period_days,),
            ).fetchone()

        return dict(row) if row else None
    except Exception as e:
        logger.error(f"Failed to get performance stats: {e}")
        return None


# ── Helper Functions ──


def get_db() -> sqlite3.Connection:
    """Get a database connection with proper factory."""
    from db.repository import get_connection

    return get_connection(DB_PATH)


def initialize_momentum_db() -> None:
    """Initialize momentum tables in the main database."""
    try:
        conn = get_db()
        ensure_momentum_tables(conn)
        conn.commit()
        conn.close()
        logger.info("Momentum database initialized")
    except Exception as e:
        logger.error(f"Failed to initialize momentum database: {e}")
        raise
