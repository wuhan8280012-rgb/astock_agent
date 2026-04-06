"""
Database repository — the ONLY module allowed to execute SQL.
All other modules access the database through functions defined here.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from config.settings import DB_PATH, SCHEMA_PATH
from decision_engine.regime_engine import derive_legacy_macro_switch

# Track DDL initialization per database file in this process.
_ddl_initialized_paths: set[str] = set()


def _ensure_ddl_once(conn: sqlite3.Connection) -> None:
    """Run DDL migrations at most once per process."""
    global _ddl_initialized_paths
    if not hasattr(conn, "execute"):
        return
    db_rows = conn.execute("PRAGMA database_list").fetchall()
    db_key = ""
    if db_rows:
        row = db_rows[0]
        db_key = str(row["file"] if isinstance(row, sqlite3.Row) else row[2])
    if db_key in _ddl_initialized_paths:
        return
    _ensure_signals_mode_column(conn)
    _ensure_signals_sentiment_columns(conn)
    _ensure_schema_registry_table(conn)
    _ensure_buy_decisions_table(conn)
    _ensure_macro_switch_log_table(conn)
    _ensure_task_locks_table(conn)
    _ensure_job_run_log_table(conn)
    _ensure_strategy_version_registry_table(conn)
    _ensure_strategy_validation_runs_table(conn)
    _ensure_strategy_ablation_results_table(conn)
    _ensure_audit_findings_table(conn)
    _ensure_system_metrics_table(conn)
    _ensure_reflection_bundles_table(conn)
    _ensure_model_review_chain_table(conn)
    _ensure_upgrade_proposals_table(conn)
    _ensure_governance_experiments_table(conn)
    _ensure_governance_approvals_table(conn)
    _ensure_live_promotion_log_table(conn)
    _ensure_regret_records_table(conn)
    _ensure_research_priorities_table(conn)
    _ensure_replay_runs_table(conn)
    _ensure_research_memos_table(conn)
    _ensure_weight_allocation_log_table(conn)
    _ensure_opus_decision_log_table(conn)
    _ensure_pass_tracking_table(conn)
    _ensure_candidate_decision_trace_table(conn)
    _ensure_prescreener_run_table(conn)
    _ensure_prescreener_candidate_table(conn)
    _ensure_prescreener_theme_table(conn)
    _ensure_prescreener_replacement_table(conn)
    _ensure_trading_lessons_table(conn)
    _ensure_weekly_review_log_table(conn)
    _ensure_prompt_suggestions_table(conn)
    _ensure_sector_membership_table(conn)
    _ensure_etf_list_table(conn)
    _ensure_etf_holdings_table(conn)
    _ddl_initialized_paths.add(db_key)
    apply_platform_migrations(conn)


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Create a new database connection with row factory."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_db(db_path: Path = DB_PATH):
    """Context manager for database connections."""
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path = DB_PATH):
    """Initialize database by executing schema.sql and all DDL migrations."""
    global _ddl_initialized_paths
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with get_db(db_path) as conn:
        conn.executescript(schema_sql)
        _ensure_signals_mode_column(conn)
        _ensure_signals_sentiment_columns(conn)
        _ensure_schema_registry_table(conn)
        _ensure_buy_decisions_table(conn)
        _ensure_macro_switch_log_table(conn)
        _ensure_task_locks_table(conn)
        _ensure_job_run_log_table(conn)
        _ensure_strategy_version_registry_table(conn)
        _ensure_strategy_validation_runs_table(conn)
        _ensure_strategy_ablation_results_table(conn)
        _ensure_audit_findings_table(conn)
        _ensure_system_metrics_table(conn)
        _ensure_reflection_bundles_table(conn)
        _ensure_model_review_chain_table(conn)
        _ensure_upgrade_proposals_table(conn)
        _ensure_governance_experiments_table(conn)
        _ensure_governance_approvals_table(conn)
        _ensure_live_promotion_log_table(conn)
        _ensure_regret_records_table(conn)
        _ensure_research_priorities_table(conn)
        _ensure_replay_runs_table(conn)
        _ensure_research_memos_table(conn)
        _ensure_weight_allocation_log_table(conn)
        _ensure_opus_decision_log_table(conn)
        _ensure_pass_tracking_table(conn)
        _ensure_candidate_decision_trace_table(conn)
        _ensure_prescreener_run_table(conn)
        _ensure_prescreener_candidate_table(conn)
        _ensure_trading_lessons_table(conn)
        _ensure_weekly_review_log_table(conn)
        _ensure_prompt_suggestions_table(conn)
        _ensure_sector_membership_table(conn)
        _ensure_etf_list_table(conn)
        _ensure_etf_holdings_table(conn)
        _ddl_initialized_paths.add(str(db_path))
        apply_platform_migrations(conn)


# ── feature_snapshots ──

def insert_feature_snapshot(
    conn: sqlite3.Connection,
    stock_code: str,
    as_of: str,
    composite_score: float,
    sector_score: float,
    capital_score: float,
    catalyst_score: float,
    structure_score: float,
    liquidity_score: float,
    sector_trace: str = "",
    capital_trace: str = "",
    catalyst_trace: str = "",
    structure_trace: str = "",
    liquidity_trace: str = "",
    feature_vector: str = "",
) -> int:
    """Insert a feature snapshot and return the row id."""
    cursor = conn.execute(
        """INSERT INTO feature_snapshots
           (stock_code, as_of, composite_score, sector_score, capital_score,
            catalyst_score, structure_score, liquidity_score,
            sector_trace, capital_trace, catalyst_trace, structure_trace, liquidity_trace,
            feature_vector)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (stock_code, as_of, composite_score, sector_score, capital_score,
         catalyst_score, structure_score, liquidity_score,
         sector_trace, capital_trace, catalyst_trace, structure_trace, liquidity_trace,
         feature_vector),
    )
    return cursor.lastrowid


def get_feature_snapshot(conn: sqlite3.Connection, stock_code: str, as_of: str) -> Optional[dict]:
    """Get the latest feature snapshot for a stock on a given date."""
    row = conn.execute(
        """SELECT * FROM feature_snapshots
           WHERE stock_code = ? AND as_of = ?
           ORDER BY id DESC LIMIT 1""",
        (stock_code, as_of),
    ).fetchone()
    return dict(row) if row else None


def get_recent_scores(conn: sqlite3.Connection, stock_code: str, days: int = 5) -> list[dict]:
    """Get recent feature snapshots for score trend analysis."""
    rows = conn.execute(
        """SELECT * FROM feature_snapshots
           WHERE stock_code = ?
           ORDER BY as_of DESC LIMIT ?""",
        (stock_code, days),
    ).fetchall()
    return [dict(r) for r in rows]


# ── signals ──

def insert_signal(
    conn: sqlite3.Connection,
    stock_code: str,
    signal_date: str,
    composite_score: float,
    entry_price: float,
    target_price: float,
    stop_price: float,
    rr_ratio: float,
    position_size: float,
    primary_driver: str,
    core_reason: str,
    decision_trace: str = "",
    rationale_text: str = "",
    mode: str = "live",
    sentiment_score: float | None = None,
    sentiment_sector: str = "",
    sentiment_phase: str = "",
    sentiment_detail: str = "",
) -> int:
    """Insert a buy signal and return the row id."""
    _ensure_ddl_once(conn)
    cursor = conn.execute(
        """INSERT INTO signals
           (stock_code, signal_date, mode, sentiment_score, sentiment_sector, sentiment_phase, sentiment_detail,
            composite_score, entry_price, target_price,
            stop_price, rr_ratio, position_size, primary_driver, core_reason,
            decision_trace, rationale_text)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (stock_code, signal_date, mode, sentiment_score, sentiment_sector, sentiment_phase, sentiment_detail,
         composite_score, entry_price, target_price,
         stop_price, rr_ratio, position_size, primary_driver, core_reason,
         decision_trace, rationale_text),
    )
    return cursor.lastrowid


def update_signal_status(conn: sqlite3.Connection, signal_id: int, status: str):
    """Update signal status (ACTIVE/CLOSED/VETOED)."""
    conn.execute(
        "UPDATE signals SET status = ? WHERE id = ?",
        (status, signal_id),
    )


def update_signal_rationale(conn: sqlite3.Connection, signal_id: int, rationale_text: str):
    """Update the rationale text for a signal."""
    conn.execute(
        "UPDATE signals SET rationale_text = ? WHERE id = ?",
        (rationale_text, signal_id),
    )


def get_active_signals(conn: sqlite3.Connection) -> list[dict]:
    """Get all active signals."""
    rows = conn.execute(
        "SELECT * FROM signals WHERE status = 'ACTIVE' ORDER BY signal_date DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def count_signals_on_date(conn: sqlite3.Connection, signal_date: str) -> int:
    """Count how many new signals were created on a given date."""
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM signals WHERE signal_date = ?",
        (signal_date,),
    ).fetchone()
    return row["cnt"]


def get_all_signals(conn: sqlite3.Connection) -> list[dict]:
    """Get all signals for audit purposes."""
    rows = conn.execute("SELECT * FROM signals ORDER BY signal_date DESC").fetchall()
    return [dict(r) for r in rows]


def get_recent_signals_lite(conn: sqlite3.Connection, limit: int = 10) -> list[dict]:
    """Get compact signal records for LLM context to reduce token usage."""
    rows = conn.execute(
        """SELECT stock_code, signal_date, composite_score, rr_ratio, status, primary_driver
           FROM signals
           ORDER BY signal_date DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def _ensure_buy_decisions_table(conn: sqlite3.Connection) -> None:
    """Best-effort table creation for backward compatibility with existing DB files."""
    conn.execute(
        """CREATE TABLE IF NOT EXISTS buy_decisions (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               decision_date TEXT NOT NULL,
               stock_code TEXT NOT NULL,
               decision TEXT NOT NULL,
               composite_score REAL,
               reason TEXT,
               pass_reason TEXT,
               model TEXT,
               created_at TEXT DEFAULT (datetime('now'))
           )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_buy_decisions_date ON buy_decisions(decision_date)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_buy_decisions_decision ON buy_decisions(decision, decision_date)"
    )


def insert_buy_decision(
    conn: sqlite3.Connection,
    decision_date: str,
    stock_code: str,
    decision: str,
    composite_score: float | None = None,
    reason: str = "",
    pass_reason: str = "",
    model: str = "",
) -> int:
    """Insert one buy_agent decision record (BUY or PASS)."""
    _ensure_ddl_once(conn)
    cursor = conn.execute(
        """INSERT INTO buy_decisions
           (decision_date, stock_code, decision, composite_score, reason, pass_reason, model)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            decision_date,
            stock_code,
            decision,
            composite_score,
            reason,
            pass_reason,
            model,
        ),
    )
    return cursor.lastrowid


# ── positions ──

def insert_position(
    conn: sqlite3.Connection,
    stock_code: str,
    signal_id: int,
    entry_date: str,
    entry_price: float,
    current_price: float,
    target_price: float,
    stop_price: float,
    position_size: float,
) -> int:
    """Insert a new position and return the row id."""
    cursor = conn.execute(
        """INSERT INTO positions
           (stock_code, signal_id, entry_date, entry_price, current_price,
            target_price, stop_price, position_size)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (stock_code, signal_id, entry_date, entry_price, current_price,
         target_price, stop_price, position_size),
    )
    return cursor.lastrowid


def get_open_positions(conn: sqlite3.Connection) -> list[dict]:
    """Get all positions with status='OPEN'."""
    rows = conn.execute(
        "SELECT * FROM positions WHERE status = 'OPEN' ORDER BY entry_date"
    ).fetchall()
    return [dict(r) for r in rows]


def close_position(
    conn: sqlite3.Connection,
    position_id: int,
    exit_date: str,
    exit_price: float,
    exit_trigger: str,
):
    """Close a position by setting status to CLOSED and recording exit info."""
    pnl_pct = None
    row = conn.execute("SELECT entry_price FROM positions WHERE id = ?", (position_id,)).fetchone()
    if row and row["entry_price"] and row["entry_price"] > 0:
        pnl_pct = (exit_price - row["entry_price"]) / row["entry_price"] * 100

    conn.execute(
        """UPDATE positions
           SET status = 'CLOSED', exit_date = ?, exit_price = ?, exit_trigger = ?,
               pnl_pct = ?, current_price = ?, updated_at = datetime('now')
           WHERE id = ?""",
        (exit_date, exit_price, exit_trigger, pnl_pct, exit_price, position_id),
    )


def update_position_price(conn: sqlite3.Connection, position_id: int, current_price: float, holding_days: int):
    """Update current price and holding days for an open position."""
    row = conn.execute("SELECT entry_price FROM positions WHERE id = ?", (position_id,)).fetchone()
    pnl_pct = 0
    if row and row["entry_price"] and row["entry_price"] > 0:
        pnl_pct = (current_price - row["entry_price"]) / row["entry_price"] * 100

    conn.execute(
        """UPDATE positions
           SET current_price = ?, holding_days = ?, pnl_pct = ?, updated_at = datetime('now')
           WHERE id = ?""",
        (current_price, holding_days, pnl_pct, position_id),
    )


def update_position_size(conn: sqlite3.Connection, position_id: int, new_size: float):
    """Update position size (for ADD/REDUCE operations)."""
    conn.execute(
        "UPDATE positions SET position_size = ?, updated_at = datetime('now') WHERE id = ?",
        (new_size, position_id),
    )


# ── position_reviews ──

def insert_position_review(
    conn: sqlite3.Connection,
    position_id: int,
    review_date: str,
    current_score: float,
    score_trend: str,
    action: str,
    size_change: float = 0,
    reason: str = "",
) -> int:
    """Insert a position review record."""
    cursor = conn.execute(
        """INSERT INTO position_reviews
           (position_id, review_date, current_score, score_trend, action, size_change, reason)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (position_id, review_date, current_score, score_trend, action, size_change, reason),
    )
    return cursor.lastrowid


# ── postmortems ──

def insert_postmortem(
    conn: sqlite3.Connection,
    position_id: int,
    trigger_reason: str,
    advocate_output: str,
    challenger_output: str,
    arbitrator_verdict: str,
    arbitrator_confidence: float,
    execution_class: str,
    param_change: str = "",
    human_queue_item: str = "",
) -> int:
    """Insert an L3 postmortem record."""
    cursor = conn.execute(
        """INSERT INTO postmortems
           (position_id, trigger_reason, advocate_output, challenger_output,
            arbitrator_verdict, arbitrator_confidence, execution_class,
            param_change, human_queue_item)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (position_id, trigger_reason, advocate_output, challenger_output,
         arbitrator_verdict, arbitrator_confidence, execution_class,
         param_change, human_queue_item),
    )
    return cursor.lastrowid


def get_recent_postmortems(conn: sqlite3.Connection, limit: int = 20) -> list[dict]:
    """Get recent postmortems for arbitrator context injection."""
    rows = conn.execute(
        """SELECT arbitrator_verdict, arbitrator_confidence AS confidence,
                  execution_class, created_at
           FROM postmortems
           ORDER BY created_at DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


# ── param_changelog ──

def insert_param_changelog(
    conn: sqlite3.Connection,
    version: str,
    changed_at: str,
    agent_name: str,
    param_name: str,
    value_from: str,
    value_to: str,
    llm_reasoning: str = "",
    execution_class: str = "",
    falsifiable_expectation: str = "",
    rollback_condition: str = "",
    rollback_date: str = "",
) -> int:
    """Insert a parameter change log entry."""
    cursor = conn.execute(
        """INSERT INTO param_changelog
           (version, changed_at, agent_name, param_name, value_from, value_to,
            llm_reasoning, execution_class, falsifiable_expectation,
            rollback_condition, rollback_date)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (version, changed_at, agent_name, param_name, value_from, value_to,
         llm_reasoning, execution_class, falsifiable_expectation,
         rollback_condition, rollback_date),
    )
    return cursor.lastrowid


def get_param_changes_in_days(conn: sqlite3.Connection, param_name: str, days: int = 90) -> list[dict]:
    """Get parameter changes within the last N days for guardrail checks."""
    rows = conn.execute(
        """SELECT * FROM param_changelog
           WHERE param_name = ?
           ORDER BY changed_at DESC""",
        (param_name,),
    ).fetchall()
    # Filter in Python to support both real and test scenarios
    from datetime import datetime, timedelta
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    return [dict(r) for r in rows if (r["changed_at"] or "") >= cutoff]


# ── Audit queries ──

def count_signals_total(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) as cnt FROM signals").fetchone()
    return row["cnt"]


def count_winning_signals(conn: sqlite3.Connection) -> int:
    """Count signals where exit_price > entry_price (from positions table)."""
    row = conn.execute(
        """SELECT COUNT(*) as cnt FROM positions
           WHERE status = 'CLOSED' AND exit_price > entry_price"""
    ).fetchone()
    return row["cnt"]


def count_closed_positions(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) as cnt FROM positions WHERE status = 'CLOSED'").fetchone()
    return row["cnt"]


def get_signals_in_date_range(conn: sqlite3.Connection, start: str, end: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM signals WHERE signal_date BETWEEN ? AND ?",
        (start, end),
    ).fetchall()
    return [dict(r) for r in rows]


def get_position_by_id(conn: sqlite3.Connection, position_id: int) -> Optional[dict]:
    """Get a single position by id."""
    row = conn.execute("SELECT * FROM positions WHERE id = ?", (position_id,)).fetchone()
    return dict(row) if row else None


def get_signal_by_id(conn: sqlite3.Connection, signal_id: int) -> Optional[dict]:
    """Get a single signal by id."""
    row = conn.execute("SELECT * FROM signals WHERE id = ?", (signal_id,)).fetchone()
    return dict(row) if row else None


def get_positions_by_exit_date(conn: sqlite3.Connection, exit_date: str) -> list[dict]:
    """Get positions closed on a given date."""
    rows = conn.execute(
        "SELECT * FROM positions WHERE exit_date = ?", (exit_date,)
    ).fetchall()
    return [dict(r) for r in rows]


def get_postmortems_in_date_range(conn: sqlite3.Connection, start: str, end: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM postmortems WHERE created_at BETWEEN ? AND ?",
        (start, end),
    ).fetchall()
    return [dict(r) for r in rows]


# ── Daily report read models ──

def get_signals_on_date(conn: sqlite3.Connection, report_date: str) -> list[dict]:
    """Get all signals generated on a specific date (YYYY-MM-DD)."""
    rows = conn.execute(
        """SELECT *
           FROM signals
           WHERE signal_date = ?
           ORDER BY composite_score DESC, stock_code ASC""",
        (report_date,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_signals_on_date_with_mode(
    conn: sqlite3.Connection,
    report_date: str,
    mode: str | None = None,
    dedup: bool = True,
) -> list[dict]:
    """Get signals on date, optionally filtering by mode ('live'|'shadow')."""
    _ensure_ddl_once(conn)
    params: tuple[Any, ...]
    mode_clause = ""
    if mode:
        mode_clause = " AND mode = ?"
        params = (report_date, mode)
    else:
        params = (report_date,)

    if dedup:
        # Daily read-model uses latest row per (stock_code, signal_date, mode) for rerun consistency.
        rows = conn.execute(
            f"""WITH ranked AS (
                    SELECT s.*,
                           ROW_NUMBER() OVER (
                               PARTITION BY s.stock_code, s.signal_date, s.mode
                               ORDER BY s.id DESC
                           ) AS rn
                    FROM signals s
                    WHERE s.signal_date = ?{mode_clause}
                )
                SELECT *
                FROM ranked
                WHERE rn = 1
                ORDER BY composite_score DESC, stock_code ASC""",
            params,
        ).fetchall()
    else:
        rows = conn.execute(
            f"""SELECT *
                FROM signals
                WHERE signal_date = ?{mode_clause}
                ORDER BY composite_score DESC, stock_code ASC""",
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def get_regime_report_context(conn: sqlite3.Connection, report_date: str) -> dict:
    """Return a single consistent market-state read model for reports."""
    row = get_latest_macro_switch_on_or_before(conn, report_date)
    if not row:
        return {
            "regime_state": "UNKNOWN",
            "regime_source": "missing",
            "legacy_macro_switch_derived_from_regime": "UNKNOWN",
            "regime_policy_effect": {},
            "regime_evidence": {},
            "csi300_close": None,
            "ma60": None,
        }
    try:
        evidence = json.loads(row.get("regime_evidence_json") or "{}")
    except Exception:
        evidence = {}
    try:
        policy = json.loads(row.get("regime_policy_json") or "{}")
    except Exception:
        policy = {}
    regime_state = str(row.get("regime_state") or "RUN")
    return {
        "regime_state": regime_state,
        "regime_source": str(row.get("source") or "scheduler"),
        "legacy_macro_switch_derived_from_regime": derive_legacy_macro_switch(regime_state),
        "regime_policy_effect": policy,
        "regime_evidence": evidence,
        "csi300_close": row.get("csi300_close"),
        "ma60": row.get("ma60"),
    }


def get_trace_stage_stock_set(
    conn: sqlite3.Connection,
    trade_date: str,
    mode: str,
    stage: str,
    status: str | None = None,
) -> set[str]:
    """Distinct stock set for a stage/status from the unified trace model."""
    rows = get_candidate_decision_trace_on_date(conn, trade_date, mode=mode)
    out: set[str] = set()
    for row in rows:
        if str(row.get("stage")) != stage:
            continue
        if status is not None and str(row.get("status")) != status:
            continue
        code = str(row.get("stock_code") or "").strip()
        if code:
            out.add(code)
    return out


def get_score_read_model_on_date(
    conn: sqlite3.Connection,
    trade_date: str,
    mode: str,
    threshold: float,
) -> dict:
    """
    Unified score source of truth for reports.

    `prescored_candidates` means candidates that reached the prescore stage and
    have a persisted prescore in trace.details_json. `trace_candidates` means
    candidates that reached the main decision trace (`opus_decision` stage).
    Reports must not mix these two sets implicitly.
    """
    rows = get_candidate_decision_trace_on_date(conn, trade_date, mode=mode)
    prescore_by_code: dict[str, float] = {}
    trace_candidates: set[str] = set()
    for row in rows:
        code = str(row.get("stock_code") or "").strip()
        if not code:
            continue
        stage = str(row.get("stage") or "")
        if stage == "prescore":
            try:
                payload = json.loads(row.get("details_json") or "{}")
            except Exception:
                payload = {}
            try:
                prescore_by_code[code] = float(payload.get("prescore"))
            except Exception:
                continue
        elif stage == "opus_decision":
            trace_candidates.add(code)

    prescored_candidates = set(prescore_by_code)
    prescore_ge_threshold = {c for c, score in prescore_by_code.items() if score >= threshold}
    trace_ge_threshold = {c for c in trace_candidates if prescore_by_code.get(c, float("-inf")) >= threshold}

    return {
        "threshold": float(threshold),
        "prescored_candidates": prescored_candidates,
        "prescore_ge_threshold": prescore_ge_threshold,
        "trace_candidates": trace_candidates,
        "trace_ge_threshold": trace_ge_threshold,
        "prescore_scores": prescore_by_code,
    }


def delete_shadow_signals_on_date(conn: sqlite3.Connection, signal_date: str) -> int:
    """Delete shadow signals on date to keep reruns idempotent."""
    _ensure_ddl_once(conn)
    cursor = conn.execute(
        "DELETE FROM signals WHERE signal_date = ? AND mode = 'shadow'",
        (signal_date,),
    )
    return int(cursor.rowcount or 0)


def get_feature_snapshots_on_date(conn: sqlite3.Connection, report_date: str) -> list[dict]:
    """Get feature snapshots for a specific date by matching date(as_of)."""
    rows = conn.execute(
        """WITH ranked AS (
               SELECT f.*,
                      ROW_NUMBER() OVER (
                          PARTITION BY f.stock_code, date(f.as_of)
                          ORDER BY f.id DESC
                      ) AS rn
               FROM feature_snapshots f
               WHERE date(f.as_of) = date(?)
           )
           SELECT *
           FROM ranked
           WHERE rn = 1
           ORDER BY composite_score DESC, stock_code ASC""",
        (report_date,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_positions_open_on_date(conn: sqlite3.Connection, report_date: str) -> list[dict]:
    """Get positions opened on report_date (regardless of current status)."""
    rows = conn.execute(
        """SELECT *
           FROM positions
           WHERE entry_date = ?
           ORDER BY stock_code ASC""",
        (report_date,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_positions_closed_on_date(conn: sqlite3.Connection, report_date: str) -> list[dict]:
    """Get positions closed on report_date."""
    rows = conn.execute(
        """SELECT *
           FROM positions
           WHERE status='CLOSED' AND exit_date = ?
           ORDER BY stock_code ASC""",
        (report_date,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_positions_since(conn: sqlite3.Connection, start_date: str) -> list[dict]:
    """Get all positions with entry_date >= start_date."""
    rows = conn.execute(
        """SELECT *
           FROM positions
           WHERE entry_date >= ?
           ORDER BY entry_date ASC, stock_code ASC""",
        (start_date,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_closed_positions_since(conn: sqlite3.Connection, start_date: str) -> list[dict]:
    """Get all closed positions with entry_date >= start_date."""
    rows = conn.execute(
        """SELECT *
           FROM positions
           WHERE status='CLOSED' AND entry_date >= ?
           ORDER BY exit_date ASC, stock_code ASC""",
        (start_date,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_buy_decisions_on_date(conn: sqlite3.Connection, report_date: str) -> list[dict]:
    """Get BUY/PASS decisions on a specific date."""
    _ensure_ddl_once(conn)
    rows = conn.execute(
        """WITH ranked AS (
               SELECT b.*,
                      ROW_NUMBER() OVER (
                          PARTITION BY b.stock_code, b.decision_date
                          ORDER BY b.id DESC
                      ) AS rn
               FROM buy_decisions b
               WHERE b.decision_date = ?
           )
           SELECT *
           FROM ranked
           WHERE rn = 1
           ORDER BY stock_code ASC""",
        (report_date,),
    ).fetchall()
    return [dict(r) for r in rows]


def delete_buy_decisions_on_date(conn: sqlite3.Connection, decision_date: str) -> int:
    """Delete buy decisions on date to keep reruns idempotent."""
    _ensure_ddl_once(conn)
    cursor = conn.execute(
        "DELETE FROM buy_decisions WHERE decision_date = ?",
        (decision_date,),
    )
    return int(cursor.rowcount or 0)


def get_llm_usage_on_date(conn: sqlite3.Connection, report_date: str, component_prefix: str = "") -> list[dict]:
    """Get llm_usage rows on a specific date, optionally filtered by component prefix."""
    if component_prefix:
        rows = conn.execute(
            """SELECT *
               FROM llm_usage
               WHERE date(created_at) = date(?) AND component LIKE ?
               ORDER BY created_at ASC""",
            (report_date, f"{component_prefix}%"),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT *
               FROM llm_usage
               WHERE date(created_at) = date(?)
               ORDER BY created_at ASC""",
            (report_date,),
        ).fetchall()
    return [dict(r) for r in rows]


def _ensure_macro_switch_log_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS macro_switch_log (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               check_date TEXT NOT NULL,
               result TEXT NOT NULL,
               csi300_close REAL,
               ma60 REAL,
               regime_state TEXT DEFAULT 'RUN',
               regime_evidence_json TEXT,
               regime_policy_json TEXT,
               source TEXT DEFAULT 'scheduler',
               created_at TEXT DEFAULT (datetime('now'))
           )"""
    )
    cols = conn.execute("PRAGMA table_info(macro_switch_log)").fetchall()
    col_names = {str(r["name"]) for r in cols}
    if "regime_state" not in col_names:
        conn.execute("ALTER TABLE macro_switch_log ADD COLUMN regime_state TEXT DEFAULT 'RUN'")
    if "regime_evidence_json" not in col_names:
        conn.execute("ALTER TABLE macro_switch_log ADD COLUMN regime_evidence_json TEXT")
    if "regime_policy_json" not in col_names:
        conn.execute("ALTER TABLE macro_switch_log ADD COLUMN regime_policy_json TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_macro_switch_date ON macro_switch_log(check_date)")


def insert_macro_switch_log(
    conn: sqlite3.Connection,
    check_date: str,
    result: str,
    csi300_close: float | None = None,
    ma60: float | None = None,
    regime_state: str | None = None,
    regime_evidence_json: str = "",
    regime_policy_json: str = "",
    source: str = "scheduler",
) -> int:
    _ensure_ddl_once(conn)
    if regime_state is None or not str(regime_state).strip():
        regime_state = "HALT" if str(result) == "HALT" else "RUN"
    # regime_state is the only primary market-state semantic. Legacy macro_switch
    # is persisted as a compatibility projection derived from regime_state.
    derived_result = derive_legacy_macro_switch(str(regime_state or "RUN"))
    cursor = conn.execute(
        """INSERT INTO macro_switch_log
           (check_date, result, csi300_close, ma60, regime_state, regime_evidence_json, regime_policy_json, source)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (check_date, derived_result, csi300_close, ma60, regime_state, regime_evidence_json, regime_policy_json, source),
    )
    return cursor.lastrowid


def get_latest_macro_switch_on_or_before(conn: sqlite3.Connection, report_date: str) -> Optional[dict]:
    _ensure_ddl_once(conn)
    row = conn.execute(
        """SELECT *
           FROM macro_switch_log
           WHERE check_date <= ?
           ORDER BY check_date DESC, id DESC
           LIMIT 1""",
        (report_date,),
    ).fetchone()
    if not row:
        return None
    out = dict(row)
    regime_state = str(out.get("regime_state") or ("HALT" if str(out.get("result")) == "HALT" else "RUN"))
    out["regime_state"] = regime_state
    out["result"] = derive_legacy_macro_switch(regime_state)
    return out


def _ensure_signals_mode_column(conn: sqlite3.Connection) -> None:
    """Add signals.mode for backward compatibility with existing DB files."""
    table_row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='signals'"
    ).fetchone()
    if table_row is None:
        return
    cols = conn.execute("PRAGMA table_info(signals)").fetchall()
    col_names = {str(r["name"]) for r in cols}
    if "mode" not in col_names:
        conn.execute("ALTER TABLE signals ADD COLUMN mode TEXT DEFAULT 'live'")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_mode_date ON signals(mode, signal_date)")


def _ensure_signals_sentiment_columns(conn: sqlite3.Connection) -> None:
    """Add sentiment columns for backward compatibility with existing DB files."""
    table_row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='signals'"
    ).fetchone()
    if table_row is None:
        return
    cols = conn.execute("PRAGMA table_info(signals)").fetchall()
    col_names = {str(r["name"]) for r in cols}
    if "sentiment_score" not in col_names:
        conn.execute("ALTER TABLE signals ADD COLUMN sentiment_score REAL")
    if "sentiment_sector" not in col_names:
        conn.execute("ALTER TABLE signals ADD COLUMN sentiment_sector TEXT")
    if "sentiment_phase" not in col_names:
        conn.execute("ALTER TABLE signals ADD COLUMN sentiment_phase TEXT")
    if "sentiment_detail" not in col_names:
        conn.execute("ALTER TABLE signals ADD COLUMN sentiment_detail TEXT")


def _ensure_schema_registry_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS schema_migrations (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               migration_id TEXT NOT NULL UNIQUE,
               applied_at TEXT DEFAULT (datetime('now')),
               notes TEXT
           )"""
    )


def register_migration(conn: sqlite3.Connection, migration_id: str, notes: str = "") -> bool:
    """Register migration once. Returns True when inserted, False when already applied."""
    _ensure_schema_registry_table(conn)
    try:
        conn.execute(
            "INSERT INTO schema_migrations (migration_id, notes) VALUES (?, ?)",
            (migration_id, notes),
        )
        return True
    except sqlite3.IntegrityError:
        return False


def get_applied_migrations(conn: sqlite3.Connection) -> list[dict]:
    _ensure_schema_registry_table(conn)
    rows = conn.execute(
        "SELECT migration_id, applied_at, notes FROM schema_migrations ORDER BY id ASC"
    ).fetchall()
    return [dict(r) for r in rows]


def apply_platform_migrations(conn: sqlite3.Connection) -> None:
    """
    Run idempotent platform migrations/backfills on existing SQLite files.
    This keeps old paper-trading DBs readable by new trace/report models.
    """
    # M1: ensure legacy dates have minimal trace rows for report compatibility.
    m1 = "20260316_trace_backfill_v1"
    if register_migration(conn, m1, notes="backfill candidate_decision_trace from legacy tables"):
        rows = conn.execute(
            """WITH dates AS (
                   SELECT decision_date AS d FROM buy_decisions
                   UNION
                   SELECT signal_date AS d FROM signals
                   UNION
                   SELECT check_date AS d FROM macro_switch_log
                   UNION
                   SELECT entry_date AS d FROM positions
               )
               SELECT DISTINCT d
               FROM dates
               WHERE d IS NOT NULL AND d <> ''
               ORDER BY d ASC"""
        ).fetchall()
        for r in rows:
            d = str(r["d"])
            backfill_trace_from_legacy(conn, d)

    # M2: annotate missing mode field semantics for old rows via defaults/indexes.
    m2 = "20260316_signal_mode_index_v1"
    register_migration(conn, m2, notes="signals mode/index compatibility")


def _ensure_task_locks_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS task_locks (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               job_name TEXT NOT NULL,
               trade_date TEXT NOT NULL,
               owner TEXT NOT NULL,
               acquired_at TEXT DEFAULT (datetime('now')),
               expires_at TEXT,
               UNIQUE(job_name, trade_date)
           )"""
    )


def acquire_task_lock(
    conn: sqlite3.Connection,
    job_name: str,
    trade_date: str,
    owner: str,
    ttl_seconds: int = 7200,
) -> bool:
    """Acquire per-job/date lock. Expired lock can be stolen."""
    if not hasattr(conn, "execute"):
        return True
    _ensure_ddl_once(conn)
    _ensure_task_locks_table(conn)
    now_utc = datetime.now(timezone.utc)
    now_ts = now_utc.strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        """DELETE FROM task_locks
           WHERE job_name = ? AND trade_date = ?
             AND expires_at IS NOT NULL AND expires_at < ?""",
        (job_name, trade_date, now_ts),
    )
    expires_at = (now_utc + timedelta(seconds=float(ttl_seconds))).strftime("%Y-%m-%d %H:%M:%S")
    try:
        conn.execute(
            """INSERT INTO task_locks (job_name, trade_date, owner, expires_at)
               VALUES (?, ?, ?, ?)""",
            (job_name, trade_date, owner, expires_at),
        )
        return True
    except sqlite3.IntegrityError:
        return False


def release_task_lock(conn: sqlite3.Connection, job_name: str, trade_date: str, owner: str = "") -> int:
    if not hasattr(conn, "execute"):
        return 0
    _ensure_ddl_once(conn)
    if owner:
        cur = conn.execute(
            "DELETE FROM task_locks WHERE job_name = ? AND trade_date = ? AND owner = ?",
            (job_name, trade_date, owner),
        )
    else:
        cur = conn.execute(
            "DELETE FROM task_locks WHERE job_name = ? AND trade_date = ?",
            (job_name, trade_date),
        )
    return int(cur.rowcount or 0)


def _ensure_job_run_log_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS job_run_log (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               run_id TEXT NOT NULL UNIQUE,
               job_name TEXT NOT NULL,
               trade_date TEXT NOT NULL,
               trigger_type TEXT NOT NULL,
               start_at TEXT DEFAULT (datetime('now')),
               end_at TEXT,
               status TEXT NOT NULL,
               retry_count INTEGER DEFAULT 0,
               is_rerun INTEGER DEFAULT 0,
               supersedes_run_id TEXT,
               error_message TEXT,
               strategy_versions_json TEXT,
               report_schema_version TEXT,
               execution_policy_version TEXT
           )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_job_run_lookup ON job_run_log(job_name, trade_date, status, id)")


def start_job_run(
    conn: sqlite3.Connection,
    job_name: str,
    trade_date: str,
    trigger_type: str = "scheduler",
    is_rerun: bool = False,
    supersedes_run_id: str = "",
    strategy_versions_json: str = "",
    report_schema_version: str = "",
    execution_policy_version: str = "",
) -> str:
    if not hasattr(conn, "execute"):
        return f"mock-{uuid4()}"
    _ensure_ddl_once(conn)
    run_id = str(uuid4())
    conn.execute(
        """INSERT INTO job_run_log
           (run_id, job_name, trade_date, trigger_type, status, is_rerun, supersedes_run_id,
            strategy_versions_json, report_schema_version, execution_policy_version)
           VALUES (?, ?, ?, ?, 'running', ?, ?, ?, ?, ?)""",
        (
            run_id,
            job_name,
            trade_date,
            trigger_type,
            1 if is_rerun else 0,
            supersedes_run_id or None,
            strategy_versions_json or "",
            report_schema_version or "",
            execution_policy_version or "",
        ),
    )
    return run_id


def finish_job_run(
    conn: sqlite3.Connection,
    run_id: str,
    status: str,
    error_message: str = "",
    retry_count: int | None = None,
) -> None:
    if not hasattr(conn, "execute"):
        return
    _ensure_ddl_once(conn)
    if retry_count is None:
        conn.execute(
            """UPDATE job_run_log
               SET end_at = datetime('now'), status = ?, error_message = ?
               WHERE run_id = ?""",
            (status, (error_message or "")[:1000], run_id),
        )
    else:
        conn.execute(
            """UPDATE job_run_log
               SET end_at = datetime('now'), status = ?, error_message = ?, retry_count = ?
               WHERE run_id = ?""",
            (status, (error_message or "")[:1000], int(retry_count), run_id),
        )


def get_latest_effective_run(conn: sqlite3.Connection, job_name: str, trade_date: str) -> Optional[dict]:
    if not hasattr(conn, "execute"):
        return None
    _ensure_ddl_once(conn)
    row = conn.execute(
        """SELECT *
           FROM job_run_log
           WHERE job_name = ? AND trade_date = ? AND status IN ('success', 'partial_success')
           ORDER BY id DESC
           LIMIT 1""",
        (job_name, trade_date),
    ).fetchone()
    return dict(row) if row else None


def get_job_runs_on_date(conn: sqlite3.Connection, job_name: str, trade_date: str) -> list[dict]:
    if not hasattr(conn, "execute"):
        return []
    _ensure_ddl_once(conn)
    rows = conn.execute(
        """SELECT *
           FROM job_run_log
           WHERE job_name = ? AND trade_date = ?
           ORDER BY id DESC""",
        (job_name, trade_date),
    ).fetchall()
    return [dict(r) for r in rows]


def get_latest_successful_run_by_trigger(
    conn: sqlite3.Connection,
    job_name: str,
    trade_date: str,
    trigger_type: str,
) -> Optional[dict]:
    """Return the latest successful run for the same job/date/trigger tuple."""
    if not hasattr(conn, "execute"):
        return None
    _ensure_ddl_once(conn)
    row = conn.execute(
        """SELECT *
           FROM job_run_log
           WHERE job_name = ?
             AND trade_date = ?
             AND trigger_type = ?
             AND status IN ('success', 'partial_success')
           ORDER BY id DESC
           LIMIT 1""",
        (job_name, trade_date, trigger_type),
    ).fetchone()
    return dict(row) if row else None


def _ensure_strategy_version_registry_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS strategy_version_registry (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               key TEXT NOT NULL,
               version_value TEXT NOT NULL,
               version_hash TEXT NOT NULL,
               experiment_tag TEXT DEFAULT 'baseline',
               updated_at TEXT DEFAULT (datetime('now')),
               UNIQUE(key, experiment_tag)
           )"""
    )


def upsert_strategy_version(
    conn: sqlite3.Connection,
    key: str,
    version_value: str,
    experiment_tag: str = "baseline",
) -> None:
    if not hasattr(conn, "execute"):
        return
    _ensure_ddl_once(conn)
    h = sha1(version_value.encode("utf-8")).hexdigest()
    conn.execute(
        """INSERT INTO strategy_version_registry (key, version_value, version_hash, experiment_tag)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(key, experiment_tag) DO UPDATE SET
               version_value = excluded.version_value,
               version_hash = excluded.version_hash,
               updated_at = datetime('now')""",
        (key, version_value, h, experiment_tag),
    )


def get_strategy_versions(conn: sqlite3.Connection, experiment_tag: str = "baseline") -> list[dict]:
    _ensure_ddl_once(conn)
    rows = conn.execute(
        """SELECT key, version_value, version_hash, experiment_tag, updated_at
           FROM strategy_version_registry
           WHERE experiment_tag = ?
           ORDER BY key ASC""",
        (experiment_tag,),
    ).fetchall()
    return [dict(r) for r in rows]


def _ensure_strategy_validation_runs_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS strategy_validation_runs (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               validation_id TEXT NOT NULL UNIQUE,
               window_start TEXT NOT NULL,
               window_end TEXT NOT NULL,
               baseline_frozen_version TEXT NOT NULL,
               candidate_frozen_version TEXT NOT NULL,
               split_registry_json TEXT NOT NULL,
               summary_json TEXT NOT NULL,
               overfit_risk TEXT DEFAULT '',
               promotion_decision TEXT DEFAULT '',
               complexity_score REAL DEFAULT 0,
               complexity_delta_vs_baseline REAL DEFAULT 0,
               created_at TEXT DEFAULT (datetime('now'))
           )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_strategy_validation_window ON strategy_validation_runs(window_end, created_at)")


def _ensure_strategy_ablation_results_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS strategy_ablation_results (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               validation_id TEXT NOT NULL,
               rule_name TEXT NOT NULL,
               rule_type TEXT DEFAULT '',
               design_delta REAL DEFAULT 0,
               validation_delta REAL DEFAULT 0,
               holdout_delta REAL DEFAULT 0,
               complexity_delta REAL DEFAULT 0,
               classification TEXT DEFAULT '',
               details_json TEXT,
               created_at TEXT DEFAULT (datetime('now'))
           )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_strategy_ablation_validation ON strategy_ablation_results(validation_id, classification)")


def insert_strategy_validation_run(
    conn: sqlite3.Connection,
    *,
    validation_id: str,
    window_start: str,
    window_end: str,
    baseline_frozen_version: str,
    candidate_frozen_version: str,
    split_registry_json: str,
    summary_json: str,
    overfit_risk: str,
    promotion_decision: str,
    complexity_score: float,
    complexity_delta_vs_baseline: float,
) -> int:
    _ensure_ddl_once(conn)
    cur = conn.execute(
        """INSERT OR REPLACE INTO strategy_validation_runs
           (validation_id, window_start, window_end, baseline_frozen_version, candidate_frozen_version,
            split_registry_json, summary_json, overfit_risk, promotion_decision,
            complexity_score, complexity_delta_vs_baseline)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            validation_id,
            window_start,
            window_end,
            baseline_frozen_version,
            candidate_frozen_version,
            split_registry_json,
            summary_json,
            overfit_risk,
            promotion_decision,
            float(complexity_score or 0.0),
            float(complexity_delta_vs_baseline or 0.0),
        ),
    )
    return int(cur.lastrowid or 0)


def replace_strategy_ablation_results(conn: sqlite3.Connection, validation_id: str, rows: list[dict]) -> int:
    _ensure_ddl_once(conn)
    conn.execute("DELETE FROM strategy_ablation_results WHERE validation_id = ?", (validation_id,))
    inserted = 0
    for row in rows:
        conn.execute(
            """INSERT INTO strategy_ablation_results
               (validation_id, rule_name, rule_type, design_delta, validation_delta,
                holdout_delta, complexity_delta, classification, details_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                validation_id,
                str(row.get("rule_name") or ""),
                str(row.get("rule_type") or ""),
                float(row.get("design_delta", 0.0) or 0.0),
                float(row.get("validation_delta", 0.0) or 0.0),
                float(row.get("holdout_delta", 0.0) or 0.0),
                float(row.get("complexity_delta", 0.0) or 0.0),
                str(row.get("classification") or ""),
                json.dumps(row, ensure_ascii=False),
            ),
        )
        inserted += 1
    return inserted


def get_latest_strategy_validation_run(conn: sqlite3.Connection) -> dict | None:
    _ensure_ddl_once(conn)
    row = conn.execute(
        """SELECT * FROM strategy_validation_runs
           ORDER BY created_at DESC, id DESC
           LIMIT 1"""
    ).fetchone()
    return dict(row) if row else None


def get_strategy_ablation_results(conn: sqlite3.Connection, validation_id: str) -> list[dict]:
    _ensure_ddl_once(conn)
    rows = conn.execute(
        """SELECT * FROM strategy_ablation_results
           WHERE validation_id = ?
           ORDER BY id ASC""",
        (validation_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def _ensure_audit_findings_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS audit_findings (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               week_end TEXT NOT NULL,
               category TEXT NOT NULL,
               severity TEXT NOT NULL,
               finding_text TEXT NOT NULL,
               evidence_json TEXT,
               suggested_action TEXT,
               status TEXT DEFAULT 'open',
               source_run_id TEXT,
               created_at TEXT DEFAULT (datetime('now'))
           )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_findings_week ON audit_findings(week_end, severity)")


def insert_audit_finding(
    conn: sqlite3.Connection,
    week_end: str,
    category: str,
    severity: str,
    finding_text: str,
    evidence_json: str = "",
    suggested_action: str = "",
    source_run_id: str = "",
) -> int:
    _ensure_ddl_once(conn)
    cur = conn.execute(
        """INSERT INTO audit_findings
           (week_end, category, severity, finding_text, evidence_json, suggested_action, source_run_id)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (week_end, category, severity, finding_text, evidence_json, suggested_action, source_run_id or None),
    )
    return int(cur.lastrowid or 0)


def get_latest_audit_findings(conn: sqlite3.Connection, limit: int = 20) -> list[dict]:
    _ensure_ddl_once(conn)
    rows = conn.execute(
        """SELECT *
           FROM audit_findings
           ORDER BY id DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def _ensure_system_metrics_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS system_metrics (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               trade_date TEXT NOT NULL,
               run_id TEXT,
               metric_name TEXT NOT NULL,
               metric_value REAL NOT NULL,
               tags_json TEXT,
               created_at TEXT DEFAULT (datetime('now'))
           )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_system_metrics_lookup ON system_metrics(trade_date, metric_name)")


def insert_metric(
    conn: sqlite3.Connection,
    trade_date: str,
    metric_name: str,
    metric_value: float,
    run_id: str = "",
    tags_json: str = "",
) -> int:
    _ensure_ddl_once(conn)
    cur = conn.execute(
        """INSERT INTO system_metrics
           (trade_date, run_id, metric_name, metric_value, tags_json)
           VALUES (?, ?, ?, ?, ?)""",
        (trade_date, run_id or None, metric_name, float(metric_value), tags_json or ""),
    )
    return int(cur.lastrowid or 0)


def get_metrics_on_date(conn: sqlite3.Connection, trade_date: str) -> list[dict]:
    _ensure_ddl_once(conn)
    rows = conn.execute(
        """SELECT *
           FROM system_metrics
           WHERE trade_date = ?
           ORDER BY id ASC""",
        (trade_date,),
    ).fetchall()
    return [dict(r) for r in rows]


def _ensure_reflection_bundles_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS reflection_bundles (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               bundle_id TEXT NOT NULL UNIQUE,
               window_start TEXT NOT NULL,
               window_end TEXT NOT NULL,
               window_days INTEGER NOT NULL,
               strategy_versions_json TEXT,
               bundle_json TEXT NOT NULL,
               sample_count INTEGER DEFAULT 0,
               created_at TEXT DEFAULT (datetime('now'))
           )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_reflection_bundle_window ON reflection_bundles(window_start, window_end)")


def insert_reflection_bundle(
    conn: sqlite3.Connection,
    bundle_id: str,
    window_start: str,
    window_end: str,
    window_days: int,
    strategy_versions_json: str,
    bundle_json: str,
    sample_count: int,
) -> int:
    _ensure_ddl_once(conn)
    cur = conn.execute(
        """INSERT INTO reflection_bundles
           (bundle_id, window_start, window_end, window_days, strategy_versions_json, bundle_json, sample_count)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (bundle_id, window_start, window_end, int(window_days), strategy_versions_json, bundle_json, int(sample_count)),
    )
    return int(cur.lastrowid or 0)


def get_latest_reflection_bundle(conn: sqlite3.Connection) -> Optional[dict]:
    _ensure_ddl_once(conn)
    row = conn.execute("SELECT * FROM reflection_bundles ORDER BY id DESC LIMIT 1").fetchone()
    return dict(row) if row else None


def _ensure_model_review_chain_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS model_review_chain (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               chain_id TEXT NOT NULL,
               bundle_id TEXT NOT NULL,
               role TEXT NOT NULL,
               model_name TEXT,
               output_json TEXT NOT NULL,
               status TEXT DEFAULT 'ok',
               created_at TEXT DEFAULT (datetime('now'))
           )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_model_review_chain_chain ON model_review_chain(chain_id, role)")


def insert_model_review_chain_row(
    conn: sqlite3.Connection,
    chain_id: str,
    bundle_id: str,
    role: str,
    model_name: str,
    output_json: str,
    status: str = "ok",
) -> int:
    _ensure_ddl_once(conn)
    cur = conn.execute(
        """INSERT INTO model_review_chain
           (chain_id, bundle_id, role, model_name, output_json, status)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (chain_id, bundle_id, role, model_name, output_json, status),
    )
    return int(cur.lastrowid or 0)


def get_chain_rows(conn: sqlite3.Connection, chain_id: str) -> list[dict]:
    _ensure_ddl_once(conn)
    rows = conn.execute(
        "SELECT * FROM model_review_chain WHERE chain_id = ? ORDER BY id ASC",
        (chain_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def _ensure_upgrade_proposals_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS upgrade_proposals (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               proposal_id TEXT NOT NULL UNIQUE,
               created_at TEXT DEFAULT (datetime('now')),
               created_from_run_window TEXT NOT NULL,
               created_from_audit_id TEXT,
               proposer_model TEXT NOT NULL,
               target_component TEXT NOT NULL,
               current_version TEXT,
               proposed_change_type TEXT NOT NULL,
               proposed_patch_summary TEXT NOT NULL,
               rationale TEXT NOT NULL,
               evidence_refs_json TEXT NOT NULL,
               expected_benefit TEXT,
               primary_risk TEXT,
               rollback_condition TEXT,
               approval_status TEXT NOT NULL DEFAULT 'draft',
               experiment_plan_json TEXT
           )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_upgrade_proposals_status ON upgrade_proposals(approval_status, proposed_change_type)")


def insert_upgrade_proposal(
    conn: sqlite3.Connection,
    proposal_id: str,
    created_from_run_window: str,
    created_from_audit_id: str,
    proposer_model: str,
    target_component: str,
    current_version: str,
    proposed_change_type: str,
    proposed_patch_summary: str,
    rationale: str,
    evidence_refs_json: str,
    expected_benefit: str = "",
    primary_risk: str = "",
    rollback_condition: str = "",
    approval_status: str = "draft",
    experiment_plan_json: str = "",
) -> int:
    _ensure_ddl_once(conn)
    try:
        refs = json.loads(evidence_refs_json or "[]")
    except Exception:
        refs = []
    if not refs:
        raise ValueError("evidence_refs_json must be non-empty")
    cur = conn.execute(
        """INSERT INTO upgrade_proposals
           (proposal_id, created_from_run_window, created_from_audit_id, proposer_model, target_component,
            current_version, proposed_change_type, proposed_patch_summary, rationale, evidence_refs_json,
            expected_benefit, primary_risk, rollback_condition, approval_status, experiment_plan_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            proposal_id,
            created_from_run_window,
            created_from_audit_id or None,
            proposer_model,
            target_component,
            current_version,
            proposed_change_type,
            proposed_patch_summary,
            rationale,
            evidence_refs_json,
            expected_benefit,
            primary_risk,
            rollback_condition,
            approval_status,
            experiment_plan_json,
        ),
    )
    return int(cur.lastrowid or 0)


def update_proposal_status(conn: sqlite3.Connection, proposal_id: str, approval_status: str) -> int:
    _ensure_ddl_once(conn)
    cur = conn.execute(
        "UPDATE upgrade_proposals SET approval_status = ? WHERE proposal_id = ?",
        (approval_status, proposal_id),
    )
    return int(cur.rowcount or 0)


def get_proposal(conn: sqlite3.Connection, proposal_id: str) -> Optional[dict]:
    _ensure_ddl_once(conn)
    row = conn.execute("SELECT * FROM upgrade_proposals WHERE proposal_id = ? LIMIT 1", (proposal_id,)).fetchone()
    return dict(row) if row else None


def get_proposals(conn: sqlite3.Connection, status: str | None = None) -> list[dict]:
    _ensure_ddl_once(conn)
    if status:
        rows = conn.execute(
            "SELECT * FROM upgrade_proposals WHERE approval_status = ? ORDER BY id DESC",
            (status,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM upgrade_proposals ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]


def _ensure_governance_experiments_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS governance_experiments (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               experiment_id TEXT NOT NULL UNIQUE,
               experiment_tag TEXT NOT NULL,
               proposal_id TEXT NOT NULL,
               base_version TEXT,
               candidate_version TEXT,
               start_date TEXT NOT NULL,
               end_date TEXT,
               success_criteria_json TEXT NOT NULL,
               stop_conditions_json TEXT NOT NULL,
               verdict TEXT DEFAULT 'inconclusive',
               verdict_note TEXT,
               status TEXT DEFAULT 'running',
               created_at TEXT DEFAULT (datetime('now'))
           )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_governance_experiments_tag ON governance_experiments(experiment_tag, status)")


def create_governance_experiment(
    conn: sqlite3.Connection,
    experiment_id: str,
    experiment_tag: str,
    proposal_id: str,
    base_version: str,
    candidate_version: str,
    start_date: str,
    success_criteria_json: str,
    stop_conditions_json: str,
) -> int:
    _ensure_ddl_once(conn)
    proposal = get_proposal(conn, proposal_id)
    if proposal is None:
        raise ValueError("proposal not found")
    status = str(proposal.get("approval_status", "draft"))
    if status not in {"approved_for_shadow", "approved_for_ab"}:
        raise PermissionError("proposal not approved for experiment")
    cur = conn.execute(
        """INSERT INTO governance_experiments
           (experiment_id, experiment_tag, proposal_id, base_version, candidate_version,
            start_date, success_criteria_json, stop_conditions_json, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running')""",
        (experiment_id, experiment_tag, proposal_id, base_version, candidate_version, start_date, success_criteria_json, stop_conditions_json),
    )
    return int(cur.lastrowid or 0)


def finalize_experiment_verdict(
    conn: sqlite3.Connection,
    experiment_id: str,
    verdict: str,
    verdict_note: str = "",
    end_date: str = "",
) -> int:
    _ensure_ddl_once(conn)
    cur = conn.execute(
        """UPDATE governance_experiments
           SET verdict = ?, verdict_note = ?, end_date = ?, status = 'finished'
           WHERE experiment_id = ?""",
        (verdict, verdict_note[:500], end_date or datetime.now().strftime("%Y-%m-%d"), experiment_id),
    )
    return int(cur.rowcount or 0)


def get_experiment(conn: sqlite3.Connection, experiment_id: str) -> Optional[dict]:
    _ensure_ddl_once(conn)
    row = conn.execute(
        "SELECT * FROM governance_experiments WHERE experiment_id = ? LIMIT 1",
        (experiment_id,),
    ).fetchone()
    return dict(row) if row else None


def get_experiments(conn: sqlite3.Connection, status: str | None = None) -> list[dict]:
    _ensure_ddl_once(conn)
    if status:
        rows = conn.execute(
            "SELECT * FROM governance_experiments WHERE status = ? ORDER BY id DESC",
            (status,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM governance_experiments ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]


def _ensure_governance_approvals_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS governance_approvals (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               entity_type TEXT NOT NULL,
               entity_id TEXT NOT NULL,
               reviewer TEXT NOT NULL,
               decision TEXT NOT NULL,
               decision_time TEXT DEFAULT (datetime('now')),
               decision_note TEXT,
               approved_scope TEXT,
               rollback_plan_ack INTEGER DEFAULT 0,
               UNIQUE(entity_type, entity_id, reviewer)
           )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_governance_approvals_entity ON governance_approvals(entity_type, entity_id)")


def insert_governance_approval(
    conn: sqlite3.Connection,
    entity_type: str,
    entity_id: str,
    reviewer: str,
    decision: str,
    decision_note: str = "",
    approved_scope: str = "",
    rollback_plan_ack: bool = False,
) -> int:
    _ensure_ddl_once(conn)
    cur = conn.execute(
        """INSERT OR REPLACE INTO governance_approvals
           (entity_type, entity_id, reviewer, decision, decision_note, approved_scope, rollback_plan_ack)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (entity_type, entity_id, reviewer, decision, decision_note[:500], approved_scope[:500], 1 if rollback_plan_ack else 0),
    )
    return int(cur.lastrowid or 0)


def get_governance_approvals(conn: sqlite3.Connection, entity_type: str, entity_id: str) -> list[dict]:
    _ensure_ddl_once(conn)
    rows = conn.execute(
        """SELECT * FROM governance_approvals
           WHERE entity_type = ? AND entity_id = ?
           ORDER BY id DESC""",
        (entity_type, entity_id),
    ).fetchall()
    return [dict(r) for r in rows]


def _ensure_live_promotion_log_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS live_promotion_log (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               promotion_id TEXT NOT NULL UNIQUE,
               proposal_id TEXT NOT NULL,
               experiment_id TEXT NOT NULL,
               promoted_version TEXT NOT NULL,
               reviewer TEXT NOT NULL,
               decision_note TEXT,
               created_at TEXT DEFAULT (datetime('now'))
           )"""
    )


def promote_proposal_to_live(
    conn: sqlite3.Connection,
    promotion_id: str,
    proposal_id: str,
    experiment_id: str,
    promoted_version: str,
    reviewer: str,
    decision_note: str = "",
) -> int:
    _ensure_ddl_once(conn)
    proposal = get_proposal(conn, proposal_id)
    if proposal is None:
        raise ValueError("proposal not found")
    if str(proposal.get("approval_status")) != "approved_for_live":
        raise PermissionError("proposal not approved_for_live")
    exp = get_experiment(conn, experiment_id)
    if exp is None:
        raise ValueError("experiment not found")
    if str(exp.get("verdict")) != "ready_for_live_promotion":
        raise PermissionError("experiment verdict not ready_for_live_promotion")
    approvals = get_governance_approvals(conn, "promotion", proposal_id)
    if not any(str(a.get("decision")) == "approve" for a in approvals):
        raise PermissionError("missing human approval for promotion")

    cur = conn.execute(
        """INSERT INTO live_promotion_log
           (promotion_id, proposal_id, experiment_id, promoted_version, reviewer, decision_note)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (promotion_id, proposal_id, experiment_id, promoted_version, reviewer, decision_note[:500]),
    )
    # Bind promoted version lineage into version registry.
    upsert_strategy_version(conn, "live_promotion_source", json.dumps({
        "proposal_id": proposal_id,
        "experiment_id": experiment_id,
        "promotion_id": promotion_id,
    }, ensure_ascii=False), experiment_tag="live")
    return int(cur.lastrowid or 0)


def get_live_promotions(conn: sqlite3.Connection) -> list[dict]:
    _ensure_ddl_once(conn)
    rows = conn.execute("SELECT * FROM live_promotion_log ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]


def _ensure_regret_records_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS regret_records (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               trade_date TEXT NOT NULL,
               stock_code TEXT NOT NULL,
               mode TEXT NOT NULL DEFAULT 'live',
               final_outcome TEXT NOT NULL,
               regret_type TEXT NOT NULL,
               severity_score REAL DEFAULT 0,
               observation_window TEXT NOT NULL,
               metrics_json TEXT NOT NULL,
               evidence_refs_json TEXT NOT NULL,
               taxonomy_version TEXT NOT NULL,
               sample_window TEXT NOT NULL,
               version_snapshot_json TEXT,
               created_at TEXT DEFAULT (datetime('now'))
           )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_regret_records_date_type ON regret_records(trade_date, regret_type, mode)")


def insert_regret_record(
    conn: sqlite3.Connection,
    trade_date: str,
    stock_code: str,
    mode: str,
    final_outcome: str,
    regret_type: str,
    severity_score: float,
    observation_window: str,
    metrics_json: str,
    evidence_refs_json: str,
    taxonomy_version: str,
    sample_window: str,
    version_snapshot_json: str = "",
) -> int:
    _ensure_ddl_once(conn)
    cur = conn.execute(
        """INSERT INTO regret_records
           (trade_date, stock_code, mode, final_outcome, regret_type, severity_score,
            observation_window, metrics_json, evidence_refs_json, taxonomy_version,
            sample_window, version_snapshot_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            trade_date,
            stock_code,
            mode,
            final_outcome,
            regret_type,
            float(severity_score),
            observation_window,
            metrics_json,
            evidence_refs_json,
            taxonomy_version,
            sample_window,
            version_snapshot_json,
        ),
    )
    return int(cur.lastrowid or 0)


def get_regret_records(
    conn: sqlite3.Connection,
    start_date: str,
    end_date: str,
    mode: str | None = None,
) -> list[dict]:
    _ensure_ddl_once(conn)
    if mode:
        rows = conn.execute(
            """SELECT * FROM regret_records
               WHERE trade_date >= ? AND trade_date <= ? AND mode = ?
               ORDER BY severity_score DESC, id DESC""",
            (start_date, end_date, mode),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT * FROM regret_records
               WHERE trade_date >= ? AND trade_date <= ?
               ORDER BY severity_score DESC, id DESC""",
            (start_date, end_date),
        ).fetchall()
    return [dict(r) for r in rows]


def _ensure_research_priorities_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS research_priorities (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               as_of_date TEXT NOT NULL,
               priority_level TEXT NOT NULL,
               problem_statement TEXT NOT NULL,
               suspected_root_cause TEXT,
               likely_target_component TEXT,
               recommended_change_type TEXT,
               supporting_evidence_json TEXT,
               score_json TEXT,
               created_at TEXT DEFAULT (datetime('now'))
           )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_research_priorities_asof ON research_priorities(as_of_date, priority_level)")


def insert_research_priority(
    conn: sqlite3.Connection,
    as_of_date: str,
    priority_level: str,
    problem_statement: str,
    suspected_root_cause: str = "",
    likely_target_component: str = "",
    recommended_change_type: str = "",
    supporting_evidence_json: str = "",
    score_json: str = "",
) -> int:
    _ensure_ddl_once(conn)
    cur = conn.execute(
        """INSERT INTO research_priorities
           (as_of_date, priority_level, problem_statement, suspected_root_cause,
            likely_target_component, recommended_change_type, supporting_evidence_json, score_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            as_of_date,
            priority_level,
            problem_statement,
            suspected_root_cause,
            likely_target_component,
            recommended_change_type,
            supporting_evidence_json,
            score_json,
        ),
    )
    return int(cur.lastrowid or 0)


def get_research_priorities(conn: sqlite3.Connection, as_of_date: str | None = None) -> list[dict]:
    _ensure_ddl_once(conn)
    if as_of_date:
        rows = conn.execute(
            "SELECT * FROM research_priorities WHERE as_of_date = ? ORDER BY id DESC",
            (as_of_date,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM research_priorities ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]


def _ensure_replay_runs_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS replay_runs (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               replay_id TEXT NOT NULL UNIQUE,
               sample_window TEXT NOT NULL,
               component_type TEXT NOT NULL,
               base_version TEXT NOT NULL,
               candidate_version TEXT NOT NULL,
               diff_summary_json TEXT NOT NULL,
               limitation_note TEXT,
               version_snapshot_json TEXT,
               created_at TEXT DEFAULT (datetime('now'))
           )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_replay_runs_window ON replay_runs(sample_window, component_type)")


def insert_replay_run(
    conn: sqlite3.Connection,
    replay_id: str,
    sample_window: str,
    component_type: str,
    base_version: str,
    candidate_version: str,
    diff_summary_json: str,
    limitation_note: str = "",
    version_snapshot_json: str = "",
) -> int:
    _ensure_ddl_once(conn)
    cur = conn.execute(
        """INSERT INTO replay_runs
           (replay_id, sample_window, component_type, base_version, candidate_version,
            diff_summary_json, limitation_note, version_snapshot_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            replay_id,
            sample_window,
            component_type,
            base_version,
            candidate_version,
            diff_summary_json,
            limitation_note,
            version_snapshot_json,
        ),
    )
    return int(cur.lastrowid or 0)


def get_replay_runs(conn: sqlite3.Connection, sample_window: str | None = None) -> list[dict]:
    _ensure_ddl_once(conn)
    if sample_window:
        rows = conn.execute(
            "SELECT * FROM replay_runs WHERE sample_window = ? ORDER BY id DESC",
            (sample_window,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM replay_runs ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]


def _ensure_research_memos_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS research_memos (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               memo_id TEXT NOT NULL UNIQUE,
               hypothesis TEXT NOT NULL,
               evidence_json TEXT NOT NULL,
               verdict TEXT NOT NULL,
               expiry_condition TEXT,
               status TEXT DEFAULT 'active',
               created_at TEXT DEFAULT (datetime('now'))
           )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_research_memos_status ON research_memos(status, created_at)")


def insert_research_memo(
    conn: sqlite3.Connection,
    memo_id: str,
    hypothesis: str,
    evidence_json: str,
    verdict: str,
    expiry_condition: str = "",
    status: str = "active",
) -> int:
    _ensure_ddl_once(conn)
    cur = conn.execute(
        """INSERT INTO research_memos
           (memo_id, hypothesis, evidence_json, verdict, expiry_condition, status)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (memo_id, hypothesis, evidence_json, verdict, expiry_condition, status),
    )
    return int(cur.lastrowid or 0)


def get_research_memos(conn: sqlite3.Connection, status: str | None = None) -> list[dict]:
    _ensure_ddl_once(conn)
    if status:
        rows = conn.execute(
            "SELECT * FROM research_memos WHERE status = ? ORDER BY id DESC",
            (status,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM research_memos ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]


def _ensure_weight_allocation_log_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS weight_allocation_log (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               trade_date TEXT NOT NULL,
               weights_json TEXT NOT NULL,
               reasoning TEXT,
               market_summary_json TEXT,
               agent_summaries_json TEXT,
               fallback_used INTEGER DEFAULT 0,
               llm_model TEXT,
               llm_latency_ms INTEGER,
               created_at TEXT DEFAULT (datetime('now'))
           )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_weight_allocation_date ON weight_allocation_log(trade_date)")


def insert_weight_allocation_log(
    conn: sqlite3.Connection,
    trade_date: str,
    weights_json: str,
    reasoning: str = "",
    market_summary_json: str = "",
    agent_summaries_json: str = "",
    fallback_used: int = 0,
    llm_model: str = "",
    llm_latency_ms: int | None = None,
) -> int:
    _ensure_ddl_once(conn)
    cursor = conn.execute(
        """INSERT INTO weight_allocation_log
           (trade_date, weights_json, reasoning, market_summary_json, agent_summaries_json,
            fallback_used, llm_model, llm_latency_ms)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            trade_date,
            weights_json,
            reasoning,
            market_summary_json,
            agent_summaries_json,
            int(fallback_used),
            llm_model,
            llm_latency_ms if llm_latency_ms is not None else None,
        ),
    )
    return cursor.lastrowid


def get_weight_allocation_on_date(conn: sqlite3.Connection, trade_date: str) -> Optional[dict]:
    _ensure_ddl_once(conn)
    row = conn.execute(
        """SELECT * FROM weight_allocation_log
           WHERE trade_date = ?
           ORDER BY id DESC
           LIMIT 1""",
        (trade_date,),
    ).fetchone()
    return dict(row) if row else None


def _ensure_opus_decision_log_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS opus_decision_log (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               trade_date TEXT NOT NULL,
               decisions_json TEXT NOT NULL,
               market_assessment TEXT,
               adversarial_summary TEXT,
               overrides_count INTEGER DEFAULT 0,
               tool_calls_json TEXT,
               conversation_json TEXT,
               total_rounds INTEGER,
               total_input_tokens INTEGER,
               total_output_tokens INTEGER,
               latency_seconds REAL,
               status TEXT DEFAULT 'success',
               error_message TEXT,
               created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
           )"""
    )
    # Migration: add columns if table already exists without them
    try:
        conn.execute("SELECT adversarial_summary FROM opus_decision_log LIMIT 0")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE opus_decision_log ADD COLUMN adversarial_summary TEXT")
        conn.execute("ALTER TABLE opus_decision_log ADD COLUMN overrides_count INTEGER DEFAULT 0")
    # Migration: add status/error fields for auditable failure reporting.
    try:
        conn.execute("SELECT status FROM opus_decision_log LIMIT 0")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE opus_decision_log ADD COLUMN status TEXT DEFAULT 'success'")
    try:
        conn.execute("SELECT error_message FROM opus_decision_log LIMIT 0")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE opus_decision_log ADD COLUMN error_message TEXT")


def insert_opus_decision_log(
    conn: sqlite3.Connection,
    trade_date: str,
    decisions_json: str,
    market_assessment: str = "",
    adversarial_summary: str = "",
    overrides_count: int = 0,
    tool_calls_json: str = "",
    conversation_json: str = "",
    total_rounds: int = 0,
    total_input_tokens: int = 0,
    total_output_tokens: int = 0,
    latency_seconds: float = 0.0,
    status: str = "success",
    error_message: str = "",
) -> int:
    _ensure_ddl_once(conn)
    cur = conn.execute(
        """INSERT INTO opus_decision_log
           (trade_date, decisions_json, market_assessment, adversarial_summary,
            overrides_count, tool_calls_json, conversation_json, total_rounds,
            total_input_tokens, total_output_tokens, latency_seconds, status, error_message)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (trade_date, decisions_json, market_assessment, adversarial_summary,
         overrides_count, tool_calls_json, conversation_json, total_rounds,
         total_input_tokens, total_output_tokens, latency_seconds, status, error_message[:500]),
    )
    return cur.lastrowid


def get_opus_decision_log(conn: sqlite3.Connection, trade_date: str) -> list[dict]:
    _ensure_ddl_once(conn)
    rows = conn.execute(
        "SELECT * FROM opus_decision_log WHERE trade_date = ? ORDER BY id DESC",
        (trade_date,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_latest_opus_decision_log(conn: sqlite3.Connection, trade_date: str) -> Optional[dict]:
    _ensure_ddl_once(conn)
    row = conn.execute(
        """SELECT *
           FROM opus_decision_log
           WHERE trade_date = ?
           ORDER BY id DESC
           LIMIT 1""",
        (trade_date,),
    ).fetchone()
    return dict(row) if row else None


# ── pass_tracking ──

def _ensure_pass_tracking_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS pass_tracking (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               decision_date TEXT NOT NULL,
               ts_code TEXT NOT NULL,
               stock_name TEXT,
               sector_name TEXT,
               prescore REAL,
               pass_reason TEXT,
               day1_pct REAL,
               day2_pct REAL,
               day3_pct REAL,
               day5_pct REAL,
               max_gain_5d REAL,
               max_loss_5d REAL,
               tracking_complete INTEGER DEFAULT 0,
               created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
           )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_pass_tracking_date ON pass_tracking(decision_date)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_pass_tracking_incomplete ON pass_tracking(tracking_complete)"
    )


def insert_pass_tracking(
    conn: sqlite3.Connection,
    decision_date: str,
    ts_code: str,
    stock_name: str = "",
    sector_name: str = "",
    prescore: float = 0.0,
    pass_reason: str = "",
) -> int:
    _ensure_ddl_once(conn)
    cur = conn.execute(
        """INSERT INTO pass_tracking
           (decision_date, ts_code, stock_name, sector_name, prescore, pass_reason)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (decision_date, ts_code, stock_name, sector_name, prescore, pass_reason),
    )
    return cur.lastrowid


def delete_pass_tracking_on_date(conn: sqlite3.Connection, decision_date: str) -> int:
    """Delete pass-tracking rows on date to keep reruns idempotent."""
    _ensure_ddl_once(conn)
    cursor = conn.execute(
        "DELETE FROM pass_tracking WHERE decision_date = ?",
        (decision_date,),
    )
    return int(cursor.rowcount or 0)


def get_incomplete_pass_tracking(conn: sqlite3.Connection) -> list[dict]:
    _ensure_ddl_once(conn)
    rows = conn.execute(
        "SELECT * FROM pass_tracking WHERE tracking_complete = 0 ORDER BY decision_date ASC"
    ).fetchall()
    return [dict(r) for r in rows]


def update_pass_tracking_prices(
    conn: sqlite3.Connection,
    record_id: int,
    day1_pct: float | None = None,
    day2_pct: float | None = None,
    day3_pct: float | None = None,
    day5_pct: float | None = None,
    max_gain_5d: float | None = None,
    max_loss_5d: float | None = None,
    tracking_complete: bool = False,
) -> None:
    _ensure_ddl_once(conn)
    conn.execute(
        """UPDATE pass_tracking
           SET day1_pct = COALESCE(?, day1_pct),
               day2_pct = COALESCE(?, day2_pct),
               day3_pct = COALESCE(?, day3_pct),
               day5_pct = COALESCE(?, day5_pct),
               max_gain_5d = COALESCE(?, max_gain_5d),
               max_loss_5d = COALESCE(?, max_loss_5d),
               tracking_complete = ?
           WHERE id = ?""",
        (day1_pct, day2_pct, day3_pct, day5_pct, max_gain_5d, max_loss_5d,
         1 if tracking_complete else 0, record_id),
    )


def get_pass_tracking_since(conn: sqlite3.Connection, start_date: str) -> list[dict]:
    _ensure_ddl_once(conn)
    rows = conn.execute(
        """SELECT * FROM pass_tracking
           WHERE decision_date >= ? AND tracking_complete = 1
           ORDER BY decision_date DESC""",
        (start_date,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_pass_tracking_stats(conn: sqlite3.Connection, start_date: str) -> dict:
    """Get PASS tracking statistics since start_date."""
    _ensure_ddl_once(conn)
    rows = conn.execute(
        """SELECT * FROM pass_tracking WHERE decision_date >= ?""",
        (start_date,),
    ).fetchall()
    total = len(rows)
    complete = sum(1 for r in rows if r["tracking_complete"])
    correct = sum(1 for r in rows if r["tracking_complete"]
                  and r["max_gain_5d"] is not None and r["max_gain_5d"] < 5.0)
    return {"total": total, "complete": complete, "correct": correct}


def get_papertrade_activity_dates_since(conn: sqlite3.Connection, start_date: str) -> list[str]:
    """Union of active trading dates across scheduler artifacts for paper-day counting."""
    _ensure_ddl_once(conn)
    rows = conn.execute(
        """WITH dates AS (
               SELECT check_date AS d FROM macro_switch_log WHERE check_date >= ?
               UNION
               SELECT trade_date AS d FROM opus_decision_log WHERE trade_date >= ?
               UNION
               SELECT signal_date AS d FROM signals WHERE signal_date >= ?
               UNION
               SELECT decision_date AS d FROM buy_decisions WHERE decision_date >= ?
               UNION
               SELECT entry_date AS d FROM positions WHERE entry_date >= ?
               UNION
               SELECT trade_date AS d FROM candidate_decision_trace WHERE trade_date >= ?
           )
           SELECT d
           FROM dates
           WHERE d IS NOT NULL AND d <> ''
           ORDER BY d ASC""",
        (start_date, start_date, start_date, start_date, start_date, start_date),
    ).fetchall()
    return [str(r["d"]) for r in rows]


# ── candidate_decision_trace (single source of truth read model) ──

def _ensure_candidate_decision_trace_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS candidate_decision_trace (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               trade_date TEXT NOT NULL,
               stock_code TEXT NOT NULL,
               mode TEXT NOT NULL DEFAULT 'live',
               stage TEXT NOT NULL,
               status TEXT NOT NULL,
               reason_code TEXT,
               details_json TEXT,
               created_at TEXT DEFAULT (datetime('now'))
           )"""
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_trace_date_mode_stock
           ON candidate_decision_trace(trade_date, mode, stock_code)"""
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_trace_stage
           ON candidate_decision_trace(trade_date, mode, stage, status)"""
    )


def reset_candidate_decision_trace_on_date(
    conn: sqlite3.Connection,
    trade_date: str,
    mode: str | None = None,
) -> int:
    """Delete trace rows on date (and mode) for rerun idempotency."""
    _ensure_ddl_once(conn)
    if mode:
        cur = conn.execute(
            "DELETE FROM candidate_decision_trace WHERE trade_date = ? AND mode = ?",
            (trade_date, mode),
        )
    else:
        cur = conn.execute(
            "DELETE FROM candidate_decision_trace WHERE trade_date = ?",
            (trade_date,),
        )
    return int(cur.rowcount or 0)


def upsert_candidate_decision_trace(
    conn: sqlite3.Connection,
    trade_date: str,
    stock_code: str,
    mode: str,
    stage: str,
    status: str,
    reason_code: str = "",
    details_json: str = "",
) -> int:
    """
    Upsert one stage row for (trade_date, stock_code, mode, stage).
    Keeps latest-only semantics for report consistency.
    """
    _ensure_ddl_once(conn)
    conn.execute(
        """DELETE FROM candidate_decision_trace
           WHERE trade_date = ? AND stock_code = ? AND mode = ? AND stage = ?""",
        (trade_date, stock_code, mode, stage),
    )
    cur = conn.execute(
        """INSERT INTO candidate_decision_trace
           (trade_date, stock_code, mode, stage, status, reason_code, details_json)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (trade_date, stock_code, mode, stage, status, reason_code, details_json),
    )
    return int(cur.lastrowid or 0)


def get_candidate_decision_trace_on_date(
    conn: sqlite3.Connection,
    trade_date: str,
    mode: str | None = None,
) -> list[dict]:
    _ensure_ddl_once(conn)
    if mode:
        rows = conn.execute(
            """SELECT *
               FROM candidate_decision_trace
               WHERE trade_date = ? AND mode = ?
               ORDER BY stock_code ASC, id ASC""",
            (trade_date, mode),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT *
               FROM candidate_decision_trace
               WHERE trade_date = ?
               ORDER BY mode ASC, stock_code ASC, id ASC""",
            (trade_date,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_candidate_final_outcomes_on_date(
    conn: sqlite3.Connection,
    trade_date: str,
    mode: str | None = None,
) -> list[dict]:
    _ensure_ddl_once(conn)
    params: tuple[Any, ...]
    mode_clause = ""
    if mode:
        mode_clause = " AND mode = ?"
        params = (trade_date, mode)
    else:
        params = (trade_date,)
    rows = conn.execute(
        f"""SELECT trade_date, stock_code, mode, status AS final_outcome, reason_code, details_json
            FROM candidate_decision_trace
            WHERE trade_date = ? AND stage = 'final_outcome'{mode_clause}
            ORDER BY stock_code ASC""",
        params,
    ).fetchall()
    return [dict(r) for r in rows]


def get_blocker_attribution_on_date(
    conn: sqlite3.Connection,
    trade_date: str,
    mode: str | None = None,
) -> list[dict]:
    """Aggregate final outcomes by reason_code for daily blocker attribution."""
    _ensure_ddl_once(conn)
    params: tuple[Any, ...]
    mode_clause = ""
    if mode:
        mode_clause = " AND mode = ?"
        params = (trade_date, mode)
    else:
        params = (trade_date,)
    rows = conn.execute(
        f"""SELECT COALESCE(reason_code, '') AS reason_code, COUNT(*) AS cnt
            FROM candidate_decision_trace
            WHERE trade_date = ? AND stage = 'final_outcome'{mode_clause}
            GROUP BY COALESCE(reason_code, '')
            ORDER BY cnt DESC, reason_code ASC""",
        params,
    ).fetchall()
    return [dict(r) for r in rows]


def backfill_trace_from_legacy(conn: sqlite3.Connection, trade_date: str) -> dict:
    """
    Best-effort backfill for legacy days lacking candidate_decision_trace.
    Marks reason/status as legacy_mapped/inferred where needed.
    """
    _ensure_ddl_once(conn)
    existing = conn.execute(
        "SELECT COUNT(*) AS c FROM candidate_decision_trace WHERE trade_date = ?",
        (trade_date,),
    ).fetchone()
    if existing and int(existing["c"] or 0) > 0:
        return {"created": 0, "mode": "existing"}

    created = 0
    decisions = get_buy_decisions_on_date(conn, trade_date)
    for d in decisions:
        code = str(d.get("stock_code", "")).strip()
        if not code:
            continue
        created += upsert_candidate_decision_trace(
            conn, trade_date, code, "live", "opus_decision",
            "buy" if str(d.get("decision", "")).upper() == "BUY" else "pass",
            "legacy_mapped",
            json.dumps({"source": "buy_decisions", "inferred": True}, ensure_ascii=False),
        )
        final = "position_opened" if str(d.get("decision", "")).upper() == "BUY" else "passed_by_opus"
        created += upsert_candidate_decision_trace(
            conn, trade_date, code, "live", "final_outcome",
            "done", final,
            json.dumps({"source": "buy_decisions", "inferred": True}, ensure_ascii=False),
        )

    sigs = get_signals_on_date_with_mode(conn, trade_date, dedup=True)
    for s in sigs:
        code = str(s.get("stock_code", "")).strip()
        if not code:
            continue
        mode = str(s.get("mode", "live") or "live")
        created += upsert_candidate_decision_trace(
            conn, trade_date, code, mode, "signal_created", "created", "legacy_mapped",
            json.dumps({"source": "signals", "inferred": True}, ensure_ascii=False),
        )
        if mode == "shadow":
            out = "shadow_only"
        else:
            pos = conn.execute(
                "SELECT 1 FROM positions WHERE stock_code = ? AND entry_date = ? LIMIT 1",
                (code, trade_date),
            ).fetchone()
            out = "position_opened" if pos else "signal_only"
        created += upsert_candidate_decision_trace(
            conn, trade_date, code, mode, "final_outcome", "done", out,
            json.dumps({"source": "signals+positions", "inferred": True}, ensure_ascii=False),
        )

    return {"created": created, "mode": "legacy_mapped"}


def compare_live_shadow_outcomes(
    conn: sqlite3.Connection,
    start_date: str,
    end_date: str | None = None,
) -> dict:
    """
    Lightweight live-vs-shadow comparison for recent N-day review.
    Counts final outcomes by mode from the unified trace model.
    """
    _ensure_ddl_once(conn)
    rows = conn.execute(
        """SELECT mode, reason_code AS final_outcome, COUNT(*) AS cnt
           FROM candidate_decision_trace
           WHERE trade_date >= ?
             AND trade_date <= COALESCE(?, trade_date)
             AND stage = 'final_outcome'
           GROUP BY mode, reason_code""",
        (start_date, end_date),
    ).fetchall()
    out = {"live": {}, "shadow": {}}
    for r in rows:
        mode = str(r["mode"] or "live")
        if mode not in out:
            out[mode] = {}
        out[mode][str(r["final_outcome"] or "")] = int(r["cnt"] or 0)
    return out


# ── prescreener structured results ──

def _ensure_prescreener_run_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS prescreener_run (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               run_id TEXT NOT NULL UNIQUE,
               trade_date TEXT NOT NULL,
               price_date TEXT NOT NULL,
               candidate_source_mode TEXT NOT NULL,
               prescreener_version TEXT NOT NULL,
               sector_scoring_version TEXT NOT NULL,
               theme_score_version TEXT DEFAULT '',
               review_rank_version TEXT DEFAULT '',
               regime_state TEXT DEFAULT 'RUN',
               market_count INTEGER DEFAULT 0,
               layer1_count INTEGER DEFAULT 0,
               top_n_count INTEGER DEFAULT 0,
               selected_for_opus_count INTEGER DEFAULT 0,
               etf_list_count INTEGER DEFAULT 0,
               etf_daily_rows INTEGER DEFAULT 0,
               funnel_json TEXT,
               top_sectors_json TEXT,
               top_themes_by_total_json TEXT,
               top_themes_by_trend_json TEXT,
               promoted_wildcard_themes_json TEXT,
               excluded_by_theme_cap_json TEXT,
               selected_sectors_json TEXT,
               tech_monitor_json TEXT,
               notes_json TEXT,
               created_at TEXT DEFAULT (datetime('now'))
           )"""
    )
    cols = conn.execute("PRAGMA table_info(prescreener_run)").fetchall()
    col_names = {str(r["name"]) for r in cols}
    if "theme_score_version" not in col_names:
        conn.execute("ALTER TABLE prescreener_run ADD COLUMN theme_score_version TEXT DEFAULT ''")
    if "review_rank_version" not in col_names:
        conn.execute("ALTER TABLE prescreener_run ADD COLUMN review_rank_version TEXT DEFAULT ''")
    if "top_themes_by_total_json" not in col_names:
        conn.execute("ALTER TABLE prescreener_run ADD COLUMN top_themes_by_total_json TEXT")
    if "top_themes_by_trend_json" not in col_names:
        conn.execute("ALTER TABLE prescreener_run ADD COLUMN top_themes_by_trend_json TEXT")
    if "promoted_wildcard_themes_json" not in col_names:
        conn.execute("ALTER TABLE prescreener_run ADD COLUMN promoted_wildcard_themes_json TEXT")
    if "excluded_by_theme_cap_json" not in col_names:
        conn.execute("ALTER TABLE prescreener_run ADD COLUMN excluded_by_theme_cap_json TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prescreener_run_lookup ON prescreener_run(trade_date, run_id)")


def _ensure_prescreener_candidate_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS prescreener_candidate (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               run_id TEXT NOT NULL,
               trade_date TEXT NOT NULL,
               price_date TEXT NOT NULL,
               stock_code TEXT NOT NULL,
               rank_position INTEGER DEFAULT 0,
               selected_for_opus INTEGER DEFAULT 0,
               selected_for_opus_rank INTEGER DEFAULT 0,
               candidate_source_mode TEXT NOT NULL,
               prescore REAL DEFAULT 0,
               sentiment_score REAL DEFAULT 0,
               trend_hotness_score REAL DEFAULT 0,
               stock_quality_score REAL DEFAULT 0,
               role_strength_score REAL DEFAULT 0,
               review_rank_score REAL DEFAULT 0,
               selected_sector_primary TEXT,
               selected_sector_secondary TEXT,
               sector_selection_reason TEXT,
               phase TEXT,
               etf_code TEXT,
               all_matched_sectors_json TEXT,
               matched_sectors_topk_json TEXT,
               sector_scores_by_name_json TEXT,
               score_components_json TEXT,
               review_rank_components_json TEXT,
               selection_bucket TEXT DEFAULT '',
               notes_json TEXT,
               created_at TEXT DEFAULT (datetime('now')),
               UNIQUE(run_id, stock_code)
           )"""
    )
    cols = conn.execute("PRAGMA table_info(prescreener_candidate)").fetchall()
    col_names = {str(r["name"]) for r in cols}
    if "role_strength_score" not in col_names:
        conn.execute("ALTER TABLE prescreener_candidate ADD COLUMN role_strength_score REAL DEFAULT 0")
    if "review_rank_score" not in col_names:
        conn.execute("ALTER TABLE prescreener_candidate ADD COLUMN review_rank_score REAL DEFAULT 0")
    if "review_rank_components_json" not in col_names:
        conn.execute("ALTER TABLE prescreener_candidate ADD COLUMN review_rank_components_json TEXT")
    if "selection_bucket" not in col_names:
        conn.execute("ALTER TABLE prescreener_candidate ADD COLUMN selection_bucket TEXT DEFAULT ''")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_prescreener_candidate_lookup ON prescreener_candidate(trade_date, run_id, rank_position)"
    )


def _ensure_prescreener_theme_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS prescreener_theme (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               run_id TEXT NOT NULL,
               trade_date TEXT NOT NULL,
               price_date TEXT NOT NULL,
               theme_name TEXT NOT NULL,
               theme_score_version TEXT NOT NULL,
               theme_total_score REAL DEFAULT 0,
               theme_trend_score REAL DEFAULT 0,
               theme_sentiment_score REAL DEFAULT 0,
               theme_rank_total INTEGER DEFAULT 0,
               theme_rank_trend INTEGER DEFAULT 0,
               quota_allocated INTEGER DEFAULT 0,
               wildcard_promoted INTEGER DEFAULT 0,
               excluded_due_to_theme_cap INTEGER DEFAULT 0,
               quota_reason TEXT DEFAULT '',
               candidate_count INTEGER DEFAULT 0,
               member_count INTEGER DEFAULT 0,
               score_components_json TEXT,
               notes_json TEXT,
               created_at TEXT DEFAULT (datetime('now')),
               UNIQUE(run_id, theme_name)
           )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prescreener_theme_lookup ON prescreener_theme(trade_date, run_id, theme_rank_total)")


def _ensure_prescreener_replacement_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS prescreener_replacement (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               run_id TEXT NOT NULL,
               trade_date TEXT NOT NULL,
               price_date TEXT NOT NULL,
               rank_slot INTEGER DEFAULT 0,
               replaced_out_stock TEXT NOT NULL,
               promoted_in_stock TEXT NOT NULL,
               old_rank_score REAL DEFAULT 0,
               new_rank_score REAL DEFAULT 0,
               sector_total_diff REAL DEFAULT 0,
               stock_quality_diff REAL DEFAULT 0,
               role_strength_diff REAL DEFAULT 0,
               replacement_reason TEXT NOT NULL,
               details_json TEXT,
               created_at TEXT DEFAULT (datetime('now'))
           )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_prescreener_replacement_lookup ON prescreener_replacement(trade_date, run_id, rank_slot)"
    )


def upsert_prescreener_run(conn: sqlite3.Connection, row: dict) -> int:
    _ensure_ddl_once(conn)
    cur = conn.execute(
        """INSERT OR REPLACE INTO prescreener_run
           (run_id, trade_date, price_date, candidate_source_mode, prescreener_version,
            sector_scoring_version, theme_score_version, review_rank_version, regime_state, market_count, layer1_count, top_n_count,
            selected_for_opus_count, etf_list_count, etf_daily_rows, funnel_json, top_sectors_json, top_themes_by_total_json,
            top_themes_by_trend_json, promoted_wildcard_themes_json, excluded_by_theme_cap_json,
            selected_sectors_json, tech_monitor_json, notes_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            row.get("run_id", ""),
            row.get("trade_date", ""),
            row.get("price_date", ""),
            row.get("candidate_source_mode", ""),
            row.get("prescreener_version", ""),
            row.get("sector_scoring_version", ""),
            row.get("theme_score_version", "") or "",
            row.get("review_rank_version", "") or "",
            row.get("regime_state", "RUN"),
            int(row.get("market_count", 0) or 0),
            int(row.get("layer1_count", 0) or 0),
            int(row.get("top_n_count", 0) or 0),
            int(row.get("selected_for_opus_count", 0) or 0),
            int(row.get("etf_list_count", 0) or 0),
            int(row.get("etf_daily_rows", 0) or 0),
            row.get("funnel_json", "") or "",
            row.get("top_sectors_json", "") or "",
            row.get("top_themes_by_total_json", "") or "",
            row.get("top_themes_by_trend_json", "") or "",
            row.get("promoted_wildcard_themes_json", "") or "",
            row.get("excluded_by_theme_cap_json", "") or "",
            row.get("selected_sectors_json", "") or "",
            row.get("tech_monitor_json", "") or "",
            row.get("notes_json", "") or "",
        ),
    )
    return int(cur.lastrowid or 0)


def replace_prescreener_candidates(conn: sqlite3.Connection, run_id: str, rows: list[dict]) -> int:
    _ensure_ddl_once(conn)
    conn.execute("DELETE FROM prescreener_candidate WHERE run_id = ?", (run_id,))
    count = 0
    for row in rows:
        conn.execute(
            """INSERT INTO prescreener_candidate
               (run_id, trade_date, price_date, stock_code, rank_position, selected_for_opus, selected_for_opus_rank,
                candidate_source_mode, prescore, sentiment_score, trend_hotness_score, stock_quality_score, role_strength_score, review_rank_score,
                selected_sector_primary, selected_sector_secondary, sector_selection_reason, phase, etf_code,
                all_matched_sectors_json, matched_sectors_topk_json, sector_scores_by_name_json, score_components_json, review_rank_components_json, selection_bucket, notes_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                row.get("trade_date", ""),
                row.get("price_date", ""),
                row.get("stock_code", ""),
                int(row.get("rank_position", 0) or 0),
                1 if row.get("selected_for_opus") else 0,
                int(row.get("selected_for_opus_rank", 0) or 0),
                row.get("candidate_source_mode", "") or "",
                float(row.get("prescore", 0) or 0),
                float(row.get("sentiment_score", 0) or 0),
                float(row.get("trend_hotness_score", 0) or 0),
                float(row.get("stock_quality_score", 0) or 0),
                float(row.get("role_strength_score", 0) or 0),
                float(row.get("review_rank_score", 0) or 0),
                row.get("selected_sector_primary", "") or "",
                row.get("selected_sector_secondary", "") or "",
                row.get("sector_selection_reason", "") or "",
                row.get("phase", "") or "",
                row.get("etf_code", "") or "",
                row.get("all_matched_sectors_json", "") or "",
                row.get("matched_sectors_topk_json", "") or "",
                row.get("sector_scores_by_name_json", "") or "",
                row.get("score_components_json", "") or "",
                row.get("review_rank_components_json", "") or "",
                row.get("selection_bucket", "") or "",
                row.get("notes_json", "") or "",
            ),
        )
        count += 1
    return count


def replace_prescreener_themes(conn: sqlite3.Connection, run_id: str, rows: list[dict]) -> int:
    _ensure_ddl_once(conn)
    conn.execute("DELETE FROM prescreener_theme WHERE run_id = ?", (run_id,))
    count = 0
    for row in rows:
        conn.execute(
            """INSERT INTO prescreener_theme
               (run_id, trade_date, price_date, theme_name, theme_score_version, theme_total_score, theme_trend_score,
                theme_sentiment_score, theme_rank_total, theme_rank_trend, quota_allocated, wildcard_promoted,
                excluded_due_to_theme_cap, quota_reason, candidate_count, member_count, score_components_json, notes_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                row.get("trade_date", ""),
                row.get("price_date", ""),
                row.get("theme_name", ""),
                row.get("theme_score_version", "") or "",
                float(row.get("theme_total_score", 0) or 0),
                float(row.get("theme_trend_score", 0) or 0),
                float(row.get("theme_sentiment_score", 0) or 0),
                int(row.get("theme_rank_total", 0) or 0),
                int(row.get("theme_rank_trend", 0) or 0),
                int(row.get("quota_allocated", 0) or 0),
                1 if row.get("wildcard_promoted") else 0,
                int(row.get("excluded_due_to_theme_cap", 0) or 0),
                row.get("quota_reason", "") or "",
                int(row.get("candidate_count", 0) or 0),
                int(row.get("member_count", 0) or 0),
                row.get("score_components_json", "") or "",
                row.get("notes_json", "") or "",
            ),
        )
        count += 1
    return count


def get_prescreener_run(conn: sqlite3.Connection, run_id: str) -> Optional[dict]:
    _ensure_ddl_once(conn)
    row = conn.execute("SELECT * FROM prescreener_run WHERE run_id = ? LIMIT 1", (run_id,)).fetchone()
    return dict(row) if row else None


def get_prescreener_candidates(conn: sqlite3.Connection, run_id: str) -> list[dict]:
    _ensure_ddl_once(conn)
    rows = conn.execute(
        """SELECT *
           FROM prescreener_candidate
           WHERE run_id = ?
           ORDER BY rank_position ASC, stock_code ASC""",
        (run_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_prescreener_themes(conn: sqlite3.Connection, run_id: str) -> list[dict]:
    _ensure_ddl_once(conn)
    rows = conn.execute(
        """SELECT *
           FROM prescreener_theme
           WHERE run_id = ?
           ORDER BY theme_rank_total ASC, theme_rank_trend ASC, theme_name ASC""",
        (run_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def replace_prescreener_replacements(conn: sqlite3.Connection, run_id: str, rows: list[dict]) -> int:
    _ensure_ddl_once(conn)
    conn.execute("DELETE FROM prescreener_replacement WHERE run_id = ?", (run_id,))
    count = 0
    for row in rows:
        conn.execute(
            """INSERT INTO prescreener_replacement
               (run_id, trade_date, price_date, rank_slot, replaced_out_stock, promoted_in_stock,
                old_rank_score, new_rank_score, sector_total_diff, stock_quality_diff, role_strength_diff,
                replacement_reason, details_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                row.get("trade_date", ""),
                row.get("price_date", ""),
                int(row.get("rank_slot", 0) or 0),
                row.get("replaced_out_stock", "") or "",
                row.get("promoted_in_stock", "") or "",
                float(row.get("old_rank_score", 0) or 0),
                float(row.get("new_rank_score", 0) or 0),
                float(row.get("sector_total_diff", 0) or 0),
                float(row.get("stock_quality_diff", 0) or 0),
                float(row.get("role_strength_diff", 0) or 0),
                row.get("replacement_reason", "") or "",
                row.get("details_json", "") or "",
            ),
        )
        count += 1
    return count


def get_prescreener_replacements(conn: sqlite3.Connection, run_id: str) -> list[dict]:
    _ensure_ddl_once(conn)
    rows = conn.execute(
        """SELECT *
           FROM prescreener_replacement
           WHERE run_id = ?
           ORDER BY rank_slot ASC, id ASC""",
        (run_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_prescreener_replacements_window(conn: sqlite3.Connection, start_date: str, end_date: str) -> list[dict]:
    _ensure_ddl_once(conn)
    rows = conn.execute(
        """SELECT *
           FROM prescreener_replacement
           WHERE trade_date >= ? AND trade_date <= ?
           ORDER BY trade_date ASC, rank_slot ASC, id ASC""",
        (start_date, end_date),
    ).fetchall()
    return [dict(r) for r in rows]


def get_prescreener_themes_window(conn: sqlite3.Connection, start_date: str, end_date: str) -> list[dict]:
    _ensure_ddl_once(conn)
    rows = conn.execute(
        """SELECT *
           FROM prescreener_theme
           WHERE trade_date >= ? AND trade_date <= ?
           ORDER BY trade_date ASC, theme_rank_total ASC, theme_rank_trend ASC, theme_name ASC""",
        (start_date, end_date),
    ).fetchall()
    return [dict(r) for r in rows]


# ── trading_lessons ──

def _ensure_trading_lessons_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS trading_lessons (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               lesson TEXT NOT NULL,
               category TEXT,
               source TEXT NOT NULL,
               priority TEXT NOT NULL,
               created_date TEXT NOT NULL,
               expiry_date TEXT,
               hit_count INTEGER DEFAULT 0,
               miss_count INTEGER DEFAULT 0,
               confidence REAL DEFAULT 0.5,
               status TEXT DEFAULT 'pending',
               review_note TEXT,
               updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
               created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
           )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_lessons_status ON trading_lessons(status)"
    )


def insert_trading_lesson(
    conn: sqlite3.Connection,
    lesson: str,
    category: str,
    source: str,
    priority: str,
    created_date: str,
    expiry_date: str | None = None,
    status: str = "pending",
) -> int:
    _ensure_ddl_once(conn)
    cur = conn.execute(
        """INSERT INTO trading_lessons
           (lesson, category, source, priority, created_date, expiry_date, status)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (lesson, category, source, priority, created_date, expiry_date, status),
    )
    return cur.lastrowid


def get_active_lessons(conn: sqlite3.Connection) -> list[dict]:
    _ensure_ddl_once(conn)
    rows = conn.execute(
        """SELECT * FROM trading_lessons
           WHERE status = 'active'
           ORDER BY
             CASE priority WHEN 'P0' THEN 0 WHEN 'P1' THEN 1 WHEN 'P2' THEN 2 ELSE 3 END,
             confidence DESC"""
    ).fetchall()
    return [dict(r) for r in rows]


def get_lessons_by_status(conn: sqlite3.Connection, status: str) -> list[dict]:
    _ensure_ddl_once(conn)
    rows = conn.execute(
        "SELECT * FROM trading_lessons WHERE status = ? ORDER BY id DESC",
        (status,),
    ).fetchall()
    return [dict(r) for r in rows]


def update_lesson_status(
    conn: sqlite3.Connection, lesson_id: int, status: str, review_note: str = ""
) -> None:
    _ensure_ddl_once(conn)
    conn.execute(
        """UPDATE trading_lessons
           SET status = ?, review_note = COALESCE(NULLIF(?, ''), review_note),
               updated_at = CURRENT_TIMESTAMP
           WHERE id = ?""",
        (status, review_note, lesson_id),
    )


def update_lesson_text(conn: sqlite3.Connection, lesson_id: int, lesson: str) -> None:
    _ensure_ddl_once(conn)
    conn.execute(
        "UPDATE trading_lessons SET lesson = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (lesson, lesson_id),
    )


def update_lesson_confidence(
    conn: sqlite3.Connection, lesson_id: int, hit: bool
) -> None:
    _ensure_ddl_once(conn)
    if hit:
        conn.execute(
            """UPDATE trading_lessons
               SET hit_count = hit_count + 1,
                   confidence = CAST(hit_count + 1 AS REAL) / (hit_count + 1 + miss_count),
                   updated_at = CURRENT_TIMESTAMP
               WHERE id = ?""",
            (lesson_id,),
        )
    else:
        conn.execute(
            """UPDATE trading_lessons
               SET miss_count = miss_count + 1,
                   confidence = CAST(hit_count AS REAL) / (hit_count + miss_count + 1),
                   updated_at = CURRENT_TIMESTAMP
               WHERE id = ?""",
            (lesson_id,),
        )
    # Auto-flag low confidence lessons
    row = conn.execute(
        "SELECT hit_count, miss_count, confidence FROM trading_lessons WHERE id = ?",
        (lesson_id,),
    ).fetchone()
    if row and (row["hit_count"] + row["miss_count"]) >= 5 and row["confidence"] < 0.3:
        conn.execute(
            "UPDATE trading_lessons SET status = 'pending_revoke' WHERE id = ? AND status = 'active'",
            (lesson_id,),
        )


def expire_outdated_lessons(conn: sqlite3.Connection, today: str) -> int:
    """Expire P1 lessons past expiry_date and P2 lessons > 30 days without validation."""
    _ensure_ddl_once(conn)
    # P1 past expiry
    conn.execute(
        """UPDATE trading_lessons
           SET status = 'expired', updated_at = CURRENT_TIMESTAMP
           WHERE status = 'active' AND priority = 'P1'
             AND expiry_date IS NOT NULL AND expiry_date < ?""",
        (today,),
    )
    # P2 older than 30 days with no validation
    conn.execute(
        """UPDATE trading_lessons
           SET status = 'expired', updated_at = CURRENT_TIMESTAMP
           WHERE status = 'active' AND priority = 'P2'
             AND (hit_count + miss_count) = 0
             AND julianday(?) - julianday(created_date) > 30""",
        (today,),
    )
    return conn.execute("SELECT changes()").fetchone()[0]


def get_lesson_stats(conn: sqlite3.Connection) -> dict:
    _ensure_ddl_once(conn)
    rows = conn.execute(
        """SELECT priority, status, COUNT(*) as cnt
           FROM trading_lessons
           GROUP BY priority, status"""
    ).fetchall()
    stats = {"P0": 0, "P1": 0, "P2": 0, "pending": 0, "total_active": 0}
    for r in rows:
        if r["status"] == "active":
            stats[r["priority"]] = stats.get(r["priority"], 0) + r["cnt"]
            stats["total_active"] += r["cnt"]
        elif r["status"] == "pending":
            stats["pending"] += r["cnt"]
    return stats


def lesson_exists(conn: sqlite3.Connection, lesson: str, source: str) -> bool:
    """Check if an identical lesson from the same source already exists."""
    _ensure_ddl_once(conn)
    row = conn.execute(
        "SELECT 1 FROM trading_lessons WHERE lesson = ? AND source = ? LIMIT 1",
        (lesson, source),
    ).fetchone()
    return row is not None


# ── weekly_review_log ──

def _ensure_weekly_review_log_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS weekly_review_log (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               week_start TEXT NOT NULL,
               week_end TEXT NOT NULL,
               review_json TEXT NOT NULL,
               advocate_json TEXT,
               challenger_json TEXT,
               arbitrator_json TEXT,
               adversarial_quality_score INTEGER,
               challenger_model TEXT,
               arbitrator_model TEXT,
               new_lessons_count INTEGER,
               revoke_suggestions_count INTEGER,
               systematic_bias TEXT,
               next_week_focus TEXT,
               opus_input_tokens INTEGER,
               opus_output_tokens INTEGER,
               cost_usd REAL,
               created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
           )"""
    )
    # Migration for existing tables
    for col, ctype in [
        ("advocate_json", "TEXT"),
        ("challenger_json", "TEXT"),
        ("arbitrator_json", "TEXT"),
        ("adversarial_quality_score", "INTEGER"),
        ("challenger_model", "TEXT"),
        ("arbitrator_model", "TEXT"),
    ]:
        try:
            conn.execute(f"SELECT {col} FROM weekly_review_log LIMIT 0")
        except sqlite3.OperationalError:
            conn.execute(f"ALTER TABLE weekly_review_log ADD COLUMN {col} {ctype}")


def insert_weekly_review(
    conn: sqlite3.Connection,
    week_start: str,
    week_end: str,
    review_json: str,
    advocate_json: str = "",
    challenger_json: str = "",
    arbitrator_json: str = "",
    adversarial_quality_score: int | None = None,
    challenger_model: str = "",
    arbitrator_model: str = "",
    new_lessons_count: int = 0,
    revoke_suggestions_count: int = 0,
    systematic_bias: str = "",
    next_week_focus: str = "",
    opus_input_tokens: int = 0,
    opus_output_tokens: int = 0,
    cost_usd: float = 0.0,
) -> int:
    _ensure_ddl_once(conn)
    cur = conn.execute(
        """INSERT INTO weekly_review_log
           (week_start, week_end, review_json, advocate_json, challenger_json,
            arbitrator_json, adversarial_quality_score, challenger_model,
            arbitrator_model, new_lessons_count, revoke_suggestions_count,
            systematic_bias, next_week_focus, opus_input_tokens,
            opus_output_tokens, cost_usd)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (week_start, week_end, review_json, advocate_json, challenger_json,
         arbitrator_json, adversarial_quality_score, challenger_model,
         arbitrator_model, new_lessons_count, revoke_suggestions_count,
         systematic_bias, next_week_focus, opus_input_tokens,
         opus_output_tokens, cost_usd),
    )
    return cur.lastrowid


def get_latest_weekly_review(conn: sqlite3.Connection) -> dict | None:
    _ensure_ddl_once(conn)
    row = conn.execute(
        "SELECT * FROM weekly_review_log ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return dict(row) if row else None


# ── prompt_suggestions ──

def _ensure_prompt_suggestions_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS prompt_suggestions (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               week_end TEXT NOT NULL,
               suggestion TEXT NOT NULL,
               source TEXT DEFAULT 'arbitrator',
               status TEXT DEFAULT 'pending',
               applied_date TEXT,
               created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
           )"""
    )


def insert_prompt_suggestion(
    conn: sqlite3.Connection,
    week_end: str,
    suggestion: str,
    source: str = "arbitrator",
) -> int:
    _ensure_ddl_once(conn)
    cur = conn.execute(
        "INSERT INTO prompt_suggestions (week_end, suggestion, source) VALUES (?, ?, ?)",
        (week_end, suggestion, source),
    )
    return cur.lastrowid


def get_pending_prompt_suggestions(conn: sqlite3.Connection) -> list[dict]:
    _ensure_ddl_once(conn)
    rows = conn.execute(
        "SELECT * FROM prompt_suggestions WHERE status = 'pending' ORDER BY id DESC"
    ).fetchall()
    return [dict(r) for r in rows]


# ── sector_membership ──

def _ensure_sector_membership_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS sector_membership (
               ts_code TEXT NOT NULL,
               sector_name TEXT NOT NULL,
               source TEXT DEFAULT 'industry',
               updated_date TEXT NOT NULL,
               PRIMARY KEY (ts_code, sector_name)
           )"""
    )


def bulk_upsert_sector_membership(conn, records: list[dict]) -> int:
    """Upsert sector membership records. Each dict has ts_code, sector_name, source, updated_date."""
    _ensure_ddl_once(conn)
    count = 0
    for r in records:
        conn.execute(
            "INSERT OR REPLACE INTO sector_membership (ts_code, sector_name, source, updated_date) VALUES (?, ?, ?, ?)",
            (r["ts_code"], r["sector_name"], r.get("source", "industry"), r["updated_date"]),
        )
        count += 1
    return count


def get_sector_mapping(conn) -> dict:
    """Return {sector_name: [ts_code, ...]} from sector_membership table."""
    _ensure_ddl_once(conn)
    rows = conn.execute("SELECT ts_code, sector_name FROM sector_membership").fetchall()
    mapping = {}
    for r in rows:
        mapping.setdefault(r["sector_name"], []).append(r["ts_code"])
    return mapping


def get_sector_membership_age_days(conn) -> int | None:
    """Return age in days of the sector membership data, or None if empty."""
    _ensure_ddl_once(conn)
    row = conn.execute("SELECT MAX(updated_date) as latest FROM sector_membership").fetchone()
    if row is None or row["latest"] is None:
        return None
    from datetime import datetime
    try:
        latest = datetime.strptime(row["latest"], "%Y%m%d")
        return (datetime.now() - latest).days
    except Exception:
        return None


# ── etf_list / etf_holdings ──

def _ensure_etf_list_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS etf_list (
               ts_code TEXT PRIMARY KEY,
               name TEXT NOT NULL,
               fund_type TEXT DEFAULT '',
               benchmark TEXT DEFAULT '',
               management TEXT DEFAULT '',
               list_date TEXT DEFAULT '',
               updated_date TEXT NOT NULL
           )"""
    )


def _ensure_etf_holdings_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS etf_holdings (
               etf_code TEXT NOT NULL,
               ts_code TEXT NOT NULL,
               weight REAL DEFAULT 0,
               updated_date TEXT NOT NULL,
               PRIMARY KEY (etf_code, ts_code)
           )"""
    )


def bulk_upsert_etf_list(conn, records: list[dict]) -> int:
    _ensure_ddl_once(conn)
    count = 0
    for r in records:
        conn.execute(
            "INSERT OR REPLACE INTO etf_list (ts_code, name, fund_type, benchmark, management, list_date, updated_date) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (r["ts_code"], r["name"], r.get("fund_type", ""), r.get("benchmark", ""),
             r.get("management", ""), r.get("list_date", ""), r["updated_date"]),
        )
        count += 1
    return count


def bulk_upsert_etf_holdings(conn, records: list[dict]) -> int:
    _ensure_ddl_once(conn)
    count = 0
    for r in records:
        conn.execute(
            "INSERT OR REPLACE INTO etf_holdings (etf_code, ts_code, weight, updated_date) "
            "VALUES (?, ?, ?, ?)",
            (r["etf_code"], r["ts_code"], r.get("weight", 0), r["updated_date"]),
        )
        count += 1
    return count


def get_etf_list(conn) -> list[dict]:
    """Return all ETFs in etf_list table."""
    _ensure_ddl_once(conn)
    rows = conn.execute("SELECT * FROM etf_list ORDER BY ts_code").fetchall()
    return [dict(r) for r in rows]


def get_etf_holdings_map(conn) -> dict:
    """Return {etf_code: [ts_code, ...]} from etf_holdings table."""
    _ensure_ddl_once(conn)
    rows = conn.execute("SELECT etf_code, ts_code FROM etf_holdings").fetchall()
    mapping: dict = {}
    for r in rows:
        mapping.setdefault(r["etf_code"], []).append(r["ts_code"])
    return mapping


def get_stock_to_etfs(conn) -> dict:
    """Return {ts_code: [(etf_code, etf_name), ...]}."""
    _ensure_ddl_once(conn)
    rows = conn.execute(
        "SELECT h.ts_code, h.etf_code, e.name AS etf_name "
        "FROM etf_holdings h JOIN etf_list e ON h.etf_code = e.ts_code"
    ).fetchall()
    mapping: dict = {}
    for r in rows:
        mapping.setdefault(r["ts_code"], []).append((r["etf_code"], r["etf_name"]))
    return mapping


def get_etf_list_age_days(conn) -> int | None:
    _ensure_ddl_once(conn)
    row = conn.execute("SELECT MAX(updated_date) as latest FROM etf_list").fetchone()
    if row is None or row["latest"] is None:
        return None
    from datetime import datetime
    try:
        latest = datetime.strptime(row["latest"], "%Y%m%d")
        return (datetime.now() - latest).days
    except Exception:
        return None
