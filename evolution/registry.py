"""
Version registry for tracking strategy evolution history.
Stores all versions, evaluations, and promotions in SQLite.
"""

import json
import sqlite3
from datetime import datetime
from typing import List, Optional

from loguru import logger

from momentum.config import MomentumConfig

from .evaluator import PerformanceReport


class EvolutionRegistry:
    """Manages version history and evolution state."""

    @staticmethod
    def _ensure_tables(conn: sqlite3.Connection) -> None:
        """Create evolution tables if they don't exist."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evolution_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT NOT NULL UNIQUE,
                parent_version TEXT,
                config_json TEXT NOT NULL,
                description TEXT,
                mutation_reason TEXT,
                created_at TEXT NOT NULL,
                promoted_at TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS evolution_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT NOT NULL,
                evaluation_date TEXT NOT NULL,
                window_days INTEGER,
                total_return REAL,
                annualized_return REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                calmar_ratio REAL,
                win_rate REAL,
                avg_turnover REAL,
                total_trades INTEGER,
                information_ratio REAL,
                evaluation_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (version) REFERENCES evolution_versions(version)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS evolution_promotions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT NOT NULL,
                promoted_at TEXT NOT NULL,
                reason TEXT,
                compared_with TEXT,
                confidence REAL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (version) REFERENCES evolution_versions(version)
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_evolution_versions_parent
            ON evolution_versions(parent_version)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_evolution_versions_created
            ON evolution_versions(created_at)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_evolution_evaluations_version
            ON evolution_evaluations(version, evaluation_date)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_evolution_promotions_version
            ON evolution_promotions(version)
        """)

    @staticmethod
    def register_version(
        conn: sqlite3.Connection,
        config: MomentumConfig,
        parent_version: Optional[str] = None,
        mutation_reason: str = "",
    ) -> str:
        """
        Register a new strategy version in the registry.

        Args:
            conn: Database connection
            config: MomentumConfig to register
            parent_version: Version this was mutated from
            mutation_reason: Description of what changed

        Returns:
            The registered version identifier
        """
        EvolutionRegistry._ensure_tables(conn)

        version = config.version
        config_json = json.dumps(config.to_dict())

        logger.info(f"Registering version {version}")

        try:
            conn.execute(
                """
                INSERT INTO evolution_versions
                (version, parent_version, config_json, description, mutation_reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    version,
                    parent_version or config.parent_version,
                    config_json,
                    config.description,
                    mutation_reason,
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()
            logger.info(f"Registered version {version}")
            return version
        except sqlite3.IntegrityError:
            logger.warning(f"Version {version} already registered")
            return version

    @staticmethod
    def get_version(
        conn: sqlite3.Connection,
        version: str,
    ) -> Optional[MomentumConfig]:
        """
        Retrieve a registered configuration by version.

        Args:
            conn: Database connection
            version: Version identifier

        Returns:
            MomentumConfig or None if not found
        """
        EvolutionRegistry._ensure_tables(conn)

        row = conn.execute(
            "SELECT config_json FROM evolution_versions WHERE version = ?",
            (version,),
        ).fetchone()

        if not row:
            logger.warning(f"Version {version} not found in registry")
            return None

        try:
            config_data = json.loads(row["config_json"])
            config = MomentumConfig.from_dict(config_data)
            logger.info(f"Retrieved version {version}")
            return config
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to deserialize version {version}: {e}")
            return None

    @staticmethod
    def get_lineage(
        conn: sqlite3.Connection,
        version: str,
    ) -> List[dict]:
        """
        Get full evolution history (lineage) of a version.

        Args:
            conn: Database connection
            version: Starting version

        Returns:
            List of dicts with version info, ordered from root to current
        """
        EvolutionRegistry._ensure_tables(conn)

        lineage = []
        current = version

        # Walk backwards to root
        while current:
            row = conn.execute(
                """
                SELECT version, parent_version, description, mutation_reason, created_at, promoted_at
                FROM evolution_versions
                WHERE version = ?
                """,
                (current,),
            ).fetchone()

            if not row:
                break

            lineage.append({
                "version": row["version"],
                "parent": row["parent_version"],
                "description": row["description"],
                "mutation_reason": row["mutation_reason"],
                "created_at": row["created_at"],
                "promoted_at": row["promoted_at"],
            })

            current = row["parent_version"]

        # Reverse to show root first
        lineage.reverse()
        logger.info(f"Retrieved lineage for {version}: {len(lineage)} versions")
        return lineage

    @staticmethod
    def promote_version(
        conn: sqlite3.Connection,
        version: str,
        compared_with: Optional[str] = None,
        confidence: float = 0.8,
        reason: str = "",
    ) -> None:
        """
        Mark a version as promoted (active in production).

        Args:
            conn: Database connection
            version: Version to promote
            compared_with: Version it was compared against
            confidence: Confidence in promotion (0-1)
            reason: Why this version was promoted
        """
        EvolutionRegistry._ensure_tables(conn)

        logger.info(f"Promoting version {version} (confidence={confidence:.1%})")

        now = datetime.now().isoformat()

        # Mark version as promoted
        conn.execute(
            "UPDATE evolution_versions SET promoted_at = ? WHERE version = ?",
            (now, version),
        )

        # Record promotion
        conn.execute(
            """
            INSERT INTO evolution_promotions
            (version, promoted_at, reason, compared_with, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (version, now, reason, compared_with, confidence, now),
        )
        conn.commit()
        logger.info(f"Version {version} promoted")

    @staticmethod
    def get_active_version(
        conn: sqlite3.Connection,
    ) -> Optional[MomentumConfig]:
        """
        Get the currently active (most recently promoted) version.

        Returns:
            MomentumConfig of active version or None
        """
        EvolutionRegistry._ensure_tables(conn)

        row = conn.execute(
            """
            SELECT version FROM evolution_versions
            WHERE promoted_at IS NOT NULL
            ORDER BY promoted_at DESC
            LIMIT 1
            """
        ).fetchone()

        if not row:
            logger.warning("No promoted version found, using baseline")
            return None

        version = row["version"]
        config = EvolutionRegistry.get_version(conn, version)
        if config:
            logger.info(f"Active version: {version}")
        return config

    @staticmethod
    def record_evaluation(
        conn: sqlite3.Connection,
        version: str,
        perf: PerformanceReport,
    ) -> None:
        """
        Record evaluation results for a version.

        Args:
            conn: Database connection
            version: Version being evaluated
            perf: PerformanceReport with metrics
        """
        EvolutionRegistry._ensure_tables(conn)

        perf_json = json.dumps({
            "total_return": perf.total_return,
            "annualized_return": perf.annualized_return,
            "sharpe_ratio": perf.sharpe_ratio,
            "max_drawdown": perf.max_drawdown,
            "calmar_ratio": perf.calmar_ratio,
            "win_rate": perf.win_rate,
            "avg_turnover": perf.avg_turnover,
            "total_trades": perf.total_trades,
            "information_ratio": perf.information_ratio,
            "num_rebalances": perf.num_rebalances,
        })

        logger.info(f"Recording evaluation for {version}")

        conn.execute(
            """
            INSERT INTO evolution_evaluations
            (version, evaluation_date, window_days, total_return, annualized_return,
             sharpe_ratio, max_drawdown, calmar_ratio, win_rate, avg_turnover,
             total_trades, information_ratio, evaluation_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                version,
                perf.as_of,
                perf.window_days,
                perf.total_return,
                perf.annualized_return,
                perf.sharpe_ratio,
                perf.max_drawdown,
                perf.calmar_ratio,
                perf.win_rate,
                perf.avg_turnover,
                perf.total_trades,
                perf.information_ratio,
                perf_json,
                datetime.now().isoformat(),
            ),
        )
        conn.commit()
        logger.info(f"Evaluation recorded for {version}")

    @staticmethod
    def get_version_history(
        conn: sqlite3.Connection,
        limit: int = 50,
    ) -> List[dict]:
        """
        Get recent version history.

        Args:
            conn: Database connection
            limit: Maximum number of versions to return

        Returns:
            List of version dicts ordered by creation time (newest first)
        """
        EvolutionRegistry._ensure_tables(conn)

        rows = conn.execute(
            """
            SELECT version, parent_version, description, created_at, promoted_at
            FROM evolution_versions
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        history = [
            {
                "version": row["version"],
                "parent": row["parent_version"],
                "description": row["description"],
                "created_at": row["created_at"],
                "promoted": row["promoted_at"] is not None,
            }
            for row in rows
        ]

        logger.info(f"Retrieved version history: {len(history)} entries")
        return history
