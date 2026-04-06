"""Integration tests for the momentum rotation + evolution system.

All tests are offline — Tushare and LLM are fully mocked.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db():
    """Create a temporary SQLite database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    yield conn
    conn.close()


@pytest.fixture
def sample_config():
    """Create a sample MomentumConfig."""
    from momentum.config import MomentumConfig
    return MomentumConfig(
        lookback_days=[20, 60],
        lookback_weights=[0.6, 0.4],
        universe_min_market_cap=5.0,
        universe_min_avg_turnover=20.0,
        top_n=5,
        rebalance_weekday=4,
        rebalance_threshold=0.3,
        max_single_weight=0.25,
        stop_loss_pct=-0.08,
        momentum_type="simple",
    )


@pytest.fixture
def mock_tushare():
    """Mock TushareClient."""
    client = MagicMock()

    # stock_basic
    client.query.return_value = pd.DataFrame({
        "ts_code": ["000001.SZ", "000002.SZ", "000063.SZ", "600036.SH", "601318.SH",
                     "000858.SZ", "002415.SZ", "300750.SZ", "600519.SH", "601012.SH"],
        "name": ["平安银行", "万科A", "中兴通讯", "招商银行", "中国平安",
                 "五粮液", "海康威视", "宁德时代", "贵州茅台", "隆基绿能"],
        "industry": ["银行", "房地产", "通信设备", "银行", "保险",
                     "白酒", "安防", "电池", "白酒", "光伏"],
        "market": ["主板", "主板", "主板", "主板", "主板",
                   "主板", "中小板", "创业板", "主板", "主板"],
        "list_date": ["19910403", "19910129", "19970118", "20020409", "20070301",
                      "20030424", "20100520", "20180611", "20010827", "20120112"],
        "list_status": ["L"] * 10,
    })

    # batch_query for daily prices
    dates = pd.date_range("20250101", "20250321", freq="B").strftime("%Y%m%d").tolist()

    def mock_batch_query(api_name, codes, start, end, **kwargs):
        rows = []
        np.random.seed(42)
        for code in codes:
            base_price = np.random.uniform(10, 100)
            for i, d in enumerate(dates):
                if d < start or d > end:
                    continue
                pct = np.random.normal(0.001, 0.02)
                base_price *= (1 + pct)
                rows.append({
                    "ts_code": code,
                    "trade_date": d,
                    "open": round(base_price * 0.99, 2),
                    "close": round(base_price, 2),
                    "high": round(base_price * 1.02, 2),
                    "low": round(base_price * 0.98, 2),
                    "vol": np.random.uniform(50000, 500000),
                    "amount": np.random.uniform(1e8, 1e10),
                    "pct_chg": round(pct * 100, 2),
                    "turnover_rate": np.random.uniform(0.5, 5.0),
                })
        return pd.DataFrame(rows)

    client.batch_query.side_effect = mock_batch_query

    # get_trade_calendar
    client.get_trade_calendar.return_value = dates

    # query for index_daily (CSI300)
    def mock_query(api_name, **kwargs):
        if api_name == "stock_basic":
            return client.query.return_value
        if api_name == "index_daily":
            base = 3800.0
            rows = []
            for i, d in enumerate(dates):
                base *= (1 + np.random.normal(0.0005, 0.01))
                rows.append({
                    "ts_code": "000300.SH",
                    "trade_date": d,
                    "close": round(base, 2),
                    "pct_chg": round(np.random.normal(0.05, 1.0), 2),
                })
            return pd.DataFrame(rows)
        if api_name == "daily_basic":
            rows = []
            for code in kwargs.get("ts_code", "").split(","):
                if not code:
                    continue
                rows.append({
                    "ts_code": code.strip(),
                    "trade_date": kwargs.get("trade_date", dates[-1]),
                    "total_mv": np.random.uniform(5e9, 5e11),
                    "circ_mv": np.random.uniform(3e9, 3e11),
                    "turnover_rate": np.random.uniform(0.5, 5.0),
                })
            return pd.DataFrame(rows)
        if api_name == "top_list":
            return pd.DataFrame()
        if api_name == "suspend_d":
            return pd.DataFrame()
        return pd.DataFrame()

    client.query.side_effect = mock_query

    def mock_query_by_date(api_name, trade_date=None, **kwargs):
        if api_name == "daily_basic":
            rows = []
            for code in ["000001.SZ", "000002.SZ", "000063.SZ", "600036.SH", "601318.SH",
                         "000858.SZ", "002415.SZ", "300750.SZ", "600519.SH", "601012.SH"]:
                rows.append({
                    "ts_code": code,
                    "trade_date": trade_date or dates[-1],
                    "total_mv": np.random.uniform(5e9, 5e11),
                    "circ_mv": np.random.uniform(3e9, 3e11),
                    "turnover_rate": np.random.uniform(0.5, 5.0),
                    "volume_ratio": np.random.uniform(0.5, 2.0),
                })
            return pd.DataFrame(rows)
        if api_name == "top_list":
            return pd.DataFrame()
        return pd.DataFrame()

    client.query_by_date.side_effect = mock_query_by_date

    return client


# ── MomentumConfig Tests ─────────────────────────────────────────────────────

class TestMomentumConfig:
    def test_create_default(self):
        from momentum.config import MomentumConfig
        cfg = MomentumConfig()
        assert cfg.top_n == 10
        assert cfg.momentum_type == "composite"
        assert len(cfg.lookback_days) == 3

    def test_to_dict_and_back(self, sample_config):
        d = sample_config.to_dict()
        assert isinstance(d, dict)
        assert d["top_n"] == 5

        from momentum.config import MomentumConfig
        cfg2 = MomentumConfig.from_dict(d)
        assert cfg2.top_n == 5
        assert cfg2.lookback_days == [20, 60]

    def test_save_and_load(self, sample_config, tmp_path):
        path = tmp_path / "test_config.json"
        sample_config.save(str(path))
        assert path.exists()

        from momentum.config import MomentumConfig
        loaded = MomentumConfig.load(str(path))
        assert loaded.top_n == sample_config.top_n

    def test_mutate(self, sample_config):
        mutated = sample_config.mutate({"top_n": 8, "stop_loss_pct": -0.1})
        assert mutated.top_n == 8
        assert mutated.stop_loss_pct == -0.1
        assert mutated.parent_version == sample_config.version
        # Original unchanged
        assert sample_config.top_n == 5


# ── Calculator Tests ─────────────────────────────────────────────────────────

class TestCalculator:
    def test_simple_momentum(self):
        from momentum.calculator import calc_simple_momentum
        prices = pd.DataFrame({
            "close": [100, 102, 105, 103, 110],
        })
        result = calc_simple_momentum(prices, lookback=4)
        # (110 - 100) / 100 = 0.10
        assert abs(result - 0.10) < 0.001

    def test_risk_adjusted_momentum(self):
        from momentum.calculator import calc_risk_adjusted_momentum
        prices = pd.DataFrame({
            "close": [100, 102, 105, 103, 110, 108, 112, 115, 113, 120],
        })
        result = calc_risk_adjusted_momentum(prices, lookback=9)
        assert isinstance(result, float)
        assert result > 0  # Positive return, should be positive


# ── Rebalancer Tests ─────────────────────────────────────────────────────────

class TestRebalancer:
    def test_compute_rebalance_empty_to_full(self, sample_config):
        from momentum.rebalancer import compute_rebalance

        target = pd.DataFrame({
            "ts_code": ["000001.SZ", "600036.SH", "600519.SH"],
            "momentum_score": [0.15, 0.12, 0.10],
            "rank": [1, 2, 3],
        })

        result = compute_rebalance(
            current_holdings=[],
            target_ranking=target,
            config=sample_config,
            total_capital=1_000_000,
        )

        assert len(result.buys) == 3
        assert len(result.sells) == 0

    def test_compute_rebalance_with_existing(self, sample_config):
        from momentum.rebalancer import compute_rebalance

        current = [
            {"ts_code": "000001.SZ", "weight": 0.2, "shares": 1000},
            {"ts_code": "000002.SZ", "weight": 0.2, "shares": 500},
        ]

        target = pd.DataFrame({
            "ts_code": ["000001.SZ", "600036.SH", "600519.SH"],
            "momentum_score": [0.15, 0.12, 0.10],
            "rank": [1, 2, 3],
        })

        result = compute_rebalance(
            current_holdings=current,
            target_ranking=target,
            config=sample_config,
            total_capital=1_000_000,
        )

        # 000002.SZ should be sold, 600036.SH and 600519.SH should be bought
        sell_codes = [s["ts_code"] for s in result.sells]
        buy_codes = [b["ts_code"] for b in result.buys]
        assert "000002.SZ" in sell_codes
        assert "600036.SH" in buy_codes


# ── Risk Control Tests ───────────────────────────────────────────────────────

class TestRiskControl:
    def test_apply_position_limits(self, sample_config):
        from momentum.risk_control import apply_position_limits

        weights = {
            "000001.SZ": 0.4,  # Over limit
            "600036.SH": 0.3,  # Over limit
            "600519.SH": 0.2,
            "000858.SZ": 0.1,
        }

        adjusted = apply_position_limits(weights, sample_config)
        for code, w in adjusted.items():
            assert w <= sample_config.max_single_weight + 0.001

    def test_apply_stop_loss(self, sample_config):
        from momentum.risk_control import apply_stop_loss

        holdings = [
            {"ts_code": "000001.SZ", "entry_price": 100.0, "peak_price": 110.0},
            {"ts_code": "600036.SH", "entry_price": 50.0, "peak_price": 55.0},
        ]

        prices = {
            "000001.SZ": 95.0,   # -13.6% from peak (below -8% stop)
            "600036.SH": 52.0,   # -5.5% from peak (above -8% stop)
        }

        triggered = apply_stop_loss(holdings, prices, sample_config)
        triggered_codes = [t["ts_code"] for t in triggered]
        assert "000001.SZ" in triggered_codes
        assert "600036.SH" not in triggered_codes


# ── DB Repository Tests ──────────────────────────────────────────────────────

class TestMomentumRepo:
    def test_ensure_tables(self, tmp_db):
        from db.momentum_repo import ensure_momentum_tables
        ensure_momentum_tables(tmp_db)

        tables = tmp_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'momentum_%'"
        ).fetchall()
        table_names = [t[0] for t in tables]
        assert "momentum_config_active" in table_names
        assert "momentum_holdings" in table_names
        assert "momentum_rebalance_log" in table_names

    def test_save_and_get_config(self, tmp_db, sample_config):
        from db.momentum_repo import ensure_momentum_tables, save_active_config, get_active_config
        ensure_momentum_tables(tmp_db)

        save_active_config(tmp_db, sample_config.to_dict(), sample_config.version)
        result = get_active_config(tmp_db)
        assert result is not None
        assert result["top_n"] == 5

    def test_save_and_get_holdings(self, tmp_db):
        from db.momentum_repo import ensure_momentum_tables, save_holdings, get_current_holdings
        ensure_momentum_tables(tmp_db)

        holdings = [
            {"ts_code": "000001.SZ", "name": "平安银行", "weight": 0.2,
             "shares": 1000, "entry_date": "2025-03-01", "entry_price": 12.5,
             "momentum_score": 0.15, "version": "v1"},
        ]
        save_holdings(tmp_db, holdings)

        result = get_current_holdings(tmp_db)
        assert len(result) >= 1

    def test_save_daily_snapshot(self, tmp_db):
        from db.momentum_repo import ensure_momentum_tables, save_daily_snapshot, get_daily_snapshots
        ensure_momentum_tables(tmp_db)

        save_daily_snapshot(tmp_db, "2025-03-21", 1_050_000, 100_000, 5, 3850.0)

        snapshots = get_daily_snapshots(tmp_db, "2025-03-01", "2025-03-31")
        assert len(snapshots) == 1


# ── Evolution Registry Tests ─────────────────────────────────────────────────

class TestEvolutionRegistry:
    def test_register_and_get(self, tmp_db, sample_config):
        from evolution.registry import EvolutionRegistry
        registry = EvolutionRegistry(tmp_db)

        version = registry.register_version(sample_config, parent_version="", mutation_reason="initial")
        assert version == sample_config.version

        retrieved = registry.get_version(version)
        assert retrieved is not None

    def test_promote_version(self, tmp_db, sample_config):
        from evolution.registry import EvolutionRegistry
        registry = EvolutionRegistry(tmp_db)

        registry.register_version(sample_config, parent_version="", mutation_reason="initial")
        registry.promote_version(sample_config.version, reason="test promotion")

        active = registry.get_active_version()
        assert active is not None


# ── Evolution Evaluator Tests ────────────────────────────────────────────────

class TestEvaluator:
    def test_performance_report_creation(self):
        from evolution.evaluator import PerformanceReport
        report = PerformanceReport(
            total_return=0.15,
            annualized_return=0.30,
            sharpe_ratio=1.5,
            max_drawdown=-0.12,
            win_rate=0.65,
            avg_turnover=0.25,
            total_trades=48,
            calmar_ratio=2.5,
            information_ratio=0.8,
            benchmark_return=0.10,
        )
        assert report.sharpe_ratio == 1.5
        assert report.max_drawdown == -0.12


# ── Run all tests ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
