#!/usr/bin/env python3
"""
Ultra Rotation Strategy – comprehensive backtest script.

Target profile (T+1 framework, after costs):
  * Annual return  : 40-80 %
  * Sharpe ratio   : 1.5-2.5
  * Max drawdown   : 15-25 %

Key design choices vs the baseline F strategy:
  ┌────────────────────────────┬──────────────┬───────────────────┐
  │ Dimension                  │ F strategy   │ Ultra Rotation    │
  ├────────────────────────────┼──────────────┼───────────────────┤
  │ Factors                    │ 4            │ 6 (orthogonal)    │
  │ Rebalance cycle            │ 20 d         │ 5 d               │
  │ Portfolio concentration    │ 15 stocks    │ 10 stocks         │
  │ Max single weight          │ 8 %          │ 12 %              │
  │ Momentum signal            │ 60 d only    │ 5/10/20/60 d      │
  │ Volume signal              │ —            │ volume surge      │
  │ Industry signal            │ —            │ relative strength │
  │ Momentum acceleration      │ —            │ 5d vs 20d trend   │
  │ Stop loss                  │ -15 %        │ -10 %             │
  │ Trend filter               │ 200d MA      │ 200d MA (same)    │
  └────────────────────────────┴──────────────┴───────────────────┘

Modes:
  full            Full 5-year backtest (default)
  train_test      Train/test split with specified date boundaries
  walk_forward    Rolling walk-forward analysis
  recent          Recent N-day backtest
  compare_pit     Side-by-side static vs PIT A/B comparison

Usage:
  # Full 5-year backtest
  python strategies/ultra_rotation_strategy/run_backtest.py

  # PIT (survivorship-bias-free)
  python strategies/ultra_rotation_strategy/run_backtest.py \\
      --dataset csi1000_5y_pit --use-pit-constituents

  # Train/test split
  python strategies/ultra_rotation_strategy/run_backtest.py \\
      --dataset csi1000_5y_pit --use-pit-constituents \\
      --train-end 20240401 --test-start 20240401

  # Walk-forward analysis (3yr train / 1yr test)
  python strategies/ultra_rotation_strategy/run_backtest.py \\
      --dataset csi1000_5y_pit --use-pit-constituents \\
      --walk-forward

  # Recent 100-day
  python strategies/ultra_rotation_strategy/run_backtest.py \\
      --dataset csi1000_5y_pit --use-pit-constituents \\
      --recent-days 100

  # Static vs PIT A/B comparison
  python strategies/ultra_rotation_strategy/run_backtest.py --compare-pit
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.pit_constituents import load_or_fetch_pit_universe
from strategies.f_strategy.scoring import ScoreFilters
from strategies.ultra_rotation_strategy.scoring import (
    UltraFactorWeights,
    score_universe_ultra,
)

BACKTEST_SCRIPT = PROJECT_ROOT / "scripts" / "backtest_strategies.py"
ENV_PATH = PROJECT_ROOT / "config" / ".env"
DEFAULT_TREND_INDEX_CODE = "000001.SH"

DATASETS: dict[str, Path] = {
    "csi1000_5y": PROJECT_ROOT / "data_exports" / "tushare_20210329_20260327_csi1000_5y" / "csi1000_market_bundle_5y.csv",
    "csi1000_5y_pit": PROJECT_ROOT / "data_exports" / "tushare_20210329_20260327_csi1000_5y_pit" / "csi1000_market_bundle_5y_pit.csv",
}
DATASET_INDEX_CODES = {
    "csi1000_5y": "000852.SH",
    "csi1000_5y_pit": "000852.SH",
}


# ══════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════

def optional_float(value: str):
    if value is None:
        return None
    if str(value).strip().lower() in {"none", "null", "off"}:
        return None
    return float(value)


def load_backtest_module():
    spec = importlib.util.spec_from_file_location("backtest_strategies", BACKTEST_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_env_token() -> str | None:
    if os.environ.get("TUSHARE_TOKEN"):
        return os.environ["TUSHARE_TOKEN"].strip()
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            if line.startswith("TUSHARE_TOKEN="):
                token = line.split("=", 1)[1].strip()
                if token:
                    return token
    return None


def load_local_trend_index_df(index_path: Path | str, trade_dates: list[str]) -> pd.DataFrame:
    idx = pd.read_csv(index_path)
    idx["trade_date"] = idx["trade_date"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)
    close_col = "close" if "close" in idx.columns else "idx_close"
    pct_col = "pct_chg" if "pct_chg" in idx.columns else "idx_pct_chg"
    idx = idx[["trade_date", close_col, pct_col]].copy()
    idx.columns = ["trade_date", "idx_close", "idx_pct_chg"]
    for col in ["idx_close", "idx_pct_chg"]:
        idx[col] = pd.to_numeric(idx[col], errors="coerce")
    idx = idx[idx["trade_date"].isin(trade_dates)].sort_values("trade_date").reset_index(drop=True)
    return idx


def fetch_trend_index_df(trade_dates: list[str], trend_index_code: str) -> pd.DataFrame | None:
    token = load_env_token()
    if not token:
        return None
    try:
        import tushare as ts
    except Exception:
        return None
    try:
        ts.set_token(token)
        pro = ts.pro_api(token)
        idx = pro.index_daily(
            ts_code=trend_index_code,
            start_date=trade_dates[0],
            end_date=trade_dates[-1],
        )
    except Exception:
        return None
    if idx is None or idx.empty:
        return None
    idx = idx[["trade_date", "close", "pct_chg"]].copy()
    idx.columns = ["trade_date", "idx_close", "idx_pct_chg"]
    idx["trade_date"] = idx["trade_date"].astype(str)
    for c in ["idx_close", "idx_pct_chg"]:
        idx[c] = pd.to_numeric(idx[c], errors="coerce")
    idx = idx.sort_values("trade_date").reset_index(drop=True)
    idx = idx[idx["trade_date"].isin(trade_dates)].copy().reset_index(drop=True)
    return idx


# ══════════════════════════════════════════════════════════════════
#  Data loading (reuse enrichment from f_strategy loader)
# ══════════════════════════════════════════════════════════════════

def load_dataset(
    data_path: Path,
    module,
    trend_index_code: str = DEFAULT_TREND_INDEX_CODE,
):
    """Load and preprocess dataset using the base module's enrichment pipeline."""
    if hasattr(module, "load_data"):
        # Try using the module-level loader for the default static dataset
        if "csi1000_market_bundle_5y.csv" in str(data_path) and data_path.exists():
            return module.load_data(trend_index_code)

    # Fall back to the f_strategy's load_dataset utility if available
    f_strategy_path = PROJECT_ROOT / "strategies" / "f_strategy" / "run_backtest.py"
    f_spec = importlib.util.spec_from_file_location("f_strategy_bt", f_strategy_path)
    f_mod = importlib.util.module_from_spec(f_spec)
    f_spec.loader.exec_module(f_mod)
    return f_mod.load_dataset(data_path, module, trend_index_code=trend_index_code)


# ══════════════════════════════════════════════════════════════════
#  Ultra Rotation Backtest class
# ══════════════════════════════════════════════════════════════════

def make_ultra_rotation_backtest(
    module,
    *,
    momentum_lookbacks: list[int] | None = None,
    momentum_lookback_weights: list[float] | None = None,
    accel_short_days: int = 5,
    accel_long_days: int = 20,
    vol_surge_short_days: int = 5,
    vol_surge_long_days: int = 20,
    volatility_days: int = 20,
    angle_trend_days: int = 10,
    angle_trend_slope_weight: float = 0.6,
    angle_trend_persistence_weight: float = 0.4,
    factor_weights: UltraFactorWeights | None = None,
    min_transition_coef: float | None = -0.1,
):
    """Create a Backtest subclass with the ultra-rotation 6-factor scoring."""

    _momentum_lookbacks = momentum_lookbacks or [5, 10, 20, 60]
    _momentum_lookback_weights = momentum_lookback_weights or [0.15, 0.25, 0.35, 0.25]
    _factor_weights = factor_weights or UltraFactorWeights()
    _filters = ScoreFilters(min_transition_coef=min_transition_coef)

    class UltraRotationBacktest(module.Backtest):
        """Backtest with 6-factor ultra-rotation scoring."""

        def _score_universe(self, date: str) -> list[tuple[str, float]]:
            universe_map = getattr(self, "_universe_codes_by_date", None)
            allowed_codes = universe_map.get(date) if isinstance(universe_map, dict) else None

            return score_universe_ultra(
                self._stock_data,
                self._basic_map,
                date,
                momentum_lookbacks=_momentum_lookbacks,
                momentum_lookback_weights=_momentum_lookback_weights,
                accel_short_days=accel_short_days,
                accel_long_days=accel_long_days,
                vol_surge_short_days=vol_surge_short_days,
                vol_surge_long_days=vol_surge_long_days,
                volatility_days=volatility_days,
                angle_trend_days=angle_trend_days,
                angle_trend_slope_weight=angle_trend_slope_weight,
                angle_trend_persistence_weight=angle_trend_persistence_weight,
                factor_weights=_factor_weights,
                min_price=self.cfg.min_price,
                min_amount_20d=self.cfg.min_amount_20d,
                min_list_days=self.cfg.min_list_days,
                filters=_filters,
                allowed_codes=allowed_codes,
            )

    return UltraRotationBacktest


# ══════════════════════════════════════════════════════════════════
#  Strategy config builder
# ══════════════════════════════════════════════════════════════════

def get_strategy_config(module, **overrides):
    """Build the optimal BacktestConfig for Ultra Rotation."""
    defaults = {
        "name": "Ultra_Rotation_极限短期轮动",
        "momentum_days": [5, 10, 20, 60],
        "momentum_weights": [0.15, 0.25, 0.35, 0.25],
        # Portfolio construction – concentrated + frequent
        "top_n": 10,
        "hold_buffer_ratio": 1.3,
        "max_single_weight": 0.12,
        "rebalance_interval": 5,
        # Risk management
        "stop_loss_pct": -0.10,
        "use_halt": False,
        "use_trend_filter": True,
        "trend_ma_days": 200,
        "trend_reduce_pct": 0.5,
        # Transaction costs (realistic)
        "commission": 0.0003,
        "stamp_tax": 0.001,
        "slippage": 0.002,
        "execution_mode": "same_close",
        # Filters
        "min_amount_20d": 1.5e8,  # 1.5亿 – tighter liquidity for concentrated portfolio
        "min_price": 3.0,
        "min_list_days": 250,
        # Disable base-class factor flags (we use custom scoring)
        "use_volatility_factor": False,
        "use_size_factor": False,
        "use_angle_trend_factor": False,
    }
    defaults.update(overrides)
    return module.BacktestConfig(**defaults)


# ══════════════════════════════════════════════════════════════════
#  Train/test split
# ══════════════════════════════════════════════════════════════════

def resolve_start_offset(trade_dates: list[str], start_date: str | None, minimum_warmup: int = 250) -> int:
    """Return start_offset for a given start_date, or default warmup."""
    if start_date is None:
        return minimum_warmup
    if start_date not in trade_dates:
        raise ValueError(f"start_date not in trade calendar: {start_date}")
    return max(minimum_warmup, trade_dates.index(start_date))


def run_train_test_split(
    bt,
    trade_dates: list[str],
    train_end: str | None,
    test_start: str | None,
) -> dict:
    """Run separate train and test windows on the same backtest instance."""
    split: dict = {}
    if train_end is not None:
        if train_end not in trade_dates:
            raise ValueError(f"train_end not in trade calendar: {train_end}")
        split["train"] = bt.run(start_offset=250, end_date=train_end)
        split["train"]["window_start"] = trade_dates[250]
        split["train"]["window_end"] = train_end
    if test_start is not None:
        test_offset = resolve_start_offset(trade_dates, test_start)
        split["test"] = bt.run(start_offset=test_offset, end_date=trade_dates[-1])
        split["test"]["window_start"] = trade_dates[test_offset]
        split["test"]["window_end"] = trade_dates[-1]
    return split


# ══════════════════════════════════════════════════════════════════
#  Walk-forward analysis
# ══════════════════════════════════════════════════════════════════

def run_walk_forward(
    bt,
    trade_dates: list[str],
    train_days: int,
    test_days: int,
    step_days: int | None = None,
) -> dict:
    """Rolling walk-forward: train N days then test M days, advance by step."""
    step = step_days or test_days
    if train_days < 250:
        raise ValueError("walk-forward train window must be >= 250 trading days")

    segments: list[dict] = []
    test_start_idx = train_days
    while test_start_idx + test_days - 1 < len(trade_dates):
        train_end_idx = test_start_idx - 1
        test_end_idx = test_start_idx + test_days - 1
        train_end = trade_dates[train_end_idx]
        test_start = trade_dates[test_start_idx]
        test_end = trade_dates[test_end_idx]

        train_result = bt.run(start_offset=250, end_date=train_end)
        test_result = bt.run(start_offset=test_start_idx, end_date=test_end)
        segments.append({
            "train_window_start": trade_dates[250],
            "train_window_end": train_end,
            "test_window_start": test_start,
            "test_window_end": test_end,
            "train": train_result,
            "test": test_result,
        })
        test_start_idx += step

    test_sharpes = [s["test"].get("sharpe", 0.0) for s in segments if isinstance(s.get("test"), dict)]
    test_returns = [s["test"].get("total_return_pct", 0.0) for s in segments if isinstance(s.get("test"), dict)]
    test_drawdowns = [s["test"].get("max_drawdown_pct", 0.0) for s in segments if isinstance(s.get("test"), dict)]
    summary = {
        "segment_count": len(segments),
        "avg_test_sharpe": round(sum(test_sharpes) / len(test_sharpes), 2) if test_sharpes else None,
        "avg_test_total_return_pct": round(sum(test_returns) / len(test_returns), 2) if test_returns else None,
        "avg_test_max_drawdown_pct": round(sum(test_drawdowns) / len(test_drawdowns), 2) if test_drawdowns else None,
        "positive_test_segments": sum(1 for v in test_returns if v > 0),
        "win_rate_pct": round(sum(1 for v in test_returns if v > 0) / len(test_returns) * 100, 1) if test_returns else None,
    }
    return {"summary": summary, "segments": segments}


# ══════════════════════════════════════════════════════════════════
#  PIT helpers
# ══════════════════════════════════════════════════════════════════

def build_pit_summary(pit_universe, daily: pd.DataFrame) -> dict:
    """Build a detailed PIT summary dict with survivorship bias annotation."""
    bundle_unique_codes = int(daily["ts_code"].astype(str).nunique())
    summary = {
        "enabled": True,
        "index_code": pit_universe.index_code,
        "snapshot_count": pit_universe.snapshot_count,
        "constituent_count_max": pit_universe.constituent_count_max,
        "unique_codes": pit_universe.unique_codes,
        "earliest_snapshot_date": pit_universe.earliest_snapshot_date,
        "latest_snapshot_date": pit_universe.latest_snapshot_date,
        "fallback_to_earliest_days": pit_universe.fallback_to_earliest_days,
        "cache_path": pit_universe.cache_path,
        "bundle_unique_codes": bundle_unique_codes,
    }
    if bundle_unique_codes >= pit_universe.unique_codes:
        summary["survivorship_bias_note"] = (
            "PIT constituent filtering is active. The bundle covers all historical "
            "constituent codes from index_weight snapshots. Residual gaps may still exist "
            "if the upstream API is missing certain delisted securities."
        )
    else:
        summary["survivorship_bias_note"] = (
            "PIT constituent filtering is active, but the bundle does NOT fully cover "
            "all historical constituent codes. Delisted/removed stocks absent from "
            "the bundle mean survivorship bias is only partially mitigated."
        )
    return summary


# ══════════════════════════════════════════════════════════════════
#  Result annotation helper
# ══════════════════════════════════════════════════════════════════

def annotate_result(
    result: dict,
    *,
    args,
    factor_weights: UltraFactorWeights,
    pit_summary: dict,
    elapsed: float,
) -> dict:
    """Attach metadata to a backtest result dict."""
    result["trend_filter_index_code"] = args.trend_index_code
    result["execution_mode"] = args.execution_mode
    result["pit_constituents"] = pit_summary
    result["factor_weights"] = asdict(factor_weights)
    result["strategy_params"] = {
        "top_n": args.top_n,
        "hold_buffer_ratio": 1.3,
        "rebalance_interval": args.rebalance_interval,
        "stop_loss_pct": args.stop_loss_pct,
        "max_single_weight": args.max_single_weight,
        "min_transition_coef": args.min_transition_coef,
        "momentum_lookbacks": [5, 10, 20, 60],
        "momentum_lookback_weights": [0.15, 0.25, 0.35, 0.25],
        "use_trend_filter": True,
        "trend_ma_days": 200,
        "trend_reduce_pct": 0.5,
        "min_amount_20d": 1.5e8,
        "commission": 0.0003,
        "stamp_tax": 0.001,
        "slippage": 0.002,
    }
    result["elapsed_sec"] = round(elapsed, 1)
    return result


# ══════════════════════════════════════════════════════════════════
#  Output filename builder
# ══════════════════════════════════════════════════════════════════

def build_output_suffix(args, mode_tag: str = "") -> str:
    """Build a descriptive filename suffix."""
    suffix = f"_{args.dataset}"
    if args.use_pit_constituents:
        suffix += "_pit"
    suffix += f"_top{args.top_n}_reb{args.rebalance_interval}d"
    if args.execution_mode != "same_close":
        suffix += f"_{args.execution_mode}"
    if mode_tag:
        suffix += f"_{mode_tag}"
    return suffix


# ══════════════════════════════════════════════════════════════════
#  Comparison printer
# ══════════════════════════════════════════════════════════════════

def print_comparison_table(results: list[dict], title: str = "Comparison"):
    """Print a formatted comparison table for multiple results."""
    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")
    print(
        f"  {'Label':<35} {'Total%':>8} {'Annual%':>8} {'Sharpe':>7} "
        f"{'Calmar':>7} {'MaxDD%':>8} {'Trades':>7} {'Days':>6}"
    )
    print(f"  {'-' * 93}")
    for r in results:
        label = r.get("_label", r.get("name", "?"))
        print(
            f"  {label:<35} {r.get('total_return_pct', 0):>7.1f}% "
            f"{r.get('annual_return_pct', 0):>7.1f}% {r.get('sharpe', 0):>7.2f} "
            f"{r.get('calmar', 0):>7.2f} {r.get('max_drawdown_pct', 0):>7.1f}% "
            f"{r.get('total_trades', 0):>7} {r.get('trading_days', 0):>6}"
        )
    print()


# ══════════════════════════════════════════════════════════════════
#  Core backtest setup (shared by all modes)
# ══════════════════════════════════════════════════════════════════

def setup_backtest(args, module, daily, idx, basic, trade_dates, factor_weights):
    """Create config, backtest class, instance, and attach PIT universe if needed."""
    cfg = get_strategy_config(
        module,
        execution_mode=args.execution_mode,
        top_n=args.top_n,
        rebalance_interval=args.rebalance_interval,
        stop_loss_pct=args.stop_loss_pct,
        max_single_weight=args.max_single_weight,
    )
    backtest_cls = make_ultra_rotation_backtest(
        module,
        factor_weights=factor_weights,
        min_transition_coef=args.min_transition_coef,
    )
    bt = backtest_cls(cfg, daily, idx, basic, trade_dates)

    pit_summary: dict = {"enabled": False}
    if args.use_pit_constituents:
        pit_index_code = (args.pit_index_code or DATASET_INDEX_CODES.get(args.dataset) or "").strip()
        if not pit_index_code:
            raise ValueError(
                f"dataset={args.dataset} has no default PIT index code; "
                "pass --pit-index-code explicitly"
            )
        pit_universe = load_or_fetch_pit_universe(
            index_code=pit_index_code,
            trade_dates=trade_dates,
            refresh=args.refresh_pit_cache,
        )
        bt._universe_codes_by_date = pit_universe.constituents_by_date
        pit_summary = build_pit_summary(pit_universe, daily)

    return bt, pit_summary


# ══════════════════════════════════════════════════════════════════
#  Mode: full (default)
# ══════════════════════════════════════════════════════════════════

def run_mode_full(args, module, daily, idx, basic, trade_dates, factor_weights):
    """Full 5-year backtest."""
    t0 = time.time()
    bt, pit_summary = setup_backtest(args, module, daily, idx, basic, trade_dates, factor_weights)
    result = bt.run(start_offset=250)
    elapsed = time.time() - t0

    annotate_result(result, args=args, factor_weights=factor_weights,
                    pit_summary=pit_summary, elapsed=elapsed)

    output = {"mode": "full", "full": result}

    suffix = build_output_suffix(args)
    out_path = PROJECT_ROOT / "backtest" / f"strategy_ultra_rotation{suffix}.json"
    out_path.write_text(json.dumps({"data_file": str(DATASETS[args.dataset]), "result": output},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"\nWROTE {out_path}")
    return output


# ══════════════════════════════════════════════════════════════════
#  Mode: train_test
# ══════════════════════════════════════════════════════════════════

def run_mode_train_test(args, module, daily, idx, basic, trade_dates, factor_weights):
    """Train/test split backtest."""
    t0 = time.time()
    bt, pit_summary = setup_backtest(args, module, daily, idx, basic, trade_dates, factor_weights)

    full_result = bt.run(start_offset=250)
    split_result = run_train_test_split(bt, trade_dates,
                                         train_end=args.train_end, test_start=args.test_start)
    elapsed = time.time() - t0

    annotate_result(full_result, args=args, factor_weights=factor_weights,
                    pit_summary=pit_summary, elapsed=elapsed)

    output = {"mode": "train_test", "full": full_result, "train_test": split_result}

    # Print comparison table
    rows = [full_result.copy()]
    rows[0]["_label"] = "Full period"
    if "train" in split_result:
        r = split_result["train"].copy()
        r["_label"] = f"Train ({r.get('window_start', '?')} ~ {r.get('window_end', '?')})"
        rows.append(r)
    if "test" in split_result:
        r = split_result["test"].copy()
        r["_label"] = f"Test  ({r.get('window_start', '?')} ~ {r.get('window_end', '?')})"
        rows.append(r)
    print_comparison_table(rows, title="Train / Test Split")

    suffix = build_output_suffix(args, mode_tag="train_test")
    out_path = PROJECT_ROOT / "backtest" / f"strategy_ultra_rotation{suffix}.json"
    out_path.write_text(json.dumps({"data_file": str(DATASETS[args.dataset]), "result": output},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WROTE {out_path}")
    return output


# ══════════════════════════════════════════════════════════════════
#  Mode: walk_forward
# ══════════════════════════════════════════════════════════════════

def run_mode_walk_forward(args, module, daily, idx, basic, trade_dates, factor_weights):
    """Walk-forward analysis."""
    t0 = time.time()
    bt, pit_summary = setup_backtest(args, module, daily, idx, basic, trade_dates, factor_weights)

    full_result = bt.run(start_offset=250)
    wf_result = run_walk_forward(
        bt, trade_dates,
        train_days=args.wf_train_days, test_days=args.wf_test_days,
        step_days=args.wf_step_days,
    )
    elapsed = time.time() - t0

    annotate_result(full_result, args=args, factor_weights=factor_weights,
                    pit_summary=pit_summary, elapsed=elapsed)

    output = {"mode": "walk_forward", "full": full_result, "walk_forward": wf_result}

    # Print walk-forward summary
    summary = wf_result.get("summary", {})
    print(f"\n{'=' * 70}")
    print("  Walk-Forward Summary")
    print(f"{'=' * 70}")
    print(f"  Segments            : {summary.get('segment_count', 0)}")
    print(f"  Avg test Sharpe     : {summary.get('avg_test_sharpe', 'N/A')}")
    print(f"  Avg test return     : {summary.get('avg_test_total_return_pct', 'N/A')}%")
    print(f"  Avg test max DD     : {summary.get('avg_test_max_drawdown_pct', 'N/A')}%")
    print(f"  Positive segments   : {summary.get('positive_test_segments', 0)}")
    print(f"  Win rate            : {summary.get('win_rate_pct', 'N/A')}%")

    # Per-segment table
    segments = wf_result.get("segments", [])
    if segments:
        rows = []
        for i, seg in enumerate(segments):
            r = seg.get("test", {}).copy()
            r["_label"] = f"Seg {i+1}: {seg['test_window_start']}~{seg['test_window_end']}"
            rows.append(r)
        print_comparison_table(rows, title="Walk-Forward Test Segments")

    suffix = build_output_suffix(args, mode_tag="walk_forward")
    out_path = PROJECT_ROOT / "backtest" / f"strategy_ultra_rotation{suffix}.json"
    out_path.write_text(json.dumps({"data_file": str(DATASETS[args.dataset]), "result": output},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WROTE {out_path}")
    return output


# ══════════════════════════════════════════════════════════════════
#  Mode: recent N-day
# ══════════════════════════════════════════════════════════════════

def run_mode_recent(args, module, daily, idx, basic, trade_dates, factor_weights):
    """Recent N-day backtest (from trade_dates[-N-250] with 250-offset warmup)."""
    n = args.recent_days
    if n <= 0 or n > len(trade_dates) - 250:
        raise ValueError(f"--recent-days must be in [1, {len(trade_dates) - 250}]")

    # start_offset so that the active trading window is exactly the last N days
    start_offset = len(trade_dates) - n

    t0 = time.time()
    bt, pit_summary = setup_backtest(args, module, daily, idx, basic, trade_dates, factor_weights)
    result = bt.run(start_offset=start_offset)
    elapsed = time.time() - t0

    annotate_result(result, args=args, factor_weights=factor_weights,
                    pit_summary=pit_summary, elapsed=elapsed)
    result["recent_days"] = n
    result["window_start"] = trade_dates[start_offset]
    result["window_end"] = trade_dates[-1]

    output = {"mode": f"recent_{n}d", "result": result}

    suffix = build_output_suffix(args, mode_tag=f"recent_{n}d")
    out_path = PROJECT_ROOT / "backtest" / f"strategy_ultra_rotation{suffix}.json"
    out_path.write_text(json.dumps({"data_file": str(DATASETS[args.dataset]), "result": output},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"\nWROTE {out_path}")
    return output


# ══════════════════════════════════════════════════════════════════
#  Mode: compare_pit (static vs PIT A/B)
# ══════════════════════════════════════════════════════════════════

def run_mode_compare_pit(args, module, factor_weights):
    """Run same config on both static and PIT datasets, print side-by-side."""
    pit_dataset = "csi1000_5y_pit"
    pit_data_path = DATASETS[pit_dataset]
    if not pit_data_path.exists() and not Path(str(pit_data_path) + ".gz").exists():
        raise FileNotFoundError(f"PIT data file not found: {pit_data_path}")

    results = []

    # ── A: Static sample ──
    static_data_path = DATASETS.get("csi1000_5y")
    if static_data_path and static_data_path.exists():
        print("\n" + "=" * 60)
        print("  [A] Static sample (potential survivorship bias)")
        print("=" * 60)
        daily_s, idx_s, basic_s, td_s = load_dataset(static_data_path, module,
                                                       trend_index_code=args.trend_index_code)
        cfg_s = get_strategy_config(module, execution_mode=args.execution_mode,
                                    top_n=args.top_n, rebalance_interval=args.rebalance_interval,
                                    stop_loss_pct=args.stop_loss_pct, max_single_weight=args.max_single_weight)
        cls_s = make_ultra_rotation_backtest(module, factor_weights=factor_weights,
                                             min_transition_coef=args.min_transition_coef)
        bt_s = cls_s(cfg_s, daily_s, idx_s, basic_s, td_s)
        r_s = bt_s.run(start_offset=250)
        r_s["_label"] = "Static (survivorship bias possible)"
        r_s["sample"] = "static"
        results.append(r_s)
    else:
        print("[SKIP] Static dataset not available")

    # ── B: PIT sample ──
    print("\n" + "=" * 60)
    print("  [B] PIT sample (survivorship-bias-free)")
    print("=" * 60)
    daily_p, idx_p, basic_p, td_p = load_dataset(pit_data_path, module,
                                                   trend_index_code=args.trend_index_code)
    cfg_p = get_strategy_config(module, execution_mode=args.execution_mode,
                                top_n=args.top_n, rebalance_interval=args.rebalance_interval,
                                stop_loss_pct=args.stop_loss_pct, max_single_weight=args.max_single_weight)
    cls_p = make_ultra_rotation_backtest(module, factor_weights=factor_weights,
                                         min_transition_coef=args.min_transition_coef)
    bt_p = cls_p(cfg_p, daily_p, idx_p, basic_p, td_p)

    pit_index_code = (args.pit_index_code or DATASET_INDEX_CODES.get(pit_dataset) or "").strip()
    if pit_index_code:
        pit_universe = load_or_fetch_pit_universe(
            index_code=pit_index_code, trade_dates=td_p,
            refresh=args.refresh_pit_cache,
        )
        bt_p._universe_codes_by_date = pit_universe.constituents_by_date

    r_p = bt_p.run(start_offset=250)
    r_p["_label"] = "PIT (survivorship-bias-free)"
    r_p["sample"] = "pit"
    results.append(r_p)

    # ── Print comparison ──
    print_comparison_table(results, title="Static vs PIT A/B Comparison")

    # ── Survivorship bias magnitude ──
    if len(results) == 2:
        delta_annual = results[0].get("annual_return_pct", 0) - results[1].get("annual_return_pct", 0)
        delta_sharpe = results[0].get("sharpe", 0) - results[1].get("sharpe", 0)
        print(f"  Survivorship bias impact:")
        print(f"    Annual return delta : {delta_annual:+.2f}%")
        print(f"    Sharpe delta        : {delta_sharpe:+.2f}")
        print()

    output = {
        "mode": "compare_pit",
        "static": results[0] if len(results) >= 1 else None,
        "pit": results[-1],
    }

    suffix = build_output_suffix(args, mode_tag="compare_pit")
    out_path = PROJECT_ROOT / "backtest" / f"strategy_ultra_rotation{suffix}.json"
    out_path.write_text(json.dumps({"data_file": str(pit_data_path), "result": output},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WROTE {out_path}")
    return output


# ══════════════════════════════════════════════════════════════════
#  CLI entry point
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Ultra Rotation Strategy – Comprehensive Backtest Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Dataset & index
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="csi1000_5y",
                        help="Data bundle to use (default: csi1000_5y)")
    parser.add_argument("--trend-index-code", type=str, default=DEFAULT_TREND_INDEX_CODE,
                        help="Trend filter index code (default: 000001.SH)")
    parser.add_argument("--execution-mode", choices=["same_close", "next_open"], default="same_close",
                        help="Trade execution mode (default: same_close)")

    # PIT
    parser.add_argument("--use-pit-constituents", action="store_true",
                        help="Enable PIT constituent filtering to remove survivorship bias")
    parser.add_argument("--pit-index-code", type=str, default=None,
                        help="Override PIT index code")
    parser.add_argument("--refresh-pit-cache", action="store_true",
                        help="Force refresh PIT constituent cache")
    parser.add_argument("--min-transition-coef", type=optional_float, default=-0.1,
                        help="Transition coefficient hard filter threshold (default: -0.1)")

    # Factor weights
    parser.add_argument("--w-momentum", type=float, default=0.25, help="Momentum composite weight")
    parser.add_argument("--w-accel", type=float, default=0.20, help="Momentum acceleration weight")
    parser.add_argument("--w-volume", type=float, default=0.15, help="Volume surge weight")
    parser.add_argument("--w-lowvol", type=float, default=0.15, help="Low volatility weight")
    parser.add_argument("--w-angle", type=float, default=0.15, help="Angle trend weight")
    parser.add_argument("--w-industry", type=float, default=0.10, help="Industry RS weight")

    # Portfolio parameters
    parser.add_argument("--top-n", type=int, default=10, help="Target number of positions")
    parser.add_argument("--rebalance-interval", type=int, default=5,
                        help="Rebalance every N trading days")
    parser.add_argument("--stop-loss-pct", type=float, default=-0.10,
                        help="Per-stock stop loss (default: -0.10)")
    parser.add_argument("--max-single-weight", type=float, default=0.12,
                        help="Max single position weight (default: 0.12)")

    # Backtest modes
    parser.add_argument("--train-end", type=str, default=None,
                        help="Train period end date (YYYYMMDD). Enables train/test split mode.")
    parser.add_argument("--test-start", type=str, default=None,
                        help="Test period start date (YYYYMMDD). Enables train/test split mode.")
    parser.add_argument("--walk-forward", action="store_true",
                        help="Enable walk-forward analysis")
    parser.add_argument("--wf-train-days", type=int, default=252 * 3,
                        help="Walk-forward train window in trading days (default: 756)")
    parser.add_argument("--wf-test-days", type=int, default=252,
                        help="Walk-forward test window in trading days (default: 252)")
    parser.add_argument("--wf-step-days", type=int, default=None,
                        help="Walk-forward step size (default: same as test window)")
    parser.add_argument("--recent-days", type=int, default=None,
                        help="Run backtest on the most recent N trading days only")
    parser.add_argument("--compare-pit", action="store_true",
                        help="Run static vs PIT A/B comparison")

    args = parser.parse_args()

    # Build factor weights from CLI
    factor_weights = UltraFactorWeights(
        momentum_composite=args.w_momentum,
        momentum_accel=args.w_accel,
        volume_surge=args.w_volume,
        low_volatility=args.w_lowvol,
        angle_trend=args.w_angle,
        industry_rs=args.w_industry,
    )

    module = load_backtest_module()

    # ── Dispatch to the appropriate mode ──

    if args.compare_pit:
        run_mode_compare_pit(args, module, factor_weights)
        return

    # Load data once for all other modes
    data_path = DATASETS[args.dataset]
    daily, idx, basic, trade_dates = load_dataset(data_path, module,
                                                   trend_index_code=args.trend_index_code)

    if args.recent_days is not None:
        run_mode_recent(args, module, daily, idx, basic, trade_dates, factor_weights)
    elif args.walk_forward:
        run_mode_walk_forward(args, module, daily, idx, basic, trade_dates, factor_weights)
    elif args.train_end or args.test_start:
        run_mode_train_test(args, module, daily, idx, basic, trade_dates, factor_weights)
    else:
        run_mode_full(args, module, daily, idx, basic, trade_dates, factor_weights)


if __name__ == "__main__":
    main()
