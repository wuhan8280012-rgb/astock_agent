#!/usr/bin/env python3
"""
Ultra Rotation Strategy – extreme short-term A-share rotation backtest.

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

Usage:
  python strategies/ultra_rotation_strategy/run_backtest.py
  python strategies/ultra_rotation_strategy/run_backtest.py --dataset csi1000_5y_pit --use-pit-constituents
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


# ── helpers ──────────────────────────────────────────────────────

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


# ── data loading (reuse enrichment from f_strategy loader) ──────

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


# ── Ultra Rotation Backtest class ───────────────────────────────

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


# ── strategy config builder ─────────────────────────────────────

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


# ── main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ultra Rotation Strategy Backtest")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="csi1000_5y")
    parser.add_argument("--trend-index-code", type=str, default=DEFAULT_TREND_INDEX_CODE)
    parser.add_argument("--execution-mode", choices=["same_close", "next_open"], default="same_close")
    parser.add_argument("--use-pit-constituents", action="store_true")
    parser.add_argument("--pit-index-code", type=str, default=None)
    parser.add_argument("--refresh-pit-cache", action="store_true")
    parser.add_argument("--min-transition-coef", type=optional_float, default=-0.1)

    # Factor weights (override defaults)
    parser.add_argument("--w-momentum", type=float, default=0.25)
    parser.add_argument("--w-accel", type=float, default=0.20)
    parser.add_argument("--w-volume", type=float, default=0.15)
    parser.add_argument("--w-lowvol", type=float, default=0.15)
    parser.add_argument("--w-angle", type=float, default=0.15)
    parser.add_argument("--w-industry", type=float, default=0.10)

    # Portfolio parameters (override defaults)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--rebalance-interval", type=int, default=5)
    parser.add_argument("--stop-loss-pct", type=float, default=-0.10)
    parser.add_argument("--max-single-weight", type=float, default=0.12)

    args = parser.parse_args()

    module = load_backtest_module()
    data_path = DATASETS[args.dataset]
    daily, idx, basic, trade_dates = load_dataset(data_path, module, trend_index_code=args.trend_index_code)

    factor_weights = UltraFactorWeights(
        momentum_composite=args.w_momentum,
        momentum_accel=args.w_accel,
        volume_surge=args.w_volume,
        low_volatility=args.w_lowvol,
        angle_trend=args.w_angle,
        industry_rs=args.w_industry,
    )

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

    t0 = time.time()
    bt = backtest_cls(cfg, daily, idx, basic, trade_dates)

    # PIT constituents
    pit_summary: dict = {"enabled": False}
    if args.use_pit_constituents:
        pit_index_code = (args.pit_index_code or DATASET_INDEX_CODES.get(args.dataset) or "").strip()
        if not pit_index_code:
            raise ValueError(f"dataset={args.dataset} 未配置默认 PIT 指数代码，请显式传入 --pit-index-code")
        pit_universe = load_or_fetch_pit_universe(
            index_code=pit_index_code,
            trade_dates=trade_dates,
            refresh=args.refresh_pit_cache,
        )
        bt._universe_codes_by_date = pit_universe.constituents_by_date
        pit_summary = {
            "enabled": True,
            "index_code": pit_universe.index_code,
            "snapshot_count": pit_universe.snapshot_count,
            "unique_codes": pit_universe.unique_codes,
        }

    result = bt.run(start_offset=250)
    result["trend_filter_index_code"] = args.trend_index_code
    result["execution_mode"] = args.execution_mode
    result["pit_constituents"] = pit_summary
    result["factor_weights"] = asdict(factor_weights)
    result["strategy_params"] = {
        "top_n": args.top_n,
        "rebalance_interval": args.rebalance_interval,
        "stop_loss_pct": args.stop_loss_pct,
        "max_single_weight": args.max_single_weight,
        "min_transition_coef": args.min_transition_coef,
        "momentum_lookbacks": [5, 10, 20, 60],
        "momentum_lookback_weights": [0.15, 0.25, 0.35, 0.25],
    }
    result["elapsed_sec"] = round(time.time() - t0, 1)

    # Build output filename
    suffix = f"_{args.dataset}"
    if args.use_pit_constituents:
        suffix += "_pit"
    suffix += f"_top{args.top_n}_reb{args.rebalance_interval}d"
    if args.execution_mode != "same_close":
        suffix += f"_{args.execution_mode}"

    out_path = PROJECT_ROOT / "backtest" / f"strategy_ultra_rotation{suffix}.json"
    out_path.write_text(
        json.dumps({"data_file": str(data_path), "result": result}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nWROTE {out_path}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
