#!/usr/bin/env python3
"""Full backtest parameter sweep for AVWAP profile strategy.

Sweeps portfolio construction, stop-loss, rebalance cycle, and market filter
variants using the best volume params (bv=1.3, pv=0.9) with daily_failure_exit
and ma60 filter as the common baseline.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from strategies.avwap_profile_strategy.run_backtest import (
    DATASETS,
    DEFAULT_TREND_INDEX_CODE,
    get_strategy_config,
    load_dataset_with_flow_signals,
    load_module,
    make_avwap_profile_backtest,
)

BACKTEST_SCRIPT = PROJECT_ROOT / "scripts" / "backtest_strategies.py"


# ---------------------------------------------------------------------------
# Shared base params (best known: bv=1.3, pv=0.9, daily_exit, ma60)
# ---------------------------------------------------------------------------

BASE_FACTORY = {
    "breakout_volume_mult": 1.3,
    "pullback_volume_frac": 0.9,
    "daily_failure_exit": True,
    "market_filter_mode": "ma60",
}

BASE_CONFIG = {
    "top_n": 6,
    "hold_buffer_ratio": 1.20,
    "max_single_weight": 0.18,
    "stop_loss_pct": -0.12,
    "min_amount_20d": 3e8,
    "min_price": 5.0,
    "min_list_days": 250,
}


def _make_experiments() -> list[dict]:
    """Build experiment list across multiple dimensions."""
    experiments = []

    # === Baseline (current best) ===
    experiments.append({
        "name": "baseline_250d",
        "holding_cycle": "weekly",
        "recent_days": 250,
        "factory": {**BASE_FACTORY},
        "config": {**BASE_CONFIG},
    })

    # === P0: 5-year full PIT verification ===
    experiments.append({
        "name": "full_5y_pit",
        "holding_cycle": "weekly",
        "recent_days": 0,  # 0 means use full dataset (start_offset=250)
        "factory": {**BASE_FACTORY},
        "config": {**BASE_CONFIG},
    })

    # === P1: Stop-loss variants ===
    for sl in [-0.15, -0.20, -0.99]:
        label = "off" if sl <= -0.99 else f"{sl:.0%}"
        experiments.append({
            "name": f"stop_loss_{label}",
            "holding_cycle": "weekly",
            "recent_days": 250,
            "factory": {**BASE_FACTORY},
            "config": {**BASE_CONFIG, "stop_loss_pct": sl},
        })

    # === P1: Rebalance cycle ===
    for cycle in ["biweekly", "monthly"]:
        experiments.append({
            "name": f"cycle_{cycle}",
            "holding_cycle": cycle,
            "recent_days": 250,
            "factory": {**BASE_FACTORY},
            "config": {**BASE_CONFIG},
        })

    # === P1: Market filter: half-position instead of empty ===
    experiments.append({
        "name": "market_half_position",
        "holding_cycle": "weekly",
        "recent_days": 250,
        "factory": {**BASE_FACTORY, "market_filter_half_position": True},
        "config": {**BASE_CONFIG},
    })

    # === P1: Market filter off (no filter at all) ===
    experiments.append({
        "name": "market_filter_off",
        "holding_cycle": "weekly",
        "recent_days": 250,
        "factory": {**BASE_FACTORY, "market_filter_mode": "off"},
        "config": {**BASE_CONFIG},
    })

    # === P1: Market filter ma120 ===
    experiments.append({
        "name": "market_filter_ma120",
        "holding_cycle": "weekly",
        "recent_days": 250,
        "factory": {**BASE_FACTORY, "market_filter_mode": "ma120"},
        "config": {**BASE_CONFIG},
    })

    # === P1: Portfolio construction variants ===
    for top_n in [8, 10]:
        experiments.append({
            "name": f"top_n={top_n}",
            "holding_cycle": "weekly",
            "recent_days": 250,
            "factory": {**BASE_FACTORY},
            "config": {**BASE_CONFIG, "top_n": top_n, "max_single_weight": round(0.95 / top_n, 2)},
        })

    for buf in [1.3, 1.5]:
        experiments.append({
            "name": f"buffer={buf}",
            "holding_cycle": "weekly",
            "recent_days": 250,
            "factory": {**BASE_FACTORY},
            "config": {**BASE_CONFIG, "hold_buffer_ratio": buf},
        })

    # === P2: Candidate pool wider + portfolio ===
    experiments.append({
        "name": "wider_pool_top8",
        "holding_cycle": "weekly",
        "recent_days": 250,
        "factory": {
            **BASE_FACTORY,
            "balance_periods_override": 6,
            "recent_breakout_lookback_override": 8,
            "range_pct_max": 0.40,
            "value_area_width_pct_max": 0.22,
            "poc_distance_max": 0.08,
            "angle_deg_min": 0.5,
            "angle_deg_max": 25.0,
        },
        "config": {**BASE_CONFIG, "top_n": 8, "max_single_weight": 0.12},
    })

    # === P2: Wider pool + biweekly + stop-loss off ===
    experiments.append({
        "name": "wider_biweekly_no_stop",
        "holding_cycle": "biweekly",
        "recent_days": 250,
        "factory": {
            **BASE_FACTORY,
            "balance_periods_override": 6,
            "recent_breakout_lookback_override": 8,
            "range_pct_max": 0.40,
            "value_area_width_pct_max": 0.22,
            "poc_distance_max": 0.08,
            "angle_deg_min": 0.5,
            "angle_deg_max": 25.0,
        },
        "config": {**BASE_CONFIG, "stop_loss_pct": -0.99, "top_n": 8, "max_single_weight": 0.12},
    })

    return experiments


def run_experiment(bt_module, daily, idx, basic, trade_dates, exp: dict) -> dict:
    """Run a single experiment and return the result dict."""
    holding_cycle = exp["holding_cycle"]
    recent_days = exp["recent_days"]
    factory_kwargs = exp["factory"]
    config_overrides = exp["config"]

    cfg = get_strategy_config(bt_module, holding_cycle, **config_overrides)
    strategy_cls = make_avwap_profile_backtest(bt_module, holding_cycle, **factory_kwargs)
    bt = strategy_cls(cfg, daily, idx, basic, trade_dates)

    if recent_days > 0:
        start_offset = max(0, len(trade_dates) - recent_days)
    else:
        start_offset = 250  # full dataset from day 251

    captured = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(captured):
        result = bt.run(start_offset=start_offset)
    result["elapsed_sec"] = round(time.time() - t0, 1)
    result["experiment"] = exp["name"]
    result["recent_days"] = recent_days if recent_days > 0 else len(trade_dates) - 250
    result["recent_start_date"] = trade_dates[start_offset]
    return result


def main():
    bt_module = load_module(BACKTEST_SCRIPT, "backtest_strategies_sweep")
    dataset = "csi1000_5y_pit"

    print("Loading data...")
    daily, idx, basic, trade_dates = load_dataset_with_flow_signals(
        DATASETS[dataset],
        bt_module,
        trend_index_code=DEFAULT_TREND_INDEX_CODE,
    )
    print(f"Data loaded. {len(trade_dates)} trade dates.")

    experiments = _make_experiments()
    results = []

    for i, exp in enumerate(experiments):
        name = exp["name"]
        t0 = time.time()
        result = run_experiment(bt_module, daily, idx, basic, trade_dates, exp)
        elapsed = round(time.time() - t0, 1)

        results.append(result)
        sharpe = result.get("sharpe", float("nan"))
        annual = result.get("annual_return_pct", float("nan"))
        maxdd = result.get("max_drawdown_pct", float("nan"))
        excess = result.get("excess_return_pct", float("nan"))
        cand_mean = result.get("candidate_count_mean", float("nan"))

        print(
            f"  [{i + 1}/{len(experiments)}] {name:30s}"
            f"  Sharpe={sharpe:+.2f}  Annual={annual:+.1f}%"
            f"  MaxDD={maxdd:.1f}%  Excess={excess:+.1f}%"
            f"  Cand={cand_mean:.1f}"
            f"  ({elapsed}s)"
        )

    # Sort by Sharpe descending
    results.sort(key=lambda r: r.get("sharpe", float("-inf")), reverse=True)

    out_path = PROJECT_ROOT / "backtest" / "strategy_avwap_profile_param_sweep_experiments.json"
    payload = {
        "dataset": dataset,
        "trend_index_code": DEFAULT_TREND_INDEX_CODE,
        "experiments": results,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWROTE {out_path}")

    # Summary table
    print("\n--- Results sorted by Sharpe ---")
    print(f"{'Name':35s} {'Sharpe':>7s} {'Annual%':>8s} {'MaxDD%':>7s} {'Excess%':>8s} {'Cand':>5s}")
    print("-" * 80)
    for r in results:
        print(
            f"{r.get('experiment', '?'):35s}"
            f" {r.get('sharpe', 0):+7.2f}"
            f" {r.get('annual_return_pct', 0):+8.1f}"
            f" {r.get('max_drawdown_pct', 0):7.1f}"
            f" {r.get('excess_return_pct', 0):+8.1f}"
            f" {r.get('candidate_count_mean', 0):5.1f}"
        )


if __name__ == "__main__":
    main()
