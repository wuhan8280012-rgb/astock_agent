#!/usr/bin/env python3
"""
Parameter sweep experiments for Ultra Rotation Strategy.

Explores the key parameter axes:
  1. Factor weight combinations
  2. Rebalance frequency (3d, 5d, 7d, 10d)
  3. Portfolio concentration (8, 10, 12, 15 stocks)
  4. Stop loss levels (-8%, -10%, -12%, -15%)
  5. Momentum lookback combinations

Usage:
  python strategies/ultra_rotation_strategy/run_param_sweep.py --dataset csi1000_5y_pit --use-pit-constituents
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from itertools import product
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from strategies.ultra_rotation_strategy.run_backtest import (
    BACKTEST_SCRIPT,
    DATASETS,
    DATASET_INDEX_CODES,
    DEFAULT_TREND_INDEX_CODE,
    UltraFactorWeights,
    get_strategy_config,
    load_dataset,
    make_ultra_rotation_backtest,
)

try:
    from data_pipeline.pit_constituents import load_or_fetch_pit_universe
except ImportError:
    load_or_fetch_pit_universe = None


def load_backtest_module():
    spec = importlib.util.spec_from_file_location("backtest_strategies", BACKTEST_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── Experiment definitions ──────────────────────────────────────

def get_factor_weight_experiments() -> list[dict]:
    """Different factor weight allocations."""
    return [
        {"label": "baseline", "weights": UltraFactorWeights()},
        {"label": "momentum_heavy", "weights": UltraFactorWeights(
            momentum_composite=0.35, momentum_accel=0.25, volume_surge=0.10,
            low_volatility=0.10, angle_trend=0.10, industry_rs=0.10)},
        {"label": "balanced_6", "weights": UltraFactorWeights(
            momentum_composite=0.20, momentum_accel=0.20, volume_surge=0.15,
            low_volatility=0.15, angle_trend=0.15, industry_rs=0.15)},
        {"label": "accel_volume_focus", "weights": UltraFactorWeights(
            momentum_composite=0.20, momentum_accel=0.25, volume_surge=0.20,
            low_volatility=0.10, angle_trend=0.15, industry_rs=0.10)},
        {"label": "quality_focus", "weights": UltraFactorWeights(
            momentum_composite=0.20, momentum_accel=0.15, volume_surge=0.10,
            low_volatility=0.25, angle_trend=0.20, industry_rs=0.10)},
    ]


def get_portfolio_param_experiments() -> list[dict]:
    """Rebalance interval × portfolio size grid."""
    rebalance_intervals = [3, 5, 7, 10]
    top_ns = [8, 10, 12, 15]

    experiments = []
    for reb, top_n in product(rebalance_intervals, top_ns):
        experiments.append({
            "label": f"reb{reb}d_top{top_n}",
            "rebalance_interval": reb,
            "top_n": top_n,
        })
    return experiments


def get_stop_loss_experiments() -> list[dict]:
    """Stop loss level sweep."""
    return [
        {"label": "stop_8pct", "stop_loss_pct": -0.08},
        {"label": "stop_10pct", "stop_loss_pct": -0.10},
        {"label": "stop_12pct", "stop_loss_pct": -0.12},
        {"label": "stop_15pct", "stop_loss_pct": -0.15},
        {"label": "stop_20pct", "stop_loss_pct": -0.20},
        {"label": "no_stop", "stop_loss_pct": -0.99},
    ]


# ── Runner ──────────────────────────────────────────────────────

def run_experiment(
    module,
    daily,
    idx,
    basic,
    trade_dates,
    label: str,
    cfg_overrides: dict,
    factor_weights: UltraFactorWeights,
    min_transition_coef: float = -0.1,
    pit_universe=None,
) -> dict:
    """Run a single experiment and return the result dict."""
    cfg = get_strategy_config(module, **cfg_overrides)
    backtest_cls = make_ultra_rotation_backtest(
        module,
        factor_weights=factor_weights,
        min_transition_coef=min_transition_coef,
    )
    bt = backtest_cls(cfg, daily, idx, basic, trade_dates)
    if pit_universe is not None:
        bt._universe_codes_by_date = pit_universe.constituents_by_date

    t0 = time.time()
    result = bt.run(start_offset=250)
    result["elapsed_sec"] = round(time.time() - t0, 1)
    result["experiment_label"] = label
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="csi1000_5y")
    parser.add_argument("--trend-index-code", type=str, default=DEFAULT_TREND_INDEX_CODE)
    parser.add_argument("--use-pit-constituents", action="store_true")
    parser.add_argument("--pit-index-code", type=str, default=None)
    parser.add_argument("--sweep", choices=["weights", "portfolio", "stoploss", "all"], default="all")
    args = parser.parse_args()

    module = load_backtest_module()
    data_path = DATASETS[args.dataset]
    daily, idx, basic, trade_dates = load_dataset(data_path, module, trend_index_code=args.trend_index_code)

    pit_universe = None
    if args.use_pit_constituents and load_or_fetch_pit_universe is not None:
        pit_index_code = (args.pit_index_code or DATASET_INDEX_CODES.get(args.dataset) or "").strip()
        if pit_index_code:
            pit_universe = load_or_fetch_pit_universe(
                index_code=pit_index_code,
                trade_dates=trade_dates,
            )

    all_results = {}

    # Factor weight experiments
    if args.sweep in ("weights", "all"):
        print("\n" + "=" * 60)
        print("SWEEP: Factor Weights")
        print("=" * 60)
        results = []
        for exp in get_factor_weight_experiments():
            print(f"\n  → {exp['label']}")
            r = run_experiment(
                module, daily, idx, basic, trade_dates,
                label=exp["label"],
                cfg_overrides={},
                factor_weights=exp["weights"],
                pit_universe=pit_universe,
            )
            results.append(r)
        all_results["factor_weights"] = results

    # Portfolio parameter experiments
    if args.sweep in ("portfolio", "all"):
        print("\n" + "=" * 60)
        print("SWEEP: Portfolio Parameters (rebalance × top_n)")
        print("=" * 60)
        results = []
        for exp in get_portfolio_param_experiments():
            print(f"\n  → {exp['label']}")
            r = run_experiment(
                module, daily, idx, basic, trade_dates,
                label=exp["label"],
                cfg_overrides={
                    "rebalance_interval": exp["rebalance_interval"],
                    "top_n": exp["top_n"],
                },
                factor_weights=UltraFactorWeights(),
                pit_universe=pit_universe,
            )
            results.append(r)
        all_results["portfolio_params"] = results

    # Stop loss experiments
    if args.sweep in ("stoploss", "all"):
        print("\n" + "=" * 60)
        print("SWEEP: Stop Loss Levels")
        print("=" * 60)
        results = []
        for exp in get_stop_loss_experiments():
            print(f"\n  → {exp['label']}")
            r = run_experiment(
                module, daily, idx, basic, trade_dates,
                label=exp["label"],
                cfg_overrides={"stop_loss_pct": exp["stop_loss_pct"]},
                factor_weights=UltraFactorWeights(),
                pit_universe=pit_universe,
            )
            results.append(r)
        all_results["stop_loss"] = results

    # Summary table
    print("\n\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"{'Experiment':<30} {'Annual%':>8} {'Sharpe':>7} {'MaxDD%':>8} {'Calmar':>7} {'Trades':>7} {'Time':>6}")
    print("-" * 100)
    for sweep_name, results in all_results.items():
        print(f"\n  [{sweep_name}]")
        for r in results:
            label = r.get("experiment_label", "?")
            print(
                f"  {label:<28} {r.get('annual_return_pct', 0):>7.1f}% "
                f"{r.get('sharpe', 0):>7.2f} {r.get('max_drawdown_pct', 0):>7.1f}% "
                f"{r.get('calmar', 0):>7.2f} {r.get('total_trades', 0):>7} "
                f"{r.get('elapsed_sec', 0):>5.0f}s"
            )

    # Save results
    suffix = f"_{args.dataset}"
    if args.use_pit_constituents:
        suffix += "_pit"
    out_path = PROJECT_ROOT / "backtest" / f"strategy_ultra_rotation_param_sweep{suffix}.json"
    out_path.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nWROTE {out_path}")


if __name__ == "__main__":
    main()
