#!/usr/bin/env python3
"""Lightweight candidate pool size sweep for AVWAP profile strategy.

Only runs _score_universe() on each rebalance date and counts candidates —
no full portfolio simulation — so it's much faster than a full backtest.
Use this to find parameter combos that yield median ≥ 6 candidates.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from strategies.avwap_profile_strategy.run_backtest import (  # noqa: E402
    DATASETS,
    DEFAULT_TREND_INDEX_CODE,
    get_strategy_config,
    load_dataset_with_flow_signals,
    load_module,
    make_avwap_profile_backtest,
)

BACKTEST_SCRIPT = PROJECT_ROOT / "scripts" / "backtest_strategies.py"


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

# Baseline: current best params
BASELINE = {
    "breakout_volume_mult": 1.3,
    "pullback_volume_frac": 0.9,
    "balance_periods_override": None,
    "recent_breakout_lookback_override": None,
    "range_pct_max": 0.32,
    "value_area_width_pct_max": 0.18,
    "poc_distance_max": 0.05,
    "angle_deg_min": 1.0,
    "angle_deg_max": 20.0,
    "min_price": 5.0,
    "min_list_days": 250,
    "min_amount_20d": 3e8,
}


def _make_experiments() -> list[dict]:
    """Build experiment list: change one dimension at a time from baseline."""
    experiments = [{"name": "baseline", **BASELINE}]

    # --- Dimension 1: balance_periods ---
    for bp in [4, 5, 6]:
        exp = {**BASELINE, "name": f"balance_periods={bp}", "balance_periods_override": bp}
        experiments.append(exp)

    # --- Dimension 2: recent_breakout_lookback ---
    for lb in [6, 8, 12]:
        exp = {**BASELINE, "name": f"lookback={lb}", "recent_breakout_lookback_override": lb}
        experiments.append(exp)

    # --- Dimension 3: range_pct_max ---
    for rp in [0.40, 0.45]:
        exp = {**BASELINE, "name": f"range_pct={rp}", "range_pct_max": rp}
        experiments.append(exp)

    # --- Dimension 4: value_area_width_pct_max ---
    for vw in [0.22, 0.25]:
        exp = {**BASELINE, "name": f"va_width={vw}", "value_area_width_pct_max": vw}
        experiments.append(exp)

    # --- Dimension 5: poc_distance_max ---
    for pd_val in [0.08, 0.10]:
        exp = {**BASELINE, "name": f"poc_dist={pd_val}", "poc_distance_max": pd_val}
        experiments.append(exp)

    # --- Dimension 6: angle_deg bounds ---
    for lo, hi in [(0.5, 25.0), (0.0, 30.0)]:
        exp = {**BASELINE, "name": f"angle={lo}-{hi}", "angle_deg_min": lo, "angle_deg_max": hi}
        experiments.append(exp)

    # --- Dimension 7: min_list_days ---
    for mld in [180]:
        exp = {**BASELINE, "name": f"min_list={mld}d", "min_list_days": mld}
        experiments.append(exp)

    # --- Dimension 8: min_price ---
    for mp in [3.0]:
        exp = {**BASELINE, "name": f"min_price={mp}", "min_price": mp}
        experiments.append(exp)

    # --- Dimension 9: min_amount_20d ---
    for ma in [2e8, 1.5e8]:
        amount_label = f"{ma / 1e8:.1f}亿"
        exp = {**BASELINE, "name": f"min_amt={amount_label}", "min_amount_20d": ma}
        experiments.append(exp)

    # --- Combined: best of each dimension (wide open) ---
    experiments.append({
        "name": "wide_open",
        "breakout_volume_mult": 1.3,
        "pullback_volume_frac": 0.9,
        "balance_periods_override": 5,
        "recent_breakout_lookback_override": 8,
        "range_pct_max": 0.40,
        "value_area_width_pct_max": 0.22,
        "poc_distance_max": 0.08,
        "angle_deg_min": 0.5,
        "angle_deg_max": 25.0,
        "min_price": 3.0,
        "min_list_days": 180,
        "min_amount_20d": 2e8,
    })

    # --- Combined: moderately wider ---
    experiments.append({
        "name": "moderate_wider",
        "breakout_volume_mult": 1.3,
        "pullback_volume_frac": 0.9,
        "balance_periods_override": 6,
        "recent_breakout_lookback_override": 6,
        "range_pct_max": 0.40,
        "value_area_width_pct_max": 0.22,
        "poc_distance_max": 0.08,
        "angle_deg_min": 0.5,
        "angle_deg_max": 25.0,
        "min_price": 5.0,
        "min_list_days": 250,
        "min_amount_20d": 3e8,
    })

    return experiments


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def count_candidates_only(
    bt_module,
    daily,
    idx,
    basic,
    trade_dates,
    holding_cycle: str,
    start_offset: int,
    factory_kwargs: dict,
    config_overrides: dict,
) -> dict:
    """Create strategy instance and count candidates on each rebalance date."""
    cfg = get_strategy_config(bt_module, holding_cycle, **config_overrides)
    strategy_cls = make_avwap_profile_backtest(
        bt_module,
        holding_cycle,
        daily_failure_exit=True,
        market_filter_mode="ma60",
        **factory_kwargs,
    )
    bt = strategy_cls(cfg, daily, idx, basic, trade_dates)

    rebalance_dates = sorted(bt._rebalance_dates)
    start_date = trade_dates[start_offset]
    dates_in_range = [d for d in rebalance_dates if d >= start_date]

    counts = []
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        for date in dates_in_range:
            scores = bt._score_universe(date)
            counts.append(len(scores))

    if not counts:
        return {
            "rebalance_dates_scanned": 0,
            "candidate_count_mean": 0.0,
            "candidate_count_median": 0.0,
            "candidate_count_min": 0,
            "candidate_count_max": 0,
            "candidate_count_p25": 0.0,
            "candidate_count_p75": 0.0,
            "enough_candidates_ratio": 0.0,
        }

    s = pd.Series(counts, dtype=float)
    return {
        "rebalance_dates_scanned": len(dates_in_range),
        "candidate_count_mean": round(float(s.mean()), 2),
        "candidate_count_median": round(float(s.median()), 2),
        "candidate_count_min": int(s.min()),
        "candidate_count_max": int(s.max()),
        "candidate_count_p25": round(float(s.quantile(0.25)), 2),
        "candidate_count_p75": round(float(s.quantile(0.75)), 2),
        "enough_candidates_ratio": round(float((s >= cfg.top_n).mean()), 4),
    }


def main():
    bt_module = load_module(BACKTEST_SCRIPT, "backtest_strategies_pool_exp")
    dataset = "csi1000_5y_pit"
    holding_cycle = "weekly"
    recent_days = 250

    print("Loading data...")
    daily, idx, basic, trade_dates = load_dataset_with_flow_signals(
        DATASETS[dataset],
        bt_module,
        trend_index_code=DEFAULT_TREND_INDEX_CODE,
    )
    start_offset = max(0, len(trade_dates) - recent_days)
    print(f"Data loaded. {len(trade_dates)} trade dates, start_offset={start_offset}")

    experiments = _make_experiments()
    results = []

    for i, exp in enumerate(experiments):
        name = exp["name"]
        # Separate factory kwargs from config overrides
        factory_keys = {
            "breakout_volume_mult",
            "pullback_volume_frac",
            "balance_periods_override",
            "recent_breakout_lookback_override",
            "range_pct_max",
            "value_area_width_pct_max",
            "poc_distance_max",
            "angle_deg_min",
            "angle_deg_max",
        }
        config_keys = {"min_price", "min_list_days", "min_amount_20d"}

        factory_kwargs = {k: v for k, v in exp.items() if k in factory_keys}
        config_overrides = {k: v for k, v in exp.items() if k in config_keys}

        t0 = time.time()
        stats = count_candidates_only(
            bt_module,
            daily,
            idx,
            basic,
            trade_dates,
            holding_cycle,
            start_offset,
            factory_kwargs,
            config_overrides,
        )
        elapsed = round(time.time() - t0, 1)

        result = {"name": name, **{k: v for k, v in exp.items() if k != "name"}, **stats, "elapsed_sec": elapsed}
        results.append(result)

        marker = "✅" if stats["candidate_count_median"] >= 6 else "❌"
        print(
            f"  [{i + 1}/{len(experiments)}] {marker} {name:30s}"
            f"  median={stats['candidate_count_median']:5.1f}"
            f"  mean={stats['candidate_count_mean']:5.1f}"
            f"  p75={stats['candidate_count_p75']:5.1f}"
            f"  enough={stats['enough_candidates_ratio']:.1%}"
            f"  ({elapsed}s)"
        )

    # Sort by candidate_count_median descending
    results.sort(key=lambda r: (r["candidate_count_median"], r["candidate_count_mean"]), reverse=True)

    out_path = PROJECT_ROOT / "backtest" / "strategy_avwap_profile_candidate_pool_experiments.json"
    payload = {
        "dataset": dataset,
        "holding_cycle": holding_cycle,
        "recent_days": recent_days,
        "trend_index_code": DEFAULT_TREND_INDEX_CODE,
        "target_median_candidates": 6,
        "experiments": results,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWROTE {out_path}")

    # Summary
    print("\n--- Top 5 by median candidate count ---")
    for r in results[:5]:
        print(f"  {r['name']:30s}  median={r['candidate_count_median']:.1f}  mean={r['candidate_count_mean']:.1f}  enough={r['enough_candidates_ratio']:.1%}")


if __name__ == "__main__":
    main()
