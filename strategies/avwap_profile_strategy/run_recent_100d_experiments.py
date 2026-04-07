#!/usr/bin/env python3
"""Run focused recent-100d experiments for AVWAP profile strategy."""

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


def main():
    bt_module = load_module(BACKTEST_SCRIPT, "backtest_strategies_avwap_experiments")
    dataset = "csi1000_5y_pit"
    holding_cycle = "weekly"
    recent_days = 100

    daily, idx, basic, trade_dates = load_dataset_with_flow_signals(
        DATASETS[dataset],
        bt_module,
        trend_index_code=DEFAULT_TREND_INDEX_CODE,
    )
    start_offset = max(0, len(trade_dates) - recent_days)

    experiments = [
        {"name": "baseline", "breakout_volume_mult": 1.5, "pullback_volume_frac": 0.8, "daily_failure_exit": False, "market_filter_mode": "off"},
        {"name": "daily_exit", "breakout_volume_mult": 1.5, "pullback_volume_frac": 0.8, "daily_failure_exit": True, "market_filter_mode": "off"},
        {"name": "daily_exit_ma60", "breakout_volume_mult": 1.5, "pullback_volume_frac": 0.8, "daily_failure_exit": True, "market_filter_mode": "ma60"},
        {"name": "looser_volume", "breakout_volume_mult": 1.3, "pullback_volume_frac": 0.9, "daily_failure_exit": True, "market_filter_mode": "ma60"},
        {"name": "stricter_volume", "breakout_volume_mult": 1.7, "pullback_volume_frac": 0.7, "daily_failure_exit": True, "market_filter_mode": "ma60"},
    ]

    results = []
    for exp in experiments:
        cfg = get_strategy_config(bt_module, holding_cycle)
        cfg.name = f"AVWAPProfile_{exp['name']}"
        strategy_cls = make_avwap_profile_backtest(
            bt_module,
            holding_cycle,
            breakout_volume_mult=exp["breakout_volume_mult"],
            pullback_volume_frac=exp["pullback_volume_frac"],
            daily_failure_exit=exp["daily_failure_exit"],
            market_filter_mode=exp["market_filter_mode"],
        )
        bt = strategy_cls(cfg, daily, idx, basic, trade_dates)
        captured = io.StringIO()
        t0 = time.time()
        with contextlib.redirect_stdout(captured):
            result = bt.run(start_offset=start_offset)
        result["elapsed_sec"] = round(time.time() - t0, 1)
        result["experiment"] = exp["name"]
        result["recent_days"] = recent_days
        result["recent_start_date"] = trade_dates[start_offset]
        results.append(result)
        print(
            exp["name"],
            {
                "total_return_pct": result["total_return_pct"],
                "annual_return_pct": result["annual_return_pct"],
                "sharpe": result["sharpe"],
                "max_drawdown_pct": result["max_drawdown_pct"],
                "excess_return_pct": result["excess_return_pct"],
                "candidate_count_mean": result["candidate_count_mean"],
            },
        )

    results.sort(key=lambda item: item["sharpe"], reverse=True)
    out_path = PROJECT_ROOT / "backtest" / "strategy_avwap_profile_csi1000_5y_pit_weekly_100d_experiments.json"
    payload = {
        "dataset": dataset,
        "holding_cycle": holding_cycle,
        "recent_days": recent_days,
        "trend_index_code": DEFAULT_TREND_INDEX_CODE,
        "experiments": results,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WROTE {out_path}")


if __name__ == "__main__":
    main()
