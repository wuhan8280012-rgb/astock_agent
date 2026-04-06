#!/usr/bin/env python3
"""Compare RANGE position sizing for official F + ma20_angle_deg >= 0."""

import json
from pathlib import Path

import pandas as pd

from run_backtest import DATASETS, load_backtest_module, load_dataset, make_angle_filtered_backtest
from run_regime_analysis import summarize_regimes


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_variant(module, daily, idx, basic, trade_dates, trend_reduce_pct: float) -> dict:
    cfg = [s for s in module.get_strategies() if s.name == "F_三因子+趋势过滤"][0]
    cfg.trend_reduce_pct = trend_reduce_pct
    bt_cls = make_angle_filtered_backtest(module, 0.0)
    bt = bt_cls(cfg, daily, idx, basic, trade_dates)
    result = bt.run(start_offset=250, include_daily=True)
    daily_df = pd.DataFrame(result["daily_records"])
    regime_summary, annual_matrix = summarize_regimes(daily_df)
    return {
        "trend_reduce_pct": trend_reduce_pct,
        "label": f"range_{int(trend_reduce_pct * 100)}pct",
        "result": {
            k: result[k]
            for k in [
                "total_return_pct",
                "annual_return_pct",
                "annual_vol_pct",
                "sharpe",
                "calmar",
                "max_drawdown_pct",
                "max_dd_date",
                "total_trades",
                "rebalance_count",
                "final_value",
                "benchmark_return_pct",
                "excess_return_pct",
                "trading_days",
            ]
        },
        "regime_summary": regime_summary,
        "regime_annual_matrix": annual_matrix,
    }


def main():
    module = load_backtest_module()
    data_path = DATASETS["csi1000_5y"]
    daily, idx, basic, trade_dates = load_dataset(data_path, module)

    variants = [0.5, 0.25, 0.0]
    runs = [run_variant(module, daily, idx, basic, trade_dates, v) for v in variants]

    out = {
        "data_file": str(data_path),
        "strategy": "F + ma20_angle_deg >= 0",
        "variants": runs,
        "note": "Official F backtester, only trend_reduce_pct changed.",
    }
    out_path = PROJECT_ROOT / "backtest" / "strategy_f_range_position_experiments_csi1000_5y.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out_path), "variants": runs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
