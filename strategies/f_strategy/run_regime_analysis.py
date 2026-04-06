#!/usr/bin/env python3
"""Run official F backtests and attribute daily returns by trend state."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_backtest import DATASETS, load_backtest_module, load_dataset, make_angle_filtered_backtest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_THRESHOLDS = [-5.0, 0.0, 5.0, 10.0, 15.0]


def annualize(total_return: float, n_days: int) -> float:
    if n_days <= 0:
        return 0.0
    base = 1.0 + total_return
    if base <= 0:
        return -100.0
    return (base ** (252.0 / n_days) - 1.0) * 100.0


def summarize_regimes(daily_df: pd.DataFrame) -> tuple[dict, dict]:
    regime_summary = {}
    annual_matrix = {}
    total_log_return = (daily_df["daily_return"].fillna(0.0) + 1.0).map(np.log).sum()

    for regime in ["BULL", "RANGE", "BEAR"]:
        grp = daily_df[daily_df["trend_state"] == regime].copy()
        if grp.empty:
            regime_summary[regime] = {
                "days": 0,
                "pct_days": 0.0,
                "total_return_pct": 0.0,
                "annual_return_pct": 0.0,
                "annual_vol_pct": 0.0,
                "sharpe": 0.0,
                "avg_position_pct": 0.0,
                "log_return_contribution_pct": 0.0,
            }
            annual_matrix[regime] = 0.0
            continue

        returns = grp["daily_return"].fillna(0.0)
        total_return = (1.0 + returns).prod() - 1.0
        annual_return = annualize(total_return, len(grp))
        annual_vol = returns.std() * (252.0 ** 0.5) * 100.0
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0.0
        log_contribution = (1.0 + returns).map(np.log).sum()
        regime_summary[regime] = {
            "days": int(len(grp)),
            "pct_days": round(len(grp) / len(daily_df) * 100.0, 2),
            "total_return_pct": round(total_return * 100.0, 2),
            "annual_return_pct": round(annual_return, 2),
            "annual_vol_pct": round(annual_vol, 2),
            "sharpe": round(sharpe, 2),
            "avg_position_pct": round(grp["position_pct"].mean() * 100.0, 2),
            "log_return_contribution_pct": round((log_contribution / total_log_return) * 100.0, 2)
            if total_log_return != 0
            else 0.0,
        }
        annual_matrix[regime] = round(annual_return, 2)

    return regime_summary, annual_matrix


def run_threshold(module, daily, idx, basic, trade_dates, threshold: float) -> dict:
    cfg = [s for s in module.get_strategies() if s.name == "F_三因子+趋势过滤"][0]
    bt_cls = make_angle_filtered_backtest(module, threshold)
    bt = bt_cls(cfg, daily, idx, basic, trade_dates)
    result = bt.run(start_offset=250, include_daily=True)
    daily_df = pd.DataFrame(result["daily_records"])
    regime_summary, annual_matrix = summarize_regimes(daily_df)

    return {
        "threshold": threshold,
        "official_result": {
            k: result[k]
            for k in [
                "name",
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
                "yearly_returns",
                "trading_days",
            ]
        },
        "regime_summary": regime_summary,
        "daily_records": result["daily_records"],
        "regime_annual_matrix": annual_matrix,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="csi1000_5y")
    parser.add_argument("--thresholds", nargs="*", type=float, default=DEFAULT_THRESHOLDS)
    args = parser.parse_args()

    module = load_backtest_module()
    data_path = DATASETS[args.dataset]
    daily, idx, basic, trade_dates = load_dataset(data_path, module)

    runs = [run_threshold(module, daily, idx, basic, trade_dates, threshold) for threshold in args.thresholds]
    matrix = {
        regime: {str(run["threshold"]).rstrip("0").rstrip("."): run["regime_annual_matrix"][regime] for run in runs}
        for regime in ["BULL", "RANGE", "BEAR"]
    }

    output = {
        "data_file": str(data_path),
        "thresholds": args.thresholds,
        "runs": runs,
        "regime_annual_matrix": matrix,
        "note": "Uses official F backtester daily records; no reimplemented trading logic.",
    }

    out_path = PROJECT_ROOT / "backtest" / f"strategy_f_official_regime_analysis_{args.dataset}.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out_path), "matrix": matrix}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
