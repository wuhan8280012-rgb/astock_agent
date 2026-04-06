#!/usr/bin/env python3
"""Test regime-aware industry 20d excess return rank bonuses on top of the current F baseline."""

from __future__ import annotations

import json
import time
from pathlib import Path

from run_backtest import (
    DATASETS,
    DEFAULT_TREND_INDEX_CODE,
    PROJECT_ROOT,
    load_backtest_module,
    load_dataset,
    load_local_trend_index_df,
    make_filtered_backtest,
    make_regime_industry_excess_rank_boost_backtest,
)


OUT_PATH = PROJECT_ROOT / "backtest" / "strategy_f_industry_excess20_regime_bonus_experiments_csi1000_5y.json"
LOCAL_TREND_INDEX_PATH = PROJECT_ROOT / "data" / "market_index_000001sh_5y.csv"


def run_variant(module, daily, idx, basic, trade_dates, label: str, bt_cls) -> dict:
    cfg = next(s for s in module.get_strategies() if s.name == "F_三因子+趋势过滤")
    bt = bt_cls(cfg, daily, idx, basic, trade_dates)
    t0 = time.time()
    result = bt.run(start_offset=250)
    cleaned = {
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
            "yearly_returns",
            "trading_days",
        ]
    }
    cleaned["elapsed_sec"] = round(time.time() - t0, 2)
    return {"experiment": label, "result": cleaned}


def main():
    module = load_backtest_module()

    daily, idx, basic, trade_dates = load_dataset(
        DATASETS["csi1000_5y"],
        module,
        trend_index_code=DEFAULT_TREND_INDEX_CODE,
        trend_index_loader=lambda trade_dates, trend_index_code: load_local_trend_index_df(
            LOCAL_TREND_INDEX_PATH,
            trade_dates,
        ),
    )

    experiments = [
        (
            "baseline",
            make_filtered_backtest(module, min_transition_coef=-0.1),
            {
                "bonus_weight": 0.0,
                "active_regimes": [],
                "note": "Current official F baseline",
            },
        ),
        (
            "industry_excess20_bonus_w0_10_range",
            make_regime_industry_excess_rank_boost_backtest(
                module,
                bonus_weight=0.10,
                active_regimes=("RANGE",),
                min_transition_coef=-0.1,
            ),
            {
                "bonus_weight": 0.10,
                "active_regimes": ["RANGE"],
                "note": "20d industry excess rank bonus only in RANGE",
            },
        ),
        (
            "industry_excess20_bonus_w0_20_range",
            make_regime_industry_excess_rank_boost_backtest(
                module,
                bonus_weight=0.20,
                active_regimes=("RANGE",),
                min_transition_coef=-0.1,
            ),
            {
                "bonus_weight": 0.20,
                "active_regimes": ["RANGE"],
                "note": "20d industry excess rank bonus only in RANGE",
            },
        ),
        (
            "industry_excess20_bonus_w0_10_bear",
            make_regime_industry_excess_rank_boost_backtest(
                module,
                bonus_weight=0.10,
                active_regimes=("BEAR",),
                min_transition_coef=-0.1,
            ),
            {
                "bonus_weight": 0.10,
                "active_regimes": ["BEAR"],
                "note": "20d industry excess rank bonus only in BEAR",
            },
        ),
        (
            "industry_excess20_bonus_w0_20_bear",
            make_regime_industry_excess_rank_boost_backtest(
                module,
                bonus_weight=0.20,
                active_regimes=("BEAR",),
                min_transition_coef=-0.1,
            ),
            {
                "bonus_weight": 0.20,
                "active_regimes": ["BEAR"],
                "note": "20d industry excess rank bonus only in BEAR",
            },
        ),
        (
            "industry_excess20_bonus_w0_10_range_bear",
            make_regime_industry_excess_rank_boost_backtest(
                module,
                bonus_weight=0.10,
                active_regimes=("RANGE", "BEAR"),
                min_transition_coef=-0.1,
            ),
            {
                "bonus_weight": 0.10,
                "active_regimes": ["RANGE", "BEAR"],
                "note": "20d industry excess rank bonus in RANGE and BEAR",
            },
        ),
        (
            "industry_excess20_bonus_w0_20_range_bear",
            make_regime_industry_excess_rank_boost_backtest(
                module,
                bonus_weight=0.20,
                active_regimes=("RANGE", "BEAR"),
                min_transition_coef=-0.1,
            ),
            {
                "bonus_weight": 0.20,
                "active_regimes": ["RANGE", "BEAR"],
                "note": "20d industry excess rank bonus in RANGE and BEAR",
            },
        ),
    ]

    runs = []
    for label, cls, meta in experiments:
        print(f"RUN {label}", flush=True)
        run = run_variant(module, daily, idx, basic, trade_dates, label, cls)
        run.update(meta)
        runs.append(run)
        result = run["result"]
        print(
            f"DONE {label} total={result['total_return_pct']} annual={result['annual_return_pct']} "
            f"sharpe={result['sharpe']} mdd={result['max_drawdown_pct']}",
            flush=True,
        )

    baseline = runs[0]["result"]
    for run in runs[1:]:
        result = run["result"]
        result["delta_total_return_pct_vs_baseline"] = round(result["total_return_pct"] - baseline["total_return_pct"], 2)
        result["delta_sharpe_vs_baseline"] = round(result["sharpe"] - baseline["sharpe"], 2)
        result["delta_max_drawdown_pct_vs_baseline"] = round(
            result["max_drawdown_pct"] - baseline["max_drawdown_pct"],
            2,
        )

    payload = {
        "data_file": str(DATASETS["csi1000_5y"]),
        "trend_filter_index_code": DEFAULT_TREND_INDEX_CODE,
        "trend_index_source": str(LOCAL_TREND_INDEX_PATH),
        "window_start_date": trade_dates[250],
        "window_end_date": trade_dates[-1],
        "note": "Industry 20d excess return is used only as a rank bonus, never as a hard filter.",
        "runs": runs,
    }

    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WROTE {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
