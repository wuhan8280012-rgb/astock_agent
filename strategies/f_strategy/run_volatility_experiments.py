#!/usr/bin/env python3
"""Sweep low-volatility factor settings for the current F strategy."""

from __future__ import annotations

import importlib.util
import json
import time
from copy import deepcopy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKTEST_SCRIPT = PROJECT_ROOT / "scripts" / "backtest_strategies.py"
RUN_SCRIPT = PROJECT_ROOT / "strategies" / "f_strategy" / "run_backtest.py"
OUT_PATH = PROJECT_ROOT / "backtest" / "strategy_f_volatility_experiments_csi1000_5y.json"

VOL_WINDOWS = [20, 60, 120]
VOL_WEIGHTS = [0.10, 0.25, 0.40, 0.60]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize(label: str, kind: str, cfg, result: dict) -> dict:
    return {
        "label": label,
        "kind": kind,
        "volatility_days": cfg.volatility_days if cfg.use_volatility_factor else None,
        "volatility_weight": cfg.volatility_weight if cfg.use_volatility_factor else 0.0,
        "total_return_pct": result["total_return_pct"],
        "annual_return_pct": result["annual_return_pct"],
        "annual_vol_pct": result["annual_vol_pct"],
        "sharpe": result["sharpe"],
        "calmar": result["calmar"],
        "max_drawdown_pct": result["max_drawdown_pct"],
        "benchmark_return_pct": result["benchmark_return_pct"],
        "excess_return_pct": result["excess_return_pct"],
        "total_trades": result["total_trades"],
        "rebalance_count": result["rebalance_count"],
        "trend_filter_index_code": result.get("trend_filter_index_code", "000001.SH"),
    }


def main():
    btm = load_module(BACKTEST_SCRIPT, "backtest_strategies")
    runm = load_module(RUN_SCRIPT, "f_run_backtest")

    daily, idx, basic, trade_dates = runm.load_dataset(
        runm.DATASETS["csi1000_5y"],
        btm,
        trend_index_code=runm.DEFAULT_TREND_INDEX_CODE,
    )
    base_cfg = [s for s in btm.get_strategies() if s.name == "F_三因子+趋势过滤"][0]
    backtest_cls = runm.make_filtered_backtest(btm, min_transition_coef=-0.1)

    experiments = []

    variants = [("baseline", base_cfg)]

    no_vol_cfg = deepcopy(base_cfg)
    no_vol_cfg.use_volatility_factor = False
    variants.append(("no_vol_factor", no_vol_cfg))

    for window in VOL_WINDOWS:
        for weight in VOL_WEIGHTS:
            cfg = deepcopy(base_cfg)
            cfg.volatility_days = window
            cfg.volatility_weight = weight
            variants.append((f"vol_{window}d_w_{str(weight).replace('.', '_')}", cfg))

    t0 = time.time()
    for label, cfg in variants:
        bt = backtest_cls(cfg, daily, idx, basic, trade_dates)
        result = bt.run(start_offset=250)
        result["trend_filter_index_code"] = runm.DEFAULT_TREND_INDEX_CODE
        kind = "baseline" if label == "baseline" else ("no_vol" if label == "no_vol_factor" else "grid")
        experiments.append(summarize(label, kind, cfg, result))

    best_by_sharpe = max(experiments, key=lambda x: (x["sharpe"], x["annual_return_pct"]))
    best_by_annual = max(experiments, key=lambda x: (x["annual_return_pct"], x["sharpe"]))
    best_by_calmar = max(experiments, key=lambda x: (x["calmar"], x["annual_return_pct"]))

    output = {
        "strategy": "F + strength_transition_coef >= -0.1",
        "trend_filter_index_code": runm.DEFAULT_TREND_INDEX_CODE,
        "dataset": "csi1000_5y",
        "elapsed_sec": round(time.time() - t0, 1),
        "experiments": experiments,
        "best_by_sharpe": best_by_sharpe,
        "best_by_annual_return": best_by_annual,
        "best_by_calmar": best_by_calmar,
    }
    OUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"WROTE {OUT_PATH}")


if __name__ == "__main__":
    main()
