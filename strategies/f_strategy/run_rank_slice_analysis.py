#!/usr/bin/env python3
"""Analyze forward returns of F + ma20_angle_deg >= 0 by rank slices."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_backtest import DATASETS, load_backtest_module, load_dataset, make_angle_filtered_backtest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def annualize(avg_period_return: float, holding_days: int) -> float:
    base = 1.0 + avg_period_return
    if base <= 0 or holding_days <= 0:
        return -100.0
    return (base ** (252.0 / holding_days) - 1.0) * 100.0


def summarize_returns(series: pd.Series, holding_days: int) -> dict:
    series = series.dropna()
    if series.empty:
        return {
            "count": 0,
            "avg_return_pct": 0.0,
            "median_return_pct": 0.0,
            "win_rate_pct": 0.0,
            "annualized_from_avg_pct": 0.0,
        }
    avg = series.mean()
    return {
        "count": int(series.size),
        "avg_return_pct": round(avg * 100.0, 2),
        "median_return_pct": round(series.median() * 100.0, 2),
        "win_rate_pct": round((series > 0).mean() * 100.0, 2),
        "annualized_from_avg_pct": round(annualize(avg, holding_days), 2),
    }


def main():
    module = load_backtest_module()
    data_path = DATASETS["csi1000_5y"]
    daily, idx, basic, trade_dates = load_dataset(data_path, module)
    cfg = [s for s in module.get_strategies() if s.name == "F_三因子+趋势过滤"][0]
    bt_cls = make_angle_filtered_backtest(module, 0.0)
    bt = bt_cls(cfg, daily, idx, basic, trade_dates)

    holding_days = cfg.rebalance_interval
    rows = []
    start_offset = 250
    last_rebalance_idx = start_offset - cfg.rebalance_interval

    for i in range(start_offset, len(trade_dates) - holding_days):
        date = trade_dates[i]
        should_rebalance = (i - last_rebalance_idx) >= cfg.rebalance_interval
        if not should_rebalance:
            continue

        max_position_pct = bt._get_trend_position(date) if cfg.use_trend_filter else 1.0
        last_rebalance_idx = i
        if max_position_pct <= 0:
            continue

        scores = bt._score_universe(date)
        if len(scores) < cfg.top_n:
            continue

        target_codes = [code for code, _ in scores[:cfg.top_n]]
        sell_date = trade_dates[i + holding_days]

        for rank, code in enumerate(target_codes, start=1):
            data = bt._stock_data.get(code)
            if data is None or date not in data.index or sell_date not in data.index:
                continue
            buy_close = float(data.loc[date, "close"])
            sell_close = float(data.loc[sell_date, "close"])
            if buy_close <= 0 or sell_close <= 0:
                continue

            buy_px = buy_close * (1 + cfg.slippage)
            sell_px = sell_close * (1 - cfg.slippage)
            gross_ret = sell_px / buy_px - 1.0
            net_ret = ((sell_px * (1 - cfg.commission - cfg.stamp_tax)) / (buy_px * (1 + cfg.commission))) - 1.0
            rows.append(
                {
                    "rebalance_date": date,
                    "sell_date": sell_date,
                    "rank": rank,
                    "ts_code": code,
                    "gross_return": gross_ret,
                    "net_return": net_ret,
                }
            )

    df = pd.DataFrame(rows)
    rank_summary = {}
    for rank in range(1, cfg.top_n + 1):
        rank_summary[str(rank)] = summarize_returns(df.loc[df["rank"] == rank, "net_return"], holding_days)

    bucket_defs = {
        "1-3": (1, 3),
        "4-6": (4, 6),
        "7-10": (7, 10),
        "11-15": (11, 15),
    }
    bucket_summary = {}
    for label, (lo, hi) in bucket_defs.items():
        bucket_summary[label] = summarize_returns(
            df.loc[(df["rank"] >= lo) & (df["rank"] <= hi), "net_return"], holding_days
        )

    best_rank = max(rank_summary.items(), key=lambda kv: kv[1]["avg_return_pct"])
    best_bucket = max(bucket_summary.items(), key=lambda kv: kv[1]["avg_return_pct"])

    output = {
        "data_file": str(data_path),
        "strategy": "F + ma20_angle_deg >= 0",
        "holding_days": holding_days,
        "trade_count": int(len(df)),
        "rebalance_count": int(df["rebalance_date"].nunique()) if not df.empty else 0,
        "rank_summary": rank_summary,
        "bucket_summary": bucket_summary,
        "best_rank": {"rank": best_rank[0], **best_rank[1]},
        "best_bucket": {"bucket": best_bucket[0], **best_bucket[1]},
    }

    out_path = PROJECT_ROOT / "backtest" / "strategy_f_rank_slice_analysis_csi1000_5y.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out_path), "best_rank": output["best_rank"], "best_bucket": output["best_bucket"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
