#!/usr/bin/env python3
"""Rank-tilted allocation experiments for F + ma20_angle_deg >= 0."""

import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/wuhan/project/stock_agent/new")

from strategies.f_strategy.run_backtest import DATASETS, load_backtest_module, load_dataset, make_angle_filtered_backtest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = PROJECT_ROOT / "backtest" / "strategy_f_rank_tilt_experiments_csi1000_5y.json"


def build_weight_map(kind: str, top_n: int) -> dict[int, float]:
    if kind == "baseline_equal":
        return {rank: 1.0 / top_n for rank in range(1, top_n + 1)}

    if kind == "tilt_4_6":
        weights = {}
        for rank in range(1, top_n + 1):
            if 4 <= rank <= 6:
                weights[rank] = 0.08
            elif 1 <= rank <= 3:
                weights[rank] = 0.07
            else:
                weights[rank] = 0.055
        return weights

    if kind == "tilt_1_3":
        weights = {}
        for rank in range(1, top_n + 1):
            if 1 <= rank <= 3:
                weights[rank] = 0.08
            else:
                weights[rank] = 0.058
        return weights

    raise ValueError(kind)


def run_tilt_backtest(module, daily, idx, basic, trade_dates, kind: str) -> dict:
    cfg = [s for s in module.get_strategies() if s.name == "F_三因子+趋势过滤"][0]
    bt_cls = make_angle_filtered_backtest(module, 0.0)
    bt = bt_cls(cfg, daily, idx, basic, trade_dates)

    capital = 1_000_000.0
    cash = capital
    positions = {}
    nav_history = []
    trade_count = 0
    rebalance_count = 0
    last_rebalance_idx = 250 - cfg.rebalance_interval
    dates = trade_dates
    weights = build_weight_map(kind, cfg.top_n)

    for i in range(250, len(dates)):
        date = dates[i]
        prices = bt._get_prices(date)

        for code, pos in list(positions.items()):
            if code in prices:
                pos["current_price"] = prices[code]

        if cfg.stop_loss_pct > -0.99:
            for code in list(positions.keys()):
                if code not in prices:
                    continue
                pos = positions[code]
                pnl = prices[code] / pos["cost_price"] - 1
                if pnl <= cfg.stop_loss_pct:
                    sell_price = prices[code] * (1 - cfg.slippage)
                    proceeds = pos["shares"] * sell_price
                    cost = proceeds * (cfg.commission + cfg.stamp_tax)
                    cash += proceeds - cost
                    trade_count += 1
                    del positions[code]

        should_rebalance = (i - last_rebalance_idx) >= cfg.rebalance_interval
        max_position_pct = bt._get_trend_position(date) if cfg.use_trend_filter else 1.0

        if should_rebalance:
            scores = bt._score_universe(date)
            if len(scores) >= cfg.top_n:
                target_codes = [s[0] for s in scores[:cfg.top_n]]
                buffer_codes = set(s[0] for s in scores[: int(cfg.top_n * cfg.hold_buffer_ratio)])

                for code in list(positions.keys()):
                    if code not in buffer_codes and code in prices:
                        sell_price = prices[code] * (1 - cfg.slippage)
                        proceeds = positions[code]["shares"] * sell_price
                        cost = proceeds * (cfg.commission + cfg.stamp_tax)
                        cash += proceeds - cost
                        trade_count += 1
                        del positions[code]

                portfolio_value_now = cash + sum(
                    pos["shares"] * prices.get(code, pos.get("current_price", pos["cost_price"]))
                    for code, pos in positions.items()
                )
                max_equity = portfolio_value_now * max_position_pct
                current_position_value = sum(
                    pos["shares"] * prices.get(code, pos.get("current_price", pos["cost_price"]))
                    for code, pos in positions.items()
                )
                available_for_equity = max_equity - current_position_value
                available_cash = min(cash, available_for_equity) if available_for_equity > 0 else 0.0

                for rank, code in enumerate(target_codes, start=1):
                    if code in positions:
                        continue
                    if code not in prices or prices[code] <= 0:
                        continue
                    if available_cash < 10000:
                        break

                    buy_price = prices[code] * (1 + cfg.slippage)
                    desired_weight = min(weights[rank], cfg.max_single_weight)
                    target_amount = min(portfolio_value_now * desired_weight, available_cash * 0.95)
                    shares = int(target_amount / buy_price / 100) * 100
                    if shares < 100:
                        continue
                    amount = shares * buy_price
                    cost = amount * cfg.commission
                    cash -= amount + cost
                    available_cash -= amount + cost
                    positions[code] = {
                        "shares": shares,
                        "cost_price": buy_price,
                        "entry_date": date,
                        "current_price": prices[code],
                    }
                    trade_count += 1

                last_rebalance_idx = i
                rebalance_count += 1

        final_value = cash + sum(
            pos["shares"] * prices.get(code, pos.get("current_price", pos["cost_price"]))
            for code, pos in positions.items()
        )
        nav_history.append({"date": date, "nav": final_value})

    nav_df = module.pd.DataFrame(nav_history)
    nav_df["daily_return"] = nav_df["nav"].pct_change()
    total_return = (nav_df["nav"].iloc[-1] / capital - 1) * 100
    years = len(nav_df) / 252
    annual_return = ((nav_df["nav"].iloc[-1] / capital) ** (1 / years) - 1) * 100
    annual_vol = nav_df["daily_return"].std() * np.sqrt(252) * 100
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    cummax = nav_df["nav"].cummax()
    drawdown = (nav_df["nav"] - cummax) / cummax
    max_dd = drawdown.min() * 100
    max_dd_date = nav_df.loc[drawdown.idxmin(), "date"]
    idx_start = bt._idx_series.get(dates[250], None)
    idx_end = bt._idx_series.get(dates[-1], None)
    benchmark_return = ((idx_end / idx_start) - 1) * 100 if idx_start and idx_end else 0

    return {
        "variant": kind,
        "weights": weights,
        "total_return_pct": round(total_return, 2),
        "annual_return_pct": round(annual_return, 2),
        "annual_vol_pct": round(annual_vol, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "max_dd_date": max_dd_date,
        "total_trades": trade_count,
        "rebalance_count": rebalance_count,
        "final_value": round(nav_df["nav"].iloc[-1], 2),
        "benchmark_return_pct": round(benchmark_return, 2),
        "excess_return_pct": round(total_return - benchmark_return, 2),
    }


def main():
    module = load_backtest_module()
    daily, idx, basic, trade_dates = load_dataset(DATASETS["csi1000_5y"], module)

    variants = ["baseline_equal", "tilt_4_6", "tilt_1_3"]
    results = []
    for kind in variants:
        t0 = time.time()
        result = run_tilt_backtest(module, daily, idx, basic, trade_dates, kind)
        result["elapsed_sec"] = round(time.time() - t0, 1)
        results.append(result)
        print(kind, result["annual_return_pct"], result["sharpe"], result["max_drawdown_pct"], flush=True)

    OUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WROTE {OUT_PATH}")


if __name__ == "__main__":
    main()
