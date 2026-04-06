#!/usr/bin/env python3
"""Compare baseline F against a RANGE-only ma10-angle exit on recent 100d."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_backtest import (
    DATASETS,
    DEFAULT_TREND_INDEX_CODE,
    load_backtest_module,
    load_dataset,
    make_filtered_backtest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "backtest" / "strategy_f_recent_100d_range_ma10_angle_exit_compare.json"


class RangeMa10ExitBacktest:
    def __init__(self, inner):
        self.inner = inner
        self.cfg = inner.cfg
        self.daily = inner.daily
        self.idx = inner.idx
        self.basic = inner.basic
        self.trade_dates = inner.trade_dates
        self._stock_data = inner._stock_data
        self._basic_map = inner._basic_map
        self._idx_series = inner._idx_series
        self._get_prices = inner._get_prices
        self._get_trend_position = inner._get_trend_position
        self._get_trend_state = inner._get_trend_state
        self._score_universe = inner._score_universe
        self._check_halt = inner._check_halt

    def run(self, start_offset: int) -> dict:
        cfg = self.cfg
        dates = self.trade_dates
        capital = 1_000_000.0
        start_date = dates[start_offset]

        cash = capital
        positions = {}
        nav_history = []
        trade_count = 0
        rebalance_count = 0
        last_rebalance_idx = start_offset - cfg.rebalance_interval

        for i in range(start_offset, len(dates)):
            date = dates[i]
            prices = self._get_prices(date)

            portfolio_value = cash
            for code, pos in list(positions.items()):
                if code in prices:
                    pos["current_price"] = prices[code]
                    portfolio_value += pos["shares"] * prices[code]
                else:
                    portfolio_value += pos["shares"] * pos.get("current_price", pos["cost_price"])

            trend_state = self._get_trend_state(date) if cfg.use_trend_filter else "BULL"

            if cfg.stop_loss_pct > -0.99:
                for code in list(positions.keys()):
                    pos = positions[code]
                    if code not in prices:
                        continue

                    should_sell = False
                    pnl = prices[code] / pos["cost_price"] - 1
                    if pnl <= cfg.stop_loss_pct:
                        should_sell = True

                    if not should_sell and trend_state == "RANGE":
                        row = self._stock_data[code].loc[date]
                        ma10_angle = pd.to_numeric(row.get("ma10_angle_deg", np.nan), errors="coerce")
                        if not pd.isna(ma10_angle) and ma10_angle < 0:
                            should_sell = True

                    if should_sell:
                        sell_price = prices[code] * (1 - cfg.slippage)
                        proceeds = pos["shares"] * sell_price
                        cost = proceeds * (cfg.commission + cfg.stamp_tax)
                        cash += proceeds - cost
                        trade_count += 1
                        del positions[code]

            should_rebalance = (i - last_rebalance_idx) >= cfg.rebalance_interval
            max_position_pct = 1.0

            if should_rebalance:
                halt = False
                if cfg.use_halt:
                    halt = self._check_halt(date)
                    if halt:
                        for code in list(positions.keys()):
                            if code in prices:
                                sell_price = prices[code] * (1 - cfg.slippage)
                                proceeds = positions[code]["shares"] * sell_price
                                cost = proceeds * (cfg.commission + cfg.stamp_tax)
                                cash += proceeds - cost
                                trade_count += 1
                        positions = {}
                        last_rebalance_idx = i
                        rebalance_count += 1

                if not halt:
                    if cfg.use_trend_filter:
                        max_position_pct = self._get_trend_position(date)

                    scores = self._score_universe(date)
                    if len(scores) >= cfg.top_n:
                        target_codes = [s[0] for s in scores[: cfg.top_n]]
                        buffer_codes = set(s[0] for s in scores[: int(cfg.top_n * cfg.hold_buffer_ratio)])

                        for code in list(positions.keys()):
                            if code not in buffer_codes:
                                if code in prices:
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
                        current_position_value = sum(
                            pos["shares"] * prices.get(code, pos.get("current_price", pos["cost_price"]))
                            for code, pos in positions.items()
                        )
                        max_equity = portfolio_value_now * max_position_pct
                        available_for_equity = max_equity - current_position_value
                        available_cash = min(cash, available_for_equity) if available_for_equity > 0 else 0

                        hold_count = len(positions)
                        buy_slots = cfg.top_n - hold_count

                        for code in target_codes:
                            if buy_slots <= 0 or available_cash < 10000:
                                break
                            if code in positions:
                                continue
                            if code not in prices or prices[code] <= 0:
                                continue

                            buy_price = prices[code] * (1 + cfg.slippage)
                            target_amount = min(portfolio_value_now * cfg.max_single_weight, available_cash * 0.95)
                            shares = int(target_amount / buy_price / 100) * 100
                            if shares >= 100:
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
                                buy_slots -= 1

                        last_rebalance_idx = i
                        rebalance_count += 1

            final_value = cash + sum(
                pos["shares"] * prices.get(code, pos.get("current_price", pos["cost_price"]))
                for code, pos in positions.items()
            )
            position_value = sum(
                pos["shares"] * prices.get(code, pos.get("current_price", pos["cost_price"]))
                for code, pos in positions.items()
            )
            nav_history.append(
                {
                    "date": date,
                    "nav": final_value,
                    "cash": cash,
                    "position_value": position_value,
                    "position_pct": position_value / final_value if final_value > 0 else 0.0,
                    "max_position_pct": max_position_pct,
                    "trend_state": trend_state,
                    "idx_close": float(self._idx_series.get(date, np.nan)),
                }
            )

        nav_df = pd.DataFrame(nav_history)
        nav_df["daily_return"] = nav_df["nav"].pct_change()
        total_return = (nav_df["nav"].iloc[-1] / capital - 1) * 100
        days = len(nav_df)
        years = days / 252
        annual_return = ((nav_df["nav"].iloc[-1] / capital) ** (1 / years) - 1) * 100
        annual_vol = nav_df["daily_return"].std() * np.sqrt(252) * 100
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0

        cummax = nav_df["nav"].cummax()
        drawdown = (nav_df["nav"] - cummax) / cummax
        max_dd = drawdown.min() * 100
        max_dd_date = nav_df.loc[drawdown.idxmin(), "date"] if len(nav_df) > 0 else ""

        idx_start = self._idx_series.get(start_date, None)
        idx_end = self._idx_series.get(dates[-1], None)
        benchmark_return = ((idx_end / idx_start) - 1) * 100 if idx_start and idx_end else 0

        result = {
            "name": cfg.name,
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
            "trading_days": days,
        }
        return result


def add_ma10_angle_column(data_path: Path, daily: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_csv(data_path, usecols=["data_type", "ts_code", "trade_date", "ma10_angle_deg"], low_memory=False)
    extra = raw[raw["data_type"] == "daily"][["ts_code", "trade_date", "ma10_angle_deg"]].copy()
    extra["trade_date"] = extra["trade_date"].astype(str)
    extra["ma10_angle_deg"] = pd.to_numeric(extra["ma10_angle_deg"], errors="coerce")
    merged = daily.merge(extra, on=["ts_code", "trade_date"], how="left")
    return merged


def to_builtin(value):
    if isinstance(value, dict):
        return {k: to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    module = load_backtest_module()
    data_path = DATASETS["csi1000_5y"]
    daily, idx, basic, trade_dates = load_dataset(data_path, module, trend_index_code=DEFAULT_TREND_INDEX_CODE)
    daily = add_ma10_angle_column(data_path, daily)

    cfg = next(s for s in module.get_strategies() if s.name == "F_三因子+趋势过滤")
    backtest_cls = make_filtered_backtest(module, min_transition_coef=-0.1)
    start_offset = len(trade_dates) - 100

    t0 = time.time()
    baseline = backtest_cls(cfg, daily, idx, basic, trade_dates).run(start_offset=start_offset)
    modified = RangeMa10ExitBacktest(backtest_cls(cfg, daily, idx, basic, trade_dates)).run(start_offset=start_offset)
    elapsed = round(time.time() - t0, 2)

    payload = {
        "data_file": str(data_path),
        "trend_filter_index_code": DEFAULT_TREND_INDEX_CODE,
        "rule": "Keep baseline -15% stop loss. Additionally, in RANGE state, sell any holding if ma10_angle_deg < 0.",
        "window_trade_days": 100,
        "window_start_date": trade_dates[start_offset],
        "window_end_date": trade_dates[-1],
        "baseline": baseline,
        "range_ma10_angle_exit": modified,
        "elapsed_sec": elapsed,
    }
    payload = to_builtin(payload)
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUTPUT_PATH)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
