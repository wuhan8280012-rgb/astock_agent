#!/usr/bin/env python3
"""Compare confirmed-entry and tight daily-rank review execution overlays for F."""

import importlib.util
import json
from pathlib import Path

import pandas as pd

from run_backtest import (
    DATASETS,
    calc_strength_transition_coef,
    load_dataset,
    make_filtered_backtest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKTEST_SCRIPT = PROJECT_ROOT / "scripts" / "backtest_strategies.py"


def load_backtest_module():
    spec = importlib.util.spec_from_file_location("backtest_strategies", BACKTEST_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ConfirmedAndTightReviewBacktest:
    """Use official F scoring/position sizing, but delay new entries and cut rank collapse faster."""

    def __init__(self, base_bt):
        self.base = base_bt
        self.cfg = base_bt.cfg
        self.trade_dates = base_bt.trade_dates
        self._idx_series = base_bt._idx_series

    def _rank_map(self, date: str):
        scores = self.base._score_universe(date)
        return {code: i + 1 for i, (code, _) in enumerate(scores)}

    def _get_prices(self, date: str):
        return self.base._get_prices(date)

    def _get_trend_position(self, date: str):
        return self.base._get_trend_position(date)

    def run(self, start_offset: int):
        cfg = self.cfg
        dates = self.trade_dates
        capital = 1_000_000.0
        cash = capital
        positions = {}
        pending = []
        nav_history = []
        trade_count = 0
        rebalance_count = 0
        last_rebalance_idx = start_offset - cfg.rebalance_interval

        for i in range(start_offset, len(dates)):
            date = dates[i]
            prices = self._get_prices(date)
            rank_map = self._rank_map(date)

            # 1) Execute pending confirmed entries
            new_pending = []
            for item in pending:
                code = item["code"]
                entry_idx = item["rebalance_idx"]
                target_day_idx = i - entry_idx
                if code in positions:
                    continue
                if code not in prices or prices[code] <= 0:
                    if target_day_idx < 3:
                        new_pending.append(item)
                    continue

                buy_now = False
                reason = None
                if target_day_idx == 1 and rank_map.get(code, 10_000) <= cfg.top_n:
                    buy_now = True
                    reason = "CONFIRM_NEXT_DAY_TOP"
                elif target_day_idx <= 3:
                    low_ok = True
                    for j in range(entry_idx + 1, min(entry_idx + 4, i + 1)):
                        d = dates[j]
                        stock_row = self.base._stock_data[code].loc[d] if d in self.base._stock_data[code].index else None
                        if stock_row is None:
                            low_ok = False
                            break
                        if float(stock_row["low"]) < item["ref_price"] * 0.95:
                            low_ok = False
                            break
                    if target_day_idx == 3 and low_ok:
                        buy_now = True
                        reason = "CONFIRM_3D_NO_BREAK_5"

                if buy_now:
                    portfolio_value = cash + sum(
                        pos["shares"] * prices.get(c, pos.get("current_price", pos["cost_price"]))
                        for c, pos in positions.items()
                    )
                    position_value = sum(
                        pos["shares"] * prices.get(c, pos.get("current_price", pos["cost_price"]))
                        for c, pos in positions.items()
                    )
                    max_position_pct = self._get_trend_position(date)
                    max_equity = portfolio_value * max_position_pct
                    available_for_equity = max_equity - position_value
                    available_cash = min(cash, available_for_equity) if available_for_equity > 0 else 0
                    if available_cash >= 10000:
                        buy_price = prices[code] * (1 + cfg.slippage)
                        target_amount = min(portfolio_value * cfg.max_single_weight, available_cash * 0.95)
                        shares = int(target_amount / buy_price / 100) * 100
                        if shares >= 100:
                            amount = shares * buy_price
                            cost = amount * cfg.commission
                            cash -= (amount + cost)
                            positions[code] = {
                                "shares": shares,
                                "cost_price": buy_price,
                                "entry_date": date,
                                "entry_idx": i,
                                "current_price": prices[code],
                                "entry_rank": rank_map.get(code, 10_000),
                            }
                            trade_count += 1
                            continue

                if target_day_idx < 3:
                    new_pending.append(item)
            pending = new_pending

            # 2) Mark prices and stop-loss / rank-review exits
            for code, pos in list(positions.items()):
                if code in prices:
                    pos["current_price"] = prices[code]

            for code in list(positions.keys()):
                pos = positions[code]
                if code not in prices:
                    continue
                sell_reason = None
                pnl = prices[code] / pos["cost_price"] - 1
                if pnl <= cfg.stop_loss_pct:
                    sell_reason = "STOP_LOSS"
                else:
                    held_days = max(0, i - pos.get("entry_idx", i))
                    rank = rank_map.get(code, 10_000)
                    entry_rank = pos.get("entry_rank", rank)
                    if held_days <= 3 and rank > 25 and (rank - entry_rank) >= 12:
                        sell_reason = "TIGHT_REVIEW_EARLY"
                    elif held_days > 3 and rank > 35:
                        sell_reason = "TIGHT_REVIEW_MATURE"

                if sell_reason:
                    sell_price = prices[code] * (1 - cfg.slippage)
                    proceeds = pos["shares"] * sell_price
                    cost = proceeds * (cfg.commission + cfg.stamp_tax)
                    cash += proceeds - cost
                    trade_count += 1
                    del positions[code]

            # 3) Rebalance creates pending candidates, sells out-of-buffer old names
            if (i - last_rebalance_idx) >= cfg.rebalance_interval:
                scores = self.base._score_universe(date)
                if len(scores) >= cfg.top_n:
                    target_codes = [code for code, _ in scores[:cfg.top_n]]
                    buffer_codes = {code for code, _ in scores[: int(cfg.top_n * cfg.hold_buffer_ratio)]}

                    for code in list(positions.keys()):
                        if code not in buffer_codes and code in prices:
                            sell_price = prices[code] * (1 - cfg.slippage)
                            proceeds = positions[code]["shares"] * sell_price
                            cost = proceeds * (cfg.commission + cfg.stamp_tax)
                            cash += proceeds - cost
                            trade_count += 1
                            del positions[code]

                    pending = []
                    for code in target_codes:
                        if code in positions or code not in prices or prices[code] <= 0:
                            continue
                        pending.append(
                            {
                                "code": code,
                                "rebalance_idx": i,
                                "ref_price": float(prices[code] * (1 + cfg.slippage)),
                            }
                        )
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
                    "daily_return": None,
                }
            )

        nav_df = pd.DataFrame(nav_history)
        nav_df["daily_return"] = nav_df["nav"].pct_change()
        total_return = (nav_df["nav"].iloc[-1] / capital - 1) * 100
        years = len(nav_df) / 252
        annual_return = ((nav_df["nav"].iloc[-1] / capital) ** (1 / years) - 1) * 100
        annual_vol = nav_df["daily_return"].std() * (252 ** 0.5) * 100
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0.0
        cummax = nav_df["nav"].cummax()
        drawdown = (nav_df["nav"] - cummax) / cummax
        max_dd = drawdown.min() * 100
        max_dd_date = nav_df.loc[drawdown.idxmin(), "date"]
        idx_start = self._idx_series.get(dates[start_offset], None)
        idx_end = self._idx_series.get(dates[-1], None)
        benchmark_return = ((idx_end / idx_start) - 1) * 100 if idx_start and idx_end else 0
        return {
            "total_return_pct": round(total_return, 2),
            "annual_return_pct": round(annual_return, 2),
            "annual_vol_pct": round(annual_vol, 2),
            "sharpe": round(sharpe, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "max_dd_date": max_dd_date,
            "benchmark_return_pct": round(benchmark_return, 2),
            "excess_return_pct": round(total_return - benchmark_return, 2),
            "total_trades": trade_count,
            "rebalance_count": rebalance_count,
            "final_value": round(nav_df["nav"].iloc[-1], 2),
            "trading_days": int(len(nav_df)),
        }


def run_window(window_name: str, window_days: int):
    module = load_backtest_module()
    daily, idx, basic, trade_dates = load_dataset(DATASETS["csi1000_5y"], module)
    cfg = [s for s in module.get_strategies() if s.name == "F_三因子+趋势过滤"][0]
    backtest_cls = make_filtered_backtest(module, min_transition_coef=-0.1)
    start_offset = len(trade_dates) - window_days

    baseline = backtest_cls(cfg, daily, idx, basic, trade_dates).run(start_offset=start_offset)
    combo = ConfirmedAndTightReviewBacktest(backtest_cls(cfg, daily, idx, basic, trade_dates)).run(start_offset=start_offset)
    return {
        "window_start_date": trade_dates[start_offset],
        "window_end_date": trade_dates[-1],
        "baseline": baseline,
        "confirmed_entry_plus_tight_review": combo,
    }


def main():
    results = {
        "data_file": str(DATASETS["csi1000_5y"]),
        "results": {
            "recent_30d": run_window("recent_30d", 30),
            "recent_100d": run_window("recent_100d", 100),
        },
    }
    out = PROJECT_ROOT / "backtest" / "strategy_f_confirmed_entry_plus_tight_review_compare.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
