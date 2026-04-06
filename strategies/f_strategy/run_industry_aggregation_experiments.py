#!/usr/bin/env python3
"""Industry aggregation experiments for F strategy."""

import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from run_backtest import DATASETS, load_dataset, make_filtered_backtest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKTEST_SCRIPT = PROJECT_ROOT / "scripts" / "backtest_strategies.py"


def load_backtest_module():
    spec = importlib.util.spec_from_file_location("backtest_strategies", BACKTEST_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class IndustryAggregatedBacktest:
    """Reuse official F engine, but restrict buys to top industries selected from stock ranks."""

    def __init__(self, base_bt, top_industries=5, per_industry_cap=None):
        self.base = base_bt
        self.cfg = base_bt.cfg
        self.trade_dates = base_bt.trade_dates
        self._idx_series = base_bt._idx_series
        self.top_industries = top_industries
        self.per_industry_cap = per_industry_cap
        self._industry_map = {}
        for code, row in base_bt._basic_map.items():
            industry = str(row.get("industry", "")).strip()
            self._industry_map[code] = industry or "UNKNOWN"

    def _get_prices(self, date: str):
        return self.base._get_prices(date)

    def _get_trend_position(self, date: str):
        return self.base._get_trend_position(date)

    def _select_by_industry(self, date: str):
        scores = self.base._score_universe(date)
        if not scores:
            return [], [], []
        ranked = [(code, float(rank), self._industry_map.get(code, "UNKNOWN")) for code, rank in scores]

        bucket = defaultdict(list)
        for code, rank, industry in ranked:
            bucket[industry].append((code, rank))

        industry_scores = []
        for industry, members in bucket.items():
            top_members = members[:3]
            # lower rank is better; use inverse-rank style aggregation
            score = sum(1.0 / r for _, r in top_members) / len(top_members)
            industry_scores.append((industry, score, len(members)))

        industry_scores.sort(key=lambda x: x[1], reverse=True)
        selected_industries = {industry for industry, _, _ in industry_scores[: self.top_industries]}

        target_codes = []
        industry_counts = defaultdict(int)
        for code, rank, industry in ranked:
            if industry not in selected_industries:
                continue
            if self.per_industry_cap is not None and industry_counts[industry] >= self.per_industry_cap:
                continue
            target_codes.append(code)
            industry_counts[industry] += 1
            if len(target_codes) >= self.cfg.top_n:
                break

        # buffer uses same industry restriction but a looser count target
        buffer_limit = int(self.cfg.top_n * self.cfg.hold_buffer_ratio)
        buffer_codes = []
        industry_counts = defaultdict(int)
        cap = None if self.per_industry_cap is None else max(self.per_industry_cap + 1, self.per_industry_cap)
        for code, rank, industry in ranked:
            if industry not in selected_industries:
                continue
            if cap is not None and industry_counts[industry] >= cap:
                continue
            buffer_codes.append(code)
            industry_counts[industry] += 1
            if len(buffer_codes) >= buffer_limit:
                break

        selected_meta = [
            {"industry": industry, "score": round(score, 6), "member_count": member_count}
            for industry, score, member_count in industry_scores[: self.top_industries]
        ]
        return target_codes, set(buffer_codes), selected_meta

    def run(self, start_offset=250):
        cfg = self.cfg
        dates = self.trade_dates
        capital = 1_000_000.0
        cash = capital
        positions = {}
        nav_history = []
        trade_count = 0
        rebalance_count = 0
        last_rebalance_idx = start_offset - cfg.rebalance_interval
        rebalance_meta = []

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

            # daily stop loss unchanged
            if cfg.stop_loss_pct > -0.99:
                for code in list(positions.keys()):
                    pos = positions[code]
                    if code in prices:
                        pnl = prices[code] / pos["cost_price"] - 1
                        if pnl <= cfg.stop_loss_pct:
                            sell_price = prices[code] * (1 - cfg.slippage)
                            proceeds = pos["shares"] * sell_price
                            cost = proceeds * (cfg.commission + cfg.stamp_tax)
                            cash += proceeds - cost
                            trade_count += 1
                            del positions[code]

            should_rebalance = (i - last_rebalance_idx) >= cfg.rebalance_interval
            max_position_pct = 1.0
            if should_rebalance:
                if cfg.use_trend_filter:
                    max_position_pct = self._get_trend_position(date)
                target_codes, buffer_codes, meta = self._select_by_industry(date)
                if len(target_codes) >= min(cfg.top_n, 5):
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
                    current_position_value = sum(
                        pos["shares"] * prices.get(code, pos.get("current_price", pos["cost_price"]))
                        for code, pos in positions.items()
                    )
                    max_equity = portfolio_value_now * max_position_pct
                    available_for_equity = max_equity - current_position_value
                    available_cash = min(cash, available_for_equity) if available_for_equity > 0 else 0
                    buy_slots = cfg.top_n - len(positions)

                    for code in target_codes:
                        if buy_slots <= 0 or available_cash < 10000:
                            break
                        if code in positions or code not in prices or prices[code] <= 0:
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
                    rebalance_meta.append({"date": date, "industries": meta})

            final_value = cash + sum(
                pos["shares"] * prices.get(code, pos.get("current_price", pos["cost_price"]))
                for code, pos in positions.items()
            )
            nav_history.append({"date": date, "nav": final_value})

        nav_df = pd.DataFrame(nav_history)
        nav_df["daily_return"] = nav_df["nav"].pct_change()
        total_return = (nav_df["nav"].iloc[-1] / capital - 1) * 100
        years = len(nav_df) / 252
        annual_return = ((nav_df["nav"].iloc[-1] / capital) ** (1 / years) - 1) * 100
        annual_vol = nav_df["daily_return"].std() * (252 ** 0.5) * 100
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        cummax = nav_df["nav"].cummax()
        drawdown = (nav_df["nav"] - cummax) / cummax
        max_dd = drawdown.min() * 100
        max_dd_date = nav_df.loc[drawdown.idxmin(), "date"] if len(nav_df) > 0 else ""
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
            "rebalance_meta": rebalance_meta,
        }


def main():
    module = load_backtest_module()
    daily, idx, basic, trade_dates = load_dataset(DATASETS["csi1000_5y"], module)
    cfg = [s for s in module.get_strategies() if s.name == "F_三因子+趋势过滤"][0]
    backtest_cls = make_filtered_backtest(module, min_transition_coef=-0.1)
    base_bt = backtest_cls(cfg, daily, idx, basic, trade_dates)
    baseline = base_bt.run(start_offset=250)
    exp_top5 = IndustryAggregatedBacktest(backtest_cls(cfg, daily, idx, basic, trade_dates), top_industries=5).run(start_offset=250)
    exp_top3 = IndustryAggregatedBacktest(backtest_cls(cfg, daily, idx, basic, trade_dates), top_industries=3).run(start_offset=250)
    exp_top5_cap3 = IndustryAggregatedBacktest(backtest_cls(cfg, daily, idx, basic, trade_dates), top_industries=5, per_industry_cap=3).run(start_offset=250)
    out = PROJECT_ROOT / "backtest" / "strategy_f_industry_aggregation_experiments_csi1000_5y.json"
    out.write_text(json.dumps({
        "data_file": str(DATASETS["csi1000_5y"]),
        "results": {
            "baseline": baseline,
            "industry_top5": exp_top5,
            "industry_top3": exp_top3,
            "industry_top5_cap3": exp_top5_cap3,
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "baseline": baseline,
        "industry_top5": {k: v for k, v in exp_top5.items() if k != "rebalance_meta"},
        "industry_top3": {k: v for k, v in exp_top3.items() if k != "rebalance_meta"},
        "industry_top5_cap3": {k: v for k, v in exp_top5_cap3.items() if k != "rebalance_meta"},
    }, ensure_ascii=False, indent=2))
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
