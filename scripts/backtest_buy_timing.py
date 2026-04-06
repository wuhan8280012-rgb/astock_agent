#!/usr/bin/env python3
"""
回测对比: 买入时点优化

现有: 周五收盘后生成信号 → 周一开盘买入 (承受周末缺口)
改进: 周五14:30生成信号 → 周五尾盘买入 (无缺口)

用日线数据模拟:
  A) next_open   — 现有逻辑: 信号日评分, 下一交易日开盘价买入
  B) same_close  — 新逻辑:   前一日数据评分, 信号日收盘价买入
                   (模拟14:30评分+尾盘买入, 收盘价≈14:30后成交均价)
  C) same_vwap   — 前一日评分, 信号日VWAP买入 (H+L+C)/3
                   (模拟14:30评分+14:30-15:00区间成交)

关键区别:
  A: 评分用 rebalance_day 数据, 买入用 next_day open → 有周末缺口
  B: 评分用 prev_day 数据, 买入用 rebalance_day close → 无缺口
  C: 评分用 prev_day 数据, 买入用 rebalance_day VWAP → 无缺口
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import signal as _builtin_signal
_saved_signal = sys.modules.pop("signal")
sys.modules.pop("signal.signal_generator", None)
from signal.signal_generator import SignalConfig, SignalGenerator
sys.modules["_builtin_signal"] = _saved_signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_v2_vs_v3 import load_all_data, BacktestEngine, Position, DailyRecord


class BuyTimingEngine(BacktestEngine):
    """买入时点回测引擎"""

    def __init__(self, data: dict, timing_mode: str = "next_open",
                 initial_capital: float = 1_000_000):
        super().__init__("v3.1", data, initial_capital)
        self.timing_mode = timing_mode
        self.skip_count = 0  # 跳过买入计数 (缺口保护触发)

    def _get_prev_date(self, date: str) -> str:
        """获取前一个交易日"""
        dates = self.data["trade_dates"]
        try:
            idx = dates.index(date)
            return dates[idx - 1] if idx > 0 else date
        except ValueError:
            return date

    def _score_stocks_as_of(self, date: str, regime: str) -> list[dict]:
        """用指定日期的数据评分 (可以是前一日)"""
        cfg = self.config
        gen = self.generator
        stock_data = self.data["stock_data"]
        stock_info = self.data["stock_info"]

        candidates = gen._filter_universe(stock_data, stock_info, date)
        weights = gen._get_regime_weights(regime)

        scores = []
        for ts_code in candidates:
            df = stock_data[ts_code]
            hist = df[df["trade_date"] <= date]
            closes = hist["close"].values.astype(float)
            avg_amount = hist.tail(20)["amount"].mean() * 1000

            score = gen._calc_composite_score(closes, avg_amount, weights=weights)
            if not np.isnan(score):
                scores.append({
                    "ts_code": ts_code,
                    "name": self.data["name_map"].get(ts_code, ts_code),
                    "score": score,
                    "close": float(closes[-1]),
                })

        # LHB加分
        if cfg.enable_lhb_factor:
            lhb_data = self.data.get("lhb_data", {})
            trade_dates = self.data["trade_dates"]
            if lhb_data:
                for s in scores:
                    net_buy, count = gen._calc_lhb_signal(
                        s["ts_code"], date, lhb_data, cfg.lhb_lookback, trade_dates
                    )
                    if count > 0 and net_buy > 0:
                        boost = 1 + cfg.lhb_weight * min(count / 2, 1.0)
                        s["score"] *= boost
                    elif count > 0 and net_buy < 0:
                        s["score"] *= (1 - cfg.lhb_weight * cfg.lhb_negative_penalty)

        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores

    def _get_buy_price(self, ts_code: str, date: str) -> float:
        """根据timing_mode获取买入价"""
        if self.timing_mode == "next_open":
            # 下一交易日开盘价
            dates = self.data["trade_dates"]
            try:
                idx = dates.index(date)
                if idx + 1 < len(dates):
                    next_date = dates[idx + 1]
                    return self._get_open_price(ts_code, next_date)
            except ValueError:
                pass
            return self._get_open_price(ts_code, date)

        elif self.timing_mode == "same_close":
            # 当日收盘价
            return self._get_price(ts_code, date)

        elif self.timing_mode == "same_vwap":
            # 当日VWAP近似
            df = self.data["stock_data"].get(ts_code)
            if df is None:
                return 0
            row = df[df["trade_date"] == date]
            if row.empty:
                return self._get_price(ts_code, date)
            r = row.iloc[0]
            h, l, c = float(r["high"]), float(r["low"]), float(r["close"])
            return (h + l + c) / 3

        return self._get_open_price(ts_code, date)

    def _get_sell_price(self, ts_code: str, date: str) -> float:
        """卖出价: 统一用当日收盘价 (尾盘卖出)"""
        if self.timing_mode == "next_open":
            return self._get_open_price(ts_code, date)
        else:
            # same_close / same_vwap: 尾盘卖出 = 收盘价
            return self._get_price(ts_code, date)

    def _execute_rebalance(self, date: str, scores: list[dict], regime: str):
        """执行调仓"""
        cfg = self.config
        total_value = self._total_value(date)
        trade_count = 0

        if regime in ("HALT", "WAIT"):
            max_position = 0.0
        elif regime == "DEFENSIVE":
            max_position = 0.5 * cfg.max_total_position
        else:
            max_position = cfg.max_total_position

        # HALT/WAIT清仓
        if regime in ("HALT", "WAIT"):
            reason = "HALT清仓" if regime == "HALT" else "WAIT空仓"
            for code in list(self.positions.keys()):
                pos = self.positions[code]
                sell_price = self._get_sell_price(code, date)
                proceeds = pos.shares * sell_price
                commission = max(proceeds * cfg.commission_rate, 5)
                stamp_tax = proceeds * cfg.stamp_tax_rate
                self.cash += proceeds - commission - stamp_tax
                self.trade_log.append({
                    "date": date, "action": "SELL", "code": code,
                    "name": pos.name, "shares": pos.shares, "price": sell_price,
                    "reason": reason,
                })
                del self.positions[code]
                trade_count += 1
            return trade_count

        buffer_codes = [s["ts_code"] for s in scores[:int(cfg.top_n * cfg.hold_buffer_ratio)]]

        # 卖出
        for code in list(self.positions.keys()):
            if code not in buffer_codes:
                pos = self.positions[code]
                sell_price = self._get_sell_price(code, date)
                proceeds = pos.shares * sell_price
                commission = max(proceeds * cfg.commission_rate, 5)
                stamp_tax = proceeds * cfg.stamp_tax_rate
                self.cash += proceeds - commission - stamp_tax
                self.trade_log.append({
                    "date": date, "action": "SELL", "code": code,
                    "name": pos.name, "shares": pos.shares, "price": sell_price,
                    "reason": "出缓冲带",
                })
                del self.positions[code]
                trade_count += 1

        # 买入
        hold_count = len(self.positions)
        buy_slots = cfg.top_n - hold_count

        scan_range = min(len(scores), cfg.top_n * 3)
        for s in scores[:scan_range]:
            code = s["ts_code"]
            if code in self.positions:
                continue
            if buy_slots <= 0:
                break

            buy_price = self._get_buy_price(code, date)
            if buy_price <= 0:
                continue

            # 缺口保护 (仅 next_open 模式)
            if self.timing_mode == "next_open" and cfg.open_gap_limit > 0:
                prev_close = s["close"]
                if prev_close > 0:
                    gap = buy_price / prev_close - 1
                    if gap > cfg.open_gap_limit:
                        self.skip_count += 1
                        continue

            exec_price = buy_price * (1 + cfg.slippage_pct)
            single_weight = max_position / cfg.top_n
            target_value = min(total_value * single_weight, self.cash * 0.95)
            current_pos_value = sum(
                p.shares * self._get_price(p.ts_code, date)
                for p in self.positions.values()
            )
            if total_value > 0 and (current_pos_value + target_value) / total_value > max_position:
                target_value = max(0, total_value * max_position - current_pos_value)

            shares = int(target_value / exec_price / 100) * 100
            if shares >= 100 and shares * exec_price >= cfg.min_position_amount:
                cost = shares * exec_price
                commission = max(cost * cfg.commission_rate, 5)
                self.cash -= (cost + commission)
                self.positions[code] = Position(
                    ts_code=code, name=s["name"], shares=shares,
                    cost_price=exec_price, entry_date=date,
                    peak_price=buy_price,
                )
                buy_slots -= 1
                trade_count += 1
                self.trade_log.append({
                    "date": date, "action": "BUY", "code": code,
                    "name": s["name"], "shares": shares, "price": exec_price,
                    "reason": f"排名#{scores.index(s)+1}",
                })

        return trade_count

    def run(self, start_date: str, end_date: str):
        """运行回测"""
        dates = self.data["trade_dates"]
        bt_dates = [d for d in dates if start_date <= d <= end_date]

        label = {"next_open": "周一开盘买", "same_close": "周五收盘买", "same_vwap": "周五VWAP买"}.get(self.timing_mode, self.timing_mode)
        print(f"\n[回测] v3.1 ({label}): {bt_dates[0]} ~ {bt_dates[-1]}, {len(bt_dates)} 交易日")

        for i, date in enumerate(bt_dates):
            regime = self._check_regime(date)
            trade_count = 0
            trade_count += self._check_stop_loss(date)

            if regime in ("HALT", "WAIT") and self.positions:
                scores = self._score_stocks(date, regime)
                trade_count += self._execute_rebalance(date, scores, regime)
                self._halt_liquidated = True
                self.rebalance_dates.append(date)
            elif regime not in ("HALT", "WAIT"):
                halt_recovery = getattr(self, "_halt_liquidated", False) and not self.positions
                days_since = i - self.last_rebalance_idx
                if (halt_recovery
                        or days_since >= self.config.rebalance_interval_days
                        or self.last_rebalance_idx < 0):

                    if self.timing_mode == "next_open":
                        # 现有逻辑: 当日数据评分, 下一日开盘买
                        scores = self._score_stocks(date, regime)
                    else:
                        # 新逻辑: 前一日数据评分, 当日收盘/VWAP买
                        # (模拟周五14:30用截至前一日的完整数据评分)
                        prev_date = self._get_prev_date(date)
                        scores = self._score_stocks_as_of(prev_date, regime)

                    trade_count += self._execute_rebalance(date, scores, regime)
                    self.last_rebalance_idx = i
                    self.rebalance_dates.append(date)
                    if halt_recovery:
                        self._halt_liquidated = False

            for pos in self.positions.values():
                p = self._get_price(pos.ts_code, date)
                pos.peak_price = max(pos.peak_price, p)

            total = self._total_value(date)
            pos_value = total - self.cash
            self.daily_records.append(DailyRecord(
                date=date, total_value=total, cash=self.cash,
                position_value=pos_value,
                position_ratio=pos_value / total if total > 0 else 0,
                regime=regime, num_positions=len(self.positions),
                trade_count=trade_count,
            ))

        print(f"[回测] 完成: 终值=¥{self.daily_records[-1].total_value:,.0f}, "
              f"调仓{len(self.rebalance_dates)}次, 交易{len(self.trade_log)}笔")
        if self.skip_count:
            print(f"[回测] 缺口保护跳过: {self.skip_count}笔")


def main():
    csv_lhb = PROJECT_ROOT / "data" / "csi1000_market_bundle_300d_lhb.csv"
    csv_300d = PROJECT_ROOT / "data" / "csi1000_market_bundle_300d.csv"
    csv_path = str(csv_lhb) if csv_lhb.exists() else str(csv_300d)
    data = load_all_data(csv_path)

    start_date = "20250206"
    end_date = "20260324"
    bt_dates = [d for d in data["trade_dates"] if start_date <= d <= end_date]

    modes = [
        ("A_周一开盘买", "next_open"),
        ("B_周五收盘买", "same_close"),
        ("C_周五VWAP买", "same_vwap"),
    ]

    engines = {}
    metrics = {}
    for label, mode in modes:
        bt = BuyTimingEngine(data, timing_mode=mode)
        bt.run(start_date, end_date)
        engines[label] = bt
        metrics[label] = bt.calc_metrics()

    # 基准
    idx = data["index_data"]
    idx_bt = idx[(idx["trade_date"] >= start_date) & (idx["trade_date"] <= end_date)]
    idx_return = float(idx_bt.iloc[-1]["close"]) / float(idx_bt.iloc[0]["close"]) - 1 if not idx_bt.empty else 0

    n_days = len(bt_dates)
    print("\n" + "=" * 100)
    print("  回测报告: 买入时点优化 (v3.1 龙虎榜增强)")
    print(f"  区间: {start_date} ~ {end_date}  |  {n_days}个交易日  |  初始资金: ¥1,000,000")
    print("=" * 100)

    print(f"\n  逻辑说明:")
    print(f"    A 周一开盘买: 周五收盘数据评分 → 周一开盘价买入 (有周末缺口)")
    print(f"    B 周五收盘买: 周四收盘数据评分 → 周五收盘价买入 (无缺口, 模拟14:30信号+尾盘执行)")
    print(f"    C 周五VWAP买: 周四收盘数据评分 → 周五VWAP买入 (无缺口, 模拟14:30后分批成交)")

    labels = [m[0] for m in modes]
    mA, mB, mC = [metrics[lb] for lb in labels]

    print(f"\n{'指标':<24s} {'A 周一开盘':>14s} {'B 周五收盘':>14s} {'C 周五VWAP':>14s} {'基准指数':>14s}")
    print("─" * 84)
    print(f"{'累计收益率':<22s} {mA['total_return']:>13.2%} {mB['total_return']:>13.2%} {mC['total_return']:>13.2%} {idx_return:>13.2%}")
    print(f"{'年化收益率':<22s} {mA['annual_return']:>13.2%} {mB['annual_return']:>13.2%} {mC['annual_return']:>13.2%} {'—':>14s}")
    print(f"{'年化波动率':<22s} {mA['annual_vol']:>13.2%} {mB['annual_vol']:>13.2%} {mC['annual_vol']:>13.2%} {'—':>14s}")
    print(f"{'夏普比率':<22s} {mA['sharpe']:>13.2f} {mB['sharpe']:>13.2f} {mC['sharpe']:>13.2f} {'—':>14s}")
    print(f"{'最大回撤':<22s} {mA['max_drawdown']:>13.2%} {mB['max_drawdown']:>13.2%} {mC['max_drawdown']:>13.2%} {'—':>14s}")
    print(f"{'最大回撤日期':<20s} {mA['max_dd_date']:>14s} {mB['max_dd_date']:>14s} {mC['max_dd_date']:>14s} {'—':>14s}")
    print(f"{'日胜率':<22s} {mA['daily_win_rate']:>13.1%} {mB['daily_win_rate']:>13.1%} {mC['daily_win_rate']:>13.1%} {'—':>14s}")
    print(f"{'总交易笔数':<22s} {mA['total_trades']:>13d} {mB['total_trades']:>13d} {mC['total_trades']:>13d} {'—':>14s}")
    print(f"{'期末净值':<22s} ¥{mA['final_value']:>11,.0f} ¥{mB['final_value']:>11,.0f} ¥{mC['final_value']:>11,.0f} {'—':>14s}")

    # 净值采样
    print(f"\n{'─'*90}")
    print("  每日净值曲线 (每10天)")
    print(f"{'─'*90}")
    print(f"  {'日期':<12s} {'A周一开盘':>10s} {'B周五收盘':>10s} {'C周五VWAP':>10s} {'基准':>10s}")
    idx_dict = dict(zip(idx_bt["trade_date"], idx_bt["close"]))
    idx_base = float(idx_bt.iloc[0]["close"]) if not idx_bt.empty else 1
    ra, rb, rc = engines[labels[0]].daily_records, engines[labels[1]].daily_records, engines[labels[2]].daily_records
    cap = 1_000_000
    for i, (a, b, c) in enumerate(zip(ra, rb, rc)):
        if i % 10 != 0 and i != len(ra) - 1:
            continue
        print(f"  {a.date:<12s} {a.total_value/cap:>9.4f} {b.total_value/cap:>9.4f} {c.total_value/cap:>9.4f} {float(idx_dict.get(a.date, idx_base))/idx_base:>9.4f}")

    # 结论
    best = max(metrics.items(), key=lambda x: x[1]["sharpe"])
    print(f"\n{'='*90}")
    print(f"  结论")
    print(f"{'='*90}")
    for lb in labels:
        m = metrics[lb]
        print(f"  {lb}: 年化{m['annual_return']:>7.2%}  夏普{m['sharpe']:>5.2f}  回撤{m['max_drawdown']:>7.2%}")

    diff_ab = mB["annual_return"] - mA["annual_return"]
    diff_sharpe = mB["sharpe"] - mA["sharpe"]
    print(f"\n  A→B: 年化{diff_ab:+.2%}  夏普{diff_sharpe:+.2f}")
    if mB["sharpe"] > mA["sharpe"]:
        print(f"  → 周五尾盘买入优于周一开盘, 建议采用")
    else:
        print(f"  → 周一开盘仍优, 保持现有逻辑")

    # 保存
    result = {"backtest_period": f"{start_date}~{end_date}", "trading_days": n_days}
    for lb in labels:
        result[lb] = metrics[lb]
    out_path = PROJECT_ROOT / "data" / "backtest_buy_timing.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    print(f"\n[保存] {out_path}")


if __name__ == "__main__":
    main()
