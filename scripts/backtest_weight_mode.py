#!/usr/bin/env python3
"""
回测对比: 等权 vs 得分加权 分配

实盘复盘发现: TOP5贡献+11.89%, TOP6-10贡献-0.27%
如果按动量得分加权, 强动量股获得更多仓位, 可能放大alpha

测试三种模式:
  A) equal    — 等权 (每只 max_single_weight=15%)
  B) score    — 得分线性加权 (TOP1权重最高, TOP10最低, 总和=max_total_position)
  C) tiered   — 分层加权 (TOP3=18%, TOP4-7=12%, TOP8-10=6%)
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from collections import Counter

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


class WeightModeBacktestEngine(BacktestEngine):
    """扩展回测引擎: 支持多种权重分配模式"""

    def __init__(self, data: dict, weight_mode: str = "equal",
                 initial_capital: float = 1_000_000):
        # 先用v3.1初始化
        super().__init__("v3.1", data, initial_capital)
        self.weight_mode = weight_mode

    def _calc_target_weights(self, scores: list[dict], n: int) -> list[float]:
        """根据模式计算目标权重"""
        cfg = self.config
        max_total = cfg.max_total_position  # 0.80

        if self.weight_mode == "equal":
            w = max_total / n
            return [w] * n

        elif self.weight_mode == "score":
            # 得分线性加权: 按得分比例分配总仓位
            top_scores = [max(s["score"], 0.001) for s in scores[:n]]
            total_score = sum(top_scores)
            if total_score <= 0:
                return [max_total / n] * n
            raw_weights = [s / total_score * max_total for s in top_scores]
            # 单只上限 20%
            capped = [min(w, 0.20) for w in raw_weights]
            # 重新归一化
            cap_sum = sum(capped)
            if cap_sum > 0:
                return [w / cap_sum * max_total for w in capped]
            return [max_total / n] * n

        elif self.weight_mode == "tiered":
            # 分层: TOP3=18%, TOP4-7=12%, TOP8-10=6%
            weights = []
            for i in range(n):
                if i < 3:
                    weights.append(0.18)
                elif i < 7:
                    weights.append(0.10)
                else:
                    weights.append(0.05)
            # 归一化到 max_total_position
            w_sum = sum(weights)
            return [w / w_sum * max_total for w in weights]

        return [max_total / n] * n

    def _execute_rebalance(self, date: str, scores: list[dict], regime: str):
        """执行调仓 (支持多种权重模式)"""
        cfg = self.config
        total_value = self._total_value(date)
        trade_count = 0

        if regime in ("HALT", "WAIT"):
            max_position = 0.0
        elif regime == "DEFENSIVE":
            max_position = 0.5 * cfg.max_total_position
        elif regime == "STRONG_RUN":
            max_position = cfg.max_total_position
        else:
            max_position = cfg.max_total_position

        # HALT/WAIT清仓
        if regime in ("HALT", "WAIT"):
            reason = "HALT清仓" if regime == "HALT" else "WAIT空仓(死叉)"
            for code in list(self.positions.keys()):
                pos = self.positions[code]
                sell_price = self._get_open_price(code, date)
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
                sell_price = self._get_open_price(code, date)
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

        # 计算目标权重
        target_weights = self._calc_target_weights(scores, cfg.top_n)

        # 买入
        hold_count = len(self.positions)
        buy_slots = cfg.top_n - hold_count
        buy_idx = 0  # 跟踪当前是第几个买入位

        scan_range = min(len(scores), cfg.top_n * 3)
        for s in scores[:scan_range]:
            code = s["ts_code"]
            if code in self.positions:
                buy_idx += 1  # 已持有的也算一个排位
                continue
            if buy_slots <= 0:
                break

            buy_price = self._get_open_price(code, date)
            if buy_price <= 0:
                continue

            # 缺口保护
            prev_close = s["close"]
            if prev_close > 0 and cfg.open_gap_limit > 0:
                gap = buy_price / prev_close - 1
                if gap > cfg.open_gap_limit:
                    buy_idx += 1
                    continue

            # 使用排位对应的权重
            weight_idx = min(buy_idx, len(target_weights) - 1)
            single_weight = target_weights[weight_idx]

            target_value = min(total_value * single_weight, self.cash * 0.95)
            current_pos_value = sum(
                p.shares * self._get_price(p.ts_code, date)
                for p in self.positions.values()
            )
            if total_value > 0 and (current_pos_value + target_value) / total_value > max_position:
                target_value = max(0, total_value * max_position - current_pos_value)

            exec_price = buy_price * (1 + cfg.slippage_pct)
            shares = int(target_value / exec_price / 100) * 100

            if shares >= 100 and shares * exec_price >= cfg.min_position_amount:
                cost = shares * exec_price
                commission = max(cost * cfg.commission_rate, 5)
                self.cash -= (cost + commission)

                self.positions[code] = Position(
                    ts_code=code,
                    name=s["name"],
                    shares=shares,
                    cost_price=exec_price,
                    entry_date=date,
                    peak_price=buy_price,
                )
                buy_slots -= 1
                trade_count += 1

                self.trade_log.append({
                    "date": date, "action": "BUY", "code": code,
                    "name": s["name"], "shares": shares, "price": exec_price,
                    "score": round(s["score"], 4),
                    "weight": f"{single_weight:.1%}",
                    "reason": f"排名#{scores.index(s)+1} 权重{single_weight:.1%}",
                })

            buy_idx += 1

        return trade_count


def main():
    csv_lhb = PROJECT_ROOT / "data" / "csi1000_market_bundle_300d_lhb.csv"
    csv_300d = PROJECT_ROOT / "data" / "csi1000_market_bundle_300d.csv"
    csv_path = str(csv_lhb) if csv_lhb.exists() else str(csv_300d)
    data = load_all_data(csv_path)

    start_date = "20250206"
    end_date = "20260324"
    trade_dates = data["trade_dates"]
    bt_dates = [d for d in trade_dates if start_date <= d <= end_date]

    modes = ["equal", "score", "tiered"]
    labels = {"equal": "A 等权", "score": "B 得分加权", "tiered": "C 分层加权"}
    engines = {}
    metrics = {}

    for mode in modes:
        bt = WeightModeBacktestEngine(data, weight_mode=mode)
        bt.run(start_date, end_date)
        engines[mode] = bt
        metrics[mode] = bt.calc_metrics()

    # 基准
    idx = data["index_data"]
    idx_bt = idx[(idx["trade_date"] >= start_date) & (idx["trade_date"] <= end_date)]
    idx_return = float(idx_bt.iloc[-1]["close"]) / float(idx_bt.iloc[0]["close"]) - 1 if not idx_bt.empty else 0

    n_days = len(bt_dates)
    print("\n" + "=" * 100)
    print("  回测报告: 权重分配模式对比 (v3.1 龙虎榜增强)")
    print(f"  区间: {start_date} ~ {end_date}  |  {n_days}个交易日  |  初始资金: ¥1,000,000")
    print("=" * 100)

    mA, mB, mC = metrics["equal"], metrics["score"], metrics["tiered"]

    print(f"\n  权重说明:")
    print(f"    A 等权:    每只 8.0% (10只×8%=80%)")
    print(f"    B 得分加权: 按动量得分比例, 单只上限20%, 总仓位80%")
    print(f"    C 分层:    TOP3=18%, TOP4-7=10%, TOP8-10=5%, 归一化到80%")

    print(f"\n{'指标':<24s} {'A 等权':>14s} {'B 得分加权':>14s} {'C 分层加权':>14s} {'基准指数':>14s}")
    print("─" * 84)
    print(f"{'累计收益率':<22s} {mA['total_return']:>13.2%} {mB['total_return']:>13.2%} {mC['total_return']:>13.2%} {idx_return:>13.2%}")
    print(f"{'年化收益率':<22s} {mA['annual_return']:>13.2%} {mB['annual_return']:>13.2%} {mC['annual_return']:>13.2%} {'—':>14s}")
    print(f"{'年化波动率':<22s} {mA['annual_vol']:>13.2%} {mB['annual_vol']:>13.2%} {mC['annual_vol']:>13.2%} {'—':>14s}")
    print(f"{'夏普比率':<22s} {mA['sharpe']:>13.2f} {mB['sharpe']:>13.2f} {mC['sharpe']:>13.2f} {'—':>14s}")
    print(f"{'最大回撤':<22s} {mA['max_drawdown']:>13.2%} {mB['max_drawdown']:>13.2%} {mC['max_drawdown']:>13.2%} {'—':>14s}")
    print(f"{'日胜率':<22s} {mA['daily_win_rate']:>13.1%} {mB['daily_win_rate']:>13.1%} {mC['daily_win_rate']:>13.1%} {'—':>14s}")
    print(f"{'总交易笔数':<22s} {mA['total_trades']:>13d} {mB['total_trades']:>13d} {mC['total_trades']:>13d} {'—':>14s}")
    print(f"{'期末净值':<22s} ¥{mA['final_value']:>11,.0f} ¥{mB['final_value']:>11,.0f} ¥{mC['final_value']:>11,.0f} {'—':>14s}")

    # 净值采样
    print(f"\n{'─'*90}")
    print("  每日净值曲线 (每10天)")
    print(f"{'─'*90}")
    print(f"  {'日期':<12s} {'A等权':>10s} {'B得分':>10s} {'C分层':>10s} {'基准':>10s}")
    idx_dict = dict(zip(idx_bt["trade_date"], idx_bt["close"]))
    idx_base = float(idx_bt.iloc[0]["close"]) if not idx_bt.empty else 1
    ra, rb, rc = engines["equal"].daily_records, engines["score"].daily_records, engines["tiered"].daily_records
    cap = engines["equal"].initial_capital
    for i, (a, b, c) in enumerate(zip(ra, rb, rc)):
        if i % 10 != 0 and i != len(ra) - 1:
            continue
        print(f"  {a.date:<12s} {a.total_value/cap:>9.4f} {b.total_value/cap:>9.4f} {c.total_value/cap:>9.4f} {float(idx_dict.get(a.date, idx_base))/idx_base:>9.4f}")

    # 结论
    best = max(metrics.items(), key=lambda x: x[1]["sharpe"])
    print(f"\n{'='*90}")
    print(f"  结论: {labels[best[0]]} 夏普最优 ({best[1]['sharpe']:.2f})")
    print(f"{'='*90}")
    for mode in modes:
        m = metrics[mode]
        print(f"  {labels[mode]}: 年化{m['annual_return']:.2%}  夏普{m['sharpe']:.2f}  回撤{m['max_drawdown']:.2%}  交易{m['total_trades']}笔")

    # 保存
    result = {
        "backtest_period": f"{start_date}~{end_date}",
        "trading_days": n_days,
    }
    for mode in modes:
        result[mode] = metrics[mode]
    out_path = PROJECT_ROOT / "data" / "backtest_weight_mode.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    print(f"\n[保存] {out_path}")


if __name__ == "__main__":
    main()
