#!/usr/bin/env python3
"""
回测对比: 修复 look-ahead bias 后的多模式对比

模式说明:
  [A] 每5交易日调仓 (T-1评分, T日open执行) — 修复后的基准引擎
  [B] 周五决策 → 周一执行 (周五close评分, 周一open执行) — 当前生产流程
  [C] 每日信号 → 次日执行 (T-1 close评分, T日open执行) — 建议的新生产流程
"""

import sys, datetime
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

import numpy as np
from copy import deepcopy
from collections import Counter
from scripts.backtest_v2_vs_v3 import load_all_data, BacktestEngine, DailyRecord


def find_data_file(project_root: Path) -> Path:
    data_dir = project_root / "data"
    for name in ["csi1000_market_bundle_700d.csv", "csi1000_market_bundle_300d_lhb.csv",
                  "csi1000_market_bundle_300d.csv", "csi1000_market_bundle_100d.csv",
                  "csi1000_market_bundle.csv"]:
        p = data_dir / name
        if p.exists():
            print(f"[数据] 使用: {p.name}")
            return p
    raise FileNotFoundError("找不到数据文件")


def get_weekday(date_str: str) -> int:
    return datetime.datetime.strptime(date_str, "%Y%m%d").weekday()


class FridayMondayEngine(BacktestEngine):
    """
    周五收盘评分 → 周一开盘执行
    注意: 评分用周五数据(已知), 无 look-ahead bias
    """

    def run(self, start_date: str, end_date: str):
        dates = self.data["trade_dates"]
        bt_dates = [d for d in dates if start_date <= d <= end_date]
        date_idx = {d: i for i, d in enumerate(dates)}

        friday_to_next = {}
        for d in bt_dates:
            if get_weekday(d) == 4:
                idx = date_idx[d]
                if idx + 1 < len(dates):
                    friday_to_next[d] = dates[idx + 1]

        print(f"\n[周五→周一] {bt_dates[0]} ~ {bt_dates[-1]}, {len(bt_dates)} 交易日")
        print(f"  周五调仓信号: {len(friday_to_next)} 次")

        friday_set = set(friday_to_next.keys())
        exec_day_map = {}

        for i, date in enumerate(bt_dates):
            regime = self._check_regime(date)
            trade_count = 0

            trade_count += self._check_stop_loss(date)

            if regime in ("HALT", "WAIT") and self.positions:
                # HALT紧急清仓: 用前一日数据评分
                prev_idx = date_idx.get(date, 0) - 1
                prev_d = dates[prev_idx] if prev_idx >= 0 else date
                scores = self._score_stocks(prev_d, regime)
                trade_count += self._execute_rebalance(date, scores, regime)
                self._halt_liquidated = True
                self.rebalance_dates.append(date)
            elif date in friday_set and regime not in ("HALT", "WAIT"):
                # 周五: 用当天收盘数据评分 (已知, 无look-ahead)
                scores = self._score_stocks(date, regime)
                next_day = friday_to_next[date]
                exec_day_map[next_day] = (date, scores, regime)

            if date in exec_day_map and regime not in ("HALT", "WAIT"):
                signal_fri, scores, sig_regime = exec_day_map.pop(date)
                trade_count += self._execute_rebalance(date, scores, sig_regime)
                self.last_rebalance_idx = i
                self.rebalance_dates.append(date)

            for pos in self.positions.values():
                p = self._get_price(pos.ts_code, date)
                pos.peak_price = max(pos.peak_price, p)

            total = self._total_value(date)
            pos_value = total - self.cash
            industries = [pos.industry for pos in self.positions.values() if pos.industry]

            self.daily_records.append(DailyRecord(
                date=date, total_value=total, cash=self.cash,
                position_value=pos_value,
                position_ratio=pos_value / total if total > 0 else 0,
                regime=regime, num_positions=len(self.positions),
                trade_count=trade_count, industries=industries,
            ))

        final_val = self.daily_records[-1].total_value if self.daily_records else self.cash
        print(f"[回测] 完成: 终值=¥{final_val:,.0f}, 调仓{len(self.rebalance_dates)}次, 交易{len(self.trade_log)}笔")


class DailyEngine(BacktestEngine):
    """
    每日信号 → 次日执行 (rebalance_interval_days=1)
    T-1 close 评分, T open 执行 (与修复后的标准引擎相同逻辑, 但每天都调仓)
    """

    def run(self, start_date: str, end_date: str):
        dates = self.data["trade_dates"]
        bt_dates = [d for d in dates if start_date <= d <= end_date]
        all_date_idx = {d: i for i, d in enumerate(dates)}

        print(f"\n[每日调仓] {bt_dates[0]} ~ {bt_dates[-1]}, {len(bt_dates)} 交易日")

        for i, date in enumerate(bt_dates):
            regime = self._check_regime(date)
            trade_count = 0

            global_idx = all_date_idx.get(date, 0)
            prev_date = dates[global_idx - 1] if global_idx > 0 else date

            trade_count += self._check_stop_loss(date)

            if regime in ("HALT", "WAIT") and self.positions:
                scores = self._score_stocks(prev_date, regime)
                trade_count += self._execute_rebalance(date, scores, regime)
                self._halt_liquidated = True
                self.rebalance_dates.append(date)
            elif regime not in ("HALT", "WAIT"):
                halt_recovery = getattr(self, "_halt_liquidated", False) and not self.positions
                # 每天都调仓
                scores = self._score_stocks(prev_date, regime)
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
            industries = [pos.industry for pos in self.positions.values() if pos.industry]

            self.daily_records.append(DailyRecord(
                date=date, total_value=total, cash=self.cash,
                position_value=pos_value,
                position_ratio=pos_value / total if total > 0 else 0,
                regime=regime, num_positions=len(self.positions),
                trade_count=trade_count, industries=industries,
            ))

        final_val = self.daily_records[-1].total_value if self.daily_records else self.cash
        print(f"[回测] 完成: 终值=¥{final_val:,.0f}, 调仓{len(self.rebalance_dates)}次, 交易{len(self.trade_log)}笔")


def make_v32_config():
    from signal.signal_generator import SignalConfig
    return SignalConfig(
        data_csv_path="__backtest__",
        adaptive_weights=False,
        enable_reversal_filter=False,
        enable_trend_window=False,
        stop_loss_pct=-0.99,
        lookback_weights=[0.5, 0.3, 0.2],
        enable_lhb_factor=True,
        lhb_weight=0.30,
        lhb_lookback=10,
        lhb_negative_penalty=0.3,
        enable_macro_calendar=False,
        enable_breaking_monitor=False,
        liquidity_weight=0.15,
        hold_buffer_ratio=1.2,
    )


def run_comparison():
    data_file = find_data_file(project_root)
    data = load_all_data(str(data_file))

    trade_dates = data["trade_dates"]
    start_date = trade_dates[25] if len(trade_dates) > 30 else trade_dates[0]
    end_date = trade_dates[-1]

    n_bt_days = len([d for d in trade_dates if start_date <= d <= end_date])
    print(f"\n{'='*70}")
    print(f"  回测区间: {start_date} ~ {end_date}")
    print(f"  交易日数: {n_bt_days}")
    print(f"  ⚠️  所有模式均已修复 look-ahead bias")
    print(f"{'='*70}")

    results = {}

    # ── [A] 每5交易日, T-1评分 T日执行 (修复后) ──
    print(f"\n{'─'*70}")
    print("  [A] 每5交易日调仓 (T-1评分, T日open执行)")
    print(f"{'─'*70}")
    bt_a = BacktestEngine("v3.1", deepcopy(data), custom_config=make_v32_config())
    bt_a.run(start_date, end_date)
    results["A"] = bt_a.calc_metrics()
    results["A"]["trade_log"] = bt_a.trade_log

    # ── [B] 周五→周一 ──
    print(f"\n{'─'*70}")
    print("  [B] 周五决策 → 周一执行 (当前生产流程)")
    print(f"{'─'*70}")
    bt_b = FridayMondayEngine("v3.1", deepcopy(data), custom_config=make_v32_config())
    bt_b.run(start_date, end_date)
    results["B"] = bt_b.calc_metrics()
    results["B"]["trade_log"] = bt_b.trade_log

    # ── [C] 每日信号, 次日执行 ──
    print(f"\n{'─'*70}")
    print("  [C] 每日信号 → 次日执行 (建议的生产流程)")
    print(f"{'─'*70}")
    bt_c = DailyEngine("v3.1", deepcopy(data), custom_config=make_v32_config())
    bt_c.run(start_date, end_date)
    results["C"] = bt_c.calc_metrics()
    results["C"]["trade_log"] = bt_c.trade_log

    # ── 报告 ──
    print(f"\n{'='*70}")
    print("  📊 修复 look-ahead bias 后的三方对比")
    print(f"{'='*70}")
    labels = {
        "A": "[A]每5日",
        "B": "[B]周五→周一",
        "C": "[C]每日→次日",
    }
    print(f"  {'指标':<16} {labels['A']:>14} {labels['B']:>14} {labels['C']:>14}")
    print(f"  {'─'*58}")

    for label, key, fmt in [
        ("总收益率", "total_return", "%"),
        ("年化收益率", "annual_return", "%"),
        ("Sharpe", "sharpe", ""),
        ("最大回撤", "max_drawdown", "%"),
        ("年化波动率", "annual_vol", "%"),
        ("交易次数", "n_trades", "d"),
        ("调仓次数", "n_rebalances", "d"),
        ("胜率", "win_rate", "%"),
    ]:
        vals = {}
        for k in ["A", "B", "C"]:
            vals[k] = results[k].get(key, 0) or 0

        if fmt == "%":
            strs = {k: f"{v*100:+.2f}%" for k, v in vals.items()}
        elif fmt == "d":
            strs = {k: str(int(v)) for k, v in vals.items()}
        else:
            strs = {k: f"{v:.2f}" for k, v in vals.items()}

        print(f"  {label:<16} {strs['A']:>14} {strs['B']:>14} {strs['C']:>14}")

    for k in ["A", "B", "C"]:
        fv = results[k].get("final_value", 0)
        print(f"\n  {labels[k]} 最终净值: ¥{fv:,.0f}")

    # 交易成本分析
    print(f"\n{'─'*70}")
    print("  📋 交易频率分析")
    print(f"{'─'*70}")
    for k in ["A", "B", "C"]:
        tl = results[k].get("trade_log", [])
        buys = [t for t in tl if t["action"] == "BUY"]
        sells = [t for t in tl if t["action"] == "SELL"]
        total_commission = 0
        for t in tl:
            shares = t.get("shares", 0)
            price = t.get("price", 0)
            value = shares * price
            total_commission += max(value * 0.00025, 5)
            if t["action"] == "SELL":
                total_commission += value * 0.001  # 印花税

        avg_hold_days = 0
        if buys and results[k].get("n_rebalances", 0) > 0:
            avg_hold_days = n_bt_days / max(results[k].get("n_rebalances", 1), 1)

        print(f"  {labels[k]}: {len(buys)}买 + {len(sells)}卖 = {len(tl)}笔")
        print(f"    平均调仓间隔: {avg_hold_days:.1f} 交易日 | 估算总交易成本: ¥{total_commission:,.0f}")

    # 解读
    print(f"\n{'='*70}")
    print("  💡 解读")
    print(f"{'='*70}")
    ra = results["A"].get("total_return", 0) * 100
    rb = results["B"].get("total_return", 0) * 100
    rc = results["C"].get("total_return", 0) * 100
    print(f"  [A] 每5日调仓 (修复后): {ra:+.2f}%")
    print(f"  [B] 周五→周一 (生产流程): {rb:+.2f}%")
    print(f"  [C] 每日→次日 (建议方案): {rc:+.2f}%")
    print(f"  [B vs A] 差异: {rb-ra:+.2f}%")
    print(f"  [C vs A] 差异: {rc-ra:+.2f}%")
    print(f"  [C vs B] 差异: {rc-rb:+.2f}%")


if __name__ == "__main__":
    run_comparison()
