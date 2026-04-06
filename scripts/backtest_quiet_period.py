#!/usr/bin/env python3
"""
回测对比: 宏观静默期 ON vs OFF

v3.1 龙虎榜增强为基准, 测试宏观静默期对收益的影响:
  A) v3.1 无静默期 (enable_macro_calendar=False) — 当前回测基准
  B) v3.1 有静默期 (enable_macro_calendar=True)  — CRITICAL日跳过调仓
  C) v3.1 有静默期 + 空仓豁免              — 空仓时忽略静默期

静默期触发条件 (CRITICAL级):
  - 总影响分 >= 1.5
  - 高影响事件 >= 2个
  - 当天有高影响事件
  → 跳过当日调仓, 仓位上限打5折
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

# 解决 signal 模块冲突
import signal as _builtin_signal
_saved_signal = sys.modules.pop("signal")
sys.modules.pop("signal.signal_generator", None)

from signal.signal_generator import SignalConfig, SignalGenerator
from signal.macro_calendar import MacroCalendar

sys.modules["_builtin_signal"] = _saved_signal


# ══════════════════════════════════════════════════════════════════════════════
#  复用 backtest_v2_vs_v3 的数据加载和引擎
# ══════════════════════════════════════════════════════════════════════════════

# 直接从同目录导入
sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_v2_vs_v3 import load_all_data, BacktestEngine, Position, DailyRecord


class QuietPeriodBacktestEngine(BacktestEngine):
    """扩展回测引擎: 支持宏观静默期"""

    def __init__(self, version: str, data: dict, initial_capital: float = 1_000_000,
                 enable_quiet_period: bool = False,
                 empty_portfolio_bypass: bool = False):
        super().__init__(version, data, initial_capital)
        self.enable_quiet_period = enable_quiet_period
        self.empty_portfolio_bypass = empty_portfolio_bypass
        self.macro_calendar = MacroCalendar() if enable_quiet_period else None
        self.quiet_days = []       # 记录哪些天被静默了
        self.bypassed_days = []    # 记录哪些天被空仓豁免了

    def run(self, start_date: str, end_date: str):
        """运行回测 (含静默期逻辑)"""
        dates = self.data["trade_dates"]
        bt_dates = [d for d in dates if start_date <= d <= end_date]

        label = "有静默期"
        if self.empty_portfolio_bypass:
            label += "+空仓豁免"
        if not self.enable_quiet_period:
            label = "无静默期"
        print(f"\n[回测] {self.version} ({label}): {bt_dates[0]} ~ {bt_dates[-1]}, {len(bt_dates)} 交易日")

        for i, date in enumerate(bt_dates):
            regime = self._check_regime(date)
            trade_count = 0

            # 止损 (每天, 不受静默期影响)
            trade_count += self._check_stop_loss(date)

            # HALT/WAIT: 立即清仓 (不受静默期影响)
            if regime in ("HALT", "WAIT") and self.positions:
                scores = self._score_stocks(date, regime)
                trade_count += self._execute_rebalance(date, scores, regime)
                self._halt_liquidated = True
                self.rebalance_dates.append(date)
            elif regime not in ("HALT", "WAIT"):
                halt_recovery = getattr(self, "_halt_liquidated", False) and not self.positions
                days_since = i - self.last_rebalance_idx
                should_rebalance = (halt_recovery
                                    or days_since >= self.config.rebalance_interval_days
                                    or self.last_rebalance_idx < 0)

                # ── 宏观静默期检查 ──
                if should_rebalance and self.enable_quiet_period and self.macro_calendar:
                    assessment = self.macro_calendar.assess(date)
                    if assessment.quiet_period:
                        # 空仓豁免: 没有持仓时允许建仓
                        if self.empty_portfolio_bypass and not self.positions:
                            self.bypassed_days.append(date)
                            # 仓位折扣仍然生效 — 通过临时修改 config
                            # (不影响, 因为 _execute_rebalance 用自己的逻辑)
                        else:
                            should_rebalance = False
                            self.quiet_days.append(date)

                if should_rebalance:
                    scores = self._score_stocks(date, regime)
                    trade_count += self._execute_rebalance(date, scores, regime)
                    self.last_rebalance_idx = i
                    self.rebalance_dates.append(date)
                    if halt_recovery:
                        self._halt_liquidated = False

            # 更新峰值
            for pos in self.positions.values():
                p = self._get_price(pos.ts_code, date)
                pos.peak_price = max(pos.peak_price, p)

            # 记录
            total = self._total_value(date)
            pos_value = total - self.cash
            self.daily_records.append(DailyRecord(
                date=date,
                total_value=total,
                cash=self.cash,
                position_value=pos_value,
                position_ratio=pos_value / total if total > 0 else 0,
                regime=regime,
                num_positions=len(self.positions),
                trade_count=trade_count,
            ))

        print(f"[回测] 完成: 终值=¥{self.daily_records[-1].total_value:,.0f}, "
              f"调仓{len(self.rebalance_dates)}次, 交易{len(self.trade_log)}笔")
        if self.quiet_days:
            print(f"[回测] 静默跳过: {len(self.quiet_days)}次 → {self.quiet_days}")
        if self.bypassed_days:
            print(f"[回测] 空仓豁免: {len(self.bypassed_days)}次 → {self.bypassed_days}")


# ══════════════════════════════════════════════════════════════════════════════
#  先扫描: 哪些调仓日会被静默期命中
# ══════════════════════════════════════════════════════════════════════════════

def scan_quiet_days(trade_dates, start_date, end_date, rebalance_interval=5):
    """预扫描: 列出所有会被静默的调仓日"""
    cal = MacroCalendar()
    bt_dates = [d for d in trade_dates if start_date <= d <= end_date]

    quiet_hits = []
    elevated_hits = []
    rebalance_idx = 0

    for i, date in enumerate(bt_dates):
        # 模拟调仓间隔
        if i > 0 and (i - rebalance_idx) < rebalance_interval:
            continue
        rebalance_idx = i

        assessment = cal.assess(date)
        if assessment.quiet_period:
            events = [(e["name"], e["impact"], e["distance"]) for e in assessment.nearby_events]
            quiet_hits.append({"date": date, "level": assessment.risk_level, "events": events})
        elif assessment.risk_level == "ELEVATED":
            events = [(e["name"], e["impact"], e["distance"]) for e in assessment.nearby_events]
            elevated_hits.append({"date": date, "level": assessment.risk_level,
                                  "discount": assessment.position_discount, "events": events})

    return quiet_hits, elevated_hits


def main():
    # 数据加载
    csv_lhb = PROJECT_ROOT / "data" / "csi1000_market_bundle_300d_lhb.csv"
    csv_300d = PROJECT_ROOT / "data" / "csi1000_market_bundle_300d.csv"
    if csv_lhb.exists():
        csv_path = str(csv_lhb)
    elif csv_300d.exists():
        csv_path = str(csv_300d)
    else:
        print("找不到数据文件")
        return

    data = load_all_data(csv_path)
    start_date = "20250206"
    end_date = "20260324"

    trade_dates = data["trade_dates"]
    bt_dates = [d for d in trade_dates if start_date <= d <= end_date]

    # ── 预扫描静默日 ──
    print("=" * 90)
    print("  静默期预扫描: 哪些调仓日会被命中")
    print("=" * 90)
    quiet_hits, elevated_hits = scan_quiet_days(trade_dates, start_date, end_date)

    print(f"\nCRITICAL (静默, 跳过调仓): {len(quiet_hits)} 次")
    for q in quiet_hits:
        print(f"  {q['date']}  {q['level']}")
        for name, impact, dist in q["events"]:
            icon = {"high": "🔴", "medium": "🟡", "low": "⚪"}.get(impact, "⚪")
            print(f"    {icon} {dist}: {name}")

    print(f"\nELEVATED (仓位75折, 不跳过): {len(elevated_hits)} 次")
    for e in elevated_hits:
        print(f"  {e['date']}  {e['level']}  仓位折扣={e['discount']:.0%}")

    # ── 三版本回测 ──
    configs = [
        ("A_无静默", False, False),
        ("B_有静默", True, False),
        ("C_静默+空仓豁免", True, True),
    ]

    engines = {}
    metrics = {}

    for label, enable_qp, bypass in configs:
        bt = QuietPeriodBacktestEngine(
            version="v3.1", data=data,
            enable_quiet_period=enable_qp,
            empty_portfolio_bypass=bypass,
        )
        bt.run(start_date, end_date)
        engines[label] = bt
        metrics[label] = bt.calc_metrics()

    # 基准指数
    idx = data["index_data"]
    idx_bt = idx[(idx["trade_date"] >= start_date) & (idx["trade_date"] <= end_date)]
    idx_return = float(idx_bt.iloc[-1]["close"]) / float(idx_bt.iloc[0]["close"]) - 1 if not idx_bt.empty else 0

    # ══════════════════════════════════════════════════════════════════════
    #  输出报告
    # ══════════════════════════════════════════════════════════════════════
    n_days = len(bt_dates)
    print("\n" + "=" * 100)
    print("  回测报告: 宏观静默期影响分析 (v3.1 龙虎榜增强)")
    print(f"  区间: {start_date} ~ {end_date}  |  {n_days}个交易日  |  初始资金: ¥1,000,000")
    print("=" * 100)

    mA, mB, mC = metrics["A_无静默"], metrics["B_有静默"], metrics["C_静默+空仓豁免"]

    print(f"\n{'指标':<24s} {'A 无静默':>14s} {'B 有静默':>14s} {'C 空仓豁免':>14s} {'基准指数':>14s}")
    print("─" * 84)
    print(f"{'累计收益率':<22s} {mA['total_return']:>13.2%} {mB['total_return']:>13.2%} {mC['total_return']:>13.2%} {idx_return:>13.2%}")
    print(f"{'年化收益率':<22s} {mA['annual_return']:>13.2%} {mB['annual_return']:>13.2%} {mC['annual_return']:>13.2%} {'—':>14s}")
    print(f"{'年化波动率':<22s} {mA['annual_vol']:>13.2%} {mB['annual_vol']:>13.2%} {mC['annual_vol']:>13.2%} {'—':>14s}")
    print(f"{'夏普比率':<22s} {mA['sharpe']:>13.2f} {mB['sharpe']:>13.2f} {mC['sharpe']:>13.2f} {'—':>14s}")
    print(f"{'最大回撤':<22s} {mA['max_drawdown']:>13.2%} {mB['max_drawdown']:>13.2%} {mC['max_drawdown']:>13.2%} {'—':>14s}")
    print(f"{'最大回撤日期':<20s} {mA['max_dd_date']:>14s} {mB['max_dd_date']:>14s} {mC['max_dd_date']:>14s} {'—':>14s}")
    print(f"{'日胜率':<22s} {mA['daily_win_rate']:>13.1%} {mB['daily_win_rate']:>13.1%} {mC['daily_win_rate']:>13.1%} {'—':>14s}")
    print(f"{'总交易笔数':<22s} {mA['total_trades']:>13d} {mB['total_trades']:>13d} {mC['total_trades']:>13d} {'—':>14s}")
    print(f"{'调仓次数':<22s} {mA['rebalance_count']:>13d} {mB['rebalance_count']:>13d} {mC['rebalance_count']:>13d} {'—':>14s}")
    print(f"{'期末净值':<22s} ¥{mA['final_value']:>11,.0f} ¥{mB['final_value']:>11,.0f} ¥{mC['final_value']:>11,.0f} {'—':>14s}")

    # 静默日明细
    print(f"\n{'─'*90}")
    print(f"  静默日明细")
    print(f"{'─'*90}")
    bt_b = engines["B_有静默"]
    bt_c = engines["C_静默+空仓豁免"]
    print(f"  B 静默跳过: {len(bt_b.quiet_days)}次")
    for d in bt_b.quiet_days:
        print(f"    {d}")
    print(f"  C 静默跳过: {len(bt_c.quiet_days)}次, 空仓豁免: {len(bt_c.bypassed_days)}次")
    for d in bt_c.quiet_days:
        print(f"    跳过 {d}")
    for d in bt_c.bypassed_days:
        print(f"    豁免 {d} (空仓, 允许建仓)")

    # 净值采样
    print(f"\n{'─'*100}")
    print("  每日净值曲线 (每10天采样)")
    print(f"{'─'*100}")
    print(f"  {'日期':<12s} {'A无静默':>10s} {'B有静默':>10s} {'C空仓豁免':>10s} {'基准':>10s}")

    idx_dict = dict(zip(idx_bt["trade_date"], idx_bt["close"]))
    idx_base = float(idx_bt.iloc[0]["close"]) if not idx_bt.empty else 1

    ra, rb, rc = engines["A_无静默"].daily_records, engines["B_有静默"].daily_records, engines["C_静默+空仓豁免"].daily_records
    cap = engines["A_无静默"].initial_capital
    for i, (a, b, c) in enumerate(zip(ra, rb, rc)):
        if i % 10 != 0 and i != len(ra) - 1:
            continue
        nav_a = a.total_value / cap
        nav_b = b.total_value / cap
        nav_c = c.total_value / cap
        idx_val = idx_dict.get(a.date, idx_base)
        idx_nav = float(idx_val) / idx_base
        quiet_mark = " ★" if a.date in bt_b.quiet_days else ""
        print(f"  {a.date:<12s} {nav_a:>9.4f} {nav_b:>9.4f} {nav_c:>9.4f} {idx_nav:>9.4f}{quiet_mark}")

    # 结论
    diff_ab = mA["annual_return"] - mB["annual_return"]
    diff_ac = mA["annual_return"] - mC["annual_return"]
    print(f"\n{'='*90}")
    print(f"  结论")
    print(f"{'='*90}")
    print(f"  A→B (加静默期): 年化 {diff_ab:+.2%}  夏普 {mA['sharpe']-mB['sharpe']:+.2f}  回撤 {mA['max_drawdown']-mB['max_drawdown']:+.2%}")
    print(f"  A→C (静默+豁免): 年化 {diff_ac:+.2%}  夏普 {mA['sharpe']-mC['sharpe']:+.2f}  回撤 {mA['max_drawdown']-mC['max_drawdown']:+.2%}")
    if mB["sharpe"] > mA["sharpe"]:
        print(f"  → 静默期有正贡献, 建议保留")
    elif mC["sharpe"] > mA["sharpe"]:
        print(f"  → 静默期+空仓豁免最优")
    else:
        print(f"  → 静默期拖累收益, 建议关闭或仅保留空仓豁免")

    # 保存
    result = {
        "backtest_period": f"{start_date}~{end_date}",
        "trading_days": n_days,
        "A_no_quiet": mA,
        "B_with_quiet": mB,
        "C_quiet_bypass": mC,
        "benchmark_return": idx_return,
        "quiet_days_B": bt_b.quiet_days,
        "quiet_days_C": bt_c.quiet_days,
        "bypassed_days_C": bt_c.bypassed_days,
    }
    out_path = PROJECT_ROOT / "data" / "backtest_quiet_period.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    print(f"\n[保存] {out_path}")


if __name__ == "__main__":
    main()
