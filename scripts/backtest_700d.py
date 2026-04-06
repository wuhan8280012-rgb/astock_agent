#!/usr/bin/env python3
"""
700天数据全面验证:
1. 因子有效性: 每5日调仓(5个偏移量) → 平均收益和标准差
2. 调仓日敏感性: 固定周几 (真实生产流程: 周X信号→次日执行)
3. 分段验证: 前350天 vs 后350天, 因子是否稳定
"""

import sys, datetime, time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

import numpy as np
from copy import deepcopy
from collections import Counter
from scripts.backtest_v2_vs_v3 import load_all_data, BacktestEngine, DailyRecord


def get_weekday(date_str: str) -> int:
    return datetime.datetime.strptime(date_str, "%Y%m%d").weekday()

WEEKDAY_NAMES = ["周一", "周二", "周三", "周四", "周五"]


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


class OffsetEngine(BacktestEngine):
    """可控起始偏移的每N日调仓引擎 (T-1评分, T open执行)"""

    def __init__(self, *args, offset: int = 0, interval: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self._start_offset = offset
        self._interval = interval

    def run(self, start_date: str, end_date: str):
        dates = self.data["trade_dates"]
        bt_dates = [d for d in dates if start_date <= d <= end_date]
        all_date_idx = {d: i for i, d in enumerate(dates)}

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
                should = (halt_recovery
                          or self.last_rebalance_idx < 0
                          or (i - self.last_rebalance_idx) >= self._interval)
                if should and (i >= self._start_offset):
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


class WeekdaySignalNextDayEngine(BacktestEngine):
    """周X收盘评分 → 次日open执行"""

    def __init__(self, *args, signal_weekday: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self._signal_weekday = signal_weekday

    def run(self, start_date: str, end_date: str):
        dates = self.data["trade_dates"]
        bt_dates = [d for d in dates if start_date <= d <= end_date]
        date_idx = {d: i for i, d in enumerate(dates)}

        signal_to_next = {}
        for d in bt_dates:
            if get_weekday(d) == self._signal_weekday:
                idx = date_idx[d]
                if idx + 1 < len(dates):
                    signal_to_next[d] = dates[idx + 1]

        signal_set = set(signal_to_next.keys())
        exec_day_map = {}

        for i, date in enumerate(bt_dates):
            regime = self._check_regime(date)
            trade_count = 0
            global_idx = date_idx.get(date, 0)
            prev_date = dates[global_idx - 1] if global_idx > 0 else date

            trade_count += self._check_stop_loss(date)

            if regime in ("HALT", "WAIT") and self.positions:
                scores = self._score_stocks(prev_date, regime)
                trade_count += self._execute_rebalance(date, scores, regime)
                self._halt_liquidated = True
                self.rebalance_dates.append(date)
            elif date in signal_set and regime not in ("HALT", "WAIT"):
                scores = self._score_stocks(date, regime)
                next_day = signal_to_next[date]
                exec_day_map[next_day] = (date, scores, regime)

            if date in exec_day_map and regime not in ("HALT", "WAIT"):
                sig_date, scores, sig_regime = exec_day_map.pop(date)
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


def run_single(engine_cls, data, start, end, label="", verbose=True, **kwargs):
    bt = engine_cls("v3.1", deepcopy(data), custom_config=make_v32_config(), **kwargs)
    bt.run(start, end)
    m = bt.calc_metrics()
    if verbose:
        tr = m.get("total_return", 0)
        ar = m.get("annual_return", 0)
        sh = m.get("sharpe", 0)
        md = m.get("max_drawdown", 0)
        nt = len(bt.trade_log)
        nr = m.get("n_rebalances", 0)
        print(f"  {label:<28s} 收益={tr*100:+7.2f}%  年化={ar*100:+7.2f}%  "
              f"Sharpe={sh:+.2f}  回撤={md*100:6.2f}%  交易={nt}")
    return m


def main():
    t_start = time.time()
    data = load_all_data("data/csi1000_market_bundle_700d.csv")
    print(f"[数据加载] {time.time()-t_start:.1f}s")

    trade_dates = data["trade_dates"]
    start = trade_dates[25]
    end = trade_dates[-1]
    mid_idx = 25 + (len(trade_dates) - 25) // 2
    mid = trade_dates[mid_idx]

    n_total = len([d for d in trade_dates if start <= d <= end])
    n_h1 = len([d for d in trade_dates if start <= d <= mid])
    n_h2 = len([d for d in trade_dates if mid < d <= end])

    print(f"\n{'='*80}")
    print(f"  700天数据全面验证")
    print(f"  全程: {start} ~ {end} ({n_total}个交易日, ~{n_total/252:.1f}年)")
    print(f"  前半: {start} ~ {mid} ({n_h1}日) | 后半: {mid} ~ {end} ({n_h2}日)")
    print(f"{'='*80}")

    # ══════════════════════════════════════════════════════
    # 测试1: 因子有效性 — 每5日调仓, 5个偏移量
    # ══════════════════════════════════════════════════════
    print(f"\n{'─'*80}")
    print("  测试1: 每5日调仓 × 5个偏移量 (T-1评分, T open执行)")
    print(f"{'─'*80}")

    offset_full = []
    offset_h1 = []
    offset_h2 = []

    for offset in range(5):
        # 全程
        m = run_single(OffsetEngine, data, start, end,
                       label=f"全程 offset={offset}", offset=offset)
        offset_full.append(m)

        # 前半
        m1 = run_single(OffsetEngine, data, start, mid,
                        label=f"  前半 offset={offset}", offset=offset)
        offset_h1.append(m1)

        # 后半
        m2 = run_single(OffsetEngine, data, mid, end,
                        label=f"  后半 offset={offset}", offset=offset)
        offset_h2.append(m2)
        print()

    def summarize(results, label):
        rets = [m.get("total_return", 0) for m in results]
        sharpes = [m.get("sharpe", 0) for m in results]
        print(f"  {label}: 收益={np.mean(rets)*100:+.2f}% ± {np.std(rets)*100:.2f}%  "
              f"Sharpe={np.mean(sharpes):.2f} ± {np.std(sharpes):.2f}")

    print(f"  {'─'*60}")
    summarize(offset_full, "全程均值")
    summarize(offset_h1, "前半均值")
    summarize(offset_h2, "后半均值")

    # ══════════════════════════════════════════════════════
    # 测试2: 真实生产流程 — 周X信号→次日执行
    # ══════════════════════════════════════════════════════
    print(f"\n{'─'*80}")
    print("  测试2: 真实生产流程 (周X收盘信号 → 次日open执行)")
    print(f"{'─'*80}")

    for wd in range(5):
        exec_wd = WEEKDAY_NAMES[wd + 1] if wd < 4 else WEEKDAY_NAMES[0]
        label = f"{WEEKDAY_NAMES[wd]}信号→{exec_wd}执行"
        if wd == 4:
            label += "(跨周末)"
        run_single(WeekdaySignalNextDayEngine, data, start, end,
                   label=label, signal_weekday=wd)

    # ══════════════════════════════════════════════════════
    # 测试3: 不同调仓间隔
    # ══════════════════════════════════════════════════════
    print(f"\n{'─'*80}")
    print("  测试3: 不同调仓间隔 (offset=0, 全程)")
    print(f"{'─'*80}")

    for interval in [3, 5, 7, 10, 15, 20]:
        run_single(OffsetEngine, data, start, end,
                   label=f"间隔={interval:2d}日", offset=0, interval=interval)

    # ══════════════════════════════════════════════════════
    # 测试4: 基准对比 — CSI1000指数
    # ══════════════════════════════════════════════════════
    print(f"\n{'─'*80}")
    print("  基准: CSI1000 指数同期表现")
    print(f"{'─'*80}")
    idx = data["index_data"]
    idx_start = idx[idx["trade_date"] >= start].iloc[0]
    idx_end = idx[idx["trade_date"] <= end].iloc[-1]
    idx_mid = idx[idx["trade_date"] <= mid].iloc[-1]
    idx_ret_full = idx_end["close"] / idx_start["close"] - 1
    idx_ret_h1 = idx_mid["close"] / idx_start["close"] - 1
    idx_ret_h2 = idx_end["close"] / idx_mid["close"] - 1
    print(f"  全程: {idx_ret_full*100:+.2f}%  | 前半: {idx_ret_h1*100:+.2f}%  | 后半: {idx_ret_h2*100:+.2f}%")

    print(f"\n{'='*80}")
    print(f"  总耗时: {time.time()-t_start:.0f}s")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
