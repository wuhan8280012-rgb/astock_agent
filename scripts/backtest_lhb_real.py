#!/usr/bin/env python3
"""
龙虎榜因子回测 — 使用真实龙虎榜数据 (top_list)
"""
from __future__ import annotations
import sys, json, numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import signal as _builtin_signal
_saved_signal = sys.modules.pop("signal")
sys.modules.pop("signal.signal_generator", None)
from signal.signal_generator import SignalConfig, SignalGenerator
sys.modules["_builtin_signal"] = _saved_signal

from scripts.backtest_v2_vs_v3 import load_all_data, BacktestEngine
import pandas as pd


def load_real_lhb(csv_path: str) -> dict:
    """
    从CSV中加载真实龙虎榜数据
    返回: {ts_code: {trade_date: {'l_buy': float, 'l_sell': float, 'net_buy': float}}}
    """
    raw = pd.read_csv(csv_path, low_memory=False)
    lhb_df = raw[raw["data_type"] == "top_list"].copy()
    
    for col in ["trade_date", "l_buy", "l_sell", "pct_chg", "amount", "turnover_rate"]:
        if col in lhb_df.columns:
            lhb_df[col] = pd.to_numeric(lhb_df[col], errors="coerce")
    lhb_df["trade_date"] = lhb_df["trade_date"].astype(int).astype(str)
    
    lhb = {}
    for _, row in lhb_df.iterrows():
        ts_code = row["ts_code"]
        date = row["trade_date"]
        l_buy = row.get("l_buy", 0) or 0
        l_sell = row.get("l_sell", 0) or 0
        
        if ts_code not in lhb:
            lhb[ts_code] = {}
        
        # 同一天可能有多条记录(不同原因), 合并
        if date in lhb[ts_code]:
            lhb[ts_code][date]["l_buy"] += l_buy
            lhb[ts_code][date]["l_sell"] += l_sell
            lhb[ts_code][date]["net_buy"] = lhb[ts_code][date]["l_buy"] - lhb[ts_code][date]["l_sell"]
        else:
            lhb[ts_code][date] = {
                "l_buy": l_buy,
                "l_sell": l_sell,
                "net_buy": l_buy - l_sell,
            }
    
    return lhb


def calc_lhb_signal(ts_code, date, lhb, lookback, trade_dates):
    """计算龙虎榜信号: 回看N天内的净买入情况"""
    if ts_code not in lhb:
        return 0.0, 0
    
    try:
        idx = trade_dates.index(date)
    except ValueError:
        return 0.0, 0
    
    window = set(trade_dates[max(0, idx - lookback):idx + 1])
    stock_lhb = lhb[ts_code]
    
    total_net = 0.0
    count = 0
    for d in window:
        if d in stock_lhb:
            total_net += stock_lhb[d]["net_buy"]
            count += 1
    
    return total_net, count


class RealLHBEngine(BacktestEngine):
    """使用真实龙虎榜数据的回测引擎"""
    
    def __init__(self, version, data, lhb, lhb_mode="boost",
                 lhb_weight=0.25, lhb_lookback=10, initial_capital=1_000_000):
        super().__init__(version, data, initial_capital)
        self.lhb = lhb
        self.lhb_mode = lhb_mode
        self.lhb_weight = lhb_weight
        self.lhb_lookback = lhb_lookback
    
    def _score_stocks(self, date, regime):
        scores = super()._score_stocks(date, regime)
        trade_dates = self.data["trade_dates"]
        
        if self.lhb_mode == "boost":
            for s in scores:
                net_buy, count = calc_lhb_signal(
                    s["ts_code"], date, self.lhb, self.lhb_lookback, trade_dates
                )
                if count > 0 and net_buy > 0:
                    boost = 1 + self.lhb_weight * min(count / 2, 1.0)
                    s["score"] *= boost
                elif count > 0 and net_buy < 0:
                    s["score"] *= (1 - self.lhb_weight * 0.3)
        
        elif self.lhb_mode == "filter":
            filtered = []
            for s in scores:
                net_buy, count = calc_lhb_signal(
                    s["ts_code"], date, self.lhb, self.lhb_lookback, trade_dates
                )
                if count > 0 and net_buy > 0:
                    filtered.append(s)
            if len(filtered) >= self.config.top_n:
                scores = filtered
        
        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores


def main():
    csv_path = str(PROJECT_ROOT / "data" / "csi1000_market_bundle_300d_lhb.csv")
    start_date = "20250206"
    end_date = "20260324"
    
    data = load_all_data(csv_path)
    lhb = load_real_lhb(csv_path)
    
    # 统计
    total = sum(len(v) for v in lhb.values())
    net_pos = sum(1 for ts in lhb.values() for d in ts.values() if d["net_buy"] > 0)
    print(f"[龙虎榜] 真实数据: {len(lhb)}只股票, {total}条记录, 净买入>0: {net_pos} ({net_pos/total:.0%})")
    
    print(f"\n{'═'*80}")
    print(f"  真实龙虎榜因子回测: {start_date} ~ {end_date}")
    print(f"{'═'*80}\n")
    
    configs = [
        # 基准
        ("v3.0 基准 (无龙虎榜)", None, None, None),
        # 方案A: boost不同权重
        ("方案A: 加分 w=0.15 回看10天", "boost", 0.15, 10),
        ("方案A: 加分 w=0.20 回看10天", "boost", 0.20, 10),
        ("方案A: 加分 w=0.25 回看10天", "boost", 0.25, 10),
        ("方案A: 加分 w=0.30 回看10天", "boost", 0.30, 10),
        ("方案A: 加分 w=0.40 回看10天", "boost", 0.40, 10),
        # 方案A: 不同回看窗口
        ("方案A: 加分 w=0.25 回看5天", "boost", 0.25, 5),
        ("方案A: 加分 w=0.25 回看15天", "boost", 0.25, 15),
        ("方案A: 加分 w=0.25 回看20天", "boost", 0.25, 20),
        # 方案B: 过滤
        ("方案B: 过滤 回看10天", "filter", 0.0, 10),
        ("方案B: 过滤 回看15天", "filter", 0.0, 15),
        ("方案B: 过滤 回看20天", "filter", 0.0, 20),
    ]
    
    print(f"{'策略':>32s} {'累计':>8s} {'年化':>8s} {'夏普':>7s} {'回撤':>8s} {'交易':>6s}")
    print("─" * 77)
    
    for label, mode, weight, lookback in configs:
        if mode is None:
            bt = BacktestEngine("v3.0", data)
        else:
            bt = RealLHBEngine("v3.0", data, lhb,
                               lhb_mode=mode, lhb_weight=weight,
                               lhb_lookback=lookback)
        bt.run(start_date, end_date)
        m = bt.calc_metrics()
        print(f"{label:>32s} {m['total_return']:>7.2%} {m['annual_return']:>7.2%} "
              f"{m['sharpe']:>6.2f} {m['max_drawdown']:>7.2%} {m['total_trades']:>5d}")


if __name__ == "__main__":
    main()
