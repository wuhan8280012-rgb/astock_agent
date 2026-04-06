#!/usr/bin/env python3
"""
龙虎榜因子回测 — 验证机构资金流向对动量策略的增强效果

设计思路:
  龙虎榜信号本质是: 某只股票在某天因异动(涨幅/换手/振幅)上榜,
  并且机构净买入 > 0. 这类股票往往有持续上涨动力.

  融入方式:
    方案A: 龙虎榜加分  — 近N天内上过龙虎榜且机构净买入的股票, 动量得分 × 加成系数
    方案B: 龙虎榜过滤  — 只从近N天上过龙虎榜的股票中选
    方案C: 龙虎榜权重  — 将龙虎榜净买入额作为独立因子加入综合评分

  由于当前没有真实龙虎榜数据, 用以下代理指标模拟:
    "模拟上榜" = 当日涨幅 > 5% 或 换手率(vol/流通股) > 8%
    "模拟机构净买入" = 上榜当日成交额的20% (假设机构占比)
    这个模拟并不精确, 但能验证"异动+资金确认"这个逻辑框架是否有alpha
"""

from __future__ import annotations
import sys, json, numpy as np
from pathlib import Path
from dataclasses import dataclass, field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import signal as _builtin_signal
_saved_signal = sys.modules.pop("signal")
sys.modules.pop("signal.signal_generator", None)
from signal.signal_generator import SignalConfig, SignalGenerator
sys.modules["_builtin_signal"] = _saved_signal

from scripts.backtest_v2_vs_v3 import load_all_data, BacktestEngine, Position
import pandas as pd


def build_simulated_lhb(stock_data: dict, trade_dates: list) -> dict:
    """
    构建模拟龙虎榜数据
    
    上榜条件 (模拟真实龙虎榜规则):
      1. 日涨幅 >= 7%  (对应真实规则: 涨幅偏离值达7%)
      2. 或 日振幅 >= 15% (price range / close)
      3. 或 换手率 >= 15% (vol相对流通盘)
    
    返回: {ts_code: {trade_date: {'net_buy': float, 'reason': str}}}
    """
    lhb = {}
    
    for ts_code, df in stock_data.items():
        df = df.sort_values("trade_date").reset_index(drop=True)
        for i, row in df.iterrows():
            date = row["trade_date"]
            pct = row.get("pct_chg", 0)
            if pd.isna(pct):
                continue
            
            high = row.get("high", 0)
            low = row.get("low", 0)
            close = row.get("close", 0)
            amount = row.get("amount", 0) * 1000  # 转换为元
            
            # 振幅
            amplitude = (high - low) / close * 100 if close > 0 else 0
            
            reason = None
            if pct >= 7:
                reason = "涨幅偏离"
            elif pct <= -7:
                reason = "跌幅偏离"
            elif amplitude >= 15:
                reason = "振幅偏离"
            
            if reason:
                # 模拟净买入: 涨的日子净买入为正, 跌的日子为负
                if pct > 0:
                    net_buy = amount * 0.15  # 假设15%为机构净买入
                else:
                    net_buy = -amount * 0.15
                
                if ts_code not in lhb:
                    lhb[ts_code] = {}
                lhb[ts_code][date] = {
                    "net_buy": net_buy,
                    "reason": reason,
                    "pct_chg": pct,
                }
    
    return lhb


def calc_lhb_score(ts_code: str, date: str, lhb: dict, lookback_days: int = 20,
                   trade_dates: list = None) -> tuple[float, int]:
    """
    计算某只股票在date之前lookback_days内的龙虎榜信号强度
    
    返回: (lhb_score, appearance_count)
      lhb_score: 归一化的龙虎榜信号强度 (净买入额加权)
      appearance_count: 上榜次数
    """
    if ts_code not in lhb:
        return 0.0, 0
    
    stock_lhb = lhb[ts_code]
    
    # 找回看窗口内的上榜记录
    if trade_dates:
        try:
            idx = trade_dates.index(date)
        except ValueError:
            return 0.0, 0
        window_dates = set(trade_dates[max(0, idx - lookback_days):idx + 1])
    else:
        window_dates = None
    
    total_net_buy = 0
    count = 0
    for d, info in stock_lhb.items():
        if window_dates and d not in window_dates:
            continue
        if d > date:
            continue
        total_net_buy += info["net_buy"]
        count += 1
    
    return total_net_buy, count


class LHBBacktestEngine(BacktestEngine):
    """带龙虎榜因子的回测引擎"""
    
    def __init__(self, version: str, data: dict, lhb: dict,
                 lhb_mode: str = "boost", lhb_weight: float = 0.2,
                 lhb_lookback: int = 20, initial_capital: float = 1_000_000):
        super().__init__(version, data, initial_capital)
        self.lhb = lhb
        self.lhb_mode = lhb_mode      # "boost" | "filter" | "factor"
        self.lhb_weight = lhb_weight
        self.lhb_lookback = lhb_lookback
    
    def _score_stocks(self, date: str, regime: str) -> list[dict]:
        """评分 + 龙虎榜因子"""
        # 基础动量评分
        scores = super()._score_stocks(date, regime)
        
        trade_dates = self.data["trade_dates"]
        
        if self.lhb_mode == "boost":
            # 方案A: 近N天上过龙虎榜且净买入>0的, 得分加成
            for s in scores:
                net_buy, count = calc_lhb_score(
                    s["ts_code"], date, self.lhb, self.lhb_lookback, trade_dates
                )
                if count > 0 and net_buy > 0:
                    # 加成: 1 + weight * min(count/3, 1) * sign(net_buy)
                    boost = 1 + self.lhb_weight * min(count / 3, 1.0)
                    s["score"] *= boost
                    s["lhb_boost"] = boost
                elif count > 0 and net_buy < 0:
                    # 机构净卖出: 轻微惩罚
                    s["score"] *= (1 - self.lhb_weight * 0.3)
        
        elif self.lhb_mode == "filter":
            # 方案B: 只保留近N天上过龙虎榜且净买入>0的
            filtered = []
            for s in scores:
                net_buy, count = calc_lhb_score(
                    s["ts_code"], date, self.lhb, self.lhb_lookback, trade_dates
                )
                if count > 0 and net_buy > 0:
                    filtered.append(s)
            # 如果过滤后不足top_n, 回退到全部
            if len(filtered) >= self.config.top_n:
                scores = filtered
        
        elif self.lhb_mode == "factor":
            # 方案C: 龙虎榜净买入作为独立因子
            # 归一化net_buy后加入综合评分
            lhb_values = []
            for s in scores:
                net_buy, count = calc_lhb_score(
                    s["ts_code"], date, self.lhb, self.lhb_lookback, trade_dates
                )
                s["lhb_raw"] = net_buy
                lhb_values.append(net_buy)
            
            if lhb_values:
                max_abs = max(abs(v) for v in lhb_values) if lhb_values else 1
                if max_abs > 0:
                    for s in scores:
                        lhb_norm = s["lhb_raw"] / max_abs  # [-1, 1]
                        s["score"] = s["score"] * (1 - self.lhb_weight) + lhb_norm * self.lhb_weight
        
        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores


def main():
    csv_300d = PROJECT_ROOT / "data" / "csi1000_market_bundle_300d.csv"
    csv_100d = PROJECT_ROOT / "data" / "csi1000_market_bundle_100d.csv"
    
    if csv_300d.exists():
        csv_path = str(csv_300d)
        start_date = "20250206"
        end_date = "20260324"
    elif csv_100d.exists():
        csv_path = str(csv_100d)
        start_date = "20251127"
        end_date = "20260323"
    else:
        print("找不到数据集")
        return
    
    data = load_all_data(csv_path)
    
    # 构建模拟龙虎榜
    print("[龙虎榜] 构建模拟数据...")
    lhb = build_simulated_lhb(data["stock_data"], data["trade_dates"])
    
    # 统计
    total_entries = sum(len(v) for v in lhb.values())
    stocks_with_lhb = len(lhb)
    print(f"[龙虎榜] {stocks_with_lhb}只股票共{total_entries}次上榜记录")
    
    # 上榜原因分布
    reasons = {}
    net_buy_positive = 0
    for ts_code, entries in lhb.items():
        for date, info in entries.items():
            r = info["reason"]
            reasons[r] = reasons.get(r, 0) + 1
            if info["net_buy"] > 0:
                net_buy_positive += 1
    print(f"[龙虎榜] 原因分布: {reasons}")
    print(f"[龙虎榜] 净买入为正: {net_buy_positive}/{total_entries} ({net_buy_positive/total_entries:.0%})")
    
    # ═══ 回测对比 ═══
    print(f"\n{'═'*80}")
    print(f"  龙虎榜因子回测: {start_date} ~ {end_date}")
    print(f"{'═'*80}\n")
    
    configs = [
        ("v3.0 基准 (无龙虎榜)", "v3.0", None, None, None),
        ("方案A: 龙虎榜加分 w=0.15", "v3.0", "boost", 0.15, 20),
        ("方案A: 龙虎榜加分 w=0.25", "v3.0", "boost", 0.25, 20),
        ("方案A: 龙虎榜加分 w=0.35", "v3.0", "boost", 0.35, 20),
        ("方案B: 龙虎榜过滤", "v3.0", "filter", 0.0, 20),
        ("方案C: 龙虎榜因子 w=0.15", "v3.0", "factor", 0.15, 20),
        ("方案C: 龙虎榜因子 w=0.25", "v3.0", "factor", 0.25, 20),
        # 不同回看窗口
        ("方案A: w=0.25 回看10天", "v3.0", "boost", 0.25, 10),
        ("方案A: w=0.25 回看5天", "v3.0", "boost", 0.25, 5),
    ]
    
    print(f"{'策略':>30s} {'累计':>8s} {'年化':>8s} {'夏普':>7s} {'回撤':>8s} {'交易':>6s}")
    print("─" * 75)
    
    results = {}
    for label, ver, mode, weight, lookback in configs:
        if mode is None:
            bt = BacktestEngine(ver, data)
        else:
            bt = LHBBacktestEngine(ver, data, lhb, 
                                    lhb_mode=mode, lhb_weight=weight, 
                                    lhb_lookback=lookback)
        bt.run(start_date, end_date)
        m = bt.calc_metrics()
        results[label] = m
        
        print(f"{label:>30s} {m['total_return']:>7.2%} {m['annual_return']:>7.2%} "
              f"{m['sharpe']:>6.2f} {m['max_drawdown']:>7.2%} {m['total_trades']:>5d}")
    
    # 保存
    out_path = PROJECT_ROOT / "data" / "backtest_lhb.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    print(f"\n[保存] {out_path}")


if __name__ == "__main__":
    main()
