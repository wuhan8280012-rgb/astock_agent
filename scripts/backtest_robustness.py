#!/usr/bin/env python3
"""
策略F稳健性检验 — 参数敏感性 + 滚动窗口验证
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from backtest_strategies import load_data, Backtest, BacktestConfig


def run_param_sensitivity(daily, idx, basic, trade_dates):
    """参数敏感性: 在策略F的基础上, 逐一变化关键参数"""

    print("\n" + "=" * 80)
    print("第一部分: 参数敏感性检验")
    print("=" * 80)

    variants = []

    # 动量周期
    for mom_days in [40, 60, 80, 100]:
        variants.append(BacktestConfig(
            name=f"动量{mom_days}d",
            momentum_days=[mom_days], momentum_weights=[1.0],
            use_volatility_factor=True, volatility_days=60, volatility_weight=0.25,
            use_size_factor=True, size_weight=0.2,
            top_n=15, hold_buffer_ratio=1.5, max_single_weight=0.08,
            rebalance_interval=20, stop_loss_pct=-0.15,
            use_trend_filter=True, trend_ma_days=200, trend_reduce_pct=0.5,
            slippage=0.002,
        ))

    # 持仓数量
    for top_n in [10, 15, 20, 25]:
        variants.append(BacktestConfig(
            name=f"持仓{top_n}只",
            momentum_days=[60], momentum_weights=[1.0],
            use_volatility_factor=True, volatility_days=60, volatility_weight=0.25,
            use_size_factor=True, size_weight=0.2,
            top_n=top_n, hold_buffer_ratio=1.5, max_single_weight=min(0.12, 1.0/top_n),
            rebalance_interval=20, stop_loss_pct=-0.15,
            use_trend_filter=True, trend_ma_days=200, trend_reduce_pct=0.5,
            slippage=0.002,
        ))

    # 调仓间隔
    for interval in [10, 15, 20, 30]:
        variants.append(BacktestConfig(
            name=f"调仓{interval}d",
            momentum_days=[60], momentum_weights=[1.0],
            use_volatility_factor=True, volatility_days=60, volatility_weight=0.25,
            use_size_factor=True, size_weight=0.2,
            top_n=15, hold_buffer_ratio=1.5, max_single_weight=0.08,
            rebalance_interval=interval, stop_loss_pct=-0.15,
            use_trend_filter=True, trend_ma_days=200, trend_reduce_pct=0.5,
            slippage=0.002,
        ))

    # 止损
    for sl in [-0.10, -0.15, -0.20, -0.99]:
        label = "无止损" if sl < -0.5 else f"止损{int(sl*100)}%"
        variants.append(BacktestConfig(
            name=label,
            momentum_days=[60], momentum_weights=[1.0],
            use_volatility_factor=True, volatility_days=60, volatility_weight=0.25,
            use_size_factor=True, size_weight=0.2,
            top_n=15, hold_buffer_ratio=1.5, max_single_weight=0.08,
            rebalance_interval=20, stop_loss_pct=sl,
            use_trend_filter=True, trend_ma_days=200, trend_reduce_pct=0.5,
            slippage=0.002,
        ))

    # 波动率权重
    for vw in [0.15, 0.25, 0.35, 0.45]:
        variants.append(BacktestConfig(
            name=f"波动率权重{vw:.0%}",
            momentum_days=[60], momentum_weights=[1.0],
            use_volatility_factor=True, volatility_days=60, volatility_weight=vw,
            use_size_factor=True, size_weight=0.2,
            top_n=15, hold_buffer_ratio=1.5, max_single_weight=0.08,
            rebalance_interval=20, stop_loss_pct=-0.15,
            use_trend_filter=True, trend_ma_days=200, trend_reduce_pct=0.5,
            slippage=0.002,
        ))

    # 趋势均线天数
    for ma in [120, 160, 200, 250]:
        variants.append(BacktestConfig(
            name=f"趋势MA{ma}d",
            momentum_days=[60], momentum_weights=[1.0],
            use_volatility_factor=True, volatility_days=60, volatility_weight=0.25,
            use_size_factor=True, size_weight=0.2,
            top_n=15, hold_buffer_ratio=1.5, max_single_weight=0.08,
            rebalance_interval=20, stop_loss_pct=-0.15,
            use_trend_filter=True, trend_ma_days=ma, trend_reduce_pct=0.5,
            slippage=0.002,
        ))

    # 滑点敏感性
    for slip in [0.001, 0.002, 0.003, 0.005]:
        variants.append(BacktestConfig(
            name=f"滑点{slip:.1%}",
            momentum_days=[60], momentum_weights=[1.0],
            use_volatility_factor=True, volatility_days=60, volatility_weight=0.25,
            use_size_factor=True, size_weight=0.2,
            top_n=15, hold_buffer_ratio=1.5, max_single_weight=0.08,
            rebalance_interval=20, stop_loss_pct=-0.15,
            use_trend_filter=True, trend_ma_days=200, trend_reduce_pct=0.5,
            slippage=slip,
        ))

    results = []
    for cfg in variants:
        bt = Backtest(cfg, daily, idx, basic, trade_dates)
        r = bt.run(start_offset=250)
        results.append(r)

    # 分组输出
    groups = {
        "动量周期": [r for r in results if r["name"].startswith("动量")],
        "持仓数量": [r for r in results if r["name"].startswith("持仓")],
        "调仓间隔": [r for r in results if r["name"].startswith("调仓")],
        "止损阈值": [r for r in results if "止损" in r["name"]],
        "波动率权重": [r for r in results if r["name"].startswith("波动率")],
        "趋势均线": [r for r in results if r["name"].startswith("趋势MA")],
        "滑点敏感性": [r for r in results if r["name"].startswith("滑点")],
    }

    print("\n\n" + "=" * 80)
    print("参数敏感性汇总")
    print("=" * 80)

    for group_name, group_results in groups.items():
        print(f"\n── {group_name} ──")
        print(f"  {'参数':<20} {'总收益':>8} {'年化':>8} {'夏普':>6} {'最大回撤':>8} {'交易次数':>8}")
        for r in group_results:
            print(f"  {r['name']:<20} {r['total_return_pct']:>7.1f}% {r['annual_return_pct']:>7.1f}% "
                  f"{r['sharpe']:>6.2f} {r['max_drawdown_pct']:>7.1f}% {r['total_trades']:>8}")

    return results


def run_walk_forward(daily, idx, basic, trade_dates):
    """滚动窗口验证"""

    print("\n\n" + "=" * 80)
    print("第二部分: 滚动窗口验证 (Walk-Forward)")
    print("=" * 80)

    cfg_template = dict(
        momentum_days=[60], momentum_weights=[1.0],
        use_volatility_factor=True, volatility_days=60, volatility_weight=0.25,
        use_size_factor=True, size_weight=0.2,
        top_n=15, hold_buffer_ratio=1.5, max_single_weight=0.08,
        rebalance_interval=20, stop_loss_pct=-0.15,
        use_trend_filter=True, trend_ma_days=200, trend_reduce_pct=0.5,
        slippage=0.002,
    )

    # 分段: 每半年
    start_idx = 250
    segment_length = 126
    wf_results = []

    while start_idx + segment_length <= len(trade_dates):
        seg_start = trade_dates[start_idx]
        seg_end = trade_dates[min(start_idx + segment_length - 1, len(trade_dates) - 1)]

        window_dates = trade_dates[:start_idx + segment_length]
        window_daily = daily[daily["trade_date"].isin(set(window_dates))]
        window_idx = idx[idx["trade_date"].isin(set(window_dates))]

        cfg = BacktestConfig(name=f"WF_{seg_start}~{seg_end}", **cfg_template)
        bt = Backtest(cfg, window_daily, window_idx, basic, window_dates)
        r = bt.run(start_offset=start_idx)
        r["segment"] = f"{seg_start}~{seg_end}"
        wf_results.append(r)

        start_idx += segment_length

    print(f"\n{'窗口':<25} {'总收益':>8} {'年化':>8} {'夏普':>6} {'最大回撤':>8}")
    print("-" * 60)
    wins = 0
    for r in wf_results:
        print(f"{r['segment']:<25} {r['total_return_pct']:>7.1f}% {r['annual_return_pct']:>7.1f}% "
              f"{r['sharpe']:>6.2f} {r['max_drawdown_pct']:>7.1f}%")
        if r['total_return_pct'] > 0:
            wins += 1

    print(f"\n盈利窗口: {wins}/{len(wf_results)} ({wins/len(wf_results)*100:.0f}%)")
    return wf_results


def main():
    daily, idx, basic, trade_dates = load_data()
    sensitivity = run_param_sensitivity(daily, idx, basic, trade_dates)
    wf = run_walk_forward(daily, idx, basic, trade_dates)

    output = {"sensitivity": sensitivity, "walk_forward": wf}
    path = Path(__file__).parent.parent / "backtest" / "robustness_check.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存至: {path}")


if __name__ == "__main__":
    main()
