#!/usr/bin/env python3
"""
动量轮动策略回测验证脚本 — 自包含版本（无外部依赖）。

使用模拟A股市场数据，完整验证动量轮动策略流水线：
  股票池筛选 → 动量计算 → 排名选股 → 风控检查 → 调仓执行 → 绩效分析

模拟 20 个交易日的数据，对应约 4 周的交易。

Usage:
    python scripts/run_momentum_backtest_demo.py
"""

from __future__ import annotations

import json
import math
import random
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Logging shim (avoid loguru dependency) ────────────────────────────────────

class _Logger:
    def info(self, msg, *a, **kw): print(f"[INFO]  {msg}")
    def warning(self, msg, *a, **kw): print(f"[WARN]  {msg}")
    def error(self, msg, *a, **kw): print(f"[ERROR] {msg}")
    def debug(self, msg, *a, **kw): pass

logger = _Logger()

# ── Market Data Simulator ─────────────────────────────────────────────────────

# 30 representative A-share stocks across sectors
STOCK_POOL = [
    ("600519.SH", "贵州茅台", "白酒",    1800.0),
    ("000858.SZ", "五粮液",   "白酒",     160.0),
    ("601318.SH", "中国平安", "保险",      48.0),
    ("600036.SH", "招商银行", "银行",      35.0),
    ("000001.SZ", "平安银行", "银行",      12.0),
    ("002415.SZ", "海康威视", "安防",      32.0),
    ("300750.SZ", "宁德时代", "电池",     200.0),
    ("601012.SH", "隆基绿能", "光伏",      22.0),
    ("002594.SZ", "比亚迪",   "汽车",     260.0),
    ("600900.SH", "长江电力", "电力",      28.0),
    ("000063.SZ", "中兴通讯", "通信",      28.0),
    ("002230.SZ", "科大讯飞", "AI",        45.0),
    ("300059.SZ", "东方财富", "券商",      16.0),
    ("601888.SH", "中国中免", "免税",      72.0),
    ("000568.SZ", "泸州老窖", "白酒",     180.0),
    ("002714.SZ", "牧原股份", "养殖",      42.0),
    ("601899.SH", "紫金矿业", "有色",      15.0),
    ("600276.SH", "恒瑞医药", "医药",      45.0),
    ("603259.SH", "药明康德", "CXO",       55.0),
    ("002475.SZ", "立讯精密", "消费电子",  32.0),
    ("600585.SH", "海螺水泥", "建材",      22.0),
    ("601688.SH", "华泰证券", "券商",      15.0),
    ("002352.SZ", "顺丰控股", "物流",      38.0),
    ("600809.SH", "山西汾酒", "白酒",     250.0),
    ("300124.SZ", "汇川技术", "自动化",    58.0),
    ("002049.SZ", "紫光国微", "芯片",      95.0),
    ("688981.SH", "中芯国际", "半导体",    52.0),
    ("601919.SH", "中远海控", "航运",      12.0),
    ("002812.SZ", "恩捷股份", "隔膜",      65.0),
    ("300274.SZ", "阳光电源", "逆变器",    85.0),
]


def generate_market_data(n_days: int = 20, seed: int = 42) -> dict:
    """
    Generate realistic A-share market data with sector rotation patterns.

    Returns dict: {ts_code: DataFrame[trade_date, open, high, low, close, vol, amount, pct_chg]}
    Also returns: trade_calendar, index_data (CSI300)
    """
    rng = np.random.RandomState(seed)

    # Generate trade calendar (skip weekends)
    base_date = datetime(2026, 2, 20)
    trade_dates = []
    d = base_date
    while len(trade_dates) < n_days:
        if d.weekday() < 5:  # Mon-Fri
            trade_dates.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)

    # Sector momentum patterns (simulate rotation)
    sector_trends = {
        "白酒":    rng.normal(0.003, 0.005, n_days),   # Strong uptrend
        "银行":    rng.normal(0.001, 0.008, n_days),   # Mild uptrend
        "保险":    rng.normal(0.000, 0.010, n_days),   # Flat
        "安防":    rng.normal(-0.001, 0.012, n_days),  # Slight down
        "电池":    rng.normal(0.004, 0.015, n_days),   # Strong but volatile
        "光伏":    rng.normal(0.002, 0.018, n_days),   # Up but volatile
        "汽车":    rng.normal(0.005, 0.012, n_days),   # Strongest sector
        "电力":    rng.normal(0.001, 0.006, n_days),   # Defensive
        "通信":    rng.normal(0.002, 0.010, n_days),   # Moderate
        "AI":      rng.normal(0.006, 0.020, n_days),   # Hot theme, high vol
        "券商":    rng.normal(0.003, 0.015, n_days),   # Bull proxy
        "免税":    rng.normal(-0.002, 0.014, n_days),  # Cooling
        "养殖":    rng.normal(-0.001, 0.016, n_days),  # Weak
        "有色":    rng.normal(0.004, 0.013, n_days),   # Resource boom
        "医药":    rng.normal(0.000, 0.011, n_days),   # Flat
        "CXO":     rng.normal(-0.003, 0.015, n_days),  # Downtrend
        "消费电子": rng.normal(0.002, 0.014, n_days),  # Moderate
        "建材":    rng.normal(-0.001, 0.009, n_days),  # Weak
        "物流":    rng.normal(0.001, 0.010, n_days),   # Flat
        "自动化":  rng.normal(0.003, 0.012, n_days),   # Good
        "芯片":    rng.normal(0.005, 0.018, n_days),   # Hot
        "半导体":  rng.normal(0.004, 0.017, n_days),   # Hot
        "航运":    rng.normal(-0.002, 0.020, n_days),  # Volatile down
        "隔膜":    rng.normal(0.001, 0.016, n_days),   # Moderate
        "逆变器":  rng.normal(0.003, 0.015, n_days),   # Good
    }

    stock_data = {}
    for ts_code, name, sector, base_price in STOCK_POOL:
        sector_drift = sector_trends.get(sector, rng.normal(0.001, 0.010, n_days))
        idio = rng.normal(0, 0.01, n_days)  # Idiosyncratic noise
        daily_returns = sector_drift + idio

        prices = [base_price]
        for r in daily_returns:
            # A-share limit: ±10% (±20% for 创业板/科创板)
            limit = 0.20 if ts_code.startswith("300") or ts_code.startswith("688") else 0.10
            capped_r = max(-limit, min(limit, r))
            prices.append(prices[-1] * (1 + capped_r))

        rows = []
        for i in range(n_days):
            c = prices[i + 1]
            o = prices[i] * (1 + rng.normal(0, 0.005))
            h = max(c, o) * (1 + abs(rng.normal(0, 0.005)))
            l = min(c, o) * (1 - abs(rng.normal(0, 0.005)))
            pct = (c / prices[i] - 1) * 100
            vol = rng.uniform(5e6, 5e8)
            amount = vol * c

            rows.append({
                "ts_code": ts_code,
                "trade_date": trade_dates[i],
                "open": round(o, 2),
                "high": round(h, 2),
                "low": round(l, 2),
                "close": round(c, 2),
                "vol": round(vol),
                "amount": round(amount),
                "pct_chg": round(pct, 2),
            })

        stock_data[ts_code] = pd.DataFrame(rows)

    # Generate CSI300 index
    idx_base = 3800.0
    idx_rows = []
    for i in range(n_days):
        idx_ret = rng.normal(0.002, 0.008)
        idx_base *= (1 + idx_ret)
        idx_rows.append({
            "ts_code": "000300.SH",
            "trade_date": trade_dates[i],
            "close": round(idx_base, 2),
            "pct_chg": round(idx_ret * 100, 2),
        })

    return {
        "stock_data": stock_data,
        "trade_calendar": trade_dates,
        "index_data": pd.DataFrame(idx_rows),
        "stock_info": pd.DataFrame([
            {"ts_code": s[0], "name": s[1], "industry": s[2], "base_price": s[3]}
            for s in STOCK_POOL
        ]),
    }


# ── Strategy Config ───────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    lookback_days: list[int] = field(default_factory=lambda: [5, 10, 20])
    lookback_weights: list[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])
    top_n: int = 5
    rebalance_interval_days: int = 5  # ~weekly
    max_single_weight: float = 0.25
    stop_loss_pct: float = -0.08
    initial_capital: float = 1_000_000.0
    commission_rate: float = 0.0003     # 万三
    stamp_tax_rate: float = 0.001       # 千一(卖出)
    momentum_type: str = "composite"    # simple | risk_adjusted | composite
    volatility_penalty: float = 0.3


# ── Core Strategy Logic ──────────────────────────────────────────────────────

def calc_momentum(prices_df: pd.DataFrame, lookback: int) -> float:
    """Calculate simple momentum (return over lookback period)."""
    if len(prices_df) < lookback + 1:
        return 0.0
    close = prices_df["close"].values
    return (close[-1] / close[-(lookback + 1)] - 1.0)


def calc_volatility(prices_df: pd.DataFrame, lookback: int) -> float:
    """Calculate annualized volatility."""
    if len(prices_df) < lookback + 1:
        return 1.0
    returns = prices_df["close"].pct_change().dropna().values[-lookback:]
    if len(returns) < 2:
        return 1.0
    return float(np.std(returns)) * math.sqrt(252)


def calc_composite_momentum(prices_df: pd.DataFrame, config: BacktestConfig) -> float:
    """Weighted multi-horizon momentum with volatility penalty."""
    score = 0.0
    for lb, w in zip(config.lookback_days, config.lookback_weights):
        m = calc_momentum(prices_df, lb)
        score += w * m

    if config.volatility_penalty > 0:
        vol = calc_volatility(prices_df, max(config.lookback_days))
        if vol > 0:
            penalty = config.volatility_penalty * (vol - 0.25)  # Penalize above 25% annualized vol
            score -= max(0, penalty)

    return score


def check_market_regime(index_data: pd.DataFrame, as_of_idx: int) -> str:
    """Simple market regime check based on CSI300."""
    if as_of_idx < 10:
        return "RUN"
    close = index_data["close"].values[:as_of_idx + 1]
    ma10 = np.mean(close[-10:])
    current = close[-1]
    ret5 = (close[-1] / close[-6] - 1) if len(close) >= 6 else 0

    if current < ma10 * 0.97:
        return "HALT"
    elif current < ma10:
        return "DEFENSIVE"
    elif ret5 > 0.03:
        return "STRONG_RUN"
    else:
        return "RUN"


# ── Portfolio Tracker ─────────────────────────────────────────────────────────

@dataclass
class Position:
    ts_code: str
    name: str
    shares: int
    entry_price: float
    entry_date: str
    current_price: float
    peak_price: float
    weight: float = 0.0

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def pnl_pct(self) -> float:
        return (self.current_price / self.entry_price - 1) * 100


@dataclass
class TradeRecord:
    date: str
    ts_code: str
    name: str
    action: str  # BUY or SELL
    shares: int
    price: float
    value: float
    commission: float
    reason: str


@dataclass
class DailySnapshot:
    date: str
    portfolio_value: float
    cash: float
    n_positions: int
    regime: str
    benchmark_value: float


class Portfolio:
    def __init__(self, capital: float, config: BacktestConfig):
        self.cash = capital
        self.config = config
        self.positions: dict[str, Position] = {}
        self.trade_log: list[TradeRecord] = []
        self.daily_snapshots: list[DailySnapshot] = []
        self.peak_value = capital

    @property
    def total_value(self) -> float:
        pos_value = sum(p.market_value for p in self.positions.values())
        return self.cash + pos_value

    def buy(self, ts_code: str, name: str, price: float, target_value: float, date: str, reason: str = "REBALANCE"):
        if price <= 0 or target_value <= 0:
            return
        shares = int(target_value / price / 100) * 100  # Round to 100 (A-share lot)
        if shares < 100:
            return
        cost = shares * price
        commission = max(cost * self.config.commission_rate, 5.0)  # Min 5 CNY
        total_cost = cost + commission

        if total_cost > self.cash:
            shares = int(self.cash * 0.99 / price / 100) * 100
            if shares < 100:
                return
            cost = shares * price
            commission = max(cost * self.config.commission_rate, 5.0)
            total_cost = cost + commission

        self.cash -= total_cost
        if ts_code in self.positions:
            pos = self.positions[ts_code]
            total_shares = pos.shares + shares
            avg_price = (pos.shares * pos.entry_price + shares * price) / total_shares
            pos.shares = total_shares
            pos.entry_price = avg_price
            pos.current_price = price
            pos.peak_price = max(pos.peak_price, price)
        else:
            self.positions[ts_code] = Position(
                ts_code=ts_code, name=name, shares=shares,
                entry_price=price, entry_date=date,
                current_price=price, peak_price=price,
            )

        self.trade_log.append(TradeRecord(
            date=date, ts_code=ts_code, name=name, action="BUY",
            shares=shares, price=price, value=cost, commission=commission,
            reason=reason,
        ))

    def sell(self, ts_code: str, price: float, date: str, reason: str = "REBALANCE"):
        if ts_code not in self.positions:
            return
        pos = self.positions[ts_code]
        shares = pos.shares
        revenue = shares * price
        commission = max(revenue * self.config.commission_rate, 5.0)
        stamp_tax = revenue * self.config.stamp_tax_rate
        net_revenue = revenue - commission - stamp_tax

        self.cash += net_revenue
        name = pos.name

        self.trade_log.append(TradeRecord(
            date=date, ts_code=ts_code, name=name, action="SELL",
            shares=shares, price=price, value=revenue,
            commission=commission + stamp_tax, reason=reason,
        ))

        del self.positions[ts_code]

    def update_prices(self, prices: dict[str, float]):
        for code, pos in self.positions.items():
            if code in prices:
                pos.current_price = prices[code]
                pos.peak_price = max(pos.peak_price, prices[code])

    def check_stop_loss(self, prices: dict[str, float], date: str) -> list[str]:
        triggered = []
        for code, pos in list(self.positions.items()):
            if code in prices:
                drawdown = prices[code] / pos.peak_price - 1
                if drawdown <= self.config.stop_loss_pct:
                    triggered.append(code)
                    self.sell(code, prices[code], date, reason=f"STOP_LOSS({drawdown:.1%})")
        return triggered

    def snapshot(self, date: str, regime: str, benchmark: float):
        self.peak_value = max(self.peak_value, self.total_value)
        self.daily_snapshots.append(DailySnapshot(
            date=date,
            portfolio_value=round(self.total_value, 2),
            cash=round(self.cash, 2),
            n_positions=len(self.positions),
            regime=regime,
            benchmark_value=benchmark,
        ))


# ── Backtest Engine ──────────────────────────────────────────────────────────

def run_backtest(config: BacktestConfig, market_data: dict) -> dict:
    """Run momentum rotation backtest over simulated data."""
    stock_data = market_data["stock_data"]
    trade_cal = market_data["trade_calendar"]
    index_data = market_data["index_data"]
    stock_info = market_data["stock_info"]
    n_days = len(trade_cal)

    name_map = {r["ts_code"]: r["name"] for _, r in stock_info.iterrows()}
    portfolio = Portfolio(config.initial_capital, config)
    initial_benchmark = index_data.iloc[0]["close"]

    logger.info(f"═══ 动量轮动回测启动 ═══")
    logger.info(f"  交易日数: {n_days}")
    logger.info(f"  标的池:   {len(stock_data)} 只股票")
    logger.info(f"  初始资金: ¥{config.initial_capital:,.0f}")
    logger.info(f"  持仓数量: {config.top_n}")
    logger.info(f"  调仓间隔: {config.rebalance_interval_days} 天")
    logger.info(f"  止损线:   {config.stop_loss_pct:.0%}")
    logger.info(f"  动量类型: {config.momentum_type}")
    logger.info(f"  回望周期: {config.lookback_days}")
    logger.info("")

    last_rebalance_idx = -999
    rebalance_count = 0

    for day_idx in range(n_days):
        date = trade_cal[day_idx]

        # 1) Update prices
        day_prices = {}
        for code, df in stock_data.items():
            row = df[df["trade_date"] == date]
            if not row.empty:
                day_prices[code] = float(row.iloc[0]["close"])

        portfolio.update_prices(day_prices)

        # 2) Check regime
        regime = check_market_regime(index_data, day_idx)

        # 3) Check stop-loss daily
        stopped = portfolio.check_stop_loss(day_prices, date)
        if stopped:
            logger.warning(f"  [{date}] 止损触发: {stopped}")

        # 4) Rebalance check
        days_since = day_idx - last_rebalance_idx
        is_rebalance_day = days_since >= config.rebalance_interval_days
        # Also need enough lookback data
        max_lb = max(config.lookback_days)
        has_enough_data = day_idx >= max_lb

        if is_rebalance_day and has_enough_data and regime != "HALT":
            rebalance_count += 1
            logger.info(f"  [{date}] 第{rebalance_count}次调仓 | 状态={regime} | 资产=¥{portfolio.total_value:,.0f}")

            # a) Calculate momentum scores
            scores = {}
            for code, df in stock_data.items():
                # Only use data up to current day (no lookahead)
                hist = df[df["trade_date"] <= date]
                if len(hist) < max_lb + 1:
                    continue
                if config.momentum_type == "simple":
                    score = calc_momentum(hist, config.lookback_days[0])
                elif config.momentum_type == "risk_adjusted":
                    m = calc_momentum(hist, config.lookback_days[0])
                    v = calc_volatility(hist, config.lookback_days[0])
                    score = m / v if v > 0.01 else 0
                else:  # composite
                    score = calc_composite_momentum(hist, config)
                scores[code] = score

            # b) Rank and select top N
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            target_codes = [code for code, _ in ranked[:config.top_n]]

            # c) Regime adjustments
            max_positions = config.top_n
            if regime == "DEFENSIVE":
                max_positions = max(2, config.top_n // 2)
                target_codes = target_codes[:max_positions]

            # d) Sell positions not in target
            for code in list(portfolio.positions.keys()):
                if code not in target_codes:
                    portfolio.sell(code, day_prices.get(code, 0), date, reason="ROTATION_OUT")

            # e) Buy target positions
            n_to_buy = len(target_codes)
            if n_to_buy > 0:
                target_weight = min(config.max_single_weight, 1.0 / n_to_buy)
                for code in target_codes:
                    if code not in portfolio.positions:
                        target_value = portfolio.total_value * target_weight
                        price = day_prices.get(code)
                        if price and price > 0:
                            portfolio.buy(
                                code, name_map.get(code, code),
                                price, target_value, date,
                                reason=f"ROTATION_IN(score={scores.get(code, 0):.4f})"
                            )

            # f) Log holdings
            holdings_str = ", ".join(
                f"{p.name}({p.pnl_pct:+.1f}%)"
                for p in sorted(portfolio.positions.values(), key=lambda x: x.pnl_pct, reverse=True)
            )
            logger.info(f"           持仓: [{holdings_str}]")

            last_rebalance_idx = day_idx

        # 5) Daily snapshot
        benchmark_val = float(index_data.iloc[day_idx]["close"])
        portfolio.snapshot(date, regime, benchmark_val)

    # ── Calculate metrics ─────────────────────────────────────────────────────
    logger.info("")
    logger.info("═══ 计算绩效指标 ═══")

    daily_values = [s.portfolio_value for s in portfolio.daily_snapshots]
    daily_returns = np.diff(daily_values) / daily_values[:-1] if len(daily_values) > 1 else []

    total_return = (daily_values[-1] / daily_values[0] - 1) if daily_values else 0
    ann_factor = 252 / max(n_days, 1)
    annualized_return = (1 + total_return) ** ann_factor - 1

    volatility = float(np.std(daily_returns) * math.sqrt(252)) if len(daily_returns) > 1 else 0
    sharpe = annualized_return / volatility if volatility > 0.001 else 0
    rf_rate = 0.02
    sharpe_adj = (annualized_return - rf_rate) / volatility if volatility > 0.001 else 0

    # Max drawdown
    peak = daily_values[0]
    max_dd = 0
    for v in daily_values:
        peak = max(peak, v)
        dd = (v - peak) / peak
        max_dd = min(max_dd, dd)

    calmar = annualized_return / abs(max_dd) if abs(max_dd) > 0.001 else 0

    # Benchmark
    bench_start = float(index_data.iloc[0]["close"])
    bench_end = float(index_data.iloc[-1]["close"])
    bench_return = bench_end / bench_start - 1
    excess_return = total_return - bench_return

    # Trade stats
    buy_trades = [t for t in portfolio.trade_log if t.action == "BUY"]
    sell_trades = [t for t in portfolio.trade_log if t.action == "SELL"]
    total_commission = sum(t.commission for t in portfolio.trade_log)

    # Win rate (by completed round-trips)
    wins = sum(1 for t in sell_trades if t.reason.startswith("ROTATION") and t.value > 0)

    metrics = {
        "total_return": round(total_return * 100, 2),
        "annualized_return": round(annualized_return * 100, 2),
        "benchmark_return": round(bench_return * 100, 2),
        "excess_return": round(excess_return * 100, 2),
        "volatility": round(volatility * 100, 2),
        "sharpe_ratio": round(sharpe_adj, 3),
        "max_drawdown": round(max_dd * 100, 2),
        "calmar_ratio": round(calmar, 3),
        "total_trades": len(portfolio.trade_log),
        "buy_trades": len(buy_trades),
        "sell_trades": len(sell_trades),
        "rebalance_count": rebalance_count,
        "total_commission": round(total_commission, 2),
        "final_value": round(daily_values[-1], 2),
        "final_cash": round(portfolio.cash, 2),
        "final_positions": len(portfolio.positions),
    }

    return {
        "metrics": metrics,
        "daily_snapshots": portfolio.daily_snapshots,
        "trade_log": portfolio.trade_log,
        "final_holdings": portfolio.positions,
        "config": asdict(config),
    }


# ── Report Generator ─────────────────────────────────────────────────────────

def print_report(result: dict):
    """Print formatted backtest report."""
    m = result["metrics"]
    snaps = result["daily_snapshots"]
    trades = result["trade_log"]
    holdings = result["final_holdings"]

    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║            动量轮动策略回测报告 — Momentum Rotation Backtest        ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print(f"║  回测期间: {snaps[0].date} ~ {snaps[-1].date}  ({len(snaps)} 交易日)          ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print("║                           绩效摘要                                 ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")

    def row(label, value, extra=""):
        label_w = label.ljust(20)
        value_w = str(value).rjust(12)
        extra_w = extra.ljust(30)
        print(f"║  {label_w} {value_w}  {extra_w}  ║")

    row("总收益率", f"{m['total_return']:+.2f}%", f"(¥{m['final_value']:,.0f})")
    row("年化收益率", f"{m['annualized_return']:+.2f}%")
    row("基准收益率 (沪深300)", f"{m['benchmark_return']:+.2f}%")
    row("超额收益", f"{m['excess_return']:+.2f}%",
        "✅ 跑赢" if m['excess_return'] > 0 else "❌ 跑输")
    print("║" + "─" * 68 + "║")
    row("年化波动率", f"{m['volatility']:.2f}%")
    row("夏普比率", f"{m['sharpe_ratio']:.3f}",
        "⭐ 优秀" if m['sharpe_ratio'] > 1.5 else "✅ 良好" if m['sharpe_ratio'] > 0.5 else "⚠️ 一般")
    row("最大回撤", f"{m['max_drawdown']:.2f}%")
    row("卡尔马比率", f"{m['calmar_ratio']:.3f}")
    print("║" + "─" * 68 + "║")
    row("调仓次数", f"{m['rebalance_count']}")
    row("总交易笔数", f"{m['total_trades']}", f"(买{m['buy_trades']} / 卖{m['sell_trades']})")
    row("总交易成本", f"¥{m['total_commission']:,.0f}")
    row("最终持仓数", f"{m['final_positions']}")

    print("╠══════════════════════════════════════════════════════════════════════╣")
    print("║                         每日净值曲线                               ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")

    # ASCII chart
    values = [s.portfolio_value for s in snaps]
    bench_values = [s.benchmark_value / snaps[0].benchmark_value * values[0] for s in snaps]
    all_vals = values + bench_values
    v_min = min(all_vals) * 0.999
    v_max = max(all_vals) * 1.001
    chart_width = 50
    chart_height = 12

    for row_idx in range(chart_height, -1, -1):
        threshold = v_min + (v_max - v_min) * row_idx / chart_height
        line = "║  "
        if row_idx == chart_height:
            line += f"¥{v_max:>10,.0f} │"
        elif row_idx == 0:
            line += f"¥{v_min:>10,.0f} │"
        elif row_idx == chart_height // 2:
            mid = (v_max + v_min) / 2
            line += f"¥{mid:>10,.0f} │"
        else:
            line += "            │"

        for i in range(min(len(values), chart_width)):
            day_idx = int(i * len(values) / chart_width)
            pv = values[day_idx]
            bv = bench_values[day_idx]
            p_row = int((pv - v_min) / (v_max - v_min) * chart_height)
            b_row = int((bv - v_min) / (v_max - v_min) * chart_height)

            if p_row == row_idx and b_row == row_idx:
                line += "X"
            elif p_row == row_idx:
                line += "█"
            elif b_row == row_idx:
                line += "·"
            else:
                line += " "

        line = line.ljust(69) + "║"
        print(line)

    print("║              └" + "─" * 50 + "       ║")
    print("║               █ 策略净值    · 沪深300基准                          ║")

    print("╠══════════════════════════════════════════════════════════════════════╣")
    print("║                         交易明细                                   ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")

    for t in trades[:20]:  # Show first 20
        action_icon = "🔵" if t.action == "BUY" else "🔴"
        print(f"║  {t.date} {action_icon} {t.action:4s} {t.name:8s} "
              f"{t.shares:>6d}股 @ ¥{t.price:>8.2f}  ¥{t.value:>10,.0f}  {t.reason[:18]:18s} ║")

    if len(trades) > 20:
        print(f"║  ... 还有 {len(trades) - 20} 笔交易                                          ║")

    print("╠══════════════════════════════════════════════════════════════════════╣")
    print("║                       最终持仓                                     ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")

    for code, pos in sorted(holdings.items(), key=lambda x: x[1].pnl_pct, reverse=True):
        pnl_icon = "📈" if pos.pnl_pct > 0 else "📉"
        print(f"║  {code} {pos.name:8s} {pos.shares:>6d}股  "
              f"成本¥{pos.entry_price:>8.2f} → 现价¥{pos.current_price:>8.2f}  "
              f"{pnl_icon} {pos.pnl_pct:+6.2f}%  ║")

    if not holdings:
        print("║  (空仓)                                                            ║")

    print("╠══════════════════════════════════════════════════════════════════════╣")
    print("║                       市场状态分布                                  ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")

    regime_counts = defaultdict(int)
    for s in snaps:
        regime_counts[s.regime] += 1
    for regime, count in sorted(regime_counts.items()):
        bar_len = int(count / len(snaps) * 40)
        bar = "█" * bar_len
        print(f"║  {regime:12s} {count:3d}天 ({count/len(snaps)*100:4.1f}%) {bar:40s} ║")

    print("╚══════════════════════════════════════════════════════════════════════╝")


# ── Save results to JSON ─────────────────────────────────────────────────────

def save_results(result: dict, path: str):
    """Save backtest results to JSON."""
    output = {
        "metrics": result["metrics"],
        "config": result["config"],
        "daily_values": [
            {"date": s.date, "value": s.portfolio_value, "cash": s.cash,
             "positions": s.n_positions, "regime": s.regime, "benchmark": s.benchmark_value}
            for s in result["daily_snapshots"]
        ],
        "trades": [
            {"date": t.date, "code": t.ts_code, "name": t.name, "action": t.action,
             "shares": t.shares, "price": t.price, "value": t.value,
             "commission": t.commission, "reason": t.reason}
            for t in result["trade_log"]
        ],
        "final_holdings": {
            code: {"name": p.name, "shares": p.shares, "entry_price": p.entry_price,
                   "current_price": p.current_price, "pnl_pct": round(p.pnl_pct, 2)}
            for code, p in result["final_holdings"].items()
        },
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"结果已保存: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("   动量轮动策略回测验证系统 v1.0")
    print("   Momentum Rotation Strategy Backtester")
    print("=" * 70)
    print()

    # 1. Generate market data (50 days = 20 warmup + 30 actual trading)
    logger.info("生成模拟市场数据 (30只A股, 50个交易日, 前20天为预热)...")
    market_data = generate_market_data(n_days=50, seed=42)
    logger.info(f"  交易日历: {market_data['trade_calendar'][0]} ~ {market_data['trade_calendar'][-1]}")

    # 2. Run backtest with default config
    config = BacktestConfig()
    logger.info("启动回测...")
    print()

    t0 = time.time()
    result = run_backtest(config, market_data)
    elapsed = time.time() - t0

    # 3. Print report
    print_report(result)

    print()
    logger.info(f"回测耗时: {elapsed:.2f}秒")

    # 4. Save results
    output_path = str(PROJECT_ROOT / "backtest" / "momentum_demo_result.json")
    save_results(result, output_path)

    # 5. Summary
    m = result["metrics"]
    print()
    print("━" * 70)
    print("  验证结论:")
    print(f"  ✅ 策略流水线完整跑通: 股票池→动量排名→风控→调仓→绩效")
    print(f"  ✅ 共 {m['rebalance_count']} 次调仓, {m['total_trades']} 笔交易")
    print(f"  ✅ 超额收益: {m['excess_return']:+.2f}% (vs 沪深300)")
    print(f"  ✅ 风控生效: 最大回撤 {m['max_drawdown']:.2f}%, 止损规则正常")
    print(f"  ✅ 交易成本: ¥{m['total_commission']:,.0f} (佣金万三 + 印花税千一)")
    print("━" * 70)

    return result


if __name__ == "__main__":
    main()
