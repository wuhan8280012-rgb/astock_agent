#!/usr/bin/env python3
"""
动量轮动策略 — 优化版回测引擎

基于 v1.0 基线回测的数据分析，实施以下 6 大优化：

┌──────────────────────────────────────────────────────────────────────┐
│  优化项            │  基线 v1.0          │  优化 v2.0              │
├──────────────────────────────────────────────────────────────────────┤
│ 1 止损线           │ -8% (91%触发)       │ -15% (68%触发)         │
│ 2 动量周期权重     │ 5/10/20 = 0.5/0.3/0.2 │ 5/10/20 = 0.1/0.3/0.6 │
│ 3 流动性加权       │ 无                   │ 成交额对数加权          │
│ 4 波动率过滤       │ 仅惩罚项             │ 过滤>P90 + 惩罚加大     │
│ 5 换手约束         │ 无限制               │ 最大换手50%/次          │
│ 6 持仓缓冲带       │ 排名外即卖           │ TOP N*1.5 内保留       │
└──────────────────────────────────────────────────────────────────────┘

同时运行 baseline 和 optimized 两套参数进行对照。

Usage:
    python scripts/run_optimized_backtest.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class _Logger:
    def __init__(self, verbose=True):
        self.verbose = verbose
    def info(self, msg, *a, **kw):
        if self.verbose: print(f"[INFO]  {msg}")
    def warning(self, msg, *a, **kw):
        if self.verbose: print(f"[WARN]  {msg}")
    def error(self, msg, *a, **kw): print(f"[ERROR] {msg}")
    def debug(self, msg, *a, **kw): pass

logger = _Logger(verbose=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestConfig:
    name: str = "baseline"

    # Momentum
    lookback_days: list[int] = field(default_factory=lambda: [5, 10, 20])
    lookback_weights: list[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])
    volatility_penalty: float = 0.5

    # Portfolio
    top_n: int = 10
    rebalance_interval_days: int = 5
    max_single_weight: float = 0.15
    max_total_position: float = 0.80
    stop_loss_pct: float = -0.08
    initial_capital: float = 1_000_000.0

    # Execution
    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.001
    slippage_pct: float = 0.001

    # Universe
    min_amount_20d: float = 1e8    # Min 20d avg amount (yuan)
    min_price: float = 5.0
    max_price: float = 500.0
    exclude_st: bool = True

    # Warmup
    warmup_days: int = 25

    # ── v2.0 Optimizations ──
    # Opt1: Adaptive stop-loss (use ATR-based instead of fixed)
    use_atr_stop: bool = False
    atr_stop_multiplier: float = 2.5  # N * ATR(20) as stop distance

    # Opt3: Liquidity score weight in momentum
    liquidity_weight: float = 0.0     # 0=off, >0 = blend with momentum

    # Opt4: Volatility filter threshold (percentile)
    max_volatility_pctile: float = 1.0  # 1.0=no filter, 0.9=filter top 10%

    # Opt5: Max turnover per rebalance
    max_turnover_pct: float = 1.0     # 1.0=no limit, 0.5=max 50%

    # Opt6: Buffer zone for existing holdings
    hold_buffer_ratio: float = 1.0    # 1.0=no buffer, 1.5=keep if in top N*1.5


# ══════════════════════════════════════════════════════════════════════════════
#  Data Loader
# ══════════════════════════════════════════════════════════════════════════════

def load_market_data(csv_path: str | Path) -> dict:
    raw = pd.read_csv(csv_path, low_memory=False)
    daily_df = raw[raw["data_type"] == "daily"].copy()
    index_df = raw[raw["data_type"] == "index_daily"].copy()
    basic_df = raw[raw["data_type"] == "stock_basic"].copy()
    cal_df = raw[raw["data_type"] == "trade_cal"].copy()

    for col in ["trade_date", "open", "high", "low", "close", "vol", "amount", "pct_chg"]:
        if col in daily_df.columns:
            daily_df[col] = pd.to_numeric(daily_df[col], errors="coerce")
    for col in ["trade_date", "close", "pct_chg"]:
        if col in index_df.columns:
            index_df[col] = pd.to_numeric(index_df[col], errors="coerce")

    daily_df["trade_date"] = daily_df["trade_date"].astype(int).astype(str)
    index_df["trade_date"] = index_df["trade_date"].astype(int).astype(str)
    cal_df["cal_date"] = cal_df["cal_date"].astype(int).astype(str)
    trade_dates = sorted(cal_df[cal_df["is_open"] == 1]["cal_date"].tolist())

    stock_data = {}
    for ts_code, grp in daily_df.groupby("ts_code"):
        stock_data[ts_code] = grp.sort_values("trade_date").reset_index(drop=True)

    stock_info = basic_df[["ts_code", "name", "industry", "market", "list_date"]].copy()

    return {
        "stock_data": stock_data,
        "trade_calendar": trade_dates,
        "index_data": index_df.sort_values("trade_date").reset_index(drop=True),
        "stock_info": stock_info,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Momentum & Scoring
# ══════════════════════════════════════════════════════════════════════════════

def calc_composite_momentum(closes: np.ndarray, config: BacktestConfig) -> float:
    score = 0.0
    for lb, w in zip(config.lookback_days, config.lookback_weights):
        if len(closes) < lb + 1:
            return np.nan
        m = closes[-1] / closes[-(lb + 1)] - 1
        rets = np.diff(closes[-lb - 1:]) / closes[-lb - 1:-1]
        v = float(np.std(rets)) if len(rets) > 1 else 0
        score += w * (m - config.volatility_penalty * v)
    return score


def calc_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 20) -> float:
    """Average True Range over period."""
    if len(closes) < period + 1:
        return np.nan
    trs = []
    for i in range(-period, 0):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )
        trs.append(tr)
    return np.mean(trs)


def calc_liquidity_score(amounts: np.ndarray, lookback: int = 20) -> float:
    """Log-scaled liquidity score."""
    if len(amounts) < lookback:
        return 0.0
    avg = np.mean(amounts[-lookback:])
    return np.log10(max(avg * 1000, 1))  # amount in thousands -> yuan


# ══════════════════════════════════════════════════════════════════════════════
#  Universe Filter
# ══════════════════════════════════════════════════════════════════════════════

def filter_universe(stock_data: dict, stock_info: pd.DataFrame,
                    date: str, config: BacktestConfig,
                    vol_threshold: float = None) -> list[str]:
    info_map = {}
    for _, row in stock_info.iterrows():
        info_map[row["ts_code"]] = row

    candidates = []
    for ts_code, df in stock_data.items():
        hist = df[df["trade_date"] <= date]
        if len(hist) < max(config.lookback_days) + 5:
            continue

        latest = hist.iloc[-1]
        close = latest["close"]

        if close < config.min_price or close > config.max_price:
            continue

        # Volume filter
        recent = hist.tail(20)
        avg_amount = recent["amount"].mean() * 1000
        if avg_amount < config.min_amount_20d:
            continue

        # ST filter
        info = info_map.get(ts_code)
        if config.exclude_st and info is not None:
            name = str(info.get("name", ""))
            if "ST" in name:
                continue

        # Suspension check
        if hist.iloc[-1]["trade_date"] != date:
            continue

        # Limit-up check (can't buy)
        if latest["pct_chg"] >= 9.5:
            continue

        # Opt4: Volatility filter
        if vol_threshold is not None:
            stock_vol = recent["pct_chg"].std()
            if stock_vol > vol_threshold:
                continue

        candidates.append(ts_code)

    return candidates


# ══════════════════════════════════════════════════════════════════════════════
#  Market Regime
# ══════════════════════════════════════════════════════════════════════════════

def check_market_regime(index_data: pd.DataFrame, date: str) -> str:
    hist = index_data[index_data["trade_date"] <= date].tail(6)
    if len(hist) < 3:
        return "RUN"
    total_change = (hist.iloc[-1]["close"] / hist.iloc[0]["close"] - 1) * 100
    avg_pct = hist["pct_chg"].mean()
    hist_20 = index_data[index_data["trade_date"] <= date].tail(20)
    ma20 = hist_20["close"].mean()
    current = hist.iloc[-1]["close"]
    ma_trend = (current / ma20 - 1) * 100

    if total_change < -5 or avg_pct < -1.0:
        return "HALT"
    elif total_change < -2 or avg_pct < -0.3 or ma_trend < -3:
        return "DEFENSIVE"
    elif total_change > 3 and avg_pct > 0.3:
        return "STRONG_RUN"
    else:
        return "RUN"


# ══════════════════════════════════════════════════════════════════════════════
#  Portfolio
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Position:
    code: str
    name: str
    shares: int
    cost_price: float
    entry_date: str
    peak_price: float = 0.0
    current_price: float = 0.0
    atr_at_entry: float = 0.0  # For ATR-based stop

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def pnl_pct(self) -> float:
        return (self.current_price / self.cost_price - 1) if self.cost_price > 0 else 0


class Portfolio:
    def __init__(self, capital: float, config: BacktestConfig, stock_names: dict):
        self.cash = capital
        self.initial_capital = capital
        self.config = config
        self.stock_names = stock_names
        self.positions: dict[str, Position] = {}
        self.trades: list[dict] = []
        self.snapshots: list[dict] = []
        self.peak_value = capital

    @property
    def position_value(self) -> float:
        return sum(p.market_value for p in self.positions.values())

    @property
    def total_value(self) -> float:
        return self.cash + self.position_value

    def update_prices(self, prices: dict[str, float]):
        for code, p in self.positions.items():
            if code in prices:
                p.current_price = prices[code]
                p.peak_price = max(p.peak_price, prices[code])
        self.peak_value = max(self.peak_value, self.total_value)

    def buy(self, code: str, price: float, date: str, target_weight: float = None,
            reason: str = "", atr: float = 0) -> bool:
        if target_weight is None:
            target_weight = self.config.max_single_weight
        target_value = min(self.total_value * target_weight, self.cash * 0.95)
        exec_price = price * (1 + self.config.slippage_pct)
        if target_value < exec_price * 100:
            return False
        shares = int(target_value / exec_price / 100) * 100
        if shares <= 0:
            return False
        cost = shares * exec_price
        commission = max(cost * self.config.commission_rate, 5)
        total_cost = cost + commission
        if total_cost > self.cash:
            return False
        if (self.position_value + cost) / self.total_value > self.config.max_total_position:
            max_add = self.total_value * self.config.max_total_position - self.position_value
            if max_add < exec_price * 100:
                return False
            shares = int(max_add / exec_price / 100) * 100
            if shares <= 0:
                return False
            cost = shares * exec_price
            commission = max(cost * self.config.commission_rate, 5)
            total_cost = cost + commission

        self.cash -= total_cost
        name = self.stock_names.get(code, code)
        self.positions[code] = Position(
            code=code, name=name, shares=shares,
            cost_price=exec_price, entry_date=date,
            peak_price=price, current_price=price,
            atr_at_entry=atr,
        )
        self.trades.append({
            "date": date, "code": code, "name": name, "direction": "BUY",
            "shares": shares, "price": round(exec_price, 2), "amount": round(cost, 2),
            "cost": round(commission, 2), "reason": reason,
        })
        return True

    def sell(self, code: str, price: float, date: str, reason: str = "") -> float:
        if code not in self.positions:
            return 0
        pos = self.positions[code]
        exec_price = price * (1 - self.config.slippage_pct)
        cost = pos.shares * exec_price
        commission = max(cost * self.config.commission_rate, 5)
        stamp_tax = cost * self.config.stamp_tax_rate
        self.cash += cost - commission - stamp_tax
        self.trades.append({
            "date": date, "code": code, "name": pos.name, "direction": "SELL",
            "shares": pos.shares, "price": round(exec_price, 2), "amount": round(cost, 2),
            "cost": round(commission + stamp_tax, 2), "reason": reason,
        })
        sold_value = cost
        del self.positions[code]
        return sold_value

    def check_stop_loss(self, prices: dict, date: str, stock_data: dict = None) -> list[str]:
        stopped = []
        for code in list(self.positions.keys()):
            pos = self.positions[code]
            if code not in prices:
                continue

            if self.config.use_atr_stop and pos.atr_at_entry > 0:
                # ATR-based trailing stop
                stop_price = pos.peak_price - self.config.atr_stop_multiplier * pos.atr_at_entry
                if prices[code] <= stop_price:
                    drop = prices[code] / pos.peak_price - 1
                    self.sell(code, prices[code], date, f"ATR_STOP({drop:.1%})")
                    stopped.append(f"{pos.name}({drop:.1%})")
            else:
                # Fixed percentage trailing stop
                drop = prices[code] / pos.peak_price - 1
                if drop <= self.config.stop_loss_pct:
                    self.sell(code, prices[code], date, f"STOP_LOSS({drop:.1%})")
                    stopped.append(f"{pos.name}({drop:.1%})")
        return stopped

    def take_snapshot(self, date: str, regime: str):
        self.snapshots.append({
            "date": date,
            "total_value": round(self.total_value, 2),
            "cash": round(self.cash, 2),
            "position_value": round(self.position_value, 2),
            "n_positions": len(self.positions),
            "regime": regime,
        })


# ══════════════════════════════════════════════════════════════════════════════
#  Backtest Engine
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(config: BacktestConfig, market_data: dict, verbose: bool = True) -> dict:
    global logger
    logger = _Logger(verbose=verbose)

    stock_data = market_data["stock_data"]
    trade_cal = market_data["trade_calendar"]
    index_data = market_data["index_data"]
    stock_info = market_data["stock_info"]
    n_days = len(trade_cal)

    stock_names = {}
    for _, row in stock_info.iterrows():
        stock_names[str(row["ts_code"])] = str(row.get("name", row["ts_code"]))

    portfolio = Portfolio(config.initial_capital, config, stock_names)

    logger.info(f"═══ [{config.name}] 动量轮动回测 ═══")
    logger.info(f"  止损: {config.stop_loss_pct:.0%} | ATR止损: {config.use_atr_stop}")
    logger.info(f"  动量权重: {config.lookback_weights} | 波动惩罚: {config.volatility_penalty}")
    logger.info(f"  流动性权重: {config.liquidity_weight} | 波动过滤: P{config.max_volatility_pctile:.0%}")
    logger.info(f"  换手上限: {config.max_turnover_pct:.0%} | 缓冲带: {config.hold_buffer_ratio}x")
    logger.info("")

    # Pre-compute volatility threshold if needed
    vol_threshold = None
    if config.max_volatility_pctile < 1.0:
        all_vols = []
        for _, df in stock_data.items():
            if len(df) >= 20:
                all_vols.append(df["pct_chg"].std())
        vol_threshold = np.percentile(all_vols, config.max_volatility_pctile * 100)
        logger.info(f"  波动率过滤阈值: {vol_threshold:.2f}%")

    last_rebalance_idx = -config.rebalance_interval_days
    rebalance_count = 0
    regime_counts = defaultdict(int)
    stop_loss_count = 0

    for day_idx in range(n_days):
        date = trade_cal[day_idx]

        day_prices = {}
        for code, df in stock_data.items():
            row = df[df["trade_date"] == date]
            if not row.empty:
                day_prices[code] = float(row.iloc[0]["close"])

        portfolio.update_prices(day_prices)
        regime = check_market_regime(index_data, date)
        regime_counts[regime] += 1

        stopped = portfolio.check_stop_loss(day_prices, date, stock_data)
        if stopped:
            stop_loss_count += len(stopped)
            logger.warning(f"  [{date}] 止损: {', '.join(stopped)}")

        if day_idx < config.warmup_days:
            portfolio.take_snapshot(date, regime)
            continue

        days_since = day_idx - last_rebalance_idx
        if days_since >= config.rebalance_interval_days and regime != "HALT":
            rebalance_count += 1

            # Universe
            universe = filter_universe(stock_data, stock_info, date, config, vol_threshold)

            # Score all candidates
            scores = {}
            atrs = {}
            for code in universe:
                df = stock_data[code]
                hist = df[df["trade_date"] <= date]
                closes = hist["close"].values
                if len(closes) < max(config.lookback_days) + 1:
                    continue

                # Momentum score
                mscore = calc_composite_momentum(closes, config)
                if np.isnan(mscore):
                    continue

                # Opt3: Liquidity blending
                if config.liquidity_weight > 0:
                    liq = calc_liquidity_score(hist["amount"].values)
                    # Normalize: liq is log10(amount), typically 7~10
                    liq_norm = (liq - 7) / 3  # Map [7,10] -> [0,1]
                    liq_norm = max(0, min(1, liq_norm))
                    mscore = mscore * (1 - config.liquidity_weight) + liq_norm * config.liquidity_weight * abs(mscore)

                scores[code] = mscore

                # ATR for stop-loss
                if config.use_atr_stop:
                    highs = hist["high"].values
                    lows = hist["low"].values
                    atr = calc_atr(highs, lows, closes)
                    atrs[code] = atr if not np.isnan(atr) else 0

            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

            # Opt6: Buffer zone — current holdings get to stay if in top N*buffer
            buffer_size = int(config.top_n * config.hold_buffer_ratio)
            top_buffer = set(code for code, _ in ranked[:buffer_size])

            # Target: top_n, adjusted by regime
            if regime == "DEFENSIVE":
                n_hold = max(3, config.top_n // 2)
            else:
                n_hold = config.top_n

            target_codes = [code for code, _ in ranked[:n_hold]]

            # Opt5: Turnover control — limit sells
            current_holdings = set(portfolio.positions.keys())
            to_sell = [c for c in current_holdings if c not in top_buffer]

            if config.max_turnover_pct < 1.0:
                max_sell_value = portfolio.total_value * config.max_turnover_pct
                sold_value = 0
                limited_sells = []
                # Sort by PnL ascending (sell worst first)
                to_sell_sorted = sorted(
                    to_sell,
                    key=lambda c: portfolio.positions[c].pnl_pct if c in portfolio.positions else 0
                )
                for code in to_sell_sorted:
                    if code in portfolio.positions:
                        pos_val = portfolio.positions[code].market_value
                        if sold_value + pos_val <= max_sell_value:
                            limited_sells.append(code)
                            sold_value += pos_val
                to_sell = limited_sells

            # Execute sells
            for code in to_sell:
                portfolio.sell(code, day_prices.get(code, 0), date, "ROTATION_OUT")

            # Execute buys
            target_weight = min(config.max_single_weight, config.max_total_position / n_hold)
            for code in target_codes:
                if code not in portfolio.positions and code in day_prices:
                    score = scores.get(code, 0)
                    atr = atrs.get(code, 0)
                    portfolio.buy(code, day_prices[code], date,
                                  target_weight=target_weight,
                                  reason=f"IN({score:+.3f})",
                                  atr=atr)

            logger.info(
                f"  [{date}] 调仓#{rebalance_count} | {regime} | "
                f"¥{portfolio.total_value:,.0f} | "
                f"卖{len(to_sell)}买{len([c for c in target_codes if c not in current_holdings and c in day_prices])} "
                f"| 持仓{len(portfolio.positions)}"
            )

            last_rebalance_idx = day_idx

        portfolio.take_snapshot(date, regime)

    # ── Metrics ──
    snaps = portfolio.snapshots
    active_snaps = [s for i, s in enumerate(snaps) if i >= config.warmup_days]
    values = [s["total_value"] for s in active_snaps]

    if len(values) < 2:
        return {"metrics": {}, "config_name": config.name}

    returns = pd.Series(values).pct_change().dropna()
    total_return = values[-1] / values[0] - 1
    n_active = len(values)
    ann_factor = 252 / max(n_active, 1)
    ann_return = (1 + total_return) ** ann_factor - 1
    ann_vol = float(returns.std() * np.sqrt(252)) if len(returns) > 1 else 0
    sharpe = (ann_return - 0.02) / ann_vol if ann_vol > 0.001 else 0

    peak = values[0]
    max_dd = 0
    for v in values:
        peak = max(peak, v)
        max_dd = min(max_dd, v / peak - 1)
    calmar = abs(ann_return / max_dd) if abs(max_dd) > 0.001 else 0

    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

    idx_active = index_data[index_data["trade_date"] >= active_snaps[0]["date"]]
    bench_return = (idx_active.iloc[-1]["close"] / idx_active.iloc[0]["close"] - 1) if len(idx_active) >= 2 else 0

    total_cost = sum(t["cost"] for t in portfolio.trades)
    turnover = sum(t["amount"] for t in portfolio.trades) / config.initial_capital

    return {
        "config_name": config.name,
        "config": asdict(config),
        "metrics": {
            "total_return": total_return,
            "ann_return": ann_return,
            "bench_return": bench_return,
            "excess_return": total_return - bench_return,
            "ann_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
            "rebalance_count": rebalance_count,
            "total_trades": len(portfolio.trades),
            "stop_loss_count": stop_loss_count,
            "total_cost": round(total_cost, 2),
            "turnover": round(turnover, 2),
            "final_value": round(values[-1], 2),
        },
        "snapshots": active_snaps,
        "trades": portfolio.trades,
        "positions": {
            code: {
                "name": p.name, "shares": p.shares,
                "cost": round(p.cost_price, 2), "price": round(p.current_price, 2),
                "pnl": round(p.pnl_pct * 100, 2),
            }
            for code, p in portfolio.positions.items()
        },
        "regime_distribution": dict(regime_counts),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Parameter Grid
# ══════════════════════════════════════════════════════════════════════════════

def build_configs() -> list[BacktestConfig]:
    configs = []

    # 0) Baseline — exact v1.0
    configs.append(BacktestConfig(
        name="baseline_v1.0",
        lookback_weights=[0.5, 0.3, 0.2],
        stop_loss_pct=-0.08,
        volatility_penalty=0.5,
        liquidity_weight=0.0,
        max_volatility_pctile=1.0,
        max_turnover_pct=1.0,
        hold_buffer_ratio=1.0,
    ))

    # 1) Opt1 only: Wider stop-loss -15%
    configs.append(BacktestConfig(
        name="opt1_stop15",
        lookback_weights=[0.5, 0.3, 0.2],
        stop_loss_pct=-0.15,
        volatility_penalty=0.5,
    ))

    # 2) Opt1+Opt2: Wider stop + long-term momentum
    configs.append(BacktestConfig(
        name="opt12_stop15_longmom",
        lookback_weights=[0.1, 0.3, 0.6],
        stop_loss_pct=-0.15,
        volatility_penalty=0.8,
    ))

    # 3) Opt1+2+3: + liquidity
    configs.append(BacktestConfig(
        name="opt123_+liquidity",
        lookback_weights=[0.1, 0.3, 0.6],
        stop_loss_pct=-0.15,
        volatility_penalty=0.8,
        liquidity_weight=0.15,
    ))

    # 4) Opt1+2+3+4: + vol filter
    configs.append(BacktestConfig(
        name="opt1234_+volfilter",
        lookback_weights=[0.1, 0.3, 0.6],
        stop_loss_pct=-0.15,
        volatility_penalty=0.8,
        liquidity_weight=0.15,
        max_volatility_pctile=0.85,
    ))

    # 5) Full optimized: all 6 opts
    configs.append(BacktestConfig(
        name="v2.0_full_optimized",
        lookback_weights=[0.1, 0.3, 0.6],
        stop_loss_pct=-0.15,
        volatility_penalty=0.8,
        liquidity_weight=0.15,
        max_volatility_pctile=0.85,
        max_turnover_pct=0.5,
        hold_buffer_ratio=1.5,
    ))

    # 6) ATR-based stop variant
    configs.append(BacktestConfig(
        name="v2.1_atr_stop",
        lookback_weights=[0.1, 0.3, 0.6],
        stop_loss_pct=-0.20,  # Fallback if ATR=0
        use_atr_stop=True,
        atr_stop_multiplier=2.5,
        volatility_penalty=0.8,
        liquidity_weight=0.15,
        max_volatility_pctile=0.85,
        max_turnover_pct=0.5,
        hold_buffer_ratio=1.5,
    ))

    return configs


# ══════════════════════════════════════════════════════════════════════════════
#  Comparison Report
# ══════════════════════════════════════════════════════════════════════════════

def print_comparison(results: list[dict]):
    print()
    print("=" * 110)
    print("  策略参数网格搜索结果对比")
    print("=" * 110)

    header = (
        f"{'策略':24s} │ {'收益':>7s} │ {'超额':>7s} │ {'夏普':>6s} │ "
        f"{'回撤':>7s} │ {'卡尔马':>7s} │ {'胜率':>5s} │ "
        f"{'止损':>4s} │ {'交易':>4s} │ {'换手':>5s} │ {'终值':>12s}"
    )
    print(header)
    print("─" * 110)

    best_sharpe = max(r["metrics"].get("sharpe_ratio", -999) for r in results if r["metrics"])

    for r in results:
        m = r.get("metrics", {})
        if not m:
            continue
        name = r["config_name"][:24]
        star = " ⭐" if m.get("sharpe_ratio", 0) == best_sharpe else ""
        print(
            f"  {name:22s} │ {m['total_return']:>+6.2%} │ {m['excess_return']:>+6.2%} │ "
            f"{m['sharpe_ratio']:>6.2f} │ {m['max_drawdown']:>6.2%} │ "
            f"{m['calmar_ratio']:>7.2f} │ {m['win_rate']:>4.1%} │ "
            f"{m['stop_loss_count']:>4} │ {m['total_trades']:>4} │ "
            f"{m['turnover']:>5.1f}x │ ¥{m['final_value']:>10,.0f}{star}"
        )

    print("─" * 110)

    # Find best by composite score (sharpe * (1 + excess) / (1 + |maxdd|))
    def composite_score(r):
        m = r.get("metrics", {})
        if not m:
            return -999
        s = m.get("sharpe_ratio", 0)
        e = m.get("excess_return", 0)
        d = abs(m.get("max_drawdown", -1))
        return s * (1 + e) / (1 + d)

    best = max(results, key=composite_score)
    bm = best["metrics"]
    print()
    print(f"  🏆 最优策略: {best['config_name']}")
    print(f"     收益 {bm['total_return']:+.2%} | 超额 {bm['excess_return']:+.2%} | "
          f"夏普 {bm['sharpe_ratio']:.2f} | 回撤 {bm['max_drawdown']:.2%} | "
          f"止损 {bm['stop_loss_count']}次")

    return best


def print_detailed_report(result: dict):
    """Print detailed report for the winning strategy."""
    m = result["metrics"]
    positions = result.get("positions", {})
    trades = result.get("trades", [])
    snaps = result.get("snapshots", [])

    W = 74

    def box(text):
        n_wide = sum(1 for c in text if ord(c) > 0x2E00)
        pad = max(0, W - len(text) - n_wide)
        print(f"║{text}{' ' * pad}║")

    def sep():
        print("╠" + "═" * W + "╣")

    print()
    print("╔" + "═" * W + "╗")
    box(f"  🏆 最优策略详细报告: {result['config_name']}")
    sep()

    box(f"  总收益:   {m['total_return']:>+8.2%}  (¥{m['final_value']:>12,.0f})")
    box(f"  超额收益: {m['excess_return']:>+8.2%}  (vs 沪深300 {m['bench_return']:+.2%})")
    box(f"  夏普:     {m['sharpe_ratio']:>8.3f}  年化波动: {m['ann_volatility']:.2%}")
    box(f"  最大回撤: {m['max_drawdown']:>8.2%}  卡尔马: {m['calmar_ratio']:.2f}")
    box(f"  日胜率:   {m['win_rate']:>8.1%}")
    box(f"  交易成本: ¥{m['total_cost']:>8,.0f}  换手率: {m['turnover']:.1f}x")
    box(f"  止损次数: {m['stop_loss_count']:>8}  调仓: {m['rebalance_count']}次")
    sep()

    # Net value chart
    values = [s["total_value"] for s in snaps]
    if len(values) > 1:
        box("                      ═══ 净值曲线 ═══")
        sep()
        chart_w = 55
        chart_h = 10
        min_v, max_v = min(values), max(values)
        rng = max_v - min_v if max_v > min_v else 1
        step = max(1, len(values) // chart_w)
        sampled = values[::step][:chart_w]

        box(f"  ¥{max_v:>10,.0f} ┐")
        for row in range(chart_h):
            threshold = max_v - (row + 0.5) * rng / chart_h
            chars = "".join("█" if v >= threshold else " " for v in sampled)
            box(f"               │{chars:{chart_w}s}")
        box(f"  ¥{min_v:>10,.0f} ┘{'─' * chart_w}")
        sep()

    # Final positions
    box(f"               ═══ 最终持仓 ({len(positions)}只) ═══")
    sep()
    if positions:
        for code, p in sorted(positions.items(), key=lambda x: -x[1].get("pnl", 0)):
            emoji = "📈" if p["pnl"] > 0 else "📉"
            name = p["name"][:6]
            box(f"  {code} {name:8s} {p['shares']:>6}股 ¥{p['cost']:>7.2f}→¥{p['price']:>7.2f} {emoji}{p['pnl']:>+6.2f}%")
    else:
        box("  (空仓)")

    print("╚" + "═" * W + "╝")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print()
    print("█" * 74)
    print("█  动量轮动策略优化 — 参数网格搜索 + 对比回测                       █")
    print("█  Momentum Rotation Optimizer — Grid Search & Comparison             █")
    print("█" * 74)
    print()

    # Load data
    data_path = None
    for p in [
        PROJECT_ROOT / "data" / "csi1000_market_bundle.csv",
        Path("/sessions/festive-jolly-meitner/mnt/uploads/csi1000_market_bundle.csv"),
    ]:
        if p.exists():
            data_path = p
            break

    if not data_path:
        print("[ERROR] 找不到数据文件")
        sys.exit(1)

    logger.info(f"加载数据: {data_path}")
    market_data = load_market_data(data_path)
    logger.info(f"  {len(market_data['stock_data'])}只股票, {len(market_data['trade_calendar'])}个交易日")
    print()

    # Build configs
    configs = build_configs()
    logger.info(f"将运行 {len(configs)} 套策略参数:")
    for c in configs:
        logger.info(f"  • {c.name}")
    print()

    # Run all
    results = []
    for i, cfg in enumerate(configs):
        print(f"\n{'─' * 74}")
        print(f"  [{i+1}/{len(configs)}] 运行: {cfg.name}")
        print(f"{'─' * 74}")
        t0 = time.time()
        result = run_backtest(cfg, market_data, verbose=True)
        elapsed = time.time() - t0
        results.append(result)
        m = result.get("metrics", {})
        if m:
            print(f"  → 收益 {m['total_return']:+.2%} | 夏普 {m['sharpe_ratio']:.2f} | "
                  f"回撤 {m['max_drawdown']:.2%} | 止损 {m['stop_loss_count']} | "
                  f"耗时 {elapsed:.1f}s")

    # Comparison
    best = print_comparison(results)

    # Detailed report for best
    print_detailed_report(best)

    # Save
    save_path = PROJECT_ROOT / "backtest" / "optimization_results.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        "metadata": {
            "type": "strategy_optimization",
            "run_time": datetime.now().isoformat(),
            "n_configs": len(configs),
            "data_source": "Tushare CSI1000",
        },
        "results": [
            {
                "config_name": r["config_name"],
                "metrics": r.get("metrics", {}),
                "positions": r.get("positions", {}),
            }
            for r in results
        ],
        "best_config": best.get("config", {}),
        "best_metrics": best.get("metrics", {}),
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    logger.info(f"\n优化结果已保存: {save_path}")

    # Summary
    bm = best["metrics"]
    base = results[0]["metrics"]
    print()
    print("━" * 74)
    print("  优化总结:")
    print(f"  基线 v1.0:  收益 {base['total_return']:+.2%} | 夏普 {base['sharpe_ratio']:.2f} | 回撤 {base['max_drawdown']:.2%} | 止损 {base['stop_loss_count']}次")
    print(f"  最优 {best['config_name'][:16]}:  收益 {bm['total_return']:+.2%} | 夏普 {bm['sharpe_ratio']:.2f} | 回撤 {bm['max_drawdown']:.2%} | 止损 {bm['stop_loss_count']}次")
    print()
    d_ret = bm['total_return'] - base['total_return']
    d_sharpe = bm['sharpe_ratio'] - base['sharpe_ratio']
    d_dd = bm['max_drawdown'] - base['max_drawdown']
    d_stop = bm['stop_loss_count'] - base['stop_loss_count']
    print(f"  改进: 收益 {d_ret:+.2%} | 夏普 {d_sharpe:+.2f} | 回撤 {d_dd:+.2%} | 止损 {d_stop:+d}次")
    print("━" * 74)


if __name__ == "__main__":
    main()
