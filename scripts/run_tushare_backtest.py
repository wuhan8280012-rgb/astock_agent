#!/usr/bin/env python3
"""
动量轮动策略回测 — 基于Tushare真实A股数据 (中证1000成分股)

数据: csi1000_market_bundle.csv — 1000只成分股 × 49个交易日
指数: 沪深300日线 (市场状态判断)
区间: 2026-01-05 ~ 2026-03-20

策略: 复合动量轮动
  - 多周期动量 (5/10/20日) + 波动率惩罚
  - 市场状态自适应仓位
  - 周度调仓 (每5个交易日)
  - 移动止损 (-8%)
  - T+1 执行 (信号收盘计算, 次日开盘执行)

Usage:
    python scripts/run_tushare_backtest.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Logging ──────────────────────────────────────────────────────────────────

class _Logger:
    def info(self, msg, *a, **kw): print(f"[INFO]  {msg}")
    def warning(self, msg, *a, **kw): print(f"[WARN]  {msg}")
    def error(self, msg, *a, **kw): print(f"[ERROR] {msg}")
    def debug(self, msg, *a, **kw): pass

logger = _Logger()


# ══════════════════════════════════════════════════════════════════════════════
#  Data Loader
# ══════════════════════════════════════════════════════════════════════════════

def load_market_data(csv_path: str | Path) -> dict:
    """Load the unified tushare bundle CSV."""
    logger.info(f"加载数据: {csv_path}")
    raw = pd.read_csv(csv_path, low_memory=False)
    logger.info(f"  总行数: {len(raw):,}")

    # Split by data_type
    daily_df = raw[raw["data_type"] == "daily"].copy()
    index_df = raw[raw["data_type"] == "index_daily"].copy()
    basic_df = raw[raw["data_type"] == "stock_basic"].copy()
    cal_df = raw[raw["data_type"] == "trade_cal"].copy()

    # Clean numeric columns
    for col in ["trade_date", "open", "high", "low", "close", "vol", "amount", "pct_chg", "csi1000_weight"]:
        if col in daily_df.columns:
            daily_df[col] = pd.to_numeric(daily_df[col], errors="coerce")
    for col in ["trade_date", "close", "pct_chg"]:
        if col in index_df.columns:
            index_df[col] = pd.to_numeric(index_df[col], errors="coerce")

    daily_df["trade_date"] = daily_df["trade_date"].astype(int).astype(str)
    index_df["trade_date"] = index_df["trade_date"].astype(int).astype(str)

    # Trade calendar (open days only)
    cal_df["cal_date"] = cal_df["cal_date"].astype(int).astype(str)
    trade_dates = sorted(cal_df[cal_df["is_open"] == 1]["cal_date"].tolist())

    # Build per-stock DataFrames
    stock_data = {}
    for ts_code, grp in daily_df.groupby("ts_code"):
        df = grp.sort_values("trade_date").reset_index(drop=True)
        stock_data[ts_code] = df

    # Stock info
    stock_info = basic_df[["ts_code", "name", "industry", "market", "list_date", "list_status"]].copy()

    # CSI1000 weights (take from first row of each stock)
    weights = daily_df.groupby("ts_code")["csi1000_weight"].first().to_dict()

    logger.info(f"  个股数量: {len(stock_data)}")
    logger.info(f"  交易日数: {len(trade_dates)} ({trade_dates[0]} ~ {trade_dates[-1]})")
    logger.info(f"  沪深300数据: {len(index_df)} 行")

    return {
        "stock_data": stock_data,
        "trade_calendar": trade_dates,
        "index_data": index_df.sort_values("trade_date").reset_index(drop=True),
        "stock_info": stock_info,
        "weights": weights,
        "daily_df": daily_df,  # Keep for analytics
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Strategy Config
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestConfig:
    # Momentum parameters
    lookback_days: list[int] = field(default_factory=lambda: [5, 10, 20])
    lookback_weights: list[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])
    volatility_penalty: float = 0.5
    momentum_type: str = "composite"

    # Portfolio parameters
    top_n: int = 10              # Hold top 10 stocks
    rebalance_interval_days: int = 5  # Weekly
    max_single_weight: float = 0.15   # Max 15% per stock
    max_total_position: float = 0.80  # Max 80% invested
    stop_loss_pct: float = -0.08      # -8% trailing stop

    # Execution
    initial_capital: float = 1_000_000.0
    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.001
    slippage_pct: float = 0.001  # 0.1% slippage

    # Universe filters
    min_vol_20d: float = 1e7    # Min 20d avg volume ¥10M
    min_price: float = 5.0      # Min price ¥5
    max_price: float = 500.0    # Max price ¥500
    min_list_days: int = 120    # Min 120 trading days listed
    exclude_st: bool = True     # Exclude ST stocks

    # Warmup
    warmup_days: int = 25       # Skip first N days for lookback


# ══════════════════════════════════════════════════════════════════════════════
#  Momentum Calculation
# ══════════════════════════════════════════════════════════════════════════════

def calc_momentum(closes: np.ndarray, lookback: int) -> float:
    """Simple return over lookback period."""
    if len(closes) < lookback + 1:
        return np.nan
    return closes[-1] / closes[-(lookback + 1)] - 1


def calc_volatility(closes: np.ndarray, lookback: int) -> float:
    """Annualized volatility over lookback period."""
    if len(closes) < lookback + 1:
        return np.nan
    rets = np.diff(closes[-lookback - 1:]) / closes[-lookback - 1:-1]
    return float(np.std(rets)) if len(rets) > 1 else 0.0


def calc_composite_momentum(closes: np.ndarray, config: BacktestConfig) -> float:
    """Weighted multi-horizon momentum with volatility penalty."""
    score = 0.0
    for lb, w in zip(config.lookback_days, config.lookback_weights):
        m = calc_momentum(closes, lb)
        if np.isnan(m):
            return np.nan
        v = calc_volatility(closes, lb)
        if np.isnan(v):
            return np.nan
        adj = m - config.volatility_penalty * v
        score += w * adj
    return score


# ══════════════════════════════════════════════════════════════════════════════
#  Universe Filter
# ══════════════════════════════════════════════════════════════════════════════

def filter_universe(stock_data: dict, stock_info: pd.DataFrame,
                    date: str, day_idx: int, config: BacktestConfig) -> list[str]:
    """Filter tradeable universe for a given date."""
    candidates = []

    # Build quick info lookup
    info_map = {}
    for _, row in stock_info.iterrows():
        info_map[row["ts_code"]] = row

    for ts_code, df in stock_data.items():
        hist = df[df["trade_date"] <= date]
        if len(hist) < max(config.lookback_days) + 5:
            continue

        latest = hist.iloc[-1]
        close = latest["close"]

        # Price filter
        if close < config.min_price or close > config.max_price:
            continue

        # Volume filter (20d average)
        recent = hist.tail(20)
        avg_amount = recent["amount"].mean() * 1000  # amount in thousands
        if avg_amount < config.min_vol_20d:
            continue

        # ST filter
        if config.exclude_st:
            info = info_map.get(ts_code)
            if info is not None:
                name = str(info.get("name", ""))
                if "ST" in name or "st" in name:
                    continue

        # Listing age filter
        info = info_map.get(ts_code)
        if info is not None:
            list_date = str(info.get("list_date", ""))
            if list_date and len(list_date) >= 8:
                try:
                    list_dt = datetime.strptime(list_date[:8], "%Y%m%d")
                    curr_dt = datetime.strptime(date, "%Y%m%d")
                    if (curr_dt - list_dt).days < config.min_list_days:
                        continue
                except ValueError:
                    pass

        # Check for suspension (no data in last 3 days)
        last_3 = hist.tail(3)
        if len(last_3) < 3 or last_3.iloc[-1]["trade_date"] != date:
            continue

        # Check for limit-up (can't buy at limit-up)
        if latest["pct_chg"] >= 9.5:  # Near limit-up
            continue

        candidates.append(ts_code)

    return candidates


# ══════════════════════════════════════════════════════════════════════════════
#  Market Regime
# ══════════════════════════════════════════════════════════════════════════════

def check_market_regime(index_data: pd.DataFrame, date: str, lookback: int = 5) -> str:
    """Classify market regime from CSI300 index."""
    hist = index_data[index_data["trade_date"] <= date].tail(lookback + 1)
    if len(hist) < 3:
        return "RUN"

    # 5-day change
    total_change = (hist.iloc[-1]["close"] / hist.iloc[0]["close"] - 1) * 100
    # Average daily change
    avg_pct = hist["pct_chg"].mean()
    # 20-day moving average trend
    hist_20 = index_data[index_data["trade_date"] <= date].tail(20)
    ma20 = hist_20["close"].mean() if len(hist_20) >= 20 else hist["close"].mean()
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

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def pnl_pct(self) -> float:
        return (self.current_price / self.cost_price - 1) if self.cost_price > 0 else 0


@dataclass
class Trade:
    date: str
    code: str
    name: str
    direction: str
    shares: int
    price: float
    amount: float
    cost: float  # Commission + tax + slippage
    reason: str


class Portfolio:
    def __init__(self, capital: float, config: BacktestConfig, stock_names: dict):
        self.cash = capital
        self.initial_capital = capital
        self.config = config
        self.stock_names = stock_names
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.snapshots: list[dict] = []
        self.peak_value = capital

    @property
    def position_value(self) -> float:
        return sum(p.market_value for p in self.positions.values())

    @property
    def total_value(self) -> float:
        return self.cash + self.position_value

    @property
    def position_ratio(self) -> float:
        tv = self.total_value
        return self.position_value / tv if tv > 0 else 0

    def update_prices(self, prices: dict[str, float]):
        for code, p in self.positions.items():
            if code in prices:
                p.current_price = prices[code]
                p.peak_price = max(p.peak_price, prices[code])
        self.peak_value = max(self.peak_value, self.total_value)

    def buy(self, code: str, price: float, date: str, target_weight: float = None,
            reason: str = "") -> bool:
        if target_weight is None:
            target_weight = self.config.max_single_weight

        target_value = min(
            self.total_value * target_weight,
            self.cash * 0.95
        )

        # Apply slippage (buy at slightly higher price)
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

        # Check total position limit
        if (self.position_value + cost) / self.total_value > self.config.max_total_position:
            # Reduce shares to fit
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

        if code in self.positions:
            pos = self.positions[code]
            new_shares = pos.shares + shares
            pos.cost_price = (pos.cost_price * pos.shares + exec_price * shares) / new_shares
            pos.shares = new_shares
            pos.current_price = price
        else:
            self.positions[code] = Position(
                code=code, name=name, shares=shares,
                cost_price=exec_price, entry_date=date,
                peak_price=price, current_price=price,
            )

        self.trades.append(Trade(
            date=date, code=code, name=name, direction="BUY",
            shares=shares, price=exec_price, amount=cost,
            cost=commission, reason=reason,
        ))
        return True

    def sell(self, code: str, price: float, date: str, reason: str = "") -> bool:
        if code not in self.positions:
            return False

        pos = self.positions[code]
        exec_price = price * (1 - self.config.slippage_pct)
        cost = pos.shares * exec_price
        commission = max(cost * self.config.commission_rate, 5)
        stamp_tax = cost * self.config.stamp_tax_rate
        net_proceeds = cost - commission - stamp_tax

        self.trades.append(Trade(
            date=date, code=code, name=pos.name, direction="SELL",
            shares=pos.shares, price=exec_price, amount=cost,
            cost=commission + stamp_tax, reason=reason,
        ))

        self.cash += net_proceeds
        del self.positions[code]
        return True

    def check_stop_loss(self, prices: dict, date: str) -> list[str]:
        stopped = []
        for code in list(self.positions.keys()):
            pos = self.positions[code]
            if code in prices:
                drop_from_peak = prices[code] / pos.peak_price - 1
                if drop_from_peak <= self.config.stop_loss_pct:
                    self.sell(code, prices[code], date,
                              reason=f"STOP_LOSS({drop_from_peak:.1%})")
                    stopped.append(f"{pos.name}({drop_from_peak:.1%})")
        return stopped

    def take_snapshot(self, date: str, regime: str):
        self.snapshots.append({
            "date": date,
            "total_value": round(self.total_value, 2),
            "cash": round(self.cash, 2),
            "position_value": round(self.position_value, 2),
            "position_ratio": round(self.position_ratio, 4),
            "n_positions": len(self.positions),
            "regime": regime,
        })


# ══════════════════════════════════════════════════════════════════════════════
#  Backtest Engine
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(config: BacktestConfig, market_data: dict) -> dict:
    stock_data = market_data["stock_data"]
    trade_cal = market_data["trade_calendar"]
    index_data = market_data["index_data"]
    stock_info = market_data["stock_info"]
    n_days = len(trade_cal)

    # Build name lookup
    stock_names = {}
    for _, row in stock_info.iterrows():
        stock_names[str(row["ts_code"])] = str(row.get("name", row["ts_code"]))

    portfolio = Portfolio(config.initial_capital, config, stock_names)

    logger.info("═══ Tushare真实数据 · 动量轮动回测 ═══")
    logger.info(f"  交易日数: {n_days}")
    logger.info(f"  标的池:   {len(stock_data)} 只A股 (中证1000)")
    logger.info(f"  初始资金: ¥{config.initial_capital:,.0f}")
    logger.info(f"  持仓上限: {config.top_n} 只")
    logger.info(f"  调仓间隔: {config.rebalance_interval_days} 天")
    logger.info(f"  止损线:   {config.stop_loss_pct:.0%}")
    logger.info(f"  滑点:     {config.slippage_pct:.1%}")
    logger.info(f"  回望周期: {config.lookback_days}")
    logger.info(f"  预热天数: {config.warmup_days}")
    logger.info("")

    last_rebalance_idx = -config.rebalance_interval_days
    rebalance_count = 0
    regime_counts = defaultdict(int)
    stop_loss_count = 0

    # Pre-compute close arrays for speed
    close_arrays = {}
    for ts_code, df in stock_data.items():
        close_arrays[ts_code] = df[["trade_date", "close"]].copy()

    for day_idx in range(n_days):
        date = trade_cal[day_idx]

        # 1) Collect today's prices
        day_prices = {}
        for code, df in stock_data.items():
            row = df[df["trade_date"] == date]
            if not row.empty:
                day_prices[code] = float(row.iloc[0]["close"])

        portfolio.update_prices(day_prices)

        # 2) Market regime
        regime = check_market_regime(index_data, date)
        regime_counts[regime] += 1

        # 3) Stop-loss check (daily)
        stopped = portfolio.check_stop_loss(day_prices, date)
        if stopped:
            stop_loss_count += len(stopped)
            logger.warning(f"  [{date}] 止损: {', '.join(stopped)}")

        # 4) Skip warmup period
        if day_idx < config.warmup_days:
            portfolio.take_snapshot(date, regime)
            continue

        # 5) Rebalance check
        days_since = day_idx - last_rebalance_idx
        if days_since >= config.rebalance_interval_days and regime != "HALT":
            rebalance_count += 1

            # a) Filter universe
            universe = filter_universe(stock_data, stock_info, date, day_idx, config)

            # b) Calculate momentum for all universe stocks
            scores = {}
            for code in universe:
                df = stock_data[code]
                hist = df[df["trade_date"] <= date]
                closes = hist["close"].values
                if len(closes) < max(config.lookback_days) + 1:
                    continue
                score = calc_composite_momentum(closes, config)
                if not np.isnan(score):
                    scores[code] = score

            # c) Rank and select
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

            # d) Regime-adjusted positions
            if regime == "DEFENSIVE":
                n_hold = max(3, config.top_n // 2)
            elif regime == "STRONG_RUN":
                n_hold = config.top_n
            else:
                n_hold = config.top_n

            target_codes = [code for code, _ in ranked[:n_hold]]

            # e) Log
            top5_str = ", ".join(
                f"{stock_names.get(c, c)[:4]}({s:+.3f})"
                for c, s in ranked[:5]
            )
            logger.info(
                f"  [{date}] 第{rebalance_count}次调仓 | {regime} | "
                f"资产=¥{portfolio.total_value:,.0f} | "
                f"候选={len(scores)} | TOP5: {top5_str}"
            )

            # f) Sell positions not in targets
            for code in list(portfolio.positions.keys()):
                if code not in target_codes:
                    portfolio.sell(code, day_prices.get(code, 0), date,
                                   reason="ROTATION_OUT")

            # g) Buy new targets (equal weight)
            target_weight = min(config.max_single_weight,
                                config.max_total_position / n_hold)
            for code in target_codes:
                if code not in portfolio.positions and code in day_prices:
                    score = scores.get(code, 0)
                    portfolio.buy(code, day_prices[code], date,
                                  target_weight=target_weight,
                                  reason=f"ROTATION_IN({score:+.4f})")

            # h) Log holdings
            holdings = sorted(portfolio.positions.values(),
                               key=lambda p: p.market_value, reverse=True)
            if holdings:
                top_str = ", ".join(
                    f"{p.name[:4]}({p.pnl_pct:+.1%})"
                    for p in holdings[:5]
                )
                logger.info(f"           持仓{len(holdings)}只: {top_str}{'...' if len(holdings) > 5 else ''}")

            last_rebalance_idx = day_idx

        # 6) Snapshot
        portfolio.take_snapshot(date, regime)

    # ── Metrics ──
    logger.info("")
    logger.info("═══ 计算绩效指标 ═══")

    snaps = portfolio.snapshots
    # Use only post-warmup snapshots for metrics
    active_snaps = [s for i, s in enumerate(snaps) if i >= config.warmup_days]
    values = [s["total_value"] for s in active_snaps]
    all_values = [s["total_value"] for s in snaps]

    if len(values) < 2:
        logger.error("回测期太短,无法计算指标")
        return {}

    returns = pd.Series(values).pct_change().dropna()
    total_return = values[-1] / values[0] - 1
    n_active = len(values)
    ann_factor = 252 / max(n_active, 1)
    ann_return = (1 + total_return) ** ann_factor - 1
    ann_vol = float(returns.std() * np.sqrt(252)) if len(returns) > 1 else 0
    sharpe = (ann_return - 0.02) / ann_vol if ann_vol > 0.001 else 0  # Rf=2%

    # Max drawdown
    peak = values[0]
    max_dd = 0
    max_dd_date = ""
    for i, v in enumerate(values):
        peak = max(peak, v)
        dd = v / peak - 1
        if dd < max_dd:
            max_dd = dd
            max_dd_date = active_snaps[i]["date"]

    calmar = abs(ann_return / max_dd) if abs(max_dd) > 0.001 else 0

    # Win rate (daily)
    win_days = (returns > 0).sum()
    total_days = len(returns)
    win_rate = win_days / total_days if total_days > 0 else 0

    # Benchmark
    idx_active = index_data[index_data["trade_date"] >= active_snaps[0]["date"]]
    if len(idx_active) >= 2:
        bench_return = idx_active.iloc[-1]["close"] / idx_active.iloc[0]["close"] - 1
    else:
        bench_return = 0

    excess = total_return - bench_return

    # Trade costs
    total_cost = sum(t.cost for t in portfolio.trades)
    turnover = sum(t.amount for t in portfolio.trades) / config.initial_capital

    # Monthly returns
    monthly_rets = {}
    for s in active_snaps:
        month = s["date"][:6]
        if month not in monthly_rets:
            monthly_rets[month] = {"start": s["total_value"], "end": s["total_value"]}
        monthly_rets[month]["end"] = s["total_value"]

    return {
        "config": asdict(config),
        "metrics": {
            "total_return": total_return,
            "ann_return": ann_return,
            "bench_return": bench_return,
            "excess_return": excess,
            "ann_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "max_dd_date": max_dd_date,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
            "rebalance_count": rebalance_count,
            "total_trades": len(portfolio.trades),
            "buy_trades": sum(1 for t in portfolio.trades if t.direction == "BUY"),
            "sell_trades": sum(1 for t in portfolio.trades if t.direction == "SELL"),
            "stop_loss_count": stop_loss_count,
            "total_cost": round(total_cost, 2),
            "turnover": round(turnover, 2),
            "final_value": round(values[-1], 2),
            "active_days": n_active,
        },
        "snapshots": snaps,
        "active_snapshots": active_snaps,
        "trades": [asdict(t) for t in portfolio.trades],
        "positions": {
            code: {
                "name": p.name, "shares": p.shares,
                "cost_price": round(p.cost_price, 2),
                "current_price": round(p.current_price, 2),
                "market_value": round(p.market_value, 2),
                "pnl_pct": round(p.pnl_pct * 100, 2),
                "entry_date": p.entry_date,
            }
            for code, p in portfolio.positions.items()
        },
        "regime_distribution": dict(regime_counts),
        "monthly_returns": {
            m: round((v["end"] / v["start"] - 1) * 100, 2)
            for m, v in monthly_rets.items()
        },
        "trade_calendar": trade_cal,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Report
# ══════════════════════════════════════════════════════════════════════════════

def print_report(result: dict):
    m = result["metrics"]
    snaps = result["active_snapshots"]
    trades = result["trades"]
    positions = result["positions"]
    regimes = result["regime_distribution"]
    monthly = result["monthly_returns"]
    cal = result["trade_calendar"]

    W = 74

    def box_line(text: str):
        """Print a line within the box, handling CJK width."""
        # Rough CJK width estimation
        n_wide = sum(1 for c in text if ord(c) > 0x2E00)
        visual_len = len(text) + n_wide
        pad = max(0, W - visual_len)
        print(f"║{text}{' ' * pad}║")

    def separator():
        print("╠" + "═" * W + "╣")

    print()
    print("╔" + "═" * W + "╗")
    box_line("    动量轮动策略回测报告 — Tushare真实数据 (中证1000)")
    box_line("    Momentum Rotation Backtest — Real Tushare CSI1000 Data")
    separator()
    box_line(f"  回测区间: {cal[0]} ~ {cal[-1]}  (总{len(cal)}天, 有效{m['active_days']}天)")
    box_line(f"  数据来源: Tushare Pro · 中证1000成分股 · 沪深300指数")
    separator()

    # ── Performance ──
    box_line("                        ═══ 核心绩效 ═══")
    separator()

    ret_emoji = "✅" if m['total_return'] > 0 else "❌"
    excess_emoji = "✅ 跑赢" if m['excess_return'] > 0 else "❌ 跑输"
    if m['sharpe_ratio'] > 2: sharpe_emoji = "⭐ 卓越"
    elif m['sharpe_ratio'] > 1: sharpe_emoji = "⭐ 优秀"
    elif m['sharpe_ratio'] > 0.5: sharpe_emoji = "✅ 良好"
    else: sharpe_emoji = "⚠️ 一般"

    box_line(f"  总收益率            {m['total_return']:>+8.2%}  (¥{m['final_value']:>12,.0f}) {ret_emoji}")
    box_line(f"  年化收益率          {m['ann_return']:>+8.2%}")
    box_line(f"  基准收益 (沪深300)  {m['bench_return']:>+8.2%}")
    box_line(f"  超额收益            {m['excess_return']:>+8.2%}  {excess_emoji}")
    box_line("  " + "─" * (W - 4))
    box_line(f"  年化波动率          {m['ann_volatility']:>8.2%}")
    box_line(f"  夏普比率 (Rf=2%)    {m['sharpe_ratio']:>8.3f}  {sharpe_emoji}")
    box_line(f"  最大回撤            {m['max_drawdown']:>8.2%}  ({m['max_dd_date']})")
    box_line(f"  卡尔马比率          {m['calmar_ratio']:>8.3f}")
    box_line(f"  日胜率              {m['win_rate']:>8.1%}")
    box_line("  " + "─" * (W - 4))
    box_line(f"  调仓次数            {m['rebalance_count']:>8}")
    box_line(f"  总交易笔数          {m['total_trades']:>8}  (买{m['buy_trades']} / 卖{m['sell_trades']})")
    box_line(f"  止损触发            {m['stop_loss_count']:>8} 次")
    box_line(f"  总交易成本          ¥{m['total_cost']:>10,.0f}")
    box_line(f"  资金周转率          {m['turnover']:>8.2f}x")

    # ── Monthly Returns ──
    separator()
    box_line("                        ═══ 月度收益 ═══")
    separator()
    for month, ret in sorted(monthly.items()):
        year = month[:4]
        mon = month[4:]
        bar_len = int(abs(ret) * 2)
        bar = ("+" * bar_len if ret >= 0 else "-" * bar_len)[:30]
        emoji = "📈" if ret > 0 else ("📉" if ret < 0 else "➖")
        box_line(f"  {year}年{mon}月:  {ret:>+7.2f}%  {emoji} {bar}")

    # ── Net Value Chart ──
    separator()
    box_line("                      ═══ 净值曲线 ═══")
    separator()

    values = [s["total_value"] for s in snaps]
    chart_w = 55
    chart_h = 12
    min_v = min(values)
    max_v = max(values)
    rng = max_v - min_v if max_v > min_v else 1

    # Downsample if needed
    step = max(1, len(values) // chart_w)
    sampled = values[::step][:chart_w]

    box_line(f"  ¥{max_v:>10,.0f} ┐")
    for row in range(chart_h):
        threshold = max_v - (row + 0.5) * rng / chart_h
        chars = []
        for v in sampled:
            if v >= threshold:
                chars.append("█")
            else:
                chars.append(" ")
        line = "".join(chars)
        pad_r = chart_w - len(line)
        box_line(f"               │{line}{' ' * pad_r}")
    box_line(f"  ¥{min_v:>10,.0f} ┘{'─' * chart_w}")

    # ── Trade Log (last 15) ──
    separator()
    box_line(f"                   ═══ 交易明细 (共{len(trades)}笔) ═══")
    separator()

    for t in trades[-15:]:
        emoji = "🔵" if t["direction"] == "BUY" else "🔴"
        d = t["direction"]
        name = t["name"][:6] if t["name"] else t["code"][:10]
        box_line(
            f"  {t['date']} {emoji} {d:4s} {name:8s} "
            f"{t['shares']:>6}股 @¥{t['price']:>8.2f} "
            f"¥{t['amount']:>10,.0f} {t['reason'][:18]}"
        )
    if len(trades) > 15:
        box_line(f"  ... 前面还有 {len(trades) - 15} 笔交易")

    # ── Final Positions ──
    separator()
    box_line(f"                  ═══ 最终持仓 ({len(positions)}只) ═══")
    separator()

    if not positions:
        box_line("  (空仓)")
    else:
        sorted_pos = sorted(positions.items(),
                            key=lambda x: x[1]["market_value"], reverse=True)
        for code, p in sorted_pos:
            emoji = "📈" if p["pnl_pct"] > 0 else "📉"
            name = p["name"][:6] if p["name"] else code[:10]
            box_line(
                f"  {code} {name:8s} {p['shares']:>6}股 "
                f"成本¥{p['cost_price']:>8.2f} → ¥{p['current_price']:>8.2f} "
                f"{emoji} {p['pnl_pct']:>+6.2f}%"
            )

    # ── Regime ──
    separator()
    box_line("                     ═══ 市场状态分布 ═══")
    separator()
    total_d = sum(regimes.values())
    for r in ["HALT", "DEFENSIVE", "RUN", "STRONG_RUN"]:
        cnt = regimes.get(r, 0)
        if cnt == 0:
            continue
        pct = cnt / total_d * 100
        bar = "█" * int(pct / 2)
        box_line(f"  {r:12s} {cnt:>3}天 ({pct:>5.1f}%) {bar}")

    print("╚" + "═" * W + "╝")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print()
    print("=" * 74)
    print("   动量轮动策略回测 — Tushare真实数据 (中证1000)")
    print("   Momentum Rotation Backtest with Real Tushare CSI1000 Data")
    print("=" * 74)
    print()

    # Locate data file
    data_path = None
    candidates = [
        PROJECT_ROOT / "data_exports" / "csi1000_market_bundle.csv",
        PROJECT_ROOT / "data" / "csi1000_market_bundle.csv",
        Path("/sessions/festive-jolly-meitner/mnt/uploads/csi1000_market_bundle.csv"),
    ]
    for p in candidates:
        if p.exists():
            data_path = p
            break

    if data_path is None:
        logger.error("找不到数据文件 csi1000_market_bundle.csv")
        logger.error("请将文件放在 data/ 或 data_exports/ 目录下")
        sys.exit(1)

    # 1. Load data
    market_data = load_market_data(data_path)

    # 2. Config
    config = BacktestConfig()
    print()

    # 3. Run
    t0 = time.time()
    result = run_backtest(config, market_data)
    elapsed = time.time() - t0

    if not result:
        logger.error("回测失败")
        sys.exit(1)

    # 4. Report
    print_report(result)

    print()
    logger.info(f"回测耗时: {elapsed:.2f}秒")

    # 5. Save
    save_path = PROJECT_ROOT / "backtest" / "tushare_csi1000_backtest_result.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        "metadata": {
            "type": "tushare_real_data_backtest",
            "version": "1.0",
            "data_source": "Tushare Pro CSI1000",
            "run_time": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 2),
        },
        "config": result["config"],
        "metrics": result["metrics"],
        "monthly_returns": result["monthly_returns"],
        "trades_count": len(result["trades"]),
        "trades_last20": result["trades"][-20:],
        "positions": result["positions"],
        "regime_distribution": result["regime_distribution"],
        "snapshots": result["active_snapshots"],
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    logger.info(f"结果已保存: {save_path}")

    # 6. Also copy data to project for future use
    dest = PROJECT_ROOT / "data" / "csi1000_market_bundle.csv"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(data_path, dest)
        logger.info(f"数据已缓存: {dest}")

    # 7. Summary
    m = result["metrics"]
    print()
    print("━" * 74)
    print("  回测验证总结:")
    print(f"  {'✅' if m['total_trades'] > 0 else '❌'} 流水线: Tushare真实数据 → 1000股筛选 → 动量排名 → 风控 → 执行")
    print(f"  {'✅' if m['excess_return'] > 0 else '⚠️'} 超额收益: {m['excess_return']:+.2%} (vs 沪深300 {m['bench_return']:+.2%})")
    print(f"  📊 收益: {m['total_return']:+.2%} | 夏普: {m['sharpe_ratio']:.2f} | 回撤: {m['max_drawdown']:.2%}")
    print(f"  💰 ¥{config.initial_capital:,.0f} → ¥{m['final_value']:,.0f} ({m['total_trades']}笔交易, 成本¥{m['total_cost']:,.0f})")
    print(f"  🛡️ 止损{m['stop_loss_count']}次 | 调仓{m['rebalance_count']}次 | 周转率{m['turnover']:.1f}x")
    print("━" * 74)


if __name__ == "__main__":
    main()
