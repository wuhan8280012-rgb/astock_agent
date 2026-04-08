#!/usr/bin/env python3
"""
F 策略官方回测引擎。

当前 `new/` 已收敛为 `F_三因子+趋势过滤` 这一条主策略线。
这里保留通用回测骨架，但默认只维护和输出 F。
"""

import json
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from strategies.f_strategy.scoring import score_universe  # noqa: E402

# ══════════════════════════════════════════════════════════════════
#  数据加载
# ══════════════════════════════════════════════════════════════════

DATA_PATH = PROJECT_ROOT / "data_exports" / "tushare_20210329_20260327_csi1000_5y" / "csi1000_market_bundle_5y.csv"
ENV_PATH = PROJECT_ROOT / "config" / ".env"
DEFAULT_TREND_INDEX_CODE = "000001.SH"


def _load_env_token() -> str | None:
    if os.environ.get("TUSHARE_TOKEN"):
        return os.environ["TUSHARE_TOKEN"].strip()
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            if line.startswith("TUSHARE_TOKEN="):
                token = line.split("=", 1)[1].strip()
                if token:
                    return token
    return None


def _fetch_trend_index_df(trade_dates: list[str], trend_index_code: str) -> pd.DataFrame | None:
    token = _load_env_token()
    if not token:
        return None
    try:
        import tushare as ts
    except Exception:
        return None
    try:
        ts.set_token(token)
        pro = ts.pro_api(token)
        idx = pro.index_daily(
            ts_code=trend_index_code,
            start_date=trade_dates[0],
            end_date=trade_dates[-1],
        )
    except Exception:
        return None
    if idx is None or idx.empty:
        return None
    idx = idx[["trade_date", "close", "pct_chg"]].copy()
    idx.columns = ["trade_date", "idx_close", "idx_pct_chg"]
    idx["trade_date"] = idx["trade_date"].astype(str)
    for c in ["idx_close", "idx_pct_chg"]:
        idx[c] = pd.to_numeric(idx[c], errors="coerce")
    idx = idx.sort_values("trade_date").reset_index(drop=True)
    idx = idx[idx["trade_date"].isin(trade_dates)].copy().reset_index(drop=True)
    return idx


def load_data(trend_index_code: str = DEFAULT_TREND_INDEX_CODE):
    """加载并预处理5年数据"""
    print("[数据] 加载CSV...")
    raw = pd.read_csv(DATA_PATH, low_memory=False)

    # 日线数据
    daily = raw[raw["data_type"] == "daily"][
        ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount", "pct_chg"]
    ].copy()
    for c in ["open", "high", "low", "close", "vol", "amount", "pct_chg"]:
        daily[c] = pd.to_numeric(daily[c], errors="coerce")
    daily["trade_date"] = daily["trade_date"].astype(str)
    daily = daily.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    # 复权因子
    adj = raw[raw["data_type"] == "adj_factor"][["ts_code", "trade_date", "adj_factor"]].copy()
    adj["adj_factor"] = pd.to_numeric(adj["adj_factor"], errors="coerce")
    adj["trade_date"] = adj["trade_date"].astype(str)

    # 合并复权因子
    daily = daily.merge(adj, on=["ts_code", "trade_date"], how="left")
    # 后复权价
    daily["adj_close"] = daily["close"] * daily["adj_factor"]

    # 市值数据
    db = raw[raw["data_type"] == "daily_basic"][["ts_code", "trade_date", "total_mv", "circ_mv"]].copy()
    db["total_mv"] = pd.to_numeric(db["total_mv"], errors="coerce")
    db["circ_mv"] = pd.to_numeric(db["circ_mv"], errors="coerce")
    db["trade_date"] = db["trade_date"].astype(str)
    daily = daily.merge(db, on=["ts_code", "trade_date"], how="left")

    # 指数数据
    idx = raw[raw["data_type"] == "index_daily"][["trade_date", "close", "pct_chg"]].copy()
    idx.columns = ["trade_date", "idx_close", "idx_pct_chg"]
    for c in ["idx_close", "idx_pct_chg"]:
        idx[c] = pd.to_numeric(idx[c], errors="coerce")
    idx["trade_date"] = idx["trade_date"].astype(str)
    idx = idx.sort_values("trade_date").reset_index(drop=True)

    # 个股信息
    basic = raw[raw["data_type"] == "stock_basic"][["ts_code", "name", "industry", "list_date"]].copy()
    basic["list_date"] = basic["list_date"].astype(str)

    # 交易日历
    trade_dates = sorted(daily["trade_date"].unique())
    fetched_idx = _fetch_trend_index_df(trade_dates, trend_index_code)
    if fetched_idx is not None and not fetched_idx.empty:
        idx = fetched_idx

    print(f"[数据] {len(daily):,} 行日线, {daily['ts_code'].nunique()} 只股票, {len(trade_dates)} 个交易日")
    print(f"[数据] 区间: {trade_dates[0]} ~ {trade_dates[-1]}")

    return daily, idx, basic, trade_dates


# ══════════════════════════════════════════════════════════════════
#  回测引擎
# ══════════════════════════════════════════════════════════════════

@dataclass
class BacktestConfig:
    name: str = "unnamed"
    # 因子参数
    momentum_days: list = field(default_factory=lambda: [60])
    momentum_weights: list = field(default_factory=lambda: [1.0])
    subtract_short_momentum: bool = False  # 减去20日短期收益
    short_momentum_days: int = 20
    use_reversal_factor: bool = False       # 反转因子 (替代动量)
    reversal_days: int = 20                 # 反转回看天数
    reversal_weight: float = 1.0            # 反转因子权重
    use_regime_switch: bool = False          # 牛熊切换: 牛市用动量, 熊市用反转
    regime_ma_days: int = 200                # 牛熊判断均线天数
    regime_bull_vol_weight: float = 0.25     # 牛市时低波权重
    regime_bear_vol_weight: float = 1.0      # 熊市时低波权重
    regime_bear_reversal_days: int = 20      # 熊市反转回看天数
    regime_bear_stop_loss_pct: float = -0.99 # 熊市止损 (默认禁用)
    regime_bear_trend_reduce: float = 1.0    # 熊市趋势仓位 (1.0=不减仓)
    use_volatility_factor: bool = False
    volatility_days: int = 60
    volatility_weight: float = 0.3
    use_size_factor: bool = False
    size_weight: float = 0.2
    use_angle_trend_factor: bool = False
    angle_trend_days: int = 10
    angle_trend_weight: float = 0.15
    angle_trend_slope_weight: float = 0.6
    angle_trend_persistence_weight: float = 0.4
    # 组合参数
    top_n: int = 15
    hold_buffer_ratio: float = 1.5
    max_single_weight: float = 0.08
    rebalance_interval: int = 20  # 交易日
    # 风控
    stop_loss_pct: float = -0.15     # 个股固定止损
    use_halt: bool = False            # HALT清仓
    use_trend_filter: bool = False    # 趋势过滤
    trend_ma_days: int = 200          # 约10个月
    trend_reduce_pct: float = 0.5     # 趋势下半仓
    # 交易成本
    commission: float = 0.0003
    stamp_tax: float = 0.001
    slippage: float = 0.002
    execution_mode: str = "same_close"  # same_close|next_open
    # 过滤
    min_amount_20d: float = 1e8       # 20日均成交额 >= 1亿
    min_price: float = 3.0
    min_list_days: int = 250          # 上市至少1年


class Backtest:
    def __init__(self, config: BacktestConfig, daily: pd.DataFrame,
                 idx: pd.DataFrame, basic: pd.DataFrame, trade_dates: list):
        self.cfg = config
        self.daily = daily
        self.idx = idx
        self.basic = basic
        self.trade_dates = trade_dates

        # 预计算: 按股票分组
        self._stock_data = {}
        for code, grp in daily.groupby("ts_code"):
            g = grp.set_index("trade_date").sort_index()
            self._stock_data[code] = g

        # 指数序列
        self._idx_series = idx.set_index("trade_date")["idx_close"].sort_index()

        # 基本信息
        self._basic_map = {}
        for _, row in basic.iterrows():
            self._basic_map[row["ts_code"]] = row

    def run(self, start_offset: int = 250, include_daily: bool = False, end_date: str | None = None) -> dict:
        """
        运行回测。start_offset: 跳过前N个交易日用于计算指标。
        """
        cfg = self.cfg
        if getattr(cfg, "execution_mode", "same_close") == "next_open":
            return self._run_next_open(start_offset=start_offset, include_daily=include_daily, end_date=end_date)
        dates = self.trade_dates
        if start_offset >= len(dates):
            return {"error": "数据不足"}
        if end_date is not None and end_date not in dates:
            return {"error": f"end_date 不在交易日历中: {end_date}"}
        end_idx = dates.index(end_date) if end_date is not None else len(dates) - 1
        if end_idx < start_offset:
            return {"error": "回测结束日早于起始偏移"}

        start_date = dates[start_offset]
        final_date = dates[end_idx]
        print(f"\n{'='*60}")
        print(f"回测: {cfg.name}")
        print(f"区间: {start_date} ~ {final_date} ({end_idx - start_offset + 1} 交易日)")
        print(f"参数: top_n={cfg.top_n}, 调仓间隔={cfg.rebalance_interval}d, "
              f"止损={cfg.stop_loss_pct:.0%}")
        print(f"{'='*60}")

        # 初始状态
        capital = 1_000_000.0
        cash = capital
        positions = {}  # {ts_code: {shares, cost_price, entry_date}}
        nav_history = []
        trade_count = 0
        rebalance_count = 0
        last_rebalance_idx = start_offset - cfg.rebalance_interval  # 允许首日调仓

        for i in range(start_offset, end_idx + 1):
            date = dates[i]

            # 1. 更新持仓价格
            prices = self._get_prices(date)
            portfolio_value = cash
            for code, pos in list(positions.items()):
                if code in prices:
                    pos["current_price"] = prices[code]
                    portfolio_value += pos["shares"] * prices[code]
                else:
                    portfolio_value += pos["shares"] * pos.get("current_price", pos["cost_price"])

            # 2. 个股止损检查 (每日)
            if cfg.stop_loss_pct > -0.99:
                for code in list(positions.keys()):
                    pos = positions[code]
                    if code in prices:
                        pnl = prices[code] / pos["cost_price"] - 1
                        if pnl <= cfg.stop_loss_pct:
                            sell_price = prices[code] * (1 - cfg.slippage)
                            proceeds = pos["shares"] * sell_price
                            cost = proceeds * (cfg.commission + cfg.stamp_tax)
                            cash += proceeds - cost
                            trade_count += 1
                            del positions[code]

            # 3. 调仓判断
            should_rebalance = (i - last_rebalance_idx) >= cfg.rebalance_interval
            max_position_pct = 1.0

            if should_rebalance:
                # 3a. HALT检查 (如果启用)
                halt = False
                if cfg.use_halt:
                    halt = self._check_halt(date)
                    if halt:
                        # 清仓
                        for code in list(positions.keys()):
                            if code in prices:
                                sell_price = prices[code] * (1 - cfg.slippage)
                                proceeds = positions[code]["shares"] * sell_price
                                cost = proceeds * (cfg.commission + cfg.stamp_tax)
                                cash += proceeds - cost
                                trade_count += 1
                        positions = {}
                        last_rebalance_idx = i
                        rebalance_count += 1

                if not halt:
                    # 3b. 趋势过滤 (如果启用)
                    if cfg.use_trend_filter:
                        max_position_pct = self._get_trend_position(date)

                    # 3c. 选股评分
                    scores = self._score_universe(date)
                    if len(scores) >= cfg.top_n:
                        target_codes = [s[0] for s in scores[:cfg.top_n]]
                        buffer_codes = set(s[0] for s in scores[:int(cfg.top_n * cfg.hold_buffer_ratio)])

                        # 卖出: 不在缓冲带内的
                        for code in list(positions.keys()):
                            if code not in buffer_codes:
                                if code in prices:
                                    sell_price = prices[code] * (1 - cfg.slippage)
                                    proceeds = positions[code]["shares"] * sell_price
                                    cost = proceeds * (cfg.commission + cfg.stamp_tax)
                                    cash += proceeds - cost
                                    trade_count += 1
                                del positions[code]

                        # 计算可用于买入的资金
                        portfolio_value_now = cash + sum(
                            pos["shares"] * prices.get(code, pos.get("current_price", pos["cost_price"]))
                            for code, pos in positions.items()
                        )
                        # 趋势过滤: 限制总仓位
                        current_position_value = sum(
                            pos["shares"] * prices.get(code, pos.get("current_price", pos["cost_price"]))
                            for code, pos in positions.items()
                        )
                        max_equity = portfolio_value_now * max_position_pct
                        available_for_equity = max_equity - current_position_value
                        available_cash = min(cash, available_for_equity) if available_for_equity > 0 else 0

                        # 买入: 在目标中但不在持仓的
                        hold_count = len(positions)
                        buy_slots = cfg.top_n - hold_count

                        for code in target_codes:
                            if buy_slots <= 0 or available_cash < 10000:
                                break
                            if code in positions:
                                continue
                            if code not in prices or prices[code] <= 0:
                                continue

                            buy_price = prices[code] * (1 + cfg.slippage)
                            target_amount = min(
                                portfolio_value_now * cfg.max_single_weight,
                                available_cash * 0.95
                            )
                            shares = int(target_amount / buy_price / 100) * 100
                            if shares >= 100:
                                amount = shares * buy_price
                                cost = amount * cfg.commission
                                cash -= (amount + cost)
                                available_cash -= (amount + cost)
                                positions[code] = {
                                    "shares": shares,
                                    "cost_price": buy_price,
                                    "entry_date": date,
                                    "current_price": prices[code],
                                }
                                trade_count += 1
                                buy_slots -= 1

                        last_rebalance_idx = i
                        rebalance_count += 1

            # 4. 记录NAV
            final_value = cash + sum(
                pos["shares"] * prices.get(code, pos.get("current_price", pos["cost_price"]))
                for code, pos in positions.items()
            )
            position_value = sum(
                pos["shares"] * prices.get(code, pos.get("current_price", pos["cost_price"]))
                for code, pos in positions.items()
            )
            trend_state = self._get_trend_state(date) if cfg.use_trend_filter else "FULL"
            nav_history.append(
                {
                    "date": date,
                    "nav": final_value,
                    "cash": cash,
                    "position_value": position_value,
                    "position_pct": position_value / final_value if final_value > 0 else 0.0,
                    "max_position_pct": max_position_pct,
                    "trend_state": trend_state,
                    "idx_close": float(self._idx_series.get(date, np.nan)),
                }
            )

        return self._build_result(
            nav_history=nav_history,
            capital=capital,
            start_date=start_date,
            end_date=final_date,
            trade_count=trade_count,
            rebalance_count=rebalance_count,
            include_daily=include_daily,
        )

    def _build_result(
        self,
        nav_history: list[dict],
        capital: float,
        start_date: str,
        end_date: str,
        trade_count: int,
        rebalance_count: int,
        include_daily: bool,
    ) -> dict:
        nav_df = pd.DataFrame(nav_history)
        nav_df["daily_return"] = nav_df["nav"].pct_change()
        total_return = (nav_df["nav"].iloc[-1] / capital - 1) * 100
        days = len(nav_df)
        years = days / 252
        annual_return = ((nav_df["nav"].iloc[-1] / capital) ** (1 / years) - 1) * 100
        annual_vol = nav_df["daily_return"].std() * np.sqrt(252) * 100
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0

        cummax = nav_df["nav"].cummax()
        drawdown = (nav_df["nav"] - cummax) / cummax
        max_dd = drawdown.min() * 100
        max_dd_date = nav_df.loc[drawdown.idxmin(), "date"] if len(nav_df) > 0 else ""

        idx_start = self._idx_series.get(start_date, None)
        idx_end = self._idx_series.get(end_date, None)
        benchmark_return = ((idx_end / idx_start) - 1) * 100 if idx_start and idx_end else 0

        nav_df["year"] = nav_df["date"].str[:4]
        yearly = {}
        for year, grp in nav_df.groupby("year"):
            if len(grp) > 10:
                yr = (grp["nav"].iloc[-1] / grp["nav"].iloc[0] - 1) * 100
                yearly[year] = round(yr, 2)

        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
        result = {
            "name": self.cfg.name,
            "execution_mode": getattr(self.cfg, "execution_mode", "same_close"),
            "total_return_pct": round(total_return, 2),
            "annual_return_pct": round(annual_return, 2),
            "annual_vol_pct": round(annual_vol, 2),
            "sharpe": round(sharpe, 2),
            "calmar": round(calmar, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "max_dd_date": max_dd_date,
            "total_trades": trade_count,
            "rebalance_count": rebalance_count,
            "final_value": round(nav_df["nav"].iloc[-1], 2),
            "benchmark_return_pct": round(benchmark_return, 2),
            "excess_return_pct": round(total_return - benchmark_return, 2),
            "yearly_returns": yearly,
            "trading_days": days,
        }

        if include_daily:
            daily_export = nav_df.copy()
            daily_export["daily_return"] = daily_export["daily_return"].fillna(0.0)
            result["daily_records"] = daily_export.to_dict(orient="records")

        print(f"\n结果: 总收益={total_return:.2f}%, 年化={annual_return:.2f}%, "
              f"夏普={sharpe:.2f}, 最大回撤={max_dd:.2f}%")
        print(f"交易次数={trade_count}, 调仓次数={rebalance_count}")
        print(f"分年度: {yearly}")
        print(f"基准: {benchmark_return:.2f}%, 超额: {total_return - benchmark_return:.2f}%")
        return result

    def _run_next_open(self, start_offset: int = 250, include_daily: bool = False, end_date: str | None = None) -> dict:
        cfg = self.cfg
        dates = self.trade_dates
        if start_offset >= len(dates):
            return {"error": "数据不足"}
        if end_date is not None and end_date not in dates:
            return {"error": f"end_date 不在交易日历中: {end_date}"}
        end_idx = dates.index(end_date) if end_date is not None else len(dates) - 1
        if end_idx < start_offset:
            return {"error": "回测结束日早于起始偏移"}

        start_date = dates[start_offset]
        final_date = dates[end_idx]
        print(f"\n{'='*60}")
        print(f"回测: {cfg.name}")
        print(f"区间: {start_date} ~ {final_date} ({end_idx - start_offset + 1} 交易日)")
        print(
            f"参数: top_n={cfg.top_n}, 调仓间隔={cfg.rebalance_interval}d, "
            f"止损={cfg.stop_loss_pct:.0%}, 成交模式=next_open"
        )
        print(f"{'='*60}")

        capital = 1_000_000.0
        cash = capital
        positions = {}
        nav_history = []
        trade_count = 0
        rebalance_count = 0
        last_rebalance_idx = start_offset - cfg.rebalance_interval
        pending_orders = None

        for i in range(start_offset, end_idx + 1):
            date = dates[i]
            open_prices = self._get_prices(date, field="open")
            close_prices = self._get_prices(date, field="close")

            if pending_orders:
                liquidate_all = bool(pending_orders.get("liquidate_all"))
                sell_codes = list(positions.keys()) if liquidate_all else list(pending_orders.get("sell_codes", []))
                for code in sell_codes:
                    if code not in positions:
                        continue
                    ref_price = open_prices.get(code, close_prices.get(code, positions[code].get("current_price", positions[code]["cost_price"])))
                    if ref_price <= 0:
                        continue
                    sell_price = ref_price * (1 - cfg.slippage)
                    proceeds = positions[code]["shares"] * sell_price
                    cost = proceeds * (cfg.commission + cfg.stamp_tax)
                    cash += proceeds - cost
                    trade_count += 1
                    del positions[code]

                if pending_orders.get("rebalance"):
                    portfolio_value_open = cash + sum(
                        pos["shares"] * open_prices.get(code, close_prices.get(code, pos.get("current_price", pos["cost_price"])))
                        for code, pos in positions.items()
                    )
                    current_position_value = sum(
                        pos["shares"] * open_prices.get(code, close_prices.get(code, pos.get("current_price", pos["cost_price"])))
                        for code, pos in positions.items()
                    )
                    max_equity = portfolio_value_open * pending_orders.get("max_position_pct", 1.0)
                    available_for_equity = max_equity - current_position_value
                    available_cash = min(cash, available_for_equity) if available_for_equity > 0 else 0.0
                    buy_slots = cfg.top_n - len(positions)

                    for code in pending_orders.get("target_codes", []):
                        if buy_slots <= 0 or available_cash < 10000:
                            break
                        if code in positions:
                            continue
                        ref_price = open_prices.get(code, close_prices.get(code, 0.0))
                        if ref_price <= 0:
                            continue
                        buy_price = ref_price * (1 + cfg.slippage)
                        target_amount = min(portfolio_value_open * cfg.max_single_weight, available_cash * 0.95)
                        shares = int(target_amount / buy_price / 100) * 100
                        if shares < 100:
                            continue
                        amount = shares * buy_price
                        cost = amount * cfg.commission
                        cash -= (amount + cost)
                        available_cash -= (amount + cost)
                        positions[code] = {
                            "shares": shares,
                            "cost_price": buy_price,
                            "entry_date": date,
                            "current_price": close_prices.get(code, ref_price),
                        }
                        trade_count += 1
                        buy_slots -= 1
                    rebalance_count += 1

                pending_orders = None

            portfolio_value = cash
            for code, pos in list(positions.items()):
                if code in close_prices:
                    pos["current_price"] = close_prices[code]
                    portfolio_value += pos["shares"] * close_prices[code]
                else:
                    portfolio_value += pos["shares"] * pos.get("current_price", pos["cost_price"])

            should_rebalance = (i - last_rebalance_idx) >= cfg.rebalance_interval
            max_position_pct = 1.0
            if cfg.use_trend_filter:
                max_position_pct = self._get_trend_position(date)

            trend_state = self._get_trend_state(date) if cfg.use_trend_filter else "FULL"

            if i < end_idx:
                next_orders = {
                    "sell_codes": set(),
                    "target_codes": [],
                    "rebalance": False,
                    "liquidate_all": False,
                    "max_position_pct": max_position_pct,
                }

                if cfg.stop_loss_pct > -0.99:
                    for code, pos in positions.items():
                        if code not in close_prices:
                            continue
                        pnl = close_prices[code] / pos["cost_price"] - 1
                        if pnl <= cfg.stop_loss_pct:
                            next_orders["sell_codes"].add(code)

                if should_rebalance:
                    halt = False
                    if cfg.use_halt:
                        halt = self._check_halt(date)
                        if halt:
                            next_orders["sell_codes"].update(positions.keys())
                            next_orders["rebalance"] = True
                            next_orders["liquidate_all"] = True
                            last_rebalance_idx = i
                    if not halt:
                        scores = self._score_universe(date)
                        if len(scores) >= cfg.top_n:
                            target_codes = [s[0] for s in scores[:cfg.top_n]]
                            buffer_codes = set(s[0] for s in scores[:int(cfg.top_n * cfg.hold_buffer_ratio)])
                            next_orders["sell_codes"].update(code for code in positions if code not in buffer_codes)
                            next_orders["target_codes"] = target_codes
                            next_orders["rebalance"] = True
                            last_rebalance_idx = i

                if next_orders["sell_codes"] or next_orders["rebalance"]:
                    pending_orders = next_orders

            final_value = cash + sum(
                pos["shares"] * close_prices.get(code, pos.get("current_price", pos["cost_price"]))
                for code, pos in positions.items()
            )
            position_value = sum(
                pos["shares"] * close_prices.get(code, pos.get("current_price", pos["cost_price"]))
                for code, pos in positions.items()
            )
            nav_history.append(
                {
                    "date": date,
                    "nav": final_value,
                    "cash": cash,
                    "position_value": position_value,
                    "position_pct": position_value / final_value if final_value > 0 else 0.0,
                    "max_position_pct": max_position_pct,
                    "trend_state": trend_state,
                    "idx_close": float(self._idx_series.get(date, np.nan)),
                }
            )

        return self._build_result(
            nav_history=nav_history,
            capital=capital,
            start_date=start_date,
            end_date=final_date,
            trade_count=trade_count,
            rebalance_count=rebalance_count,
            include_daily=include_daily,
        )

    def _get_prices(self, date: str, field: str = "close") -> dict:
        """获取指定日期所有股票的价格字段。"""
        mask = (self.daily["trade_date"] == date)
        d = self.daily.loc[mask, ["ts_code", field]]
        return dict(zip(d["ts_code"], d[field]))

    def _check_halt(self, date: str) -> bool:
        """检查是否触发HALT (与现有系统相同的逻辑)"""
        idx_data = self.idx[self.idx["trade_date"] <= date].tail(6)
        if len(idx_data) < 3:
            return False
        total_change = (idx_data.iloc[-1]["idx_close"] / idx_data.iloc[0]["idx_close"] - 1) * 100
        avg_pct = idx_data["idx_pct_chg"].mean()
        return total_change < -5 or avg_pct < -1.0

    def _get_trend_position(self, date: str) -> float:
        """趋势过滤: 返回最大仓位比例"""
        idx_data = self._idx_series[self._idx_series.index <= date]
        if len(idx_data) < self.cfg.trend_ma_days:
            return 1.0
        ma = idx_data.iloc[-self.cfg.trend_ma_days:].mean()
        current = idx_data.iloc[-1]
        if current > ma:
            return 1.0
        elif current > ma * 0.95:
            return self.cfg.trend_reduce_pct
        else:
            return 0.0

    def _get_trend_state(self, date: str) -> str:
        """与趋势过滤一致的三态标签，用于归因分析。"""
        idx_data = self._idx_series[self._idx_series.index <= date]
        if len(idx_data) < self.cfg.trend_ma_days:
            return "BULL"
        ma = idx_data.iloc[-self.cfg.trend_ma_days:].mean()
        current = idx_data.iloc[-1]
        if current > ma:
            return "BULL"
        elif current > ma * 0.95:
            return "RANGE"
        return "BEAR"

    def _score_universe(self, date: str) -> list:
        """对全市场评分，返回 [(ts_code, score), ...]，分数越小越好。"""
        universe_map = getattr(self, "_universe_codes_by_date", None)
        allowed_codes = universe_map.get(date) if isinstance(universe_map, dict) else None
        return score_universe(self._stock_data, self._basic_map, self.cfg, date, allowed_codes=allowed_codes)


# ══════════════════════════════════════════════════════════════════
#  策略定义
# ══════════════════════════════════════════════════════════════════

def get_strategies():
    """当前只保留 F 这一条主策略。"""

    return [BacktestConfig(
        name="F_三因子+趋势过滤",
        momentum_days=[60],
        momentum_weights=[1.0],
        use_volatility_factor=True,
        volatility_days=60,
        volatility_weight=0.25,
        use_size_factor=True,
        size_weight=0.2,
        use_angle_trend_factor=True,
        angle_trend_days=10,
        angle_trend_weight=0.15,
        angle_trend_slope_weight=0.6,
        angle_trend_persistence_weight=0.4,
        top_n=15,
        hold_buffer_ratio=1.5,
        max_single_weight=0.08,
        rebalance_interval=20,
        stop_loss_pct=-0.15,
        use_halt=False,
        use_trend_filter=True,
        trend_ma_days=200,
        trend_reduce_pct=0.5,
        slippage=0.002,
    )]


# ══════════════════════════════════════════════════════════════════
#  主程序
# ══════════════════════════════════════════════════════════════════

def main():
    daily, idx, basic, trade_dates = load_data()

    strategies = get_strategies()
    results = []

    for cfg in strategies:
        bt = Backtest(cfg, daily, idx, basic, trade_dates)
        t0 = time.time()
        result = bt.run(start_offset=250)
        result["elapsed_sec"] = round(time.time() - t0, 1)
        results.append(result)

    # 输出对比表
    print("\n\n" + "=" * 80)
    print("策略对比总表")
    print("=" * 80)
    print(f"{'策略':<35} {'总收益':>8} {'年化':>8} {'夏普':>6} {'Calmar':>7} {'最大回撤':>8} {'超额':>8} {'交易次数':>8}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<35} {r['total_return_pct']:>7.1f}% {r['annual_return_pct']:>7.1f}% "
              f"{r['sharpe']:>6.2f} {r['calmar']:>7.2f} {r['max_drawdown_pct']:>7.1f}% "
              f"{r['excess_return_pct']:>7.1f}% {r['total_trades']:>8}")

    # 保存结果
    output_path = Path(__file__).parent.parent / "backtest" / "multi_strategy_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
