#!/usr/bin/env python3
"""
回测引擎 — v2.1 自适应 vs v3.0 基准 vs v3.1 龙虎榜增强

数据: data/csi1000_market_bundle_300d_lhb.csv (300个交易日, 含龙虎榜)
回测窗口: 前30天用于回望期, 其余为回测区间
调仓频率: 每5个交易日
初始资金: 100万

版本:
  v2.1 自适应    — 市场状态自适应权重, -8%止损
  v3.0 基准      — 固定权重, 无个股止损, HALT清仓保留
  v3.1 龙虎榜    — v3.0 + 龙虎榜净买入加分 (w=0.30, 回看10天)
"""

from __future__ import annotations

import sys
import os
import json
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 解决内置 signal 模块与项目 signal/ 目录冲突
# 临时移除内置 signal, 让 Python 找到我们的 signal 包
import signal as _builtin_signal
_saved_signal = sys.modules.pop("signal")
sys.modules.pop("signal.signal_generator", None)

# 现在导入我们的模块
from signal.signal_generator import SignalConfig, SignalGenerator  # noqa: E402

# 恢复内置 signal
sys.modules["_builtin_signal"] = _saved_signal


# ══════════════════════════════════════════════════════════════════════════════
#  数据加载
# ══════════════════════════════════════════════════════════════════════════════

def load_all_data(csv_path: str) -> dict:
    raw = pd.read_csv(csv_path, low_memory=False)

    daily_df = raw[raw["data_type"] == "daily"].copy()
    num_cols = ["trade_date", "open", "high", "low", "close", "vol", "amount", "pct_chg",
                "total_mv", "circ_mv", "adj_factor"]
    for col in num_cols:
        if col in daily_df.columns:
            daily_df[col] = pd.to_numeric(daily_df[col], errors="coerce")
    daily_df["trade_date"] = daily_df["trade_date"].astype(int).astype(str)

    # 前复权处理: close_adj = close * adj_factor / 最新adj_factor
    if "adj_factor" in daily_df.columns:
        latest_adj = daily_df.sort_values("trade_date").groupby("ts_code")["adj_factor"].last()
        daily_df = daily_df.merge(
            latest_adj.rename("latest_adj"), left_on="ts_code", right_index=True, how="left"
        )
        mask = daily_df["adj_factor"].notna() & daily_df["latest_adj"].notna() & (daily_df["latest_adj"] > 0)
        ratio = daily_df["adj_factor"] / daily_df["latest_adj"]
        for col in ["open", "high", "low", "close"]:
            daily_df.loc[mask, col] = daily_df.loc[mask, col] * ratio[mask]
        daily_df.drop(columns=["latest_adj"], inplace=True)
        print(f"[数据] 前复权处理完成")

    index_df = raw[raw["data_type"] == "index_daily"].copy()
    for col in ["trade_date", "close", "pct_chg"]:
        if col in index_df.columns:
            index_df[col] = pd.to_numeric(index_df[col], errors="coerce")
    index_df["trade_date"] = index_df["trade_date"].astype(int).astype(str)
    index_df = index_df.sort_values("trade_date").reset_index(drop=True)

    basic_df = raw[raw["data_type"] == "stock_basic"][
        ["ts_code", "name", "industry", "market", "list_date", "list_status"]
    ].copy()

    cal_df = raw[raw["data_type"] == "trade_cal"].copy()
    cal_df["cal_date"] = cal_df["cal_date"].astype(int).astype(str)
    trade_dates = sorted(cal_df[cal_df["is_open"] == 1]["cal_date"].tolist())

    stock_data = {}
    for ts_code, grp in daily_df.groupby("ts_code"):
        stock_data[ts_code] = grp.sort_values("trade_date").reset_index(drop=True)

    name_map = dict(zip(basic_df["ts_code"], basic_df["name"].astype(str)))
    ind_map = dict(zip(basic_df["ts_code"], basic_df["industry"].astype(str)))

    # ── 龙虎榜数据 ──
    lhb_data = {}
    lhb_df = raw[raw["data_type"] == "top_list"].copy()
    if not lhb_df.empty:
        for col in ["trade_date", "l_buy", "l_sell"]:
            if col in lhb_df.columns:
                lhb_df[col] = pd.to_numeric(lhb_df[col], errors="coerce")
        lhb_df["trade_date"] = lhb_df["trade_date"].astype(int).astype(str)
        for _, row in lhb_df.iterrows():
            ts_code = row["ts_code"]
            date = row["trade_date"]
            l_buy = row.get("l_buy", 0) or 0
            l_sell = row.get("l_sell", 0) or 0
            if ts_code not in lhb_data:
                lhb_data[ts_code] = {}
            if date in lhb_data[ts_code]:
                lhb_data[ts_code][date]["l_buy"] += l_buy
                lhb_data[ts_code][date]["l_sell"] += l_sell
                lhb_data[ts_code][date]["net_buy"] = lhb_data[ts_code][date]["l_buy"] - lhb_data[ts_code][date]["l_sell"]
            else:
                lhb_data[ts_code][date] = {"l_buy": l_buy, "l_sell": l_sell, "net_buy": l_buy - l_sell}
        total_lhb = sum(len(v) for v in lhb_data.values())
        print(f"[数据] 龙虎榜: {len(lhb_data)} 只股票, {total_lhb} 条记录")

    return {
        "stock_data": stock_data,
        "index_data": index_df,
        "stock_info": basic_df,
        "trade_dates": trade_dates,
        "name_map": name_map,
        "ind_map": ind_map,
        "lhb_data": lhb_data,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  回测引擎
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Position:
    ts_code: str
    name: str
    shares: int
    cost_price: float
    entry_date: str
    peak_price: float
    industry: str = ""


@dataclass
class DailyRecord:
    date: str
    total_value: float
    cash: float
    position_value: float
    position_ratio: float
    regime: str
    num_positions: int
    trade_count: int  # 当日交易笔数
    industries: list = field(default_factory=list)


class BacktestEngine:
    """轻量回测引擎, 直接复用 SignalGenerator 的评分逻辑"""

    def __init__(self, version: str, data: dict, initial_capital: float = 1_000_000,
                 custom_config: "SignalConfig | None" = None):
        self.version = version
        self.data = data
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: dict[str, Position] = {}
        self.daily_records: list[DailyRecord] = []
        self.trade_log: list[dict] = []
        self.rebalance_dates: list[str] = []

        # SignalConfig / SignalGenerator 已在模块顶部导入

        if custom_config is not None:
            # 进化系统传入自定义配置
            self.config = custom_config
        elif version == "v2.1":
            # 旧版: 自适应权重, -8%止损, 无缺口保护
            self.config = SignalConfig(
                data_csv_path="__backtest__",
                adaptive_weights=True,
                enable_reversal_filter=False,
                enable_trend_window=False,
                stop_loss_pct=-0.08,
                open_gap_limit=0.0,         # v2.1不启用缺口保护
                slippage_pct=0.001,          # v2.1原始滑点
                enable_lhb_factor=False,
                enable_macro_calendar=False,
                enable_breaking_monitor=False,
            )
        elif version == "v3.0":
            # 基准: 固定权重, 无个股止损, HALT保留
            self.config = SignalConfig(
                data_csv_path="__backtest__",
                adaptive_weights=False,
                enable_reversal_filter=False,
                enable_trend_window=False,
                stop_loss_pct=-0.99,
                lookback_weights=[0.5, 0.3, 0.2],
                enable_lhb_factor=False,
                enable_macro_calendar=False,
                enable_breaking_monitor=False,
            )
        elif version == "v3.1":
            # 龙虎榜增强: v3.0 + LHB boost (w=0.30, 回看10天)
            self.config = SignalConfig(
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
            )
        else:
            raise ValueError(f"Unknown version: {version}")

        self.generator = SignalGenerator(self.config)
        self.last_rebalance_idx = -999

    def _get_price(self, ts_code: str, date: str) -> float:
        df = self.data["stock_data"].get(ts_code)
        if df is None:
            return 0
        row = df[df["trade_date"] == date]
        if row.empty:
            # 停牌: 用最近价格
            hist = df[df["trade_date"] <= date]
            return float(hist.iloc[-1]["close"]) if not hist.empty else 0
        return float(row.iloc[0]["close"])

    def _get_open_price(self, ts_code: str, date: str) -> float:
        df = self.data["stock_data"].get(ts_code)
        if df is None:
            return 0
        row = df[df["trade_date"] == date]
        if row.empty:
            return self._get_price(ts_code, date)
        return float(row.iloc[0]["open"])

    def _total_value(self, date: str) -> float:
        pos_value = sum(
            p.shares * self._get_price(p.ts_code, date)
            for p in self.positions.values()
        )
        return self.cash + pos_value

    def _check_regime(self, date: str) -> str:
        return self.generator._check_market_regime(self.data["index_data"], date)

    def _score_stocks(self, date: str, regime: str) -> list[dict]:
        """对所有候选股评分"""
        cfg = self.config
        gen = self.generator
        stock_data = self.data["stock_data"]
        stock_info = self.data["stock_info"]
        ind_map = self.data["ind_map"]

        # 过滤
        candidates = gen._filter_universe(stock_data, stock_info, date)

        # 自适应权重
        weights = gen._get_regime_weights(regime)

        scores = []
        for ts_code in candidates:
            df = stock_data[ts_code]
            hist = df[df["trade_date"] <= date]
            closes = hist["close"].values.astype(float)
            avg_amount = hist.tail(20)["amount"].mean() * 1000

            score = gen._calc_composite_score(
                closes, avg_amount, weights=weights
            )
            if not np.isnan(score):
                scores.append({
                    "ts_code": ts_code,
                    "name": self.data["name_map"].get(ts_code, ts_code),
                    "score": score,
                    "close": float(closes[-1]),
                })

        # 龙虎榜加分 (v3.1)
        if cfg.enable_lhb_factor:
            lhb_data = self.data.get("lhb_data", {})
            trade_dates = self.data["trade_dates"]
            if lhb_data:
                for s in scores:
                    net_buy, count = self.generator._calc_lhb_signal(
                        s["ts_code"], date, lhb_data, cfg.lhb_lookback, trade_dates
                    )
                    if count > 0 and net_buy > 0:
                        boost = 1 + cfg.lhb_weight * min(count / 2, 1.0)
                        s["score"] *= boost
                    elif count > 0 and net_buy < 0:
                        s["score"] *= (1 - cfg.lhb_weight * cfg.lhb_negative_penalty)

        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores

    def _execute_rebalance(self, date: str, scores: list[dict], regime: str):
        """执行调仓"""
        cfg = self.config
        ind_map = self.data["ind_map"]
        total_value = self._total_value(date)
        trade_count = 0

        # 仓位上限
        if regime in ("HALT", "WAIT"):
            max_position = 0.0
        elif regime == "DEFENSIVE":
            max_position = 0.5 * cfg.max_total_position
        elif regime == "STRONG_RUN":
            max_position = cfg.max_total_position
        else:
            max_position = cfg.max_total_position

        # HALT/WAIT: 清仓
        if regime in ("HALT", "WAIT"):
            reason = "HALT清仓" if regime == "HALT" else "WAIT空仓(死叉)"
            for code in list(self.positions.keys()):
                pos = self.positions[code]
                sell_price = self._get_open_price(code, date)
                proceeds = pos.shares * sell_price
                commission = max(proceeds * cfg.commission_rate, 5)
                stamp_tax = proceeds * cfg.stamp_tax_rate
                self.cash += proceeds - commission - stamp_tax
                self.trade_log.append({
                    "date": date, "action": "SELL", "code": code,
                    "name": pos.name, "shares": pos.shares, "price": sell_price,
                    "reason": reason,
                })
                del self.positions[code]
                trade_count += 1
            return trade_count

        target_codes = [s["ts_code"] for s in scores[:cfg.top_n]]
        buffer_codes = [s["ts_code"] for s in scores[:int(cfg.top_n * cfg.hold_buffer_ratio)]]

        # 卖出: 不在缓冲带的
        for code in list(self.positions.keys()):
            if code not in buffer_codes:
                pos = self.positions[code]
                sell_price = self._get_open_price(code, date)
                proceeds = pos.shares * sell_price
                commission = max(proceeds * cfg.commission_rate, 5)
                stamp_tax = proceeds * cfg.stamp_tax_rate
                self.cash += proceeds - commission - stamp_tax
                self.trade_log.append({
                    "date": date, "action": "SELL", "code": code,
                    "name": pos.name, "shares": pos.shares, "price": sell_price,
                    "reason": "出缓冲带",
                })
                del self.positions[code]
                trade_count += 1

        # 买入
        hold_count = len(self.positions)
        buy_slots = cfg.top_n - hold_count

        scan_range = min(len(scores), cfg.top_n * 3)
        for s in scores[:scan_range]:
            code = s["ts_code"]
            if code in self.positions:
                continue
            if buy_slots <= 0:
                break

            buy_price = self._get_open_price(code, date)
            if buy_price <= 0:
                continue

            # 限价单保护: 开盘价高开超过阈值则跳过 (实盘验证: 高开追入当日大概率回撤)
            prev_close = s["close"]  # 上一日收盘价 (信号日)
            if prev_close > 0 and cfg.open_gap_limit > 0:
                gap = buy_price / prev_close - 1
                if gap > cfg.open_gap_limit:
                    continue  # 高开过多, 不追

            target_value = min(total_value * cfg.max_single_weight, self.cash * 0.95)
            current_pos_value = sum(
                p.shares * self._get_price(p.ts_code, date)
                for p in self.positions.values()
            )
            if total_value > 0 and (current_pos_value + target_value) / total_value > max_position:
                target_value = max(0, total_value * max_position - current_pos_value)

            exec_price = buy_price * (1 + cfg.slippage_pct)
            shares = int(target_value / exec_price / 100) * 100

            if shares >= 100 and shares * exec_price >= cfg.min_position_amount:
                cost = shares * exec_price
                commission = max(cost * cfg.commission_rate, 5)
                self.cash -= (cost + commission)

                self.positions[code] = Position(
                    ts_code=code,
                    name=s["name"],
                    shares=shares,
                    cost_price=exec_price,
                    entry_date=date,
                    peak_price=buy_price,
                )
                buy_slots -= 1
                trade_count += 1

                self.trade_log.append({
                    "date": date, "action": "BUY", "code": code,
                    "name": s["name"], "shares": shares, "price": exec_price,
                    "score": round(s["score"], 4),
                    "reason": f"排名#{scores.index(s)+1}",
                })

        return trade_count

    def _check_stop_loss(self, date: str) -> int:
        """止损检查"""
        cfg = self.config
        trade_count = 0
        for code in list(self.positions.keys()):
            pos = self.positions[code]
            current = self._get_price(code, date)
            pos.peak_price = max(pos.peak_price, current)
            if pos.peak_price > 0:
                drop = current / pos.peak_price - 1
                if drop <= cfg.stop_loss_pct:
                    sell_price = current
                    proceeds = pos.shares * sell_price
                    commission = max(proceeds * cfg.commission_rate, 5)
                    stamp_tax = proceeds * cfg.stamp_tax_rate
                    self.cash += proceeds - commission - stamp_tax
                    self.trade_log.append({
                        "date": date, "action": "STOP_LOSS", "code": code,
                        "name": pos.name, "shares": pos.shares, "price": sell_price,
                        "reason": f"止损{drop:.1%}",
                    })
                    del self.positions[code]
                    trade_count += 1
        return trade_count

    def run(self, start_date: str, end_date: str):
        """运行回测

        重要: 评分使用 T-1 日收盘数据, 执行使用 T 日开盘价
        即: 昨日盘后跑信号 → 今日开盘执行, 消除 look-ahead bias
        """
        dates = self.data["trade_dates"]
        bt_dates = [d for d in dates if start_date <= d <= end_date]

        # 构建全局交易日→index映射 (用于找前一交易日)
        all_date_idx = {d: i for i, d in enumerate(dates)}

        print(f"\n[回测] {self.version}: {bt_dates[0]} ~ {bt_dates[-1]}, {len(bt_dates)} 交易日")

        for i, date in enumerate(bt_dates):
            regime = self._check_regime(date)
            trade_count = 0

            # 找前一交易日 (用于评分, 消除 look-ahead bias)
            global_idx = all_date_idx.get(date, 0)
            prev_date = dates[global_idx - 1] if global_idx > 0 else date

            # 止损 (每天检查)
            trade_count += self._check_stop_loss(date)

            # HALT/WAIT状态: 立即清仓
            if regime in ("HALT", "WAIT") and self.positions:
                scores = self._score_stocks(prev_date, regime)
                trade_count += self._execute_rebalance(date, scores, regime)
                self._halt_liquidated = True  # 标记: 清仓过
                self.rebalance_dates.append(date)
            # 非清仓状态
            elif regime not in ("HALT", "WAIT"):
                # 清仓恢复后首次: 立即允许建仓, 无视调仓间隔
                halt_recovery = getattr(self, "_halt_liquidated", False) and not self.positions
                days_since = i - self.last_rebalance_idx
                if (halt_recovery
                        or days_since >= self.config.rebalance_interval_days
                        or self.last_rebalance_idx < 0):
                    scores = self._score_stocks(prev_date, regime)
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
            industries = [pos.industry for pos in self.positions.values() if pos.industry]

            self.daily_records.append(DailyRecord(
                date=date,
                total_value=total,
                cash=self.cash,
                position_value=pos_value,
                position_ratio=pos_value / total if total > 0 else 0,
                regime=regime,
                num_positions=len(self.positions),
                trade_count=trade_count,
                industries=industries,
            ))

        print(f"[回测] 完成: 终值=¥{self.daily_records[-1].total_value:,.0f}, "
              f"调仓{len(self.rebalance_dates)}次, 交易{len(self.trade_log)}笔")

    def calc_metrics(self) -> dict:
        """计算绩效指标"""
        if not self.daily_records:
            return {}

        values = [r.total_value for r in self.daily_records]
        dates = [r.date for r in self.daily_records]

        # 累计收益率
        total_return = values[-1] / self.initial_capital - 1

        # 日收益率序列
        daily_rets = np.diff(values) / np.array(values[:-1])

        # 年化收益
        n_days = len(values)
        annual_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 1 else 0

        # 年化波动率
        annual_vol = np.std(daily_rets) * np.sqrt(252) if len(daily_rets) > 1 else 0

        # 夏普比率 (无风险2.5%)
        rf = 0.025
        sharpe = (annual_return - rf) / annual_vol if annual_vol > 0 else 0

        # 最大回撤
        peak = np.maximum.accumulate(values)
        drawdowns = (np.array(values) - peak) / peak
        max_dd = float(np.min(drawdowns))
        max_dd_date = dates[int(np.argmin(drawdowns))]

        # 胜率 (调仓后5天的收益)
        wins = sum(1 for r in daily_rets if r > 0)
        win_rate = wins / len(daily_rets) if daily_rets.size > 0 else 0

        # 行业分散度 (平均每次持仓覆盖多少行业)
        ind_diversities = []
        for r in self.daily_records:
            if r.industries:
                ind_diversities.append(len(set(r.industries)))
        avg_ind_diversity = np.mean(ind_diversities) if ind_diversities else 0

        # 最大单行业集中度
        max_ind_concentration = 0
        for r in self.daily_records:
            if r.industries:
                counts = Counter(r.industries)
                if counts:
                    max_count = max(counts.values())
                    max_ind_concentration = max(max_ind_concentration, max_count)

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_vol": annual_vol,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "max_dd_date": max_dd_date,
            "daily_win_rate": win_rate,
            "total_trades": len(self.trade_log),
            "rebalance_count": len(self.rebalance_dates),
            "avg_industry_diversity": avg_ind_diversity,
            "max_industry_concentration": max_ind_concentration,
            "final_value": values[-1],
        }


# ══════════════════════════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # 优先使用含龙虎榜的300天数据集
    csv_300d_lhb = PROJECT_ROOT / "data" / "csi1000_market_bundle_300d_lhb.csv"
    csv_300d = PROJECT_ROOT / "data" / "csi1000_market_bundle_300d.csv"
    csv_100d = PROJECT_ROOT / "data" / "csi1000_market_bundle_100d.csv"
    csv_24d = PROJECT_ROOT / "data" / "csi1000_market_bundle.csv"
    if csv_300d_lhb.exists():
        csv_path = str(csv_300d_lhb)
        start_date = "20250206"
        end_date = "20260324"
        print(f"[数据] 使用300天数据集(含龙虎榜): {csv_path}")
    elif csv_300d.exists():
        csv_path = str(csv_300d)
        start_date = "20250206"
        end_date = "20260324"
        print(f"[数据] 使用300天数据集: {csv_path}")
    elif csv_100d.exists():
        csv_path = str(csv_100d)
        start_date = "20251127"
        end_date = "20260323"
        print(f"[数据] 使用100天数据集: {csv_path}")
    else:
        csv_path = str(csv_24d)
        start_date = "20260209"
        end_date = "20260320"
        print(f"[数据] 使用24天数据集: {csv_path}")

    data = load_all_data(csv_path)

    # 确认回测日期在交易日范围内
    trade_dates = data["trade_dates"]
    bt_dates = [d for d in trade_dates if start_date <= d <= end_date]
    if not bt_dates:
        # 自动找合适的起始日 (跳过前25天回望期)
        if len(trade_dates) > 25:
            start_date = trade_dates[25]
            end_date = trade_dates[-1]
            bt_dates = [d for d in trade_dates if start_date <= d <= end_date]
            print(f"[数据] 自动调整回测区间: {start_date} ~ {end_date}, {len(bt_dates)}个交易日")

    # ── 三版本回测 ──
    versions = ["v2.1", "v3.0", "v3.1"]
    engines = {}
    metrics = {}

    for ver in versions:
        bt = BacktestEngine(ver, data)
        bt.run(start_date, end_date)
        engines[ver] = bt
        metrics[ver] = bt.calc_metrics()

    # 基准指数
    idx = data["index_data"]
    idx_bt = idx[(idx["trade_date"] >= start_date) & (idx["trade_date"] <= end_date)]
    if not idx_bt.empty:
        idx_start = float(idx_bt.iloc[0]["close"])
        idx_end = float(idx_bt.iloc[-1]["close"])
        idx_return = idx_end / idx_start - 1
    else:
        idx_return = 0

    # ══════════════════════════════════════════════════════════════════════
    #  输出报告
    # ══════════════════════════════════════════════════════════════════════

    n_days = len(bt_dates) if bt_dates else 0
    print("\n" + "=" * 100)
    print("  回测报告: v2.1 旧版 vs v3.0 基准 vs v3.1 龙虎榜增强")
    print(f"  区间: {start_date} ~ {end_date}  |  {n_days}个交易日  |  初始资金: ¥1,000,000")
    print("=" * 100)

    m2, m3, m31 = metrics["v2.1"], metrics["v3.0"], metrics["v3.1"]

    print(f"\n{'指标':<24s} {'v2.1 旧版':>14s} {'v3.0 基准':>14s} {'v3.1 龙虎榜':>14s} {'基准指数':>14s}")
    print("─" * 84)
    print(f"{'累计收益率':<22s} {m2['total_return']:>13.2%} {m3['total_return']:>13.2%} {m31['total_return']:>13.2%} {idx_return:>13.2%}")
    print(f"{'年化收益率':<22s} {m2['annual_return']:>13.2%} {m3['annual_return']:>13.2%} {m31['annual_return']:>13.2%} {'—':>14s}")
    print(f"{'年化波动率':<22s} {m2['annual_vol']:>13.2%} {m3['annual_vol']:>13.2%} {m31['annual_vol']:>13.2%} {'—':>14s}")
    print(f"{'夏普比率':<22s} {m2['sharpe']:>13.2f} {m3['sharpe']:>13.2f} {m31['sharpe']:>13.2f} {'—':>14s}")
    print(f"{'最大回撤':<22s} {m2['max_drawdown']:>13.2%} {m3['max_drawdown']:>13.2%} {m31['max_drawdown']:>13.2%} {'—':>14s}")
    print(f"{'最大回撤日期':<20s} {m2['max_dd_date']:>14s} {m3['max_dd_date']:>14s} {m31['max_dd_date']:>14s} {'—':>14s}")
    print(f"{'日胜率':<22s} {m2['daily_win_rate']:>13.1%} {m3['daily_win_rate']:>13.1%} {m31['daily_win_rate']:>13.1%} {'—':>14s}")
    print(f"{'总交易笔数':<22s} {m2['total_trades']:>13d} {m3['total_trades']:>13d} {m31['total_trades']:>13d} {'—':>14s}")
    print(f"{'调仓次数':<22s} {m2['rebalance_count']:>13d} {m3['rebalance_count']:>13d} {m31['rebalance_count']:>13d} {'—':>14s}")
    print(f"{'期末净值':<22s} ¥{m2['final_value']:>11,.0f} ¥{m3['final_value']:>11,.0f} ¥{m31['final_value']:>11,.0f} {'—':>14s}")

    # 每日净值对比 (每5天打印一行)
    print(f"\n{'─'*100}")
    print("  每日净值曲线 (每5天采样)")
    print(f"{'─'*100}")
    print(f"  {'日期':<12s} {'v2.1':>10s} {'v3.0':>10s} {'v3.1':>10s} {'基准':>10s} {'状态':>12s}")

    idx_dict = dict(zip(idx_bt["trade_date"], idx_bt["close"]))
    idx_base = float(idx_bt.iloc[0]["close"]) if not idx_bt.empty else 1

    bt_v2, bt_v3, bt_v31 = engines["v2.1"], engines["v3.0"], engines["v3.1"]
    for i, (r2, r3, r31) in enumerate(zip(bt_v2.daily_records, bt_v3.daily_records, bt_v31.daily_records)):
        if i % 5 != 0 and i != len(bt_v2.daily_records) - 1:
            continue
        v2_nav = r2.total_value / bt_v2.initial_capital
        v3_nav = r3.total_value / bt_v3.initial_capital
        v31_nav = r31.total_value / bt_v31.initial_capital
        idx_val = idx_dict.get(r2.date, idx_base)
        idx_nav = float(idx_val) / idx_base

        marker = ""
        if r2.date in bt_v2.rebalance_dates:
            marker = " ◆"
        print(f"  {r2.date:<12s} {v2_nav:>9.4f} {v3_nav:>9.4f} {v31_nav:>9.4f} {idx_nav:>9.4f}  "
              f"{r3.regime:>12s}{marker}")

    # 交易统计摘要
    print(f"\n{'─'*90}")
    print(f"  交易统计摘要")
    print(f"{'─'*90}")
    for ver in versions:
        bt = engines[ver]
        buys = [t for t in bt.trade_log if t["action"] == "BUY"]
        sells = [t for t in bt.trade_log if t["action"] == "SELL"]
        stops = [t for t in bt.trade_log if t["action"] == "STOP_LOSS"]
        print(f"  {ver}: 买入{len(buys)}笔 | 卖出{len(sells)}笔 | 止损{len(stops)}笔 | 总{len(bt.trade_log)}笔")

    # 最终持仓行业分布
    print(f"\n{'─'*90}")
    print(f"  最终持仓行业分布")
    print(f"{'─'*90}")
    for ver in versions:
        bt = engines[ver]
        inds = Counter(pos.industry for pos in bt.positions.values() if pos.industry)
        holdings = [(pos.name, pos.industry, pos.shares * bt._get_price(pos.ts_code, end_date))
                     for pos in bt.positions.values()]
        holdings.sort(key=lambda x: x[2], reverse=True)
        print(f"\n  {ver}: {len(bt.positions)}只, {len(inds)}个行业")
        for name, ind, mv in holdings[:10]:  # 最多显示10只
            print(f"    {name:8s} {ind:10s} 市值 ¥{mv:>10,.0f}")
        if len(holdings) > 10:
            print(f"    ... 还有{len(holdings)-10}只")
        if inds:
            print(f"    行业: {dict(inds)}")

    # 保存结果
    result = {
        "backtest_period": f"{start_date}~{end_date}",
        "trading_days": n_days,
        "v2.1": m2,
        "v3.0": m3,
        "v3.1": m31,
        "benchmark_return": idx_return,
    }
    for ver in versions:
        bt = engines[ver]
        result[f"{ver}_daily"] = [{"date": r.date, "nav": r.total_value / bt.initial_capital}
                                   for r in bt.daily_records]

    out_path = PROJECT_ROOT / "data" / "backtest_v1_vs_v2.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    print(f"\n[保存] 回测数据: {out_path}")


if __name__ == "__main__":
    main()
