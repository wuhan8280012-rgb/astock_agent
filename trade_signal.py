"""
Skill 8: 交易信号生成器 (trade_signal.py)
==========================================
定位: 整个系统的"扣扳机"模块

其他 Skill 回答的是"这只股票好不好" (评估)
本模块回答的是"今天要不要动手" (触发)

三大职责:
  1. 买入信号扫描: 候选池 + 今日行情 → 谁触发了入场条件
  2. 卖出信号扫描: 持仓 + 今日行情 → 谁触发了退出条件
  3. 仓位计算: 入场价 + 止损价 + 总资金 → 买多少

触发类型 (买入):
  - 阶梯突破: 收盘 > 整理区间上沿 + 放量 ≥ 1.5x
  - EMA突破: 首次站上 EMA10 且 EMA10 > EMA20
  - 放量阳线: 缩量整理后首日 涨幅>2% + 量比>2
  - 缺口突破: 跳空高开不回补

触发类型 (卖出):
  - 止损: 浮亏 ≥ 止损线 → 无条件 (优先级最高)
  - 趋势破位: 收盘跌破 EMA20 + 放量
  - 板块退出: 所在板块进入黑名单
  - 止盈分批: 盈利达 1R/2R/3R 分批卖出
  - 时间退出: 持仓超 N 天且浮盈 < 5%

A股适配:
  - T+1: 今日买入的股票不能当天卖出
  - 涨跌停: 触及跌停无法卖出，标记为"流动性锁死"
  - 集合竞价: 缺口突破以 9:25 定价为准
  - 10cm/20cm: 主板10%、创业板/科创板20%涨跌幅

输入: 候选池(Pipeline/StageFilter) + 持仓(RiskControl格式) + 日行情
输出: TradeSignal 列表 → 供 DebateEngine 做最终决策

调用方式:
  scanner = TradeSignalScanner()
  buy_signals = scanner.scan_buy(watchlist, env_score=72)
  sell_signals = scanner.scan_sell(holdings)
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_fetcher import get_fetcher

try:
    from config import SKILL_PARAMS
    PARAMS = SKILL_PARAMS.get("trade_signal", {})
except ImportError:
    PARAMS = {}


# ════════════════════════════════════════════════════════════
#  参数
# ════════════════════════════════════════════════════════════

DEFAULT_PARAMS = {
    # ── 买入触发 ──
    "breakout_vol_ratio": 1.5,    # 突破日量比阈值 (vs 20日均量)
    "ema_periods": [10, 20, 50],  # EMA 周期
    "gap_min_pct": 1.0,           # 缺口最小幅度 %
    "vol_expansion_ratio": 2.0,   # 放量阳线量比阈值
    "min_yang_pct": 2.0,          # 放量阳线最小涨幅 %

    # ── 卖出触发 ──
    "stop_loss_pct": 0.07,        # 止损线 7%
    "trend_break_vol_ratio": 1.3, # 趋势破位放量阈值
    "time_exit_days": 20,         # 时间退出天数
    "time_exit_min_profit": 0.05, # 时间退出最低收益
    "blacklist_exit_days": 3,     # 板块黑名单退出宽限天数

    # ── 止盈分批 ──
    "profit_take_1r": 0.33,       # 盈利1R时卖出比例
    "profit_take_2r": 0.33,       # 盈利2R时卖出比例
    "trailing_stop_atr": 2.0,     # 移动止损 ATR 倍数

    # ── 仓位计算 ──
    "risk_per_trade": 0.01,       # 单笔最大亏损占总资金 1%
    "max_single_position": 0.08,  # 单只最大仓位 8%
    "max_sector_position": 0.25,  # 板块最大仓位 25%
    "stop_atr_multiple": 2.0,     # 止损 = 入场 - N×ATR
    "target_atr_multiple": 6.0,   # 目标 = 入场 + N×ATR
    "min_rr_ratio": 3.0,          # 最低盈亏比
}

P = {**DEFAULT_PARAMS, **PARAMS}


# ════════════════════════════════════════════════════════════
#  数据结构
# ════════════════════════════════════════════════════════════

@dataclass
class TradeSignal:
    """交易信号"""
    ts_code: str
    name: str
    action: str                  # "BUY" / "SELL"
    signal_type: str             # 触发类型 (见下方常量)
    trigger_price: float = 0     # 触发价格 (突破价/跌破价)
    current_price: float = 0     # 当前价格
    position_size_pct: float = 0 # 建议仓位 % (BUY时)
    sell_ratio: float = 0        # 卖出比例 (SELL时, 0-1)
    stop_price: float = 0        # 止损价
    target_price: float = 0      # 目标价
    rr_ratio: float = 0          # 盈亏比
    urgency: str = "今日"         # "立即" / "今日" / "观察" / "3日内"
    reason: str = ""             # 触发理由
    sector: str = ""             # 所在板块
    pnl_pct: float = 0           # 当前盈亏% (SELL时)
    holding_days: int = 0        # 持仓天数 (SELL时)

    def to_dict(self) -> dict:
        return {
            "ts_code": self.ts_code,
            "name": self.name,
            "action": self.action,
            "signal_type": self.signal_type,
            "trigger_price": self.trigger_price,
            "current_price": self.current_price,
            "position_size_pct": round(self.position_size_pct, 3),
            "sell_ratio": round(self.sell_ratio, 2),
            "stop_price": self.stop_price,
            "target_price": self.target_price,
            "rr_ratio": round(self.rr_ratio, 1),
            "urgency": self.urgency,
            "reason": self.reason,
        }

    def to_brief(self) -> str:
        if self.action == "BUY":
            return (
                f"🟢 BUY {self.name}({self.ts_code}) "
                f"{self.signal_type} | "
                f"仓位{self.position_size_pct*100:.1f}% "
                f"入{self.current_price} 止{self.stop_price} "
                f"目标{self.target_price} R:R={self.rr_ratio:.1f} | "
                f"{self.urgency}"
            )
        else:
            return (
                f"🔴 SELL {self.name}({self.ts_code}) "
                f"{self.signal_type} | "
                f"卖{self.sell_ratio*100:.0f}% "
                f"盈亏{self.pnl_pct*100:+.1f}% "
                f"持{self.holding_days}天 | "
                f"{self.urgency}"
            )


# 信号类型常量
class SignalType:
    # 买入
    BREAKOUT = "阶梯突破"           # 价格突破整理区间 + 放量
    EMA_BREAK = "EMA突破"           # 首次站上 EMA + 均线多头
    VOLUME_YANG = "放量阳线"         # 缩量后首日放量大阳
    GAP_UP = "缺口突破"             # 跳空高开不回补
    # 卖出
    STOP_LOSS = "止损"              # 浮亏达止损线
    TREND_BREAK = "趋势破位"        # 跌破 EMA20 + 放量
    SECTOR_EXIT = "板块退出"        # 板块进入黑名单
    PROFIT_TAKE = "止盈分批"        # 盈利达 R 倍数
    TIME_EXIT = "时间退出"          # 持仓过久无表现
    TRAILING_STOP = "移动止损"      # 移动止损触发
    LIQUIDITY_LOCK = "流动性锁死"   # 跌停无法卖出


@dataclass
class SignalReport:
    """交易信号报告"""
    date: str
    buy_signals: List[TradeSignal] = field(default_factory=list)
    sell_signals: List[TradeSignal] = field(default_factory=list)
    scanned_watchlist: int = 0
    scanned_holdings: int = 0
    env_score: int = 0
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "env_score": self.env_score,
            "buy_signals": [s.to_dict() for s in self.buy_signals],
            "sell_signals": [s.to_dict() for s in self.sell_signals],
            "summary": self.summary,
        }

    def to_brief(self) -> str:
        buys = [f"{s.name}({s.signal_type})" for s in self.buy_signals[:3]]
        sells = [f"{s.name}({s.signal_type})" for s in self.sell_signals[:3]]
        return (
            f"[信号] 环境{self.env_score} | "
            f"买入{len(self.buy_signals)}只: {', '.join(buys) or '无'} | "
            f"卖出{len(self.sell_signals)}只: {', '.join(sells) or '无'}"
        )


# ════════════════════════════════════════════════════════════
#  核心引擎
# ════════════════════════════════════════════════════════════

class TradeSignalScanner:
    """
    交易信号扫描器

    盘后运行:
      scanner = TradeSignalScanner()
      report = scanner.scan(watchlist, holdings, env_score, blacklisted_sectors)
    """

    def __init__(self):
        self.fetcher = get_fetcher()

    def scan(
        self,
        watchlist: List[Dict] = None,
        holdings: List[Dict] = None,
        env_score: int = 65,
        blacklisted_sectors: List[str] = None,
        total_capital: float = 1_000_000,
        current_total_position: float = 0.0,
        verbose: bool = True,
    ) -> SignalReport:
        """
        完整扫描 = 买入扫描 + 卖出扫描

        Parameters
        ----------
        watchlist : Pipeline/StageFilter 输出的候选列表
            每项: {"ts_code", "name", "sector", "breakout_price",
                   "suggested_stop", "stage_score", ...}
        holdings : 当前持仓 (RiskControl 格式)
            每项: {"ts_code", "name", "sector", "position_pct",
                   "cost_price", "current_price", "buy_date", "stop_price"}
        env_score : 环境评分 (0-100)
        blacklisted_sectors : 黑名单板块名称列表
        total_capital : 总资金
        current_total_position : 当前总仓位 (0-1)
        """
        date = self.fetcher.get_latest_trade_date()
        report = SignalReport(date=date, env_score=env_score)

        watchlist = watchlist or []
        holdings = holdings or []
        blacklisted_sectors = blacklisted_sectors or []

        # ── 卖出信号优先扫描 (风控第一) ──
        if holdings:
            if verbose:
                print(f"[TradeSignal] 扫描卖出信号 ({len(holdings)}只持仓)...")
            report.sell_signals = self._scan_sell(
                holdings, blacklisted_sectors, verbose
            )
            report.scanned_holdings = len(holdings)

        # ── 买入信号扫描 ──
        if watchlist and env_score >= 60:
            if verbose:
                print(f"[TradeSignal] 扫描买入信号 ({len(watchlist)}只候选)...")
            report.buy_signals = self._scan_buy(
                watchlist, env_score, total_capital,
                current_total_position, blacklisted_sectors, verbose
            )
            report.scanned_watchlist = len(watchlist)
        elif env_score < 60:
            if verbose:
                print(f"[TradeSignal] 环境{env_score}<60，跳过买入扫描")

        # 排序: 卖出按紧急度，买入按盈亏比
        urgency_order = {"立即": 0, "今日": 1, "3日内": 2, "观察": 3}
        report.sell_signals.sort(key=lambda s: urgency_order.get(s.urgency, 9))
        report.buy_signals.sort(key=lambda s: -s.rr_ratio)

        # 摘要
        sell_immediate = sum(1 for s in report.sell_signals if s.urgency == "立即")
        report.summary = (
            f"环境{env_score} | "
            f"卖出信号{len(report.sell_signals)}只"
            f"(紧急{sell_immediate}) | "
            f"买入信号{len(report.buy_signals)}只"
        )

        return report

    # ────────────────────────────────────────────────────────
    #  买入信号扫描
    # ────────────────────────────────────────────────────────

    def _scan_buy(
        self,
        watchlist: List[Dict],
        env_score: int,
        total_capital: float,
        current_position: float,
        blacklisted: List[str],
        verbose: bool,
    ) -> List[TradeSignal]:
        """
        对候选池逐只扫描今日行情，检测入场事件

        返回: 触发了入场条件的信号列表
        """
        signals = []
        position_budget = self._max_position_for_env(env_score) - current_position

        if position_budget <= 0.02:
            if verbose:
                print(f"  仓位已满(剩余{position_budget*100:.1f}%)，跳过买入扫描")
            return signals

        for item in watchlist:
            ts_code = item.get("ts_code", "")
            name = item.get("name", ts_code)
            sector = item.get("sector", "")

            # 黑名单过滤
            if sector in blacklisted:
                continue

            try:
                df = self.fetcher.get_stock_daily(ts_code, days=60)
                if df is None or len(df) < 25:
                    continue

                df = df.sort_values("trade_date").reset_index(drop=True)

                # 计算技术指标
                df = self._add_indicators(df)
                today = df.iloc[-1]
                yesterday = df.iloc[-2]

                # 外部传入的突破价和止损价
                ext_breakout = item.get("breakout_price", 0)
                ext_stop = item.get("suggested_stop", 0)

                # ── 检测4种买入触发 ──
                signal = None

                # 1. 阶梯突破
                signal = signal or self._check_breakout(
                    df, today, yesterday, ext_breakout, name, ts_code, sector
                )
                # 2. EMA突破
                signal = signal or self._check_ema_break(
                    df, today, yesterday, name, ts_code, sector
                )
                # 3. 放量阳线
                signal = signal or self._check_volume_yang(
                    df, today, yesterday, name, ts_code, sector
                )
                # 4. 缺口突破
                signal = signal or self._check_gap_up(
                    df, today, yesterday, name, ts_code, sector
                )

                if signal:
                    # 计算止损/目标/盈亏比
                    self._compute_buy_levels(df, signal, ext_stop)

                    # 盈亏比过滤
                    if signal.rr_ratio < P["min_rr_ratio"]:
                        continue

                    # 仓位计算
                    signal.position_size_pct = self._calc_position_size(
                        signal.current_price,
                        signal.stop_price,
                        total_capital,
                    )

                    # 仓位预算检查
                    if signal.position_size_pct > position_budget:
                        signal.position_size_pct = round(position_budget, 3)

                    if signal.position_size_pct >= 0.01:  # 至少1%才值得
                        signals.append(signal)
                        position_budget -= signal.position_size_pct

                time.sleep(0.08)

            except Exception as e:
                if verbose:
                    print(f"  {ts_code} 扫描异常: {e}")
                continue

        return signals

    def _check_breakout(self, df, today, yesterday, ext_breakout, name, ts_code, sector):
        """
        阶梯突破: 收盘突破整理区间上沿 + 放量

        条件:
          1. 今日收盘 > 整理区间高点 (或外部传入的 breakout_price)
          2. 昨日收盘 ≤ 整理区间高点 (今天才突破)
          3. 今日成交量 ≥ 1.5x 20日均量
        """
        # 整理区间上沿: 用近20日最高价，或外部传入
        if ext_breakout > 0:
            resistance = ext_breakout
        else:
            resistance = df["high"].iloc[-21:-1].max() if len(df) > 21 else 0

        if resistance <= 0:
            return None

        close = float(today["close"])
        prev_close = float(yesterday["close"])
        vol_ratio = float(today["vol"]) / df["vol"].iloc[-21:-1].mean() if len(df) > 21 else 0

        if (close > resistance and
            prev_close <= resistance and
            vol_ratio >= P["breakout_vol_ratio"]):
            return TradeSignal(
                ts_code=ts_code, name=name, sector=sector,
                action="BUY",
                signal_type=SignalType.BREAKOUT,
                trigger_price=round(resistance, 2),
                current_price=round(close, 2),
                urgency="今日",
                reason=f"收盘{close:.2f}突破区间高点{resistance:.2f}, 量比{vol_ratio:.1f}x",
            )
        return None

    def _check_ema_break(self, df, today, yesterday, name, ts_code, sector):
        """
        EMA突破: 从下方首次站上 EMA10 且 EMA10 > EMA20

        条件:
          1. 昨日收盘 < EMA10
          2. 今日收盘 ≥ EMA10
          3. EMA10 > EMA20 (均线方向向上)
        """
        close = float(today["close"])
        prev_close = float(yesterday["close"])
        ema10 = float(today.get("ema10", 0))
        ema20 = float(today.get("ema20", 0))
        prev_ema10 = float(yesterday.get("ema10", 0))

        if (ema10 <= 0 or ema20 <= 0):
            return None

        if (prev_close < prev_ema10 and
            close >= ema10 and
            ema10 > ema20):
            return TradeSignal(
                ts_code=ts_code, name=name, sector=sector,
                action="BUY",
                signal_type=SignalType.EMA_BREAK,
                trigger_price=round(ema10, 2),
                current_price=round(close, 2),
                urgency="今日",
                reason=f"站上EMA10({ema10:.2f}), EMA10>EMA20({ema20:.2f}), 多头排列",
            )
        return None

    def _check_volume_yang(self, df, today, yesterday, name, ts_code, sector):
        """
        放量阳线: 缩量整理后首日大阳 + 放量

        条件:
          1. 前5日平均量 < 20日均量的 0.7 (之前在缩量)
          2. 今日涨幅 ≥ 2%
          3. 今日量比 ≥ 2x (vs 前5日)
        """
        if len(df) < 25:
            return None

        close = float(today["close"])
        prev_close = float(yesterday["close"])
        pct_change = (close / prev_close - 1) * 100

        vol_5d = df["vol"].iloc[-6:-1].mean()
        vol_20d = df["vol"].iloc[-21:-1].mean()
        today_vol = float(today["vol"])

        if vol_20d <= 0 or vol_5d <= 0:
            return None

        was_quiet = vol_5d < vol_20d * 0.7
        is_yang = pct_change >= P["min_yang_pct"]
        vol_expansion = today_vol / vol_5d

        if was_quiet and is_yang and vol_expansion >= P["vol_expansion_ratio"]:
            return TradeSignal(
                ts_code=ts_code, name=name, sector=sector,
                action="BUY",
                signal_type=SignalType.VOLUME_YANG,
                trigger_price=round(prev_close, 2),
                current_price=round(close, 2),
                urgency="今日",
                reason=f"缩量后放量阳线: 涨{pct_change:.1f}%, 量比{vol_expansion:.1f}x(vs前5日)",
            )
        return None

    def _check_gap_up(self, df, today, yesterday, name, ts_code, sector):
        """
        缺口突破: 跳空高开且未回补

        条件:
          1. 今开 > 昨高 + gap阈值
          2. 今低 > 昨高 (缺口未回补)
          3. 收阳 (收盘 > 开盘)
        """
        today_open = float(today["open"])
        today_low = float(today["low"])
        today_close = float(today["close"])
        yest_high = float(yesterday["high"])

        gap_pct = (today_open / yest_high - 1) * 100

        if (gap_pct >= P["gap_min_pct"] and
            today_low > yest_high and
            today_close > today_open):
            return TradeSignal(
                ts_code=ts_code, name=name, sector=sector,
                action="BUY",
                signal_type=SignalType.GAP_UP,
                trigger_price=round(yest_high, 2),
                current_price=round(today_close, 2),
                urgency="观察",  # 缺口突破次日确认更安全
                reason=f"跳空缺口{gap_pct:.1f}%未回补, 收阳确认",
            )
        return None

    def _compute_buy_levels(self, df, signal: TradeSignal, ext_stop: float):
        """计算止损价、目标价、盈亏比"""
        if len(df) < 14:
            return

        atr = self._calc_atr(df, 14)
        entry = signal.current_price

        # 止损: 优先用外部传入 (Pipeline/StageFilter 计算的)，否则用 ATR
        if ext_stop > 0 and ext_stop < entry:
            stop = ext_stop
        else:
            stop = entry - P["stop_atr_multiple"] * atr

        # 止损不超过10% (A股硬约束)
        stop = max(stop, entry * 0.90)

        # 目标: ATR 外推，但不超过30%
        target = entry + P["target_atr_multiple"] * atr
        target = min(target, entry * 1.30)

        risk = entry - stop
        reward = target - entry

        signal.stop_price = round(stop, 2)
        signal.target_price = round(target, 2)
        signal.rr_ratio = round(reward / risk, 1) if risk > 0 else 0

    # ────────────────────────────────────────────────────────
    #  卖出信号扫描
    # ────────────────────────────────────────────────────────

    def _scan_sell(
        self,
        holdings: List[Dict],
        blacklisted: List[str],
        verbose: bool,
    ) -> List[TradeSignal]:
        """
        对持仓逐只扫描，检测退出事件

        优先级排序:
          1. 止损 → 立即
          2. 流动性锁死 (跌停) → 标记但无法执行
          3. 趋势破位 → 今日
          4. 板块退出 → 3日内
          5. 止盈分批 → 今日
          6. 时间退出 → 观察
        """
        signals = []

        for pos in holdings:
            ts_code = pos.get("ts_code", "")
            name = pos.get("name", ts_code)
            sector = pos.get("sector", "")
            cost = pos.get("cost_price", 0)
            current = pos.get("current_price", 0)
            pos_stop = pos.get("stop_price", 0)  # 持仓记录中的止损价
            buy_date = pos.get("buy_date", "")

            if cost <= 0 or current <= 0:
                continue

            pnl_pct = current / cost - 1
            holding_days = self._calc_holding_days(buy_date)

            try:
                df = self.fetcher.get_stock_daily(ts_code, days=30)
                if df is None or len(df) < 10:
                    continue
                df = df.sort_values("trade_date").reset_index(drop=True)
                df = self._add_indicators(df)
                today = df.iloc[-1]
            except Exception:
                continue

            # ── 优先级1: 止损 ──
            stop_line = pos_stop if pos_stop > 0 else cost * (1 - P["stop_loss_pct"])
            if current <= stop_line:
                signals.append(TradeSignal(
                    ts_code=ts_code, name=name, sector=sector,
                    action="SELL", signal_type=SignalType.STOP_LOSS,
                    current_price=round(current, 2),
                    sell_ratio=1.0,
                    pnl_pct=round(pnl_pct, 4),
                    holding_days=holding_days,
                    urgency="立即",
                    reason=f"浮亏{pnl_pct*100:.1f}%, 跌破止损{stop_line:.2f}",
                ))
                # 检查是否跌停锁死
                limit_down = self._get_limit_down(cost, ts_code, name)
                if current <= limit_down * 1.001:
                    signals[-1].signal_type = SignalType.LIQUIDITY_LOCK
                    signals[-1].reason += " [跌停无法卖出]"
                    signals[-1].urgency = "立即"  # 标记但实际无法执行
                continue  # 止损信号不叠加其他

            # ── 优先级2: 趋势破位 ──
            ema20 = float(today.get("ema20", 0))
            vol_ratio = float(today["vol"]) / df["vol"].iloc[-21:-1].mean() if len(df) > 21 else 0
            prev_close = float(df.iloc[-2]["close"])

            if (ema20 > 0 and
                current < ema20 and
                prev_close >= float(df.iloc[-2].get("ema20", ema20)) and
                vol_ratio >= P["trend_break_vol_ratio"]):
                signals.append(TradeSignal(
                    ts_code=ts_code, name=name, sector=sector,
                    action="SELL", signal_type=SignalType.TREND_BREAK,
                    trigger_price=round(ema20, 2),
                    current_price=round(current, 2),
                    sell_ratio=1.0,
                    pnl_pct=round(pnl_pct, 4),
                    holding_days=holding_days,
                    urgency="今日",
                    reason=f"跌破EMA20({ema20:.2f}), 放量{vol_ratio:.1f}x",
                ))
                continue

            # ── 优先级3: 板块退出 ──
            if sector in blacklisted:
                signals.append(TradeSignal(
                    ts_code=ts_code, name=name, sector=sector,
                    action="SELL", signal_type=SignalType.SECTOR_EXIT,
                    current_price=round(current, 2),
                    sell_ratio=1.0,
                    pnl_pct=round(pnl_pct, 4),
                    holding_days=holding_days,
                    urgency="3日内",
                    reason=f"板块{sector}进入黑名单, {P['blacklist_exit_days']}日内退出",
                ))
                continue

            # ── 优先级4: 止盈分批 ──
            if pnl_pct > 0 and pos_stop > 0:
                risk_per_share = cost - pos_stop  # 1R
                if risk_per_share > 0:
                    r_multiple = (current - cost) / risk_per_share

                    if r_multiple >= 3:
                        signals.append(TradeSignal(
                            ts_code=ts_code, name=name, sector=sector,
                            action="SELL", signal_type=SignalType.TRAILING_STOP,
                            current_price=round(current, 2),
                            sell_ratio=P["profit_take_2r"],
                            pnl_pct=round(pnl_pct, 4),
                            holding_days=holding_days,
                            urgency="今日",
                            reason=f"盈利{r_multiple:.1f}R, 卖出{P['profit_take_2r']*100:.0f}%, "
                                   f"移动止损上移至+{r_multiple-1:.0f}R",
                        ))
                    elif r_multiple >= 2:
                        signals.append(TradeSignal(
                            ts_code=ts_code, name=name, sector=sector,
                            action="SELL", signal_type=SignalType.PROFIT_TAKE,
                            current_price=round(current, 2),
                            sell_ratio=P["profit_take_1r"],
                            pnl_pct=round(pnl_pct, 4),
                            holding_days=holding_days,
                            urgency="今日",
                            reason=f"盈利{r_multiple:.1f}R, 卖出{P['profit_take_1r']*100:.0f}%, "
                                   f"止损上移至+1R({cost + risk_per_share:.2f})",
                        ))
                    elif r_multiple >= 1:
                        signals.append(TradeSignal(
                            ts_code=ts_code, name=name, sector=sector,
                            action="SELL", signal_type=SignalType.PROFIT_TAKE,
                            current_price=round(current, 2),
                            sell_ratio=P["profit_take_1r"],
                            pnl_pct=round(pnl_pct, 4),
                            holding_days=holding_days,
                            urgency="今日",
                            reason=f"盈利{r_multiple:.1f}R, 卖出{P['profit_take_1r']*100:.0f}%, "
                                   f"止损上移至成本({cost:.2f})",
                        ))

            # ── 优先级5: 时间退出 ──
            if (holding_days >= P["time_exit_days"] and
                pnl_pct < P["time_exit_min_profit"]):
                signals.append(TradeSignal(
                    ts_code=ts_code, name=name, sector=sector,
                    action="SELL", signal_type=SignalType.TIME_EXIT,
                    current_price=round(current, 2),
                    sell_ratio=1.0,
                    pnl_pct=round(pnl_pct, 4),
                    holding_days=holding_days,
                    urgency="观察",
                    reason=f"持仓{holding_days}天, 浮盈仅{pnl_pct*100:.1f}%<{P['time_exit_min_profit']*100:.0f}%, "
                           f"资金效率低",
                ))

            time.sleep(0.05)

        return signals

    # ────────────────────────────────────────────────────────
    #  仓位计算
    # ────────────────────────────────────────────────────────

    def _calc_position_size(
        self,
        entry: float,
        stop: float,
        total_capital: float,
    ) -> float:
        """
        固定风险仓位计算

        公式: 仓位% = (总资金 × 单笔风险%) ÷ ((入场-止损) × 入场价 ÷ 入场价)
             简化: 股数 = (总资金 × 1%) ÷ (入场 - 止损)
                   仓位% = 股数 × 入场 ÷ 总资金

        约束:
          1. 单只上限 8%
          2. 止损距离 < 1% 时按1%计算 (防除零)
        """
        if entry <= 0 or stop <= 0 or stop >= entry:
            return 0

        risk_per_share = entry - stop
        risk_per_share = max(risk_per_share, entry * 0.01)  # 至少1%止损距离

        max_loss = total_capital * P["risk_per_trade"]
        shares = max_loss / risk_per_share
        position_value = shares * entry
        position_pct = position_value / total_capital

        # 上限
        position_pct = min(position_pct, P["max_single_position"])

        return round(position_pct, 4)

    def _max_position_for_env(self, env_score: int) -> float:
        """环境评分 → 允许的最大总仓位"""
        if env_score >= 80:
            return 0.80
        elif env_score >= 70:
            return 0.60
        elif env_score >= 60:
            return 0.40
        else:
            return 0.0  # 不允许买入

    # ────────────────────────────────────────────────────────
    #  工具方法
    # ────────────────────────────────────────────────────────

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """给日线数据添加 EMA 指标"""
        for period in P["ema_periods"]:
            col = f"ema{period}"
            if col not in df.columns:
                df[col] = df["close"].ewm(span=period, adjust=False).mean()
        return df

    def _calc_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """计算ATR"""
        if len(df) < period + 1:
            return 0
        d = df.copy()
        d["prev_close"] = d["close"].shift(1)
        d["tr"] = np.maximum(
            d["high"] - d["low"],
            np.maximum(
                abs(d["high"] - d["prev_close"]),
                abs(d["low"] - d["prev_close"])
            )
        )
        return d["tr"].tail(period).mean()

    def _calc_holding_days(self, buy_date: str) -> int:
        """计算持仓天数"""
        if not buy_date:
            return 0
        try:
            buy = datetime.strptime(buy_date[:8], "%Y%m%d")
            now = datetime.now()
            return (now - buy).days
        except Exception:
            return 0

    def _get_limit_down(self, cost: float, ts_code: str, name: str = "") -> float:
        """
        获取跌停价

        A股规则:
          主板 (60xxxx, 00xxxx): ±10%
          创业板 (30xxxx): ±20%
          科创板 (68xxxx): ±20%
          ST (名称含ST): ±5%

        注: ST 股通过名称识别 (如 "ST国华", "*ST金科")
            代码本身是正常6位数字
        """
        code = ts_code[:6] if isinstance(ts_code, str) else ""
        name = name or ""

        # ST 判断优先 (名称中含 ST)
        if "ST" in name.upper():
            return cost * 0.95
        elif code.startswith("30") or code.startswith("68"):
            return cost * 0.80
        else:
            return cost * 0.90


# ════════════════════════════════════════════════════════════
#  独立运行
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("  交易信号扫描器")
    print("=" * 80)

    # 示例: 需要配合真实数据运行
    # scanner = TradeSignalScanner()
    # report = scanner.scan(
    #     watchlist=[...],    # Pipeline 输出
    #     holdings=[...],     # 持仓数据
    #     env_score=72,
    #     blacklisted_sectors=["房地产", "建筑"],
    #     total_capital=1_000_000,
    #     current_total_position=0.36,
    # )
    # print(report.to_brief())

    print("  请配合 Pipeline 候选池和持仓数据运行")
    print("  示例: python -c 'from trade_signal import *; ...'")
    print("=" * 80)
