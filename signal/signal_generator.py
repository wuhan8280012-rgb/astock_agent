#!/usr/bin/env python3
"""
动量轮动策略 — 每日信号生成器 (半自动实盘版)

功能:
  1. 通过 tushare API 拉取最新行情 (或从CSV加载)
  2. 运行 v2.0 优化策略, 计算全市场动量排名
  3. 与当前持仓对比, 生成调仓指令 (买入/卖出/持有)
  4. 输出结构化信号 (JSON + 可读文本)

用法:
  from signal.signal_generator import SignalGenerator
  gen = SignalGenerator(tushare_token="xxx")
  result = gen.generate()  # 返回 SignalResult

设计原则:
  - 零外部依赖 (仅 pandas, numpy, tushare)
  - 持仓状态以 JSON 文件持久化, 无需数据库
  - 可独立运行, 也可被调度器调用
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# ── Project path ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ══════════════════════════════════════════════════════════════════════════════
#  Data Classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SignalConfig:
    """
    信号生成配置 — v3.1 生产版

    策略核心:
      固定权重[0.5,0.3,0.2]单因子动量 + 龙虎榜加分 + 无个股止损 + HALT清仓保护

    v3.1 相比 v3.0 的变更 (300天真实龙虎榜数据回测验证):
      1. enable_lhb_factor=True   — 龙虎榜净买入加分, w=0.30, 回看10天
         年化 41.71% vs 26.45% (基准), 夏普 1.05 vs 0.63

    v3.0 相比 v2.1 的变更 (300天回测验证):
      1. adaptive_weights=False  — 固定权重比自适应多赚10% (29% vs 17%)
      2. stop_loss_pct=-0.99     — 移除个股止损, 减少无效交易摩擦
      3. enable_trend_window=False — 死叉过滤因交易成本反而拖累收益
      4. HALT清仓机制保留        — 强制组合刷新是alpha核心来源
    """
    # 动量参数
    lookback_days: list[int] = field(default_factory=lambda: [5, 10, 20])
    lookback_weights: list[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])
    volatility_penalty: float = 0.5
    liquidity_weight: float = 0.15          # v3.2: 进化验证, 偏好高流动性降低隐性滑点

    # ── 市场状态自适应权重 ──
    # DEFENSIVE: 降低5日短期权重, 偏中长期趋势, 避免追高位反转股
    defensive_weights: list[float] = field(default_factory=lambda: [0.2, 0.4, 0.4])
    # STRONG_RUN: 加大5日短期权重, 捕捉热点轮动
    strong_run_weights: list[float] = field(default_factory=lambda: [0.6, 0.25, 0.15])
    # HALT: 最大化长期趋势权重 (仅用于评分排名, 实际不买入)
    halt_weights: list[float] = field(default_factory=lambda: [0.1, 0.3, 0.6])
    # 是否启用自适应权重 (关闭则所有状态用 lookback_weights)
    adaptive_weights: bool = False          # v3.0: 固定权重更优

    # ── 趋势窗口过滤 ──
    # 指数MA5 < MA20 (死叉) 时, 降级为WAIT; 因交易摩擦过高默认关闭
    enable_trend_window: bool = False       # v3.0: 关闭
    trend_ma_short: int = 5               # 短期均线天数
    trend_ma_long: int = 20               # 长期均线天数

    # ── 反转保护 (超买惩罚) ──
    enable_reversal_filter: bool = False    # 默认关闭; 强动量市场中会拖累收益
    rsi_lookback: int = 10
    rsi_overbought: float = 85.0       # RSI超过此值视为超买
    rsi_penalty: float = 0.5           # 单项超买惩罚系数
    bias_lookback: int = 5
    bias_overbought: float = 0.15      # 乖离率超过15%视为超买

    # 组合参数
    top_n: int = 10
    hold_buffer_ratio: float = 1.2           # v3.2: 进化验证, 收紧缓冲带加快淘汰动量衰退持仓
    max_single_weight: float = 0.15
    min_position_amount: float = 10_000  # 单只最小持仓金额, 低于此不建仓
    max_total_position: float = 0.80
    stop_loss_pct: float = -0.99        # v3.0: 无个股止损 (仅HALT系统性保护)
    rebalance_interval_days: int = 5

    # 过滤参数
    min_amount_20d: float = 1e8     # 20日均成交额 >= 1亿
    min_price: float = 5.0
    max_price: float = 500.0
    min_list_days: int = 120
    exclude_st: bool = True

    # ── 龙虎榜因子 (v3.1) ──
    # 回测最优: boost模式, w=0.30, 回看10天 → 年化41.71%, 夏普1.05
    enable_lhb_factor: bool = True           # v3.1: 启用龙虎榜加分
    lhb_weight: float = 0.30                 # 净买入加分权重
    lhb_lookback: int = 10                   # 回看天数
    lhb_negative_penalty: float = 0.3        # 净卖出惩罚系数 (乘以lhb_weight)

    # 执行参数
    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.001
    slippage_pct: float = 0.002        # v3.1: 上调至0.2% (实盘建仓日均高开1.12%, 限价单可控制在0.2%内)
    # 限价单保护: 开盘价偏离上日收盘超过此阈值则跳过买入
    open_gap_limit: float = 0.05       # 高开超过5%不追 (实盘国科微高开7.3%当日跌回)

    # 数据源
    tushare_token: str = ""
    data_csv_path: str = ""         # 可选: CSV文件路径 (离线模式)
    universe: str = "csi1000"       # csi1000 | csi500 | csi300

    # 持仓文件路径
    portfolio_path: str = ""

    # 消息层配置
    enable_macro_calendar: bool = True   # 启用宏观事件日历
    enable_sentiment: bool = False       # 启用舆情扫描 (需要tushare新闻接口)
    enable_breaking_monitor: bool = True # 启用突发事件监听
    custom_events_path: str = ""         # 自定义事件JSON路径


@dataclass
class TradeSignal:
    """单条交易信号"""
    action: str              # BUY / SELL / STOP_LOSS / HOLD
    ts_code: str
    name: str
    current_price: float
    target_weight: float     # 目标权重 (0~1)
    target_shares: int       # 目标股数 (100的整数倍)
    target_amount: float     # 预估金额
    momentum_score: float    # 动量得分
    rank: int                # 当前排名
    reason: str              # 说明
    urgency: str = "NORMAL"  # HIGH(止损) / NORMAL / LOW(缓冲带保留)


@dataclass
class SignalResult:
    """信号生成结果"""
    success: bool
    generated_at: str
    trade_date: str
    market_regime: str
    total_value: float
    cash: float
    signals: list[TradeSignal] = field(default_factory=list)
    top_momentum: list[dict] = field(default_factory=list)  # 前30名动量排行
    current_holdings: list[dict] = field(default_factory=list)
    summary_text: str = ""
    error_message: str = ""
    macro_risk: dict = field(default_factory=dict)      # 宏观风险评估
    sentiment: dict = field(default_factory=dict)        # 舆情情绪
    breaking_alerts: dict = field(default_factory=dict)  # 突发事件告警

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_json(self, path: str = None) -> str:
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text, encoding="utf-8")
        return text


# ══════════════════════════════════════════════════════════════════════════════
#  Portfolio State (JSON-based persistence)
# ══════════════════════════════════════════════════════════════════════════════

class PortfolioState:
    """
    JSON文件持久化的持仓状态管理器。
    无需数据库, 适合个人投资者使用。
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text("utf-8"))
        return {
            "initial_capital": 1_000_000.0,
            "cash": 1_000_000.0,
            "positions": {},       # {ts_code: {shares, cost_price, entry_date, peak_price, name}}
            "last_rebalance_date": "",
            "rebalance_count": 0,
            "trade_history": [],
            "updated_at": "",
        }

    def save(self):
        self.data["updated_at"] = datetime.now().isoformat()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @property
    def cash(self) -> float:
        return self.data.get("cash", 0)

    @property
    def positions(self) -> dict:
        return self.data.get("positions", {})

    @property
    def total_value(self) -> float:
        pos_value = sum(
            p.get("shares", 0) * p.get("current_price", p.get("cost_price", 0))
            for p in self.positions.values()
        )
        return self.cash + pos_value

    @property
    def last_rebalance_date(self) -> str:
        return self.data.get("last_rebalance_date", "")

    def update_position_prices(self, prices: dict[str, float]):
        """用最新价格更新所有持仓"""
        for code, pos in self.positions.items():
            if code in prices:
                pos["current_price"] = prices[code]
                pos["peak_price"] = max(pos.get("peak_price", 0), prices[code])

    def apply_trade(self, signal: TradeSignal, exec_price: float):
        """执行交易后更新状态 (手动确认后调用)"""
        if signal.action == "BUY":
            cost = signal.target_shares * exec_price
            commission = max(cost * 0.0003, 5)
            self.data["cash"] -= (cost + commission)
            if signal.ts_code in self.positions:
                pos = self.positions[signal.ts_code]
                old_shares = pos["shares"]
                pos["cost_price"] = (pos["cost_price"] * old_shares + exec_price * signal.target_shares) / (old_shares + signal.target_shares)
                pos["shares"] += signal.target_shares
            else:
                self.data["positions"][signal.ts_code] = {
                    "shares": signal.target_shares,
                    "cost_price": exec_price,
                    "entry_date": datetime.now().strftime("%Y%m%d"),
                    "peak_price": exec_price,
                    "current_price": exec_price,
                    "name": signal.name,
                }
        elif signal.action in ("SELL", "STOP_LOSS"):
            if signal.ts_code in self.positions:
                pos = self.positions[signal.ts_code]
                proceeds = pos["shares"] * exec_price
                commission = max(proceeds * 0.0003, 5)
                stamp_tax = proceeds * 0.001
                self.data["cash"] += (proceeds - commission - stamp_tax)
                del self.data["positions"][signal.ts_code]

        self.data["trade_history"].append({
            "date": datetime.now().strftime("%Y%m%d"),
            "action": signal.action,
            "ts_code": signal.ts_code,
            "name": signal.name,
            "shares": signal.target_shares,
            "price": exec_price,
        })
        self.save()


# ══════════════════════════════════════════════════════════════════════════════
#  Signal Generator
# ══════════════════════════════════════════════════════════════════════════════

class SignalGenerator:
    """每日交易信号生成器"""

    def __init__(self, config: SignalConfig = None):
        self.config = config or SignalConfig()
        # 持仓状态
        portfolio_path = self.config.portfolio_path or str(
            PROJECT_ROOT / "data" / "portfolio_state.json"
        )
        self.portfolio = PortfolioState(portfolio_path)

        # 消息层 (延迟初始化)
        self._macro_calendar = None
        self._sentiment_scanner = None
        self._breaking_monitor = None

    @property
    def macro_calendar(self):
        if self._macro_calendar is None and self.config.enable_macro_calendar:
            from signal.macro_calendar import MacroCalendar
            custom_path = self.config.custom_events_path or str(
                PROJECT_ROOT / "config" / "custom_events.json"
            )
            self._macro_calendar = MacroCalendar(custom_events_path=custom_path)
        return self._macro_calendar

    @property
    def sentiment_scanner(self):
        if self._sentiment_scanner is None and self.config.enable_sentiment:
            from signal.sentiment_scanner import SentimentScanner
            self._sentiment_scanner = SentimentScanner(
                tushare_token=self.config.tushare_token
            )
        return self._sentiment_scanner

    @property
    def breaking_monitor(self):
        if self._breaking_monitor is None and self.config.enable_breaking_monitor:
            from signal.breaking_monitor import BreakingMonitor
            self._breaking_monitor = BreakingMonitor(
                tushare_token=self.config.tushare_token,
                alert_log_path=str(PROJECT_ROOT / "data" / "breaking_alerts.json"),
            )
        return self._breaking_monitor

    # ── 数据获取 ──

    def _fetch_tushare_data(self) -> dict:
        """通过 tushare API 拉取最近30个交易日数据"""
        import tushare as ts
        pro = ts.pro_api(self.config.tushare_token)

        today = datetime.now().strftime("%Y%m%d")

        # 获取交易日历 (取最近60天覆盖)
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y%m%d")
        cal = pro.trade_cal(start_date=start_date, end_date=today)
        trade_dates = sorted(
            cal[cal["is_open"] == 1]["cal_date"].tolist()
        )
        # 取最近35个交易日 (20日回望 + 余量)
        recent_dates = trade_dates[-35:]
        fetch_start = recent_dates[0]

        print(f"[数据] 获取 {fetch_start} ~ {today} 行情...")

        # 获取中证1000成分股
        if self.config.universe == "csi1000":
            index_code = "000852.SH"
        elif self.config.universe == "csi500":
            index_code = "000905.SH"
        else:
            index_code = "000300.SH"

        members = pro.index_weight(index_code=index_code, start_date=today, end_date=today)
        if members.empty:
            # 取最近一期
            members = pro.index_weight(index_code=index_code)
            members = members.head(1000)
        ts_codes = members["con_code"].tolist()
        print(f"[数据] 成分股: {len(ts_codes)} 只")

        # 分批获取日线数据
        all_daily = []
        batch_size = 50
        for i in range(0, len(ts_codes), batch_size):
            batch = ts_codes[i:i + batch_size]
            codes_str = ",".join(batch)
            df = pro.daily(
                ts_code=codes_str,
                start_date=fetch_start,
                end_date=today,
                fields="ts_code,trade_date,open,high,low,close,vol,amount,pct_chg"
            )
            if not df.empty:
                all_daily.append(df)
            # tushare 限速
            import time
            time.sleep(0.3)

        daily_df = pd.concat(all_daily, ignore_index=True) if all_daily else pd.DataFrame()
        print(f"[数据] 日线数据: {len(daily_df):,} 行")

        # 获取个股基本信息
        basic = pro.stock_basic(fields="ts_code,name,industry,market,list_date,list_status")
        basic = basic[basic["ts_code"].isin(ts_codes)]

        # 获取沪深300指数日线 (市场状态判断)
        idx = pro.index_daily(ts_code="000300.SH", start_date=fetch_start, end_date=today)

        # 整理为统一格式
        stock_data = {}
        for ts_code, grp in daily_df.groupby("ts_code"):
            df = grp.sort_values("trade_date").reset_index(drop=True)
            stock_data[ts_code] = df

        # ── 龙虎榜数据 (tushare top_list) ──
        # top_list 是单日接口, 必须传 trade_date, 不支持区间查询
        # 逐日拉取最近 lhb_lookback 个交易日的龙虎榜
        lhb_data = {}
        if self.config.enable_lhb_factor:
            import time
            lhb_dates = recent_dates[-self.config.lhb_lookback:]
            total_rows = 0
            for lhb_date in lhb_dates:
                try:
                    lhb_df = pro.top_list(
                        trade_date=lhb_date,
                        fields="trade_date,ts_code,name,close,pct_change,turnover_rate,amount,l_sell,l_buy,net_mf_amount"
                    )
                    if lhb_df is not None and not lhb_df.empty:
                        for _, row in lhb_df.iterrows():
                            ts_code = row["ts_code"]
                            date = str(row["trade_date"])
                            l_buy = float(row.get("l_buy", 0) or 0)
                            l_sell = float(row.get("l_sell", 0) or 0)
                            if ts_code not in lhb_data:
                                lhb_data[ts_code] = {}
                            if date in lhb_data[ts_code]:
                                lhb_data[ts_code][date]["l_buy"] += l_buy
                                lhb_data[ts_code][date]["l_sell"] += l_sell
                                lhb_data[ts_code][date]["net_buy"] = lhb_data[ts_code][date]["l_buy"] - lhb_data[ts_code][date]["l_sell"]
                            else:
                                lhb_data[ts_code][date] = {"l_buy": l_buy, "l_sell": l_sell, "net_buy": l_buy - l_sell}
                        total_rows += len(lhb_df)
                    time.sleep(0.3)  # tushare 限速
                except Exception as e:
                    print(f"[数据] 龙虎榜 {lhb_date} 获取失败: {e}")
                    continue
            if lhb_data:
                print(f"[数据] 龙虎榜: {len(lhb_data)} 只股票, {total_rows} 条记录 (回看{len(lhb_dates)}天)")

        return {
            "stock_data": stock_data,
            "trade_calendar": recent_dates,
            "index_data": idx.sort_values("trade_date").reset_index(drop=True),
            "stock_info": basic,
            "daily_df": daily_df,
            "lhb_data": lhb_data,
        }

    def _load_csv_data(self) -> dict:
        """从CSV文件加载数据 (离线模式)"""
        csv_path = self.config.data_csv_path
        print(f"[数据] 加载CSV: {csv_path}")
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
            df = grp.sort_values("trade_date").reset_index(drop=True)
            stock_data[ts_code] = df

        # ── 龙虎榜数据 (top_list) ──
        lhb_data = {}
        if self.config.enable_lhb_factor:
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
                print(f"[数据] 龙虎榜: {len(lhb_data)} 只股票, {sum(len(v) for v in lhb_data.values())} 条记录")

        return {
            "stock_data": stock_data,
            "trade_calendar": trade_dates,
            "index_data": index_df.sort_values("trade_date").reset_index(drop=True),
            "stock_info": basic_df[["ts_code", "name", "industry", "market", "list_date", "list_status"]].copy(),
            "daily_df": daily_df,
            "lhb_data": lhb_data,
        }

    # ── 策略计算 ──

    def _get_regime_weights(self, regime: str) -> list[float]:
        """根据市场状态返回对应的 lookback_weights。

        设计逻辑:
          DEFENSIVE → [0.2, 0.4, 0.4]  降低5日权重, 偏中长期
            复盘依据: 20260320 DEFENSIVE 状态下, 5日暴涨股全线回撤-8.58%,
            而10/20日稳健趋势股仅-0.05%. 短期动量在弱市极易反转.
          STRONG_RUN → [0.6, 0.25, 0.15] 短期动量主导, 热点轮动快
          HALT → [0.1, 0.3, 0.6] 最大化长期趋势 (实际不买入, 仅排名参考)
          RUN → [0.5, 0.3, 0.2] 默认均衡配置
        """
        cfg = self.config
        if not cfg.adaptive_weights:
            return cfg.lookback_weights

        weight_map = {
            "HALT": cfg.halt_weights,
            "DEFENSIVE": cfg.defensive_weights,
            "STRONG_RUN": cfg.strong_run_weights,
        }
        return weight_map.get(regime, cfg.lookback_weights)

    def _calc_momentum(self, closes: np.ndarray, lookback: int) -> float:
        if len(closes) < lookback + 1:
            return np.nan
        return closes[-1] / closes[-(lookback + 1)] - 1

    def _calc_volatility(self, closes: np.ndarray, lookback: int) -> float:
        if len(closes) < lookback + 1:
            return np.nan
        rets = np.diff(closes[-lookback - 1:]) / closes[-lookback - 1:-1]
        return float(np.std(rets)) if len(rets) > 1 else 0.0

    def _calc_rsi(self, closes: np.ndarray, lookback: int = 10) -> float:
        """RSI 相对强弱指标, 用于超买检测。"""
        if len(closes) < lookback + 1:
            return 50.0
        changes = np.diff(closes[-(lookback + 1):])
        gains = np.maximum(changes, 0)
        losses = np.abs(np.minimum(changes, 0))
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1.0 + rs)

    def _calc_bias(self, closes: np.ndarray, lookback: int = 5) -> float:
        """乖离率 — 当前价相对N日均线的偏离程度。"""
        if len(closes) < lookback:
            return 0.0
        ma = np.mean(closes[-lookback:])
        if ma <= 0:
            return 0.0
        return float(closes[-1] / ma - 1)

    def _calc_composite_score(self, closes: np.ndarray, avg_amount: float,
                              weights: list[float] = None) -> float:
        """计算复合动量得分 (含反转保护)"""
        cfg = self.config
        active_weights = weights or cfg.lookback_weights
        score = 0.0

        for lb, w in zip(cfg.lookback_days, active_weights):
            m = self._calc_momentum(closes, lb)
            if np.isnan(m):
                return np.nan

            v = self._calc_volatility(closes, lb)
            if np.isnan(v):
                return np.nan

            score += w * (m - cfg.volatility_penalty * v)

        # 流动性加分 (对数缩放)
        if cfg.liquidity_weight > 0 and avg_amount > 0:
            liq_score = np.log10(max(avg_amount, 1e6)) / 12.0
            score = (1 - cfg.liquidity_weight) * score + cfg.liquidity_weight * liq_score

        # 反转保护: RSI + 乖离率超买惩罚
        if cfg.enable_reversal_filter:
            rsi = self._calc_rsi(closes, cfg.rsi_lookback)
            bias = self._calc_bias(closes, cfg.bias_lookback)

            if rsi > cfg.rsi_overbought and bias > cfg.bias_overbought:
                score *= cfg.rsi_penalty * 0.7   # 双重超买: 强惩罚 (0.35x)
            elif rsi > cfg.rsi_overbought:
                score *= cfg.rsi_penalty          # RSI超买: 0.5x
            elif bias > cfg.bias_overbought:
                score *= (cfg.rsi_penalty + 1.0) / 2  # 乖离率超买: 0.75x

        return score

    def _check_market_regime(self, index_data: pd.DataFrame, date: str) -> str:
        """市场状态判断 — 含趋势窗口过滤"""
        cfg = self.config
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
            regime = "STRONG_RUN"
        else:
            regime = "RUN"

        # ── 趋势窗口过滤: 死叉空仓 ──
        # MA_short < MA_long 时, 降级为 WAIT (完全空仓, 规避垃圾时间)
        if cfg.enable_trend_window and regime in ("RUN", "STRONG_RUN"):
            hist_long = index_data[index_data["trade_date"] <= date].tail(cfg.trend_ma_long)
            hist_short = index_data[index_data["trade_date"] <= date].tail(cfg.trend_ma_short)
            if len(hist_long) >= cfg.trend_ma_long and len(hist_short) >= cfg.trend_ma_short:
                ma_short = hist_short["close"].mean()
                ma_long = hist_long["close"].mean()
                if ma_short < ma_long:
                    regime = "WAIT"

        return regime

    def _calc_lhb_signal(self, ts_code: str, date: str, lhb_data: dict,
                         lookback: int, trade_dates: list[str]) -> tuple[float, int]:
        """计算龙虎榜信号: 回看N天内的净买入情况"""
        if ts_code not in lhb_data:
            return 0.0, 0
        try:
            idx = trade_dates.index(date)
        except ValueError:
            return 0.0, 0
        window = set(trade_dates[max(0, idx - lookback):idx + 1])
        stock_lhb = lhb_data[ts_code]
        total_net = 0.0
        count = 0
        for d in window:
            if d in stock_lhb:
                total_net += stock_lhb[d]["net_buy"]
                count += 1
        return total_net, count

    def _filter_universe(self, stock_data: dict, stock_info: pd.DataFrame,
                         date: str) -> list[str]:
        """过滤可交易标的"""
        cfg = self.config
        candidates = []

        info_map = {}
        for _, row in stock_info.iterrows():
            info_map[row["ts_code"]] = row

        for ts_code, df in stock_data.items():
            hist = df[df["trade_date"] <= date]
            if len(hist) < max(cfg.lookback_days) + 5:
                continue

            latest = hist.iloc[-1]
            close = latest["close"]
            if np.isnan(close) or close < cfg.min_price or close > cfg.max_price:
                continue

            # 成交额过滤
            recent = hist.tail(20)
            avg_amount = recent["amount"].mean() * 1000
            if avg_amount < cfg.min_amount_20d:
                continue

            # ST过滤
            if cfg.exclude_st:
                info = info_map.get(ts_code)
                if info is not None:
                    name = str(info.get("name", ""))
                    if "ST" in name.upper():
                        continue

            # 上市天数过滤
            info = info_map.get(ts_code)
            if info is not None:
                list_date = str(info.get("list_date", ""))
                if list_date and len(list_date) >= 8:
                    try:
                        list_dt = datetime.strptime(list_date[:8], "%Y%m%d")
                        curr_dt = datetime.strptime(date, "%Y%m%d")
                        if (curr_dt - list_dt).days < cfg.min_list_days:
                            continue
                    except ValueError:
                        pass

            # 停牌检查
            if latest["trade_date"] != date:
                continue

            # 涨停不买
            if latest["pct_chg"] >= 9.5:
                continue

            candidates.append(ts_code)

        return candidates

    # ── 信号生成 ──

    def generate(self, as_of_date: str = None) -> SignalResult:
        """
        生成今日交易信号。

        Args:
            as_of_date: 指定日期 (YYYYMMDD), 默认最新交易日

        Returns:
            SignalResult 对象
        """
        cfg = self.config
        now = datetime.now().isoformat()

        try:
            # 1. 获取数据
            if cfg.data_csv_path:
                market_data = self._load_csv_data()
            else:
                market_data = self._fetch_tushare_data()

            stock_data = market_data["stock_data"]
            index_data = market_data["index_data"]
            stock_info = market_data["stock_info"]
            trade_cal = market_data["trade_calendar"]

            # 确定分析日期
            if as_of_date:
                trade_date = as_of_date
            else:
                trade_date = trade_cal[-1]  # 最新交易日

            print(f"\n[信号] 分析日期: {trade_date}")
            print(f"[信号] 标的池: {len(stock_data)} 只")

            # 2. 市场状态
            regime = self._check_market_regime(index_data, trade_date)
            print(f"[信号] 市场状态: {regime}")

            # 2b. 消息层: 宏观事件日历
            macro_assessment = None
            if self.macro_calendar:
                macro_assessment = self.macro_calendar.assess(trade_date)
                if macro_assessment.alert_text:
                    print(f"\n{macro_assessment.alert_text}")

            # 2c. 消息层: 舆情扫描
            sentiment_result = None
            if self.sentiment_scanner:
                sentiment_result = self.sentiment_scanner.scan(date=trade_date)
                if sentiment_result.alert_text:
                    print(f"\n{sentiment_result.alert_text}")

            # 2d. 消息层: 突发事件监听
            breaking_result = None
            if self.breaking_monitor:
                breaking_result = self.breaking_monitor.scan()
                if breaking_result.alerts:
                    print(f"\n{breaking_result.emergency_text}")

            # 3. 过滤宇宙
            candidates = self._filter_universe(stock_data, stock_info, trade_date)
            print(f"[信号] 可交易标的: {len(candidates)} 只")

            # 4. 计算动量得分 (自适应权重)
            active_weights = self._get_regime_weights(regime)
            weight_desc = f"[{'/'.join(f'{w:.0%}' for w in active_weights)}]"
            print(f"[信号] 动量权重 (5d/10d/20d): {weight_desc} ← {regime}")

            name_map = {}
            info_map_dict = {}
            for _, row in stock_info.iterrows():
                name_map[row["ts_code"]] = str(row.get("name", row["ts_code"]))
                info_map_dict[row["ts_code"]] = row.to_dict()

            scores = []
            for ts_code in candidates:
                df = stock_data[ts_code]
                hist = df[df["trade_date"] <= trade_date]
                closes = hist["close"].values.astype(float)
                avg_amount = hist.tail(20)["amount"].mean() * 1000
                score = self._calc_composite_score(
                    closes, avg_amount, weights=active_weights
                )
                if not np.isnan(score):
                    scores.append({
                        "ts_code": ts_code,
                        "name": name_map.get(ts_code, ts_code),
                        "score": score,
                        "close": float(closes[-1]),
                        "avg_amount": avg_amount,
                        "pct_chg_5d": float(self._calc_momentum(closes, 5)) if len(closes) > 5 else 0,
                    })

            # 4b. 龙虎榜加分 (v3.1)
            lhb_data = market_data.get("lhb_data", {})
            lhb_boost_count = 0
            if cfg.enable_lhb_factor and lhb_data:
                for s in scores:
                    net_buy, count = self._calc_lhb_signal(
                        s["ts_code"], trade_date, lhb_data, cfg.lhb_lookback, trade_cal
                    )
                    if count > 0 and net_buy > 0:
                        boost = 1 + cfg.lhb_weight * min(count / 2, 1.0)
                        s["score"] *= boost
                        s["lhb_boost"] = boost
                        lhb_boost_count += 1
                    elif count > 0 and net_buy < 0:
                        penalty = 1 - cfg.lhb_weight * cfg.lhb_negative_penalty
                        s["score"] *= penalty
                        s["lhb_boost"] = penalty
                if lhb_boost_count > 0:
                    print(f"[信号] 龙虎榜加分: {lhb_boost_count} 只 (回看{cfg.lhb_lookback}天, w={cfg.lhb_weight})")

            scores.sort(key=lambda x: x["score"], reverse=True)
            print(f"[信号] 有效评分标的: {len(scores)} 只")

            # 前30名排行
            top_momentum = scores[:30]
            for i, s in enumerate(top_momentum):
                s["rank"] = i + 1

            # 5. 更新持仓价格
            latest_prices = {}
            for ts_code, df in stock_data.items():
                hist = df[df["trade_date"] <= trade_date]
                if not hist.empty:
                    latest_prices[ts_code] = float(hist.iloc[-1]["close"])

            self.portfolio.update_position_prices(latest_prices)
            total_value = self.portfolio.total_value
            cash = self.portfolio.cash

            # 6. 生成信号
            signals = []

            # 6a. 止损检查
            for code, pos in list(self.portfolio.positions.items()):
                peak = pos.get("peak_price", pos.get("cost_price", 0))
                current = latest_prices.get(code, pos.get("current_price", 0))
                if peak > 0 and current > 0:
                    drop = current / peak - 1
                    if drop <= cfg.stop_loss_pct:
                        signals.append(TradeSignal(
                            action="STOP_LOSS",
                            ts_code=code,
                            name=pos.get("name", code),
                            current_price=current,
                            target_weight=0,
                            target_shares=pos.get("shares", 0),
                            target_amount=pos.get("shares", 0) * current,
                            momentum_score=0,
                            rank=0,
                            reason=f"移动止损触发: 从峰值{peak:.2f}下跌{drop:.1%}",
                            urgency="HIGH",
                        ))

            # 6b. 市场状态仓位控制
            if regime == "HALT":
                max_position = 0.0
                regime_note = "市场熔断状态, 建议全部清仓"
            elif regime == "DEFENSIVE":
                max_position = 0.5 * cfg.max_total_position
                regime_note = "市场防御状态, 半仓运行"
            elif regime == "STRONG_RUN":
                max_position = cfg.max_total_position
                regime_note = "市场强势, 满仓运行"
            else:
                max_position = cfg.max_total_position
                regime_note = "市场正常"

            # 6b+. 消息层仓位信息展示 (不再折扣, 回测验证折扣拖累收益)
            info_notes = []
            if macro_assessment and macro_assessment.position_discount < 1.0:
                info_notes.append(f"宏观事件[{macro_assessment.risk_level}]")
            if sentiment_result and getattr(sentiment_result, 'position_adjustment', 1.0) < 1.0:
                info_notes.append(f"舆情偏空")
            if breaking_result and getattr(breaking_result, 'position_override', 1.0) < 1.0:
                info_notes.append(f"突发事件[{breaking_result.highest_level}]")
            if info_notes:
                regime_note += f" | 消息层提示: {', '.join(info_notes)} (仅展示)"

            # 宏观事件: 仅信息展示, 不阻断调仓
            # 回测验证: 静默期跳过调仓导致年化从41.71%暴跌至17.44%
            # 动量轮动的alpha来源是定期强制刷新, 跳过=持有过期持仓=踏空
            macro_quiet = False
            if macro_assessment and macro_assessment.quiet_period and regime != "HALT":
                print("[信号] ℹ️ 宏观事件密集(仅提示, 不阻断调仓)")
            if breaking_result and breaking_result.should_halt_trading:
                print(f"[信号] 🚨 突发事件 [{breaking_result.highest_level}] (仅提示, 不阻断调仓)")

            # 6c. HALT状态 — 立即清仓, 不受调仓间隔约束
            if regime == "HALT":
                for code, pos in self.portfolio.positions.items():
                    already_stop = any(s.ts_code == code and s.action == "STOP_LOSS" for s in signals)
                    if not already_stop:
                        signals.append(TradeSignal(
                            action="SELL",
                            ts_code=code,
                            name=pos.get("name", code),
                            current_price=latest_prices.get(code, 0),
                            target_weight=0,
                            target_shares=pos.get("shares", 0),
                            target_amount=pos.get("shares", 0) * latest_prices.get(code, 0),
                            momentum_score=0,
                            rank=0,
                            reason=f"HALT状态清仓(无视调仓间隔)",
                            urgency="HIGH",
                        ))
                should_rebalance = False  # HALT时不需要再调仓买入
                if self.portfolio.positions:
                    self.portfolio.data["halt_liquidated"] = True
                print(f"[信号] HALT状态: 立即清仓, 无视调仓间隔")

            # 6d. 判断是否需要调仓 (非HALT时才检查间隔)
            if regime != "HALT":
                # HALT恢复检测: 空仓 + 上次信号是HALT清仓 → 立即允许建仓
                halt_recovery = (not self.portfolio.positions
                                 and self.portfolio.data.get("halt_liquidated", False))
                if halt_recovery:
                    should_rebalance = True
                    self.portfolio.data["halt_liquidated"] = False
                    print(f"[信号] HALT恢复: 空仓状态, 立即允许建仓")
                else:
                    should_rebalance = True
                    last_rb = self.portfolio.last_rebalance_date
                    if last_rb:
                        if last_rb in trade_cal:
                            last_idx = trade_cal.index(last_rb)
                            curr_idx = trade_cal.index(trade_date) if trade_date in trade_cal else len(trade_cal) - 1
                            days_since = curr_idx - last_idx
                            if days_since < cfg.rebalance_interval_days:
                                should_rebalance = False
                                print(f"[信号] 距上次调仓仅 {days_since} 天, 不满 {cfg.rebalance_interval_days} 天")

                # 宏观静默期已关闭 (回测验证: 阻断调仓拖累年化24%+)

            # 6e. 正常调仓信号
            if should_rebalance and regime != "HALT":
                target_codes = [s["ts_code"] for s in scores[:cfg.top_n]]
                buffer_codes = [s["ts_code"] for s in scores[:int(cfg.top_n * cfg.hold_buffer_ratio)]]

                # 先确定卖出: 不在缓冲带内的持仓
                for code, pos in list(self.portfolio.positions.items()):
                    already_signal = any(s.ts_code == code for s in signals)
                    if already_signal:
                        continue
                    if code not in buffer_codes:
                        signals.append(TradeSignal(
                            action="SELL",
                            ts_code=code,
                            name=pos.get("name", code),
                            current_price=latest_prices.get(code, 0),
                            target_weight=0,
                            target_shares=pos.get("shares", 0),
                            target_amount=pos.get("shares", 0) * latest_prices.get(code, 0),
                            momentum_score=0,
                            rank=buffer_codes.index(code) + 1 if code in buffer_codes else 999,
                            reason=f"排名跌出缓冲带(TOP {int(cfg.top_n * cfg.hold_buffer_ratio)})",
                            urgency="NORMAL",
                        ))
                    else:
                        # 在缓冲带内, 保持持有
                        rank = buffer_codes.index(code) + 1
                        signals.append(TradeSignal(
                            action="HOLD",
                            ts_code=code,
                            name=pos.get("name", code),
                            current_price=latest_prices.get(code, 0),
                            target_weight=cfg.max_single_weight,
                            target_shares=pos.get("shares", 0),
                            target_amount=pos.get("shares", 0) * latest_prices.get(code, 0),
                            momentum_score=next((s["score"] for s in scores if s["ts_code"] == code), 0),
                            rank=rank,
                            reason=f"缓冲带内保留 (排名#{rank})" if code not in target_codes else f"TOP{cfg.top_n} 持有 (排名#{rank})",
                            urgency="LOW",
                        ))

                # 买入: 在TOP N中但不在当前持仓的, 带行业分散约束
                current_codes = set(self.portfolio.positions.keys())
                # 计算卖出后可用资金的近似值
                sell_proceeds = sum(
                    s.target_amount * (1 - cfg.commission_rate - cfg.stamp_tax_rate)
                    for s in signals if s.action == "SELL"
                )
                available_cash = cash + sell_proceeds
                # 目标买入数量 = TOP N - 已持有且保留的
                hold_count = sum(1 for s in signals if s.action == "HOLD")
                buy_slots = cfg.top_n - hold_count

                # 行业计数器 (含已持有的)
                from collections import Counter
                industry_counter = Counter()

                # 从排名靠前的候选中按行业约束选择
                # 扩大扫描范围, 确保行业约束下仍能选满
                scan_range = min(len(scores), cfg.top_n * 3)

                for s in scores[:scan_range]:
                    code = s["ts_code"]
                    if code in current_codes:
                        continue
                    if buy_slots <= 0:
                        break

                    price = s["close"]
                    single_weight = cfg.max_single_weight

                    target_value = min(total_value * single_weight, available_cash * 0.95)
                    # 仓位上限
                    if (total_value - available_cash + target_value) / total_value > max_position:
                        target_value = max(0, total_value * max_position - (total_value - available_cash))

                    exec_price = price * (1 + cfg.slippage_pct)
                    shares = int(target_value / exec_price / 100) * 100

                    if shares >= 100 and shares * exec_price >= cfg.min_position_amount:
                        amount = shares * exec_price
                        rank_num = scores.index(s) + 1
                        reason = f"动量排名#{rank_num}, 得分{s['score']:.4f}"

                        signals.append(TradeSignal(
                            action="BUY",
                            ts_code=code,
                            name=s["name"],
                            current_price=price,
                            target_weight=amount / total_value if total_value > 0 else 0,
                            target_shares=shares,
                            target_amount=amount,
                            momentum_score=s["score"],
                            rank=rank_num,
                            reason=reason,
                            urgency="NORMAL",
                        ))
                        available_cash -= (amount + max(amount * cfg.commission_rate, 5))
                        buy_slots -= 1

            elif not should_rebalance:
                # 非调仓日, 只输出持仓状态
                for code, pos in self.portfolio.positions.items():
                    already = any(s.ts_code == code for s in signals)
                    if not already:
                        rank_info = next((s for s in scores if s["ts_code"] == code), None)
                        signals.append(TradeSignal(
                            action="HOLD",
                            ts_code=code,
                            name=pos.get("name", code),
                            current_price=latest_prices.get(code, 0),
                            target_weight=cfg.max_single_weight,
                            target_shares=pos.get("shares", 0),
                            target_amount=pos.get("shares", 0) * latest_prices.get(code, 0),
                            momentum_score=rank_info["score"] if rank_info else 0,
                            rank=rank_info["rank"] if rank_info else 999,
                            reason="非调仓日, 继续持有",
                            urgency="LOW",
                        ))

            # 7. 组装结果
            current_holdings = []
            for code, pos in self.portfolio.positions.items():
                price = latest_prices.get(code, pos.get("current_price", 0))
                cost = pos.get("cost_price", 0)
                pnl = (price / cost - 1) if cost > 0 else 0
                current_holdings.append({
                    "ts_code": code,
                    "name": pos.get("name", code),
                    "shares": pos.get("shares", 0),
                    "cost_price": round(cost, 2),
                    "current_price": round(price, 2),
                    "pnl_pct": round(pnl * 100, 2),
                    "market_value": round(pos.get("shares", 0) * price, 2),
                })

            # 8. 生成摘要文本
            summary = self._format_summary(
                trade_date, regime, regime_note, total_value, cash,
                signals, top_momentum, current_holdings, should_rebalance,
                macro_assessment=macro_assessment,
                sentiment_result=sentiment_result,
                breaking_result=breaking_result,
            )

            result = SignalResult(
                success=True,
                generated_at=now,
                trade_date=trade_date,
                market_regime=regime,
                total_value=round(total_value, 2),
                cash=round(cash, 2),
                signals=signals,
                top_momentum=top_momentum,
                current_holdings=current_holdings,
                summary_text=summary,
                macro_risk=macro_assessment.to_dict() if macro_assessment else {},
                sentiment=sentiment_result.to_dict() if sentiment_result else {},
                breaking_alerts=breaking_result.to_dict() if breaking_result and breaking_result.alerts else {},
            )

            # 保存信号到文件
            signal_dir = PROJECT_ROOT / "data" / "signals"
            signal_dir.mkdir(parents=True, exist_ok=True)
            result.to_json(str(signal_dir / f"signal_{trade_date}.json"))
            print(f"\n[信号] 已保存至 {signal_dir / f'signal_{trade_date}.json'}")

            return result

        except Exception as e:
            traceback.print_exc()
            return SignalResult(
                success=False,
                generated_at=now,
                trade_date=as_of_date or "",
                market_regime="UNKNOWN",
                total_value=0,
                cash=0,
                error_message=str(e),
            )

    # ── 格式化输出 ──

    def _format_summary(self, trade_date, regime, regime_note,
                        total_value, cash, signals, top_momentum,
                        current_holdings, is_rebalance_day,
                        macro_assessment=None, sentiment_result=None,
                        breaking_result=None) -> str:
        """生成人类可读的信号摘要"""
        lines = []
        lines.append("=" * 60)
        lines.append(f"  动量轮动策略 · 每日信号报告")
        lines.append(f"  日期: {trade_date}  {'[调仓日]' if is_rebalance_day else '[持仓监控日]'}")
        lines.append("=" * 60)
        lines.append("")

        # 突发事件告警 (最高优先级, 置顶)
        if breaking_result and breaking_result.alerts:
            lines.append(breaking_result.emergency_text)
            lines.append("")

        # 市场状态
        regime_emoji = {"HALT": "🔴", "DEFENSIVE": "🟡", "RUN": "🟢", "STRONG_RUN": "🟢🟢"}.get(regime, "⚪")
        lines.append(f"市场状态: {regime_emoji} {regime} — {regime_note}")
        # 显示自适应权重
        active_w = self._get_regime_weights(regime)
        weight_labels = [f"{d}d={w:.0%}" for d, w in zip(self.config.lookback_days, active_w)]
        adaptive_tag = "自适应" if self.config.adaptive_weights else "固定"
        lines.append(f"动量权重: [{', '.join(weight_labels)}] ({adaptive_tag})")
        lines.append(f"账户总值: ¥{total_value:,.2f}  |  现金: ¥{cash:,.2f}")
        lines.append("")

        # 消息层: 宏观事件 + 舆情
        has_msg_layer = False
        if macro_assessment and macro_assessment.alert_text:
            lines.append(macro_assessment.alert_text)
            has_msg_layer = True
        if sentiment_result and sentiment_result.alert_text:
            lines.append(sentiment_result.alert_text)
            has_msg_layer = True
        if has_msg_layer:
            lines.append("")

        # 止损警报
        stop_signals = [s for s in signals if s.action == "STOP_LOSS"]
        if stop_signals:
            lines.append("⚠️ 止损警报:")
            for s in stop_signals:
                lines.append(f"  🔴 {s.name}({s.ts_code}) ¥{s.current_price:.2f} — {s.reason}")
            lines.append("")

        # 卖出信号
        sell_signals = [s for s in signals if s.action == "SELL"]
        if sell_signals:
            lines.append("📤 卖出信号:")
            for s in sell_signals:
                lines.append(f"  {s.name}({s.ts_code}) {s.target_shares}股 × ¥{s.current_price:.2f} = ¥{s.target_amount:,.0f}  [{s.reason}]")
            lines.append("")

        # 买入信号 (仅调仓日展示)
        buy_signals = [s for s in signals if s.action == "BUY"]
        if buy_signals:
            if is_rebalance_day:
                lines.append("📥 买入信号:")
                for s in buy_signals:
                    lines.append(f"  {s.name}({s.ts_code}) {s.target_shares}股 × ¥{s.current_price:.2f} ≈ ¥{s.target_amount:,.0f}  [{s.reason}]")
            else:
                lines.append("📋 候选观察 (非调仓日, 仅供参考, 不执行):")
                for s in buy_signals:
                    lines.append(f"  {s.name}({s.ts_code}) 得分{s.momentum_score:+.4f} ¥{s.current_price:.2f}  [{s.reason}]")
            lines.append("")

        # 持有
        hold_signals = [s for s in signals if s.action == "HOLD"]
        if hold_signals:
            lines.append("📋 继续持有:")
            for s in hold_signals:
                lines.append(f"  {s.name}({s.ts_code}) {s.target_shares}股 ¥{s.current_price:.2f}  [{s.reason}]")
            lines.append("")

        # 当前持仓汇总
        if current_holdings:
            lines.append("当前持仓:")
            total_mv = 0
            for h in current_holdings:
                pnl_icon = "📈" if h["pnl_pct"] >= 0 else "📉"
                lines.append(f"  {h['name']}  {h['shares']}股  成本¥{h['cost_price']:.2f}  现价¥{h['current_price']:.2f}  {pnl_icon}{h['pnl_pct']:+.2f}%  市值¥{h['market_value']:,.0f}")
                total_mv += h["market_value"]
            lines.append(f"  持仓合计: ¥{total_mv:,.0f}  仓位: {total_mv/total_value*100:.1f}%")
            lines.append("")

        # 动量排行 TOP 10
        lines.append("动量排行 TOP 10:")
        for s in top_momentum[:10]:
            in_portfolio = "⭐" if any(h["ts_code"] == s["ts_code"] for h in current_holdings) else "  "
            lines.append(f"  {s['rank']:>2}. {in_portfolio} {s['name']:<8s} {s['ts_code']}  得分:{s['score']:+.4f}  价格:¥{s['close']:.2f}  5日:{s['pct_chg_5d']:+.1%}")
        lines.append("")

        # 操作摘要
        n_buy = len(buy_signals)
        n_sell = len(sell_signals)
        n_stop = len(stop_signals)
        n_hold = len(hold_signals)

        if n_buy + n_sell + n_stop == 0:
            lines.append("今日无操作, 继续持有。")
        else:
            actions = []
            if n_stop: actions.append(f"止损{n_stop}只")
            if n_sell: actions.append(f"卖出{n_sell}只")
            if n_buy: actions.append(f"买入{n_buy}只")
            if n_hold: actions.append(f"持有{n_hold}只")
            lines.append(f"操作汇总: {', '.join(actions)}")

            total_buy = sum(s.target_amount for s in buy_signals)
            total_sell = sum(s.target_amount for s in sell_signals) + sum(s.target_amount for s in stop_signals)
            if total_buy > 0:
                lines.append(f"预估买入金额: ¥{total_buy:,.0f}")
            if total_sell > 0:
                lines.append(f"预估卖出金额: ¥{total_sell:,.0f}")

        lines.append("")
        lines.append("=" * 60)
        lines.append("⚠️ 以上为策略信号, 请人工确认后在广发易淘金APP执行下单")
        lines.append("=" * 60)

        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI Entry
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="动量轮动信号生成器")
    parser.add_argument("--token", type=str, help="Tushare API token")
    parser.add_argument("--csv", type=str, help="CSV数据文件路径 (离线模式)")
    parser.add_argument("--date", type=str, help="指定分析日期 YYYYMMDD")
    parser.add_argument("--portfolio", type=str, help="持仓状态JSON路径")
    args = parser.parse_args()

    config = SignalConfig()
    if args.token:
        config.tushare_token = args.token
    if args.csv:
        config.data_csv_path = args.csv
    if args.portfolio:
        config.portfolio_path = args.portfolio

    gen = SignalGenerator(config)
    result = gen.generate(as_of_date=args.date)

    if result.success:
        print(result.summary_text)
    else:
        print(f"[错误] 信号生成失败: {result.error_message}")
