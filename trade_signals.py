"""
trade_signals.py — 5类买卖信号自动生成
========================================
原始产出：阶段④ 自动化交易信号生成

5类信号：
  1. 阶梯突破 (ladder_breakout)  — 基于 daily_signal v6.0
  2. 龙头突破 (leader_breakout)  — the me leader 策略
  3. CANSLIM 突破 (canslim)      — C/A/N/S/L/I/M 七因子
  4. 板块轮动 (sector_rotation)  — 板块强弱轮动切换
  5. 价值低估 (value_undervalued) — 价值投资信号

每类信号输出统一格式：
  {
    "type": "ladder_breakout",
    "symbol": "600219.SH",
    "name": "南山铝业",
    "score": 82,
    "action": "BUY",
    "price": {"close": 5.31, "breakout": 5.14, "stop_loss": 4.41},
    "confidence": 0.78,
    "reasons": ["阶梯整理30天", "冲量45%", "板块强共振"],
    "risk_notes": [],
    "auxiliary_info": {...},  # v6.0 辅助信息
  }

与新架构集成：
  - 注册到 ToolCenter，Router 可自动路由
  - 信号结果写入 EventLog
  - Brain.commit_decision() 记录决策
"""

import time
import os
import json
import signal as _signal
import threading
from datetime import datetime, timedelta
from typing import Optional, Any

import numpy as np
import pandas as pd


class _Timeout(Exception):
    """单次股票数据拉取超时"""
    pass


def _timeout_handler(signum, frame):
    raise _Timeout("stock data fetch timeout")


# ═══════════════════════════════════════════
# 统一风控参数（对齐 trade_signal.py Skill 8）
# 所有策略共享此标准，各策略不再自行硬编码
# ═══════════════════════════════════════════
UNIFIED_PARAMS = {
    # 止损/目标
    "stop_atr_multiple": 2.0,
    "target_atr_multiple": 6.0,
    "max_stop_pct": 0.10,
    "max_target_pct": 0.30,
    # 盈亏比
    "min_rr_ratio": 2.0,
    # 量能
    "min_vol_ratio": 1.2,
    "breakout_vol_ratio": 1.5,
    # 乖离率（不再淘汰，改为入场阶段判定）
    "bias_immediate_buy": 2.0,
    "bias_watch_zone": 8.0,
    "bias_extreme": 15.0,
    # 回踩入场
    "pullback_valid_days": 5,
    "pullback_tolerance": 0.01,
    "stop_below_breakout": 0.03,
    # 环境联动
    "env_no_buy": 60,
    "env_reduce_position": 70,
    # 仓位管理
    "risk_per_trade": 0.01,
    "max_single_position": 0.08,
    "default_capital": 1_000_000,
    # 评分
    "min_score": 65,
    # ATR
    "atr_period": 14,
    # 安全检查（最高优先级）
    "safety_max_daily_drop": -5.0,
    "safety_max_daily_rise": 15.0,
    "safety_limit_reject": True,
    "safety_3d_max_drop": -10.0,
    "safety_vol_spike_threshold": 5.0,
    "safety_require_yang": True,
}
UP = UNIFIED_PARAMS
OBSERVE_SIGNALS_FILE = "observe_signals.json"


def _validate_param_alignment():
    """启动时检查与 Skill 8 (trade_signal.py) 关键参数是否对齐"""
    try:
        from trade_signal import DEFAULT_PARAMS as SKILL8
        drifts = []
        checks = [
            ("stop_atr_multiple", "stop_atr_multiple"),
            ("risk_per_trade", "risk_per_trade"),
            ("max_single_position", "max_single_position"),
        ]
        for up_key, s8_key in checks:
            up_val = UP.get(up_key)
            s8_val = SKILL8.get(s8_key)
            if up_val != s8_val:
                drifts.append(f"  {up_key}: unified={up_val}, skill8={s8_val}")
        if drifts:
            print("⚠️ 参数漂移检测 (trade_signals vs trade_signal):")
            for d in drifts:
                print(d)
    except Exception:
        pass


class TradeSignalGenerator:
    """
    交易信号生成器

    整合 5 类信号源，统一输出格式。
    """

    def __init__(self, data_fetcher=None, market_env=None,
                 event_log=None, brain=None, data_cache=None):
        """
        Parameters
        ----------
        data_fetcher : DataFetcher 实例
        market_env : MarketEnvironment 实例（v6.0 市场环境评估）
        event_log : EventLog 实例
        brain : AgentBrain 实例
        """
        self.fetcher = data_fetcher
        self.market_env = market_env
        self.event_log = event_log
        self.brain = brain
        self.data_cache = data_cache
        self._dynamic_pool = None
        self._pool_result = None
        self._progress_step = int(os.getenv("SIGNAL_PROGRESS_STEP", "50") or 50)
        self._fetch_timeout_sec = int(os.getenv("SIGNAL_FETCH_TIMEOUT_SEC", "15") or 15)
        self._min_rr_ratio = float(os.getenv("SIGNAL_MIN_RR_RATIO", "2.0") or 2.0)
        self._data_cache = {}   # key: (symbol, days) -> DataFrame|None
        self._name_map = None
        self._fetch_method = None
        self._cache_hits = 0
        self._cache_misses = 0
        self._network_fetches = 0

    # ═══════════════════════════════════════════
    # 主入口
    # ═══════════════════════════════════════════

    def scan_all(self, date: str = None, stock_pool: list = None) -> dict:
        """
        全量扫描，生成所有类型的信号

        Returns
        -------
        {
            "date": "20260224",
            "market_env": {...},          # 市场环境评估
            "signals": [...],             # 所有信号列表
            "by_type": {                  # 按类型分组
                "ladder_breakout": [...],
                "leader_breakout": [...],
                ...
            },
            "summary": "...",             # 文字摘要
        }
        """
        date = date or datetime.now().strftime("%Y%m%d")
        self._data_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self._network_fetches = 0
        self._ensure_name_map()

        # 1. 市场环境评估
        env_result = None
        env_score = 65
        if self.market_env:
            try:
                env_result = self.market_env.evaluate(date)
                env_score = int(env_result.get("total_score", 65) or 65)
            except Exception as e:
                env_result = {
                    "total_score": 65,
                    "level": "一般",
                    "advice": "谨慎交易",
                    "error": str(e),
                }

        # 2. 动态股票池（未外部传入时）
        if stock_pool:
            self._dynamic_pool = stock_pool
            self._pool_result = {
                "pool": list(stock_pool),
                "pool_size": len(stock_pool),
                "market_breadth": "外部传入",
                "filter_funnel": {},
            }
            print(f"  📊 使用外部传入池: {len(stock_pool)}只")
        elif self.fetcher or self.data_cache:
            try:
                from dynamic_pool import DynamicPool
                pool_builder = DynamicPool(data_fetcher=self.fetcher, data_cache=self.data_cache)
                self._pool_result = pool_builder.build(date, verbose=True)
                self._dynamic_pool = self._pool_result.get("pool", [])
                print(
                    f"  📊 动态池: {self._pool_result.get('pool_size', len(self._dynamic_pool))}只 "
                    f"({self._pool_result.get('market_breadth', '未知')})"
                )
                breadth = str(self._pool_result.get("market_breadth", ""))
                pool_size = int(self._pool_result.get("pool_size", len(self._dynamic_pool)) or 0)
                if ("偏弱" in breadth or "弱势" in breadth) and env_score >= 60:
                    print(f"  ⚠️ 池子仅{pool_size}只（{breadth}），建议降低仓位")
            except Exception as e:
                print(f"  ⚠️ 动态池构建失败，回退默认池: {e}")
                self._dynamic_pool = None
                self._pool_result = None
        else:
            self._dynamic_pool = None
            self._pool_result = None

        # 3. 预加载
        pool = self._dynamic_pool or self._get_stock_pool()
        dynamic_limit = int(os.getenv("SIGNAL_DYNAMIC_POOL_LIMIT", "500") or 500)
        if dynamic_limit > 0 and len(pool) > dynamic_limit:
            print(f"  ⚙️ 动态池限流: {len(pool)} -> {dynamic_limit}")
            pool = pool[:dynamic_limit]
            self._dynamic_pool = pool
        self._prefetch_pool_data(pool, date, days=90)

        # 4. 扫描各策略（入口条件不变）
        all_raw = []
        by_type = {}
        for scan_name, scan_func in [
            ("ladder_breakout", self._scan_ladder_breakout),
            ("leader_breakout", self._scan_leader_breakout),
            ("canslim", self._scan_canslim),
            ("sector_rotation", self._scan_sector_rotation),
            ("sector_stage", self._scan_sector_stage),
        ]:
            try:
                t0 = time.time()
                print(f"  ▶ 扫描 {scan_name} ...")
                raw = scan_func(date, pool)
                dt = time.time() - t0
                print(f"  ✅ {scan_name}: {len(raw)} 条原始 ({dt:.1f}s)")
                by_type[scan_name] = raw
                all_raw.extend(raw)
            except Exception as e:
                by_type[scan_name] = []
                self._emit_error(scan_name, e)
                print(f"  ❌ {scan_name}: {e}")

        # 5. 统一过滤（突破发现 + 入场分类）
        print(f"  🔍 统一过滤: {len(all_raw)} 条原始 → ", end="")
        new_signals = self._apply_unified_filter(all_raw, date, env_score)
        buy_count = sum(1 for s in new_signals if s.get("signal_stage") == "buy")
        observe_count = sum(1 for s in new_signals if s.get("signal_stage") == "observe")
        print(f"{len(new_signals)} 条通过 (🟢买入:{buy_count} 🔔观察:{observe_count})")

        # 6. 检查历史观察信号回踩
        old_observe = self.load_observe_signals()
        pullback_result = {"triggered": [], "watching": [], "expired": []}
        if old_observe:
            print(f"  🔄 检查 {len(old_observe)} 条历史观察信号回踩情况...")
            pullback_result = self.check_pullback_entries(old_observe, date)
            t = len(pullback_result["triggered"])
            w = len(pullback_result["watching"])
            e = len(pullback_result["expired"])
            print(f"  ✅ 回踩触发:{t} 继续观察:{w} 已过期:{e}")

        # 7. 合并买入/观察
        buy_signals = [s for s in new_signals if s.get("signal_stage") == "buy"]
        buy_signals.extend(pullback_result.get("triggered", []))
        observe_signals = [s for s in new_signals if s.get("signal_stage") == "observe"]
        observe_signals.extend(pullback_result.get("watching", []))

        # 8. 保存观察信号（去重：同symbol保留评分更高）
        dedup = {}
        for sig in observe_signals:
            sym = sig.get("symbol")
            if not sym:
                continue
            prev = dedup.get(sym)
            if prev is None or float(sig.get("score", 0) or 0) > float(prev.get("score", 0) or 0):
                dedup[sym] = sig
        observe_signals = list(dedup.values())
        self.save_observe_signals(observe_signals)

        # 9. 合并输出
        all_signals = buy_signals + observe_signals
        all_signals.sort(key=lambda s: (
            0 if s.get("signal_stage") == "buy" else 1,
            -float(s.get("entry_rr_ratio", 0) or 0),
            -float(s.get("score", 0) or 0),
        ))

        # 10. 摘要 + 事件
        summary = self._build_summary_v2(
            date, env_result, buy_signals, observe_signals, pullback_result
        )
        print(
            f"  📊 数据缓存统计: hit={self._cache_hits} miss={self._cache_misses} "
            f"net={self._network_fetches} cache={len(self._data_cache)}"
        )
        self._emit_signals(date, all_signals, env_result)

        return {
            "date": date,
            "market_env": env_result,
            "signals": all_signals,
            "buy_signals": buy_signals,
            "observe_signals": observe_signals,
            "pullback_result": pullback_result,
            "by_type": by_type,
            "summary": summary,
            "pool_info": {
                "pool_size": len(self._dynamic_pool) if self._dynamic_pool else 0,
                "market_breadth": (
                    self._pool_result.get("market_breadth", "未知")
                    if self._pool_result else "固定池"
                ),
                "filter_funnel": (
                    self._pool_result.get("filter_funnel", {})
                    if self._pool_result else {}
                ),
            },
        }

    def _calculate_bias(self, close_prices) -> float:
        """
        计算乖离率: (close - MA20) / MA20 × 100
        """
        try:
            arr = np.array(close_prices, dtype=float)
            if arr.size == 0:
                return 0.0
            current = float(arr[-1])
            ma20 = float(np.mean(arr[-20:])) if arr.size >= 20 else float(np.mean(arr))
            if ma20 <= 0:
                return 0.0
            return round((current - ma20) / ma20 * 100, 2)
        except Exception:
            return 0.0

    def scan_type(self, signal_type: str, date: str = None,
                  stock_pool: list = None) -> list:
        """扫描指定类型的信号"""
        date = date or datetime.now().strftime("%Y%m%d")
        func_map = {
            "ladder_breakout": self._scan_ladder_breakout,
            "leader_breakout": self._scan_leader_breakout,
            "canslim": self._scan_canslim,
            "sector_rotation": self._scan_sector_rotation,
            "sector_stage": self._scan_sector_stage,
            "value_undervalued": self._scan_value_undervalued,
        }
        func = func_map.get(signal_type)
        if not func:
            return []
        return func(date, stock_pool)

    # ═══════════════════════════════════════════
    # 信号类型 1: 阶梯突破 (daily_signal v6.0)
    # ═══════════════════════════════════════════

    def _scan_ladder_breakout(self, date: str, pool: list = None) -> list:
        """
        阶梯突破信号 — v6.0 最终版

        核心逻辑（v4.1 不变）：
        1. 阶梯形态识别：整理期 ≥ 15天，整理幅度 < 15%
        2. 突破确认：收盘价突破整理区间上沿
        3. 量能配合：突破日量比 ≥ 1.2
        4. 评分 ≥ 65 入选

        v6.0 新增（仅辅助信息，不影响选股）：
        - 60天新高判断
        - 形态质量评级
        - 量比数值
        - 板块共振
        - 风险提示
        """
        signals = []
        stocks = pool or self._get_stock_pool()
        canslim_limit = int(os.getenv("SIGNAL_CANSLIM_POOL_LIMIT", "120") or 120)
        if canslim_limit > 0 and len(stocks) > canslim_limit:
            stocks = stocks[:canslim_limit]

        total = len(stocks)
        t0 = time.time()
        for idx, symbol in enumerate(stocks, 1):
            try:
                # 前置安全检查：大跌股直接跳过，避免无意义形态计算
                df_quick = self._get_stock_data(symbol, date, days=5)
                if df_quick is not None and len(df_quick) >= 2:
                    quick_pct = (float(df_quick["close"].iloc[-1]) / float(df_quick["close"].iloc[-2]) - 1) * 100
                    if quick_pct < -3:
                        continue
                sig = self._detect_ladder_breakout(symbol, date)
                if sig:
                    # v6.0 辅助信息
                    sig["auxiliary_info"] = self._get_auxiliary_info(
                        symbol, date, df_cache=self._data_cache.get((symbol, 90))
                    )
                    signals.append(sig)
            except Exception:
                continue
            if self._progress_step > 0 and (idx % self._progress_step == 0 or idx == total):
                elapsed = time.time() - t0
                print(f"    ...ladder {idx}/{total} 已扫, 命中{len(signals)}")

        return sorted(signals, key=lambda s: -s["score"])

    def _detect_ladder_breakout(self, symbol: str, date: str) -> Optional[dict]:
        """
        检测单只股票的阶梯突破信号

        阶梯形态定义：
        - 股价在一个水平通道内整理（振幅 < 15%）
        - 整理天数 ≥ 15
        - 整理前有一段上涨（形成"阶梯"）
        - 突破整理区间上沿，量能放大
        """
        if not self.fetcher:
            return None

        df = self._get_stock_data(symbol, date, days=90)
        if df is None or len(df) < 30:
            return None

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        volume = df["vol"].values if "vol" in df.columns else df.get("volume", [0]*len(df))

        current = close[-1]

        # 寻找整理区间
        consolidation = self._find_consolidation(close, high, low)
        if not consolidation:
            return None

        c_start, c_end, c_high, c_low, c_days = consolidation

        # 突破确认
        if current <= c_high:
            return None  # 未突破

        # 突破日必须收阳，且不能是明显下跌日
        if UP.get("safety_require_yang", True) and "open" in df.columns:
            today_open = float(df["open"].iloc[-1] or 0)
            if today_open > 0 and current < today_open:
                return None
        if len(close) >= 2:
            prev_close = float(close[-2] or 0)
            day_pct = (current / prev_close - 1) * 100 if prev_close > 0 else 0
            if day_pct < -2:
                return None

        # 量能计算（保留，不在入口硬过滤）
        if len(volume) >= 20:
            avg_vol = sum(volume[-20:]) / 20
            vol_ratio = volume[-1] / avg_vol if avg_vol > 0 else 1.0
        else:
            vol_ratio = 1.0

        # 评分
        score = 50
        score += min(15, c_days)                    # 整理天数 +1/天，上限15
        score += min(10, int(vol_ratio * 5))        # 量比贡献
        amplitude = (c_high - c_low) / c_low * 100 if c_low > 0 else 99
        if amplitude < 8:
            score += 10  # 整理越窄越好
        elif amplitude < 12:
            score += 5

        # 突破幅度
        breakout_pct = (current - c_high) / c_high * 100
        if 1 < breakout_pct < 5:
            score += 5  # 温和突破

        bias = self._calculate_bias(close)
        score = max(0, min(100, score))

        name = self._get_stock_name(symbol)

        return {
            "type": "ladder_breakout",
            "symbol": symbol,
            "name": name,
            "score": score,
            "action": "BUY",
            "price": {
                "close": round(current, 2),
                "breakout": round(c_high, 2),
            },
            "bias": round(bias, 2),
            "confidence": round(score / 100, 2),
            "reasons": [
                f"阶梯整理{c_days}天",
                f"振幅{amplitude:.1f}%",
                f"量比{vol_ratio:.2f}",
                f"突破{breakout_pct:.1f}%",
            ],
            "risk_notes": self._get_risk_notes(symbol, current, vol_ratio, bias=bias),
        }

    def _find_consolidation(self, close, high, low, min_days=15, max_amp=15):
        """
        寻找最近的整理区间

        从最新数据往前找，找到一个振幅 < max_amp% 且持续 ≥ min_days 的区间
        """
        n = len(close)
        if n < min_days + 5:
            return None

        # 从倒数第2天往前找（倒数第1天是今天的突破日）
        for end in range(n - 2, min_days, -1):
            for start in range(end - min_days, max(0, end - 60), -1):
                segment_high = max(high[start:end + 1])
                segment_low = min(low[start:end + 1])
                if segment_low <= 0:
                    continue
                amp = (segment_high - segment_low) / segment_low * 100
                if amp < max_amp:
                    days = end - start + 1
                    if days >= min_days:
                        return (start, end, segment_high, segment_low, days)

        return None

    # ═══════════════════════════════════════════
    # 信号类型 2: 龙头突破 (the me leader)
    # ═══════════════════════════════════════════

    def _scan_leader_breakout(self, date: str, pool: list = None) -> list:
        """
        龙头突破信号 — The Me Leader 策略

        核心逻辑：
        1. 板块龙头识别：板块内涨幅最大 + 市值前列 + 换手率活跃
        2. 龙头形态：经历调整后再度突破（二次启动）
        3. 板块联动：板块整体走强时龙头突破更有效

        龙头定义：
        - 近 20 日板块内涨幅 Top3
        - 流通市值板块内 Top30%
        - 近 5 日平均换手率 > 板块中位数
        """
        signals = []
        sectors = self._get_active_sectors(date)
        # Leader 扫描使用板块成分股，可能不在主池内，先批量预热减少逐只阻塞
        leader_pool = sorted({c for stocks in sectors.values() for c in (stocks or [])})
        self._prefetch_pool_data(leader_pool, date, days=60)
        hit0, miss0 = self._cache_hits, self._cache_misses

        sector_items = list(sectors.items())
        total = len(sector_items)
        for idx, (sector_name, sector_stocks) in enumerate(sector_items, 1):
            try:
                leaders = self._identify_leaders(sector_stocks, date)
                for leader in leaders:
                    sig = self._detect_leader_breakout(leader, sector_name, date)
                    if sig:
                        signals.append(sig)
            except Exception:
                continue
            if self._progress_step > 0 and (idx % max(1, self._progress_step // 10) == 0 or idx == total):
                print(f"    ...leader 板块 {idx}/{total}, 命中{len(signals)}")

        dh = self._cache_hits - hit0
        dm = self._cache_misses - miss0
        total_lookup = dh + dm
        if total_lookup > 0:
            print(f"    ...leader 缓存命中率 {dh}/{total_lookup} ({dh/total_lookup:.1%})")

        return sorted(signals, key=lambda s: -s["score"])

    def _identify_leaders(self, stocks: list, date: str) -> list:
        """识别板块龙头（涨幅+市值+换手率）"""
        scored = []
        for symbol in stocks:
            try:
                # 识别龙头阶段优先用缓存，避免大量慢请求
                df = self._get_stock_data(symbol, date, days=20, allow_network=False)
                if df is None or len(df) < 10:
                    continue
                pct_20d = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
                avg_turnover = df["turnover_rate"].mean() if "turnover_rate" in df.columns else 0
                scored.append({
                    "symbol": symbol,
                    "pct_20d": pct_20d,
                    "turnover": avg_turnover,
                })
            except Exception:
                continue

        # Top5 by 涨幅
        scored.sort(key=lambda x: -x["pct_20d"])
        return [s["symbol"] for s in scored[:5]]

    def _detect_leader_breakout(self, symbol: str, sector: str, date: str) -> Optional[dict]:
        """检测龙头突破信号"""
        df = self._get_stock_data(symbol, date, days=60)
        if df is None or len(df) < 30:
            return None

        close = df["close"].values
        high = df["high"].values
        current = close[-1]

        # 当日大跌/收阴直接淘汰
        if len(close) >= 2:
            prev_close = float(close[-2] or 0)
            day_pct = (current / prev_close - 1) * 100 if prev_close > 0 else 0
            if day_pct < -3:
                return None
        if UP.get("safety_require_yang", True) and "open" in df.columns:
            today_open = float(df["open"].iloc[-1] or 0)
            if today_open > 0 and current < today_open:
                return None

        # 近 60 日最高价
        high_60d = max(high)
        near_high = current >= high_60d * 0.97

        # 近 20 日有过回调（至少 -5%）后反弹
        min_20d = min(close[-20:])
        pullback = (max(close[-20:]) - min_20d) / max(close[-20:]) * 100 if max(close[-20:]) > 0 else 0
        has_pullback = pullback > 5

        if not (near_high and has_pullback):
            return None

        # 评分
        score = 60
        vol_ratio = 1.0
        if current >= high_60d:
            score += 10  # 创新高
        if pullback > 8:
            score += 5   # 充分调整
        # 量能
        if "vol" in df.columns and len(df) >= 20:
            vol_ratio = df["vol"].iloc[-1] / df["vol"].iloc[-20:].mean()
            if vol_ratio > 1.5:
                score += 10

        bias = self._calculate_bias(close)
        score = max(0, min(100, score))

        name = self._get_stock_name(symbol)

        return {
            "type": "leader_breakout",
            "symbol": symbol,
            "name": name,
            "score": max(0, min(100, score)),
            "action": "BUY",
            "price": {
                "close": round(current, 2),
                "high_60d": round(high_60d, 2),
            },
            "bias": round(bias, 2),
            "confidence": round(score / 100, 2),
            "reasons": [
                f"板块龙头({sector})",
                f"回调{pullback:.1f}%后突破",
                "创60日新高" if current >= high_60d else "接近60日新高",
            ],
            "risk_notes": self._get_risk_notes(symbol, current, vol_ratio, bias=bias),
        }

    # ═══════════════════════════════════════════
    # 信号类型 3: CANSLIM
    # ═══════════════════════════════════════════

    def _scan_canslim(self, date: str, pool: list = None) -> list:
        """
        CANSLIM 信号

        委托给已有的 CANSLIM Skill（如果存在）
        这里提供接口桥接
        """
        try:
            from skills.canslim_skill import CANSLIMSkill
            skill = CANSLIMSkill(self.fetcher)
            return self._apply_bias_guard_signals(skill.scan(date), date)
        except ImportError:
            print("  ↪ canslim_skill 不可用，使用内嵌 fallback")

        # fallback: 内嵌 CANSLIM（基本面优先走缓存）
        signals = []
        stocks = pool or self._get_stock_pool()
        canslim_pass_fundamental = 0
        for symbol in stocks:
            try:
                fund = self.data_cache.get_fundamental(symbol, date) if self.data_cache else {}
                if not fund:
                    continue

                profit_yoy = float(fund.get("profit_yoy", 0) or 0)
                roe = float(fund.get("roe", 0) or 0)
                circ_mv = float(fund.get("circ_mv", 0) or 0)
                if profit_yoy < 20 or roe < 15 or circ_mv < 1_000_000:
                    continue
                canslim_pass_fundamental += 1

                df = self._get_stock_data(symbol, date, days=60)
                if df is None or len(df) < 30:
                    continue

                close = df["close"].values
                high = df["high"].values
                current = float(close[-1])
                if len(close) >= 2:
                    prev_close = float(close[-2] or 0)
                    day_pct = (current / prev_close - 1) * 100 if prev_close > 0 else 0
                    if day_pct < -3:
                        continue
                high_60d = float(np.max(high))
                is_near_high = current >= high_60d * 0.95
                if not is_near_high:
                    continue

                turnover = float(fund.get("turnover_rate", 0) or 0)
                if turnover < 0.5:
                    continue

                vol = df["vol"].values if "vol" in df.columns else None
                vol_ratio = (
                    float(vol[-1] / np.mean(vol[-20:]))
                    if vol is not None and len(vol) >= 20 and np.mean(vol[-20:]) > 0 else 1.0
                )
                bias = self._calculate_bias(close)

                score = 55
                if profit_yoy > 50:
                    score += 10
                elif profit_yoy > 30:
                    score += 5
                if roe > 25:
                    score += 10
                elif roe > 20:
                    score += 5
                if current >= high_60d:
                    score += 10
                if vol_ratio > 1.5:
                    score += 5
                score = max(0, min(100, score))

                name = self._get_stock_name(symbol)
                signals.append({
                    "type": "canslim",
                    "symbol": symbol,
                    "name": name,
                    "score": score,
                    "action": "BUY",
                    "price": {
                        "close": round(current, 2),
                        "breakout": round(high_60d, 2),
                    },
                    "bias": round(bias, 2),
                    "confidence": round(score / 100, 2),
                    "reasons": [
                        f"利润+{profit_yoy:.0f}% ROE={roe:.1f}%",
                        f"流通市值{circ_mv/10000:.0f}亿",
                        "创60日新高" if current >= high_60d else f"距新高{(high_60d/current-1)*100:.1f}%",
                        f"量比{vol_ratio:.2f}",
                    ],
                    "risk_notes": self._get_risk_notes(symbol, current, vol_ratio, bias=bias),
                })
            except Exception:
                continue
        print(f"    CANSLIM: 基本面通过{canslim_pass_fundamental}, 最终{len(signals)}条")
        return sorted(signals, key=lambda s: -float(s["score"]))

    # ═══════════════════════════════════════════
    # 信号类型 4: 板块轮动
    # ═══════════════════════════════════════════

    def _scan_sector_rotation(self, date: str, pool: list = None) -> list:
        """
        板块轮动信号

        委托给已有的 SectorRotationSkill
        """
        try:
            from skills.sector_rotation_skill import SectorRotationSkill
            skill = SectorRotationSkill(self.fetcher)
            return self._apply_bias_guard_signals(skill.scan(date), date)
        except ImportError:
            print("  ↪ sector_rotation_skill 不可用，使用内嵌 fallback")

        # fallback: 简化版板块轮动
        signals = []
        sectors = self._get_active_sectors(date)
        if not sectors:
            return signals

        sector_perf = {}
        for sector_name, sector_stocks in sectors.items():
            perfs = []
            for symbol in (sector_stocks or [])[:10]:
                try:
                    df = self._get_stock_data(symbol, date, days=10)
                    if df is not None and len(df) >= 5:
                        pct = (df["close"].iloc[-1] / df["close"].iloc[-5] - 1) * 100
                        perfs.append(float(pct))
                except Exception:
                    continue
            if perfs:
                sector_perf[sector_name] = float(np.mean(perfs))
        if not sector_perf:
            return signals

        top_sectors = sorted(sector_perf, key=lambda k: -sector_perf[k])[:3]
        for sector_name in top_sectors:
            for symbol in sectors.get(sector_name, []):
                try:
                    df = self._get_stock_data(symbol, date, days=60)
                    if df is None or len(df) < 20:
                        continue
                    close = df["close"].values
                    current = float(close[-1])
                    if len(close) >= 2:
                        prev_close = float(close[-2] or 0)
                        day_pct = (current / prev_close - 1) * 100 if prev_close > 0 else 0
                        if day_pct < -3:
                            continue
                    high_20d = float(max(df["high"].values[-20:]))
                    vol = df["vol"].values if "vol" in df.columns else None
                    vol_ratio = (
                        float(vol[-1] / np.mean(vol[-20:]))
                        if vol is not None and len(vol) >= 20 and np.mean(vol[-20:]) > 0 else 1.0
                    )
                    if current < high_20d * 0.98 or vol_ratio < 1.3:
                        continue
                    bias = self._calculate_bias(close)
                    score = 55 + int(sector_perf[sector_name]) + min(10, int(vol_ratio * 3))
                    score = max(0, min(100, score))
                    name = self._get_stock_name(symbol)
                    signals.append({
                        "type": "sector_rotation",
                        "symbol": symbol,
                        "name": name,
                        "score": score,
                        "action": "BUY",
                        "price": {
                            "close": round(current, 2),
                        },
                        "bias": round(bias, 2),
                        "confidence": round(score / 100, 2),
                        "reasons": [
                            f"强势板块({sector_name}, 5日+{sector_perf[sector_name]:.1f}%)",
                            f"量比{vol_ratio:.2f}",
                            "创20日新高" if current >= high_20d else "接近新高",
                        ],
                        "risk_notes": self._get_risk_notes(symbol, current, vol_ratio, bias=bias),
                    })
                except Exception:
                    continue
        return sorted(signals, key=lambda s: -s["score"])

    # ═══════════════════════════════════════════
    # 信号类型 5: 价值低估
    # ═══════════════════════════════════════════

    def _scan_sector_stage(self, date: str, pool: list = None) -> list:
        """
        板块Stage联合过滤信号 — 替代 value_undervalued
        """
        try:
            from skills.sector_stage_filter import SectorStageFilter
            stage_filter = SectorStageFilter()
            result = stage_filter.analyze(verbose=False)
            signals = []
            candidates = getattr(result, "candidates", []) or []
            for candidate in candidates:
                sig = self._convert_stage_candidate(candidate, date)
                if sig:
                    signals.append(sig)
            return sorted(signals, key=lambda s: -float(s.get("score", 0) or 0))
        except ImportError:
            pass
        except Exception as e:
            print(f"  ⚠️ sector_stage_filter 执行失败，使用内嵌 fallback: {e}")

        signals = []
        stocks = pool or self._get_stock_pool()
        for symbol in stocks:
            try:
                df = self._get_stock_data(symbol, date, days=120)
                if df is None or len(df) < 60:
                    continue
                close = np.array(df["close"].values, dtype=float)
                high = np.array(df["high"].values, dtype=float)
                low = np.array(df["low"].values, dtype=float)
                current = float(close[-1])
                if len(close) >= 2:
                    prev_close = float(close[-2] or 0)
                    day_pct = (current / prev_close - 1) * 100 if prev_close > 0 else 0
                    if day_pct < -3:
                        continue

                window = low[-120:] if len(low) >= 120 else low
                min_idx = int(np.argmin(window))
                if min_idx > len(window) - 20:
                    continue
                bottom_price = float(window[min_idx])
                recovery_pct = (current / bottom_price - 1) * 100 if bottom_price > 0 else 0
                if recovery_pct < 10 or recovery_pct > 80:
                    continue

                if len(close) < 50:
                    continue
                recent_high = float(np.max(high[-10:]))
                recent_low = float(np.min(low[-10:]))
                tight_range = (recent_high - recent_low) / recent_low * 100 if recent_low > 0 else 99
                if tight_range > 5:
                    continue

                ma10 = float(np.mean(close[-10:]))
                ma20 = float(np.mean(close[-20:]))
                ma50 = float(np.mean(close[-50:]))
                if not (ma10 > ma20 > ma50):
                    continue

                vol = np.array(df["vol"].values, dtype=float) if "vol" in df.columns else None
                vol_ratio = 1.0
                if vol is not None and len(vol) >= 20:
                    vol_5d = float(np.mean(vol[-5:]))
                    vol_20d = float(np.mean(vol[-20:]))
                    vol_ratio = vol_5d / vol_20d if vol_20d > 0 else 1.0
                    if vol_ratio > 1.0:
                        continue

                score = 60
                if tight_range < 3:
                    score += 10
                if recovery_pct > 20:
                    score += 5
                if vol_ratio < 0.6:
                    score += 5
                score = max(0, min(100, score))

                signals.append({
                    "type": "sector_stage",
                    "symbol": symbol,
                    "name": self._get_stock_name(symbol),
                    "score": score,
                    "action": "BUY",
                    "price": {
                        "close": round(current, 2),
                        "breakout": round(recent_high, 2),
                    },
                    "confidence": round(score / 100, 2),
                    "reasons": [
                        f"底部蓄力(回升{recovery_pct:.0f}%)",
                        f"10日收紧{tight_range:.1f}%",
                        "EMA多头排列",
                        f"缩量蓄力(量比{vol_ratio:.2f})",
                    ],
                    "risk_notes": [],
                })
            except Exception:
                continue
        return sorted(signals, key=lambda s: -float(s.get("score", 0) or 0))

    def _convert_stage_candidate(self, candidate, date: str) -> Optional[dict]:
        """将 SectorStageFilter 候选转为统一信号格式"""
        try:
            symbol = getattr(candidate, "ts_code", "")
            if not symbol:
                return None
            name = getattr(candidate, "name", "") or self._get_stock_name(symbol)
            score = int(round(float(getattr(candidate, "stage_score", 0) or 0)))
            close = 0.0
            breakout = 0.0
            df = self._get_stock_data(symbol, date, days=20)
            if df is not None and len(df) > 0:
                close = float(df["close"].iloc[-1])
                breakout = float(np.max(df["high"].tail(10)))
            reasons = [
                f"{getattr(candidate, 'stage_grade', 'Stage')}{getattr(candidate, 'trigger', '')}",
                f"R:R={float(getattr(candidate, 'rr_ratio', 0) or 0):.1f}",
                f"板块:{getattr(candidate, 'sector', '未知')}",
            ]
            return {
                "type": "sector_stage",
                "symbol": symbol,
                "name": name,
                "score": max(0, min(100, score)),
                "action": "BUY",
                "price": {
                    "close": round(close, 2),
                    "breakout": round(breakout, 2) if breakout > 0 else round(close, 2),
                },
                "confidence": round(max(0, min(100, score)) / 100, 2),
                "reasons": reasons,
                "risk_notes": list(getattr(candidate, "risk_notes", []) or []),
            }
        except Exception:
            return None

    def _scan_value_undervalued(self, date: str, pool: list = None) -> list:
        """
        价值低估信号

        委托给 ValueInvestorSkill
        """
        try:
            from value_investor import ValueInvestorSkill
            skill = ValueInvestorSkill(self.fetcher)
            base_pool = pool or self._get_stock_pool()
            value_limit = int(os.getenv("SIGNAL_VALUE_POOL_LIMIT", "80") or 80)
            scoped_pool = base_pool[:value_limit] if value_limit > 0 else base_pool
            scan_timeout = int(os.getenv("SIGNAL_VALUE_SCAN_TIMEOUT_SEC", "45") or 45)

            def _run():
                return skill.scan(date, pool=scoped_pool)

            fund_prof = {"calls": 0, "sec": 0.0, "name_calls": 0, "name_sec": 0.0}
            if hasattr(skill, "_get_fundamental"):
                _orig_get_fund = skill._get_fundamental

                def _timed_get_fund(symbol, aspect):
                    t0 = time.time()
                    try:
                        return _orig_get_fund(symbol, aspect)
                    finally:
                        fund_prof["calls"] += 1
                        fund_prof["sec"] += time.time() - t0

                skill._get_fundamental = _timed_get_fund

            if hasattr(skill, "_get_name"):
                _orig_get_name = skill._get_name

                def _timed_get_name(symbol):
                    t0 = time.time()
                    try:
                        return _orig_get_name(symbol)
                    finally:
                        fund_prof["name_calls"] += 1
                        fund_prof["name_sec"] += time.time() - t0

                skill._get_name = _timed_get_name

            try:
                scanned = self._call_with_timeout(_run, scan_timeout)
            except _Timeout:
                print(f"  ⚠️ value_undervalued 扫描超时({scan_timeout}s)，本轮跳过")
                scanned = []
            else:
                if fund_prof["calls"] > 0 or fund_prof["name_calls"] > 0:
                    print(
                        "  📈 value_undervalued 耗时剖析: "
                        f"fund={fund_prof['calls']}次/{fund_prof['sec']:.1f}s, "
                        f"name={fund_prof['name_calls']}次/{fund_prof['name_sec']:.1f}s"
                    )
            return self._apply_bias_guard_signals(scanned, date)
        except ImportError:
            print("  ↪ value_investor 不可用，使用内嵌 fallback")

        # fallback: 简化版价值低估
        signals = []
        pool_limit = int(os.getenv("SIGNAL_VALUE_POOL_LIMIT", "60") or 60)
        stocks = (pool or self._get_stock_pool())[:pool_limit]
        for symbol in stocks:
            try:
                if not self.fetcher or not hasattr(self.fetcher, "pro"):
                    continue
                if hasattr(self.fetcher, "_throttle"):
                    self.fetcher._throttle()
                indicators = self.fetcher.pro.daily_basic(
                    ts_code=symbol,
                    trade_date=date,
                    fields="ts_code,pe_ttm,pb,turnover_rate",
                )
                if indicators is None or indicators.empty:
                    continue
                pe = float(indicators.iloc[0].get("pe_ttm", 0) or 0)
                pb = float(indicators.iloc[0].get("pb", 0) or 0)
                if pe <= 0 or pe > 20 or pb <= 0 or pb > 2:
                    continue
                df = self._get_stock_data(symbol, date, days=30)
                if df is None or len(df) < 10:
                    continue
                close = df["close"].values
                current = float(close[-1])
                pct_5d = (current / float(close[-5]) - 1) * 100 if len(close) >= 5 else 0
                if pct_5d < 0:
                    continue
                bias = self._calculate_bias(close)
                score = 60
                if pe < 10:
                    score += 10
                if pb < 1:
                    score += 10
                if pct_5d > 3:
                    score += 5
                if bias < 0:
                    score += 5
                score = max(0, min(100, score))
                name = self._get_stock_name(symbol)
                signals.append({
                    "type": "value_undervalued",
                    "symbol": symbol,
                    "name": name,
                    "score": score,
                    "action": "BUY",
                    "price": {
                        "close": round(current, 2),
                    },
                    "bias": round(bias, 2),
                    "confidence": round(score / 100, 2),
                    "reasons": [
                        f"PE={pe:.1f} PB={pb:.2f}",
                        f"5日涨{pct_5d:.1f}%",
                        f"乖离率{bias:.1f}%",
                    ],
                    "risk_notes": self._get_risk_notes(symbol, current, 1.0, bias=bias),
                })
            except Exception:
                continue
        return sorted(signals, key=lambda s: -s["score"])

    def _apply_bias_guard_signals(self, signals: list, date: str) -> list:
        """
        对外部技能返回的信号统一施加乖离率保护。
        """
        guarded = []
        for sig in signals or []:
            try:
                sym = sig.get("symbol")
                if not sym:
                    continue
                df = self._get_stock_data(sym, date, days=60)
                if df is None or df.empty:
                    # 没有行情数据时仅补全 target 字段
                    p = sig.setdefault("price", {})
                    close = float(p.get("close", 0) or 0)
                    if close > 0 and "target" not in p:
                        p["target"] = round(close * 1.10, 2)
                    sig.setdefault("bias", 0.0)
                    guarded.append(sig)
                    continue

                close_arr = df["close"].values
                current = float(close_arr[-1])
                bias = self._calculate_bias(close_arr)
                if bias > 8:
                    continue
                score = float(sig.get("score", 0) or 0)
                if bias > 5:
                    score -= 30
                elif bias > 2:
                    score -= 10

                sig["score"] = int(max(0, min(100, round(score))))
                sig["bias"] = round(bias, 2)
                p = sig.setdefault("price", {})
                p.setdefault("close", round(current, 2))
                if "stop_loss" not in p and current > 0:
                    p["stop_loss"] = round(current * 0.93, 2)
                p.setdefault("target", round(float(p.get("close", current)) * 1.10, 2))

                risks = list(sig.get("risk_notes") or [])
                if bias > 5:
                    risks.append(f"⚠️乖离率{bias:.2f}%，追高风险")
                sig["risk_notes"] = list(dict.fromkeys(risks))
                guarded.append(sig)
            except Exception:
                continue
        return guarded

    def _apply_unified_filter(self, signals: list, date: str, env_score: int = 65) -> list:
        """
        统一出口过滤器 v2 — 突破发现 + 回踩入场模型
        """
        if env_score < UP["env_no_buy"]:
            return []

        filtered = []
        rejected_safety = 0
        for sig in signals or []:
            try:
                symbol = sig.get("symbol")
                if not symbol:
                    continue

                df = self._get_stock_data(symbol, date, days=90)
                if df is None or len(df) < 30:
                    continue

                # 主防线：当日行情安全检查（最高优先级）
                safety_ok, safety_reason = self._safety_check(df, symbol)
                if not safety_ok:
                    rejected_safety += 1
                    if rejected_safety <= 5:
                        print(f"    🚫 安全拦截: {symbol} — {safety_reason}")
                    continue

                close_arr = np.array(df["close"].values, dtype=float)
                high_arr = np.array(df["high"].values, dtype=float)
                current = float(close_arr[-1])
                if current <= 0:
                    continue

                atr = self._calc_atr_unified(df)
                if atr <= 0:
                    continue

                # 乖离率仅用于阶段判断；极端过热保底过滤
                bias = self._calculate_bias(close_arr)
                if bias > UP["bias_extreme"]:
                    continue

                breakout_price = float(sig.get("price", {}).get("breakout", 0) or 0)
                if breakout_price <= 0:
                    if len(high_arr) > 20:
                        breakout_price = float(np.max(high_arr[-21:-1]))
                    elif len(high_arr) > 1:
                        breakout_price = float(np.max(high_arr[:-1]))
                    else:
                        breakout_price = current

                ema10 = self._calc_ema(close_arr, 10)
                entry_low = max(breakout_price, ema10)
                entry_high = entry_low * (1 + UP["pullback_tolerance"])
                entry_basis = (
                    f"EMA10回踩({ema10:.2f})"
                    if ema10 >= breakout_price
                    else f"突破位回踩({breakout_price:.2f})"
                )

                stop_a = breakout_price * (1 - UP["stop_below_breakout"])
                stop_b = entry_low - UP["stop_atr_multiple"] * atr
                stop = max(stop_a, stop_b)
                stop = max(stop, entry_low * (1 - UP["max_stop_pct"]))

                target = entry_low + UP["target_atr_multiple"] * atr
                target = min(target, entry_low * (1 + UP["max_target_pct"]))

                chase_risk = current - stop
                chase_reward = target - current
                chase_rr = round(chase_reward / chase_risk, 1) if chase_risk > 0 else 0

                entry_risk = entry_low - stop
                entry_reward = target - entry_low
                entry_rr = round(entry_reward / entry_risk, 1) if entry_risk > 0 else 0
                if entry_rr < UP["min_rr_ratio"]:
                    continue

                if bias < UP["bias_immediate_buy"]:
                    signal_stage = "buy"
                    actual_entry = current
                    actual_rr = chase_rr
                    stage_reason = f"乖离率{bias:.1f}%<2%，价格在合理位置，可直接买入"
                elif bias < UP["bias_watch_zone"]:
                    signal_stage = "observe"
                    actual_entry = entry_low
                    actual_rr = entry_rr
                    stage_reason = f"乖离率{bias:.1f}%，等回踩至{entry_low:.2f}~{entry_high:.2f}"
                else:
                    signal_stage = "observe"
                    actual_entry = entry_low
                    actual_rr = entry_rr
                    stage_reason = f"乖离率{bias:.1f}%偏高，必须等回踩至{entry_low:.2f}"

                vol_ratio = float(sig.get("vol_ratio", 0) or 0)
                if vol_ratio <= 0:
                    vol = np.array(df["vol"].values, dtype=float) if "vol" in df.columns else None
                    if vol is not None and len(vol) >= 20:
                        avg_vol = float(np.mean(vol[-20:]))
                        if avg_vol > 0:
                            vol_ratio = float(vol[-1]) / avg_vol
                if vol_ratio <= 0:
                    vol_ratio = 1.0

                sig_type = str(sig.get("type", ""))
                min_vol = UP["breakout_vol_ratio"] if "breakout" in sig_type else UP["min_vol_ratio"]
                if vol_ratio < min_vol:
                    continue

                score = float(sig.get("score", 0) or 0)
                if entry_rr >= 5:
                    score += 15
                elif entry_rr >= 4:
                    score += 10
                elif entry_rr >= 3:
                    score += 5

                if vol_ratio >= 2.5:
                    score += 10
                elif vol_ratio >= 2.0:
                    score += 5

                for r in (sig.get("reasons") or []):
                    if "整理" in r:
                        nums = "".join(ch for ch in r if ch.isdigit())
                        if nums:
                            days_num = int(nums)
                            if days_num >= 25:
                                score += 10
                            elif days_num >= 20:
                                score += 5
                        break

                score = int(max(0, min(100, round(score))))
                if score < UP["min_score"]:
                    continue

                position_pct = self._calc_position_unified(actual_entry, stop)
                if env_score < UP["env_reduce_position"]:
                    position_pct *= 0.5
                position_pct = round(min(position_pct, UP["max_single_position"]), 4)

                p0 = sig.get("price", {}) or {}
                valid_until = self._calc_valid_until(date, UP["pullback_valid_days"])
                sig["price"] = {
                    "close": round(current, 2),
                    "breakout": p0.get("breakout", round(current, 2)),
                    "stop_loss": round(stop, 2),
                    "target": round(target, 2),
                }
                sig["suggested_entry"] = {
                    "price_low": round(entry_low, 2),
                    "price_high": round(entry_high, 2),
                    "basis": entry_basis,
                }
                sig["score"] = score
                sig["signal_stage"] = signal_stage
                sig["stage_reason"] = stage_reason
                sig["bias"] = round(bias, 2)
                sig["bias_at_breakout"] = round(bias, 2)
                sig["rr_ratio"] = chase_rr
                sig["entry_rr_ratio"] = entry_rr
                sig["vol_ratio"] = round(vol_ratio, 2)
                sig["position_pct"] = position_pct
                sig["confidence"] = round(score / 100, 2)
                sig["breakout_date"] = date
                sig["valid_until"] = valid_until

                risks = list(sig.get("risk_notes") or [])
                if bias > UP["bias_watch_zone"]:
                    risks.append(f"乖离率{bias:.1f}%偏高，需等充分回踩")
                if env_score < UP["env_reduce_position"]:
                    risks.append(f"⚠️ 环境{env_score}分，仓位已减半")
                if chase_rr < UP["min_rr_ratio"]:
                    risks.append(f"追突破R:R仅{chase_rr}，回踩入场可改善至{entry_rr}")
                sig["risk_notes"] = list(dict.fromkeys(risks))

                filtered.append(sig)
            except Exception:
                continue

        if rejected_safety > 0:
            print(f"    🚫 安全检查共拦截 {rejected_safety} 只")

        stage_order = {"buy": 0, "observe": 1}
        filtered.sort(key=lambda s: (
            stage_order.get(s.get("signal_stage", "observe"), 9),
            -float(s.get("entry_rr_ratio", 0) or 0),
            -float(s.get("score", 0) or 0),
        ))
        return filtered

    def _calc_atr_unified(self, df, period=None):
        """统一 ATR 计算（与 Skill 8 对齐）"""
        period = period or UP["atr_period"]
        return self._calc_atr_simple(df, period=period)

    def _calc_position_unified(self, entry: float, stop: float) -> float:
        """
        固定风险法仓位：股数=(总资金×1%)÷(入场-止损), 仓位%=股数×入场÷总资金
        """
        if entry <= 0 or stop <= 0 or stop >= entry:
            return 0.0
        capital = float(UP["default_capital"])
        risk_per_share = max(entry - stop, entry * 0.01)
        max_loss = capital * float(UP["risk_per_trade"])
        shares = max_loss / risk_per_share
        pct = shares * entry / capital
        return float(min(pct, float(UP["max_single_position"])))

    def _calc_ema(self, close_arr, period: int) -> float:
        """计算EMA最新值"""
        if close_arr is None or len(close_arr) == 0:
            return 0.0
        arr = np.array(close_arr, dtype=float)
        if len(arr) < period:
            return float(np.mean(arr))
        multiplier = 2 / (period + 1)
        ema = float(arr[0])
        for price in arr[1:]:
            ema = float(price) * multiplier + ema * (1 - multiplier)
        return round(ema, 4)

    def _calc_valid_until(self, date_str: str, trading_days: int) -> str:
        """计算有效截止日（简化：交易日×1.5）"""
        try:
            dt = datetime.strptime(date_str, "%Y%m%d")
            calendar_days = int(trading_days * 1.5)
            return (dt + timedelta(days=calendar_days)).strftime("%Y%m%d")
        except Exception:
            return date_str

    def _get_limit_pct(self, symbol: str) -> float:
        """
        获取涨跌停幅度：
        主板10%，创业板/科创板20%，北交所30%
        """
        code = ""
        if isinstance(symbol, str):
            code = symbol.split(".")[0]
        if code.startswith("30") or code.startswith("68"):
            return 20.0
        if code.startswith("8") or code.startswith("4"):
            return 30.0
        return 10.0

    def _safety_check(self, df: pd.DataFrame, symbol: str) -> tuple:
        """
        当日行情安全检查：在任何策略逻辑前拦截高风险标的。
        """
        if df is None or len(df) < 2:
            return False, "数据不足"
        try:
            today = df.iloc[-1]
            yesterday = df.iloc[-2]
            close = float(today["close"])
            prev_close = float(yesterday["close"])
            today_open = float(today.get("open", close))
            if close <= 0 or prev_close <= 0:
                return False, "价格异常"

            pct_change = (close / prev_close - 1) * 100
            if pct_change < float(UP["safety_max_daily_drop"]):
                return False, f"当日跌{pct_change:.1f}%(<{UP['safety_max_daily_drop']}%)"

            if bool(UP.get("safety_limit_reject", True)):
                limit_pct = self._get_limit_pct(symbol)
                if pct_change <= -limit_pct + 0.5:
                    return False, f"触及跌停({pct_change:.1f}%)"
                if pct_change >= limit_pct - 0.5:
                    return False, f"触及涨停({pct_change:.1f}%)封板买不到"

            if pct_change > float(UP["safety_max_daily_rise"]):
                return False, f"当日涨{pct_change:.1f}%追高风险"

            if len(df) >= 4:
                close_3d_ago = float(df.iloc[-4]["close"])
                pct_3d = (close / close_3d_ago - 1) * 100 if close_3d_ago > 0 else 0
                if pct_3d < float(UP["safety_3d_max_drop"]):
                    return False, f"近3日累跌{pct_3d:.1f}%"

            if "vol" in df.columns and len(df) >= 20:
                vol = float(today["vol"])
                avg_vol = float(df["vol"].iloc[-21:-1].mean())
                vol_ratio = vol / avg_vol if avg_vol > 0 else 1.0
                if vol_ratio > float(UP["safety_vol_spike_threshold"]) and pct_change < -2:
                    return False, f"异常放量{vol_ratio:.1f}x+跌{pct_change:.1f}%(疑似出货)"

                body_pct = (close - today_open) / today_open * 100 if today_open > 0 else 0
                if body_pct < -5 and vol_ratio > 1.5:
                    return False, f"放量大阴线(实体{body_pct:.1f}%,量比{vol_ratio:.1f}x)"

            return True, ""
        except Exception as e:
            return False, f"安全检查异常:{e}"

    def check_pullback_entries(self, observe_signals: list, date: str) -> dict:
        """每日检查：观察信号是否触发回踩入场"""
        triggered = []
        still_watching = []
        expired = []

        for sig in observe_signals or []:
            symbol = sig.get("symbol")
            if not symbol:
                continue
            valid_until = sig.get("valid_until", "")
            if not valid_until:
                valid_until = self._calc_valid_until(
                    sig.get("breakout_date", date), UP["pullback_valid_days"]
                )
                sig["valid_until"] = valid_until
            if date > valid_until:
                sig["signal_stage"] = "expired"
                sig["stage_reason"] = f"超过有效期({valid_until})未回踩，放弃"
                expired.append(sig)
                continue
            try:
                df = self._get_stock_data(symbol, date, days=5)
                if df is None or len(df) == 0:
                    still_watching.append(sig)
                    continue
                today = df.iloc[-1]
                today_low = float(today["low"])
                today_close = float(today["close"])
                entry = sig.get("suggested_entry", {}) or {}
                entry_low = float(entry.get("price_low", 0) or 0)
                entry_high = float(entry.get("price_high", 0) or 0)
                stop = float(sig.get("price", {}).get("stop_loss", 0) or 0)
                if entry_low <= 0:
                    still_watching.append(sig)
                    continue

                if today_low < stop:
                    sig["signal_stage"] = "invalidated"
                    sig["stage_reason"] = f"最低价{today_low:.2f}跌破止损{stop:.2f}，信号失效"
                    expired.append(sig)
                    continue

                tolerance = entry_low * UP["pullback_tolerance"]
                if today_low <= entry_high + tolerance:
                    actual_entry_price = min(today_close, entry_high)
                    actual_entry_price = max(actual_entry_price, entry_low)
                    target = float(sig.get("price", {}).get("target", 0) or 0)
                    risk = actual_entry_price - stop
                    reward = target - actual_entry_price
                    new_rr = round(reward / risk, 1) if risk > 0 else 0
                    if new_rr < UP["min_rr_ratio"]:
                        still_watching.append(sig)
                        continue
                    new_bias = self._calculate_bias(df["close"].values)
                    sig["signal_stage"] = "buy"
                    sig["stage_reason"] = (
                        f"回踩触发！今日低点{today_low:.2f}进入入场区间"
                        f"({entry_low:.2f}~{entry_high:.2f})"
                    )
                    sig.setdefault("price", {})
                    sig["price"]["entry"] = round(actual_entry_price, 2)
                    sig["rr_ratio"] = new_rr
                    sig["entry_rr_ratio"] = new_rr
                    sig["bias"] = round(new_bias, 2)
                    sig["pullback_date"] = date
                    sig["position_pct"] = round(self._calc_position_unified(actual_entry_price, stop), 4)
                    triggered.append(sig)
                    continue

                sig.setdefault("price", {})
                sig["price"]["close"] = round(today_close, 2)
                sig["bias"] = round(self._calculate_bias(df["close"].values), 2)
                sig["days_watching"] = int(sig.get("days_watching", 0) or 0) + 1
                still_watching.append(sig)
            except Exception:
                still_watching.append(sig)
                continue

        return {
            "triggered": triggered,
            "watching": still_watching,
            "expired": expired,
        }

    def save_observe_signals(self, signals: list):
        """保存观察信号到本地文件"""
        try:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OBSERVE_SIGNALS_FILE)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._to_json_safe(signals or []), f, ensure_ascii=False, indent=2)
            print(f"  💾 已保存 {len(signals or [])} 条观察信号")
        except Exception as e:
            print(f"  ⚠️ 保存观察信号失败: {e}")

    def load_observe_signals(self) -> list:
        """加载已有观察信号"""
        try:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OBSERVE_SIGNALS_FILE)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    signals = json.load(f)
                valid = [s for s in (signals or []) if isinstance(s, dict) and s.get("symbol")]
                print(f"  📂 加载 {len(valid)} 条历史观察信号")
                return valid
        except Exception as e:
            print(f"  ⚠️ 加载观察信号失败: {e}")
        return []

    def _to_json_safe(self, obj):
        """将 numpy/pandas 类型转换为 JSON 可序列化的原生类型"""
        if isinstance(obj, dict):
            return {str(k): self._to_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._to_json_safe(v) for v in obj]
        if isinstance(obj, tuple):
            return [self._to_json_safe(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return [self._to_json_safe(v) for v in obj.tolist()]
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    # ═══════════════════════════════════════════
    # v6.0 辅助信息
    # ═══════════════════════════════════════════

    def _get_auxiliary_info(self, symbol: str, date: str, df_cache=None) -> dict:
        """
        v6.0 辅助信息（不影响选股，仅供参考）
        """
        info = {}
        try:
            df = df_cache if df_cache is not None else self._get_stock_data(symbol, date, days=60)
            if df is None or len(df) < 10:
                return info

            close = df["close"].values
            high = df["high"].values
            current = close[-1]
            info["bias"] = self._calculate_bias(close)

            # [1] 60天新高
            high_60d = max(high)
            info["is_60d_high"] = current >= high_60d * 0.99
            info["pct_from_high"] = round((current / high_60d - 1) * 100, 1)

            # [2] 形态质量
            consolidation = self._find_consolidation(close, high, df["low"].values)
            if consolidation:
                _, _, _, _, days = consolidation
                if days >= 25:
                    info["pattern_quality"] = "优秀"
                elif days >= 18:
                    info["pattern_quality"] = "良好"
                else:
                    info["pattern_quality"] = "一般"
            else:
                info["pattern_quality"] = "无整理"

            # [3] 量比
            if "vol" in df.columns and len(df) >= 20:
                avg = df["vol"].iloc[-20:].mean()
                info["vol_ratio"] = round(df["vol"].iloc[-1] / avg, 2) if avg > 0 else 0

            # [4] 板块共振 (简化版)
            info["sector_resonance"] = "待计算"

            # [5] 风险提示
            info["risk_notes"] = self._get_risk_notes(
                symbol, current, info.get("vol_ratio", 1), bias=info.get("bias", 0)
            )

        except Exception:
            pass

        return info

    def _get_risk_notes(self, symbol: str, price: float, vol_ratio: float, bias: float = 0.0) -> list:
        """生成风险提示"""
        risks = []
        if vol_ratio > 3:
            risks.append("量能过大，警惕冲高回落")
        if vol_ratio < 0.8:
            risks.append("量能不足，突破可靠性存疑")
        if bias > 5:
            risks.append(f"⚠️乖离率{bias:.2f}%，追高风险")
        return risks

    # ═══════════════════════════════════════════
    # 输出格式化
    # ═══════════════════════════════════════════

    def format_signals(self, result: dict) -> str:
        """
        格式化信号输出（v6.0 精简格式）

        输出示例：
        市场环境: 76/100 (良好) - 可交易

        [1] 南山铝业 (600219.SH) 评分82
            60天新高: 是 (+2.9%)
            量比: 1.75
            板块: 强共振(3只)
            形态: 优秀 (冲量45% + 整理30天)
            价格: 收5.31 / 突破5.14 / 止损4.41
            风险: 无
        """
        lines = []

        # 市场环境
        env = result.get("market_env")
        if env:
            lines.append(env.get("summary", f"市场环境: {env.get('total_score', '?')}/100"))
            lines.append("")

        signals = result.get("signals", [])
        if not signals:
            lines.append("今日无信号")
            return "\n".join(lines)

        lines.append(f"共 {len(signals)} 个信号:")
        lines.append("")

        for i, sig in enumerate(signals, 1):
            # 基础信息
            lines.append(
                f"[{i}] {sig.get('name', '?')} ({sig['symbol']}) "
                f"评分{sig['score']} [{sig['type']}]"
            )

            # 辅助信息（v6.0）
            aux = sig.get("auxiliary_info", {})
            if aux.get("is_60d_high") is not None:
                yn = "是" if aux["is_60d_high"] else "否"
                lines.append(f"    60天新高: {yn} ({aux.get('pct_from_high', '?')}%)")
            if aux.get("vol_ratio"):
                lines.append(f"    量比: {aux['vol_ratio']}")
            if aux.get("pattern_quality"):
                lines.append(f"    形态: {aux['pattern_quality']}")

            # 价格
            p = sig.get("price", {})
            if p:
                parts = []
                if "close" in p:
                    parts.append(f"收{p['close']}")
                if "breakout" in p:
                    parts.append(f"突破{p['breakout']}")
                if "stop_loss" in p:
                    parts.append(f"止损{p['stop_loss']}")
                lines.append(f"    价格: {' / '.join(parts)}")

            # 原因
            reasons = sig.get("reasons", [])
            if reasons:
                lines.append(f"    依据: {', '.join(reasons)}")

            # 风险
            risks = sig.get("risk_notes") or aux.get("risk_notes", [])
            lines.append(f"    风险: {'、'.join(risks) if risks else '无'}")
            lines.append("")

        return "\n".join(lines)

    def format_dashboard(self, result: dict, env_score: int) -> str:
        date_str = result.get("date") or datetime.now().strftime("%Y%m%d")
        try:
            date_fmt = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        except Exception:
            date_fmt = date_str

        # 输出前最终安全兜底
        buy_signals = self._final_safety_sweep(result.get("buy_signals", []) or [])
        observe_signals = self._final_safety_sweep(result.get("observe_signals", []) or [])
        pullback = result.get("pullback_result", {}) or {}
        triggered = self._final_safety_sweep(pullback.get("triggered", []) or [])
        expired = pullback.get("expired", []) or []

        env_band = "较差-观望" if env_score < 60 else "一般-谨慎" if env_score < 75 else "良好-可交易"
        pool_info = result.get("pool_info", {}) or {}
        pool_size = int(pool_info.get("pool_size", 0) or 0)
        breadth = pool_info.get("market_breadth", "未知")
        lines = [
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"📊 {date_fmt} 信号仪表盘",
            f"环境: {env_score}/100 ({env_band})",
            f"股票池: {pool_size}只 ({breadth})",
            f"🟢买入:{len(buy_signals)} 🔔观察:{len(observe_signals)} 🎯回踩触发:{len(triggered)} ⏰过期:{len(expired)}",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "",
        ]

        buy_base = [s for s in buy_signals if not s.get("pullback_date")]
        if buy_base:
            lines.append("【🟢 买入信号 — 可操作】")
            lines.append("")
            for sig in buy_base:
                lines.extend(self._format_signal_detail(sig, env_score))
                lines.append("")

        if triggered:
            lines.append("【🎯 回踩触发 — 今日升级为买入】")
            lines.append("")
            for sig in triggered:
                lines.extend(self._format_signal_detail(sig, env_score))
                pd = sig.get("pullback_date", "")
                bd = sig.get("breakout_date", "")
                lines.append(f"  📅 突破日{bd} → 回踩触发日{pd}")
                lines.append("")

        if observe_signals:
            lines.append("【🔔 观察信号 — 等回踩入场】")
            lines.append("")
            for sig in observe_signals:
                lines.extend(self._format_observe_detail(sig))
                lines.append("")

        if expired:
            lines.append(f"【⏰ 过期/失效: {len(expired)}只】")
            for sig in expired:
                name = sig.get("name", sig.get("symbol", "?"))
                reason = sig.get("stage_reason", "超期")
                lines.append(f"  {name} — {reason}")
            lines.append("")

        if not buy_signals and not observe_signals and not triggered:
            lines.append("今日无新信号")
            lines.append("")

        lines.extend([
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"💡 建议: {self._env_advice(env_score)}",
        ])
        return "\n".join(lines)

    def _final_safety_sweep(self, signals: list) -> list:
        """
        输出前最终安全兜底，检查价格结构异常。
        """
        clean = []
        for sig in signals or []:
            try:
                p = sig.get("price", {}) or {}
                close = float(p.get("close", 0) or 0)
                stop = float(p.get("stop_loss", 0) or 0)
                target = float(p.get("target", 0) or 0)
                if close <= 0:
                    continue
                if stop <= 0 or stop >= close:
                    continue
                if target <= close:
                    continue
                stop_pct = (close - stop) / close * 100
                if stop_pct > 15:
                    continue
                entry = sig.get("suggested_entry", {}) or {}
                entry_low = float(entry.get("price_low", 0) or 0)
                if entry_low > 0 and entry_low > close * 1.15:
                    continue
                clean.append(sig)
            except Exception:
                continue
        return clean

    def _format_signal_detail(self, sig: dict, env_score: int) -> list:
        lines = []
        symbol = sig.get("symbol", "?")
        name = sig.get("name", symbol)
        sig_type = self._friendly_signal_type(sig.get("type", ""))
        score = int(sig.get("score", 0) or 0)
        lines.append(f"🟢 买入 | {name}({symbol}) | {sig_type} {score}分")
        lines.append(f"  📌 {sig.get('stage_reason', '')}")

        p = sig.get("price", {}) or {}
        close = float(p.get("close", 0) or 0)
        stop_loss = float(p.get("stop_loss", 0) or 0)
        target = float(p.get("target", 0) or 0)
        entry_price = float(p.get("entry", close) or close)
        entry_rr = float(sig.get("entry_rr_ratio", 0) or 0)
        stop_pct = ((stop_loss / entry_price - 1) * 100) if entry_price > 0 and stop_loss > 0 else 0
        target_pct = ((target / entry_price - 1) * 100) if entry_price > 0 and target > 0 else 0
        lines.append(
            f"  💰 入场{entry_price:.2f} | 止损{stop_loss:.2f}({stop_pct:+.1f}%) | 目标{target:.2f}({target_pct:+.1f}%) | R:R={entry_rr:.1f}"
        )

        pos_pct = float(sig.get("position_pct", 0) or 0)
        if pos_pct > 0 and entry_price > 0:
            shares = int(UP["default_capital"] * pos_pct / entry_price / 100) * 100
            lines.append(f"  📐 建议仓位: {pos_pct*100:.1f}% ({shares}股)")

        checks = self._build_checklist_v2(sig, env_score)
        if checks:
            lines.append(f"  {' '.join(checks)}")

        news = sig.get("news") or {}
        if news:
            summary = news.get("summary", "")
            adj = int(news.get("score_adj", 0) or 0)
            if summary:
                mark = f"+{adj}" if adj > 0 else str(adj)
                suffix = f" → {mark}分" if adj != 0 else ""
                lines.append(f"  📰 近期: {summary}{suffix}")
        return lines

    def _format_observe_detail(self, sig: dict) -> list:
        lines = []
        symbol = sig.get("symbol", "?")
        name = sig.get("name", symbol)
        sig_type = self._friendly_signal_type(sig.get("type", ""))
        score = int(sig.get("score", 0) or 0)
        bias = float(sig.get("bias", 0) or 0)
        entry_rr = float(sig.get("entry_rr_ratio", 0) or 0)
        valid_until = sig.get("valid_until", "")
        days_watching = int(sig.get("days_watching", 0) or 0)
        entry = sig.get("suggested_entry", {}) or {}
        entry_low = float(entry.get("price_low", 0) or 0)
        entry_high = float(entry.get("price_high", 0) or 0)
        entry_basis = entry.get("basis", "")
        p = sig.get("price", {}) or {}
        current = float(p.get("close", 0) or 0)
        stop = float(p.get("stop_loss", 0) or 0)
        target = float(p.get("target", 0) or 0)
        distance = ((current / entry_high - 1) * 100) if entry_high > 0 and current > 0 else 0

        lines.append(f"🔔 观察 | {name}({symbol}) | {sig_type} {score}分")
        lines.append(f"  📌 乖离率{bias:.1f}%，等回踩入场")
        lines.append(
            f"  🎯 入场区间: {entry_low:.2f}~{entry_high:.2f} ({entry_basis}) | 当前{current:.2f} (距入场{distance:+.1f}%)"
        )
        lines.append(f"  💰 止损{stop:.2f} | 目标{target:.2f} | 回踩入场R:R={entry_rr:.1f}")
        valid_fmt = (
            f"{valid_until[:4]}-{valid_until[4:6]}-{valid_until[6:8]}"
            if isinstance(valid_until, str) and len(valid_until) == 8 else valid_until
        )
        lines.append(f"  ⏰ 有效至{valid_fmt} | 已观察{days_watching}天")
        return lines

    def _build_checklist_v2(self, sig: dict, env_score: int) -> list:
        checks = []
        reasons = " ".join(sig.get("reasons", []) or [])
        if "阶梯整理" in reasons:
            checks.append(next((f"✅ {r}" for r in sig.get("reasons", []) if "阶梯整理" in r), "✅ 阶梯形态"))
        elif "龙头" in reasons:
            checks.append("✅ 板块龙头")

        vol_ratio = float(sig.get("vol_ratio", 0) or 0)
        if vol_ratio >= 1.5:
            checks.append(f"✅ 量比{vol_ratio:.2f}")
        elif vol_ratio >= 1.2:
            checks.append(f"⚠️ 量比{vol_ratio:.2f}")
        else:
            checks.append("❌ 量能不足")

        entry_rr = float(sig.get("entry_rr_ratio", 0) or 0)
        if entry_rr >= 3:
            checks.append(f"✅ R:R={entry_rr:.1f}")
        elif entry_rr >= 2:
            checks.append(f"⚠️ R:R={entry_rr:.1f}")

        if env_score >= 75:
            checks.append(f"✅ 环境{env_score}")
        elif env_score >= 60:
            checks.append(f"⚠️ 环境{env_score}")
        else:
            checks.append(f"❌ 环境{env_score}")
        return checks

    def _env_advice(self, env_score: int) -> str:
        if env_score < 60:
            return "环境较差，建议不操作，仅观察"
        if env_score < 75:
            return "环境一般，轻仓操作，优先乖离率<2%标的"
        return "环境良好，可正常执行策略"

    def _friendly_signal_type(self, sig_type: str) -> str:
        mapping = {
            "ladder_breakout": "阶梯突破",
            "leader_breakout": "龙头突破",
            "canslim": "CANSLIM",
            "sector_rotation": "板块轮动",
            "sector_stage": "板块Stage",
            "value_undervalued": "价值低估",
        }
        return mapping.get(sig_type, sig_type)

    # ═══════════════════════════════════════════
    # 摘要
    # ═══════════════════════════════════════════

    def _build_summary(self, date, env_result, signals, by_type) -> str:
        lines = [f"=== 交易信号扫描 {date} ==="]
        if env_result:
            lines.append(env_result.get("summary", ""))
        lines.append(f"总信号: {len(signals)} 个")
        for t, sigs in by_type.items():
            if sigs:
                lines.append(f"  {t}: {len(sigs)} 个")
        if signals:
            top = signals[0]
            lines.append(f"最强信号: {top.get('name', '?')} ({top['symbol']}) 评分{top['score']}")
        return "\n".join(lines)

    def _build_summary_v2(self, date, env_result, buy_signals, observe_signals, pullback_result) -> str:
        """v2 摘要：区分买入和观察"""
        lines = [f"=== 交易信号扫描 {date} ==="]
        if env_result:
            lines.append(env_result.get("summary", ""))
        lines.append(f"🟢 可买入: {len(buy_signals)} 只")
        lines.append(f"🔔 观察等回踩: {len(observe_signals)} 只")

        triggered = pullback_result.get("triggered", []) or []
        if triggered:
            lines.append(f"🎯 今日回踩触发: {len(triggered)} 只")
            for s in triggered:
                lines.append(f"  → {s.get('name', '?')} ({s.get('symbol', '?')})")

        expired = pullback_result.get("expired", []) or []
        if expired:
            lines.append(f"⏰ 今日过期: {len(expired)} 只")

        if buy_signals:
            top = buy_signals[0]
            lines.append(
                f"最强买入: {top.get('name', '?')} ({top.get('symbol', '?')}) "
                f"评分{top.get('score', 0)} R:R={top.get('entry_rr_ratio', 0)}"
            )
        return "\n".join(lines)

    # ═══════════════════════════════════════════
    # EventLog
    # ═══════════════════════════════════════════

    def _emit_signals(self, date, signals, env_result):
        if not self.event_log:
            return
        try:
            for sig in signals[:10]:  # 最多记录 10 条
                self.event_log.emit(
                    f"signal.{sig.get('action', 'watch').lower()}",
                    {
                        "type": sig["type"],
                        "symbol": sig["symbol"],
                        "score": sig["score"],
                        "price": sig.get("price", {}),
                    },
                    source="trade_signals",
                )
        except Exception:
            pass

    def _emit_error(self, scan_name, error):
        if self.event_log:
            try:
                self.event_log.emit("system.error", {
                    "module": "trade_signals",
                    "scan": scan_name,
                    "error": str(error),
                }, source="trade_signals")
            except Exception:
                pass

    # ═══════════════════════════════════════════
    # 数据获取适配
    # ═══════════════════════════════════════════

    def _resolve_fetch_method(self):
        """探测并缓存可用的数据获取方法，避免每次 fallback 链"""
        if self._fetch_method:
            return self._fetch_method
        for method in ["get_daily", "get_stock_daily", "get_k_data"]:
            if hasattr(self.fetcher, method):
                self._fetch_method = method
                return method
        return None

    def _call_with_timeout(self, fn, timeout_sec: int):
        """
        单次调用超时保护（Unix）。
        保留并恢复外层 alarm（例如 daily_workflow 的全局超时）。
        """
        if timeout_sec <= 0:
            return fn()
        if not hasattr(_signal, "SIGALRM"):
            return fn()
        if threading.current_thread() is not threading.main_thread():
            return fn()

        prev_handler = _signal.getsignal(_signal.SIGALRM)
        prev_remaining = _signal.alarm(0)  # 返回之前剩余秒数
        prev_deadline = time.monotonic() + prev_remaining if prev_remaining > 0 else None
        try:
            _signal.signal(_signal.SIGALRM, _timeout_handler)
            _signal.alarm(max(1, int(timeout_sec)))
            return fn()
        finally:
            _signal.alarm(0)
            _signal.signal(_signal.SIGALRM, prev_handler)
            if prev_deadline is not None:
                left = int(max(1, round(prev_deadline - time.monotonic())))
                _signal.alarm(left)

    def _get_stock_data(self, symbol, date, days=60, allow_network=True):
        cache_key = (symbol, days)
        if cache_key in self._data_cache:
            self._cache_hits += 1
            return self._data_cache[cache_key]

        # 复用更长天数缓存
        longer_keys = [k for k in self._data_cache.keys() if k[0] == symbol and k[1] >= days]
        if longer_keys:
            best = min(longer_keys, key=lambda x: x[1])
            cached_df = self._data_cache.get(best)
            if cached_df is not None:
                sliced = cached_df.tail(days).reset_index(drop=True) if hasattr(cached_df, "tail") else cached_df
                self._data_cache[cache_key] = sliced
                self._cache_hits += 1
                return sliced

        # 优先本地数据缓存层
        if self.data_cache:
            try:
                df = self.data_cache.get_daily(symbol, end_date=date, days=days)
                if df is not None and len(df) > 0:
                    self._data_cache[cache_key] = df
                    self._cache_hits += 1
                    return df
            except Exception:
                pass

        if not self.fetcher:
            self._data_cache[cache_key] = None
            return None

        self._cache_misses += 1
        if not allow_network:
            self._data_cache[cache_key] = None
            return None

        method = self._resolve_fetch_method()
        if not method:
            self._data_cache[cache_key] = None
            return None

        try:
            def _do_call():
                return getattr(self.fetcher, method)(symbol, days=days, end_date=date)

            self._network_fetches += 1
            result = self._call_with_timeout(_do_call, self._fetch_timeout_sec)
            self._data_cache[cache_key] = result
            return result
        except _Timeout:
            print(f"    ⏰ {symbol} 数据获取超时({self._fetch_timeout_sec}s)，跳过")
            self._data_cache[cache_key] = None
            return None
        except Exception:
            self._data_cache[cache_key] = None
            return None

    def _calc_atr_simple(self, df, period=14):
        """简易 ATR 计算"""
        try:
            if df is None or len(df) < period + 1:
                return 0.0
            highs = np.array(df["high"].values, dtype=float)
            lows = np.array(df["low"].values, dtype=float)
            closes = np.array(df["close"].values, dtype=float)
            trs = []
            for i in range(1, len(df)):
                tr = max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1]),
                )
                trs.append(tr)
            if not trs:
                return 0.0
            n = min(period, len(trs))
            return float(np.mean(trs[-n:]))
        except Exception:
            return 0.0

    def _prefetch_pool_data(self, stock_pool: list, date: str, days: int = 90):
        """
        批量预加载候选池行情（best-effort）。
        成功则写入 _data_cache；失败自动回退逐只拉取。
        """
        if not stock_pool:
            return
        if not self.fetcher or not hasattr(self.fetcher, "pro"):
            return
        if not hasattr(self.fetcher.pro, "daily"):
            return
        end_dt = datetime.strptime(date, "%Y%m%d")
        start_dt = end_dt - timedelta(days=int(days * 1.5))
        start_date = start_dt.strftime("%Y%m%d")
        t0 = time.time()
        cached_before = len(self._data_cache)
        loaded = 0
        invalid_count = 0
        try:
            print(f"  📦 批量预加载 {len(stock_pool)} 只 ({start_date}~{date}) ...")
            batch_size = 50
            wanted = set(stock_pool)
            for i in range(0, len(stock_pool), batch_size):
                batch = stock_pool[i:i + batch_size]
                ts_code_csv = ",".join(batch)
                if hasattr(self.fetcher, "_throttle"):
                    self.fetcher._throttle()
                df = self.fetcher.pro.daily(
                    ts_code=ts_code_csv,
                    start_date=start_date,
                    end_date=date,
                )
                if df is None or df.empty:
                    continue

                # 尽量统一为前复权口径，减少与 pro_bar(adj='qfq') 差异
                try:
                    if hasattr(self.fetcher.pro, "adj_factor"):
                        if hasattr(self.fetcher, "_throttle"):
                            self.fetcher._throttle()
                        af = self.fetcher.pro.adj_factor(
                            ts_code=ts_code_csv,
                            start_date=start_date,
                            end_date=date,
                            fields="ts_code,trade_date,adj_factor",
                        )
                    else:
                        af = None
                except Exception:
                    af = None

                if af is not None and not af.empty and "adj_factor" in af.columns:
                    af = af.copy()
                    af["adj_factor"] = np.nan_to_num(
                        np.array(af["adj_factor"], dtype=float), nan=1.0
                    )
                    merged = df.merge(
                        af[["ts_code", "trade_date", "adj_factor"]],
                        on=["ts_code", "trade_date"],
                        how="left",
                    )
                    merged["adj_factor"] = merged["adj_factor"].fillna(1.0)
                    for code, g in merged.groupby("ts_code"):
                        if code not in wanted:
                            continue
                        g = g.sort_values("trade_date").reset_index(drop=True)
                        if len(g) >= 2:
                            prev_close = float(g["close"].iloc[-2] or 0)
                            last_close = float(g["close"].iloc[-1] or 0)
                            if prev_close > 0:
                                day_pct = (last_close / prev_close - 1) * 100
                                if abs(day_pct) > 25:
                                    invalid_count += 1
                                    continue
                        latest_af = float(g["adj_factor"].iloc[-1] or 1.0)
                        if latest_af <= 0:
                            latest_af = 1.0
                        scale = g["adj_factor"].astype(float) / latest_af
                        for col in ("open", "high", "low", "close"):
                            if col in g.columns:
                                g[col] = g[col].astype(float) * scale
                        # 补常用列，兼容后续逻辑
                        if "vol" not in g.columns and "volume" in g.columns:
                            g["vol"] = g["volume"]
                        self._data_cache[(code, days)] = g
                        loaded += 1
                    time.sleep(0.08)
                    continue

                # 补常用列，兼容后续逻辑
                if "vol" not in df.columns and "volume" in df.columns:
                    df["vol"] = df["volume"]
                for code, g in df.groupby("ts_code"):
                    if code not in wanted:
                        continue
                    g = g.sort_values("trade_date").reset_index(drop=True)
                    if len(g) >= 2:
                        prev_close = float(g["close"].iloc[-2] or 0)
                        last_close = float(g["close"].iloc[-1] or 0)
                        if prev_close > 0:
                            day_pct = (last_close / prev_close - 1) * 100
                            if abs(day_pct) > 25:
                                invalid_count += 1
                                continue
                    self._data_cache[(code, days)] = g
                    loaded += 1
                time.sleep(0.08)
            dt = time.time() - t0
            print(f"  ✅ 预加载完成: 缓存{loaded}只 ({dt:.1f}s)")
            if invalid_count > 0:
                print(f"  ⚠️ {invalid_count} 只股票复权数据异常，已排除")
        except Exception as e:
            # 仅告警，不阻断主流程
            print(f"  ⚠️ 预加载失败，回退逐只请求: {e}")
            if len(self._data_cache) == cached_before:
                return

    def _get_stock_name(self, symbol):
        if self.data_cache:
            try:
                return self.data_cache.get_stock_name(symbol)
            except Exception:
                pass
        self._ensure_name_map()
        if self._name_map:
            return self._name_map.get(symbol, symbol)
        if self.fetcher and hasattr(self.fetcher, "get_stock_name"):
            try:
                return self.fetcher.get_stock_name(symbol)
            except Exception:
                return symbol
        return symbol

    def _ensure_name_map(self):
        """加载 ts_code->name 映射（仅首次）。"""
        if self.data_cache:
            return
        if self._name_map is not None:
            return
        if self._name_map is None:
            self._name_map = {}
            if self.fetcher and hasattr(self.fetcher, "pro"):
                try:
                    if hasattr(self.fetcher, "_throttle"):
                        self.fetcher._throttle()
                    df = self.fetcher.pro.stock_basic(
                        list_status="L",
                        fields="ts_code,name",
                    )
                    if df is not None and not df.empty:
                        self._name_map = dict(zip(df["ts_code"], df["name"]))
                        print(f"  📋 股票名称表已加载: {len(self._name_map)} 只")
                except Exception as e:
                    print(f"  ⚠️ 名称表加载失败: {e}")

    def _get_stock_pool(self):
        if self._dynamic_pool is not None:
            return self._dynamic_pool
        if self.fetcher and hasattr(self.fetcher, 'get_stock_pool'):
            try:
                return self.fetcher.get_stock_pool()
            except Exception:
                pass
        return []

    def _get_active_sectors(self, date) -> dict:
        """获取活跃板块及其成分股"""
        if self.fetcher and hasattr(self.fetcher, 'get_sector_stocks'):
            try:
                return self.fetcher.get_sector_stocks(date)
            except Exception:
                pass
        return {}


_validate_param_alignment()
