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
from datetime import datetime
from typing import Optional, Any

import numpy as np


class TradeSignalGenerator:
    """
    交易信号生成器

    整合 5 类信号源，统一输出格式。
    """

    def __init__(self, data_fetcher=None, market_env=None,
                 event_log=None, brain=None):
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

        # 1. 市场环境评估（v6.0）
        env_result = None
        if self.market_env:
            try:
                env_result = self.market_env.evaluate(date)
            except Exception as e:
                env_result = {"total_score": 65, "level": "一般",
                              "advice": "谨慎交易", "error": str(e)}

        # 2. 扫描各类信号
        all_signals = []
        by_type = {}

        for scan_name, scan_func in [
            ("ladder_breakout", self._scan_ladder_breakout),
            ("leader_breakout", self._scan_leader_breakout),
            ("canslim", self._scan_canslim),
            ("sector_rotation", self._scan_sector_rotation),
            ("value_undervalued", self._scan_value_undervalued),
        ]:
            try:
                signals = scan_func(date, stock_pool)
                by_type[scan_name] = signals
                all_signals.extend(signals)
            except Exception as e:
                by_type[scan_name] = []
                self._emit_error(scan_name, e)

        # 3. 排序（按 score 降序）
        all_signals.sort(key=lambda s: -s.get("score", 0))

        # 4. 生成摘要
        summary = self._build_summary(date, env_result, all_signals, by_type)

        # 5. EventLog
        self._emit_signals(date, all_signals, env_result)

        return {
            "date": date,
            "market_env": env_result,
            "signals": all_signals,
            "by_type": by_type,
            "summary": summary,
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

        for symbol in stocks:
            try:
                sig = self._detect_ladder_breakout(symbol, date)
                if sig and sig["score"] >= 65:
                    # v6.0 辅助信息
                    sig["auxiliary_info"] = self._get_auxiliary_info(symbol, date)
                    signals.append(sig)
            except Exception:
                continue

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

        # 量能确认
        if len(volume) >= 20:
            avg_vol = sum(volume[-20:]) / 20
            vol_ratio = volume[-1] / avg_vol if avg_vol > 0 else 1.0
        else:
            vol_ratio = 1.0

        if vol_ratio < 1.2:
            return None  # 量能不足

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

        # ── 乖离率保护 ──
        bias = self._calculate_bias(close)
        if bias > 8:
            return None
        if bias > 5:
            score -= 30
        elif bias > 2:
            score -= 10

        score = max(0, min(100, score))

        # 止损位
        stop_loss = c_low * 0.97

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
                "stop_loss": round(stop_loss, 2),
                "target": round(current * 1.10, 2),
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

        for sector_name, sector_stocks in sectors.items():
            try:
                leaders = self._identify_leaders(sector_stocks, date)
                for leader in leaders:
                    sig = self._detect_leader_breakout(leader, sector_name, date)
                    if sig and sig["score"] >= 65:
                        signals.append(sig)
            except Exception:
                continue

        return sorted(signals, key=lambda s: -s["score"])

    def _identify_leaders(self, stocks: list, date: str) -> list:
        """识别板块龙头（涨幅+市值+换手率）"""
        scored = []
        for symbol in stocks:
            try:
                df = self._get_stock_data(symbol, date, days=20)
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

        # Top3 by 涨幅
        scored.sort(key=lambda x: -x["pct_20d"])
        return [s["symbol"] for s in scored[:3]]

    def _detect_leader_breakout(self, symbol: str, sector: str, date: str) -> Optional[dict]:
        """检测龙头突破信号"""
        df = self._get_stock_data(symbol, date, days=60)
        if df is None or len(df) < 30:
            return None

        close = df["close"].values
        high = df["high"].values
        current = close[-1]

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
        if current >= high_60d:
            score += 10  # 创新高
        if pullback > 8:
            score += 5   # 充分调整
        # 量能
        if "vol" in df.columns and len(df) >= 20:
            vol_ratio = df["vol"].iloc[-1] / df["vol"].iloc[-20:].mean()
            if vol_ratio > 1.5:
                score += 10

        # ── 乖离率保护 ──
        bias = self._calculate_bias(close)
        if bias > 8:
            return None
        if bias > 5:
            score -= 30
        elif bias > 2:
            score -= 10

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
                "stop_loss": round(min_20d * 0.97, 2),
                "target": round(current * 1.10, 2),
            },
            "bias": round(bias, 2),
            "confidence": round(max(0, min(100, score)) / 100, 2),
            "reasons": [
                f"板块龙头({sector})",
                f"回调{pullback:.1f}%后突破",
                "创60日新高" if current >= high_60d else "接近60日新高",
            ],
            "risk_notes": self._get_risk_notes(symbol, current, vol_ratio if "vol_ratio" in locals() else 1.0, bias=bias),
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
            pass

        # 如果 Skill 不可用，返回空
        return []

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
            pass
        return []

    # ═══════════════════════════════════════════
    # 信号类型 5: 价值低估
    # ═══════════════════════════════════════════

    def _scan_value_undervalued(self, date: str, pool: list = None) -> list:
        """
        价值低估信号

        委托给 ValueInvestorSkill
        """
        try:
            from value_investor import ValueInvestorSkill
            skill = ValueInvestorSkill(self.fetcher)
            return self._apply_bias_guard_signals(skill.scan(date), date)
        except ImportError:
            pass
        return []

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

    # ═══════════════════════════════════════════
    # v6.0 辅助信息
    # ═══════════════════════════════════════════

    def _get_auxiliary_info(self, symbol: str, date: str) -> dict:
        """
        v6.0 辅助信息（不影响选股，仅供参考）
        """
        info = {}
        try:
            df = self._get_stock_data(symbol, date, days=60)
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
        """
        决策仪表盘输出（用于终端/推送）
        """
        date_str = result.get("date") or datetime.now().strftime("%Y%m%d")
        try:
            date_fmt = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        except Exception:
            date_fmt = date_str

        signals = result.get("signals", []) or []
        classified = []
        for sig in signals:
            level = self._classify_signal_level(sig, env_score)
            classified.append((level, sig))

        buy_count = sum(1 for level, _ in classified if level == "buy")
        watch_count = sum(1 for level, _ in classified if level == "watch")
        avoid_count = sum(1 for level, _ in classified if level == "avoid")

        env_band = "较差-观望" if env_score < 60 else "一般-谨慎" if env_score < 75 else "良好-可交易"
        lines = [
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"📊 {date_fmt} 信号仪表盘",
            f"环境: {env_score}/100 ({env_band}) | 命中: {len(signals)}只 | 🟢买入:{buy_count} 🟡观望:{watch_count} 🔴回避:{avoid_count}",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "",
        ]

        level_map = {
            "buy": ("🟢 买入", 0),
            "watch": ("🟡 观望", 1),
            "avoid": ("🔴 回避", 2),
        }
        sorted_items = sorted(classified, key=lambda x: (level_map[x[0]][1], -float(x[1].get("score", 0) or 0)))

        for level, sig in sorted_items:
            level_text = level_map[level][0]
            symbol = sig.get("symbol", "?")
            name = sig.get("name", symbol)
            sig_type = self._friendly_signal_type(sig.get("type", ""))
            score = int(sig.get("score", 0) or 0)
            bias = float(sig.get("bias", 0) or 0)
            lines.append(f"{level_text} | {name}({symbol}) | {sig_type} {score}分")
            lines.append(f"  📌 {self._build_core_conclusion(sig, bias)}")

            p = sig.get("price", {}) or {}
            close = float(p.get("close", 0) or 0)
            stop_loss = float(p.get("stop_loss", 0) or 0)
            target = float(p.get("target", close * 1.10 if close > 0 else 0) or 0)
            stop_pct = ((stop_loss / close - 1) * 100) if close > 0 and stop_loss > 0 else 0
            target_pct = ((target / close - 1) * 100) if close > 0 and target > 0 else 0
            lines.append(
                f"  💰 买入{close:.2f} | 止损{stop_loss:.2f}({stop_pct:+.1f}%) | 目标{target:.2f}({target_pct:+.1f}%)"
            )

            checklist = self._build_checklist(sig, env_score, bias)
            if checklist:
                lines.append(f"  {' '.join(checklist)}")

            news = sig.get("news") or {}
            if news:
                summary = news.get("summary", "无近期新闻")
                adj = int(news.get("score_adj", 0) or 0)
                mark = f"+{adj}" if adj > 0 else str(adj)
                suffix = f" → {'⚠️' if adj < 0 else ''}{mark}分" if adj != 0 else ""
                lines.append(f"  📰 近期: {summary}{suffix}")

            lines.append("")

        lines.extend([
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"💡 建议: {self._env_advice(env_score)}",
        ])
        return "\n".join(lines)

    def _classify_signal_level(self, sig: dict, env_score: int) -> str:
        score = float(sig.get("score", 0) or 0)
        bias = float(sig.get("bias", 0) or 0)
        if bias > 8 or score < 65:
            level = "avoid"
        elif score >= 75 and env_score >= 65 and bias < 5:
            level = "buy"
        elif score >= 65:
            level = "watch"
        else:
            level = "avoid"
        if env_score < 60 and level == "buy":
            level = "watch"
        return level

    def _build_core_conclusion(self, sig: dict, bias: float) -> str:
        reasons = sig.get("reasons", []) or []
        shape = reasons[0] if reasons else self._friendly_signal_type(sig.get("type", ""))
        if bias < 2:
            bias_view = f"乖离率{bias:.1f}%，最佳买点"
        elif bias < 5:
            bias_view = f"乖离率{bias:.1f}%，可谨慎参与"
        elif bias <= 8:
            bias_view = f"乖离率{bias:.1f}%，追高风险偏高"
        else:
            bias_view = f"乖离率{bias:.1f}%，建议回避"
        return f"{shape}，{bias_view}"

    def _build_checklist(self, sig: dict, env_score: int, bias: float) -> list:
        checks = []
        reasons = " ".join(sig.get("reasons", []) or [])
        risks = " ".join(sig.get("risk_notes", []) or [])

        if "阶梯整理" in reasons:
            checks.append(next((f"✅ {r}" for r in sig.get("reasons", []) if "阶梯整理" in r), "✅ 阶梯形态"))
        elif "龙头" in reasons:
            checks.append("✅ 板块龙头")

        vol_reason = next((r for r in (sig.get("reasons") or []) if "量比" in r), "")
        if vol_reason:
            checks.append(f"✅ {vol_reason}")
        elif "量能不足" in risks:
            checks.append("❌ 成交缩量")
        else:
            checks.append("⚠️ 量能未明")

        if bias > 8:
            checks.append(f"❌ 乖离率{bias:.1f}%")
        elif bias > 5:
            checks.append(f"⚠️ 乖离率{bias:.1f}%")
        elif bias > 2:
            checks.append(f"⚠️ 乖离率{bias:.1f}%")
        else:
            checks.append(f"✅ 乖离率{bias:.1f}%")

        if env_score < 60:
            checks.append(f"❌ 环境{env_score}分较差")
        elif env_score < 75:
            checks.append(f"⚠️ 环境{env_score}分一般")
        else:
            checks.append(f"✅ 环境{env_score}分")
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

    def _get_stock_data(self, symbol, date, days=60):
        if not self.fetcher:
            return None
        for method in ['get_daily', 'get_stock_daily', 'get_k_data']:
            if hasattr(self.fetcher, method):
                try:
                    return getattr(self.fetcher, method)(symbol, days=days, end_date=date)
                except Exception:
                    continue
        return None

    def _get_stock_name(self, symbol):
        if self.fetcher and hasattr(self.fetcher, 'get_stock_name'):
            try:
                return self.fetcher.get_stock_name(symbol)
            except Exception:
                pass
        return symbol

    def _get_stock_pool(self):
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
