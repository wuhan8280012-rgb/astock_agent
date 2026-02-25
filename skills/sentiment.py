"""
Skill 1: A股市场情绪监控

监控指标:
  1. 涨跌比（市场宽度）
  2. 涨停/跌停家数
  3. 北向资金流向
  4. 两融余额变化
  5. 成交额水平

输出: 情绪评级 + 建议仓位
"""

from dataclasses import dataclass, field
from typing import Optional, List
import pandas as pd
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_fetcher import get_fetcher

try:
    from config import SKILL_PARAMS
    PARAMS = SKILL_PARAMS.get("sentiment", {})
except ImportError:
    PARAMS = {}


@dataclass
class SentimentSignal:
    """单个情绪信号"""
    name: str
    value: float
    level: str          # "极度贪婪" / "贪婪" / "中性" / "恐慌" / "极度恐慌"
    score: int          # -2 到 +2 (-2=极度恐慌, +2=极度贪婪)
    detail: str = ""


@dataclass
class SentimentReport:
    """情绪评估报告"""
    date: str
    signals: List[SentimentSignal] = field(default_factory=list)
    overall_score: float = 0.0
    overall_level: str = "中性"
    suggested_position: str = ""
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "overall_score": self.overall_score,
            "overall_level": self.overall_level,
            "suggested_position": self.suggested_position,
            "summary": self.summary,
            "signals": [
                {"name": s.name, "value": s.value, "level": s.level, "score": s.score, "detail": s.detail}
                for s in self.signals
            ],
        }

    def to_brief(self) -> str:
        """压缩报告 (~50 tokens)，供 LLM 决策上下文"""
        sig_parts = []
        for s in self.signals[:5]:
            sig_parts.append(f"{s.name}{s.score:+d}")
        sigs = ",".join(sig_parts)
        warn = " ⚠降级" if any("⚠" in (s.detail or "") for s in self.signals) else ""
        return (
            f"[情绪] {self.overall_level}({self.overall_score:+.1f}) "
            f"仓位:{self.suggested_position} | {sigs}{warn}"
        )


class SentimentSkill:
    """A股市场情绪分析"""

    def __init__(self):
        self.fetcher = get_fetcher()

    def analyze(self) -> SentimentReport:
        """
        执行情绪分析 (v6.2: 职责瘦身)

        只保留短期温度计指标:
          1. 涨跌比 (今日) — 市场宽度
          2. 涨跌停 (今日) — 极端情绪
          3. 北向资金 (今日+3日) — 外资情绪

        已移除 (避免与 Macro/ME 重复):
          - 两融余额 → MacroSkill 负责 (中期杠杆水位)
          - 成交额   → MarketEnvironment 负责 (量能维度)
        """
        report = SentimentReport(date=self.fetcher.get_latest_trade_date())

        # 1. 涨跌比分析
        sig = self._analyze_breadth()
        if sig:
            report.signals.append(sig)

        # 2. 涨跌停分析
        sig = self._analyze_limits()
        if sig:
            report.signals.append(sig)

        # 3. 北向资金分析 (短期: 今日+3日)
        sig = self._analyze_north_flow()
        if sig:
            report.signals.append(sig)

        # 综合评分
        self._compute_overall(report)
        return report

    # --------------------------------------------------------
    # 信号1: 涨跌比
    # --------------------------------------------------------
    def _analyze_breadth(self) -> Optional[SentimentSignal]:
        """涨跌家数比"""
        try:
            breadth = self.fetcher.get_market_breadth()
            ratio = breadth["ratio"]

            hot_thresh = PARAMS.get("limit_up_ratio_hot", 3.0)
            panic_thresh = PARAMS.get("limit_up_ratio_panic", 0.33)

            if ratio >= hot_thresh:
                level, score = "极度贪婪", 2
            elif ratio >= 2.0:
                level, score = "贪婪", 1
            elif ratio >= 0.5:
                level, score = "中性", 0
            elif ratio >= panic_thresh:
                level, score = "恐慌", -1
            else:
                level, score = "极度恐慌", -2

            detail = f"上涨{breadth['up']}家 / 下跌{breadth['down']}家 / 平盘{breadth['flat']}家"
            return SentimentSignal(
                name="涨跌比",
                value=ratio,
                level=level,
                score=score,
                detail=detail,
            )
        except Exception as e:
            print(f"[Sentiment] 涨跌比分析失败: {e}")
            return None

    # --------------------------------------------------------
    # 信号2: 涨跌停家数
    # --------------------------------------------------------
    def _analyze_limits(self) -> Optional[SentimentSignal]:
        """涨停/跌停家数分析"""
        try:
            df = self.fetcher.get_limit_list()
            if df.empty:
                return None

            limit_up = len(df[df["limit"] == "U"]) if "limit" in df.columns else 0
            limit_down = len(df[df["limit"] == "D"]) if "limit" in df.columns else 0

            # 有些版本的tushare字段名不同
            if limit_up == 0 and limit_down == 0:
                if "limit_type" in df.columns:
                    limit_up = len(df[df["limit_type"] == "U"])
                    limit_down = len(df[df["limit_type"] == "D"])

            net = limit_up - limit_down

            if net > 50:
                level, score = "极度贪婪", 2
            elif net > 20:
                level, score = "贪婪", 1
            elif net > -20:
                level, score = "中性", 0
            elif net > -50:
                level, score = "恐慌", -1
            else:
                level, score = "极度恐慌", -2

            return SentimentSignal(
                name="涨跌停",
                value=net,
                level=level,
                score=score,
                detail=f"涨停{limit_up}家 / 跌停{limit_down}家",
            )
        except Exception as e:
            print(f"[Sentiment] 涨跌停分析失败: {e}")
            return None

    # --------------------------------------------------------
    # 信号3: 北向资金
    # --------------------------------------------------------
    def _analyze_north_flow(self) -> Optional[SentimentSignal]:
        """资金流向分析（北向→ETF→两融 三层降级）"""
        try:
            try:
                from fund_flow_proxy import FundFlowProxy
                proxy = FundFlowProxy(self.fetcher)
                flow = proxy.get_flow(days=10)
            except Exception:
                flow = None

            if flow is None:
                # 兼容回退：仍尝试旧北向接口
                df = self.fetcher.get_north_flow(days=10)
                if df.empty or "north_money_yi" not in df.columns:
                    return None
                latest = float(df.iloc[-1]["north_money_yi"])
                recent_3d = float(df.tail(3)["north_money_yi"].sum())
                source_tag = "北向"
                degraded = False
                confidence = 1.0
            else:
                if flow.source == "none":
                    return None
                latest = float(flow.latest_flow)
                recent_3d = float(flow.flow_3d)
                source_tag = {
                    "northbound": "北向",
                    "etf_fund_flow": "ETF资金流",
                    "margin_enhanced": "融资替代",
                }.get(flow.source, "?")
                degraded = bool(flow.degraded)
                confidence = float(flow.confidence)

            warn_thresh = PARAMS.get("north_flow_warn_threshold", -50)
            bull_thresh = PARAMS.get("north_flow_bull_threshold", 100)

            if latest > bull_thresh or recent_3d > bull_thresh * 2:
                level, score = "贪婪", 1
            elif latest > 0:
                level, score = "中性偏多", 0
            elif latest > warn_thresh:
                level, score = "中性偏空", 0
            elif recent_3d < warn_thresh * 2:
                level, score = "极度恐慌", -2
            else:
                level, score = "恐慌", -1

            # 降级时按置信度收敛极端值
            if degraded:
                score = int(round(score * confidence))

            return SentimentSignal(
                name="资金流向",
                value=round(latest, 2),
                level=level,
                score=score,
                detail=(
                    f"[{source_tag}] 今日{latest:+.1f}亿 | 近3日{recent_3d:+.1f}亿"
                    f"{' ⚠降级' if degraded else ''}"
                ),
            )
        except Exception as e:
            print(f"[Sentiment] 资金流向分析失败: {e}")
            return None

    # ── 以下方法已废弃(v6.2)，保留代码供回退 ──
    # 两融余额已移交 MacroSkill，成交额已移交 MarketEnvironment
    # --------------------------------------------------------
    # 信号4: 两融余额
    # --------------------------------------------------------
    def _analyze_margin(self) -> Optional[SentimentSignal]:
        """两融余额变化"""
        try:
            df = self.fetcher.get_margin_data(days=10)
            if df.empty or "rzye" not in df.columns:
                return None

            # 融资余额变化率 (5日)
            if len(df) >= 6:
                latest_rz = df.iloc[-1]["rzye"]
                prev_rz = df.iloc[-6]["rzye"]
                change_rate = (latest_rz - prev_rz) / prev_rz if prev_rz > 0 else 0
            else:
                change_rate = 0
                latest_rz = df.iloc[-1]["rzye"] if len(df) > 0 else 0

            warn_rate = PARAMS.get("margin_change_rate_warn", 0.03)

            if change_rate > warn_rate * 2:
                level, score = "极度贪婪", 2
            elif change_rate > warn_rate:
                level, score = "贪婪", 1
            elif change_rate > -warn_rate:
                level, score = "中性", 0
            elif change_rate > -warn_rate * 2:
                level, score = "恐慌", -1
            else:
                level, score = "极度恐慌", -2

            return SentimentSignal(
                name="两融余额",
                value=round(change_rate * 100, 2),
                level=level,
                score=score,
                detail=f"融资余额{latest_rz/1e8:.0f}亿 | 5日变化{change_rate*100:+.2f}%",
            )
        except Exception as e:
            print(f"[Sentiment] 两融分析失败: {e}")
            return None

    # --------------------------------------------------------
    # 信号5: 成交额
    # --------------------------------------------------------
    def _analyze_volume(self) -> Optional[SentimentSignal]:
        """全市场成交额分析"""
        try:
            # 用上证指数成交额代表
            df = self.fetcher.get_index_daily("000001.SH", days=30)
            if df.empty or "amount" not in df.columns:
                return None

            df = df.sort_values("trade_date").reset_index(drop=True)
            latest_amt = df.iloc[-1]["amount"]
            avg_20 = df["amount"].tail(20).mean()
            ratio = latest_amt / avg_20 if avg_20 > 0 else 1

            if ratio > 1.5:
                level, score = "贪婪", 1
                detail_tag = "放量"
            elif ratio > 1.0:
                level, score = "中性偏多", 0
                detail_tag = "温和放量"
            elif ratio > 0.7:
                level, score = "中性偏空", 0
                detail_tag = "温和缩量"
            else:
                level, score = "恐慌", -1
                detail_tag = "显著缩量"

            return SentimentSignal(
                name="成交额",
                value=round(ratio, 2),
                level=level,
                score=score,
                detail=f"{detail_tag} | 今日/20日均值={ratio:.2f}x",
            )
        except Exception as e:
            print(f"[Sentiment] 成交额分析失败: {e}")
            return None

    # --------------------------------------------------------
    # 综合评分
    # --------------------------------------------------------
    def _compute_overall(self, report: SentimentReport):
        """
        综合评分及仓位建议 (v6.2)

        score 语义: +2=极热, -2=极冷 (事实描述)
        仓位建议: 逆向策略 — 越热越谨慎，越冷越积极
        """
        if not report.signals:
            report.overall_level = "数据不足"
            report.summary = "无法获取足够的市场数据来进行情绪评估"
            return

        scores = [s.score for s in report.signals]
        avg_score = np.mean(scores)
        report.overall_score = round(avg_score, 2)

        # 等级 = 事实描述
        if avg_score >= 1.5:
            report.overall_level = "极度贪婪"
        elif avg_score >= 0.5:
            report.overall_level = "贪婪"
        elif avg_score >= -0.5:
            report.overall_level = "中性"
        elif avg_score >= -1.5:
            report.overall_level = "恐慌"
        else:
            report.overall_level = "极度恐慌"

        # 仓位建议 = 逆向策略
        contrarian_position = {
            "极度贪婪": "≤30%仓位(逆向:极热→防守)",
            "贪婪": "50-70%仓位(逆向:偏热→谨慎)",
            "中性": "50%仓位(均衡配置)",
            "恐慌": "60-80%仓位(逆向:偏冷→积极)",
            "极度恐慌": "≥80%仓位(逆向:极冷→进攻)",
        }
        report.suggested_position = contrarian_position.get(
            report.overall_level, "50%仓位(均衡配置)"
        )

        parts = [f"{s.name}:{s.level}({s.score:+d})" for s in report.signals]
        report.summary = " | ".join(parts)


if __name__ == "__main__":
    skill = SentimentSkill()
    result = skill.analyze()
    print(f"\n{'='*60}")
    print(f"  A股情绪监控报告  {result.date}")
    print(f"{'='*60}")
    for sig in result.signals:
        print(f"  [{sig.name}] {sig.level} (评分:{sig.score:+d}) - {sig.detail}")
    print(f"\n  综合评级: {result.overall_level} (得分:{result.overall_score:+.1f})")
    print(f"  仓位建议: {result.suggested_position}")
    print(f"{'='*60}")
