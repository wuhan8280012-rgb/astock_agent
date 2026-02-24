"""
Skill 3: 宏观流动性监控 (A股版)

A股流动性核心指标:
  1. SHIBOR (上海银行间同业拆放利率) - 类比美股的 SOFR
  2. 北向资金趋势 - 外资风向标
  3. 央行公开市场操作 - MLF/逆回购净投放
  4. 人民币汇率 - 资金流向指标
  5. 两融余额趋势 - 杠杆资金水位

输出: 流动性评级 + 对市场的影响判断
"""

from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_fetcher import get_fetcher

try:
    from config import SKILL_PARAMS
    PARAMS = SKILL_PARAMS.get("macro", {})
except ImportError:
    PARAMS = {}


@dataclass
class MacroSignal:
    name: str
    value: float
    trend: str      # "收紧" / "宽松" / "中性"
    score: int       # -2 ~ +2
    detail: str = ""


@dataclass
class MacroReport:
    date: str
    signals: List[MacroSignal] = field(default_factory=list)
    overall_score: float = 0
    liquidity_level: str = "中性"
    market_impact: str = ""
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "overall_score": self.overall_score,
            "liquidity_level": self.liquidity_level,
            "market_impact": self.market_impact,
            "summary": self.summary,
            "signals": [
                {"name": s.name, "value": s.value, "trend": s.trend, "score": s.score, "detail": s.detail}
                for s in self.signals
            ],
        }


class MacroSkill:
    """宏观流动性监控"""

    def __init__(self):
        self.fetcher = get_fetcher()

    def analyze(self) -> MacroReport:
        report = MacroReport(date=self.fetcher.get_latest_trade_date())

        # 1. SHIBOR 分析
        sig = self._analyze_shibor()
        if sig:
            report.signals.append(sig)

        # 2. 北向资金趋势
        sig = self._analyze_north_trend()
        if sig:
            report.signals.append(sig)

        # 3. 两融趋势
        sig = self._analyze_margin_trend()
        if sig:
            report.signals.append(sig)

        # 4. 市场成交额趋势
        sig = self._analyze_turnover_trend()
        if sig:
            report.signals.append(sig)

        # 综合
        self._compute_overall(report)
        return report

    def _analyze_shibor(self) -> Optional[MacroSignal]:
        """SHIBOR 利率分析"""
        try:
            df = self.fetcher.get_shibor(days=30)
            if df.empty:
                return None

            df = df.sort_values("date").reset_index(drop=True)

            # 隔夜SHIBOR
            latest_on = df.iloc[-1].get("on", None)  # overnight
            # 1周SHIBOR
            latest_1w = df.iloc[-1].get("1w", None)

            if latest_on is None and latest_1w is None:
                return None

            rate = latest_1w if latest_1w is not None else latest_on
            rate_name = "1周SHIBOR" if latest_1w is not None else "隔夜SHIBOR"

            # 计算趋势 (最近5日 vs 前5日)
            if len(df) >= 10:
                col = "1w" if latest_1w is not None else "on"
                recent = df[col].tail(5).mean()
                prev = df[col].iloc[-10:-5].mean()
                trend_change = recent - prev
            else:
                trend_change = 0

            warn = PARAMS.get("shibor_warn", 3.0)

            if rate > warn and trend_change > 0.1:
                trend, score = "收紧", -2
            elif rate > warn:
                trend, score = "偏紧", -1
            elif trend_change < -0.1:
                trend, score = "宽松", 1
            elif trend_change < -0.3:
                trend, score = "明显宽松", 2
            else:
                trend, score = "中性", 0

            return MacroSignal(
                name="SHIBOR",
                value=round(rate, 4),
                trend=trend,
                score=score,
                detail=f"{rate_name}={rate:.4f}% | 趋势变化{trend_change:+.3f}%",
            )
        except Exception as e:
            print(f"[Macro] SHIBOR分析失败: {e}")
            return None

    def _analyze_north_trend(self) -> Optional[MacroSignal]:
        """北向资金中期趋势（区别于情绪Skill的短期分析）"""
        try:
            df = self.fetcher.get_north_flow(days=20)
            if df.empty or "north_money_yi" not in df.columns:
                return None

            # 累计净流入
            total_20d = df["north_money_yi"].sum()
            # 近5日 vs 前5日
            recent_5d = df.tail(5)["north_money_yi"].sum()
            prev_5d = df.iloc[-10:-5]["north_money_yi"].sum() if len(df) >= 10 else 0

            if total_20d > 200:
                trend, score = "持续流入", 2
            elif total_20d > 50:
                trend, score = "温和流入", 1
            elif total_20d > -50:
                trend, score = "中性", 0
            elif total_20d > -200:
                trend, score = "温和流出", -1
            else:
                trend, score = "持续流出", -2

            return MacroSignal(
                name="北向资金趋势",
                value=round(total_20d, 2),
                trend=trend,
                score=score,
                detail=f"20日累计{total_20d:+.1f}亿 | 近5日{recent_5d:+.1f}亿 vs 前5日{prev_5d:+.1f}亿",
            )
        except Exception as e:
            print(f"[Macro] 北向趋势分析失败: {e}")
            return None

    def _analyze_margin_trend(self) -> Optional[MacroSignal]:
        """两融余额趋势"""
        try:
            df = self.fetcher.get_margin_data(days=20)
            if df.empty or "rzye" not in df.columns:
                return None

            df = df.sort_values("trade_date").reset_index(drop=True)

            latest = df.iloc[-1]["rzye"]
            if len(df) >= 20:
                start = df.iloc[0]["rzye"]
                change_pct = (latest - start) / start * 100
            else:
                change_pct = 0

            # 两融余额水位判断
            latest_yi = latest / 1e8  # 转为亿

            if change_pct > 5:
                trend, score = "杠杆加速上升", 1  # 短期利好但要警惕
            elif change_pct > 0:
                trend, score = "杠杆温和上升", 0
            elif change_pct > -5:
                trend, score = "杠杆小幅回落", 0
            else:
                trend, score = "去杠杆", -1

            return MacroSignal(
                name="两融余额",
                value=round(latest_yi, 0),
                trend=trend,
                score=score,
                detail=f"融资余额{latest_yi:.0f}亿 | 20日变化{change_pct:+.2f}%",
            )
        except Exception as e:
            print(f"[Macro] 两融趋势分析失败: {e}")
            return None

    def _analyze_turnover_trend(self) -> Optional[MacroSignal]:
        """全市场成交额趋势"""
        try:
            df = self.fetcher.get_index_daily("000001.SH", days=30)
            if df.empty or "amount" not in df.columns:
                return None

            df = df.sort_values("trade_date").reset_index(drop=True)

            # 5日均量 / 20日均量
            avg_5 = df["amount"].tail(5).mean()
            avg_20 = df["amount"].tail(20).mean()
            ratio = avg_5 / avg_20 if avg_20 > 0 else 1

            # 成交额绝对水平(上证, 单位亿)
            latest_amount = df.iloc[-1]["amount"] / 1e4  # tushare单位是千元

            if ratio > 1.3:
                trend, score = "放量", 1
            elif ratio > 0.9:
                trend, score = "正常", 0
            elif ratio > 0.7:
                trend, score = "缩量", -1
            else:
                trend, score = "极度缩量", -2

            return MacroSignal(
                name="成交额趋势",
                value=round(ratio, 2),
                trend=trend,
                score=score,
                detail=f"5日/20日量比={ratio:.2f} | 今日上证成交{latest_amount:.0f}亿",
            )
        except Exception as e:
            print(f"[Macro] 成交额趋势分析失败: {e}")
            return None

    def _compute_overall(self, report: MacroReport):
        if not report.signals:
            report.liquidity_level = "数据不足"
            return

        scores = [s.score for s in report.signals]
        avg = np.mean(scores)
        report.overall_score = round(avg, 2)

        if avg >= 1.0:
            report.liquidity_level = "宽松"
            report.market_impact = "流动性充裕，利好风险资产，可适度进攻"
        elif avg >= 0:
            report.liquidity_level = "中性偏松"
            report.market_impact = "流动性尚可，市场有支撑，维持均衡"
        elif avg >= -1.0:
            report.liquidity_level = "中性偏紧"
            report.market_impact = "流动性有所收紧，注意仓位管理"
        else:
            report.liquidity_level = "紧张"
            report.market_impact = "流动性紧张，防御为主，控制仓位"

        parts = [f"{s.name}:{s.trend}" for s in report.signals]
        report.summary = " | ".join(parts)


if __name__ == "__main__":
    skill = MacroSkill()
    result = skill.analyze()
    print(f"\n{'='*60}")
    print(f"  宏观流动性监控  {result.date}")
    print(f"{'='*60}")
    for sig in result.signals:
        print(f"  [{sig.name}] {sig.trend} (评分:{sig.score:+d}) - {sig.detail}")
    print(f"\n  流动性评级: {result.liquidity_level} (得分:{result.overall_score:+.1f})")
    print(f"  市场影响: {result.market_impact}")
    print(f"{'='*60}")
