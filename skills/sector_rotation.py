"""
Skill 2: 板块轮动排名

逻辑:
  1. 计算所有申万一级行业的相对强度 (20日/60日)
  2. 综合排名，选出Top N强势板块
  3. 识别"强趋势+缩量整理"的板块（即将发动的信号）
  4. 识别"弱转强"的板块（轮动切换信号）

输出: 板块排名表 + 推荐关注板块
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
    PARAMS = SKILL_PARAMS.get("sector_rotation", {})
except ImportError:
    PARAMS = {}


@dataclass
class SectorInfo:
    """板块信息"""
    ts_code: str
    name: str
    close: float = 0
    ret_5d: float = 0       # 5日涨跌幅
    ret_20d: float = 0      # 20日涨跌幅
    ret_60d: float = 0      # 60日涨跌幅
    vol_ratio: float = 1.0  # 5日/20日量比
    composite_score: float = 0  # 综合得分
    signal: str = ""        # 信号: "强势持续" / "缩量整理" / "弱转强" / "高位放量"
    rank: int = 0


@dataclass
class SectorReport:
    """板块轮动报告"""
    date: str
    sectors: List[SectorInfo] = field(default_factory=list)
    top_sectors: List[SectorInfo] = field(default_factory=list)
    rotation_signals: List[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "top_sectors": [
                {"name": s.name, "rank": s.rank, "score": s.composite_score,
                 "ret_20d": s.ret_20d, "signal": s.signal}
                for s in self.top_sectors
            ],
            "rotation_signals": self.rotation_signals,
            "summary": self.summary,
        }


class SectorRotationSkill:
    """板块轮动分析"""

    def __init__(self):
        self.fetcher = get_fetcher()
        self.top_n = PARAMS.get("top_n_sectors", 5)

    def analyze(self) -> SectorReport:
        """执行板块轮动分析"""
        report = SectorReport(date=self.fetcher.get_latest_trade_date())

        # 1. 拉取所有板块数据
        perf_df = self.fetcher.get_all_sector_performance(days=120)
        if perf_df.empty:
            report.summary = "无法获取板块数据"
            return report

        # 2. 计算综合得分
        sectors = self._compute_scores(perf_df)
        report.sectors = sectors

        # 3. 取Top N
        sorted_sectors = sorted(sectors, key=lambda x: x.composite_score, reverse=True)
        for i, s in enumerate(sorted_sectors):
            s.rank = i + 1
        report.top_sectors = sorted_sectors[:self.top_n]

        # 4. 识别轮动信号
        report.rotation_signals = self._detect_signals(sorted_sectors)

        # 5. 生成摘要
        top_names = [f"{s.name}({s.ret_20d:+.1f}%)" for s in report.top_sectors]
        report.summary = f"强势板块: {', '.join(top_names)}"
        if report.rotation_signals:
            report.summary += f" | 信号: {'; '.join(report.rotation_signals)}"

        return report

    def _compute_scores(self, df: pd.DataFrame) -> List[SectorInfo]:
        """计算板块综合得分"""
        sectors = []

        # 对涨跌幅做百分位排名 (0~100)
        for col in ["ret_5d", "ret_20d", "ret_60d"]:
            if col in df.columns:
                df[f"{col}_rank"] = df[col].rank(pct=True) * 100

        for _, row in df.iterrows():
            # 综合得分 = 短期40% + 中期40% + 长期20%
            score = (
                row.get("ret_5d_rank", 50) * 0.30 +
                row.get("ret_20d_rank", 50) * 0.40 +
                row.get("ret_60d_rank", 50) * 0.30
            )

            # 判断信号
            signal = self._classify_signal(
                ret_5d=row["ret_5d"],
                ret_20d=row["ret_20d"],
                ret_60d=row["ret_60d"],
                vol_ratio=row["vol_ratio"],
            )

            sectors.append(SectorInfo(
                ts_code=row["ts_code"],
                name=row["name"],
                close=row["close"],
                ret_5d=row["ret_5d"],
                ret_20d=row["ret_20d"],
                ret_60d=row["ret_60d"],
                vol_ratio=row["vol_ratio"],
                composite_score=round(score, 1),
                signal=signal,
            ))

        return sectors

    def _classify_signal(self, ret_5d, ret_20d, ret_60d, vol_ratio) -> str:
        """判断板块状态信号"""
        vol_shrink = PARAMS.get("volume_shrink_ratio", 0.6)

        # 强趋势 + 缩量整理 = 蓄势待发
        if ret_20d > 5 and ret_60d > 10 and abs(ret_5d) < 2 and vol_ratio < vol_shrink:
            return "⭐ 缩量整理(关注突破)"

        # 中期强 + 短期弱转强
        if ret_20d > 5 and ret_5d > 3:
            return "🔥 强势持续"

        # 中期弱 + 短期反弹明显
        if ret_20d < 0 and ret_5d > 3:
            return "🔄 弱转强(轮动信号)"

        # 高位放量回落
        if ret_60d > 15 and ret_5d < -3 and vol_ratio > 1.5:
            return "⚠️ 高位放量(警惕)"

        # 持续弱势
        if ret_20d < -5 and ret_60d < -10:
            return "❌ 弱势回避"

        return "— 观望"

    def _detect_signals(self, sorted_sectors: List[SectorInfo]) -> List[str]:
        """检测板块轮动信号"""
        signals = []

        # 找出"缩量整理"的强势板块
        consolidating = [s for s in sorted_sectors[:10]
                         if "缩量整理" in s.signal]
        if consolidating:
            names = [s.name for s in consolidating]
            signals.append(f"缩量蓄势关注: {', '.join(names)}")

        # 找出"弱转强"的板块
        turning = [s for s in sorted_sectors
                   if "弱转强" in s.signal]
        if turning:
            names = [s.name for s in turning[:3]]
            signals.append(f"轮动切换信号: {', '.join(names)}")

        # 找出需要警惕的板块
        warning = [s for s in sorted_sectors[:10]
                   if "高位放量" in s.signal]
        if warning:
            names = [s.name for s in warning]
            signals.append(f"高位警惕: {', '.join(names)}")

        return signals


if __name__ == "__main__":
    skill = SectorRotationSkill()
    result = skill.analyze()
    print(f"\n{'='*70}")
    print(f"  板块轮动排名  {result.date}")
    print(f"{'='*70}")
    print(f"{'排名':<4} {'板块':<8} {'5日':<8} {'20日':<8} {'60日':<8} {'量比':<6} {'得分':<6} {'信号'}")
    print("-" * 70)
    for s in sorted(result.sectors, key=lambda x: x.composite_score, reverse=True)[:15]:
        print(f"{s.rank:<4} {s.name:<8} {s.ret_5d:>+6.2f}% {s.ret_20d:>+6.2f}% "
              f"{s.ret_60d:>+6.2f}% {s.vol_ratio:>5.2f} {s.composite_score:>5.1f}  {s.signal}")
    print(f"\n{'='*70}")
    print(f"  {result.summary}")
    for sig in result.rotation_signals:
        print(f"  → {sig}")
    print(f"{'='*70}")
