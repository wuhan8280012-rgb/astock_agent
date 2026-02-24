"""
Skill 6: 板块轮动 + CANSLIM 联合选股流水线

完整流程:
  1. SectorRotation → 筛出 Top N 强势板块
  2. 获取板块成分股
  3. CANSLIM 逐只评分
  4. 叠加"缩量整理突破"技术形态过滤
  5. 结合风控参数 → 输出最终买入候选

这是你原有策略的升级整合版:
  - 板块轮动 (你的 sector rotation 项目)
  - CANSLIM (你的 CANSLIM 回测项目)
  - 缩量整理 (你的 consolidation pattern 模块)
  合为一个完整的选股 pipeline
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

import pandas as pd
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_fetcher import get_fetcher
from skills.sector_rotation import SectorRotationSkill
from skills.canslim_screener import CanslimScreener
from skills.sentiment import SentimentSkill

try:
    from config import SKILL_PARAMS
except ImportError:
    SKILL_PARAMS = {}


@dataclass
class ConsolidationPattern:
    """缩量整理形态"""
    ts_code: str
    name: str
    pattern_type: str = ""    # "窄幅震荡" / "三角收敛" / "箱体整理"
    consolidation_days: int = 0
    volume_shrink_ratio: float = 0  # 当前量 / 整理前量
    range_pct: float = 0      # 整理区间振幅 (%)
    near_breakout: bool = False
    breakout_price: float = 0
    detail: str = ""


@dataclass
class PipelineCandidate:
    """流水线输出的候选股"""
    ts_code: str
    name: str
    sector: str
    canslim_grade: str
    canslim_score: float
    consolidation: Optional[ConsolidationPattern] = None
    sector_rank: int = 0
    sector_signal: str = ""
    final_score: float = 0
    buy_signal_strength: str = ""  # "强" / "中" / "弱"
    suggested_entry: float = 0
    suggested_stop: float = 0
    flags: List[str] = field(default_factory=list)


@dataclass
class PipelineReport:
    """流水线报告"""
    date: str
    market_sentiment: str = ""
    top_sectors: List[str] = field(default_factory=list)
    candidates: List[PipelineCandidate] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "market_sentiment": self.market_sentiment,
            "top_sectors": self.top_sectors,
            "summary": self.summary,
            "candidates": [
                {
                    "ts_code": c.ts_code, "name": c.name,
                    "sector": c.sector, "grade": c.canslim_grade,
                    "canslim_score": c.canslim_score,
                    "final_score": c.final_score,
                    "signal_strength": c.buy_signal_strength,
                    "entry": c.suggested_entry, "stop": c.suggested_stop,
                    "has_consolidation": c.consolidation is not None,
                    "flags": c.flags,
                }
                for c in self.candidates
            ],
        }


class StockPipeline:
    """联合选股流水线"""

    def __init__(self):
        self.fetcher = get_fetcher()
        self.sector_skill = SectorRotationSkill()
        self.canslim = CanslimScreener()
        self.consolidation_days = SKILL_PARAMS.get("sector_rotation", {}).get("consolidation_days", 10)
        self.volume_shrink = SKILL_PARAMS.get("sector_rotation", {}).get("volume_shrink_ratio", 0.6)

    def run(self, top_n_result: int = 10) -> PipelineReport:
        """执行完整选股流水线"""
        date = self.fetcher.get_latest_trade_date()
        report = PipelineReport(date=date)

        # ---- Stage 1: 市场情绪 ----
        print("[Pipeline Stage 1/5] 分析市场情绪...")
            sentiment = SentimentSkill().analyze()
            report.market_sentiment = sentiment.overall_level

            if sentiment.overall_level in ["极度恐慌", "恐慌"]:
                report.summary = f"市场情绪={sentiment.overall_level}，当前不适合选股，建议观望"
                return report

            # ---- Stage 2: 板块轮动 ----
            if verbose:
                print("[Pipeline Stage 2/5] 运行板块轮动...")
            sector_result = self.sector_skill.analyze()
            top_sectors = sector_result.top_sectors[:5]
            report.top_sectors = [s.name for s in top_sectors]

            if not top_sectors:
                report.summary = "未找到强势板块"
                return report

            if verbose:
                print(f"  强势板块: {', '.join(s.name for s in top_sectors)}")

            # ---- Stage 3: 获取板块成分股 ----
            if verbose:
                print("[Pipeline Stage 3/5] 获取板块成分股...")
            sector_stock_map = {}  # ts_code → sector_name
            stock_pool = []

            for sector_info in top_sectors:
                try:
                    members = self.fetcher.pro.index_member(index_code=sector_info.ts_code)
                    if members is not None and not members.empty:
                        codes = members["con_code"].tolist()
                        for code in codes:
                            sector_stock_map[code] = sector_info.name
                        stock_pool.extend(codes)
                        if verbose:
                            print(f"  {sector_info.name}: {len(codes)}只成分股")
                except Exception as e:
                    if verbose:
                        print(f"  {sector_info.name}: 获取成分股失败 - {e}")
                time.sleep(0.2)

            stock_pool = list(set(stock_pool))
            if verbose:
                print(f"  合计候选池: {len(stock_pool)}只")

            if not stock_pool:
                report.summary = "无法获取板块成分股"
                return report

            # ---- Stage 4: CANSLIM 筛选 ----
            if verbose:
                print("[Pipeline Stage 4/5] CANSLIM 评分...")
            canslim_result = self.canslim.screen(
                stock_pool=stock_pool[:100],  # 限制数量
                top_n=30,
                market_sentiment=sentiment.overall_level,
            )

            if not canslim_result.candidates:
                report.summary = "CANSLIM筛选后无候选"
                return report

            if verbose:
                print(f"  CANSLIM通过: {len(canslim_result.candidates)}只")

            # ---- Stage 5: 缩量整理过滤 + 最终排名 ----
            if verbose:
                print("[Pipeline Stage 5/5] 技术形态过滤...")
            candidates = []

            for stock in canslim_result.candidates:
                # 检查缩量整理形态
                pattern = self._check_consolidation(stock.ts_code)

                sector_name = sector_stock_map.get(stock.ts_code, "")
                sector_rank = 0
                sector_signal = ""
                for i, s in enumerate(top_sectors):
                    if s.name == sector_name:
                        sector_rank = i + 1
                        sector_signal = s.signal
                        break

                # 计算最终得分
                final = stock.total_score

                # 缩量整理加分
                if pattern and pattern.near_breakout:
                    final += 15
                    stock.flags.append("临近突破")
                elif pattern:
                    final += 8
                    stock.flags.append("缩量整理")

                # 板块排名加分
                if sector_rank <= 2:
                    final += 10
                elif sector_rank <= 4:
                    final += 5

                # 计算建议入场价和止损价
                entry_price, stop_price = self._calc_entry_stop(stock.ts_code)

                # 信号强度
                if final >= 80 and pattern and pattern.near_breakout:
                    strength = "强"
                elif final >= 65:
                    strength = "中"
                else:
                    strength = "弱"

                candidates.append(PipelineCandidate(
                    ts_code=stock.ts_code,
                    name=stock.name,
                    sector=sector_name,
                    canslim_grade=stock.grade,
                    canslim_score=stock.total_score,
                    consolidation=pattern,
                    sector_rank=sector_rank,
                    sector_signal=sector_signal,
                    final_score=round(final, 1),
                    buy_signal_strength=strength,
                    suggested_entry=entry_price,
                    suggested_stop=stop_price,
                    flags=stock.flags,
                ))

            # 排序
            candidates.sort(key=lambda x: x.final_score, reverse=True)
            report.candidates = candidates[:top_n_result]

            strong = sum(1 for c in report.candidates if c.buy_signal_strength == "强")
            medium = sum(1 for c in report.candidates if c.buy_signal_strength == "中")
            report.summary = (
                f"情绪:{sentiment.overall_level} | "
                f"强势板块:{','.join(report.top_sectors[:3])} | "
                f"候选{len(report.candidates)}只 (强信号{strong} 中信号{medium})"
            )

            return report
        finally:
            if as_of_date:
                self.fetcher.clear_as_of_date()

    def _check_consolidation(self, ts_code: str) -> Optional[ConsolidationPattern]:
        """检查是否处于缩量整理形态"""
        try:
            df = self.fetcher.get_stock_daily(ts_code, days=40)
            if df.empty or len(df) < 20:
                return None

            df = df.sort_values("trade_date").reset_index(drop=True)

            # 最近 consolidation_days 的数据
            n = self.consolidation_days
            recent = df.tail(n)
            prev = df.iloc[-(n + 10):-n] if len(df) >= n + 10 else df.head(10)

            if recent.empty or prev.empty:
                return None

            # 条件1: 量缩
            vol_recent = recent["vol"].mean()
            vol_prev = prev["vol"].mean()
            vol_ratio = vol_recent / vol_prev if vol_prev > 0 else 1

            # 条件2: 价格区间收窄
            high_n = recent["high"].max()
            low_n = recent["low"].min()
            range_pct = (high_n - low_n) / low_n * 100 if low_n > 0 else 999

            # 判断是否为缩量整理
            is_consolidation = (vol_ratio < self.volume_shrink) and (range_pct < 10)

            if not is_consolidation:
                return None

            # 判断是否临近突破
            latest_close = df.iloc[-1]["close"]
            near_breakout = (latest_close >= high_n * 0.97)

            # 分类
            if range_pct < 5:
                ptype = "窄幅震荡"
            elif range_pct < 8:
                ptype = "箱体整理"
            else:
                ptype = "三角收敛"

            return ConsolidationPattern(
                ts_code=ts_code,
                name="",
                pattern_type=ptype,
                consolidation_days=n,
                volume_shrink_ratio=round(vol_ratio, 2),
                range_pct=round(range_pct, 2),
                near_breakout=near_breakout,
                breakout_price=round(high_n, 2),
                detail=f"{ptype} | 量比{vol_ratio:.2f} | 振幅{range_pct:.1f}% | "
                       f"{'临近突破' if near_breakout else '整理中'}",
            )

        except Exception:
            return None

    def _calc_entry_stop(self, ts_code: str) -> tuple:
        """计算建议入场价和止损价"""
        try:
            df = self.fetcher.get_stock_daily(ts_code, days=30)
            if df.empty:
                return 0, 0

            df = df.sort_values("trade_date")
            latest = df.iloc[-1]

            # 入场价: 当前价 (或突破价)
            entry = round(float(latest["close"]), 2)

            # 止损价: ATR 法
            if len(df) >= 14:
                # 简化ATR: 14日平均真实波幅
                df["tr"] = np.maximum(
                    df["high"] - df["low"],
                    np.maximum(
                        abs(df["high"] - df["close"].shift(1)),
                        abs(df["low"] - df["close"].shift(1))
                    )
                )
                atr = df["tr"].tail(14).mean()
                stop = round(entry - 2 * atr, 2)
            else:
                stop = round(entry * 0.95, 2)  # 默认5%止损

            return entry, stop

        except Exception:
            return 0, 0


if __name__ == "__main__":
    pipeline = StockPipeline()
    result = pipeline.run(top_n_result=10)

    print(f"\n{'='*80}")
    print(f"  📊 联合选股流水线  {result.date}")
    print(f"  {result.summary}")
    print(f"{'='*80}")

    if result.candidates:
        print(f"\n{'信号':<4} {'代码':<12} {'名称':<8} {'板块':<8} {'CANSLIM':<8} "
              f"{'总分':<6} {'入场':<8} {'止损':<8} {'标记'}")
        print("-" * 80)
        for c in result.candidates:
            strength_icon = {"强": "🔥", "中": "⭐", "弱": "○"}.get(c.buy_signal_strength, "")
            flags_str = " ".join(c.flags[:3])
            print(f"  {strength_icon}{c.buy_signal_strength}  {c.ts_code:<12} {c.name:<8} "
                  f"{c.sector:<8} {c.canslim_grade}({c.canslim_score:.0f})  "
                  f"{c.final_score:>5.1f}  {c.suggested_entry:>7.2f} "
                  f"{c.suggested_stop:>7.2f}  {flags_str}")

        # 缩量整理详情
        consolidating = [c for c in result.candidates if c.consolidation]
        if consolidating:
            print(f"\n  --- 缩量整理形态详情 ---")
            for c in consolidating:
                p = c.consolidation
                print(f"  {c.name}: {p.detail} | 突破价={p.breakout_price}")

    print(f"\n{'='*80}")
