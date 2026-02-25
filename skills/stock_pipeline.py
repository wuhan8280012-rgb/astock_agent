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
from typing import List, Dict, Optional, Tuple
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

    def to_brief(self) -> str:
        """压缩报告 (~80 tokens)"""
        strong = [c for c in self.candidates if c.buy_signal_strength == "强"]
        medium = [c for c in self.candidates if c.buy_signal_strength == "中"]
        parts = [
            f"[选股] 情绪:{self.market_sentiment} 板块:{','.join(self.top_sectors[:3])}",
            f"  候选{len(self.candidates)}只: 强{len(strong)} 中{len(medium)}",
        ]
        for c in self.candidates[:3]:
            parts.append(
                f"  {c.buy_signal_strength} {c.name}({c.sector}) "
                f"CANSLIM:{c.canslim_grade}({c.canslim_score:.0f}) "
                f"总分{c.final_score:.0f} 入{c.suggested_entry} 止{c.suggested_stop}"
            )
        return "\n".join(parts)


class StockPipeline:
    """联合选股流水线"""

    def __init__(self):
        self.fetcher = get_fetcher()
        self.sector_skill = SectorRotationSkill()
        self.canslim = CanslimScreener()
        self.consolidation_days = SKILL_PARAMS.get("sector_rotation", {}).get("consolidation_days", 10)
        self.volume_shrink = SKILL_PARAMS.get("sector_rotation", {}).get("volume_shrink_ratio", 0.6)

    def run(
        self,
        top_n_result: int = 10,
        as_of_date: Optional[str] = None,
        verbose: bool = True,
        sentiment_result=None,
        sector_result=None,
        stage_filter_result=None,
    ) -> PipelineReport:
        """执行完整选股流水线"""
        if as_of_date and hasattr(self.fetcher, "set_as_of_date"):
            self.fetcher.set_as_of_date(as_of_date)
        date = self.fetcher.get_latest_trade_date()
        report = PipelineReport(date=date)
        try:
            # ---- Stage 1: 市场情绪 ----
            if verbose:
                print("[Pipeline Stage 1/7] 分析市场情绪...")
            if sentiment_result is not None:
                sentiment = sentiment_result
                if verbose:
                    print("  (使用已有情绪结果，跳过重复分析)")
            else:
                sentiment = SentimentSkill().analyze()

            if isinstance(sentiment, dict):
                sentiment_level = sentiment.get("overall_level", "中性")
            else:
                sentiment_level = getattr(sentiment, "overall_level", "中性")
            report.market_sentiment = sentiment_level

            if sentiment_level in ["极度恐慌", "恐慌"]:
                report.summary = f"市场情绪={sentiment_level}，当前不适合选股，建议观望"
                return report

            # ---- Stage 2: 板块轮动 ----
            if verbose:
                print("[Pipeline Stage 2/7] 运行板块轮动...")
            if sector_result is None:
                sector_result = self.sector_skill.analyze()
            elif verbose:
                print("  (使用已有板块结果，跳过重复分析)")

            if isinstance(sector_result, dict):
                top_items = sector_result.get("top_sectors", [])[:5]
                top_sectors = []
                for i, item in enumerate(top_items, 1):
                    top_sectors.append(type("SectorLite", (), {
                        "ts_code": item.get("ts_code", ""),
                        "name": item.get("name", ""),
                        "signal": item.get("signal", ""),
                        "rank": item.get("rank", i),
                    })())
                # dict 结果若没有 ts_code 无法继续拉取成分股，回退重算
                if top_sectors and not any(getattr(s, "ts_code", "") for s in top_sectors):
                    if verbose:
                        print("  (注入板块结果缺少 ts_code，回退实时分析)")
                    sector_result = self.sector_skill.analyze()
                    top_sectors = getattr(sector_result, "top_sectors", [])[:5]
            else:
                top_sectors = getattr(sector_result, "top_sectors", [])[:5]
            report.top_sectors = [s.name for s in top_sectors]

            if not top_sectors:
                report.summary = "未找到强势板块"
                return report

            if verbose:
                print(f"  强势板块: {', '.join(s.name for s in top_sectors)}")

            # ---- Stage 3: 获取板块成分股 ----
            if verbose:
                print("[Pipeline Stage 3/7] 获取板块成分股...")
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

            # ---- Stage 2.5: Stage 联合过滤（可选）----
            stage_candidates = {}
            if stage_filter_result is not None:
                blacklisted_sectors = {s.name for s in getattr(stage_filter_result, "blacklisted", [])}
                stock_pool = [
                    code for code in stock_pool
                    if sector_stock_map.get(code, "") not in blacklisted_sectors
                ]
                stage_candidates = {
                    c.ts_code: c for c in getattr(stage_filter_result, "candidates", [])
                }
                if verbose:
                    print(f"[Pipeline Stage 2.5/7] Stage过滤后候选池: {len(stock_pool)}只")

            # ---- Stage 4: 批量预筛 ----
            if verbose:
                print("[Pipeline Stage 4/7] 批量预筛...")
            stock_pool, prefetch_map = self._batch_prefilter(stock_pool, verbose=verbose)
            if not stock_pool:
                report.summary = "批量预筛后无候选"
                return report

            # ---- Stage 5: 缩量整理预计算 ----
            if verbose:
                print("[Pipeline Stage 5/7] 缩量整理扫描...")
            stock_pool, pattern_cache = self._consolidation_prefilter(stock_pool, verbose=verbose)

            # ---- Stage 6: CANSLIM 精评 ----
            if verbose:
                print(f"[Pipeline Stage 6/7] CANSLIM评分 ({len(stock_pool)}只)...")
            canslim_result = self.canslim.screen(
                stock_pool=stock_pool,
                top_n=30,
                market_sentiment=sentiment_level,
                prefetch_map=prefetch_map,
            )

            if not canslim_result.candidates:
                report.summary = "CANSLIM筛选后无候选"
                return report

            if verbose:
                print(f"  CANSLIM通过: {len(canslim_result.candidates)}只")

            # ---- Stage 7: 最终排名 ----
            if verbose:
                print("[Pipeline Stage 7/7] 最终排名...")
            candidates = []

            for stock in canslim_result.candidates:
                pattern = pattern_cache.get(stock.ts_code)

                sector_name = sector_stock_map.get(stock.ts_code, "")
                sector_rank = 0
                sector_signal = ""
                for i, s in enumerate(top_sectors):
                    if s.name == sector_name:
                        sector_rank = i + 1
                        sector_signal = s.signal
                        break

                final = stock.total_score
                if pattern and pattern.near_breakout:
                    final += 15
                    stock.flags.append("临近突破")
                elif pattern:
                    final += 8
                    stock.flags.append("缩量整理")

                if sector_rank <= 2:
                    final += 10
                elif sector_rank <= 4:
                    final += 5

                # Stage 候选加分（前置过滤的结构优势）
                if stock.ts_code in stage_candidates:
                    stage_c = stage_candidates[stock.ts_code]
                    final += min(15, round(float(getattr(stage_c, "stage_score", 0)) / 8))
                    stock.flags.append("Stage过滤通过")

                entry_price, stop_price = self._calc_entry_stop(stock.ts_code)

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

            candidates.sort(key=lambda x: x.final_score, reverse=True)
            report.candidates = candidates[:top_n_result]

            strong = sum(1 for c in report.candidates if c.buy_signal_strength == "强")
            medium = sum(1 for c in report.candidates if c.buy_signal_strength == "中")
            report.summary = (
                f"情绪:{sentiment_level} | "
                f"强势板块:{','.join(report.top_sectors[:3])} | "
                f"候选{len(report.candidates)}只 (强信号{strong} 中信号{medium})"
            )
            return report
        finally:
            if as_of_date and hasattr(self.fetcher, "clear_as_of_date"):
                self.fetcher.clear_as_of_date()

    def _batch_prefilter(
        self,
        stock_pool: List[str],
        verbose: bool = True,
    ) -> Tuple[List[str], Dict[str, dict]]:
        """
        批量预筛：尽量用少量批量接口快速淘汰明显低质量标的。
        """
        survivors = set(stock_pool)
        trade_date = self.fetcher.get_latest_trade_date()

        st_codes = set()
        cap_small = set()
        cap_big = set()
        illiquid = set()
        limit_codes = set()
        rs_weak = set()
        prefetch_map: Dict[str, dict] = {}

        try:
            basic = self.fetcher.get_daily_basic(trade_date)
            if basic is not None and not basic.empty:
                pool_basic = basic[basic["ts_code"].isin(survivors)].copy()
                if not pool_basic.empty:
                    # 规则2: 市值过滤 (circ_mv 单位：万元)
                    if "circ_mv" in pool_basic.columns:
                        cap_small = set(pool_basic[pool_basic["circ_mv"] < 30_0000]["ts_code"])
                        cap_big = set(pool_basic[pool_basic["circ_mv"] > 500_0000]["ts_code"])
                        survivors -= cap_small
                        survivors -= cap_big

                    # 规则3: 流动性过滤 (amount 单位：千元)
                    if "amount" in pool_basic.columns:
                        illiquid = set(pool_basic[pool_basic["amount"] < 50000]["ts_code"])
                        survivors -= illiquid

                    # 规则5: 涨跌停排除 (近似口径)
                    if "pct_chg" in pool_basic.columns:
                        limit_codes = set(pool_basic[pd.to_numeric(pool_basic["pct_chg"], errors="coerce").abs() > 9.5]["ts_code"])
                        survivors -= limit_codes

            # 规则1: ST 排除 (name 不在 daily_basic 中，改为一次 stock_basic 批量)
            if survivors:
                if hasattr(self.fetcher, "_throttle"):
                    self.fetcher._throttle()
                stock_info = self.fetcher.pro.stock_basic(
                    exchange="",
                    list_status="L",
                    fields="ts_code,name",
                )
                if stock_info is not None and not stock_info.empty:
                    info = stock_info[stock_info["ts_code"].isin(survivors)].copy()
                    st_codes = set(
                        info[info["name"].astype(str).str.contains(r"\*?ST", na=False)]["ts_code"]
                    )
                    survivors -= st_codes
                    name_map = dict(zip(info["ts_code"], info["name"]))
                else:
                    name_map = {}
            else:
                name_map = {}

            # 规则4: RS 动量淘汰后50%
            if survivors:
                prev_date = self.fetcher.get_prev_trade_date(60)
                if hasattr(self.fetcher, "_throttle"):
                    self.fetcher._throttle()
                today_df = self.fetcher.pro.daily(trade_date=trade_date, fields="ts_code,close")
                if hasattr(self.fetcher, "_throttle"):
                    self.fetcher._throttle()
                prev_df = self.fetcher.pro.daily(trade_date=prev_date, fields="ts_code,close")

                if today_df is not None and prev_df is not None and not today_df.empty and not prev_df.empty:
                    merged = today_df.merge(prev_df, on="ts_code", suffixes=("_now", "_prev"))
                    merged["ret_60d"] = (
                        pd.to_numeric(merged["close_now"], errors="coerce")
                        / pd.to_numeric(merged["close_prev"], errors="coerce")
                        - 1
                    ) * 100
                    pool_rs = merged[merged["ts_code"].isin(survivors)].dropna(subset=["ret_60d"]).copy()
                    if not pool_rs.empty:
                        pool_rs["rs_pctile"] = pool_rs["ret_60d"].rank(pct=True) * 100
                        rs_weak = set(pool_rs[pool_rs["rs_pctile"] < 50]["ts_code"])
                        survivors -= rs_weak
                        rs_map = dict(zip(pool_rs["ts_code"], pool_rs["rs_pctile"]))
                    else:
                        rs_map = {}
                else:
                    rs_map = {}
            else:
                rs_map = {}

            # 构建传给 CANSLIM 的预拉取 map（仅保留存活标的）
            if basic is not None and not basic.empty:
                base_survivors = basic[basic["ts_code"].isin(survivors)].copy()
            else:
                base_survivors = pd.DataFrame(columns=["ts_code", "circ_mv"])

            for _, row in base_survivors.iterrows():
                code = str(row.get("ts_code"))
                circ_mv_wan = float(row.get("circ_mv", 0) or 0)  # 万元
                prefetch_map[code] = {
                    "name": name_map.get(code, code),
                    "circ_mv": circ_mv_wan * 1e4,  # 转元，兼容 CANSLIM 原逻辑
                    "is_prefiltered": True,
                    "rs_pctile": float(rs_map.get(code, 0) or 0),
                }

        except Exception as e:
            if verbose:
                print(f"  [预筛] 异常，回退原候选池: {e}")
            return stock_pool, {}

        result = [c for c in stock_pool if c in survivors]
        if verbose:
            print(
                f"  [预筛] {len(stock_pool)} → {len(result)} "
                f"(ST{len(st_codes)} 市值{len(cap_small | cap_big)} "
                f"流动性{len(illiquid)} 涨跌停{len(limit_codes)} RS弱{len(rs_weak)})"
            )
        return result, prefetch_map

    def _consolidation_prefilter(
        self,
        stock_pool: List[str],
        verbose: bool = True,
    ) -> Tuple[List[str], Dict[str, ConsolidationPattern]]:
        """
        缩量整理预计算：不做硬过滤，避免误杀，仅缓存形态供后续打分复用。
        """
        survivors: List[str] = []
        patterns: Dict[str, ConsolidationPattern] = {}
        for ts_code in stock_pool:
            pattern = self._check_consolidation(ts_code)
            if pattern:
                patterns[ts_code] = pattern
            survivors.append(ts_code)
            time.sleep(0.03)

        if verbose:
            print(f"  [整理] 缩量整理: {len(patterns)}/{len(stock_pool)}只")
        return survivors, patterns

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
