"""
Skill 7: 主导主题龙头突破 (Dominant Theme Leader Breakout)

原始策略逻辑:
  1. 当日成交额Top15，筛出涨幅>5%的票
  2. 同一主题≥2只 → 认定主导主题
  3. 主导主题里取最强的一等股 (pct_chg最大)
  4. 一等股必须满足"首次突破"6个月新高

重构修复:
  ✅ 接入 DataFetcher 缓存，避免重复API调用
  ✅ 主题识别改用 stock_basic.industry 为主 + concept 为增强(可选)
     → 解决原版遍历全部concept导致API爆限的问题
  ✅ 增加大盘情绪过滤 (引用 SentimentSkill)
  ✅ 首次突破窗口可配置，增加"宽松模式"
  ✅ 输出标准化 dataclass，可被 daily_agent / weekly_agent 调用
  ✅ 支持多主题并行输出 (不只返回一个主题)
  ✅ 增加连板/加速检测，区分"首板突破"和"接力板"

使用:
  python skills/theme_leader.py                     # 默认最近交易日
  python skills/theme_leader.py --date 20260221     # 指定日期
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
except ImportError:
    SKILL_PARAMS = {}


# ============================================================
# 参数 (可通过 config.py 的 SKILL_PARAMS["theme_leader"] 覆盖)
# ============================================================
DEFAULT_PARAMS = {
    # --- 候选筛选 ---
    "top_n_amount": 15,          # 成交额Top N
    "min_pct_chg": 5.0,          # 最低涨幅(%)
    "min_theme_count": 2,        # 同主题最少股票数

    # --- 突破参数 ---
    "lookback_days": 140,        # 6个月新高回溯期(交易日)
    "first_breakout_window": 40, # 首次突破过滤窗口(原版60太严，改40)
    "breakout_buffer": 0.003,    # 突破缓冲 0.3% (原版0，太敏感)
    "near_breakout_pct": 0.02,   # 接近突破阈值 2%

    # --- K线形态 ---
    "body_to_range_min": 0.55,   # 实体/振幅 ≥ 55% (原版0.6偏严)
    "close_in_top_pct": 0.70,    # 收盘在振幅上部 ≥ 70% (原版0.75偏严)

    # --- A档触发 ---
    "tier_a_pct_chg": 9.5,       # A档涨幅 (原版10，改9.5兼容非一字板)
    "tier_a_vol_ratio": 2.0,     # A档量比

    # --- B档触发 ---
    "tier_b_pct_chg": 6.0,       # B档涨幅 (原版7，适度放宽)
    "tier_b_vol_ratio": 1.5,     # B档量比 (原版1.6)

    # --- 风控 ---
    "max_consecutive_limits": 3, # 连续涨停≥N天视为加速段，降级处理
    "vol_avg_window": 20,        # 量比均线窗口

    # --- 大盘过滤 ---
    "skip_in_panic": True,       # 大盘恐慌时不选股
}

PARAMS = {**DEFAULT_PARAMS, **SKILL_PARAMS.get("theme_leader", {})}


# ============================================================
# 数据结构
# ============================================================
@dataclass
class LeaderCandidate:
    """主题龙头候选"""
    ts_code: str
    name: str
    pct_chg: float
    amount: float           # 成交额(千元)
    close: float
    theme: str              # 所属主题
    tier: str               # "A" / "B" / "NONE"
    is_leader: bool = False # 是否为该主题一等股

    # 突破详情
    prior_high_6m: float = 0
    dist_to_high_pct: float = 0  # 距6月新高百分比
    vol_ratio: float = 0
    body_to_range: float = 0
    is_first_breakout: bool = False
    consecutive_limits: int = 0  # 连板天数

    flags: List[str] = field(default_factory=list)
    detail: str = ""


@dataclass
class ThemeGroup:
    """主题组"""
    theme_name: str
    stock_count: int
    total_amount: float
    max_pct_chg: float
    leader: Optional[LeaderCandidate] = None
    members: List[LeaderCandidate] = field(default_factory=list)


@dataclass
class ThemeLeaderReport:
    """主题龙头选股报告"""
    date: str
    market_sentiment: str = ""
    skipped_reason: str = ""
    top_amount_count: int = 0
    candidate_count: int = 0
    themes_found: int = 0
    theme_groups: List[ThemeGroup] = field(default_factory=list)
    signals: List[LeaderCandidate] = field(default_factory=list)  # 有效信号(A/B档)
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "market_sentiment": self.market_sentiment,
            "skipped_reason": self.skipped_reason,
            "themes_found": self.themes_found,
            "summary": self.summary,
            "signals": [
                {
                    "ts_code": s.ts_code, "name": s.name, "tier": s.tier,
                    "theme": s.theme, "pct_chg": s.pct_chg,
                    "vol_ratio": s.vol_ratio, "is_first_breakout": s.is_first_breakout,
                    "flags": s.flags,
                }
                for s in self.signals
            ],
            "themes": [
                {
                    "name": g.theme_name, "count": g.stock_count,
                    "leader": g.leader.ts_code if g.leader else None,
                }
                for g in self.theme_groups
            ],
        }


# ============================================================
# 主类
# ============================================================
class ThemeLeaderSkill:
    """主导主题龙头突破选股"""

    def __init__(self):
        self.fetcher = get_fetcher()
        self.p = PARAMS
        self._stock_industry = None  # 缓存行业映射

    def analyze(self, trade_date: str = None, sentiment_level: str = "") -> ThemeLeaderReport:
        """
        执行主题龙头选股

        Args:
            trade_date: 交易日(YYYYMMDD)，None则自动取最近交易日
            sentiment_level: 大盘情绪等级(来自SentimentSkill)
        """
        if trade_date is None:
            trade_date = self._resolve_trade_date()

        report = ThemeLeaderReport(date=trade_date, market_sentiment=sentiment_level)

        # === 大盘过滤 ===
        if self.p["skip_in_panic"] and sentiment_level in ["极度恐慌", "恐慌"]:
            report.skipped_reason = f"大盘情绪={sentiment_level}，跳过主题选股"
            report.summary = report.skipped_reason
            return report

        # === Step 1: 获取全市场当日行情 ===
        daily = self._get_daily(trade_date)
        if daily is None or daily.empty:
            report.skipped_reason = f"{trade_date}无行情数据"
            report.summary = report.skipped_reason
            return report

        # === Step 2: 成交额Top N + 涨幅过滤 ===
        top = daily.sort_values("amount", ascending=False).head(self.p["top_n_amount"]).copy()
        candidates = top[top["pct_chg"] > self.p["min_pct_chg"]].copy()
        report.top_amount_count = len(top)
        report.candidate_count = len(candidates)

        if candidates.empty:
            report.skipped_reason = (
                f"Top{self.p['top_n_amount']}成交额中无涨幅>{self.p['min_pct_chg']}%的股票"
            )
            report.summary = report.skipped_reason
            return report

        # === Step 3: 主题分组 ===
        ts_codes = candidates["ts_code"].tolist()
        industry_map = self._get_industry_map(ts_codes)
        theme_groups = self._group_by_theme(candidates, industry_map)
        report.themes_found = len(theme_groups)

        if not theme_groups:
            report.skipped_reason = "未找到同主题≥2只的组合"
            report.summary = report.skipped_reason
            return report

        report.theme_groups = theme_groups

        # === Step 4: 对每个主题的龙头做突破检测 ===
        all_signals = []
        for group in theme_groups:
            leader = group.leader
            if leader is None:
                continue

            # 拉取历史数据做突破检测
            tier, detail = self._check_breakout(leader.ts_code, trade_date)
            leader.tier = tier
            leader.detail = str(detail)

            # 填充突破信息
            if isinstance(detail, dict):
                leader.prior_high_6m = detail.get("prior_high_6m", 0)
                leader.dist_to_high_pct = round(detail.get("dist_to_high", 0) * 100, 2)
                leader.vol_ratio = round(detail.get("vol_ratio", 0), 2)
                leader.body_to_range = round(detail.get("body_to_range", 0), 2)
                leader.is_first_breakout = detail.get("is_first_breakout_today", False)
                leader.consecutive_limits = detail.get("consecutive_limits", 0)

                # 标记
                if leader.is_first_breakout:
                    leader.flags.append("首次突破")
                if leader.consecutive_limits >= 2:
                    leader.flags.append(f"{leader.consecutive_limits}连板")
                if leader.vol_ratio >= 3.0:
                    leader.flags.append("巨量")
                if detail.get("has_recent_breakout"):
                    leader.flags.append("⚠️非首次突破")

            if tier in ["A", "B"]:
                all_signals.append(leader)

        report.signals = all_signals

        # === 生成摘要 ===
        if all_signals:
            sig_parts = []
            for s in all_signals:
                flags_str = " ".join(s.flags[:2])
                sig_parts.append(f"{s.tier}档 {s.name}({s.ts_code}) "
                                 f"+{s.pct_chg:.1f}% 量比{s.vol_ratio:.1f} "
                                 f"[{s.theme}] {flags_str}")
            report.summary = f"发现{len(all_signals)}个信号: " + " | ".join(sig_parts)
        else:
            themes_str = ", ".join(g.theme_name for g in theme_groups[:3])
            report.summary = f"主导主题: {themes_str}，但龙头均未触发突破条件"

        return report

    # ========================================================
    # 数据获取 (使用DataFetcher缓存)
    # ========================================================
    def _resolve_trade_date(self) -> str:
        """获取最近有数据的交易日"""
        date = self.fetcher.get_latest_trade_date()
        # 验证该日有数据
        df = self._get_daily(date)
        if df is not None and not df.empty:
            return date

        # 回退
        dates = self.fetcher.get_trade_dates()
        for d in reversed(dates[-10:]):
            df = self._get_daily(d)
            if df is not None and not df.empty:
                return d
        return date

    def _get_daily(self, trade_date: str) -> Optional[pd.DataFrame]:
        """获取全市场日线"""
        return self.fetcher._fetch_with_cache(
            "daily_all_theme",
            self.fetcher.pro.daily,
            trade_date=trade_date,
            fields="ts_code,trade_date,open,high,low,close,vol,amount,pct_chg"
        )

    def _get_industry_map(self, ts_codes: List[str]) -> Dict[str, str]:
        """
        获取行业映射 (用stock_basic，一次API调用搞定)

        修复原版问题: 原版遍历全部concept板块，API调用量爆炸
        改为: 用行业分类作为主分组依据，简单高效
        """
        if self._stock_industry is None:
            basic = self.fetcher._fetch_with_cache(
                "stock_basic_industry",
                self.fetcher.pro.stock_basic,
                exchange="",
                list_status="L",
                fields="ts_code,name,industry"
            )
            if basic is not None and not basic.empty:
                self._stock_industry = dict(zip(
                    basic["ts_code"],
                    list(zip(basic["name"].fillna(""), basic["industry"].fillna("")))
                ))
            else:
                self._stock_industry = {}

        result = {}
        for code in ts_codes:
            info = self._stock_industry.get(code, ("", ""))
            result[code] = info  # (name, industry)
        return result

    # ========================================================
    # 主题分组
    # ========================================================
    def _group_by_theme(
        self,
        candidates: pd.DataFrame,
        industry_map: Dict[str, tuple]
    ) -> List[ThemeGroup]:
        """按行业分组，找出同主题≥2只的组合"""
        min_count = self.p["min_theme_count"]

        # 构建 code → (name, industry, pct_chg, amount)
        records = []
        for _, row in candidates.iterrows():
            code = row["ts_code"]
            name, industry = industry_map.get(code, ("", ""))
            if not industry:
                continue
            records.append({
                "ts_code": code,
                "name": name,
                "industry": industry,
                "pct_chg": float(row["pct_chg"]),
                "amount": float(row["amount"]),
                "close": float(row["close"]),
                "open": float(row.get("open", 0)),
                "high": float(row.get("high", 0)),
                "low": float(row.get("low", 0)),
            })

        if not records:
            return []

        df = pd.DataFrame(records)

        # 按行业分组
        groups = []
        for industry, grp in df.groupby("industry"):
            if len(grp) < min_count:
                continue

            grp_sorted = grp.sort_values(["pct_chg", "amount"], ascending=[False, False])

            members = []
            for _, r in grp_sorted.iterrows():
                members.append(LeaderCandidate(
                    ts_code=r["ts_code"],
                    name=r["name"],
                    pct_chg=r["pct_chg"],
                    amount=r["amount"],
                    close=r["close"],
                    theme=industry,
                    tier="NONE",
                ))

            # 一等股 = 涨幅最大的
            leader = members[0]
            leader.is_leader = True

            groups.append(ThemeGroup(
                theme_name=industry,
                stock_count=len(members),
                total_amount=grp["amount"].sum(),
                max_pct_chg=grp["pct_chg"].max(),
                leader=leader,
                members=members,
            ))

        # 按(股票数, 总成交额, 最大涨幅)降序排列
        groups.sort(key=lambda g: (g.stock_count, g.total_amount, g.max_pct_chg), reverse=True)
        return groups

    # ========================================================
    # 突破检测 (核心逻辑重构)
    # ========================================================
    def _check_breakout(self, ts_code: str, trade_date: str) -> Tuple[str, dict]:
        """
        检测是否为首次突破

        修复原版问题:
        1. 增加连板检测 → 连续涨停≥3天降级
        2. 突破缓冲从0改为0.3% → 减少噪声
        3. K线形态阈值适度放宽 → 兼容非一字板的长阳
        """
        lookback = self.p["lookback_days"]
        extra = 80  # 额外历史数据，用于计算均量和首次突破窗口

        # 拉取历史数据
        end_dt = datetime.strptime(trade_date, "%Y%m%d")
        start_dt = end_dt - timedelta(days=(lookback + extra) * 2)
        start_date = start_dt.strftime("%Y%m%d")

        hist = self.fetcher._fetch_with_cache(
            f"hist_theme_{ts_code}",
            self.fetcher.pro.daily,
            ts_code=ts_code,
            start_date=start_date,
            end_date=trade_date,
            fields="ts_code,trade_date,open,high,low,close,vol,amount,pct_chg"
        )

        if hist is None or hist.empty or len(hist) < 80:
            return "NONE", {"reason": "insufficient_history", "bars": len(hist) if hist is not None else 0}

        hist = hist.sort_values("trade_date").reset_index(drop=True)
        # 只保留需要的长度
        max_keep = lookback + extra
        if len(hist) > max_keep:
            hist = hist.iloc[-max_keep:].copy().reset_index(drop=True)

        i = len(hist) - 1  # 当日索引
        last = hist.iloc[i]

        # --- 基础数据 ---
        close_ = float(last["close"])
        open_ = float(last["open"])
        high_ = float(last["high"])
        low_ = float(last["low"])
        pct = float(last["pct_chg"])
        vol = float(last["vol"])

        # --- 6个月新高 ---
        start_idx = max(0, i - lookback)
        prior_high = float(hist["high"].iloc[start_idx:i].max())

        buf = self.p["breakout_buffer"]
        is_breakout = close_ > prior_high * (1.0 + buf)
        near_breakout = close_ >= prior_high * (1.0 - self.p["near_breakout_pct"])
        dist_to_high = (close_ / prior_high - 1.0) if prior_high > 0 else 0

        # --- 首次突破检测 ---
        # 滚动标记历史突破事件
        highs = hist["high"].astype(float).to_numpy()
        closes = hist["close"].astype(float).to_numpy()
        events = np.zeros(len(hist), dtype=bool)

        for j in range(1, len(hist)):
            s = max(0, j - lookback)
            ph = float(np.max(highs[s:j]))
            events[j] = closes[j] > ph * (1.0 + buf)

        # 过去N天是否已有突破事件
        fb_window = self.p["first_breakout_window"]
        j0 = max(0, i - fb_window)
        has_recent_breakout = bool(np.any(events[j0:i]))  # 不含当日
        is_first_breakout = bool(events[i] and not has_recent_breakout)

        # --- K线形态 ---
        rng = max(1e-9, high_ - low_)
        body = close_ - open_
        body_to_range = body / rng
        close_in_range = (close_ - low_) / rng
        is_long_bull = (
            body > 0
            and body_to_range >= self.p["body_to_range_min"]
            and close_in_range >= self.p["close_in_top_pct"]
        )

        # --- 量比 ---
        vol_window = self.p["vol_avg_window"]
        if i >= vol_window + 1:
            vol_ma = hist["vol"].iloc[i - vol_window - 1:i - 1].mean()
            vol_ratio = vol / vol_ma if vol_ma > 0 else 0
        else:
            vol_ratio = 0

        # --- 连板检测 (A股特色) ---
        consecutive_limits = 0
        for j in range(i, max(i - 10, -1), -1):
            if float(hist.iloc[j]["pct_chg"]) >= 9.5:  # 主板涨停≈10%
                consecutive_limits += 1
            else:
                break

        # --- 连板过多降级 ---
        max_consec = self.p["max_consecutive_limits"]
        force_downgrade = consecutive_limits >= max_consec

        # --- 分档判断 ---
        # 核心修复: allow_first = 首次突破 OR (非首次但当日是突破日且窗口内无其他突破)
        allow_first = not has_recent_breakout

        ok_A = (
            allow_first
            and not force_downgrade
            and is_breakout
            and is_long_bull
            and pct >= self.p["tier_a_pct_chg"]
            and vol_ratio >= self.p["tier_a_vol_ratio"]
        )

        ok_B = (
            allow_first
            and not force_downgrade
            and near_breakout
            and is_long_bull
            and pct >= self.p["tier_b_pct_chg"]
            and vol_ratio >= self.p["tier_b_vol_ratio"]
        )

        # 连板降级: A→B, B→NONE
        if force_downgrade and ok_A:
            tier = "B"
        elif force_downgrade and ok_B:
            tier = "NONE"
        else:
            tier = "A" if ok_A else ("B" if ok_B else "NONE")

        info = {
            "close": close_,
            "open": open_,
            "high": high_,
            "low": low_,
            "pct_chg": pct,
            "vol": vol,
            "prior_high_6m": prior_high,
            "dist_to_high": dist_to_high,
            "is_breakout": is_breakout,
            "near_breakout": near_breakout,
            "has_recent_breakout": has_recent_breakout,
            "is_first_breakout_today": is_first_breakout,
            "first_breakout_window": fb_window,
            "body_to_range": body_to_range,
            "close_in_range": close_in_range,
            "is_long_bull": is_long_bull,
            "vol_ratio": vol_ratio,
            "consecutive_limits": consecutive_limits,
            "force_downgrade": force_downgrade,
        }

        return tier, info

    # ========================================================
    # 增强: 尝试用概念板块补充主题 (可选，需要高积分)
    # ========================================================
    def enhance_with_concepts(self, ts_codes: List[str]) -> Dict[str, List[str]]:
        """
        尝试获取概念板块信息

        注意: 这个方法API调用量大，仅在需要时手动调用
        不在默认analyze流程中使用
        """
        concept_map = {c: [] for c in ts_codes}
        try:
            concepts = self.fetcher.pro.concept(src="ts", fields="code,name")
            if concepts is None or concepts.empty:
                return concept_map

            cpn = dict(zip(concepts["code"], concepts["name"]))
            target = set(ts_codes)

            for _, row in concepts.iterrows():
                cid = row["code"]
                try:
                    detail = self.fetcher.pro.concept_detail(id=cid, fields="id,ts_code")
                    if detail is None or detail.empty:
                        continue
                    matched = detail[detail["ts_code"].isin(target)]
                    if matched.empty:
                        continue
                    cname = cpn.get(cid, cid)
                    for code in matched["ts_code"].unique():
                        concept_map[code].append(cname)
                except Exception:
                    continue
                time.sleep(0.15)  # 控制频率

            return concept_map
        except Exception as e:
            print(f"[ThemeLeader] 概念增强失败: {e}")
            return concept_map


# ============================================================
# CLI 入口
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="主导主题龙头突破选股")
    parser.add_argument("--date", type=str, default=None, help="交易日 YYYYMMDD")
    parser.add_argument("--no-sentiment", action="store_true", help="跳过情绪过滤")
    args = parser.parse_args()

    skill = ThemeLeaderSkill()

    # 可选: 先获取情绪
    sentiment = ""
    if not args.no_sentiment:
        try:
            from skills.sentiment import SentimentSkill
            s = SentimentSkill().analyze()
            sentiment = s.overall_level
            print(f"[情绪] {sentiment} ({s.overall_score:+.1f})")
        except Exception as e:
            print(f"[情绪] 获取失败: {e}，跳过情绪过滤")

    result = skill.analyze(trade_date=args.date, sentiment_level=sentiment)

    print(f"\n{'='*70}")
    print(f"  🐎 主导主题龙头突破  {result.date}")
    print(f"{'='*70}")
    print(f"  {result.summary}")
    print()

    if result.skipped_reason:
        print(f"  ⚠️ {result.skipped_reason}")
    else:
        # 主题组
        for g in result.theme_groups:
            leader_tag = ""
            if g.leader:
                leader_tag = f" → 龙头: {g.leader.name}({g.leader.ts_code}) +{g.leader.pct_chg:.1f}%"
            print(f"  📌 [{g.theme_name}] {g.stock_count}只 | "
                  f"总成交{g.total_amount/1e4:.0f}万{leader_tag}")

            # 龙头突破详情
            if g.leader and g.leader.tier != "NONE":
                l = g.leader
                print(f"     🔥 {l.tier}档信号 | 量比{l.vol_ratio:.1f} | "
                      f"距新高{l.dist_to_high_pct:+.1f}% | "
                      f"{' '.join(l.flags)}")

        # 汇总信号
        if result.signals:
            print(f"\n  === 有效信号 ({len(result.signals)}个) ===")
            for s in result.signals:
                icon = "🔥" if s.tier == "A" else "⭐"
                print(f"  {icon} [{s.tier}] {s.name}({s.ts_code}) "
                      f"+{s.pct_chg:.1f}% | 量比{s.vol_ratio:.1f} | "
                      f"[{s.theme}] | {' '.join(s.flags)}")
        else:
            print(f"\n  ❌ 无有效突破信号")

    print(f"\n{'='*70}")
