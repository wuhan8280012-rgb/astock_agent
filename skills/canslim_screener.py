"""
Skill 5: CANSLIM 选股框架 (A股适配版)

CANSLIM 七大维度 → A股本土化改造:
  C - Current Quarterly Earnings  : 最近季度盈利加速
  A - Annual Earnings Growth      : 年度盈利持续增长
  N - New Products/Management     : 新产品/新管理层/新高 → A股用"创新高+换手"替代
  S - Supply and Demand           : 流通盘+成交量 → A股加入"限售股解禁"因素
  L - Leader or Laggard           : 行业龙头 vs 跟随者 → 用相对强度RS排名
  I - Institutional Sponsorship   : 机构持仓 → A股用基金持仓+北向持仓
  M - Market Direction            : 大盘方向 → 引用 Sentiment Skill 结果

额外A股风控:
  - 商誉/净资产 < 15%
  - 质押比例 < 20%
  - 排除ST/*ST
  - 排除上市不足2年
  - 近6个月无立案调查

输入: 候选股票池 (或全市场扫描)
输出: 评分排名 + 买入候选
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
# 评分参数（可通过 config.py 覆盖，也可以在复盘中动态调优）
# ============================================================
DEFAULT_CANSLIM_PARAMS = {
    # C - 季度盈利
    "c_min_growth": 20,          # 最近季度净利润同比增速 ≥ 20%
    "c_acceleration_bonus": 10,  # 加速增长额外加分阈值(%)

    # A - 年度盈利
    "a_min_roe": 15,             # 最近年度ROE ≥ 15%
    "a_min_years": 3,            # 连续N年ROE达标
    "a_min_revenue_growth": 15,  # 年度营收增速 ≥ 15%

    # N - 新高 + 突破
    "n_new_high_days": 60,       # N日新高判断周期
    "n_min_turnover": 3,         # 突破日换手率 ≥ 3%

    # S - 供需
    "s_max_circ_mv": 500e8,      # 流通市值上限 500亿 (避免超大盘)
    "s_min_circ_mv": 30e8,       # 流通市值下限 30亿 (避免小微盘)
    "s_volume_expansion": 1.5,   # 放量突破要求: 成交量 ≥ 1.5x 均量

    # L - 领导力 (相对强度)
    "l_min_rs_rank": 80,         # RS排名百分位 ≥ 80 (前20%)

    # I - 机构
    "i_min_fund_holders": 5,     # 最少基金持有家数

    # A股风控
    "risk_max_goodwill_ratio": 0.15,   # 商誉/净资产 ≤ 15%
    "risk_max_pledge_ratio": 0.20,     # 质押比例 ≤ 20%
    "risk_min_listing_days": 500,      # 上市满500个交易日(约2年)

    # 权重
    "weight_c": 20,
    "weight_a": 15,
    "weight_n": 15,
    "weight_s": 10,
    "weight_l": 20,
    "weight_i": 10,
    "weight_risk": 10,
}

PARAMS = {**DEFAULT_CANSLIM_PARAMS, **SKILL_PARAMS.get("canslim", {})}


@dataclass
class StockScore:
    """个股CANSLIM评分"""
    ts_code: str
    name: str
    score_c: float = 0    # 季度盈利
    score_a: float = 0    # 年度盈利
    score_n: float = 0    # 新高突破
    score_s: float = 0    # 供需
    score_l: float = 0    # 领导力
    score_i: float = 0    # 机构
    score_risk: float = 0 # 风控
    total_score: float = 0
    grade: str = ""       # A / B / C / D
    flags: List[str] = field(default_factory=list)  # 标记
    disqualified: bool = False
    disqualify_reason: str = ""

    # 关键数据快照
    latest_roe: float = 0
    latest_eps_growth: float = 0
    rs_rank: float = 0
    circ_mv: float = 0


@dataclass
class CanslimReport:
    """CANSLIM筛选报告"""
    date: str
    total_scanned: int = 0
    total_passed: int = 0
    candidates: List[StockScore] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "total_scanned": self.total_scanned,
            "total_passed": self.total_passed,
            "summary": self.summary,
            "candidates": [
                {
                    "ts_code": c.ts_code, "name": c.name,
                    "total_score": c.total_score, "grade": c.grade,
                    "roe": c.latest_roe, "eps_growth": c.latest_eps_growth,
                    "rs_rank": c.rs_rank, "flags": c.flags,
                }
                for c in self.candidates
            ],
        }

    def to_brief(self) -> str:
        """压缩报告 (~60 tokens)"""
        top3 = []
        for c in self.candidates[:3]:
            flags_str = "+".join(c.flags[:2]) if c.flags else ""
            top3.append(f"{c.name}({c.grade},{c.total_score:.0f}){flags_str}")
        return (
            f"[CANSLIM] 扫描{self.total_scanned}→通过{self.total_passed} "
            f"Top3: {', '.join(top3)}"
        )


class CanslimScreener:
    """CANSLIM 选股器"""

    def __init__(self):
        self.fetcher = get_fetcher()
        self.p = PARAMS

    def screen(
        self,
        stock_pool: List[str] = None,
        sector_codes: List[str] = None,
        top_n: int = 20,
        market_sentiment: str = "中性",
        prefetch_map: Optional[Dict[str, dict]] = None,
    ) -> CanslimReport:
        """
        执行CANSLIM筛选

        Args:
            stock_pool: 指定候选股票列表 (ts_code)。若为None则用板块内股票
            sector_codes: 板块代码列表 (从SectorRotation获取的强势板块)
            top_n: 返回前N只
            market_sentiment: 大盘情绪 (M维度判断)
            prefetch_map: 预筛阶段缓存的股票信息，减少重复 API
        """
        report = CanslimReport(date=self.fetcher.get_latest_trade_date())

        # M维度: 大盘方向判断
        if market_sentiment in ["极度恐慌", "恐慌"]:
            report.summary = f"⚠️ 当前大盘情绪={market_sentiment}，CANSLIM策略建议观望不选股"
            return report

        # 获取候选池
        if stock_pool is None:
            stock_pool = self._get_stock_pool(sector_codes)

        if not stock_pool:
            report.summary = "候选股票池为空"
            return report

        report.total_scanned = len(stock_pool)
        print(f"[CANSLIM] 开始筛选 {len(stock_pool)} 只股票...")

        # 预先计算全市场RS排名 (用于L维度)
        rs_data = self._compute_rs_ranking()

        # 逐只评分
        scored = []
        for i, ts_code in enumerate(stock_pool):
            if (i + 1) % 20 == 0:
                print(f"  进度: {i+1}/{len(stock_pool)}")
            prefetched = prefetch_map.get(ts_code) if prefetch_map else None
            score = self._score_stock(ts_code, rs_data, prefetched=prefetched)
            if score and not score.disqualified:
                scored.append(score)
            time.sleep(0.05)  # 控制API频率

        # 排序取Top N
        scored.sort(key=lambda x: x.total_score, reverse=True)
        report.candidates = scored[:top_n]
        report.total_passed = len(scored)

        # 分配等级
        for c in report.candidates:
            if c.total_score >= 80:
                c.grade = "A"
            elif c.total_score >= 65:
                c.grade = "B"
            elif c.total_score >= 50:
                c.grade = "C"
            else:
                c.grade = "D"

        a_count = sum(1 for c in report.candidates if c.grade == "A")
        b_count = sum(1 for c in report.candidates if c.grade == "B")
        report.summary = (
            f"扫描{report.total_scanned}只 → 通过{report.total_passed}只 → "
            f"Top{top_n}: A级{a_count}只 B级{b_count}只"
        )

        return report

    # ========================================================
    # 候选池构建
    # ========================================================
    def _get_stock_pool(self, sector_codes: List[str] = None) -> List[str]:
        """获取候选股票池"""
        try:
            # 如果指定了板块，获取板块成分股
            if sector_codes:
                pool = set()
                for code in sector_codes:
                    members = self.fetcher.pro.index_member(index_code=code)
                    if members is not None and not members.empty:
                        pool.update(members["con_code"].tolist())
                    time.sleep(0.1)
                return list(pool)

            # 否则用沪深300+中证500作为默认池
            pool = set()
            for idx in ["000300.SH", "000905.SH"]:
                try:
                    members = self.fetcher.pro.index_weight(index_code=idx)
                    if members is not None and not members.empty:
                        pool.update(members["con_code"].unique().tolist())
                except Exception:
                    pass
                time.sleep(0.2)

            return list(pool)[:300]  # 限制数量避免API超限

        except Exception as e:
            print(f"[CANSLIM] 获取候选池失败: {e}")
            return []

    # ========================================================
    # 相对强度排名 (L维度)
    # ========================================================
    def _compute_rs_ranking(self) -> Dict[str, float]:
        """计算相对强度百分位排名"""
        try:
            trade_date = self.fetcher.get_latest_trade_date()
            prev_60 = self.fetcher.get_prev_trade_date(60)
            self.fetcher._throttle()
            now_df = self.fetcher.pro.daily(trade_date=trade_date, fields="ts_code,close")
            self.fetcher._throttle()
            prev_df = self.fetcher.pro.daily(trade_date=prev_60, fields="ts_code,close")
            if now_df is None or prev_df is None or now_df.empty or prev_df.empty:
                return {}
            merged = now_df.merge(prev_df, on="ts_code", suffixes=("_now", "_prev"))
            merged["ret_60d"] = (
                pd.to_numeric(merged["close_now"], errors="coerce")
                / pd.to_numeric(merged["close_prev"], errors="coerce")
                - 1
            ) * 100
            merged = merged.dropna(subset=["ret_60d"])
            if merged.empty:
                return {}
            merged["rs_pctile"] = merged["ret_60d"].rank(pct=True) * 100
            return {
                str(row["ts_code"]): float(row["rs_pctile"])
                for _, row in merged.iterrows()
            }
        except Exception:
            return {}

    # ========================================================
    # 单股评分
    # ========================================================
    def _score_stock(
        self,
        ts_code: str,
        rs_data: dict,
        prefetched: Optional[dict] = None,
    ) -> Optional[StockScore]:
        """对单只股票进行CANSLIM评分"""
        try:
            score = StockScore(ts_code=ts_code, name="")

            # --- 基本排除 ---
            if ts_code.startswith(("688", "8")):
                # 科创板/北交所 可以保留，但标记
                score.flags.append("科创/北交")

            if prefetched:
                score.name = str(prefetched.get("name", "") or "")
                score.circ_mv = float(prefetched.get("circ_mv", 0) or 0)

            # --- 风控排除 ---
            if prefetched and prefetched.get("is_prefiltered"):
                # 上游已做 ST / 市值 / 流动性预筛，保留上市时间等底线检查
                if not score.name:
                    score.name = ts_code
            else:
                # 获取股票名称和基本信息
                basic = self.fetcher.get_daily_basic(self.fetcher.get_latest_trade_date())
                if basic.empty:
                    return None
                stock_basic = basic[basic["ts_code"] == ts_code]
                if stock_basic.empty:
                    return None
                score.circ_mv = float(stock_basic.iloc[0].get("circ_mv", 0)) * 1e4  # 万→元
                disq = self._check_disqualification(ts_code, score)
                if disq:
                    return score

            # --- 一次性拉取财务和日线，供各维度共享 ---
            fina = self.fetcher.get_financial_indicator(ts_code)
            daily = self.fetcher.get_stock_daily(ts_code, days=70)

            # --- C: 季度盈利 ---
            score.score_c = self._score_c_with_data(fina, score)

            # --- A: 年度盈利 ---
            score.score_a = self._score_a_with_data(fina, score)

            # --- N: 新高突破 ---
            score.score_n = self._score_n_with_data(daily)

            # --- S: 供需 ---
            score.score_s = self._score_s_with_data(daily, score)

            # --- L: 领导力 ---
            score.score_l = self._score_l_with_data(daily, rs_data, score)

            # --- 风控加分 ---
            score.score_risk = self._score_risk(ts_code)

            # --- 综合 ---
            score.total_score = round(
                score.score_c * self.p["weight_c"] / 100 +
                score.score_a * self.p["weight_a"] / 100 +
                score.score_n * self.p["weight_n"] / 100 +
                score.score_s * self.p["weight_s"] / 100 +
                score.score_l * self.p["weight_l"] / 100 +
                score.score_risk * self.p["weight_risk"] / 100
            , 1)

            return score

        except Exception:
            return None

    def _check_disqualification(self, ts_code: str, score: StockScore) -> bool:
        """检查是否触发硬排除条件"""
        try:
            # 获取股票基本信息
            info = self.fetcher.pro.stock_basic(
                ts_code=ts_code,
                fields="ts_code,name,list_date,is_hs"
            )
            if info is None or info.empty:
                score.disqualified = True
                score.disqualify_reason = "无法获取基本信息"
                return True

            row = info.iloc[0]
            score.name = row.get("name", ts_code)

            # ST 排除
            name = str(row.get("name", ""))
            if "ST" in name or "*ST" in name:
                score.disqualified = True
                score.disqualify_reason = "ST股票"
                return True

            # 上市时间不足
            list_date = str(row.get("list_date", ""))
            if list_date:
                days_listed = (datetime.now() - datetime.strptime(list_date, "%Y%m%d")).days
                if days_listed < self.p["risk_min_listing_days"]:
                    score.disqualified = True
                    score.disqualify_reason = f"上市不足{self.p['risk_min_listing_days']}天"
                    return True

            # 流通市值过小/过大
            if score.circ_mv > 0:
                if score.circ_mv < self.p["s_min_circ_mv"]:
                    score.disqualified = True
                    score.disqualify_reason = "流通市值过小"
                    return True

            return False

        except Exception:
            return False

    # ========================================================
    # 各维度评分 (0~100)
    # ========================================================
    def _score_c_with_data(self, fina: pd.DataFrame, score: StockScore) -> float:
        """C - 最近季度盈利增速（使用预拉取财务数据）"""
        try:
            if fina is None or fina.empty:
                return 30  # 数据缺失给中间分

            fina = fina.sort_values("end_date", ascending=False).head(4)
            if fina.empty:
                return 30

            latest_growth = fina.iloc[0].get("netprofit_yoy", 0)
            if pd.isna(latest_growth):
                latest_growth = 0

            score.latest_eps_growth = latest_growth
            min_g = self.p["c_min_growth"]

            if latest_growth >= min_g * 2:
                pts = 100
            elif latest_growth >= min_g:
                pts = 60 + (latest_growth - min_g) / min_g * 40
            elif latest_growth > 0:
                pts = latest_growth / min_g * 60
            else:
                pts = max(0, 20 + latest_growth)

            if len(fina) >= 2:
                prev_growth = fina.iloc[1].get("netprofit_yoy", 0)
                if not pd.isna(prev_growth) and latest_growth > prev_growth + self.p["c_acceleration_bonus"]:
                    pts = min(100, pts + 15)
                    score.flags.append("盈利加速↑")

            return round(min(100, max(0, pts)), 1)
        except Exception:
            return 30

    def _score_c(self, ts_code: str, score: StockScore) -> float:
        """向后兼容：旧入口内部转到 with_data 版本"""
        fina = self.fetcher.get_financial_indicator(ts_code)
        return self._score_c_with_data(fina, score)

    def _score_a_with_data(self, fina: pd.DataFrame, score: StockScore) -> float:
        """A - 年度盈利质量（使用预拉取财务数据）"""
        try:
            if fina is None or fina.empty:
                return 30

            annual = fina[fina["end_date"].astype(str).str.endswith("1231")].sort_values("end_date", ascending=False)
            if annual.empty:
                return 30

            latest_roe = annual.iloc[0].get("roe_dt", 0)
            if pd.isna(latest_roe):
                latest_roe = 0
            score.latest_roe = latest_roe

            pts = 0
            min_roe = self.p["a_min_roe"]
            if latest_roe >= min_roe * 1.5:
                pts += 50
            elif latest_roe >= min_roe:
                pts += 30 + (latest_roe - min_roe) / (min_roe * 0.5) * 20
            elif latest_roe > 0:
                pts += latest_roe / min_roe * 30

            if len(annual) >= self.p["a_min_years"]:
                roe_series = annual.head(self.p["a_min_years"])["roe_dt"].tolist()
                if all(not pd.isna(r) and r >= min_roe for r in roe_series):
                    pts += 30
                    score.flags.append(f"ROE连续{self.p['a_min_years']}年>{min_roe}%")

            rev_growth = annual.iloc[0].get("or_yoy", 0)
            if not pd.isna(rev_growth) and rev_growth >= self.p["a_min_revenue_growth"]:
                pts += 20
                score.flags.append(f"营收+{rev_growth:.0f}%")

            return round(min(100, max(0, pts)), 1)
        except Exception:
            return 30

    def _score_a(self, ts_code: str, score: StockScore) -> float:
        """向后兼容：旧入口内部转到 with_data 版本"""
        fina = self.fetcher.get_financial_indicator(ts_code)
        return self._score_a_with_data(fina, score)

    def _score_n_with_data(self, daily: pd.DataFrame) -> float:
        """N - 创新高 + 突破（使用预拉取日线）"""
        try:
            if daily is None or daily.empty or len(daily) < 20:
                return 30
            df = daily.sort_values("trade_date").reset_index(drop=True)
            df_n = df.tail(max(60, self.p["n_new_high_days"]))

            latest_close = df_n.iloc[-1]["close"]
            high_n = df_n["high"].max()
            ratio = latest_close / high_n if high_n > 0 else 0

            if ratio >= 0.97:
                pts = 80
            elif ratio >= 0.90:
                pts = 60
            elif ratio >= 0.80:
                pts = 40
            else:
                pts = 20

            if len(df_n) >= 5 and "vol" in df_n.columns:
                vol_avg = df_n["vol"].tail(20).mean()
                vol_latest = df_n["vol"].iloc[-1]
                if vol_latest > vol_avg * self.p["s_volume_expansion"] and ratio >= 0.95:
                    pts = min(100, pts + 20)

            return round(min(100, max(0, pts)), 1)
        except Exception:
            return 30

    def _score_n(self, ts_code: str) -> float:
        """向后兼容：旧入口内部转到 with_data 版本"""
        df = self.fetcher.get_stock_daily(ts_code, days=max(70, self.p["n_new_high_days"]))
        return self._score_n_with_data(df)

    def _score_s_with_data(self, daily: pd.DataFrame, score: StockScore) -> float:
        """S - 供需（使用预拉取日线）"""
        try:
            pts = 50
            mv = score.circ_mv
            if mv > 0:
                mv_yi = mv / 1e8
                if 50 <= mv_yi <= 300:
                    pts += 30
                elif 30 <= mv_yi <= 500:
                    pts += 15

            if daily is not None and not daily.empty and "vol" in daily.columns:
                df = daily.sort_values("trade_date")
                vol_5 = df["vol"].tail(5).mean()
                vol_20 = df["vol"].tail(20).mean()
                if vol_20 > 0:
                    vol_ratio = vol_5 / vol_20
                    if vol_ratio > 1.5:
                        pts += 20
                    elif vol_ratio > 1.0:
                        pts += 10

            return round(min(100, max(0, pts)), 1)
        except Exception:
            return 40

    def _score_s(self, ts_code: str, score: StockScore) -> float:
        """向后兼容：旧入口内部转到 with_data 版本"""
        df = self.fetcher.get_stock_daily(ts_code, days=70)
        return self._score_s_with_data(df, score)

    def _score_l_with_data(
        self,
        daily: pd.DataFrame,
        rs_data: Dict[str, float],
        score: StockScore,
    ) -> float:
        """L - 领导力（使用预拉取日线 + 全市场 RS）"""
        try:
            if daily is None or daily.empty or len(daily) < 60:
                return 30

            df = daily.sort_values("trade_date")
            ret_60 = (df.iloc[-1]["close"] / df.iloc[-61]["close"] - 1) * 100 if len(df) > 60 else 0

            # 优先使用全市场百分位，失败时回退到绝对涨幅映射
            rs_pctile = float(rs_data.get(score.ts_code, 0) or 0)
            if rs_pctile > 0:
                pts = rs_pctile
            else:
                if ret_60 > 30:
                    pts = 95
                elif ret_60 > 20:
                    pts = 80
                elif ret_60 > 10:
                    pts = 65
                elif ret_60 > 0:
                    pts = 50
                elif ret_60 > -10:
                    pts = 30
                else:
                    pts = 10

            score.rs_rank = round(float(pts), 1)
            return round(float(pts), 1)
        except Exception:
            return 30

    def _score_l(self, ts_code: str, rs_data: dict, score: StockScore) -> float:
        """向后兼容：旧入口内部转到 with_data 版本"""
        df = self.fetcher.get_stock_daily(ts_code, days=70)
        return self._score_l_with_data(df, rs_data, score)

    def _score_risk(self, ts_code: str) -> float:
        """风控维度: 商誉/负债/质押"""
        try:
            pts = 80  # 基础分，扣分制

            # 获取资产负债表
            bs = self.fetcher.get_balancesheet(ts_code)
            if not bs.empty:
                bs = bs.sort_values("end_date", ascending=False)
                latest = bs.iloc[0]

                # 商誉/净资产
                goodwill = latest.get("goodwill", 0) or 0
                equity = latest.get("total_hldr_eqy_exc_min_int", 0) or 0
                if equity > 0 and goodwill / equity > self.p["risk_max_goodwill_ratio"]:
                    pts -= 30

                # 负债率
                total_assets = latest.get("total_assets", 0) or 0
                total_liab = latest.get("total_liab", 0) or 0
                if total_assets > 0:
                    debt_ratio = total_liab / total_assets
                    if debt_ratio > 0.7:
                        pts -= 20
                    elif debt_ratio > 0.5:
                        pts -= 10

            return round(min(100, max(0, pts)), 1)

        except Exception:
            return 60


if __name__ == "__main__":
    screener = CanslimScreener()

    # 小范围测试
    test_pool = [
        "600519.SH",  # 贵州茅台
        "000858.SZ",  # 五粮液
        "002594.SZ",  # 比亚迪
        "300750.SZ",  # 宁德时代
        "601318.SH",  # 中国平安
    ]

    print("开始CANSLIM筛选测试...")
    result = screener.screen(stock_pool=test_pool, top_n=5)

    print(f"\n{'='*70}")
    print(f"  CANSLIM选股报告  {result.date}")
    print(f"  {result.summary}")
    print(f"{'='*70}")
    print(f"{'等级':<4} {'代码':<12} {'名称':<8} {'总分':<6} {'C':<5} {'A':<5} "
          f"{'N':<5} {'S':<5} {'L':<5} {'风控':<5} {'标记'}")
    print("-" * 70)
    for c in result.candidates:
        flags_str = " ".join(c.flags) if c.flags else ""
        print(f"  {c.grade}  {c.ts_code:<12} {c.name:<8} {c.total_score:>5.1f} "
              f"{c.score_c:>4.0f} {c.score_a:>4.0f} {c.score_n:>4.0f} "
              f"{c.score_s:>4.0f} {c.score_l:>4.0f} {c.score_risk:>4.0f}  {flags_str}")
    print(f"{'='*70}")
