"""
Skill 7: 板块 Stage 联合过滤器（A股适配）

目标：
  1. 基于板块相对强度做 Top/Bottom 分层
  2. 生成黑名单板块（Bottom N）
  3. 在强势板块内筛出「底部 + 收紧 + R:R达标」候选
"""

from dataclasses import dataclass, field
from typing import List, Optional
import time
import numpy as np

from data_fetcher import get_fetcher
from skills.sector_rotation import SectorRotationSkill

try:
    from config import SKILL_PARAMS
    PARAMS = SKILL_PARAMS.get("sector_stage_filter", {})
except ImportError:
    PARAMS = {}


@dataclass
class SectorStageItem:
    name: str
    ts_code: str = ""
    rs_20d: float = 0.0
    rs_60d: float = 0.0
    score: float = 0.0
    warning: str = ""


@dataclass
class StageCandidate:
    ts_code: str
    name: str
    sector: str
    stage_grade: str
    trigger: str
    rr_ratio: float
    stage_score: float


@dataclass
class SectorStageReport:
    date: str
    leaders: List[SectorStageItem] = field(default_factory=list)
    blacklisted: List[SectorStageItem] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    candidates: List[StageCandidate] = field(default_factory=list)
    scanned: int = 0
    passed: int = 0
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "leaders": [vars(x) for x in self.leaders],
            "blacklisted": [vars(x) for x in self.blacklisted],
            "warnings": self.warnings,
            "candidates": [vars(x) for x in self.candidates],
            "scanned": self.scanned,
            "passed": self.passed,
            "summary": self.summary,
        }

    def to_brief(self) -> str:
        leaders = ", ".join([f"{x.name}(RS{x.score:+.1f})" for x in self.leaders[:3]]) or "无"
        black = ", ".join([x.name for x in self.blacklisted[:3]]) or "无"
        cands = ", ".join([f"{c.name}({c.stage_grade},R:R={c.rr_ratio:.1f})" for c in self.candidates[:3]]) or "无"
        return (
            f"[板块Stage] 领涨: {leaders} | 黑名单: {black}\n"
            f"  扫描{self.scanned}→通过{self.passed}: {cands}"
        )


class SectorStageFilter:
    """板块 Stage 联合过滤器"""

    def __init__(self):
        self.fetcher = get_fetcher()
        self.sector_skill = SectorRotationSkill()
        self.top_n = int(PARAMS.get("top_n", 5))
        self.blacklist_n = int(PARAMS.get("blacklist_n", 5))
        self.base_min_days = int(PARAMS.get("base_min_days", 60))
        self.tight_range_max_pct = float(PARAMS.get("tight_range_max_pct", 5))
        self.min_rr_ratio = float(PARAMS.get("min_rr_ratio", 3.0))
        self.max_scan_stocks = int(PARAMS.get("max_scan_stocks", 120))

    def analyze(self, sector_report=None, verbose: bool = False) -> SectorStageReport:
        date = self.fetcher.get_latest_trade_date()
        report = SectorStageReport(date=date)

        if sector_report is None:
            sector_report = self.sector_skill.analyze()

        sectors = getattr(sector_report, "sectors", []) or []
        if not sectors:
            report.summary = "无板块数据"
            return report

        ranked = []
        for s in sectors:
            score = float(s.ret_20d) * 0.6 + float(s.ret_60d) * 0.4
            warning = ""
            if float(s.ret_5d) < 0 and float(s.ret_20d) > 0:
                warning = "短期RS弱于中期，动能衰退"
            ranked.append(SectorStageItem(
                name=s.name,
                ts_code=s.ts_code,
                rs_20d=float(s.ret_20d),
                rs_60d=float(s.ret_60d),
                score=round(score, 2),
                warning=warning,
            ))

        ranked.sort(key=lambda x: x.score, reverse=True)
        report.leaders = ranked[: self.top_n]
        report.blacklisted = list(reversed(ranked[-self.blacklist_n:]))
        report.warnings = [f"⚠️ {x.name}: {x.warning}" for x in report.leaders if x.warning]

        blacklist_names = {x.name for x in report.blacklisted}
        top_codes = [x.ts_code for x in report.leaders if x.ts_code]

        stock_pool = []
        sector_map = {}
        for sec in report.leaders:
            if sec.name in blacklist_names or not sec.ts_code:
                continue
            try:
                members = self.fetcher.pro.index_member(index_code=sec.ts_code)
                if members is not None and not members.empty:
                    for code in members["con_code"].tolist():
                        stock_pool.append(code)
                        sector_map[code] = sec.name
                time.sleep(0.1)
            except Exception:
                continue

        stock_pool = list(dict.fromkeys(stock_pool))[: self.max_scan_stocks]
        report.scanned = len(stock_pool)
        if verbose:
            print(f"[StageFilter] 扫描股票数: {report.scanned}")

        for ts_code in stock_pool:
            cand = self._evaluate_stock(ts_code, sector_map.get(ts_code, ""))
            if cand:
                report.candidates.append(cand)

        report.candidates.sort(key=lambda x: x.stage_score, reverse=True)
        report.passed = len(report.candidates)

        leaders_text = ", ".join([x.name for x in report.leaders[:3]]) or "无"
        black_text = ", ".join([x.name for x in report.blacklisted[:3]]) or "无"
        report.summary = f"领涨:{leaders_text} | 黑名单:{black_text} | 扫描{report.scanned}→通过{report.passed}"
        return report

    def _evaluate_stock(self, ts_code: str, sector_name: str) -> Optional[StageCandidate]:
        try:
            df = self.fetcher.get_daily(ts_code, days=260)
            if df is None or df.empty or len(df) < self.base_min_days:
                return None
            df = df.sort_values("trade_date").reset_index(drop=True)
            close = df["close"].astype(float).values
            high = df["high"].astype(float).values
            low = df["low"].astype(float).values
            current = float(close[-1])

            low_60 = float(np.min(low[-60:]))
            high_260 = float(np.max(high))
            span = max(high_260 - low_60, 1e-6)
            base_pos = (current - low_60) / span
            if base_pos <= 0.35:
                grade = "A级大底"
                grade_score = 40
            elif base_pos <= 0.60:
                grade = "B级中底"
                grade_score = 28
            else:
                return None

            ema10 = self._ema(close, 10)[-1]
            ema20 = self._ema(close, 20)[-1]
            ema50 = self._ema(close, 50)[-1]
            range_10 = (float(np.max(high[-10:])) - float(np.min(low[-10:]))) / max(float(np.min(low[-10:])), 1e-6) * 100
            tightening = (range_10 <= self.tight_range_max_pct) and (ema10 > ema20 > ema50) and (current >= ema20)
            if not tightening:
                return None

            atr = self._atr(high, low, close, period=14)
            if atr <= 0:
                return None
            stop = current - 2 * atr
            target = current + 6 * atr
            rr = (target - current) / max(current - stop, 1e-6)
            if rr < self.min_rr_ratio:
                return None

            trigger = "🔥 周线收紧" if range_10 <= 3 else "⭐ 日线收紧"
            stage_score = grade_score + (12 if trigger.startswith("🔥") else 8) + min(20, (rr - self.min_rr_ratio) * 8)

            return StageCandidate(
                ts_code=ts_code,
                name=self.fetcher.get_stock_name(ts_code),
                sector=sector_name,
                stage_grade=grade,
                trigger=trigger,
                rr_ratio=round(float(rr), 2),
                stage_score=round(float(stage_score), 1),
            )
        except Exception:
            return None

    @staticmethod
    def _ema(arr, period):
        alpha = 2 / (period + 1.0)
        out = np.zeros_like(arr, dtype=float)
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
        return out

    @staticmethod
    def _atr(high, low, close, period=14):
        if len(close) < period + 1:
            return 0.0
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
        return float(np.mean(tr[-period:]))
