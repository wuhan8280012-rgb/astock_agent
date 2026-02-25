"""
fund_flow_proxy.py — 资金流向代理
=================================

职责:
  1. 首选北向资金
  2. 失败时降级 ETF 份额资金流
  3. 再失败降级两融变化
"""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timedelta
import time

import pandas as pd


@dataclass
class FundFlowResult:
    """资金流向统一结果"""
    latest_flow: float
    flow_3d: float
    flow_5d: float
    flow_20d: float
    trend: str
    source: str
    degraded: bool
    degradation_note: str
    confidence: float

    def to_dict(self) -> dict:
        return {
            "latest_flow": self.latest_flow,
            "flow_3d": self.flow_3d,
            "flow_5d": self.flow_5d,
            "flow_20d": self.flow_20d,
            "trend": self.trend,
            "source": self.source,
            "degraded": self.degraded,
            "degradation_note": self.degradation_note,
            "confidence": self.confidence,
        }


class FundFlowProxy:
    """三层容灾的资金流向代理"""

    ETF_PROXIES = [
        ("510300.SH", "沪深300ETF", 1.0),
        ("510500.SH", "中证500ETF", 0.6),
        ("159919.SZ", "创业板ETF", 0.4),
    ]

    def __init__(self, fetcher):
        self.fetcher = fetcher

    def get_flow(self, days: int = 20, timeout: float = 5.0) -> FundFlowResult:
        # timeout 参数为预留接口，当前由 fetcher 内部节流和重试控制
        _ = timeout

        result = self._try_northbound(days)
        if result is not None:
            return result

        result = self._try_etf_flow(days)
        if result is not None:
            return result

        result = self._try_margin_enhanced(days)
        if result is not None:
            return result

        return FundFlowResult(
            latest_flow=0,
            flow_3d=0,
            flow_5d=0,
            flow_20d=0,
            trend="未知",
            source="none",
            degraded=True,
            degradation_note="北向/ETF/两融接口全部失败, 资金流向未知",
            confidence=0.0,
        )

    def _try_northbound(self, days: int) -> Optional[FundFlowResult]:
        try:
            df = self.fetcher.get_north_flow(days=days)
            if df is None or df.empty or "north_money_yi" not in df.columns:
                return None

            valid = pd.to_numeric(df["north_money_yi"], errors="coerce").dropna()
            if len(valid) < min(3, days):
                return None

            latest = float(valid.iloc[-1])
            flow_3d = float(valid.tail(3).sum())
            flow_5d = float(valid.tail(5).sum())
            flow_20d = float(valid.tail(min(20, len(valid))).sum())
            trend = self._classify_trend(flow_20d, flow_5d)

            return FundFlowResult(
                latest_flow=round(latest, 2),
                flow_3d=round(flow_3d, 2),
                flow_5d=round(flow_5d, 2),
                flow_20d=round(flow_20d, 2),
                trend=trend,
                source="northbound",
                degraded=False,
                degradation_note="",
                confidence=1.0,
            )
        except Exception as e:
            print(f"[FundFlowProxy] 北向接口失败: {e}")
            return None

    def _try_etf_flow(self, days: int) -> Optional[FundFlowResult]:
        try:
            flow_map = {}
            any_ok = False

            for etf_code, _name, weight in self.ETF_PROXIES:
                share_df = self._get_etf_share_change(etf_code, days)
                if share_df is None or share_df.empty:
                    continue

                px_df = self.fetcher.get_stock_daily(etf_code, days=days + 5)
                if px_df is None or px_df.empty or "close" not in px_df.columns:
                    continue

                s = share_df.sort_values("trade_date").set_index("trade_date")
                p = px_df.sort_values("trade_date").set_index("trade_date")[["close"]]
                merged = s.join(p, how="inner")
                if merged.empty:
                    continue

                merged["fd_share"] = pd.to_numeric(merged["fd_share"], errors="coerce")
                merged["close"] = pd.to_numeric(merged["close"], errors="coerce")
                merged = merged.dropna(subset=["fd_share", "close"])
                if len(merged) < 3:
                    continue

                # 万份 * 元 / 1e4 => 亿
                merged["flow_yi"] = merged["fd_share"].diff() * merged["close"] / 1e4
                merged["flow_yi"] = merged["flow_yi"] * weight

                for d, v in merged["flow_yi"].dropna().items():
                    flow_map[d] = float(flow_map.get(d, 0.0) + float(v))
                any_ok = True
                time.sleep(0.05)

            if not any_ok or not flow_map:
                return None

            ser = pd.Series(flow_map).sort_index()
            valid = pd.to_numeric(ser, errors="coerce").dropna()
            if len(valid) < 3:
                return None

            latest = float(valid.iloc[-1])
            flow_3d = float(valid.tail(3).sum())
            flow_5d = float(valid.tail(5).sum())
            flow_20d = float(valid.tail(min(20, len(valid))).sum())
            trend = self._classify_trend(flow_20d, flow_5d)

            return FundFlowResult(
                latest_flow=round(latest, 2),
                flow_3d=round(flow_3d, 2),
                flow_5d=round(flow_5d, 2),
                flow_20d=round(flow_20d, 2),
                trend=trend,
                source="etf_fund_flow",
                degraded=True,
                degradation_note="北向接口不可用, 使用ETF资金流替代 (置信度80%)",
                confidence=0.8,
            )
        except Exception as e:
            print(f"[FundFlowProxy] ETF资金流获取失败: {e}")
            return None

    def _get_etf_share_change(self, etf_code: str, days: int) -> Optional[pd.DataFrame]:
        try:
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=days + 15)).strftime("%Y%m%d")
            if hasattr(self.fetcher, "_throttle"):
                self.fetcher._throttle()
            df = self.fetcher.pro.fund_share(
                ts_code=etf_code,
                start_date=start_date,
                end_date=end_date,
                fields="ts_code,trade_date,fd_share",
            )
            if df is not None and not df.empty:
                return df.sort_values("trade_date")
            return None
        except Exception:
            return None

    def _try_margin_enhanced(self, days: int) -> Optional[FundFlowResult]:
        try:
            df = self.fetcher.get_margin_data(days=days)
            if df is None or df.empty or "rzye" not in df.columns:
                return None

            df = df.sort_values("trade_date").reset_index(drop=True)
            df["rzye"] = pd.to_numeric(df["rzye"], errors="coerce")
            df["rz_change_yi"] = df["rzye"].diff() / 1e8
            valid = df["rz_change_yi"].dropna()
            if len(valid) < 3:
                return None

            latest = float(valid.iloc[-1])
            flow_3d = float(valid.tail(3).sum())
            flow_5d = float(valid.tail(5).sum())
            flow_20d = float(valid.tail(min(20, len(valid))).sum())
            trend = self._classify_trend(flow_20d, flow_5d)

            return FundFlowResult(
                latest_flow=round(latest, 2),
                flow_3d=round(flow_3d, 2),
                flow_5d=round(flow_5d, 2),
                flow_20d=round(flow_20d, 2),
                trend=trend,
                source="margin_enhanced",
                degraded=True,
                degradation_note="北向+ETF均不可用, 使用融资余额变化替代 (置信度60%)",
                confidence=0.6,
            )
        except Exception as e:
            print(f"[FundFlowProxy] 两融增强失败: {e}")
            return None

    def _classify_trend(self, flow_20d: float, flow_5d: float) -> str:
        _ = flow_5d
        if flow_20d > 200:
            return "持续流入"
        if flow_20d > 50:
            return "温和流入"
        if flow_20d > -50:
            return "中性"
        if flow_20d > -200:
            return "温和流出"
        return "持续流出"
