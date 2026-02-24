"""
value_investor.py — 6维度价值投资分析 (Skill #9)
==================================================
原始产出：阶段③ 参考方案对标

6维度评估：
  1. 估值水平 (PE/PB/PS 与行业对比)
  2. 盈利质量 (ROE 连续性、毛利率趋势)
  3. 成长性   (营收/净利润增速，连续增长季度数)
  4. 财务健康 (资产负债率、现金流)
  5. 分红回购 (股息率、回购力度)
  6. 机构认可 (北向持仓、基金重仓)

输出：综合评分 0-100，各维度子分 0-100
"""

from datetime import datetime
from typing import Optional, Any


class ValueInvestorSkill:
    """
    价值投资分析 Skill

    用法：
        skill = ValueInvestorSkill(data_fetcher)
        result = skill.analyze("600519.SH")  # 单股深度分析
        signals = skill.scan("20260224")      # 批量扫描低估值
    """

    def __init__(self, data_fetcher=None):
        self.fetcher = data_fetcher

    # ─────────────────────────────────────────
    # 单股深度分析
    # ─────────────────────────────────────────

    def analyze(self, symbol: str, date: str = None) -> dict:
        """
        6 维度深度分析

        Returns
        -------
        {
            "symbol": "600519.SH",
            "name": "贵州茅台",
            "total_score": 78,
            "dimensions": {
                "valuation": {"score": 65, "PE": 28.5, "PB": 9.2, ...},
                "profitability": {"score": 90, "ROE": 32.5, ...},
                "growth": {"score": 72, "revenue_growth": 15.2, ...},
                "health": {"score": 85, "debt_ratio": 22.1, ...},
                "dividend": {"score": 70, "yield": 2.1, ...},
                "institution": {"score": 80, "north_hold_pct": 8.5, ...},
            },
            "conclusion": "优质价值股，估值偏高但盈利质量极佳",
            "action": "WATCH",
        }
        """
        date = date or datetime.now().strftime("%Y%m%d")

        dims = {}
        dims["valuation"] = self._eval_valuation(symbol, date)
        dims["profitability"] = self._eval_profitability(symbol, date)
        dims["growth"] = self._eval_growth(symbol, date)
        dims["health"] = self._eval_health(symbol, date)
        dims["dividend"] = self._eval_dividend(symbol, date)
        dims["institution"] = self._eval_institution(symbol, date)

        # 加权总分
        weights = {
            "valuation": 0.20,
            "profitability": 0.20,
            "growth": 0.20,
            "health": 0.15,
            "dividend": 0.10,
            "institution": 0.15,
        }
        total = sum(dims[k]["score"] * weights[k] for k in weights)
        total = int(total)

        # 结论
        action, conclusion = self._make_conclusion(total, dims)

        return {
            "symbol": symbol,
            "name": self._get_name(symbol),
            "total_score": total,
            "dimensions": dims,
            "conclusion": conclusion,
            "action": action,
        }

    # ─────────────────────────────────────────
    # 批量扫描
    # ─────────────────────────────────────────

    def scan(self, date: str = None, pool: list = None,
             min_score: int = 70) -> list:
        """
        扫描低估值+高质量标的

        Returns list of signal dicts (与 TradeSignalGenerator 格式兼容)
        """
        date = date or datetime.now().strftime("%Y%m%d")
        stocks = pool or self._get_value_pool()
        signals = []

        for symbol in stocks:
            try:
                result = self.analyze(symbol, date)
                if result["total_score"] >= min_score:
                    signals.append({
                        "type": "value_undervalued",
                        "symbol": symbol,
                        "name": result["name"],
                        "score": result["total_score"],
                        "action": result["action"],
                        "price": {},
                        "confidence": round(result["total_score"] / 100, 2),
                        "reasons": self._extract_reasons(result),
                        "risk_notes": [],
                        "value_detail": result["dimensions"],
                    })
            except Exception:
                continue

        return sorted(signals, key=lambda s: -s["score"])

    # ─────────────────────────────────────────
    # 6 维度评估
    # ─────────────────────────────────────────

    def _eval_valuation(self, symbol: str, date: str) -> dict:
        """估值水平：PE/PB/PS 与行业对比"""
        result = {"score": 60, "note": "默认"}
        try:
            data = self._get_fundamental(symbol, "valuation")
            if not data:
                return result

            pe = data.get("pe", 0)
            pb = data.get("pb", 0)
            industry_pe = data.get("industry_pe", pe)

            score = 50
            # PE 低于行业均值 → 加分
            if pe > 0 and industry_pe > 0:
                ratio = pe / industry_pe
                if ratio < 0.7:
                    score += 25
                elif ratio < 0.9:
                    score += 15
                elif ratio > 1.3:
                    score -= 15
            # PB
            if 0 < pb < 1.5:
                score += 15
            elif pb < 3:
                score += 5
            elif pb > 8:
                score -= 10

            result = {
                "score": max(0, min(100, score)),
                "PE": round(pe, 1),
                "PB": round(pb, 1),
                "industry_PE": round(industry_pe, 1),
            }
        except Exception:
            pass
        return result

    def _eval_profitability(self, symbol: str, date: str) -> dict:
        """盈利质量：ROE、毛利率"""
        result = {"score": 60, "note": "默认"}
        try:
            data = self._get_fundamental(symbol, "profitability")
            if not data:
                return result

            roe = data.get("roe", 0)
            gross_margin = data.get("gross_margin", 0)

            score = 50
            if roe > 20:
                score += 25
            elif roe > 15:
                score += 15
            elif roe > 10:
                score += 5
            elif roe < 5:
                score -= 15

            if gross_margin > 50:
                score += 15
            elif gross_margin > 30:
                score += 5

            result = {
                "score": max(0, min(100, score)),
                "ROE": round(roe, 1),
                "gross_margin": round(gross_margin, 1),
            }
        except Exception:
            pass
        return result

    def _eval_growth(self, symbol: str, date: str) -> dict:
        """成长性：营收/净利润增速"""
        result = {"score": 60, "note": "默认"}
        try:
            data = self._get_fundamental(symbol, "growth")
            if not data:
                return result

            rev_growth = data.get("revenue_growth", 0)
            profit_growth = data.get("profit_growth", 0)
            consecutive_quarters = data.get("consecutive_growth_quarters", 0)

            score = 50
            if profit_growth > 30:
                score += 20
            elif profit_growth > 15:
                score += 10
            elif profit_growth < 0:
                score -= 15

            if rev_growth > 20:
                score += 10
            if consecutive_quarters >= 4:
                score += 10

            result = {
                "score": max(0, min(100, score)),
                "revenue_growth": round(rev_growth, 1),
                "profit_growth": round(profit_growth, 1),
                "consecutive_quarters": consecutive_quarters,
            }
        except Exception:
            pass
        return result

    def _eval_health(self, symbol: str, date: str) -> dict:
        """财务健康：资产负债率、现金流"""
        result = {"score": 65, "note": "默认"}
        try:
            data = self._get_fundamental(symbol, "health")
            if not data:
                return result

            debt_ratio = data.get("debt_ratio", 50)
            ocf = data.get("operating_cash_flow", 0)

            score = 50
            if debt_ratio < 30:
                score += 20
            elif debt_ratio < 50:
                score += 10
            elif debt_ratio > 70:
                score -= 20

            if ocf > 0:
                score += 15
            else:
                score -= 10

            result = {
                "score": max(0, min(100, score)),
                "debt_ratio": round(debt_ratio, 1),
                "cash_flow_positive": ocf > 0,
            }
        except Exception:
            pass
        return result

    def _eval_dividend(self, symbol: str, date: str) -> dict:
        """分红回购：股息率"""
        result = {"score": 55, "note": "默认"}
        try:
            data = self._get_fundamental(symbol, "dividend")
            if not data:
                return result

            div_yield = data.get("dividend_yield", 0)

            score = 50
            if div_yield > 4:
                score += 25
            elif div_yield > 2:
                score += 15
            elif div_yield > 1:
                score += 5

            result = {
                "score": max(0, min(100, score)),
                "dividend_yield": round(div_yield, 2),
            }
        except Exception:
            pass
        return result

    def _eval_institution(self, symbol: str, date: str) -> dict:
        """机构认可：北向、基金"""
        result = {"score": 60, "note": "默认"}
        try:
            data = self._get_fundamental(symbol, "institution")
            if not data:
                return result

            north_pct = data.get("north_hold_pct", 0)
            fund_count = data.get("fund_holder_count", 0)

            score = 50
            if north_pct > 5:
                score += 15
            elif north_pct > 2:
                score += 8
            if fund_count > 100:
                score += 15
            elif fund_count > 30:
                score += 8

            result = {
                "score": max(0, min(100, score)),
                "north_hold_pct": round(north_pct, 2),
                "fund_holder_count": fund_count,
            }
        except Exception:
            pass
        return result

    # ─────────────────────────────────────────
    # 辅助
    # ─────────────────────────────────────────

    def _make_conclusion(self, total, dims) -> tuple:
        vd = dims.get("valuation", {})
        pd = dims.get("profitability", {})

        if total >= 80:
            action = "BUY"
            conclusion = "高质量价值股，值得重点关注"
        elif total >= 70:
            if vd.get("score", 0) < 50:
                action = "WATCH"
                conclusion = "质地优秀但估值偏高，等待回调"
            else:
                action = "BUY"
                conclusion = "价值与质量兼具"
        elif total >= 60:
            action = "WATCH"
            conclusion = "中等质量，需进一步观察"
        else:
            action = "HOLD"
            conclusion = "综合评分较低，暂不推荐"

        return action, conclusion

    def _extract_reasons(self, result) -> list:
        reasons = []
        dims = result["dimensions"]
        if dims.get("profitability", {}).get("ROE", 0) > 15:
            reasons.append(f"ROE={dims['profitability']['ROE']}%")
        if dims.get("growth", {}).get("profit_growth", 0) > 15:
            reasons.append(f"净利+{dims['growth']['profit_growth']}%")
        if dims.get("valuation", {}).get("score", 0) > 70:
            reasons.append("估值偏低")
        if dims.get("dividend", {}).get("dividend_yield", 0) > 2:
            reasons.append(f"股息{dims['dividend']['dividend_yield']}%")
        return reasons or ["综合评分达标"]

    def _get_name(self, symbol):
        if self.fetcher and hasattr(self.fetcher, 'get_stock_name'):
            try:
                return self.fetcher.get_stock_name(symbol)
            except Exception:
                pass
        return symbol

    def _get_fundamental(self, symbol, aspect):
        if not self.fetcher:
            return None
        for method in [f'get_{aspect}', 'get_fundamental', 'get_finance_data']:
            if hasattr(self.fetcher, method):
                try:
                    return getattr(self.fetcher, method)(symbol)
                except Exception:
                    continue
        return None

    def _get_value_pool(self):
        if self.fetcher and hasattr(self.fetcher, 'get_stock_pool'):
            try:
                return self.fetcher.get_stock_pool()
            except Exception:
                pass
        return []
