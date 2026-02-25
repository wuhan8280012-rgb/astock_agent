"""
market_environment.py — 市场环境评估模块 v6.1
===============================================
v6.0 → v6.1 升级：解决盘中评分失真问题。

v6.1 修正内容：
  1. 趋势维度新增"当日动量"因子 — 节后涨1%但MA还没翻上来也能加分
  2. 支持盘中数据覆盖(intraday) — 不依赖Tushare收盘后更新
  3. 权重调整 — 降低趋势权重(MA滞后)，提高情绪权重(更实时)
  4. 成交量绝对水平兜底 — 万亿成交额直接给高分，不只看量比
  5. 诊断模式 — 每个维度给出计算过程

评估维度（4+1）：
  1. 大盘趋势 (0-100)：均线系统 + 当日动量
  2. 市场情绪 (0-100)：涨跌比、涨停跌停、北向资金
  3. 成交量   (0-100)：量比 + 绝对水平
  4. 板块强度 (0-100)：领涨板块数、板块轮动活跃度

权重（v6.1 调整）：
  趋势 0.25（↓ 从0.30降，MA滞后）
  情绪 0.30（↑ 从0.25升，更实时）
  成交量 0.20（不变）
  板块 0.25（不变）

综合评分 → 交易建议：
  ≥75: 良好 → 可交易
  60-74: 一般 → 谨慎交易
  <60: 较差 → 建议观望

用法：
  # 收盘后（自动从Tushare取数据）
  env = MarketEnvironment(data_fetcher)
  result = env.evaluate(date="20260224")

  # 盘中（手动输入实时数据）
  result = env.evaluate(date="20260224", intraday={
      "index_pct": 1.01,       # 上证涨幅%
      "up_count": 4186,        # 涨家数
      "down_count": 1215,      # 跌家数
      "limit_up": 61,          # 涨停
      "limit_down": 16,        # 跌停
      "amount_billion": 10284, # 成交额(亿)
      "north_flow": 75.89,     # 北向净流入(亿)
  })

  # 诊断模式
  result = env.evaluate(date="20260224", diagnose=True)
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Any


class MarketEnvironment:
    """
    市场环境评估器 v6.1

    输出示例：
        市场环境: 74/100 (一般) - 谨慎交易
          大盘趋势: 70/100
          市场情绪: 81/100
          成交量:   72/100
          板块强度: 75/100
    """

    # v6.1 权重（趋势降、情绪升）
    WEIGHTS = {
        "trend": 0.25,
        "sentiment": 0.30,
        "volume": 0.20,
        "sector": 0.25,
    }

    def __init__(self, data_fetcher=None):
        self.fetcher = data_fetcher

    def evaluate(self, date: str = None, index_code: str = "000001.SH",
                 intraday: dict = None, diagnose: bool = False) -> dict:
        """
        评估市场环境

        Parameters
        ----------
        date : 评估日期 YYYYMMDD
        index_code : 参考指数
        intraday : 盘中实时数据（覆盖Tushare延迟数据）
            {
                "index_pct": float,       # 指数涨幅%
                "up_count": int,          # 涨家数
                "down_count": int,        # 跌家数
                "limit_up": int,          # 涨停
                "limit_down": int,        # 跌停
                "amount_billion": float,  # 成交额(亿)
                "north_flow": float,      # 北向净流入(亿)
            }
        diagnose : 是否输出诊断信息

        Returns
        -------
        dict: scores, total_score, level, advice, summary, details
        """
        date = date or datetime.now().strftime("%Y%m%d")
        intraday = intraday or {}
        log = []  # 诊断日志

        # 各维度评分
        trend_score, trend_detail = self._evaluate_trend(
            date, index_code, intraday, log)
        sentiment_score, sentiment_detail = self._evaluate_sentiment(
            date, intraday, log)
        volume_score, volume_detail = self._evaluate_volume(
            date, index_code, intraday, log)
        sector_score, sector_detail = self._evaluate_sector(
            date, intraday, log)

        scores = {
            "trend": trend_score,
            "sentiment": sentiment_score,
            "volume": volume_score,
            "sector": sector_score,
        }

        # 加权综合评分
        W = self.WEIGHTS
        total = int(
            scores["trend"] * W["trend"] +
            scores["sentiment"] * W["sentiment"] +
            scores["volume"] * W["volume"] +
            scores["sector"] * W["sector"]
        )

        # 评级
        if total >= 75:
            level, advice = "良好", "可交易"
        elif total >= 60:
            level, advice = "一般", "谨慎交易"
        else:
            level, advice = "较差", "建议观望"

        summary = f"市场环境: {total}/100 ({level}) - {advice}"

        # 数据来源标记
        source = "intraday" if intraday else "tushare"

        result = {
            "date": date,
            "version": "v6.1",
            "source": source,
            "scores": scores,
            "total_score": total,
            "level": level,
            "advice": advice,
            "summary": summary,
            "weights": dict(W),
            "details": {
                "trend": trend_detail,
                "sentiment": sentiment_detail,
                "volume": volume_detail,
                "sector": sector_detail,
            },
        }

        if diagnose:
            result["diagnose_log"] = log
            self._print_diagnose(result, log)

        return result

    def format_report(self, result: dict) -> str:
        """格式化为可读报告"""
        s = result["scores"]
        src = result.get("source", "?")
        lines = [
            result["summary"],
            f"  大盘趋势: {s['trend']}/100",
            f"  市场情绪: {s['sentiment']}/100",
            f"  成交量:   {s['volume']}/100",
            f"  板块强度: {s['sector']}/100",
            f"  (数据源: {src}, 版本: {result.get('version', 'v6.0')})",
        ]
        return "\n".join(lines)

    def to_brief(self, result: dict) -> str:
        """压缩评分 (~30 tokens)"""
        s = result["scores"]
        return (
            f"[环境] {result['total_score']}/100({result['level']}) "
            f"趋势{s['trend']} 情绪{s['sentiment']} 量能{s['volume']} 板块{s['sector']} "
            f"→ {result['advice']}"
        )

    # ─────────────────────────────────────────
    # 维度 1: 大盘趋势 (v6.1: +当日动量)
    # ─────────────────────────────────────────

    def _evaluate_trend(self, date, index_code, intraday, log) -> tuple:
        detail = {}
        try:
            if not self.fetcher:
                return 65, {"note": "无数据源"}

            df = self._get_index_data(index_code, date, days=60)
            if df is None or len(df) < 20:
                return 65, {"note": "数据不足"}

            close = df["close"].values
            current = close[-1]

            # 均线
            ma5 = float(np.mean(close[-5:]))
            ma10 = float(np.mean(close[-10:]))
            ma20 = float(np.mean(close[-20:]))
            ma60 = float(np.mean(close[-60:])) if len(close) >= 60 else ma20

            above_count = sum([
                current > ma5, current > ma10,
                current > ma20, current > ma60,
            ])

            pct_5d = (current / close[-6] - 1) * 100 if len(close) >= 6 else 0
            bullish_order = ma5 > ma10 > ma20

            # 基础评分（与v6.0相同）
            score = 50
            score += above_count * 8
            score += min(10, max(-10, pct_5d * 2))
            if bullish_order:
                score += 8

            log.append(f"趋势基础: 50 + MA上方{above_count}×8 + 5日{pct_5d:.1f}%×2 + 多头{'8' if bullish_order else '0'} = {score}")

            # ── v6.1 新增: 当日动量修正 ──
            index_pct = intraday.get("index_pct")
            if index_pct is not None:
                if index_pct > 2.0:
                    momentum = 20
                elif index_pct > 1.0:
                    momentum = 15
                elif index_pct > 0.5:
                    momentum = 10
                elif index_pct > 0:
                    momentum = 5
                elif index_pct > -0.5:
                    momentum = 0
                elif index_pct > -1.0:
                    momentum = -10
                elif index_pct > -2.0:
                    momentum = -15
                else:
                    momentum = -20

                score += momentum
                log.append(f"趋势动量: 指数{index_pct:+.2f}% → +{momentum}")
                detail["intraday_pct"] = index_pct
                detail["momentum_bonus"] = momentum

            score = max(0, min(100, int(score)))

            detail.update({
                "close": round(current, 2),
                "ma5": round(ma5, 2),
                "ma10": round(ma10, 2),
                "ma20": round(ma20, 2),
                "ma60": round(ma60, 2),
                "above_ma_count": above_count,
                "pct_5d": round(pct_5d, 2),
                "bullish_order": bullish_order,
            })
            log.append(f"趋势最终: {score}")
            return score, detail

        except Exception as e:
            return 65, {"error": str(e)}

    # ─────────────────────────────────────────
    # 维度 2: 市场情绪 (v6.1: 支持盘中覆盖)
    # ─────────────────────────────────────────

    def _evaluate_sentiment(self, date, intraday, log) -> tuple:
        detail = {}
        score = 50

        # 优先用盘中数据
        up = intraday.get("up_count")
        down = intraday.get("down_count")
        limit_up = intraday.get("limit_up")
        limit_down = intraday.get("limit_down")
        north_flow = intraday.get("north_flow")
        fund_source = "intraday" if north_flow is not None else None

        # 不在 intraday 里的字段，回退到 Tushare
        if up is None or down is None:
            stats = self._get_market_stats(date)
            if stats:
                up = up if up is not None else stats.get("up_count", 0)
                down = down if down is not None else stats.get("down_count", 0)
                limit_up = limit_up if limit_up is not None else stats.get("limit_up", 0)
                limit_down = limit_down if limit_down is not None else stats.get("limit_down", 0)
                if north_flow is None and "north_flow" in stats:
                    north_flow = stats.get("north_flow")
                    fund_source = "tushare"
            else:
                log.append("情绪: 无数据（Tushare+intraday均无）→ 默认60")
                return 60, {"note": "无数据"}
        elif north_flow is None:
            # up/down 已有盘中数据，但 north_flow 可能缺失，补拉 market_stats
            stats = self._get_market_stats(date)
            if stats and "north_flow" in stats:
                north_flow = stats.get("north_flow")
                fund_source = "tushare"

        # 涨跌比 (v6.1: 放大系数 40→60，涨跌比信号更强)
        total_stocks = (up or 0) + (down or 0)
        if total_stocks > 0:
            up_ratio = up / total_stocks
            score += int((up_ratio - 0.5) * 60)
            log.append(f"情绪涨跌比: {up}/{down} = {up_ratio:.3f} → +{int((up_ratio-0.5)*60)}")

        # 涨停跌停 (v6.1: 加大分值)
        if limit_up is not None:
            if limit_up > 50:
                score += 12
            elif limit_up > 30:
                score += 6
            elif limit_up > 15:
                score += 3
            log.append(f"情绪涨停: {limit_up} → +{12 if limit_up>50 else 6 if limit_up>30 else 3 if limit_up>15 else 0}")

        if limit_down is not None:
            if limit_down > 30:
                score -= 12
            elif limit_down > 15:
                score -= 5
            log.append(f"情绪跌停: {limit_down} → {-12 if limit_down>30 else -5 if limit_down>15 else 0}")

        # 资金流向 (v6.2: 北向失败时三层降级)
        if north_flow is None:
            try:
                from fund_flow_proxy import FundFlowProxy
                proxy = FundFlowProxy(self.fetcher)
                fund_result = proxy.get_flow(days=5)
                if fund_result.source != "none":
                    north_flow = fund_result.latest_flow
                    fund_source = fund_result.source
                else:
                    fund_source = "none"
            except Exception as e:
                fund_source = "none"
                log.append(f"情绪资金流: Proxy失败 {e}")

        if north_flow is not None:
            if north_flow > 80:
                bonus = 12
            elif north_flow > 30:
                bonus = 8
            elif north_flow > 0:
                bonus = 4
            elif north_flow > -30:
                bonus = 0
            elif north_flow > -80:
                bonus = -8
            else:
                bonus = -12

            # ETF/两融替代时降低影响幅度
            if fund_source not in ("intraday", "northbound", "tushare", None):
                bonus = int(round(bonus * 0.7))

            score += bonus
            log.append(f"情绪资金流: {north_flow}亿 [{fund_source}] → +{bonus}")
        else:
            # 显式标记缺失并做轻微负向补偿，防虚高
            log.append("情绪资金流: 全部数据源失败 ⚠️ 评分可能偏高")
            score -= 3
            log.append("情绪资金流: 缺失补偿 -3")

        score = max(0, min(100, score))

        detail = {
            "up_count": up,
            "down_count": down,
            "limit_up": limit_up,
            "limit_down": limit_down,
            "north_flow": north_flow,
            "fund_source": fund_source,
            "source": "intraday" if intraday.get("up_count") is not None else "tushare",
        }
        log.append(f"情绪最终: {score}")
        return score, detail

    # ─────────────────────────────────────────
    # 维度 3: 成交量 (v6.1: +绝对水平兜底)
    # ─────────────────────────────────────────

    def _evaluate_volume(self, date, index_code, intraday, log) -> tuple:
        detail = {}
        score = 50

        # 方式1: 盘中输入成交额
        amount_b = intraday.get("amount_billion")
        if amount_b is not None:
            # v6.2: 时间因子修正 — 将盘中成交额换算为全天预估
            projected = self._project_full_day_amount(amount_b)
            actual_for_score = projected if projected > amount_b else amount_b

            # 绝对水平评分（A股正常 8000-12000 亿，用全天预估值）
            if actual_for_score > 15000:
                score = 90
            elif actual_for_score > 12000:
                score = 80
            elif actual_for_score > 10000:
                score = 72
            elif actual_for_score > 8000:
                score = 65
            elif actual_for_score > 6000:
                score = 55
            else:
                score = 40

            log.append(f"成交量(盘中): 实际{amount_b:.0f}亿 预估全天{actual_for_score:.0f}亿 → {score}")
            detail = {
                "amount_billion": amount_b,
                "projected_full_day": round(actual_for_score),
                "source": "intraday",
                "method": "time_adjusted",
            }
            return score, detail

        # 方式2: Tushare 量比 + 全市场成交额
        try:
            df = self._get_index_data(index_code, date, days=25)
            if df is not None and len(df) >= 20:
                vol_col = "amount" if "amount" in df.columns else "vol"
                if vol_col in df.columns:
                    volumes = df[vol_col].values
                    today_vol = volumes[-1]
                    avg_20 = float(np.mean(volumes[-20:]))
                    vol_ratio = today_vol / avg_20 if avg_20 > 0 else 1.0

                    # 量比评分
                    if vol_ratio > 1.5:
                        score = 80
                    elif vol_ratio > 1.3:
                        score = 70
                    elif vol_ratio > 1.0:
                        score = 60
                    elif vol_ratio > 0.8:
                        score = 50
                    elif vol_ratio > 0.7:
                        score = 42
                    else:
                        score = 35

                    # 绝对水平兜底：用全市场成交额（亿元）
                    # index_daily amount 是上证单一交易所（千元），不可靠
                    # 优先从 market_stats 拿全市场成交额
                    full_market_billion = 0
                    stats = self._get_market_stats(date)
                    if stats and stats.get("amount_billion"):
                        full_market_billion = float(stats["amount_billion"])
                    else:
                        # 兜底：index amount 千元→亿元，×2 粗估全市场
                        full_market_billion = today_vol / 1e5 * 2

                    if full_market_billion > 15000:
                        score = max(score, 85)
                    elif full_market_billion > 12000:
                        score = max(score, 75)
                    elif full_market_billion > 10000:
                        score = max(score, 68)
                    elif full_market_billion > 8000:
                        score = max(score, 62)

                    # 量能趋势
                    if len(volumes) >= 3:
                        if volumes[-1] > volumes[-2] > volumes[-3]:
                            score = min(100, score + 8)
                        elif volumes[-1] < volumes[-2] < volumes[-3]:
                            score = max(0, score - 8)

                    log.append(f"成交量(Tushare): 量比{vol_ratio:.2f} 全市场{full_market_billion:.0f}亿 → {score}")
                    detail = {
                        "today_amount": round(today_vol, 0),
                        "avg_20_amount": round(avg_20, 0),
                        "vol_ratio": round(vol_ratio, 2),
                        "full_market_billion": round(full_market_billion),
                        "source": "tushare",
                    }
                    return max(0, min(100, score)), detail
        except Exception as e:
            log.append(f"成交量异常: {e}")

        log.append(f"成交量: 无数据 → 默认65")
        return 65, {"note": "无数据"}

    # ─────────────────────────────────────────
    # 维度 4: 板块强度
    # ─────────────────────────────────────────

    def _evaluate_sector(self, date, intraday, log) -> tuple:
        detail = {}
        try:
            sectors = self._get_sector_data(date)
            if sectors:
                strong = sum(1 for s in sectors if s.get("pct_change", 0) > 2)
                weak = sum(1 for s in sectors if s.get("pct_change", 0) < -2)
                total_s = len(sectors)

                score = 50
                if total_s > 0:
                    score += int(strong / total_s * 30)
                    score -= int(weak / total_s * 20)
                if strong >= 5:
                    score += 10

                score = max(0, min(100, score))
                log.append(f"板块: 强{strong}/弱{weak}/总{total_s} → {score}")
                detail = {
                    "strong_sectors": strong,
                    "weak_sectors": weak,
                    "total_sectors": total_s,
                    "source": "tushare",
                }
                return score, detail

        except Exception:
            pass

        # 回退: 用涨跌比推算
        up = intraday.get("up_count", 0)
        down = intraday.get("down_count", 0)
        if up + down > 0:
            ratio = up / (up + down)
            score = int(40 + ratio * 40)
            log.append(f"板块(推算): 涨跌比{ratio:.2f} → {score}")
            return max(0, min(100, score)), {"source": "inferred_from_breadth"}

        log.append("板块: 无数据 → 默认60")
        return 60, {"note": "无板块数据"}

    # ─────────────────────────────────────────
    # 诊断输出
    # ─────────────────────────────────────────

    def _print_diagnose(self, result, log):
        """打印诊断信息"""
        print("\n" + "=" * 60)
        print(f"  环境评分诊断 v6.1 ({result['date']})")
        print("=" * 60)

        for line in log:
            print(f"  {line}")

        s = result["scores"]
        W = self.WEIGHTS
        print(f"\n  综合计算:")
        for k in ["trend", "sentiment", "volume", "sector"]:
            name = {"trend": "趋势", "sentiment": "情绪",
                    "volume": "成交量", "sector": "板块"}[k]
            print(f"    {name:4s} {s[k]:3d} × {W[k]:.2f} = {s[k]*W[k]:.0f}")
        print(f"    {'─'*30}")
        print(f"    {result['summary']}")
        print("=" * 60)

    # ─────────────────────────────────────────
    # 时间因子
    # ─────────────────────────────────────────

    @staticmethod
    def _project_full_day_amount(current_amount: float) -> float:
        """
        将盘中成交额换算为全天预估

        A股交易时间 4 小时（9:30-11:30 + 13:00-15:00）= 240分钟
        各时段成交量占比（经验值）:
          09:30-10:30  ~30%
          10:30-11:30  ~22%
          13:00-14:00  ~25%
          14:00-15:00  ~23%

        通过当前时间推算已过交易时间占比，再反推全天量
        """
        from datetime import datetime
        now = datetime.now()
        h, m = now.hour, now.minute

        # 计算已过的交易分钟数
        if h < 9 or (h == 9 and m < 30):
            elapsed_pct = 0.05  # 未开盘，给5%兜底
        elif h < 11 or (h == 11 and m <= 30):
            # 上午场 9:30-11:30
            minutes = (h - 9) * 60 + m - 30
            minutes = max(0, min(120, minutes))
            # 上午占全天 ~52%
            elapsed_pct = (minutes / 120) * 0.52
        elif h < 13:
            # 午休 11:30-13:00
            elapsed_pct = 0.52
        elif h < 15:
            # 下午场 13:00-15:00
            minutes = (h - 13) * 60 + m
            minutes = max(0, min(120, minutes))
            # 下午占全天 ~48%
            elapsed_pct = 0.52 + (minutes / 120) * 0.48
        else:
            # 收盘后
            elapsed_pct = 1.0

        if elapsed_pct <= 0.05:
            return current_amount

        projected = current_amount / elapsed_pct
        return round(projected)

    # ─────────────────────────────────────────
    # 数据获取（适配 DataFetcher）
    # ─────────────────────────────────────────

    def _get_index_data(self, index_code, date, days=60):
        if not self.fetcher:
            return None
        if hasattr(self.fetcher, 'get_index_daily'):
            try:
                return self.fetcher.get_index_daily(index_code, days=days, end_date=date)
            except Exception:
                pass
        if hasattr(self.fetcher, 'get_daily'):
            try:
                return self.fetcher.get_daily(index_code, days=days, end_date=date)
            except Exception:
                pass
        return None

    def _get_market_stats(self, date):
        if not self.fetcher:
            return None
        for method in ['get_market_stats', 'get_market_overview']:
            if hasattr(self.fetcher, method):
                try:
                    return getattr(self.fetcher, method)(date)
                except Exception:
                    continue
        return None

    def _get_sector_data(self, date):
        if not self.fetcher:
            return None
        for method in ['get_sector_performance', 'get_sectors']:
            if hasattr(self.fetcher, method):
                try:
                    return getattr(self.fetcher, method)(date)
                except Exception:
                    continue
        return None


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def main():
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="市场环境评估 v6.1")
    parser.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--index-pct", type=float, help="指数涨幅%%")
    parser.add_argument("--up", type=int, help="涨家数")
    parser.add_argument("--down", type=int, help="跌家数")
    parser.add_argument("--amount", type=float, help="成交额(亿)")
    parser.add_argument("--limit-up", type=int, help="涨停")
    parser.add_argument("--limit-down", type=int, help="跌停")
    parser.add_argument("--north", type=float, help="北向净流入(亿)")
    parser.add_argument("--diagnose", action="store_true", help="诊断模式")
    args = parser.parse_args()

    # 构建 intraday
    intraday = {}
    if args.index_pct is not None:
        intraday["index_pct"] = args.index_pct
    if args.up is not None:
        intraday["up_count"] = args.up
    if args.down is not None:
        intraday["down_count"] = args.down
    if args.amount is not None:
        intraday["amount_billion"] = args.amount
    if args.limit_up is not None:
        intraday["limit_up"] = args.limit_up
    if args.limit_down is not None:
        intraday["limit_down"] = args.limit_down
    if args.north is not None:
        intraday["north_flow"] = args.north

    # 初始化
    fetcher = None
    try:
        sys.path.insert(0, ".")
        from data_fetcher import DataFetcher
        fetcher = DataFetcher()
    except Exception as e:
        print(f"⚠️ DataFetcher: {e}")

    env = MarketEnvironment(fetcher)
    result = env.evaluate(
        date=args.date,
        intraday=intraday if intraday else None,
        diagnose=True,
    )
    print("\n" + env.format_report(result))


if __name__ == "__main__":
    main()
