"""
backtest_v6.py — v6.0 市场环境评分 vs 实际胜率 回测验证
=========================================================
核心命题：
  "知道何时不交易，比知道何时交易更重要"
  11月环境差 → v4.1 胜率18.9%(-6.63%) → v6.0 暂停 → 估计-2%
  12月环境好 → 保持 +5%
  全年改善 → +2.3pp

验证内容：
  1. 逐日计算市场环境评分（4维度）
  2. 逐日扫描阶梯突破信号
  3. 跟踪每个信号的实际盈亏（T+5 / T+10 / 止损）
  4. 按环境评分分桶，计算各桶胜率和收益
  5. 对比 v4.1（全做）vs v6.0（环境差时停手）

用法：
  # 快速回测（用你的 data_fetcher）
  python3 backtest_v6.py

  # 指定日期范围
  python3 backtest_v6.py --start 20251101 --end 20251231

  # 只看环境评分
  python3 backtest_v6.py --env-only

  # 导出详细数据
  python3 backtest_v6.py --export results.csv

程序化调用：
  from backtest_v6 import BacktestV6
  bt = BacktestV6(data_fetcher=fetcher)
  report = bt.run("20251101", "20251231")
  print(report["summary"])
"""

import json
import csv
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from typing import Optional, Any
import math


# ═══════════════════════════════════════════════════════════════
#  核心回测引擎
# ═══════════════════════════════════════════════════════════════

class BacktestV6:
    """
    v6.0 回测引擎

    流程：
      for each trading_day in [start, end]:
        1. 计算市场环境评分
        2. 扫描阶梯突破信号
        3. 记录信号入场价、止损价
        4. 跟踪后续 N 天表现
      统计 → 相关性分析 → v4.1 vs v6.0 对比
    """

    def __init__(self, data_fetcher=None, index_code: str = "000001.SH"):
        """
        Parameters
        ----------
        data_fetcher : DataFetcher 实例
        index_code : 大盘参考指数
        """
        self.fetcher = data_fetcher
        self.index_code = index_code

        # 回测参数
        self.hold_days = 10           # 持有天数（用于计算收益）
        self.stop_loss_pct = -7.0     # 止损线 %
        self.take_profit_pct = 15.0   # 止盈线 %
        self.min_score = 65           # 信号最低分
        self.env_threshold = 60       # v6.0 环境阈值（低于此分停手）

        # 结果存储
        self.daily_env = []           # [{date, scores, total_score, level, ...}]
        self.signals = []             # [{date, symbol, score, entry, stop, ...}]
        self.trades = []              # [{signal_info..., exit_price, pnl, hold_days, outcome}]

    # ─────────────────────────────────────────
    # 主入口
    # ─────────────────────────────────────────

    def run(self, start_date: str = "20251101", end_date: str = "20251231",
            stock_pool: list = None, verbose: bool = True) -> dict:
        """
        执行完整回测

        Returns
        -------
        {
            "period": "20251101-20251231",
            "trading_days": 42,
            "daily_env": [...],
            "total_signals": 85,
            "v41": {...},           # v4.1 全做的表现
            "v60": {...},           # v6.0 环境过滤后的表现
            "correlation": {...},   # 环境评分 vs 胜率相关性
            "buckets": {...},       # 按评分分桶统计
            "summary": "...",       # 文字报告
        }
        """
        if verbose:
            print("=" * 65)
            print("  v6.0 回测验证: 市场环境评分 vs 实际胜率")
            print(f"  {start_date} ~ {end_date}")
            print("=" * 65)

        # Step 1: 获取交易日历
        trading_days = self._get_trading_days(start_date, end_date)
        if verbose:
            print(f"\n交易日: {len(trading_days)} 天")

        # Step 2: 逐日评估环境 + 扫描信号
        if verbose:
            print("\n[Phase 1] 逐日评估环境 + 扫描信号...")

        for i, day in enumerate(trading_days):
            if verbose and (i + 1) % 10 == 0:
                print(f"  进度: {i+1}/{len(trading_days)} ({day})")

            # 环境评分
            env = self._evaluate_env(day)
            self.daily_env.append(env)

            # 信号扫描
            day_signals = self._scan_signals(day, stock_pool)
            self.signals.extend(day_signals)

        if verbose:
            print(f"  环境评分: {len(self.daily_env)} 天")
            print(f"  信号总数: {len(self.signals)} 个")

        # Step 3: 跟踪信号盈亏
        if verbose:
            print("\n[Phase 2] 跟踪信号盈亏...")

        # 需要额外数据：信号日之后 hold_days 天的价格
        end_ext = self._date_add(end_date, 20)  # 延伸 20 天获取后续价格
        for sig in self.signals:
            trade = self._track_trade(sig, end_ext)
            if trade:
                self.trades.append(trade)

        if verbose:
            print(f"  有效交易: {len(self.trades)} 笔")

        # Step 4: 分析
        if verbose:
            print("\n[Phase 3] 分析...")

        v41 = self._calc_v41_performance()
        v60 = self._calc_v60_performance()
        correlation = self._calc_correlation()
        buckets = self._calc_bucket_stats()
        monthly = self._calc_monthly_breakdown()

        # Step 5: 生成报告
        summary = self._generate_summary(
            trading_days, v41, v60, correlation, buckets, monthly
        )

        result = {
            "period": f"{start_date}-{end_date}",
            "trading_days": len(trading_days),
            "daily_env": self.daily_env,
            "total_signals": len(self.signals),
            "total_trades": len(self.trades),
            "v41": v41,
            "v60": v60,
            "correlation": correlation,
            "buckets": buckets,
            "monthly": monthly,
            "summary": summary,
        }

        if verbose:
            print("\n" + summary)

        return result

    def run_env_only(self, start_date: str, end_date: str) -> list:
        """只跑环境评分，不扫信号"""
        trading_days = self._get_trading_days(start_date, end_date)
        results = []
        for day in trading_days:
            env = self._evaluate_env(day)
            results.append(env)
            print(f"  {day}: {env['total_score']}/100 ({env['level']})")
        return results

    # ═══════════════════════════════════════════════════════════
    #  Step 2: 环境评估（复用 MarketEnvironment 逻辑）
    # ═══════════════════════════════════════════════════════════

    def _evaluate_env(self, date: str) -> dict:
        """
        计算某日的市场环境评分

        优先使用 MarketEnvironment 模块，不可用时用内置简化版。
        """
        # 尝试用正式模块
        try:
            from market_environment import MarketEnvironment
            me = MarketEnvironment(self.fetcher)
            return me.evaluate(date, self.index_code)
        except Exception:
            pass

        # 内置简化版
        return self._evaluate_env_builtin(date)

    def _evaluate_env_builtin(self, date: str) -> dict:
        """
        内置简化版环境评估

        用大盘数据直接计算 4 维评分，不依赖 MarketEnvironment 模块。
        """
        scores = {"trend": 65, "sentiment": 60, "volume": 65, "sector": 60}
        detail = {}

        try:
            df = self._get_index_daily(date, days=60)
            if df is not None and len(df) >= 20:
                close = df["close"].values
                current = close[-1]

                # ── 趋势 ──
                ma5 = _mean(close[-5:])
                ma10 = _mean(close[-10:])
                ma20 = _mean(close[-20:])

                above = sum([current > ma5, current > ma10, current > ma20])
                pct_5d = (current / close[-6] - 1) * 100 if len(close) >= 6 else 0
                bullish = ma5 > ma10 > ma20

                trend_score = 50 + above * 10 + min(10, max(-10, pct_5d * 2))
                if bullish:
                    trend_score += 8
                scores["trend"] = _clamp(trend_score)

                # ── 成交量 ──
                vol_col = "amount" if "amount" in df.columns else "vol"
                if vol_col in df.columns:
                    vols = df[vol_col].values
                    avg20 = _mean(vols[-20:])
                    ratio = vols[-1] / avg20 if avg20 > 0 else 1.0
                    vol_score = 50
                    if ratio > 1.3:
                        vol_score += 20
                    elif ratio > 1.0:
                        vol_score += 10
                    elif ratio < 0.7:
                        vol_score -= 15
                    scores["volume"] = _clamp(vol_score)
                    detail["vol_ratio"] = round(ratio, 2)

                # ── 情绪（用涨跌幅近5日波动代理）──
                if len(close) >= 10:
                    recent_pcts = [(close[i] / close[i-1] - 1) * 100
                                   for i in range(-5, 0)]
                    up_days = sum(1 for p in recent_pcts if p > 0)
                    avg_pct = _mean(recent_pcts)
                    sent_score = 50 + up_days * 6 + min(12, max(-12, avg_pct * 4))
                    scores["sentiment"] = _clamp(sent_score)

                # ── 板块（简化：用个股分散度代理）──
                # 如果有板块数据就用，没有就用趋势+情绪的均值
                scores["sector"] = _clamp(
                    int((scores["trend"] + scores["sentiment"]) / 2)
                )

                detail["close"] = round(current, 2)
                detail["ma20"] = round(ma20, 2)
                detail["pct_5d"] = round(pct_5d, 2)

        except Exception as e:
            detail["error"] = str(e)

        # v6.1 权重（趋势降、情绪升）
        total = int(
            scores["trend"] * 0.25 +
            scores["sentiment"] * 0.30 +
            scores["volume"] * 0.20 +
            scores["sector"] * 0.25
        )

        if total >= 75:
            level, advice = "良好", "可交易"
        elif total >= 60:
            level, advice = "一般", "谨慎交易"
        else:
            level, advice = "较差", "建议观望"

        return {
            "date": date,
            "scores": scores,
            "total_score": total,
            "level": level,
            "advice": advice,
            "summary": f"环境: {total}/100 ({level})",
            "details": detail,
        }

    # ═══════════════════════════════════════════════════════════
    #  Step 2: 信号扫描（阶梯突破）
    # ═══════════════════════════════════════════════════════════

    def _scan_signals(self, date: str, pool: list = None) -> list:
        """
        扫描某日的阶梯突破信号

        优先使用 TradeSignalGenerator，不可用时用内置简化版。
        """
        # 尝试正式模块
        try:
            from trade_signals import TradeSignalGenerator
            gen = TradeSignalGenerator(data_fetcher=self.fetcher)
            sigs = gen._scan_ladder_breakout(date, pool)
            for s in sigs:
                s["env_date"] = date
            return sigs
        except Exception:
            pass

        # 内置简化版
        return self._scan_signals_builtin(date, pool)

    def _scan_signals_builtin(self, date: str, pool: list = None) -> list:
        """
        内置阶梯突破扫描

        简化逻辑：
        1. 获取股票池
        2. 对每只股票：检查是否有整理后突破 + 量能配合
        3. 评分 ≥ 65 入选
        """
        signals = []
        stocks = pool or self._get_stock_pool()

        for symbol in stocks:
            try:
                df = self._get_stock_daily(symbol, date, days=90)
                if df is None or len(df) < 30:
                    continue

                close = df["close"].values
                high = df["high"].values
                low = df["low"].values

                current = close[-1]

                # 寻找整理区间 (最近 60 天内，至少 15 天)
                consol = self._find_consolidation(close, high, low)
                if not consol:
                    continue

                c_start, c_end, c_high, c_low, c_days = consol

                # 突破确认
                if current <= c_high:
                    continue

                # 量能
                vol_col = "vol" if "vol" in df.columns else "volume"
                if vol_col in df.columns:
                    vols = df[vol_col].values
                    avg_vol = _mean(vols[-20:]) if len(vols) >= 20 else _mean(vols)
                    vol_ratio = vols[-1] / avg_vol if avg_vol > 0 else 1.0
                else:
                    vol_ratio = 1.5  # 无量数据，假设通过

                if vol_ratio < 1.2:
                    continue

                # 评分
                amp = (c_high - c_low) / c_low * 100 if c_low > 0 else 99
                score = 50
                score += min(15, c_days)
                score += min(10, int(vol_ratio * 5))
                if amp < 8:
                    score += 10
                elif amp < 12:
                    score += 5

                breakout_pct = (current - c_high) / c_high * 100
                if 1 < breakout_pct < 5:
                    score += 5

                score = min(100, score)
                if score < self.min_score:
                    continue

                stop_loss = c_low * 0.97

                signals.append({
                    "type": "ladder_breakout",
                    "symbol": symbol,
                    "name": self._get_name(symbol),
                    "score": score,
                    "env_date": date,
                    "price": {
                        "close": round(current, 2),
                        "breakout": round(c_high, 2),
                        "stop_loss": round(stop_loss, 2),
                    },
                    "vol_ratio": round(vol_ratio, 2),
                    "consolidation_days": c_days,
                    "amplitude": round(amp, 1),
                })

            except Exception:
                continue

        return signals

    def _find_consolidation(self, close, high, low, min_days=15, max_amp=15):
        """寻找整理区间"""
        n = len(close)
        if n < min_days + 5:
            return None

        for end in range(n - 2, min_days, -1):
            for start in range(end - min_days, max(0, end - 60), -1):
                seg_high = max(high[start:end + 1])
                seg_low = min(low[start:end + 1])
                if seg_low <= 0:
                    continue
                amp = (seg_high - seg_low) / seg_low * 100
                if amp < max_amp:
                    days = end - start + 1
                    if days >= min_days:
                        return (start, end, seg_high, seg_low, days)
        return None

    # ═══════════════════════════════════════════════════════════
    #  Step 3: 跟踪信号盈亏
    # ═══════════════════════════════════════════════════════════

    def _track_trade(self, signal: dict, end_date_ext: str) -> Optional[dict]:
        """
        跟踪单个信号的实际盈亏

        规则：
        - 信号日收盘价买入
        - 持有期内：触发止损(-7%)立即退出，触发止盈(+15%)立即退出
        - 持有期满：按最后一天收盘价退出
        - 记录：持有天数、退出价、盈亏%、结果(win/loss/flat)
        """
        symbol = signal["symbol"]
        entry_date = signal["env_date"]
        entry_price = signal["price"]["close"]
        stop_price = signal["price"]["stop_loss"]

        # 获取后续 N+5 天数据
        future_df = self._get_future_prices(symbol, entry_date, self.hold_days + 5)
        if future_df is None or len(future_df) < 2:
            return None

        # 逐日检查
        exit_price = entry_price
        exit_day = 0
        outcome = "expired"

        for i in range(min(self.hold_days, len(future_df))):
            row = future_df.iloc[i]
            day_low = row.get("low", row["close"])
            day_high = row.get("high", row["close"])
            day_close = row["close"]

            # 止损检查（日内最低价触发）
            pct_low = (day_low / entry_price - 1) * 100
            if pct_low <= self.stop_loss_pct:
                exit_price = entry_price * (1 + self.stop_loss_pct / 100)
                exit_day = i + 1
                outcome = "stop_loss"
                break

            # 止盈检查（日内最高价触发）
            pct_high = (day_high / entry_price - 1) * 100
            if pct_high >= self.take_profit_pct:
                exit_price = entry_price * (1 + self.take_profit_pct / 100)
                exit_day = i + 1
                outcome = "take_profit"
                break

            exit_price = day_close
            exit_day = i + 1

        pnl_pct = (exit_price / entry_price - 1) * 100
        is_win = pnl_pct > 0

        # 查找该信号日的环境评分
        env_score = 65  # 默认
        for env in self.daily_env:
            if env["date"] == entry_date:
                env_score = env["total_score"]
                break

        return {
            "symbol": signal["symbol"],
            "name": signal.get("name", ""),
            "signal_score": signal["score"],
            "entry_date": entry_date,
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "hold_days": exit_day,
            "pnl_pct": round(pnl_pct, 2),
            "is_win": is_win,
            "outcome": outcome,
            "env_score": env_score,
            "env_month": entry_date[:6],   # YYYYMM
            "vol_ratio": signal.get("vol_ratio", 0),
            "consolidation_days": signal.get("consolidation_days", 0),
        }

    # ═══════════════════════════════════════════════════════════
    #  Step 4: 分析
    # ═══════════════════════════════════════════════════════════

    def _calc_v41_performance(self) -> dict:
        """v4.1: 全部信号都做"""
        if not self.trades:
            return {"trades": 0, "win_rate": 0, "total_pnl": 0, "avg_pnl": 0}

        wins = sum(1 for t in self.trades if t["is_win"])
        total_pnl = sum(t["pnl_pct"] for t in self.trades)
        count = len(self.trades)

        return {
            "trades": count,
            "wins": wins,
            "losses": count - wins,
            "win_rate": round(wins / count * 100, 1) if count else 0,
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(total_pnl / count, 2) if count else 0,
            "max_win": round(max((t["pnl_pct"] for t in self.trades), default=0), 2),
            "max_loss": round(min((t["pnl_pct"] for t in self.trades), default=0), 2),
            "avg_hold_days": round(sum(t["hold_days"] for t in self.trades) / count, 1) if count else 0,
        }

    def _calc_v60_performance(self) -> dict:
        """v6.0: 环境评分 < threshold 时停手"""
        filtered = [t for t in self.trades if t["env_score"] >= self.env_threshold]
        skipped = [t for t in self.trades if t["env_score"] < self.env_threshold]

        if not filtered:
            return {
                "trades": 0, "win_rate": 0, "total_pnl": 0,
                "skipped": len(skipped),
                "skipped_pnl": round(sum(t["pnl_pct"] for t in skipped), 2),
            }

        wins = sum(1 for t in filtered if t["is_win"])
        total_pnl = sum(t["pnl_pct"] for t in filtered)
        count = len(filtered)

        skipped_pnl = sum(t["pnl_pct"] for t in skipped)

        return {
            "trades": count,
            "wins": wins,
            "losses": count - wins,
            "win_rate": round(wins / count * 100, 1) if count else 0,
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(total_pnl / count, 2) if count else 0,
            "max_win": round(max((t["pnl_pct"] for t in filtered), default=0), 2),
            "max_loss": round(min((t["pnl_pct"] for t in filtered), default=0), 2),
            "avg_hold_days": round(sum(t["hold_days"] for t in filtered) / count, 1) if count else 0,
            "skipped": len(skipped),
            "skipped_pnl": round(skipped_pnl, 2),
            "saved_from_loss": round(-skipped_pnl, 2) if skipped_pnl < 0 else 0,
        }

    def _calc_correlation(self) -> dict:
        """
        计算环境评分与胜率的相关性

        Pearson 相关系数（简化计算，不依赖 scipy）
        """
        if len(self.trades) < 5:
            return {"pearson_r": 0, "note": "样本不足"}

        # 按日聚合：每天的环境分 vs 当天信号的平均盈亏
        day_data = defaultdict(lambda: {"env": 0, "pnls": []})
        for t in self.trades:
            d = t["entry_date"]
            day_data[d]["env"] = t["env_score"]
            day_data[d]["pnls"].append(t["pnl_pct"])

        xs = []  # env scores
        ys = []  # avg pnl
        for d, info in day_data.items():
            if info["pnls"]:
                xs.append(info["env"])
                ys.append(_mean(info["pnls"]))

        if len(xs) < 3:
            return {"pearson_r": 0, "note": "数据点不足"}

        r = _pearson(xs, ys)

        return {
            "pearson_r": round(r, 3),
            "data_points": len(xs),
            "interpretation": (
                "强正相关" if r > 0.5 else
                "中等正相关" if r > 0.3 else
                "弱正相关" if r > 0.1 else
                "无明显相关" if r > -0.1 else
                "弱负相关" if r > -0.3 else
                "中等负相关"
            ),
        }

    def _calc_bucket_stats(self) -> dict:
        """
        按环境评分分桶统计

        桶：<50, 50-59, 60-69, 70-79, ≥80
        每桶：交易数、胜率、平均盈亏
        """
        bucket_ranges = [
            ("< 50", 0, 50),
            ("50-59", 50, 60),
            ("60-69", 60, 70),
            ("70-79", 70, 80),
            ("≥ 80", 80, 101),
        ]

        buckets = {}
        for label, lo, hi in bucket_ranges:
            trades = [t for t in self.trades if lo <= t["env_score"] < hi]
            count = len(trades)
            if count == 0:
                buckets[label] = {"trades": 0, "win_rate": 0, "avg_pnl": 0}
            else:
                wins = sum(1 for t in trades if t["is_win"])
                avg_pnl = _mean([t["pnl_pct"] for t in trades])
                buckets[label] = {
                    "trades": count,
                    "wins": wins,
                    "win_rate": round(wins / count * 100, 1),
                    "avg_pnl": round(avg_pnl, 2),
                    "total_pnl": round(sum(t["pnl_pct"] for t in trades), 2),
                }

        return buckets

    def _calc_monthly_breakdown(self) -> dict:
        """按月分解表现"""
        months = defaultdict(lambda: {"trades": [], "envs": []})

        for t in self.trades:
            m = t["env_month"]
            months[m]["trades"].append(t)

        for env in self.daily_env:
            m = env["date"][:6]
            months[m]["envs"].append(env["total_score"])

        result = {}
        for m in sorted(months.keys()):
            trades = months[m]["trades"]
            envs = months[m]["envs"]
            count = len(trades)

            if count == 0:
                result[m] = {
                    "trades": 0, "win_rate": 0, "avg_pnl": 0,
                    "avg_env": round(_mean(envs), 1) if envs else 0,
                }
                continue

            wins = sum(1 for t in trades if t["is_win"])
            result[m] = {
                "trades": count,
                "wins": wins,
                "win_rate": round(wins / count * 100, 1),
                "total_pnl": round(sum(t["pnl_pct"] for t in trades), 2),
                "avg_pnl": round(_mean([t["pnl_pct"] for t in trades]), 2),
                "avg_env": round(_mean(envs), 1) if envs else 0,
            }

        return result

    # ═══════════════════════════════════════════════════════════
    #  Step 5: 报告
    # ═══════════════════════════════════════════════════════════

    def _generate_summary(self, trading_days, v41, v60, corr, buckets, monthly) -> str:
        lines = []
        lines.append("=" * 65)
        lines.append("  v6.0 回测报告")
        lines.append("=" * 65)

        # 总览
        lines.append(f"\n交易日: {len(trading_days)} 天")
        lines.append(f"信号总数: {len(self.signals)} | 有效交易: {len(self.trades)}")

        # ── 环境评分分布 ──
        lines.append("\n--- 环境评分分布 ---")
        env_scores = [e["total_score"] for e in self.daily_env]
        if env_scores:
            lines.append(f"  均值: {_mean(env_scores):.1f}")
            lines.append(f"  范围: {min(env_scores)} ~ {max(env_scores)}")
            bad_days = sum(1 for s in env_scores if s < 60)
            lines.append(f"  较差(<60): {bad_days} 天 ({bad_days/len(env_scores)*100:.0f}%)")

        # ── v4.1 vs v6.0 ──
        lines.append("\n--- v4.1 (全做) vs v6.0 (环境过滤) ---")
        lines.append(f"  {'指标':<16s} {'v4.1':>10s} {'v6.0':>10s} {'差异':>10s}")
        lines.append(f"  {'-'*46}")

        for key, label in [
            ("trades", "交易笔数"),
            ("win_rate", "胜率(%)"),
            ("total_pnl", "总收益(%)"),
            ("avg_pnl", "平均收益(%)"),
        ]:
            val41 = v41.get(key, 0)
            val60 = v60.get(key, 0)
            diff = val60 - val41
            lines.append(f"  {label:<16s} {val41:>10} {val60:>10} {diff:>+10.1f}")

        if v60.get("skipped"):
            lines.append(f"\n  跳过交易: {v60['skipped']} 笔")
            lines.append(f"  跳过的总盈亏: {v60.get('skipped_pnl', 0):.2f}%")
            saved = v60.get("saved_from_loss", 0)
            if saved > 0:
                lines.append(f"  ✅ 避免亏损: {saved:.2f}%")

        # ── 分桶统计 ──
        lines.append("\n--- 按环境评分分桶 ---")
        lines.append(f"  {'评分区间':<10s} {'交易':>6s} {'胜率':>8s} {'平均盈亏':>10s} {'总盈亏':>10s}")
        lines.append(f"  {'-'*44}")
        for label, stats in buckets.items():
            wr = f"{stats['win_rate']}%" if stats["trades"] else "-"
            ap = f"{stats['avg_pnl']}%" if stats["trades"] else "-"
            tp = f"{stats.get('total_pnl', 0)}%" if stats["trades"] else "-"
            lines.append(f"  {label:<10s} {stats['trades']:>6d} {wr:>8s} {ap:>10s} {tp:>10s}")

        # ── 月度分解 ──
        lines.append("\n--- 月度分解 ---")
        lines.append(f"  {'月份':<8s} {'环境均分':>8s} {'交易':>6s} {'胜率':>8s} {'总盈亏':>10s}")
        lines.append(f"  {'-'*40}")
        for m, stats in monthly.items():
            wr = f"{stats['win_rate']}%" if stats["trades"] else "-"
            tp = f"{stats.get('total_pnl', 0)}%" if stats["trades"] else "-"
            lines.append(f"  {m:<8s} {stats['avg_env']:>8.1f} {stats['trades']:>6d} {wr:>8s} {tp:>10s}")

        # ── 相关性 ──
        lines.append(f"\n--- 相关性分析 ---")
        lines.append(f"  Pearson r: {corr['pearson_r']}")
        lines.append(f"  解读: {corr.get('interpretation', '?')}")
        lines.append(f"  数据点: {corr.get('data_points', 0)}")

        # ── 结论 ──
        lines.append("\n--- 结论 ---")
        pnl_diff = v60.get("total_pnl", 0) - v41.get("total_pnl", 0)
        wr_diff = v60.get("win_rate", 0) - v41.get("win_rate", 0)
        if pnl_diff > 0:
            lines.append(f"  ✅ v6.0 优于 v4.1: 收益 +{pnl_diff:.2f}%, 胜率 +{wr_diff:.1f}pp")
            lines.append(f"  核心价值: 通过环境过滤避免了低质量交易")
        elif pnl_diff == 0:
            lines.append(f"  ≈ v6.0 与 v4.1 表现相当")
        else:
            lines.append(f"  ⚠️ v6.0 劣于 v4.1: 收益 {pnl_diff:.2f}%")
            lines.append(f"  可能原因: 环境阈值({self.env_threshold})需要调优")

        if corr["pearson_r"] > 0.3:
            lines.append(f"  ✅ 环境评分与盈亏呈{corr['interpretation']}（r={corr['pearson_r']}）")
            lines.append(f"     → 市场环境评估有预测价值")
        else:
            lines.append(f"  ⚠️ 环境评分与盈亏相关性较弱（r={corr['pearson_r']}）")
            lines.append(f"     → 需要优化评分维度或权重")

        lines.append("\n" + "=" * 65)
        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════
    #  导出
    # ═══════════════════════════════════════════════════════════

    def export_csv(self, filepath: str = "backtest_trades.csv"):
        """导出交易明细到 CSV"""
        if not self.trades:
            print("无交易数据")
            return

        fields = [
            "entry_date", "symbol", "name", "signal_score", "env_score",
            "entry_price", "exit_price", "hold_days", "pnl_pct",
            "is_win", "outcome", "vol_ratio", "consolidation_days",
        ]

        with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for trade in self.trades:
                writer.writerow(trade)

        print(f"已导出 {len(self.trades)} 笔交易到 {filepath}")

    def export_env_csv(self, filepath: str = "backtest_env.csv"):
        """导出环境评分到 CSV"""
        if not self.daily_env:
            print("无环境数据")
            return

        with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "total_score", "trend", "sentiment",
                            "volume", "sector", "level", "advice"])
            for env in self.daily_env:
                s = env["scores"]
                writer.writerow([
                    env["date"], env["total_score"],
                    s["trend"], s["sentiment"], s["volume"], s["sector"],
                    env["level"], env["advice"],
                ])

        print(f"已导出 {len(self.daily_env)} 天环境数据到 {filepath}")

    # ═══════════════════════════════════════════════════════════
    #  敏感性分析
    # ═══════════════════════════════════════════════════════════

    def sensitivity_analysis(self, thresholds: list = None) -> dict:
        """
        环境阈值敏感性分析

        测试不同阈值下的 v6.0 表现，找到最优阈值。

        Parameters
        ----------
        thresholds : 要测试的阈值列表（默认 [50,55,60,65,70,75]）

        Returns
        -------
        {50: {...}, 55: {...}, ...}  每个阈值对应的表现
        """
        if not self.trades:
            print("请先运行 run() 生成交易数据")
            return {}

        thresholds = thresholds or [50, 55, 60, 65, 70, 75]
        results = {}

        print("\n--- 阈值敏感性分析 ---")
        print(f"{'阈值':>6s} {'交易':>6s} {'跳过':>6s} {'胜率':>8s} {'总盈亏':>10s} {'避免亏损':>10s}")
        print("-" * 50)

        for threshold in thresholds:
            filtered = [t for t in self.trades if t["env_score"] >= threshold]
            skipped = [t for t in self.trades if t["env_score"] < threshold]

            count = len(filtered)
            skip_count = len(skipped)
            wins = sum(1 for t in filtered if t["is_win"])
            total_pnl = sum(t["pnl_pct"] for t in filtered)
            skipped_pnl = sum(t["pnl_pct"] for t in skipped)
            saved = max(0, -skipped_pnl)

            wr = round(wins / count * 100, 1) if count else 0

            results[threshold] = {
                "trades": count,
                "skipped": skip_count,
                "win_rate": wr,
                "total_pnl": round(total_pnl, 2),
                "saved": round(saved, 2),
            }

            print(f"{threshold:>6d} {count:>6d} {skip_count:>6d} "
                  f"{wr:>7.1f}% {total_pnl:>+9.2f}% {saved:>+9.2f}%")

        # 找最优
        best = max(results.items(), key=lambda x: x[1]["total_pnl"])
        print(f"\n最优阈值: {best[0]} (总收益 {best[1]['total_pnl']}%)")

        return results

    # ═══════════════════════════════════════════════════════════
    #  数据获取适配
    # ═══════════════════════════════════════════════════════════

    def _get_trading_days(self, start, end) -> list:
        """获取交易日列表"""
        if self.fetcher and hasattr(self.fetcher, 'get_trading_days'):
            try:
                return self.fetcher.get_trading_days(start, end)
            except Exception:
                pass
        # 生成简化版（排除周末）
        return _generate_weekdays(start, end)

    def _get_index_daily(self, date, days=60):
        if not self.fetcher:
            return None
        for method in ['get_index_daily', 'get_daily']:
            if hasattr(self.fetcher, method):
                try:
                    return getattr(self.fetcher, method)(self.index_code, days=days, end_date=date)
                except Exception:
                    continue
        return None

    def _get_stock_daily(self, symbol, date, days=90):
        if not self.fetcher:
            return None
        for method in ['get_daily', 'get_stock_daily', 'get_k_data']:
            if hasattr(self.fetcher, method):
                try:
                    return getattr(self.fetcher, method)(symbol, days=days, end_date=date)
                except Exception:
                    continue
        return None

    def _get_future_prices(self, symbol, start_date, days):
        """获取某日之后 N 天的价格"""
        if not self.fetcher:
            return None
        end = self._date_add(start_date, days + 5)
        for method in ['get_daily', 'get_stock_daily', 'get_k_data']:
            if hasattr(self.fetcher, method):
                try:
                    df = getattr(self.fetcher, method)(symbol, start_date=start_date, end_date=end)
                    if df is not None and len(df) > 1:
                        # 排除入场日本身，取后续数据
                        if "trade_date" in df.columns:
                            df = df[df["trade_date"] > start_date]
                        elif "date" in df.columns:
                            df = df[df["date"] > start_date]
                        else:
                            df = df.iloc[1:]  # 跳过第一行（入场日）
                        return df.head(days)
                except Exception:
                    continue
        return None

    def _get_stock_pool(self):
        if self.fetcher and hasattr(self.fetcher, 'get_stock_pool'):
            try:
                return self.fetcher.get_stock_pool()
            except Exception:
                pass
        return []

    def _get_name(self, symbol):
        if self.fetcher and hasattr(self.fetcher, 'get_stock_name'):
            try:
                return self.fetcher.get_stock_name(symbol)
            except Exception:
                pass
        return symbol

    def _date_add(self, date_str, days):
        dt = datetime.strptime(date_str, "%Y%m%d") + timedelta(days=days)
        return dt.strftime("%Y%m%d")


# ═══════════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════════

def _mean(arr) -> float:
    arr = list(arr)
    return sum(arr) / len(arr) if arr else 0

def _clamp(val, lo=0, hi=100) -> int:
    return max(lo, min(hi, int(val)))

def _pearson(xs, ys) -> float:
    """Pearson 相关系数（不依赖 scipy）"""
    n = len(xs)
    if n < 3:
        return 0
    mx = _mean(xs)
    my = _mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx * dy == 0:
        return 0
    return num / (dx * dy)

def _generate_weekdays(start_str, end_str) -> list:
    """生成日期范围内的工作日列表"""
    start = datetime.strptime(start_str, "%Y%m%d")
    end = datetime.strptime(end_str, "%Y%m%d")
    days = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            days.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    return days


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="v6.0 回测验证")
    parser.add_argument("--start", default="20251101", help="开始日期 YYYYMMDD")
    parser.add_argument("--end", default="20251231", help="结束日期 YYYYMMDD")
    parser.add_argument("--env-only", action="store_true", help="只看环境评分")
    parser.add_argument("--export", type=str, help="导出交易明细到 CSV")
    parser.add_argument("--sensitivity", action="store_true", help="阈值敏感性分析")
    parser.add_argument("--threshold", type=int, default=60, help="环境阈值（默认60）")
    parser.add_argument("--hold", type=int, default=10, help="持有天数（默认10）")
    parser.add_argument("--stop", type=float, default=-7.0, help="止损线%（默认-7）")
    args = parser.parse_args()

    # 初始化
    fetcher = None
    try:
        from data_fetcher import DataFetcher
        fetcher = DataFetcher()
        print("✅ DataFetcher 已加载")
    except Exception as e:
        print(f"⚠️ DataFetcher 未加载: {e}")
        print("将使用简化模式（需要手动提供股票池）")

    bt = BacktestV6(data_fetcher=fetcher)
    bt.env_threshold = args.threshold
    bt.hold_days = args.hold
    bt.stop_loss_pct = args.stop

    if args.env_only:
        print(f"\n环境评分: {args.start} ~ {args.end}")
        bt.run_env_only(args.start, args.end)
        return

    # 完整回测
    report = bt.run(args.start, args.end)

    # 敏感性分析
    if args.sensitivity:
        bt.sensitivity_analysis()

    # 导出
    if args.export:
        bt.export_csv(args.export)
        bt.export_env_csv(args.export.replace(".csv", "_env.csv"))

    # 保存报告
    report_path = Path("data/backtest")
    report_path.mkdir(parents=True, exist_ok=True)
    fname = f"report_{args.start}_{args.end}.json"
    with open(report_path / fname, "w", encoding="utf-8") as f:
        # trades 和 daily_env 太大，只存摘要
        save_data = {k: v for k, v in report.items()
                     if k not in ("daily_env", "signals")}
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {report_path / fname}")


if __name__ == "__main__":
    main()
