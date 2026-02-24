"""
策略复盘自动化: 追踪Agent系统的推荐绩效

功能:
  1. 记录每次选股推荐 (候选股 + 入场价 + 止损价)
  2. 追踪推荐后的实际表现 (5日/10日/20日收益)
  3. 分析各维度胜率 (CANSLIM等级/板块/信号强度)
  4. 输出优化建议 → 反馈到 Skills 参数

这就是文章中说的"系统在帮我优化系统"的复利效应
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_fetcher import get_fetcher


RECOMMENDATION_LOG = "./knowledge/recommendations.json"
REVIEW_LOG = "./knowledge/reviews.json"


class StrategyReviewer:
    """策略复盘器"""

    def __init__(self):
        self.fetcher = get_fetcher()
        self.rec_path = Path(RECOMMENDATION_LOG)
        self.review_path = Path(REVIEW_LOG)
        self.rec_path.parent.mkdir(parents=True, exist_ok=True)

    # ========================================================
    # 记录推荐
    # ========================================================
    def log_recommendation(self, candidates: List[dict], date: str = ""):
        """
        记录一批选股推荐

        candidates: 来自 PipelineReport 或 CanslimReport 的候选列表
        每个candidate需要包含: ts_code, name, entry_price, stop_price, grade, score
        """
        if not date:
            date = self.fetcher.get_latest_trade_date()

        records = self._load_records(self.rec_path)

        batch = {
            "date": date,
            "timestamp": datetime.now().isoformat(),
            "stocks": [],
            "reviewed": False,
        }

        for c in candidates:
            batch["stocks"].append({
                "ts_code": c.get("ts_code", ""),
                "name": c.get("name", ""),
                "entry_price": c.get("suggested_entry", c.get("entry_price", 0)),
                "stop_price": c.get("suggested_stop", c.get("stop_price", 0)),
                "grade": c.get("canslim_grade", c.get("grade", "")),
                "score": c.get("final_score", c.get("canslim_score", 0)),
                "sector": c.get("sector", ""),
                "signal_strength": c.get("buy_signal_strength", ""),
                "flags": c.get("flags", []),
                # 以下字段在复盘时填充
                "ret_5d": None,
                "ret_10d": None,
                "ret_20d": None,
                "hit_stop": None,
                "max_drawdown": None,
                "max_gain": None,
            })

        records.append(batch)
        self._save_records(self.rec_path, records)
        print(f"[Reviewer] 记录{len(batch['stocks'])}只推荐 ({date})")

    # ========================================================
    # 执行复盘
    # ========================================================
    def review_past_recommendations(self, lookback_days: int = 30) -> Dict:
        """
        复盘过去的推荐记录，计算实际收益

        Returns: 复盘统计摘要
        """
        records = self._load_records(self.rec_path)
        if not records:
            print("[Reviewer] 无历史推荐记录")
            return {}

        cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y%m%d")
        updated_count = 0

        for batch in records:
            if batch.get("reviewed"):
                continue
            if batch["date"] < cutoff:
                continue

            # 检查是否已过足够天数来评估
            rec_date = batch["date"]
            trade_dates = self.fetcher.get_trade_dates(start=rec_date)
            if len(trade_dates) < 6:  # 至少要5个交易日后才复盘
                continue

            for stock in batch["stocks"]:
                if stock["ret_5d"] is not None:
                    continue  # 已经复盘过

                self._fill_performance(stock, rec_date, trade_dates)
                updated_count += 1
                time.sleep(0.1)

            # 如果所有股票都复盘完了
            all_reviewed = all(s["ret_5d"] is not None for s in batch["stocks"])
            if all_reviewed:
                batch["reviewed"] = True

        if updated_count > 0:
            self._save_records(self.rec_path, records)
            print(f"[Reviewer] 更新了 {updated_count} 条推荐的绩效数据")

        # 生成统计
        stats = self._compute_stats(records)
        return stats

    def _fill_performance(self, stock: dict, rec_date: str, trade_dates: List[str]):
        """填充单只股票的实际表现"""
        try:
            ts_code = stock["ts_code"]
            entry = stock["entry_price"]
            stop = stock["stop_price"]

            if entry <= 0:
                return

            df = self.fetcher.get_stock_daily(ts_code, days=30)
            if df.empty:
                return

            df = df.sort_values("trade_date").reset_index(drop=True)
            # 只取推荐日之后的数据
            df = df[df["trade_date"] > rec_date].reset_index(drop=True)

            if df.empty:
                return

            # 5日收益
            if len(df) >= 5:
                stock["ret_5d"] = round((df.iloc[4]["close"] / entry - 1) * 100, 2)
            elif len(df) > 0:
                stock["ret_5d"] = round((df.iloc[-1]["close"] / entry - 1) * 100, 2)

            # 10日收益
            if len(df) >= 10:
                stock["ret_10d"] = round((df.iloc[9]["close"] / entry - 1) * 100, 2)

            # 20日收益
            if len(df) >= 20:
                stock["ret_20d"] = round((df.iloc[19]["close"] / entry - 1) * 100, 2)

            # 最大回撤和最大涨幅
            if not df.empty:
                stock["max_drawdown"] = round((df["low"].min() / entry - 1) * 100, 2)
                stock["max_gain"] = round((df["high"].max() / entry - 1) * 100, 2)

            # 是否触及止损
            if stop > 0:
                stock["hit_stop"] = bool(df["low"].min() <= stop)

        except Exception as e:
            print(f"[Reviewer] {stock.get('ts_code', '?')} 复盘失败: {e}")

    # ========================================================
    # 统计分析
    # ========================================================
    def _compute_stats(self, records: list) -> dict:
        """计算推荐绩效统计"""
        all_stocks = []
        for batch in records:
            for stock in batch["stocks"]:
                if stock["ret_5d"] is not None:
                    stock["rec_date"] = batch["date"]
                    all_stocks.append(stock)

        if not all_stocks:
            return {"total": 0, "message": "暂无可统计的复盘数据"}

        df = pd.DataFrame(all_stocks)

        stats = {
            "total_recommendations": len(df),
            "date_range": f"{df['rec_date'].min()} ~ {df['rec_date'].max()}",
        }

        # 5日胜率
        if "ret_5d" in df.columns:
            win_5d = (df["ret_5d"] > 0).sum()
            stats["win_rate_5d"] = round(win_5d / len(df) * 100, 1)
            stats["avg_ret_5d"] = round(df["ret_5d"].mean(), 2)
            stats["median_ret_5d"] = round(df["ret_5d"].median(), 2)

        # 10日胜率
        valid_10d = df[df["ret_10d"].notna()]
        if len(valid_10d) > 0:
            win_10d = (valid_10d["ret_10d"] > 0).sum()
            stats["win_rate_10d"] = round(win_10d / len(valid_10d) * 100, 1)
            stats["avg_ret_10d"] = round(valid_10d["ret_10d"].mean(), 2)

        # 20日胜率
        valid_20d = df[df["ret_20d"].notna()]
        if len(valid_20d) > 0:
            win_20d = (valid_20d["ret_20d"] > 0).sum()
            stats["win_rate_20d"] = round(win_20d / len(valid_20d) * 100, 1)
            stats["avg_ret_20d"] = round(valid_20d["ret_20d"].mean(), 2)

        # 止损触发率
        if "hit_stop" in df.columns:
            hit_count = df["hit_stop"].sum()
            stats["stop_hit_rate"] = round(hit_count / len(df) * 100, 1)

        # 按等级统计
        grade_stats = {}
        for grade in ["A", "B", "C"]:
            g_df = df[df["grade"] == grade]
            if len(g_df) >= 3:  # 至少3个样本
                grade_stats[grade] = {
                    "count": len(g_df),
                    "win_rate_5d": round((g_df["ret_5d"] > 0).sum() / len(g_df) * 100, 1),
                    "avg_ret_5d": round(g_df["ret_5d"].mean(), 2),
                }
        stats["by_grade"] = grade_stats

        # 按信号强度统计
        strength_stats = {}
        for strength in ["强", "中", "弱"]:
            s_df = df[df["signal_strength"] == strength]
            if len(s_df) >= 3:
                strength_stats[strength] = {
                    "count": len(s_df),
                    "win_rate_5d": round((s_df["ret_5d"] > 0).sum() / len(s_df) * 100, 1),
                    "avg_ret_5d": round(s_df["ret_5d"].mean(), 2),
                }
        stats["by_signal_strength"] = strength_stats

        # 按板块统计
        sector_stats = {}
        for sector in df["sector"].unique():
            if not sector:
                continue
            sec_df = df[df["sector"] == sector]
            if len(sec_df) >= 3:
                sector_stats[sector] = {
                    "count": len(sec_df),
                    "win_rate_5d": round((sec_df["ret_5d"] > 0).sum() / len(sec_df) * 100, 1),
                    "avg_ret_5d": round(sec_df["ret_5d"].mean(), 2),
                }
        stats["by_sector"] = sector_stats

        return stats

    # ========================================================
    # 优化建议
    # ========================================================
    def generate_optimization_suggestions(self, stats: dict) -> List[str]:
        """根据复盘统计生成参数优化建议"""
        suggestions = []

        if not stats or stats.get("total_recommendations", 0) < 10:
            return ["数据量不足(需至少10条记录)，暂无优化建议"]

        # 1. 整体胜率判断
        wr = stats.get("win_rate_5d", 50)
        if wr < 40:
            suggestions.append(
                f"⚠️ 5日胜率仅{wr}%，偏低。建议:\n"
                f"  - 提高CANSLIM筛选门槛(如 c_min_growth 从20%调到25%)\n"
                f"  - 增加缩量整理条件权重\n"
                f"  - 在非强势情绪下减少选股频率"
            )
        elif wr > 65:
            suggestions.append(
                f"✅ 5日胜率{wr}%，表现良好。可适度扩大候选池或放宽筛选条件"
            )

        # 2. 按等级分析
        by_grade = stats.get("by_grade", {})
        for grade, g_stats in by_grade.items():
            if g_stats["win_rate_5d"] < 35:
                suggestions.append(
                    f"⚠️ {grade}级股票胜率仅{g_stats['win_rate_5d']}%，"
                    f"建议重新审视{grade}级的评分标准"
                )

        # 3. 信号强度分析
        by_strength = stats.get("by_signal_strength", {})
        if "强" in by_strength and "弱" in by_strength:
            strong_wr = by_strength["强"]["win_rate_5d"]
            weak_wr = by_strength["弱"]["win_rate_5d"]
            if strong_wr - weak_wr < 10:
                suggestions.append(
                    f"📊 强信号({strong_wr}%)和弱信号({weak_wr}%)胜率差距不大，"
                    f"信号区分度不够，建议调整信号分类阈值"
                )

        # 4. 止损分析
        stop_rate = stats.get("stop_hit_rate", 0)
        if stop_rate > 30:
            suggestions.append(
                f"⚠️ 止损触发率{stop_rate}%偏高，建议:\n"
                f"  - 放宽止损幅度(如从5%调到7%)\n"
                f"  - 或收紧入场条件(减少假突破)"
            )
        elif stop_rate < 10:
            suggestions.append(
                f"📊 止损触发率仅{stop_rate}%，止损线可能设置过松，"
                f"建议适当收紧以控制单笔亏损"
            )

        # 5. 板块分析
        by_sector = stats.get("by_sector", {})
        best_sector = max(by_sector.items(), key=lambda x: x[1]["avg_ret_5d"]) if by_sector else None
        worst_sector = min(by_sector.items(), key=lambda x: x[1]["avg_ret_5d"]) if by_sector else None
        if best_sector and worst_sector:
            suggestions.append(
                f"📊 板块表现差异:\n"
                f"  最佳: {best_sector[0]} (5日均收益{best_sector[1]['avg_ret_5d']:+.2f}%)\n"
                f"  最差: {worst_sector[0]} (5日均收益{worst_sector[1]['avg_ret_5d']:+.2f}%)\n"
                f"  可考虑在板块轮动中增加历史胜率权重"
            )

        return suggestions

    # ========================================================
    # 工具方法
    # ========================================================
    def _load_records(self, path: Path) -> list:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _save_records(self, path: Path, data: list):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def print_report(self, stats: dict):
        """打印复盘报告"""
        if not stats or "total_recommendations" not in stats:
            print("暂无复盘数据")
            return

        print(f"\n{'='*60}")
        print(f"  📈 策略复盘报告")
        print(f"  推荐总数: {stats['total_recommendations']} | "
              f"时间范围: {stats.get('date_range', 'N/A')}")
        print(f"{'='*60}")

        # 总体绩效
        print(f"\n  --- 整体绩效 ---")
        for period in ["5d", "10d", "20d"]:
            wr = stats.get(f"win_rate_{period}")
            avg = stats.get(f"avg_ret_{period}")
            if wr is not None:
                print(f"  {period}: 胜率{wr}% | 均收益{avg:+.2f}%")

        if "stop_hit_rate" in stats:
            print(f"  止损触发率: {stats['stop_hit_rate']}%")

        # 按等级
        if stats.get("by_grade"):
            print(f"\n  --- 按CANSLIM等级 ---")
            for grade, gs in stats["by_grade"].items():
                print(f"  {grade}级: {gs['count']}只 | 胜率{gs['win_rate_5d']}% | "
                      f"均收益{gs['avg_ret_5d']:+.2f}%")

        # 按信号
        if stats.get("by_signal_strength"):
            print(f"\n  --- 按信号强度 ---")
            for strength, ss in stats["by_signal_strength"].items():
                print(f"  {strength}: {ss['count']}只 | 胜率{ss['win_rate_5d']}% | "
                      f"均收益{ss['avg_ret_5d']:+.2f}%")

        # 按板块
        if stats.get("by_sector"):
            print(f"\n  --- 按板块 ---")
            sorted_sectors = sorted(stats["by_sector"].items(),
                                    key=lambda x: x[1]["avg_ret_5d"], reverse=True)
            for sector, ss in sorted_sectors:
                print(f"  {sector}: {ss['count']}只 | 胜率{ss['win_rate_5d']}% | "
                      f"均收益{ss['avg_ret_5d']:+.2f}%")

        # 优化建议
        suggestions = self.generate_optimization_suggestions(stats)
        if suggestions:
            print(f"\n  --- 优化建议 ---")
            for s in suggestions:
                print(f"  {s}")

        print(f"\n{'='*60}")


if __name__ == "__main__":
    reviewer = StrategyReviewer()

    # 模拟记录一批推荐
    test_recs = [
        {"ts_code": "600519.SH", "name": "贵州茅台", "suggested_entry": 1500,
         "suggested_stop": 1425, "canslim_grade": "A", "final_score": 85,
         "sector": "食品饮料", "buy_signal_strength": "强", "flags": ["ROE连续5年>15%"]},
        {"ts_code": "002594.SZ", "name": "比亚迪", "suggested_entry": 250,
         "suggested_stop": 237, "canslim_grade": "B", "final_score": 72,
         "sector": "汽车", "buy_signal_strength": "中", "flags": ["缩量整理"]},
    ]

    print("记录测试推荐...")
    reviewer.log_recommendation(test_recs, date="20260210")

    print("\n运行复盘...")
    stats = reviewer.review_past_recommendations()
    reviewer.print_report(stats)
