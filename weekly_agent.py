#!/usr/bin/env python3
"""
周度选股 Agent - 完整调度入口

每周运行一次，执行:
  1. 情绪判断 → 决定是否选股
  2. 板块轮动 → 筛出强势板块
  3. CANSLIM + 缩量整理 → 联合选股
  4. 风控检查 → 审核现有持仓
  5. 历史模式匹配 → 风险提示
  6. 复盘上周推荐 → 策略优化
  7. 生成完整周报

运行:
  python weekly_agent.py              # 完整周报
  python weekly_agent.py --screen     # 只运行选股
  python weekly_agent.py --review     # 只运行复盘

定时任务:
  0 18 * * 5 cd /path/to/astock_agent && python weekly_agent.py
  (每周五18:00收盘后运行)
"""

import sys
import os
import json
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_fetcher import get_fetcher
from skills.sentiment import SentimentSkill
from skills.sector_rotation import SectorRotationSkill
from skills.macro_monitor import MacroSkill
from skills.risk_control import RiskControlSkill
from skills.canslim_screener import CanslimScreener
from skills.stock_pipeline import StockPipeline
from skills.strategy_reviewer import StrategyReviewer
from knowledge_base import KnowledgeBase, init_default_knowledge
from report_generator import ReportGenerator


POSITIONS_FILE = "./knowledge/positions.json"
REPORT_DIR = "./reports"


def load_positions():
    path = Path(POSITIONS_FILE)
    if path.exists():
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


class WeeklyAgent:
    """周度选股Agent"""

    def __init__(self):
        self.fetcher = get_fetcher()
        self.kb = init_default_knowledge()

    def run_full(self) -> str:
        """完整周度报告"""
        print(f"\n{'='*70}")
        print(f"  🤖 A股周度选股 Agent")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")

        date = self.fetcher.get_latest_trade_date()
        lines = []
        lines.append(f"# 📊 A股周度选股报告 | {date[:4]}-{date[4:6]}-{date[6:]}")
        lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("")

        # ============ Part 1: 市场环境 ============
        print("[1/6] 市场环境评估...")
        sentiment = SentimentSkill().analyze()
        macro = MacroSkill().analyze()

        lines.append("## 一、市场环境")
        lines.append("")
        lines.append(f"| 维度 | 评级 | 得分 | 建议 |")
        lines.append(f"|------|------|------|------|")
        lines.append(f"| 情绪 | {sentiment.overall_level} | {sentiment.overall_score:+.1f} | "
                      f"{sentiment.suggested_position} |")
        lines.append(f"| 流动性 | {macro.liquidity_level} | {macro.overall_score:+.1f} | "
                      f"{macro.market_impact} |")
        lines.append("")

        # ============ Part 2: 板块轮动 ============
        print("[2/6] 板块轮动排名...")
        sector = SectorRotationSkill().analyze()

        lines.append("## 二、板块轮动排名")
        lines.append("")
        if sector.sectors:
            sorted_sectors = sorted(sector.sectors,
                                     key=lambda x: x.composite_score, reverse=True)
            lines.append("| 排名 | 板块 | 5日 | 20日 | 60日 | 量比 | 得分 | 信号 |")
            lines.append("|------|------|-----|------|------|------|------|------|")
            for i, s in enumerate(sorted_sectors[:15]):
                lines.append(
                    f"| {i+1} | {s.name} | {s.ret_5d:+.1f}% | {s.ret_20d:+.1f}% | "
                    f"{s.ret_60d:+.1f}% | {s.vol_ratio:.2f} | {s.composite_score:.0f} | {s.signal} |"
                )
            lines.append("")
            for sig in sector.rotation_signals:
                lines.append(f"- {sig}")
            lines.append("")

        # ============ Part 3: 联合选股 ============
        print("[3/6] 联合选股流水线...")
        pipeline_result = None
        if sentiment.overall_level not in ["极度恐慌", "恐慌"]:
            pipeline = StockPipeline()
            pipeline_result = pipeline.run(top_n_result=15)

            lines.append("## 三、选股结果")
            lines.append("")
            lines.append(f"**{pipeline_result.summary}**")
            lines.append("")

            if pipeline_result.candidates:
                lines.append("| 信号 | 代码 | 名称 | 板块 | CANSLIM | 总分 | 入场 | 止损 | 标记 |")
                lines.append("|------|------|------|------|---------|------|------|------|------|")
                for c in pipeline_result.candidates:
                    strength_icon = {"强": "🔥", "中": "⭐", "弱": "○"}.get(c.buy_signal_strength, "")
                    flags = " ".join(c.flags[:2])
                    lines.append(
                        f"| {strength_icon}{c.buy_signal_strength} | {c.ts_code} | {c.name} | "
                        f"{c.sector} | {c.canslim_grade}({c.canslim_score:.0f}) | "
                        f"{c.final_score:.0f} | {c.suggested_entry:.2f} | "
                        f"{c.suggested_stop:.2f} | {flags} |"
                    )
                lines.append("")

                # 记录推荐（用于后续复盘）
                reviewer = StrategyReviewer()
                rec_data = [
                    {
                        "ts_code": c.ts_code, "name": c.name,
                        "suggested_entry": c.suggested_entry,
                        "suggested_stop": c.suggested_stop,
                        "canslim_grade": c.canslim_grade,
                        "final_score": c.final_score,
                        "sector": c.sector,
                        "buy_signal_strength": c.buy_signal_strength,
                        "flags": c.flags,
                    }
                    for c in pipeline_result.candidates
                ]
                reviewer.log_recommendation(rec_data, date=date)
        else:
            lines.append("## 三、选股结果")
            lines.append("")
            lines.append(f"⚠️ 当前市场情绪={sentiment.overall_level}，本周不推荐新建仓")
            lines.append("")

        # ============ Part 4: 持仓风控 ============
        print("[4/6] 持仓风控检查...")
        positions = load_positions()
        lines.append("## 四、持仓风控")
        lines.append("")
        if positions:
            risk_skill = RiskControlSkill()
            risk_result = risk_skill.analyze(positions, sentiment.overall_level, date)
            lines.append(f"**{risk_result.summary}**")
            lines.append("")
            for action in risk_result.actions:
                lines.append(f"- {action}")
            for alert in risk_result.portfolio_alerts:
                lines.append(f"- {alert}")
        else:
            lines.append("未配置持仓信息")
        lines.append("")

        # ============ Part 5: 历史模式匹配 ============
        print("[5/6] 历史模式匹配...")
        current_state = {
            "sentiment": sentiment.overall_level,
            "liquidity": macro.liquidity_level,
        }
        for sig in macro.signals:
            if "北向" in sig.name:
                current_state["north_flow_trend"] = sig.trend
            if "两融" in sig.name:
                current_state["margin_trend"] = sig.trend
            if "成交额" in sig.name:
                current_state["turnover_trend"] = sig.trend

        matches = self.kb.match_pattern(current_state)
        if matches:
            lines.append("## 五、历史模式匹配")
            lines.append("")
            for m in matches[:3]:
                lines.append(f"### 📌 {m['name']} (匹配度 {m['match_score']:.0%})")
                lines.append(f"- 典型结果: {m['typical_outcome']}")
                lines.append(f"- 置信度: {m.get('confidence', '中')}")
                lines.append("")

        # ============ Part 6: 策略复盘 ============
        print("[6/6] 策略复盘...")
        reviewer = StrategyReviewer()
        stats = reviewer.review_past_recommendations()

        if stats and stats.get("total_recommendations", 0) > 0:
            lines.append("## 六、策略复盘")
            lines.append("")
            lines.append(f"**复盘样本: {stats['total_recommendations']}只 | "
                          f"时间: {stats.get('date_range', 'N/A')}**")
            lines.append("")

            wr = stats.get("win_rate_5d")
            avg = stats.get("avg_ret_5d")
            if wr is not None:
                lines.append(f"- 5日胜率: {wr}% | 均收益: {avg:+.2f}%")
            wr10 = stats.get("win_rate_10d")
            avg10 = stats.get("avg_ret_10d")
            if wr10 is not None:
                lines.append(f"- 10日胜率: {wr10}% | 均收益: {avg10:+.2f}%")
            if "stop_hit_rate" in stats:
                lines.append(f"- 止损触发率: {stats['stop_hit_rate']}%")
            lines.append("")

            # 优化建议
            suggestions = reviewer.generate_optimization_suggestions(stats)
            if suggestions:
                lines.append("### 优化建议")
                for s in suggestions:
                    lines.append(f"\n{s}")
                lines.append("")

        # ============ 保存 ============
        lines.append("---")
        lines.append("*以上分析由Agent系统自动生成，仅供参考。*")

        report_content = "\n".join(lines)
        report_dir = Path(REPORT_DIR)
        report_dir.mkdir(parents=True, exist_ok=True)
        filepath = report_dir / f"weekly_{date}.md"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report_content)

        print(f"\n[Agent] ✅ 周报已保存: {filepath}")
        return str(filepath)

    def run_screen_only(self):
        """只运行选股"""
        print("运行联合选股...")
        pipeline = StockPipeline()
        result = pipeline.run(top_n_result=15)
        print(f"\n{result.summary}")
        if result.candidates:
            for c in result.candidates:
                icon = {"强": "🔥", "中": "⭐", "弱": "○"}.get(c.buy_signal_strength, "")
                print(f"  {icon} {c.ts_code} {c.name} | {c.sector} | "
                      f"CANSLIM:{c.canslim_grade}({c.canslim_score:.0f}) | "
                      f"总分{c.final_score:.0f} | {c.suggested_entry:.2f}/{c.suggested_stop:.2f}")

    def run_review_only(self):
        """只运行复盘"""
        reviewer = StrategyReviewer()
        stats = reviewer.review_past_recommendations()
        reviewer.print_report(stats)


def main():
    parser = argparse.ArgumentParser(description="A股周度选股Agent")
    parser.add_argument("--screen", action="store_true", help="只运行选股")
    parser.add_argument("--review", action="store_true", help="只运行复盘")
    args = parser.parse_args()

    agent = WeeklyAgent()

    if args.screen:
        agent.run_screen_only()
    elif args.review:
        agent.run_review_only()
    else:
        agent.run_full()


if __name__ == "__main__":
    main()
