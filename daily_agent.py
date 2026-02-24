#!/usr/bin/env python3
"""
每日A股前瞻 Agent - 主调度入口

运行方式:
  python daily_agent.py              # 生成今日完整报告
  python daily_agent.py --sentiment  # 只运行情绪分析
  python daily_agent.py --sector     # 只运行板块轮动
  python daily_agent.py --macro      # 只运行宏观监控
  python daily_agent.py --risk       # 只运行风控检查(需配置持仓)
  python daily_agent.py --push       # 生成报告并推送

定时任务 (crontab -e):
  30 6 * * 1-5 cd /path/to/astock_agent && python daily_agent.py --push
"""

import sys
import os
import json
import argparse
import traceback
from datetime import datetime
from pathlib import Path

# 确保可以找到模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_fetcher import get_fetcher
from skills.sentiment import SentimentSkill
from skills.sector_rotation import SectorRotationSkill
from skills.macro_monitor import MacroSkill
from skills.risk_control import RiskControlSkill
from knowledge_base import KnowledgeBase, init_default_knowledge
from report_generator import ReportGenerator
from scratchpad import Scratchpad


# ============================================================
# 持仓配置 (手动更新或从交易系统自动同步)
# ============================================================
POSITIONS_FILE = "./knowledge/positions.json"


def load_positions() -> list:
    """加载当前持仓"""
    path = Path(POSITIONS_FILE)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def save_positions(positions: list):
    """保存持仓"""
    path = Path(POSITIONS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(positions, f, ensure_ascii=False, indent=2)


# ============================================================
# 推送功能
# ============================================================
def push_report(report_path: str, summary: str = ""):
    """推送报告到微信/飞书"""
    try:
        from config import WECHAT_WEBHOOK, FEISHU_WEBHOOK, SERVERCHAN_KEY
    except ImportError:
        print("[Push] 未配置推送")
        return

    report_content = ""
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report_content = f.read()

    # Server酱 (微信推送)
    if SERVERCHAN_KEY:
        try:
            import requests
            url = f"https://sctapi.ftqq.com/{SERVERCHAN_KEY}.send"
            data = {
                "title": f"A股前瞻 {datetime.now().strftime('%m/%d')}",
                "desp": report_content[:4000],  # Server酱有长度限制
            }
            resp = requests.post(url, data=data)
            print(f"[Push] Server酱: {resp.status_code}")
        except Exception as e:
            print(f"[Push] Server酱失败: {e}")

    # 企业微信机器人
    if WECHAT_WEBHOOK:
        try:
            import requests
            data = {
                "msgtype": "markdown",
                "markdown": {"content": report_content[:4000]},
            }
            resp = requests.post(WECHAT_WEBHOOK, json=data)
            print(f"[Push] 企业微信: {resp.status_code}")
        except Exception as e:
            print(f"[Push] 企业微信失败: {e}")

    # 飞书机器人
    if FEISHU_WEBHOOK:
        try:
            import requests
            data = {
                "msg_type": "text",
                "content": {"text": summary or report_content[:2000]},
            }
            resp = requests.post(FEISHU_WEBHOOK, json=data)
            print(f"[Push] 飞书: {resp.status_code}")
        except Exception as e:
            print(f"[Push] 飞书失败: {e}")


# ============================================================
# 主流程
# ============================================================
class DailyAgent:
    """每日前瞻 Agent"""

    def __init__(self):
        self.fetcher = get_fetcher()
        self.kb = init_default_knowledge()
        self.reporter = ReportGenerator()
        self.sp = Scratchpad()

    def run_full(self, push: bool = False) -> str:
        """运行完整的每日前瞻流程"""
        print(f"\n{'='*60}")
        print(f"  🤖 A股每日前瞻 Agent 启动")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")

        date = self.fetcher.get_latest_trade_date()
        print(f"[Agent] 分析日期: {date}")
        self.sp.log_start("daily_agent", trade_date=date)

        # ---- Step 1: 拉取指数快照 ----
        print("\n[Step 1/6] 拉取市场数据...")
        index_snapshot = self.fetcher.get_all_index_snapshot()
        self._print_snapshot(index_snapshot)

        # ---- Step 2: 情绪分析 ----
        print("\n[Step 2/6] 运行情绪分析...")
        sentiment_skill = SentimentSkill()
        sentiment_result = sentiment_skill.analyze()
        self.sp.log("sentiment", output_data=sentiment_result.to_dict())
        print(f"  情绪评级: {sentiment_result.overall_level} ({sentiment_result.overall_score:+.1f})")
        print(f"  仓位建议: {sentiment_result.suggested_position}")

        # ---- Step 3: 宏观流动性 ----
        print("\n[Step 3/6] 运行宏观监控...")
        macro_skill = MacroSkill()
        macro_result = macro_skill.analyze()
        self.sp.log("macro", output_data=macro_result.to_dict())
        print(f"  流动性: {macro_result.liquidity_level} ({macro_result.overall_score:+.1f})")
        print(f"  影响: {macro_result.market_impact}")

        # ---- Step 4: 板块轮动 ----
        print("\n[Step 4/6] 运行板块轮动...")
        sector_skill = SectorRotationSkill()
        sector_result = sector_skill.analyze()
        self.sp.log("sector_rotation", output_data=sector_result.to_dict())
        if sector_result.top_sectors:
            print(f"  Top板块: {', '.join(s.name for s in sector_result.top_sectors)}")
        for sig in sector_result.rotation_signals:
            print(f"  信号: {sig}")

        # ---- Step 5: 风控检查 ----
        print("\n[Step 5/6] 运行风控检查...")
        positions = load_positions()
        risk_result = None
        if positions:
            risk_skill = RiskControlSkill()
            risk_obj = risk_skill.analyze(
                positions=positions,
                sentiment_level=sentiment_result.overall_level,
                date=date,
            )
            risk_result = risk_obj.to_dict()
            print(f"  {risk_obj.summary}")
            for action in risk_obj.actions:
                print(f"  {action}")
        else:
            print("  ⚠️ 未配置持仓信息，跳过风控检查")
            print(f"  💡 请编辑 {POSITIONS_FILE} 添加持仓")

        # ---- Step 6: 知识库匹配 ----
        print("\n[Step 6/6] 匹配历史模式...")
        current_state = {
            "sentiment": sentiment_result.overall_level,
            "liquidity": macro_result.liquidity_level,
        }
        # 从宏观数据补充状态
        for sig in macro_result.signals:
            if "北向" in sig.name:
                current_state["north_flow_trend"] = sig.trend
            if "两融" in sig.name:
                current_state["margin_trend"] = sig.trend
            if "成交额" in sig.name:
                current_state["turnover_trend"] = sig.trend

        pattern_matches = self.kb.match_pattern(current_state)
        if pattern_matches:
            for m in pattern_matches[:3]:
                print(f"  匹配: {m['name']} (匹配度:{m['match_score']:.0%})")
        else:
            print("  未匹配到历史模式")

        # 获取相关经验教训
        lessons = self.kb.get_lessons()

        # ---- 生成报告 ----
        print("\n[Report] 生成报告...")
        report_path = self.reporter.generate(
            date=date,
            index_snapshot=index_snapshot,
            sentiment_report=sentiment_result.to_dict(),
            sector_report=sector_result.to_dict(),
            macro_report=macro_result.to_dict(),
            risk_report=risk_result,
            pattern_matches=pattern_matches,
            lessons=lessons,
        )
        print(f"[Report] ✅ 报告已保存: {report_path}")

        # ---- 推送 ----
        if push:
            summary = (
                f"【A股前瞻 {date}】\n"
                f"情绪: {sentiment_result.overall_level} | "
                f"流动性: {macro_result.liquidity_level}\n"
                f"建议: {sentiment_result.suggested_position}\n"
                f"强势板块: {', '.join(s.name for s in sector_result.top_sectors[:3])}"
            )
            push_report(report_path, summary)

        print(f"\n{'='*60}")
        print(f"  ✅ Agent 执行完成")
        print(f"{'='*60}\n")
        self.sp.log_end("daily_agent", report_path=report_path)

        return report_path

    def run_sentiment(self):
        """只运行情绪分析"""
        skill = SentimentSkill()
        result = skill.analyze()
        self.sp.log("sentiment", output_data=result.to_dict())
        print(f"\n情绪评级: {result.overall_level} ({result.overall_score:+.1f})")
        print(f"仓位建议: {result.suggested_position}")
        for sig in result.signals:
            print(f"  [{sig.name}] {sig.level} - {sig.detail}")

    def run_sector(self):
        """只运行板块轮动"""
        skill = SectorRotationSkill()
        result = skill.analyze()
        self.sp.log("sector_rotation", output_data=result.to_dict())
        print(f"\n板块轮动 ({result.date})")
        for s in result.top_sectors:
            print(f"  #{s.rank} {s.name}: {s.ret_20d:+.1f}% | {s.signal}")
        for sig in result.rotation_signals:
            print(f"  → {sig}")

    def run_macro(self):
        """只运行宏观监控"""
        skill = MacroSkill()
        result = skill.analyze()
        self.sp.log("macro", output_data=result.to_dict())
        print(f"\n流动性: {result.liquidity_level} ({result.overall_score:+.1f})")
        print(f"影响: {result.market_impact}")
        for sig in result.signals:
            print(f"  [{sig.name}] {sig.trend} - {sig.detail}")

    def run_risk(self):
        """只运行风控检查"""
        positions = load_positions()
        if not positions:
            print(f"\n⚠️ 未配置持仓，请编辑 {POSITIONS_FILE}")
            self._show_position_template()
            return
        # 先获取情绪
        sentiment = SentimentSkill().analyze()
        skill = RiskControlSkill()
        result = skill.analyze(positions, sentiment.overall_level,
                               self.fetcher.get_latest_trade_date())
        print(f"\n{result.summary}")
        for p in result.position_checks:
            print(f"  {p.name}: {p.action} | 盈亏{p.current_pnl_pct*100:.1f}%")
        for action in result.actions:
            print(f"  {action}")

    def _print_snapshot(self, snapshot: dict):
        for code, info in snapshot.items():
            arrow = "↑" if info["change_pct"] > 0 else "↓" if info["change_pct"] < 0 else "→"
            print(f"  {info['name']}: {info['close']:.2f} {arrow} {info['change_pct']:+.2f}%")

    def _show_position_template(self):
        template = [
            {
                "ts_code": "600519.SH",
                "name": "贵州茅台",
                "position_pct": 0.08,
                "cost_price": 1500.0,
                "current_price": 1550.0,
                "sector": "食品饮料",
                "buy_date": "20260210",
            }
        ]
        print(f"\n持仓文件模板 ({POSITIONS_FILE}):")
        print(json.dumps(template, ensure_ascii=False, indent=2))


# ============================================================
# CLI 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="A股每日前瞻 Agent")
    parser.add_argument("--sentiment", action="store_true", help="只运行情绪分析")
    parser.add_argument("--sector", action="store_true", help="只运行板块轮动")
    parser.add_argument("--macro", action="store_true", help="只运行宏观监控")
    parser.add_argument("--risk", action="store_true", help="只运行风控检查")
    parser.add_argument("--push", action="store_true", help="生成报告并推送")
    args = parser.parse_args()

    agent = DailyAgent()

    try:
        if args.sentiment:
            agent.run_sentiment()
        elif args.sector:
            agent.run_sector()
        elif args.macro:
            agent.run_macro()
        elif args.risk:
            agent.run_risk()
        else:
            agent.run_full(push=args.push)
    except KeyboardInterrupt:
        print("\n[Agent] 用户中断")
    except Exception as e:
        print(f"\n[Agent] 运行出错: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
