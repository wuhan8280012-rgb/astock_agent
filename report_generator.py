"""
报告生成层: 将各 Skill 的分析结果整合为结构化报告

支持两种模式:
  1. 本地模板模式 (默认): 不需要 API，使用模板生成
  2. Claude API 模式 (可选): 用 Claude 生成自然语言分析
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from config import REPORT_DIR, REPORT_FORMAT, CLAUDE_API_KEY, CLAUDE_MODEL
except ImportError:
    REPORT_DIR = "./reports"
    REPORT_FORMAT = "md"
    CLAUDE_API_KEY = ""
    CLAUDE_MODEL = "claude-sonnet-4-20250514"


class ReportGenerator:
    """报告生成器"""

    def __init__(self):
        self.report_dir = Path(REPORT_DIR)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.use_ai = bool(CLAUDE_API_KEY)

    def generate(
        self,
        date: str,
        index_snapshot: dict,
        sentiment_report: dict,
        sector_report: dict,
        macro_report: dict,
        risk_report: dict = None,
        pattern_matches: list = None,
        lessons: list = None,
    ) -> str:
        """
        生成每日前瞻报告

        Returns: 报告文件路径
        """
        if self.use_ai:
            content = self._generate_with_ai(
                date, index_snapshot, sentiment_report,
                sector_report, macro_report, risk_report,
                pattern_matches, lessons,
            )
        else:
            content = self._generate_template(
                date, index_snapshot, sentiment_report,
                sector_report, macro_report, risk_report,
                pattern_matches, lessons,
            )

        # 保存报告
        filename = f"daily_preview_{date}.{REPORT_FORMAT}"
        filepath = self.report_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"[Report] 报告已保存: {filepath}")
        return str(filepath)

    def _generate_template(
        self, date, index_snapshot, sentiment, sector, macro,
        risk=None, patterns=None, lessons=None,
    ) -> str:
        """使用模板生成报告"""
        lines = []
        date_fmt = f"{date[:4]}-{date[4:6]}-{date[6:]}" if len(date) == 8 else date

        # ========== 标题 ==========
        lines.append(f"# 📊 A股每日前瞻 | {date_fmt}")
        lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("")

        # ========== 指数快照 ==========
        lines.append("## 一、市场概览")
        lines.append("")
        if index_snapshot:
            lines.append("| 指数 | 收盘 | 涨跌幅 |")
            lines.append("|------|------|--------|")
            for code, info in index_snapshot.items():
                chg = info['change_pct']
                arrow = "🔴" if chg < 0 else "🟢" if chg > 0 else "⚪"
                lines.append(f"| {info['name']} | {info['close']:.2f} | {arrow} {chg:+.2f}% |")
            lines.append("")

        # ========== 情绪评估 ==========
        lines.append("## 二、市场情绪")
        lines.append("")
        if sentiment:
            level = sentiment.get("overall_level", "未知")
            score = sentiment.get("overall_score", 0)
            pos = sentiment.get("suggested_position", "")

            # 情绪图标
            emoji_map = {
                "极度贪婪": "🔴🔴", "贪婪": "🔴",
                "中性": "🟡", "恐慌": "🟢", "极度恐慌": "🟢🟢",
            }
            emoji = emoji_map.get(level, "⚪")

            lines.append(f"**综合情绪: {emoji} {level}** (得分: {score:+.1f})")
            lines.append(f"**仓位建议: {pos}**")
            lines.append("")

            if sentiment.get("signals"):
                lines.append("| 指标 | 状态 | 评分 | 详情 |")
                lines.append("|------|------|------|------|")
                for sig in sentiment["signals"]:
                    lines.append(f"| {sig['name']} | {sig['level']} | {sig['score']:+d} | {sig['detail']} |")
                lines.append("")

        # ========== 宏观流动性 ==========
        lines.append("## 三、宏观流动性")
        lines.append("")
        if macro:
            level = macro.get("liquidity_level", "未知")
            impact = macro.get("market_impact", "")
            lines.append(f"**流动性评级: {level}**")
            lines.append(f"**市场影响: {impact}**")
            lines.append("")

            if macro.get("signals"):
                for sig in macro["signals"]:
                    lines.append(f"- **{sig['name']}**: {sig['trend']} — {sig['detail']}")
                lines.append("")

        # ========== 板块轮动 ==========
        lines.append("## 四、板块轮动")
        lines.append("")
        if sector:
            if sector.get("top_sectors"):
                lines.append("### 强势板块 Top 5")
                lines.append("")
                lines.append("| 排名 | 板块 | 20日涨幅 | 综合得分 | 信号 |")
                lines.append("|------|------|----------|----------|------|")
                for s in sector["top_sectors"]:
                    lines.append(
                        f"| {s['rank']} | {s['name']} | {s['ret_20d']:+.1f}% | "
                        f"{s['score']:.1f} | {s['signal']} |"
                    )
                lines.append("")

            if sector.get("rotation_signals"):
                lines.append("### 轮动信号")
                for sig in sector["rotation_signals"]:
                    lines.append(f"- {sig}")
                lines.append("")

        # ========== 风控 ==========
        if risk:
            lines.append("## 五、风控状态")
            lines.append("")
            lines.append(f"**{risk.get('summary', '')}**")
            lines.append("")

            if risk.get("portfolio_alerts"):
                for alert in risk["portfolio_alerts"]:
                    lines.append(f"- {alert}")
                lines.append("")

            if risk.get("actions"):
                lines.append("### 行动建议")
                for action in risk["actions"]:
                    lines.append(f"- {action}")
                lines.append("")

        # ========== 历史模式匹配 ==========
        if patterns:
            lines.append("## 六、历史模式匹配")
            lines.append("")
            for p in patterns[:3]:
                lines.append(f"### 📌 {p['name']} (匹配度: {p['match_score']:.0%})")
                lines.append(f"- 典型结果: {p['typical_outcome']}")
                lines.append(f"- 置信度: {p.get('confidence', '中')}")
                if p.get("occurrences"):
                    lines.append(f"- 历史发生: {', '.join(p['occurrences'][:5])}")
                lines.append("")

        # ========== 经验提醒 ==========
        if lessons:
            lines.append("## 七、经验提醒")
            lines.append("")
            for l in lessons[:5]:
                lines.append(f"- 💡 {l['lesson']}")
            lines.append("")

        # ========== 免责 ==========
        lines.append("---")
        lines.append("*以上分析由Agent系统自动生成，仅供参考，不构成投资建议。投资决策需结合个人判断。*")

        return "\n".join(lines)

    def _generate_with_ai(
        self, date, index_snapshot, sentiment, sector, macro,
        risk=None, patterns=None, lessons=None,
    ) -> str:
        """使用 Claude API 生成自然语言报告"""
        try:
            import anthropic
        except ImportError:
            print("[Report] anthropic库未安装，使用模板模式")
            return self._generate_template(
                date, index_snapshot, sentiment, sector, macro,
                risk, patterns, lessons,
            )

        # 构建提示词
        data_summary = json.dumps({
            "date": date,
            "index_snapshot": index_snapshot,
            "sentiment": sentiment,
            "macro": macro,
            "sector": sector,
            "risk": risk,
            "pattern_matches": patterns,
        }, ensure_ascii=False, indent=2)

        prompt = f"""你是一位资深A股投研分析师。请根据以下数据，生成今日市场前瞻报告。

要求：
1. 语言专业但不晦涩，适合有一定基础的投资者阅读
2. 重点突出：最关键的1-2个信号放在最前面
3. 给出明确的操作建议（仓位、关注板块、风险提示）
4. 如果有历史模式匹配，说明其参考价值
5. 控制在800字以内
6. 用Markdown格式

数据：
{data_summary}"""

        client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


if __name__ == "__main__":
    # 测试模板报告
    gen = ReportGenerator()
    content = gen._generate_template(
        date="20260223",
        index_snapshot={
            "000001.SH": {"name": "上证指数", "close": 3250.00, "change_pct": 0.85},
            "399006.SZ": {"name": "创业板指", "close": 2100.00, "change_pct": -0.32},
        },
        sentiment={
            "overall_level": "中性",
            "overall_score": 0.2,
            "suggested_position": "50%仓位，均衡配置",
            "signals": [
                {"name": "涨跌比", "level": "中性", "score": 0, "detail": "上涨2500家/下跌2200家"},
                {"name": "北向资金", "level": "贪婪", "score": 1, "detail": "今日+68亿"},
            ],
        },
        sector={
            "top_sectors": [
                {"rank": 1, "name": "有色金属", "ret_20d": 8.5, "score": 85.2, "signal": "🔥 强势持续"},
                {"rank": 2, "name": "计算机", "ret_20d": 6.2, "score": 78.1, "signal": "⭐ 缩量整理"},
            ],
            "rotation_signals": ["缩量蓄势关注: 计算机"],
        },
        macro={
            "liquidity_level": "中性偏松",
            "market_impact": "流动性尚可，市场有支撑",
            "signals": [
                {"name": "SHIBOR", "trend": "中性", "detail": "1周SHIBOR=1.85%"},
            ],
        },
    )
    print(content)
