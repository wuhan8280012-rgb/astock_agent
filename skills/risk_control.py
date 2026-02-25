"""
Skill 4: T+1 风控模型

A股特色风控:
  1. T+1 约束下的隔夜风险管理
  2. 涨跌停约束下的流动性风险
  3. 仓位管理（结合情绪评级动态调整）
  4. 止损纪律与连续亏损熔断

输入: 当前持仓 + 情绪评级
输出: 风控建议 + 仓位调整方案
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import json
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import SKILL_PARAMS, TRADE_LOG
    PARAMS = SKILL_PARAMS.get("risk", {})
except ImportError:
    PARAMS = {}
    TRADE_LOG = "./knowledge/trade_log.json"


@dataclass
class PositionCheck:
    """单个持仓的风控检查"""
    ts_code: str
    name: str
    position_pct: float       # 占总资金百分比
    current_pnl_pct: float    # 当前盈亏百分比
    sector: str = ""
    alerts: List[str] = field(default_factory=list)
    action: str = ""          # "持有" / "减仓" / "止损" / "清仓"


@dataclass
class RiskReport:
    """风控报告"""
    date: str
    total_position: float = 0        # 总仓位百分比
    max_position_limit: float = 1.0  # 根据情绪调整的最大仓位
    position_checks: List[PositionCheck] = field(default_factory=list)
    portfolio_alerts: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "total_position": self.total_position,
            "max_position_limit": self.max_position_limit,
            "portfolio_alerts": self.portfolio_alerts,
            "actions": self.actions,
            "summary": self.summary,
            "positions": [
                {"code": p.ts_code, "name": p.name, "pct": p.position_pct,
                 "pnl": p.current_pnl_pct, "alerts": p.alerts, "action": p.action}
                for p in self.position_checks
            ],
        }

    def to_brief(self) -> str:
        """压缩报告 (~50 tokens)"""
        stops = [p.name for p in self.position_checks if p.action == "止损"]
        warns = [p.name for p in self.position_checks if "减仓" in p.action]
        parts = [
            f"[风控] 仓位{self.total_position*100:.0f}%/{self.max_position_limit*100:.0f}%上限",
            f"预警{len(self.portfolio_alerts)}条",
        ]
        if stops:
            parts.append(f"止损:{','.join(stops)}")
        if warns:
            parts.append(f"减仓:{','.join(warns)}")
        if not stops and not warns:
            parts.append("持仓正常")
        return " | ".join(parts)


class RiskControlSkill:
    """T+1 风控模型"""

    def __init__(self):
        self.max_single = PARAMS.get("max_single_position", 0.10)
        self.max_sector = PARAMS.get("max_sector_exposure", 0.30)
        self.stop_loss = PARAMS.get("stop_loss_pct", 0.05)
        self.consec_limit = PARAMS.get("consecutive_stop_limit", 2)
        self.forced_limit = PARAMS.get("forced_position_limit", 0.30)
        self.trade_log_path = Path(TRADE_LOG)

    def analyze(
        self,
        positions: List[Dict],
        sentiment_level: str = "中性",
        date: str = ""
    ) -> RiskReport:
        """
        执行风控检查

        Args:
            positions: 持仓列表，每项包含:
                {
                    "ts_code": "000001.SZ",
                    "name": "平安银行",
                    "position_pct": 0.08,    # 占总资金8%
                    "cost_price": 10.5,
                    "current_price": 10.2,
                    "sector": "银行",
                    "buy_date": "20260220",
                }
            sentiment_level: 情绪Skill输出的等级
            date: 日期
        """
        report = RiskReport(date=date)

        # 1. 根据情绪等级设置最大仓位
        report.max_position_limit = self._get_max_position(sentiment_level)

        # 2. 检查总仓位
        total_pos = sum(p.get("position_pct", 0) for p in positions)
        report.total_position = round(total_pos, 4)

        if total_pos > report.max_position_limit:
            excess = total_pos - report.max_position_limit
            report.portfolio_alerts.append(
                f"⚠️ 总仓位{total_pos*100:.1f}%超过当前情绪建议上限"
                f"{report.max_position_limit*100:.0f}%，需减仓{excess*100:.1f}%"
            )

        # 3. 检查连续止损
        consec_stops = self._check_consecutive_stops()
        if consec_stops >= self.consec_limit:
            report.portfolio_alerts.append(
                f"🔴 近期连续止损{consec_stops}次，触发熔断机制，"
                f"强制降仓至{self.forced_limit*100:.0f}%以下"
            )
            report.max_position_limit = min(report.max_position_limit, self.forced_limit)

        # 4. 逐个持仓检查
        sector_exposure = {}
        for pos in positions:
            check = self._check_position(pos)
            report.position_checks.append(check)

            # 累计板块暴露
            sector = pos.get("sector", "未分类")
            sector_exposure[sector] = sector_exposure.get(sector, 0) + pos.get("position_pct", 0)

        # 5. 检查板块集中度
        for sector, exposure in sector_exposure.items():
            if exposure > self.max_sector:
                report.portfolio_alerts.append(
                    f"⚠️ {sector}板块暴露{exposure*100:.1f}%超过上限{self.max_sector*100:.0f}%"
                )

        # 6. 生成行动建议
        report.actions = self._generate_actions(report)

        # 7. 摘要
        alert_count = len(report.portfolio_alerts)
        stop_count = sum(1 for p in report.position_checks if p.action == "止损")
        report.summary = (
            f"总仓位{total_pos*100:.1f}% | 上限{report.max_position_limit*100:.0f}% | "
            f"情绪:{sentiment_level} | 预警{alert_count}条 | 止损信号{stop_count}只"
        )

        return report

    def _get_max_position(self, sentiment_level: str) -> float:
        """
        根据情绪等级确定最大仓位 — 逆向策略

        逻辑: 市场越热 → 允许仓位越低；市场越冷 → 允许仓位越高
        这是一种纪律约束，不是预测。
        """
        contrarian_map = {
            "极度贪婪": 0.30,
            "贪婪": 0.50,
            "中性": 0.70,
            "恐慌": 0.80,
            "极度恐慌": 0.90,
        }
        return contrarian_map.get(sentiment_level, 0.70)

    def get_position_perspectives(self, sentiment_level: str) -> dict:
        """
        三视角仓位建议 — 供 debate.py 风控环节使用
        """
        contrarian = self._get_max_position(sentiment_level)
        trend_map = {
            "极度贪婪": 0.90,
            "贪婪": 0.80,
            "中性": 0.50,
            "恐慌": 0.30,
            "极度恐慌": 0.20,
        }
        trend_follow = trend_map.get(sentiment_level, 0.50)
        return {
            "contrarian": contrarian,
            "trend_follow": trend_follow,
            "neutral": round((contrarian + trend_follow) / 2, 2),
        }

    def _check_position(self, pos: dict) -> PositionCheck:
        """检查单个持仓"""
        cost = pos.get("cost_price", 0)
        current = pos.get("current_price", 0)
        pnl_pct = (current / cost - 1) if cost > 0 else 0

        check = PositionCheck(
            ts_code=pos.get("ts_code", ""),
            name=pos.get("name", ""),
            position_pct=pos.get("position_pct", 0),
            current_pnl_pct=round(pnl_pct, 4),
            sector=pos.get("sector", ""),
        )

        # 单只仓位过大
        if check.position_pct > self.max_single:
            check.alerts.append(
                f"仓位{check.position_pct*100:.1f}%超过单只上限{self.max_single*100:.0f}%"
            )

        # 止损检查
        if pnl_pct < -self.stop_loss:
            check.alerts.append(f"亏损{pnl_pct*100:.1f}%触及止损线-{self.stop_loss*100:.0f}%")
            check.action = "止损"
        elif pnl_pct < -self.stop_loss * 0.6:
            check.alerts.append(f"亏损{pnl_pct*100:.1f}%接近止损线，密切关注")
            check.action = "减仓观察"
        elif pnl_pct > self.stop_loss * 3:
            # 盈利超过3倍止损幅度 → 考虑止盈
            check.alerts.append(f"盈利{pnl_pct*100:.1f}%，建议设置移动止盈")
            check.action = "持有(设移动止盈)"
        else:
            check.action = "持有"

        return check

    def _check_consecutive_stops(self) -> int:
        """检查近期连续止损次数"""
        if not self.trade_log_path.exists():
            return 0
        try:
            with open(self.trade_log_path, "r") as f:
                logs = json.load(f)
            # 取最近10条交易记录
            recent = logs[-10:] if len(logs) > 10 else logs
            # 从最近往前数连续止损
            count = 0
            for trade in reversed(recent):
                if trade.get("exit_reason") == "止损":
                    count += 1
                else:
                    break
            return count
        except Exception:
            return 0

    def _generate_actions(self, report: RiskReport) -> List[str]:
        """生成综合行动建议"""
        actions = []

        # 需要止损的
        stops = [p for p in report.position_checks if p.action == "止损"]
        if stops:
            names = [f"{p.name}({p.current_pnl_pct*100:.1f}%)" for p in stops]
            actions.append(f"🔴 止损: {', '.join(names)}")

        # 需要减仓的
        reduces = [p for p in report.position_checks if "减仓" in p.action]
        if reduces:
            names = [p.name for p in reduces]
            actions.append(f"🟡 减仓观察: {', '.join(names)}")

        # 总仓位调整
        if report.total_position > report.max_position_limit:
            excess = (report.total_position - report.max_position_limit) * 100
            actions.append(f"📉 需要减仓{excess:.1f}%使总仓位降至{report.max_position_limit*100:.0f}%以下")

        if not actions:
            actions.append("✅ 所有持仓风控正常")

        return actions


# ========================================================
# 交易日志工具
# ========================================================
def log_trade(trade: dict, log_path: str = None):
    """
    记录一笔交易
    trade = {
        "date": "20260223",
        "ts_code": "000001.SZ",
        "name": "平安银行",
        "direction": "sell",
        "price": 10.2,
        "exit_reason": "止损",  # 止损/止盈/调仓/手动
        "pnl_pct": -0.05,
    }
    """
    path = Path(log_path or TRADE_LOG)
    path.parent.mkdir(parents=True, exist_ok=True)
    logs = []
    if path.exists():
        try:
            with open(path, "r") as f:
                logs = json.load(f)
        except Exception:
            logs = []
    logs.append(trade)
    with open(path, "w") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 模拟持仓测试
    test_positions = [
        {
            "ts_code": "600519.SH", "name": "贵州茅台",
            "position_pct": 0.08, "cost_price": 1500, "current_price": 1550,
            "sector": "食品饮料",
        },
        {
            "ts_code": "000858.SZ", "name": "五粮液",
            "position_pct": 0.06, "cost_price": 120, "current_price": 115,
            "sector": "食品饮料",
        },
        {
            "ts_code": "002594.SZ", "name": "比亚迪",
            "position_pct": 0.12, "cost_price": 250, "current_price": 230,
            "sector": "汽车",
        },
        {
            "ts_code": "300750.SZ", "name": "宁德时代",
            "position_pct": 0.10, "cost_price": 200, "current_price": 210,
            "sector": "电气设备",
        },
    ]

    skill = RiskControlSkill()
    result = skill.analyze(test_positions, sentiment_level="贪婪", date="20260223")

    print(f"\n{'='*60}")
    print(f"  风控报告  {result.date}")
    print(f"{'='*60}")
    print(f"  {result.summary}")
    print()
    for p in result.position_checks:
        status = "⚠️" if p.alerts else "✅"
        print(f"  {status} {p.name}: 仓位{p.position_pct*100:.1f}% | "
              f"盈亏{p.current_pnl_pct*100:.1f}% | {p.action}")
        for alert in p.alerts:
            print(f"     → {alert}")
    print()
    for alert in result.portfolio_alerts:
        print(f"  {alert}")
    print()
    for action in result.actions:
        print(f"  {action}")
    print(f"{'='*60}")
