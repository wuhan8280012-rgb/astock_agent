#!/usr/bin/env python3
"""
宏观事件日历 — 消息层第一层

功能:
  1. 维护已知宏观事件日期表 (美联储议息、中国LPR、CPI/PMI等)
  2. 检测给定日期是否处于事件敏感窗口 (前后N个交易日)
  3. 输出风险等级和建议仓位折扣系数
  4. 支持自定义事件和在线拉取增量事件

设计原则:
  - 零外部依赖 (仅标准库 + 可选tushare)
  - 事件表可热更新 (JSON文件)
  - 输出可直接乘以信号生成器的仓位上限
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
#  数据结构
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MacroEvent:
    """单条宏观事件"""
    date: str                  # YYYYMMDD
    name: str                  # 事件名称
    category: str              # fed / pboc / data / geopolitical / market
    impact: str                # high / medium / low
    region: str = "CN"         # CN / US / global
    description: str = ""      # 补充说明

    @property
    def impact_score(self) -> float:
        return {"high": 1.0, "medium": 0.6, "low": 0.3}.get(self.impact, 0.3)


@dataclass
class RiskAssessment:
    """宏观风险评估结果"""
    date: str
    risk_level: str            # CRITICAL / ELEVATED / NORMAL
    position_discount: float   # 仓位折扣系数: 0.0 ~ 1.0 (乘到 max_position 上)
    nearby_events: list[dict] = field(default_factory=list)
    quiet_period: bool = False  # 是否建议静默 (不调仓)
    alert_text: str = ""       # 推送告警文本

    def to_dict(self) -> dict:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════════════
#  2026年宏观事件日历 (静态 + 可扩展)
# ══════════════════════════════════════════════════════════════════════════════

# 美联储FOMC议息会议 2026 (决议公布日 北京时间凌晨)
FOMC_2026 = [
    ("20260129", "FOMC议息会议决议"),
    ("20260319", "FOMC议息+经济预测+点阵图"),
    ("20260430", "FOMC议息会议决议"),
    ("20260618", "FOMC议息+经济预测+点阵图"),
    ("20260730", "FOMC议息会议决议"),
    ("20260917", "FOMC议息+经济预测+点阵图"),
    ("20261029", "FOMC议息会议决议"),
    ("20261217", "FOMC议息+经济预测+点阵图"),
]

# 中国LPR公布日 (每月20日, 遇节假日顺延)
LPR_2026 = [
    ("20260120", "1月LPR公布"),
    ("20260220", "2月LPR公布"),
    ("20260320", "3月LPR公布"),
    ("20260420", "4月LPR公布"),
    ("20260520", "5月LPR公布"),
    ("20260620", "6月LPR公布"),
    ("20260720", "7月LPR公布"),
    ("20260820", "8月LPR公布"),
    ("20260921", "9月LPR公布"),  # 20号周日, 顺延
    ("20261020", "10月LPR公布"),
    ("20261120", "11月LPR公布"),
    ("20261221", "12月LPR公布"),  # 20号周日, 顺延
]

# 中国重要经济数据发布 (大致日期, 可通过JSON覆盖精确日期)
CN_DATA_2026 = [
    # CPI/PPI (每月9-12日)
    ("20260110", "12月CPI/PPI"),
    ("20260210", "1月CPI/PPI"),
    ("20260312", "2月CPI/PPI"),
    ("20260411", "3月CPI/PPI"),
    ("20260512", "4月CPI/PPI"),
    ("20260610", "5月CPI/PPI"),
    ("20260710", "6月CPI/PPI"),
    ("20260812", "7月CPI/PPI"),
    ("20260910", "8月CPI/PPI"),
    ("20261015", "9月CPI/PPI"),
    ("20261110", "10月CPI/PPI"),
    ("20261210", "11月CPI/PPI"),
    # PMI (每月最后一个工作日或1号)
    ("20260131", "1月PMI"),
    ("20260228", "2月PMI"),
    ("20260331", "3月PMI"),
    ("20260430", "4月PMI"),
    ("20260531", "5月PMI"),
    ("20260630", "6月PMI"),
    ("20260731", "7月PMI"),
    ("20260831", "8月PMI"),
    ("20260930", "9月PMI"),
    ("20261031", "10月PMI"),
    ("20261130", "11月PMI"),
    ("20261231", "12月PMI"),
    # 社融/M2 (每月10-15日)
    ("20260113", "12月社融/M2"),
    ("20260213", "1月社融/M2"),
    ("20260313", "2月社融/M2"),
    ("20260414", "3月社融/M2"),
    ("20260513", "4月社融/M2"),
    ("20260612", "5月社融/M2"),
    ("20260714", "6月社融/M2"),
    ("20260813", "7月社融/M2"),
    ("20260914", "8月社融/M2"),
    ("20261014", "9月社融/M2"),
    ("20261113", "10月社融/M2"),
    ("20261214", "11月社融/M2"),
    # GDP (1/4/7/10月中旬)
    ("20260120", "2025Q4 GDP"),
    ("20260417", "2026Q1 GDP"),
    ("20260716", "2026Q2 GDP"),
    ("20261020", "2026Q3 GDP"),
]

# A股特殊时段
MARKET_EVENTS_2026 = [
    ("20260105", "新年开门, 机构调仓", "medium"),
    ("20260216", "春节后首个交易日", "medium"),
    ("20260331", "Q1末机构调仓", "medium"),
    ("20260428", "年报/一季报集中披露截止", "high"),
    ("20260630", "H1末机构调仓", "medium"),
    ("20260831", "中报集中披露截止", "high"),
    ("20260930", "Q3末机构调仓/国庆前", "medium"),
    ("20261031", "三季报集中披露截止", "high"),
    ("20261231", "年末机构调仓", "medium"),
]

# 期权/期货交割日 (每月第三个周五)
# 2026年三个周五序列
DELIVERY_2026 = [
    "20260116", "20260220", "20260320", "20260417", "20260515",
    "20260619", "20260717", "20260821", "20260918", "20261016",
    "20261120", "20261218",
]


def _build_default_events() -> list[MacroEvent]:
    """构建默认事件日历"""
    events = []

    # FOMC
    for date, name in FOMC_2026:
        events.append(MacroEvent(
            date=date, name=f"美联储{name}",
            category="fed", impact="high", region="US",
            description="含点阵图" if "点阵图" in name else "",
        ))

    # LPR
    for date, name in LPR_2026:
        events.append(MacroEvent(
            date=date, name=name,
            category="pboc", impact="medium", region="CN",
        ))

    # 经济数据
    for date, name in CN_DATA_2026:
        impact = "high" if "GDP" in name else "medium"
        events.append(MacroEvent(
            date=date, name=name,
            category="data", impact=impact, region="CN",
        ))

    # 市场特殊时段
    for date, name, imp in MARKET_EVENTS_2026:
        events.append(MacroEvent(
            date=date, name=name,
            category="market", impact=imp, region="CN",
        ))

    # 交割日
    for date in DELIVERY_2026:
        events.append(MacroEvent(
            date=date, name="股指期货/期权交割日",
            category="market", impact="medium", region="CN",
            description="三大期指+ETF期权集中交割",
        ))

    return events


# ══════════════════════════════════════════════════════════════════════════════
#  宏观日历引擎
# ══════════════════════════════════════════════════════════════════════════════

class MacroCalendar:
    """
    宏观事件日历。

    用法:
        cal = MacroCalendar()
        assessment = cal.assess(trade_date="20260320")
        if assessment.quiet_period:
            print("建议今日不调仓")
        adjusted_position = max_position * assessment.position_discount
    """

    # 事件影响的窗口范围 (自然日)
    WINDOW_BEFORE = {
        "high": 1,      # 高影响事件: 前1天进入警戒
        "medium": 1,    # 中影响事件: 前1天
        "low": 0,       # 低影响事件: 仅当天
    }
    WINDOW_AFTER = {
        "high": 1,      # 高影响事件: 后1天消化
        "medium": 0,
        "low": 0,
    }

    # 仓位折扣
    DISCOUNT_MAP = {
        "CRITICAL": 0.5,    # 多个高影响事件叠加
        "ELEVATED": 0.75,   # 单个高影响或多个中影响
        "NORMAL": 1.0,      # 无近期事件
    }

    def __init__(self, custom_events_path: str | Path = None):
        """
        Args:
            custom_events_path: 自定义事件JSON文件路径 (可覆盖/新增事件)
        """
        self.events = _build_default_events()

        # 加载自定义事件
        if custom_events_path:
            p = Path(custom_events_path)
            if p.exists():
                try:
                    custom = json.loads(p.read_text("utf-8"))
                    for e in custom:
                        self.events.append(MacroEvent(**e))
                    print(f"[宏观日历] 加载 {len(custom)} 条自定义事件")
                except Exception as ex:
                    print(f"[宏观日历] 加载自定义事件失败: {ex}")

        # 按日期排序
        self.events.sort(key=lambda e: e.date)

    def get_events_in_range(self, start: str, end: str) -> list[MacroEvent]:
        """获取日期范围内的事件"""
        return [e for e in self.events if start <= e.date <= end]

    def get_nearby_events(self, date: str, window_days: int = 3) -> list[MacroEvent]:
        """获取指定日期附近的事件 (自然日窗口)"""
        try:
            dt = datetime.strptime(date, "%Y%m%d")
        except ValueError:
            return []

        nearby = []
        for event in self.events:
            try:
                evt_dt = datetime.strptime(event.date, "%Y%m%d")
            except ValueError:
                continue

            delta = (evt_dt - dt).days
            before = self.WINDOW_BEFORE.get(event.impact, 0)
            after = self.WINDOW_AFTER.get(event.impact, 0)

            # 事件在 [date - after, date + before] 范围内视为"近期"
            # (即: 事件在date之后before天内, 或event在date之前after天内)
            if -after <= delta <= before + 1:
                nearby.append(event)

        return nearby

    def assess(self, trade_date: str) -> RiskAssessment:
        """
        评估指定交易日的宏观风险。

        Returns:
            RiskAssessment 包含风险等级、仓位折扣、近期事件
        """
        # 向前看3天 + 向后看1天的事件
        try:
            dt = datetime.strptime(trade_date, "%Y%m%d")
        except ValueError:
            return RiskAssessment(
                date=trade_date, risk_level="NORMAL",
                position_discount=1.0,
            )

        nearby = []
        for event in self.events:
            try:
                evt_dt = datetime.strptime(event.date, "%Y%m%d")
            except ValueError:
                continue

            delta = (evt_dt - dt).days  # 正数=事件在未来, 负数=事件已过

            before = self.WINDOW_BEFORE.get(event.impact, 0)
            after = self.WINDOW_AFTER.get(event.impact, 0)

            # 当天或前窗口或后窗口内
            in_window = False
            distance_label = ""

            if delta == 0:
                in_window = True
                distance_label = "今日"
            elif 0 < delta <= before + 1:
                # 事件在未来 before+1 天内
                in_window = True
                distance_label = f"明日" if delta == 1 else f"{delta}天后"
            elif -after <= delta < 0:
                # 事件刚过去 after 天内
                in_window = True
                distance_label = f"昨日" if delta == -1 else f"{-delta}天前"

            if in_window:
                nearby.append({
                    "date": event.date,
                    "name": event.name,
                    "category": event.category,
                    "impact": event.impact,
                    "region": event.region,
                    "delta_days": delta,
                    "distance": distance_label,
                    "impact_score": event.impact_score,
                })

        # 计算综合风险
        if not nearby:
            return RiskAssessment(
                date=trade_date, risk_level="NORMAL",
                position_discount=1.0,
                nearby_events=nearby,
            )

        total_impact = sum(e["impact_score"] for e in nearby)
        high_count = sum(1 for e in nearby if e["impact"] == "high")
        today_high = sum(1 for e in nearby if e["impact"] == "high" and e["delta_days"] == 0)

        # 风险等级判定
        if total_impact >= 1.5 or high_count >= 2 or today_high >= 1:
            risk_level = "CRITICAL"
        elif total_impact >= 0.6 or high_count >= 1:
            risk_level = "ELEVATED"
        else:
            risk_level = "NORMAL"

        discount = self.DISCOUNT_MAP[risk_level]

        # 是否建议静默
        quiet = risk_level == "CRITICAL"

        # 构建告警文本
        alert_lines = [f"⚠️ 宏观风险提示 [{risk_level}]"]
        for e in sorted(nearby, key=lambda x: x["impact_score"], reverse=True):
            icon = {"high": "🔴", "medium": "🟡", "low": "⚪"}.get(e["impact"], "⚪")
            alert_lines.append(f"  {icon} {e['distance']}: {e['name']} ({e['region']})")
        if quiet:
            alert_lines.append(f"  📛 建议今日不做主动调仓, 仓位上限折扣至 {discount:.0%}")
        elif discount < 1.0:
            alert_lines.append(f"  ⚡ 仓位上限折扣至 {discount:.0%}")

        return RiskAssessment(
            date=trade_date,
            risk_level=risk_level,
            position_discount=discount,
            nearby_events=nearby,
            quiet_period=quiet,
            alert_text="\n".join(alert_lines),
        )

    def get_month_overview(self, year_month: str) -> list[MacroEvent]:
        """获取某月全部事件, 用于Opus反思时提供宏观上下文
        Args:
            year_month: "202603" 格式
        """
        prefix = year_month[:6]
        return [e for e in self.events if e.date.startswith(prefix)]

    def format_opus_context(self, trade_date: str, lookback_days: int = 30,
                            lookahead_days: int = 14) -> str:
        """
        为Opus反思生成宏观上下文摘要。

        Returns:
            可直接插入Opus prompt的宏观环境描述文本
        """
        try:
            dt = datetime.strptime(trade_date, "%Y%m%d")
        except ValueError:
            return ""

        start = (dt - timedelta(days=lookback_days)).strftime("%Y%m%d")
        end = (dt + timedelta(days=lookahead_days)).strftime("%Y%m%d")

        past_events = [e for e in self.events if start <= e.date <= trade_date]
        future_events = [e for e in self.events if trade_date < e.date <= end]

        lines = [f"宏观事件日历 (截至 {trade_date}):\n"]

        if past_events:
            lines.append("【近期已发生事件】")
            for e in past_events[-10:]:  # 最近10条
                lines.append(f"  {e.date} [{e.impact.upper()}] {e.name}")

        if future_events:
            lines.append("\n【未来两周待发生事件】")
            for e in future_events[:10]:
                lines.append(f"  {e.date} [{e.impact.upper()}] {e.name}")

        # 统计当前密集度
        assessment = self.assess(trade_date)
        lines.append(f"\n当前宏观风险等级: {assessment.risk_level}")
        lines.append(f"建议仓位折扣: {assessment.position_discount:.0%}")

        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="宏观事件日历")
    parser.add_argument("--date", type=str, help="查询日期 YYYYMMDD")
    parser.add_argument("--month", type=str, help="查询月份 YYYYMM")
    parser.add_argument("--opus-context", action="store_true", help="输出Opus反思上下文")
    args = parser.parse_args()

    cal = MacroCalendar()

    if args.month:
        events = cal.get_month_overview(args.month)
        print(f"\n{args.month} 宏观事件日历 ({len(events)} 条):\n")
        for e in events:
            icon = {"high": "🔴", "medium": "🟡", "low": "⚪"}.get(e.impact, "⚪")
            print(f"  {e.date}  {icon} [{e.category:>8s}] {e.name}")
    elif args.date:
        assessment = cal.assess(args.date)
        print(assessment.alert_text or "无近期宏观事件")
        if args.opus_context:
            print("\n" + "=" * 50)
            print(cal.format_opus_context(args.date))
    else:
        # 默认: 今天
        today = datetime.now().strftime("%Y%m%d")
        assessment = cal.assess(today)
        print(assessment.alert_text or f"{today}: 无近期宏观事件, 风险正常")
