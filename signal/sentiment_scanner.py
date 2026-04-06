#!/usr/bin/env python3
"""
舆情情绪扫描器 — 消息层第二层

功能:
  1. 通过tushare新闻接口/东方财富/同花顺 拉取市场新闻标题
  2. 关键词情绪打分 (无需NLP模型, 纯规则)
  3. 输出恐慌/贪婪指数, 可叠加到信号生成器的仓位判断中
  4. 为Opus进化反思提供舆情上下文

设计原则:
  - 轻量: 只做关键词匹配, 不依赖NLP库
  - 可降级: tushare新闻接口不可用时返回中性
  - 输出可量化: 恐慌指数 -1.0 ~ +1.0
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
#  情绪词典
# ══════════════════════════════════════════════════════════════════════════════

# 恐慌/利空关键词 (权重越大 = 越恐慌)
FEAR_KEYWORDS: dict[str, float] = {
    # 极端恐慌 (权重 3.0)
    "暴跌": 3.0, "崩盘": 3.0, "熔断": 3.0, "股灾": 3.0, "危机": 3.0,
    "爆雷": 3.0, "踩踏": 3.0, "恐慌": 3.0, "黑天鹅": 3.0, "系统性风险": 3.0,

    # 强利空 (权重 2.0)
    "大跌": 2.0, "急跌": 2.0, "跳水": 2.0, "重挫": 2.0, "暴雷": 2.0,
    "违约": 2.0, "制裁": 2.0, "脱钩": 2.0, "战争": 2.0, "冲突升级": 2.0,
    "加征关税": 2.0, "贸易战": 2.0, "美债收益率飙升": 2.0,
    "资金外流": 2.0, "外资撤离": 2.0, "千股跌停": 2.0,
    "退市": 2.0, "ST": 2.0, "造假": 2.0, "调查": 2.0,

    # 中度利空 (权重 1.0)
    "下跌": 1.0, "回调": 1.0, "下行": 1.0, "承压": 1.0, "走弱": 1.0,
    "缩量": 1.0, "减持": 1.0, "解禁": 1.0, "高位套牢": 1.0,
    "利空": 1.0, "担忧": 1.0, "不确定性": 1.0, "风险": 1.0,
    "监管收紧": 1.0, "整顿": 1.0, "叫停": 1.0, "处罚": 1.0,
    "通胀超预期": 1.0, "加息": 1.0, "缩表": 1.0,
    "失业率上升": 1.0, "经济放缓": 1.0, "衰退": 1.5,

    # 轻度利空 (权重 0.5)
    "震荡": 0.5, "分化": 0.5, "观望": 0.5, "谨慎": 0.5,
}

# 贪婪/利好关键词 (权重越大 = 越乐观)
GREED_KEYWORDS: dict[str, float] = {
    # 极度乐观 (权重 3.0)
    "暴涨": 3.0, "大涨": 2.5, "涨停潮": 3.0, "井喷": 3.0,
    "牛市": 3.0, "史上最高": 3.0, "全面上涨": 3.0,

    # 强利好 (权重 2.0)
    "上涨": 1.5, "大幅上涨": 2.0, "强势": 2.0, "放量上攻": 2.0,
    "突破": 2.0, "新高": 2.0, "资金流入": 2.0, "外资加仓": 2.0,
    "北向资金大幅净买入": 2.0, "降息": 2.0, "降准": 2.0,
    "刺激政策": 2.0, "利好": 2.0, "重大利好": 2.5,
    "超预期": 1.5,

    # 中度利好 (权重 1.0)
    "回升": 1.0, "反弹": 1.0, "企稳": 1.0, "走强": 1.0,
    "活跃": 1.0, "增持": 1.0, "回购": 1.0, "景气": 1.0,
    "转暖": 1.0, "复苏": 1.0, "增长超预期": 1.5,

    # 轻度利好 (权重 0.5)
    "稳定": 0.5, "向好": 0.5, "乐观": 0.5, "信心": 0.5,
}

# 行业/题材风险词 (用于行业层面的情绪判断)
SECTOR_RISK_KEYWORDS: dict[str, list[str]] = {
    "房地产": ["房企违约", "楼市暴跌", "烂尾楼", "停贷", "房地产危机"],
    "科技": ["芯片制裁", "技术封锁", "出口管制", "实体清单"],
    "金融": ["银行暴雷", "信贷收紧", "坏账率", "流动性危机"],
    "新能源": ["产能过剩", "补贴退坡", "锂价暴跌", "光伏倒闭"],
    "消费": ["消费降级", "需求萎缩", "零售数据不及预期"],
}


# ══════════════════════════════════════════════════════════════════════════════
#  数据结构
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SentimentResult:
    """情绪扫描结果"""
    date: str
    sentiment_score: float     # -1.0(极度恐慌) ~ +1.0(极度贪婪)
    fear_count: int            # 恐慌词命中数
    greed_count: int           # 贪婪词命中数
    total_news: int            # 分析的新闻总数
    position_adjustment: float # 仓位调整建议: 0.5 ~ 1.2
    level: str                 # EXTREME_FEAR / FEAR / NEUTRAL / GREED / EXTREME_GREED
    top_fear_words: list[str] = field(default_factory=list)
    top_greed_words: list[str] = field(default_factory=list)
    alert_text: str = ""
    source: str = "unknown"    # tushare / manual / fallback

    def to_dict(self) -> dict:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════════════
#  舆情扫描器
# ══════════════════════════════════════════════════════════════════════════════

class SentimentScanner:
    """
    市场舆情情绪扫描器。

    用法:
        scanner = SentimentScanner(tushare_token="xxx")
        result = scanner.scan(date="20260320")
        adjusted_position = max_position * result.position_adjustment
    """

    # 情绪等级阈值
    THRESHOLDS = {
        "EXTREME_FEAR": -0.5,
        "FEAR": -0.2,
        "NEUTRAL_LOW": -0.2,
        "NEUTRAL_HIGH": 0.2,
        "GREED": 0.2,
        "EXTREME_GREED": 0.5,
    }

    # 仓位调整映射 (反向思维: 极度恐慌时不一定要减仓, 但要收紧止损)
    POSITION_MAP = {
        "EXTREME_FEAR": 0.6,     # 极度恐慌: 降仓至60%
        "FEAR": 0.8,             # 恐慌: 降仓至80%
        "NEUTRAL": 1.0,          # 中性: 不调整
        "GREED": 1.0,            # 贪婪: 不加仓(防追高)
        "EXTREME_GREED": 0.85,   # 极度贪婪: 反而收紧(物极必反)
    }

    def __init__(self, tushare_token: str = ""):
        self.tushare_token = tushare_token

    def _fetch_tushare_news(self, date: str) -> list[str]:
        """通过tushare拉取新闻标题"""
        if not self.tushare_token:
            return []

        try:
            import tushare as ts
            pro = ts.pro_api(self.tushare_token)

            # tushare news 接口
            # 格式化日期
            dt = datetime.strptime(date, "%Y%m%d")
            start = dt.strftime("%Y-%m-%d 00:00:00")
            end = dt.strftime("%Y-%m-%d 23:59:59")

            df = pro.news(src="sina", start_date=start, end_date=end)
            if df is not None and not df.empty:
                return df["title"].tolist()[:200]  # 最多200条
        except Exception as e:
            print(f"[舆情] tushare新闻拉取失败: {e}")

        return []

    def _analyze_headlines(self, headlines: list[str]) -> dict:
        """对标题进行关键词情绪打分"""
        fear_score = 0.0
        greed_score = 0.0
        fear_hits: dict[str, int] = {}
        greed_hits: dict[str, int] = {}

        for title in headlines:
            for kw, weight in FEAR_KEYWORDS.items():
                if kw in title:
                    fear_score += weight
                    fear_hits[kw] = fear_hits.get(kw, 0) + 1

            for kw, weight in GREED_KEYWORDS.items():
                if kw in title:
                    greed_score += weight
                    greed_hits[kw] = greed_hits.get(kw, 0) + 1

        return {
            "fear_score": fear_score,
            "greed_score": greed_score,
            "fear_hits": fear_hits,
            "greed_hits": greed_hits,
        }

    def scan(self, date: str = None, headlines: list[str] = None) -> SentimentResult:
        """
        执行舆情扫描。

        Args:
            date: 扫描日期 YYYYMMDD (默认今天)
            headlines: 手动传入标题列表 (跳过tushare拉取)

        Returns:
            SentimentResult
        """
        if date is None:
            date = datetime.now().strftime("%Y%m%d")

        source = "manual"

        # 获取新闻标题
        if headlines is None:
            headlines = self._fetch_tushare_news(date)
            source = "tushare" if headlines else "fallback"

        if not headlines:
            # 无数据, 返回中性
            return SentimentResult(
                date=date,
                sentiment_score=0.0,
                fear_count=0,
                greed_count=0,
                total_news=0,
                position_adjustment=1.0,
                level="NEUTRAL",
                alert_text="[舆情] 无新闻数据, 使用中性假设",
                source="fallback",
            )

        # 分析
        analysis = self._analyze_headlines(headlines)
        total = analysis["fear_score"] + analysis["greed_score"]

        if total > 0:
            # 归一化到 [-1, 1]
            raw_score = (analysis["greed_score"] - analysis["fear_score"]) / total
        else:
            raw_score = 0.0

        # 用sigmoid-like函数压缩到合理范围, 避免极端值
        import math
        sentiment = max(-1.0, min(1.0, raw_score * 1.5))

        # 确定等级
        if sentiment <= -0.5:
            level = "EXTREME_FEAR"
        elif sentiment <= -0.2:
            level = "FEAR"
        elif sentiment >= 0.5:
            level = "EXTREME_GREED"
        elif sentiment >= 0.2:
            level = "GREED"
        else:
            level = "NEUTRAL"

        pos_adj = self.POSITION_MAP.get(level, 1.0)

        # 排序关键词
        top_fear = sorted(analysis["fear_hits"].items(), key=lambda x: x[1], reverse=True)[:5]
        top_greed = sorted(analysis["greed_hits"].items(), key=lambda x: x[1], reverse=True)[:5]

        # 告警文本
        level_icons = {
            "EXTREME_FEAR": "🔴🔴",
            "FEAR": "🔴",
            "NEUTRAL": "⚪",
            "GREED": "🟢",
            "EXTREME_GREED": "🟢🟢",
        }
        alert_lines = [
            f"📰 舆情情绪: {level_icons.get(level, '⚪')} {level} (得分: {sentiment:+.2f})",
            f"   分析 {len(headlines)} 条新闻 | 恐慌词 {len(analysis['fear_hits'])} 种 | 乐观词 {len(analysis['greed_hits'])} 种",
        ]
        if top_fear:
            fear_str = ", ".join(f"{k}({v})" for k, v in top_fear[:3])
            alert_lines.append(f"   恐慌热词: {fear_str}")
        if top_greed:
            greed_str = ", ".join(f"{k}({v})" for k, v in top_greed[:3])
            alert_lines.append(f"   乐观热词: {greed_str}")
        if pos_adj < 1.0:
            alert_lines.append(f"   ⚡ 建议仓位调整至 {pos_adj:.0%}")

        return SentimentResult(
            date=date,
            sentiment_score=round(sentiment, 4),
            fear_count=len(analysis["fear_hits"]),
            greed_count=len(analysis["greed_hits"]),
            total_news=len(headlines),
            position_adjustment=pos_adj,
            level=level,
            top_fear_words=[f"{k}({v})" for k, v in top_fear],
            top_greed_words=[f"{k}({v})" for k, v in top_greed],
            alert_text="\n".join(alert_lines),
            source=source,
        )

    def format_opus_context(self, result: SentimentResult) -> str:
        """为Opus反思生成舆情上下文"""
        lines = [
            f"市场舆情情绪 ({result.date}):",
            f"  情绪等级: {result.level} ({result.sentiment_score:+.2f})",
            f"  新闻数量: {result.total_news}",
            f"  恐慌词种类: {result.fear_count} | 乐观词种类: {result.greed_count}",
        ]
        if result.top_fear_words:
            lines.append(f"  恐慌热词: {', '.join(result.top_fear_words[:5])}")
        if result.top_greed_words:
            lines.append(f"  乐观热词: {', '.join(result.top_greed_words[:5])}")
        lines.append(f"  仓位调整建议: {result.position_adjustment:.0%}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="舆情情绪扫描器")
    parser.add_argument("--token", type=str, help="Tushare API token")
    parser.add_argument("--date", type=str, help="扫描日期 YYYYMMDD")
    parser.add_argument("--test", action="store_true", help="用模拟标题测试")
    args = parser.parse_args()

    scanner = SentimentScanner(tushare_token=args.token or "")

    if args.test:
        # 模拟恐慌场景
        test_headlines = [
            "A股三大指数集体大跌 沪指跌破3000点",
            "千股跌停再现 投资者恐慌情绪蔓延",
            "美联储鹰派表态 加息预期升温",
            "中美贸易摩擦再度升级 加征关税",
            "北向资金大幅净卖出 外资加速撤离",
            "多家房企违约 房地产危机蔓延",
            "创业板指暴跌4% 科技股全线走弱",
            "央行紧急表态维稳 市场信心有待恢复",
            "社融数据不及预期 经济放缓信号明显",
            "机构紧急减持 基金赎回压力增大",
        ]
        result = scanner.scan(headlines=test_headlines, date=args.date)
        print(result.alert_text)
        print(f"\n情绪得分: {result.sentiment_score}")
        print(f"仓位调整: {result.position_adjustment}")
    elif args.date:
        result = scanner.scan(date=args.date)
        print(result.alert_text)
    else:
        print("用法: python sentiment_scanner.py --test")
        print("      python sentiment_scanner.py --token YOUR_TOKEN --date 20260320")
