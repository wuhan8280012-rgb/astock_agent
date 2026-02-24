"""
AutoJournal — 盘后自动复盘日志
================================
将 CronEngine 的 post_market 任务输出自动存入 Brain 的复盘日志。

集成方式：
  方式 A（推荐）: 传给 CronEngine 的 on_result 回调
  方式 B: 在 post_market 任务执行后手动调用

自动做的事：
  1. AI 复盘输出 → brain.write_journal()
  2. 从输出中提取关键洞察 → brain.remember()
  3. 尝试提取情绪判断 → brain.update_emotion()（如果 AI 输出了结构化情绪）
  4. 写入 EventLog
"""

import re
import json
import time
from datetime import datetime
from typing import Optional, Any

try:
    from event_log import EventLog, EventType
    HAS_EVENT_LOG = True
except ImportError:
    HAS_EVENT_LOG = False


class AutoJournal:
    """
    盘后自动复盘日志生成器

    用法：
        journal = AutoJournal(brain=brain, event_log=event_log)

        # 方式 A: 作为 CronEngine 的 on_result
        cron = CronEngine(..., on_result=journal.on_cron_result)

        # 方式 B: 手动调用
        journal.process("post_market", ai_output_text)
    """

    def __init__(self, brain=None, event_log=None,
                 auto_remember: bool = True,
                 auto_emotion: bool = True):
        """
        Parameters
        ----------
        brain : AgentBrain 实例
        event_log : EventLog 实例
        auto_remember : 是否自动从复盘输出中提取洞察存入记忆
        auto_emotion : 是否尝试从输出中提取情绪判断并更新
        """
        self.brain = brain
        self.event_log = event_log
        self.auto_remember = auto_remember
        self.auto_emotion = auto_emotion

        # 需要自动存日志的任务 ID
        self.journal_jobs = {"post_market", "weekly_review"}

        # 需要自动存记忆的任务 ID
        self.memory_jobs = {
            "post_market", "weekly_review",
            "pre_market", "morning_scan", "closing_scan",
        }

    def on_cron_result(self, job_id: str, result: str):
        """
        CronEngine 的 on_result 回调

        接入方式：
            cron = CronEngine(..., on_result=journal.on_cron_result)
        """
        if not result or not self.brain:
            return

        self.process(job_id, result)

    def process(self, job_id: str, ai_output: str):
        """
        处理 AI 任务输出

        1. 写复盘日志（post_market / weekly_review）
        2. 提取并存储洞察
        3. 尝试提取情绪更新
        """
        if not ai_output or not self.brain:
            return

        date_str = datetime.now().strftime("%Y-%m-%d")
        stored_items = []

        # ── 1. 写复盘日志 ──
        if job_id in self.journal_jobs:
            self._write_journal(job_id, ai_output, date_str)
            stored_items.append("journal")

        # ── 2. 提取洞察存入记忆 ──
        if self.auto_remember and job_id in self.memory_jobs:
            insights = self._extract_insights(job_id, ai_output)
            for insight in insights:
                self.brain.remember(**insight)
                stored_items.append(f"memory:{insight['key']}")

        # ── 3. 尝试提取情绪更新 ──
        if self.auto_emotion and job_id == "post_market":
            emotion = self._extract_emotion(ai_output)
            if emotion:
                self.brain.update_emotion(**emotion)
                stored_items.append("emotion")

        # ── 4. EventLog 记录 ──
        if self.event_log and HAS_EVENT_LOG:
            self.event_log.emit("journal.processed", {
                "job_id": job_id,
                "date": date_str,
                "output_length": len(ai_output),
                "stored": stored_items,
            }, source="auto_journal")

    # ─────────────────────────────────────────
    # 写日志
    # ─────────────────────────────────────────

    def _write_journal(self, job_id: str, ai_output: str, date_str: str):
        """写入每日/每周复盘日志"""
        if job_id == "weekly_review":
            # 周报用独立文件名
            week_num = datetime.now().isocalendar()[1]
            filename_date = f"{date_str}_W{week_num}"
        else:
            filename_date = date_str

        # 如果当天已有日志，追加而不是覆盖
        existing = self.brain.read_journal(filename_date)
        if existing:
            # 追加到已有日志
            timestamp = datetime.now().strftime("%H:%M")
            content = f"{existing}\n\n---\n\n## {self._job_title(job_id)} ({timestamp})\n\n{ai_output}"
        else:
            content = ai_output

        self.brain.write_journal(content, filename_date)

    @staticmethod
    def _job_title(job_id: str) -> str:
        titles = {
            "pre_market": "盘前扫描",
            "morning_scan": "早盘动量",
            "midday_review": "午间复盘",
            "closing_scan": "尾盘扫描",
            "post_market": "盘后复盘",
            "weekly_review": "周度复盘",
            "risk_check": "风控检查",
        }
        return titles.get(job_id, job_id)

    # ─────────────────────────────────────────
    # 提取洞察
    # ─────────────────────────────────────────

    def _extract_insights(self, job_id: str, ai_output: str) -> list[dict]:
        """
        从 AI 输出中提取值得记住的洞察

        提取规则：
        - 包含"教训"/"经验"/"规律"/"注意"/"关键"的段落 → trade_lesson / market_pattern
        - 包含板块名 + 判断性词汇的段落 → sector_insight
        - 包含策略相关词的段落 → strategy_note

        每个洞察自动生成 key，30 天过期。
        """
        insights = []
        date_str = datetime.now().strftime("%Y%m%d")

        # 按段落分割（双换行 或 markdown 标题）
        paragraphs = re.split(r'\n\n+|\n(?=#+\s)', ai_output)
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 30]

        for i, para in enumerate(paragraphs):
            category = None
            tags = []

            # 判断段落类型
            if any(kw in para for kw in ["教训", "经验", "错误", "失误", "亏损", "应该"]):
                category = "trade_lesson"
                tags.append("教训")
            elif any(kw in para for kw in ["规律", "趋势", "历史上", "通常", "往往"]):
                category = "market_pattern"
                tags.append("规律")
            elif any(kw in para for kw in ["板块", "轮动", "主线", "龙头", "切换"]):
                category = "sector_insight"
                tags.append("板块")
            elif any(kw in para for kw in ["策略", "参数", "调整", "优化", "胜率"]):
                category = "strategy_note"
                tags.append("策略")

            if not category:
                continue  # 普通段落不存

            # 提取更多 tags
            sector_keywords = [
                "AI", "算力", "芯片", "半导体", "消费", "医药", "银行",
                "地产", "军工", "新能源", "白酒", "汽车",
            ]
            for sk in sector_keywords:
                if sk in para:
                    tags.append(sk)

            # 截取前 400 字
            content = para[:400]
            if len(para) > 400:
                content += "..."

            key = f"auto_{job_id}_{date_str}_{i:02d}"

            insights.append({
                "key": key,
                "category": category,
                "content": content,
                "confidence": 0.65,
                "tags": list(set(tags))[:8],
                "source": f"auto_journal_{job_id}",
                "expiry_days": 30,
            })

        # 最多存 5 条（避免垃圾记忆过多）
        return insights[:5]

    # ─────────────────────────────────────────
    # 提取情绪
    # ─────────────────────────────────────────

    def _extract_emotion(self, ai_output: str) -> Optional[dict]:
        """
        尝试从 AI 复盘输出中提取情绪判断

        如果 AI 输出中包含结构化的情绪描述，自动解析并更新。
        如果无法解析，返回 None（不更新，保留上次的判断）。

        支持两种格式：
        1. AI 按约定格式输出（如果在 prompt 中要求了）
        2. 从自然语言中模糊提取
        """
        text = ai_output.lower()

        # 尝试提取贪婪恐惧指数
        greed_fear = self._extract_number(ai_output, [
            r"贪婪恐惧[指数]*[：:\s]*([0-9.]+)",
            r"恐慌贪婪[：:\s]*([0-9.]+)",
            r"greed.?fear[：:\s]*([0-9.]+)",
        ])

        if greed_fear is None:
            # 从关键词推断
            if any(kw in text for kw in ["极度恐慌", "恐慌", "暴跌"]):
                greed_fear = 0.2
            elif any(kw in text for kw in ["偏悲观", "谨慎", "弱势"]):
                greed_fear = 0.35
            elif any(kw in text for kw in ["中性", "震荡", "观望"]):
                greed_fear = 0.5
            elif any(kw in text for kw in ["偏乐观", "回暖", "温和"]):
                greed_fear = 0.6
            elif any(kw in text for kw in ["乐观", "强势", "放量上涨"]):
                greed_fear = 0.75
            elif any(kw in text for kw in ["极度贪婪", "疯狂", "全面暴涨"]):
                greed_fear = 0.9
            else:
                return None  # 无法判断，不更新

        # 推断趋势偏向
        if greed_fear < 0.3:
            trend = "bearish"
        elif greed_fear < 0.45:
            trend = "cautious"
        elif greed_fear < 0.55:
            trend = "neutral"
        elif greed_fear < 0.7:
            trend = "cautious_bullish"
        else:
            trend = "bullish"

        # 推断波动感知
        if any(kw in text for kw in ["恐慌", "暴跌", "急跌", "大幅波动"]):
            vol = "panic"
        elif any(kw in text for kw in ["震荡", "波动", "不稳"]):
            vol = "nervous"
        elif any(kw in text for kw in ["平稳", "窄幅", "缩量"]):
            vol = "calm"
        else:
            vol = "normal"

        # 推断市场阶段
        if any(kw in text for kw in ["底部", "筑底", "吸筹"]):
            phase = "accumulation"
        elif any(kw in text for kw in ["上涨", "拉升", "突破", "放量涨"]):
            phase = "markup"
        elif any(kw in text for kw in ["高位", "滞涨", "分歧", "缩量涨"]):
            phase = "distribution"
        elif any(kw in text for kw in ["下跌", "破位", "杀跌"]):
            phase = "markdown"
        else:
            phase = "markup"  # 默认

        # 提取板块热度（从"板块"相关段落中）
        sector_heat = self._extract_sector_heat(ai_output)

        # 提取推理依据（取最相关的一句）
        reasoning = self._extract_reasoning(ai_output)

        return {
            "greed_fear_index": greed_fear,
            "trend_bias": trend,
            "volatility_feel": vol,
            "sector_heat": sector_heat,
            "market_phase": phase,
            "confidence": 0.6,  # 自动提取的置信度稍低
            "reasoning": reasoning,
        }

    def _extract_sector_heat(self, text: str) -> dict:
        """从文本中提取板块热度"""
        sectors = {
            "AI算力": ["AI算力", "算力", "AI芯片"],
            "AI应用": ["AI应用", "人工智能应用"],
            "消费电子": ["消费电子", "消费电"],
            "半导体": ["半导体", "芯片"],
            "新能源": ["新能源", "光伏", "锂电"],
            "医药": ["医药", "医疗", "生物医药"],
            "银行": ["银行", "金融"],
            "消费": ["消费", "白酒", "食品饮料"],
            "军工": ["军工", "国防"],
            "地产": ["地产", "房地产"],
        }

        heat = {}
        text_lower = text.lower()
        for sector, keywords in sectors.items():
            count = sum(text_lower.count(kw.lower()) for kw in keywords)
            if count > 0:
                # 简单映射：出现次数 → 热度 0-1
                heat[sector] = min(1.0, count * 0.15)

        # 归一化：最高的设为 0.9
        if heat:
            max_heat = max(heat.values())
            if max_heat > 0:
                factor = 0.9 / max_heat
                heat = {k: round(min(1.0, v * factor), 2) for k, v in heat.items()}

        return heat

    @staticmethod
    def _extract_reasoning(text: str) -> str:
        """提取情绪判断的核心依据（最相关的一句话）"""
        # 找包含判断性词汇的句子
        sentences = re.split(r'[。！\n]', text)
        judgment_kws = ["因此", "总体", "综合", "整体来看", "市场", "情绪",
                        "判断", "认为", "预计", "建议", "结论"]
        for s in reversed(sentences):  # 从后往前找（结论通常在后面）
            s = s.strip()
            if len(s) > 15 and any(kw in s for kw in judgment_kws):
                return s[:150]
        # 找不到就用最后一个有意义的句子
        for s in reversed(sentences):
            s = s.strip()
            if len(s) > 15:
                return s[:150]
        return "自动提取"

    @staticmethod
    def _extract_number(text: str, patterns: list[str]) -> Optional[float]:
        """用正则提取数字"""
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    val = float(match.group(1))
                    if 0 <= val <= 1:
                        return val
                    elif 1 < val <= 100:
                        return val / 100
                except ValueError:
                    continue
        return None
