"""
Brain — Agent 认知状态管理（生产版）
====================================
适配你的系统架构：Router → Skill → EventLog

集成方式：
  1. 在 router.py 中初始化 Brain
  2. Router._route_with_llm() 调用前，注入 brain.get_context_for_prompt()
  3. Skill 执行完毕后，可选调用 brain.remember() 存储洞察
  4. 盘后复盘时，调用 brain.commit_decision() 记录决策

目录结构：
  data/brain/
  ├── memory.jsonl          长期记忆库
  ├── emotion.json          当前市场情绪
  ├── commits.jsonl         决策提交历史
  └── journal/              每日复盘日志
      └── 2025-02-23.md

依赖：event_log.py（可选）
配置：config.py 中 USE_BRAIN = True, BRAIN_DIR = "data/brain"
"""

import json
import time
import os
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional, Any
from pathlib import Path
from collections import defaultdict

# 可选依赖
try:
    from event_log import EventLog, EventType
    HAS_EVENT_LOG = True
except ImportError:
    HAS_EVENT_LOG = False


# ═══════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════

# 尝试从 config 读取，读不到用默认值
try:
    from config import USE_BRAIN, BRAIN_DIR
except ImportError:
    USE_BRAIN = True
    BRAIN_DIR = "data/brain"


# ═══════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════

@dataclass
class Memory:
    """一条长期记忆"""
    key: str
    category: str       # market_pattern / trade_lesson / sector_insight / strategy_note
    content: str
    confidence: float
    tags: list[str]
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    recall_count: int = 0
    source: str = ""
    expiry: Optional[float] = None

    def is_expired(self) -> bool:
        return self.expiry is not None and time.time() > self.expiry

    def relevance_score(self, query_tags: list[str]) -> float:
        if not query_tags:
            return self.confidence
        tag_overlap = len(set(self.tags) & set(query_tags))
        tag_score = tag_overlap / max(len(query_tags), 1)
        recency = max(0, 1 - (time.time() - self.updated_at) / (90 * 86400))
        recall_bonus = min(0.2, self.recall_count * 0.02)
        return tag_score * 0.4 + self.confidence * 0.3 + recency * 0.2 + recall_bonus * 0.1


@dataclass
class MarketEmotion:
    """市场情绪状态"""
    greed_fear_index: float         # 0(极度恐惧) - 1(极度贪婪)
    trend_bias: str                 # bearish / cautious / neutral / cautious_bullish / bullish
    volatility_feel: str            # calm / normal / nervous / panic
    sector_heat: dict               # {"AI算力": 0.8, ...}
    market_phase: str               # accumulation / markup / distribution / markdown
    confidence: float
    reasoning: str
    updated_at: float = field(default_factory=time.time)

    def to_prompt_context(self) -> str:
        heat_str = ", ".join(
            f"{k}:{v:.0%}" for k, v in
            sorted(self.sector_heat.items(), key=lambda x: -x[1])[:5]
        )
        return (
            f"[市场情绪] (置信度{self.confidence:.0%})\n"
            f"  趋势: {self.trend_bias} | 贪婪恐惧: {self.greed_fear_index:.2f} | "
            f"波动: {self.volatility_feel} | 阶段: {self.market_phase}\n"
            f"  板块热度: {heat_str}\n"
            f"  依据: {self.reasoning}"
        )


@dataclass
class DecisionCommit:
    """交易决策提交"""
    commit_id: str
    action: str                     # BUY / SELL / HOLD / WATCH
    symbol: str
    reasoning_chain: list[str]
    signals_used: list[str]         # ["canslim", "sector_rotation", ...]
    confidence: float
    risk_assessment: dict
    market_context: dict
    timestamp: float = field(default_factory=time.time)
    outcome: Optional[dict] = None

    @staticmethod
    def generate_id(symbol: str, action: str, ts: float) -> str:
        return hashlib.sha256(f"{symbol}_{action}_{ts}".encode()).hexdigest()[:12]


# ═══════════════════════════════════════════
# Brain 主类
# ═══════════════════════════════════════════

class AgentBrain:
    """
    Agent 认知状态管理

    与你系统的集成点：
    1. Router 调用 LLM 前 → brain.get_context_for_prompt(question)
    2. Skill 产出洞察后 → brain.remember(...)
    3. 做出交易决策时 → brain.commit_decision(...)
    4. 盘后复盘时 → brain.update_emotion(...) + brain.write_journal(...)
    """

    def __init__(self, brain_dir: str = None, event_log: Optional[Any] = None):
        self.brain_dir = Path(brain_dir or BRAIN_DIR)
        self.brain_dir.mkdir(parents=True, exist_ok=True)
        (self.brain_dir / "journal").mkdir(exist_ok=True)

        self.memory_file = self.brain_dir / "memory.jsonl"
        self.emotion_file = self.brain_dir / "emotion.json"
        self.commits_file = self.brain_dir / "commits.jsonl"

        self.event_log = event_log
        self._memories: dict[str, Memory] = self._load_memories()
        self._emotion: Optional[MarketEmotion] = self._load_emotion()

    # ─────────────────────────────────────────
    # 长期记忆
    # ─────────────────────────────────────────

    def remember(self, key: str, category: str, content: str,
                 confidence: float = 0.7, tags: list[str] = None,
                 source: str = "", expiry_days: Optional[int] = None) -> Memory:
        """
        存储一条市场洞察

        category:
          "market_pattern"  : 市场规律（如"两连板后大概率分歧"）
          "trade_lesson"    : 交易教训（如"追高板块末端亏损"）
          "sector_insight"  : 板块洞察（如"AI板块轮动路径"）
          "strategy_note"   : 策略笔记（如"CANSLIM在弱市中胜率下降"）

        用法：
            brain.remember(
                "lesson_chase_ai_0120", "trade_lesson",
                "AI板块连涨5日后追高，次日跌停。教训：板块末期不追高。",
                confidence=0.9, tags=["AI", "追高", "教训"],
                source="post_market_review",
            )
        """
        expiry = time.time() + expiry_days * 86400 if expiry_days else None

        if key in self._memories:
            mem = self._memories[key]
            mem.content = content
            mem.confidence = confidence
            mem.tags = tags or mem.tags
            mem.updated_at = time.time()
            mem.source = source or mem.source
            mem.expiry = expiry
        else:
            mem = Memory(
                key=key, category=category, content=content,
                confidence=confidence, tags=tags or [],
                source=source, expiry=expiry,
            )
            self._memories[key] = mem

        self._save_memories()
        self._emit("brain.memory_added", {
            "key": key, "category": category,
            "content_preview": content[:200], "tags": tags or [],
        })
        return mem

    def recall(self, query_tags: list[str] = None, category: str = None,
               top_k: int = 5, min_confidence: float = 0.3) -> list[Memory]:
        """召回相关记忆"""
        candidates = []
        for mem in self._memories.values():
            if mem.is_expired() or mem.confidence < min_confidence:
                continue
            if category and mem.category != category:
                continue
            score = mem.relevance_score(query_tags or [])
            candidates.append((score, mem))

        candidates.sort(key=lambda x: -x[0])
        results = [mem for _, mem in candidates[:top_k]]
        for mem in results:
            mem.recall_count += 1
        if results:
            self._save_memories()
        return results

    def forget(self, key: str) -> bool:
        if key in self._memories:
            del self._memories[key]
            self._save_memories()
            return True
        return False

    def get_all_memories(self, category: str = None) -> list[Memory]:
        mems = list(self._memories.values())
        if category:
            mems = [m for m in mems if m.category == category]
        return sorted(mems, key=lambda m: -m.updated_at)

    @property
    def memory_count(self) -> int:
        return len(self._memories)

    # ─────────────────────────────────────────
    # 市场情绪
    # ─────────────────────────────────────────

    def update_emotion(self, greed_fear_index: float, trend_bias: str,
                       volatility_feel: str, sector_heat: dict,
                       market_phase: str, confidence: float,
                       reasoning: str) -> MarketEmotion:
        """
        更新市场情绪判断（通常在盘前/盘后调用）

        用法：
            brain.update_emotion(
                greed_fear_index=0.65, trend_bias="cautious_bullish",
                volatility_feel="normal",
                sector_heat={"AI算力": 0.85, "消费电子": 0.6},
                market_phase="markup", confidence=0.7,
                reasoning="成交额放大，北向流入，但高位板块分歧"
            )
        """
        self._emotion = MarketEmotion(
            greed_fear_index=greed_fear_index, trend_bias=trend_bias,
            volatility_feel=volatility_feel, sector_heat=sector_heat,
            market_phase=market_phase, confidence=confidence,
            reasoning=reasoning,
        )
        self._save_emotion()
        self._emit("brain.emotion_updated", {
            "greed_fear": greed_fear_index, "trend": trend_bias,
            "phase": market_phase, "confidence": confidence,
        })
        return self._emotion

    def get_emotion(self) -> Optional[MarketEmotion]:
        return self._emotion

    # ─────────────────────────────────────────
    # 决策提交
    # ─────────────────────────────────────────

    def commit_decision(self, action: str, symbol: str,
                        reasoning_chain: list[str],
                        signals_used: list[str],
                        confidence: float,
                        risk_assessment: dict = None,
                        market_context: dict = None) -> DecisionCommit:
        """
        提交交易决策（类 git commit）

        用法：
            brain.commit_decision(
                action="BUY", symbol="002415.SZ",
                reasoning_chain=[
                    "1. CANSLIM筛选通过: C=88, A=82",
                    "2. 板块轮动确认: 消费电子轮入",
                    "3. 技术突破: 突破前高, 量比2x",
                ],
                signals_used=["canslim", "sector_rotation", "technical"],
                confidence=0.78,
                risk_assessment={"stop_loss": "-5%", "target": "+15%"},
                market_context={"上证": 3250, "成交额": "1.2万亿"},
            )
        """
        ts = time.time()
        commit = DecisionCommit(
            commit_id=DecisionCommit.generate_id(symbol, action, ts),
            action=action, symbol=symbol,
            reasoning_chain=reasoning_chain, signals_used=signals_used,
            confidence=confidence,
            risk_assessment=risk_assessment or {},
            market_context=market_context or {},
            timestamp=ts,
        )
        with open(self.commits_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(commit), ensure_ascii=False) + "\n")

        self._emit("brain.decision_committed", {
            "commit_id": commit.commit_id, "action": action,
            "symbol": symbol, "confidence": confidence,
            "signals": signals_used,
        })
        return commit

    def update_outcome(self, commit_id: str, outcome: dict):
        """事后标注决策结果（用于复盘和胜率统计）"""
        commits = self._load_commits()
        for c in commits:
            if c.get("commit_id") == commit_id:
                c["outcome"] = outcome
                break
        with open(self.commits_file, "w", encoding="utf-8") as f:
            for c in commits:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

    def get_commits(self, symbol: str = None, action: str = None,
                    days_back: int = 30, limit: int = 20) -> list[dict]:
        commits = self._load_commits()
        cutoff = time.time() - days_back * 86400
        results = []
        for c in reversed(commits):
            if c.get("timestamp", 0) < cutoff:
                break
            if symbol and c.get("symbol") != symbol:
                continue
            if action and c.get("action") != action:
                continue
            results.append(c)
            if len(results) >= limit:
                break
        return results

    def get_win_rate(self, days_back: int = 90) -> dict:
        """统计历史胜率"""
        commits = self.get_commits(days_back=days_back, limit=1000)
        total = wins = 0
        by_signal = defaultdict(lambda: {"total": 0, "wins": 0})
        for c in commits:
            outcome = c.get("outcome")
            if not outcome:
                continue
            total += 1
            pnl = outcome.get("pnl", "0%")
            is_win = not pnl.startswith("-") and pnl != "0%"
            if is_win:
                wins += 1
            for sig in c.get("signals_used", []):
                by_signal[sig]["total"] += 1
                if is_win:
                    by_signal[sig]["wins"] += 1
        return {
            "overall": {"total": total, "wins": wins,
                        "rate": wins / max(total, 1)},
            "by_signal": {
                k: {**v, "rate": v["wins"] / max(v["total"], 1)}
                for k, v in by_signal.items()
            },
        }

    # ─────────────────────────────────────────
    # 每日复盘日志
    # ─────────────────────────────────────────

    def write_journal(self, content: str, date: str = None):
        date = date or datetime.now().strftime("%Y-%m-%d")
        path = self.brain_dir / "journal" / f"{date}.md"
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# 交易日复盘 - {date}\n\n{content}")

    def read_journal(self, date: str = None) -> Optional[str]:
        date = date or datetime.now().strftime("%Y-%m-%d")
        path = self.brain_dir / "journal" / f"{date}.md"
        return path.read_text(encoding="utf-8") if path.exists() else None

    # ─────────────────────────────────────────
    # ★ 核心输出：注入 Router LLM 的上下文
    # ─────────────────────────────────────────

    def get_context_for_prompt(self, question: str = "",
                               include_emotion: bool = True,
                               include_memories: bool = True,
                               include_recent_commits: bool = True,
                               memory_top_k: int = 3,
                               commit_limit: int = 3) -> str:
        """
        生成认知上下文，注入 Router 发给 LLM 的 prompt。

        与前一版的区别：直接接受 question 字符串（而非 tags 列表），
        内部自动提取关键词匹配记忆，和你的 Router 调用方式一致。

        用法（在 router.py 的 _route_with_llm 中）：
            brain_context = brain.get_context_for_prompt(question)
            # 拼接到 system/user prompt 中
        """
        # 从 question 自动提取中文关键词作为 tags
        tags = self._extract_tags(question) if question else []

        sections = []

        if include_emotion and self._emotion:
            sections.append(self._emotion.to_prompt_context())

        if include_memories:
            memories = self.recall(tags, top_k=memory_top_k)
            if memories:
                lines = ["[历史经验]"]
                for i, m in enumerate(memories, 1):
                    age = (time.time() - m.created_at) / 86400
                    lines.append(
                        f"  {i}. [{m.category}] (置信度:{m.confidence:.0%}, "
                        f"{age:.0f}天前) {m.content}"
                    )
                sections.append("\n".join(lines))

        if include_recent_commits:
            commits = self.get_commits(days_back=7, limit=commit_limit)
            if commits:
                lines = ["[近期决策]"]
                for c in commits:
                    outcome = c.get("outcome") or {}
                    pnl = outcome.get("pnl", "待定")
                    lines.append(
                        f"  - {c['action']} {c['symbol']} "
                        f"(信号:{','.join(c.get('signals_used', []))}, "
                        f"置信:{c.get('confidence', 0):.0%}, 结果:{pnl})"
                    )
                sections.append("\n".join(lines))

        if not sections:
            return ""
        return "\n\n".join(["[Agent 认知状态]"] + sections)

    # ─────────────────────────────────────────
    # 从 Skill 输出中自动提取并存储记忆
    # ─────────────────────────────────────────

    def learn_from_output(self, skill_name: str, question: str,
                          output: str, auto_key: bool = True) -> Optional[Memory]:
        """
        从 Skill 执行结果中提取并存储洞察

        用法（在 Router._execute_skills 完成后可选调用）：
            brain.learn_from_output("sector_rotation", question, result_text)

        自动生成 key，避免手动命名的麻烦。
        只有当 output 包含有价值的洞察时才存储（长度 > 50 字符）。
        """
        if not output or len(output.strip()) < 50:
            return None

        tags = self._extract_tags(question) + self._extract_tags(output)
        tags = list(set(tags))[:10]  # 去重，限制数量

        # 自动生成 key
        date_str = datetime.now().strftime("%Y%m%d")
        key = f"auto_{skill_name}_{date_str}_{hash(question) % 10000:04d}" if auto_key else None
        if not key:
            return None

        # 截取 output 前 500 字作为记忆内容
        content = output.strip()[:500]
        if len(output.strip()) > 500:
            content += "..."

        category_map = {
            "sector_rotation": "sector_insight",
            "sentiment": "market_pattern",
            "market_sentiment": "market_pattern",
            "canslim": "strategy_note",
            "stock_analysis": "strategy_note",
            "risk_check": "trade_lesson",
        }
        category = category_map.get(skill_name, "strategy_note")

        return self.remember(
            key=key, category=category, content=content,
            confidence=0.6, tags=tags, source=skill_name,
            expiry_days=30,  # 自动记忆 30 天过期
        )

    # ─────────────────────────────────────────
    # 内部方法
    # ─────────────────────────────────────────

    @staticmethod
    def _extract_tags(text: str) -> list[str]:
        """从文本中提取中文关键词（简易版，无需分词库）"""
        import re
        # 提取连续中文片段（2-4 字）
        chunks = re.findall(r'[\u4e00-\u9fff]{2,4}', text)
        # 常见 A 股领域关键词过滤
        domain_keywords = {
            "板块", "轮动", "情绪", "龙头", "主线", "题材", "概念",
            "追高", "止损", "止盈", "仓位", "回撤", "风控",
            "涨停", "跌停", "连板", "放量", "缩量", "突破",
            "北向", "资金", "成交", "消息", "利好", "利空",
            "算力", "芯片", "半导", "新能", "消费", "医药",
            "银行", "地产", "军工", "白酒",
        }
        # 保留：在 domain_keywords 中的 + 高频出现的
        tags = []
        seen = set()
        for chunk in chunks:
            if chunk in seen:
                continue
            seen.add(chunk)
            # 完全匹配或部分匹配 domain keywords
            if chunk in domain_keywords or any(kw in chunk for kw in domain_keywords):
                tags.append(chunk)
        return tags[:8]

    def _emit(self, event_type: str, payload: dict):
        """向 EventLog 发射事件（如果有的话）"""
        if self.event_log and HAS_EVENT_LOG:
            try:
                self.event_log.emit(event_type, payload, source="brain")
            except Exception:
                pass

    # ─────────────────────────────────────────
    # 持久化
    # ─────────────────────────────────────────

    def _load_memories(self) -> dict[str, Memory]:
        memories = {}
        if self.memory_file.exists():
            for line in self.memory_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    mem = Memory(**data)
                    if not mem.is_expired():
                        memories[mem.key] = mem
                except (json.JSONDecodeError, TypeError):
                    continue
        return memories

    def _save_memories(self):
        with open(self.memory_file, "w", encoding="utf-8") as f:
            for mem in self._memories.values():
                f.write(json.dumps(asdict(mem), ensure_ascii=False) + "\n")

    def _load_emotion(self) -> Optional[MarketEmotion]:
        if self.emotion_file.exists():
            try:
                return MarketEmotion(**json.loads(
                    self.emotion_file.read_text(encoding="utf-8")
                ))
            except (json.JSONDecodeError, TypeError):
                return None
        return None

    def _save_emotion(self):
        if self._emotion:
            self.emotion_file.write_text(
                json.dumps(asdict(self._emotion), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _load_commits(self) -> list[dict]:
        commits = []
        if self.commits_file.exists():
            for line in self.commits_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    commits.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return commits


# ═══════════════════════════════════════════
# 全局实例（方便各模块直接 import）
# ═══════════════════════════════════════════

_brain_instance: Optional[AgentBrain] = None


def get_brain(event_log=None) -> AgentBrain:
    """获取全局 Brain 实例（懒加载单例）"""
    global _brain_instance
    if _brain_instance is None:
        _brain_instance = AgentBrain(event_log=event_log)
    return _brain_instance
