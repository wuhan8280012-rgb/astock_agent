"""
brain.py — AgentBrain 认知状态（Context Engineering 重构）
==========================================================
设计原则（来自 Agent Skills for Context Engineering）：

1. Structured Compaction — 记忆不是自由文本，而是固定4段结构
   ## 当前意图: 验证 v6.1 盘中评分
   ## 状态清单: market_environment.py(v6.1), realtime_fetcher.py(待验证)
   ## 决策记录: 趋势权重0.30→0.25(MA滞后)
   ## 下一步: 明天盘中验证数据源连通

2. Tokens-per-task — 记忆检索不塞全部历史，只返回与当前任务相关的最小集
3. Observation Masking — learn_from_output 时提取结论、丢弃原始数据
4. Progressive Disclosure — get_context_for_prompt 分层返回
   Level 1: 结构化状态块（~200 tokens，总是返回）
   Level 2: 相关记忆（~500 tokens，关键词匹配）
   Level 3: 事件摘要（~300 tokens，从 EventLog 获取）

用法：
  brain = AgentBrain()
  brain.set_intent("验证 v6.1 盘中评分准确性")
  brain.update_status("market_environment.py", "v6.1已部署")
  brain.record_decision("趋势权重 0.30→0.25", "MA滞后导致节后首日误判")
  brain.set_next_step("明天盘中验证 AKShare 数据源连通")

  # 给 LLM 的上下文（自动分层，最少 token）
  ctx = brain.get_context_for_prompt("盘前分析 环境评分")

  # 从工具输出中学习（observation masking）
  brain.learn_from_output("trade_signals", "信号扫描", raw_output)
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Any


class AgentBrain:
    """
    认知状态管理器（Structured Compaction 模式）

    核心状态（始终保持在 ~200 tokens 以内）：
      intent     — 当前意图（1 句话）
      status     — 状态清单（key: value pairs）
      decisions  — 最近 N 条决策记录
      next_steps — 下一步行动

    长期记忆（JSONL，按需检索）：
      memory.jsonl    — 学到的知识/洞察
      commits.jsonl   — 交易决策提交记录
      journal/        — 每日复盘日志
    """

    MAX_DECISIONS = 10      # 结构化状态中保留最近 N 条决策
    MAX_STATUS_ITEMS = 15   # 状态清单最多 N 项
    MAX_MEMORIES_IN_CTX = 3 # 上下文中最多返回 N 条相关记忆

    def __init__(self, base_dir: str = "data/brain", event_log=None):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        (self.base_dir / "journal").mkdir(exist_ok=True)

        self.event_log = event_log

        # ── 结构化状态（内存中，定期持久化）──
        self._state = {
            "intent": "",
            "status": {},       # key → value
            "decisions": [],    # [{when, what, why}, ...]
            "next_steps": [],   # [str, ...]
        }

        # ── JSONL 文件 ──
        self._memory_file = self.base_dir / "memory.jsonl"
        self._commits_file = self.base_dir / "commits.jsonl"
        self._state_file = self.base_dir / "state.json"
        self._emotion_file = self.base_dir / "emotion.json"

        # 确保 JSONL 文件有 schema 首行
        self._ensure_jsonl_schema(self._memory_file, {
            "_schema": True,
            "fields": {"ts": "timestamp", "source": "module", "insight": "learned insight", "keywords": "list of keywords"},
        })
        self._ensure_jsonl_schema(self._commits_file, {
            "_schema": True,
            "fields": {"ts": "timestamp", "symbol": "stock", "action": "buy/sell/skip", "reason": "why", "env_score": "int"},
        })

        # 加载上次状态
        self._load_state()

    # ═══════════════════════════════════════════════════════════
    #  结构化状态操作（Structured Compaction 的 4 段）
    # ═══════════════════════════════════════════════════════════

    def set_intent(self, intent: str):
        """设置当前意图（1 句话描述当前在做什么）"""
        self._state["intent"] = intent
        self._save_state()
        self._emit("brain.intent", {"intent": intent})

    def update_status(self, key: str, value: str):
        """更新状态清单中的某一项"""
        self._state["status"][key] = value
        # 保持 status 不超过上限
        if len(self._state["status"]) > self.MAX_STATUS_ITEMS:
            oldest_key = next(iter(self._state["status"]))
            del self._state["status"][oldest_key]
        self._save_state()

    def remove_status(self, key: str):
        """移除状态清单中的某一项"""
        self._state["status"].pop(key, None)
        self._save_state()

    def record_decision(self, what: str, why: str = ""):
        """记录一条决策"""
        entry = {
            "when": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "what": what,
            "why": why,
        }
        self._state["decisions"].append(entry)
        # 只保留最近 N 条
        self._state["decisions"] = self._state["decisions"][-self.MAX_DECISIONS:]
        self._save_state()
        self._emit("brain.decision", entry)

    def set_next_step(self, *steps: str):
        """设置下一步行动（覆盖之前的）"""
        self._state["next_steps"] = list(steps)
        self._save_state()

    def add_next_step(self, step: str):
        """追加一个下一步"""
        self._state["next_steps"].append(step)
        self._save_state()

    # ═══════════════════════════════════════════════════════════
    #  上下文生成（Progressive Disclosure）
    # ═══════════════════════════════════════════════════════════

    def get_context_for_prompt(self, keywords: str = "") -> str:
        """
        为 LLM prompt 生成上下文（3 层渐进披露）

        Layer 1（总是返回，~200 tokens）: 结构化状态块
        Layer 2（关键词匹配，~500 tokens）: 相关记忆
        Layer 3（有 EventLog 时，~300 tokens）: 最近事件摘要

        Parameters
        ----------
        keywords : 空格分隔的关键词，用于记忆检索

        Returns
        -------
        str: 格式化的上下文文本
        """
        sections = []

        # ── Layer 1: 结构化状态块（总是返回）──
        sections.append(self._render_state_block())

        # ── Layer 2: 相关记忆（关键词匹配）──
        if keywords:
            memories = self._search_memories(keywords)
            if memories:
                mem_lines = [f"  - {m['insight'][:80]}" for m in memories[:2]]
                sections.append("## 相关记忆\n" + "\n".join(mem_lines))

        # ── Layer 3: 事件摘要（最近 10 条）──
        if self.event_log:
            try:
                event_ctx = self.event_log.get_context_block(last_n=5)
                if event_ctx and "(无事件)" not in event_ctx:
                    sections.append(event_ctx)
            except Exception:
                pass

        return "\n\n".join(sections)

    def _render_state_block(self) -> str:
        """渲染结构化状态块"""
        s = self._state
        lines = ["## 认知状态"]

        # 意图
        intent = s["intent"] or "(未设置)"
        lines.append(f"当前意图: {intent}")

        # 状态清单（固定结构 + 压缩显示，控制上下文 token）
        status_items = list(s["status"].items())
        lines.append("状态清单:")
        if status_items:
            shown = status_items[-6:]
            for k, v in shown:
                lines.append(f"  {k}: {v}")
            hidden = len(status_items) - len(shown)
            if hidden > 0:
                lines.append(f"  ...(+{hidden}项)")
        else:
            lines.append("  (无)")

        # 最近决策（固定结构 + 仅最近3条）
        lines.append("最近决策:")
        decisions = s["decisions"][-3:]
        if decisions:
            for d in decisions:
                lines.append(f"  [{d['when']}] {d['what']}")
        else:
            lines.append("  (无)")

        # 下一步（固定结构）
        lines.append("下一步: " + ("; ".join(s["next_steps"]) if s["next_steps"] else "(未设置)"))

        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════
    #  学习（Observation Masking）
    # ═══════════════════════════════════════════════════════════

    def learn_from_output(self, source: str, label: str, raw_output: str):
        """
        从工具输出中学习（Observation Masking）

        不存储原始输出（可能几千行），只提取关键洞察。
        原始数据在 EventLog 里有完整记录。

        Parameters
        ----------
        source : 来源模块名
        label : 操作标签
        raw_output : 原始输出文本
        """
        # 提取关键信息（Observation Masking）
        insight = self._extract_insight(source, label, raw_output)

        if insight:
            # 写入 JSONL 长期记忆
            keywords = self._extract_keywords(insight)
            entry = {
                "ts": datetime.now().isoformat(),
                "source": source,
                "insight": insight,
                "keywords": keywords,
            }
            self._append_jsonl(self._memory_file, entry)

            # 事件记录
            self._emit("brain.learned", {
                "source": source,
                "insight": insight[:100],
            })

    def _extract_insight(self, source: str, label: str, raw: str) -> str:
        """
        从原始输出提取一行洞察（Observation Masking 核心）

        替换规则：
          几千行扫描结果 → "扫描300股，命中3只阶梯突破信号"
          环境评分详情 → "环境74分(一般)，情绪81最高，趋势73待确认"
          复盘长文 → "今日+1.01%，成交万亿，北向75亿，观察明日确认"
        """
        raw_lower = raw.lower() if raw else ""
        raw_len = len(raw) if raw else 0

        # 信号扫描结果：提取命中数
        if source in ("trade_signals", "signal"):
            signal_count = raw.count("信号") + raw.count("突破") + raw.count("买入")
            symbols = re.findall(r'\d{6}\.[A-Z]{2}', raw)
            return f"[{label}] 命中{len(symbols)}只: {', '.join(symbols[:5])}" if symbols else f"[{label}] 无命中信号"

        # 环境评分
        if source in ("market_env", "market_environment"):
            score_match = re.search(r'(\d+)/100', raw)
            level_match = re.search(r'(良好|一般|较差)', raw)
            if score_match:
                return f"[环境] {score_match.group()} ({level_match.group() if level_match else '?'})"
            return f"[环境] {raw[:80]}"

        # 其他：截取前 100 字符
        summary = raw[:100].replace("\n", " ").strip()
        return f"[{label}] {summary}" if summary else ""

    def _extract_keywords(self, text: str) -> list:
        """从文本提取关键词用于后续检索"""
        # 股票代码
        codes = re.findall(r'\d{6}\.[A-Z]{2}', text)
        # 中文关键词
        cn_words = re.findall(r'[突破|信号|涨停|环境|趋势|情绪|成交量|板块|龙头|阶梯|CANSLIM|价值|北向|轮动]+', text)
        # 合并去重
        return list(set(codes + cn_words))[:10]

    # ═══════════════════════════════════════════════════════════
    #  记忆检索
    # ═══════════════════════════════════════════════════════════

    def _search_memories(self, keywords: str) -> list:
        """
        按关键词搜索长期记忆（简单匹配，够用）

        不用向量数据库——对 11 个 Skill 的日常记忆，
        关键词匹配的命中率已经足够。
        """
        kw_list = keywords.split()
        if not self._memory_file.exists():
            return []

        scored = []
        with open(self._memory_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if obj.get("_schema"):
                        continue
                    # 计算关键词命中数
                    insight = obj.get("insight", "")
                    entry_kw = obj.get("keywords", [])
                    all_text = insight + " " + " ".join(entry_kw)
                    hits = sum(1 for k in kw_list if k in all_text)
                    if hits > 0:
                        scored.append((hits, obj))
                except json.JSONDecodeError:
                    continue

        # 按命中数排序，取 top N
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:self.MAX_MEMORIES_IN_CTX]]

    @property
    def memory_count(self) -> int:
        """记忆条数"""
        if not self._memory_file.exists():
            return 0
        count = 0
        with open(self._memory_file, "r") as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    if not obj.get("_schema"):
                        count += 1
                except Exception:
                    pass
        return count

    # ═══════════════════════════════════════════════════════════
    #  交易决策提交
    # ═══════════════════════════════════════════════════════════

    def commit_decision(self, symbol: str, action: str, reason: str,
                        env_score: int = 0, **kwargs) -> dict:
        """
        提交一条交易决策

        Parameters
        ----------
        symbol : 股票代码
        action : "buy" / "sell" / "skip"
        reason : 决策理由
        env_score : 当时的环境评分
        """
        entry = {
            "ts": datetime.now().isoformat(),
            "symbol": symbol,
            "action": action,
            "reason": reason,
            "env_score": env_score,
            **kwargs,
        }

        self._append_jsonl(self._commits_file, entry)

        # 同时记录到结构化状态
        self.record_decision(
            f"{action.upper()} {symbol}",
            reason[:50],
        )

        self._emit("brain.commit", entry)
        return entry

    # ═══════════════════════════════════════════════════════════
    #  情绪状态
    # ═══════════════════════════════════════════════════════════

    def get_emotion(self) -> dict:
        """获取市场情绪状态"""
        if self._emotion_file.exists():
            try:
                with open(self._emotion_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"level": "neutral", "score": 50, "note": ""}

    def set_emotion(self, level: str, score: int, note: str = ""):
        """更新市场情绪"""
        data = {
            "level": level,
            "score": score,
            "note": note,
            "updated": datetime.now().isoformat(),
        }
        with open(self._emotion_file, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self._emit("brain.emotion", data)

    # ═══════════════════════════════════════════════════════════
    #  复盘日志
    # ═══════════════════════════════════════════════════════════

    def write_journal(self, content: str, date: str = None):
        """写入每日复盘日志"""
        date = date or datetime.now().strftime("%Y%m%d")
        journal_file = self.base_dir / "journal" / f"{date}.md"
        with open(journal_file, "w") as f:
            f.write(content)
        self._emit("brain.journal", {"date": date, "length": len(content)})

    def read_journal(self, date: str = None) -> str:
        """读取某日复盘日志"""
        date = date or datetime.now().strftime("%Y%m%d")
        journal_file = self.base_dir / "journal" / f"{date}.md"
        if journal_file.exists():
            return journal_file.read_text()
        return ""

    # ═══════════════════════════════════════════════════════════
    #  持久化
    # ═══════════════════════════════════════════════════════════

    def _save_state(self):
        """保存结构化状态到磁盘"""
        with open(self._state_file, "w") as f:
            json.dump(self._state, f, ensure_ascii=False, indent=2)

    def _load_state(self):
        """从磁盘加载结构化状态"""
        if self._state_file.exists():
            try:
                with open(self._state_file, "r") as f:
                    loaded = json.load(f)
                self._state.update(loaded)
            except Exception:
                pass

    def _ensure_jsonl_schema(self, filepath: Path, schema: dict):
        """确保 JSONL 首行是 schema"""
        if not filepath.exists():
            with open(filepath, "w") as f:
                f.write(json.dumps(schema, ensure_ascii=False) + "\n")

    def _append_jsonl(self, filepath: Path, data: dict):
        """追加一条 JSONL"""
        with open(filepath, "a") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def _emit(self, event_type: str, data: dict):
        """发事件到 EventLog"""
        if self.event_log:
            try:
                self.event_log.emit(event_type, data, source="brain")
            except Exception:
                pass

    # ═══════════════════════════════════════════════════════════
    #  诊断
    # ═══════════════════════════════════════════════════════════

    def diagnose(self) -> str:
        """打印 Brain 当前状态（诊断用）"""
        lines = [
            "=" * 50,
            "  Brain 诊断",
            "=" * 50,
            f"  记忆数: {self.memory_count}",
            f"  决策数: {len(self._state['decisions'])}",
            f"  状态项: {len(self._state['status'])}",
            f"  意图: {self._state['intent'] or '(空)'}",
            f"  下一步: {'; '.join(self._state['next_steps']) or '(空)'}",
        ]

        ctx = self.get_context_for_prompt("")
        token_est = len(ctx) // 2  # 粗估 token 数
        lines.append(f"  上下文估计: ~{token_est} tokens")

        emotion = self.get_emotion()
        lines.append(f"  情绪: {emotion.get('level', '?')} ({emotion.get('score', '?')})")
        lines.append("=" * 50)

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  便捷函数
# ═══════════════════════════════════════════════════════════════

_global_brain = None

def get_brain(base_dir: str = "data/brain", event_log=None) -> AgentBrain:
    """获取全局 Brain 实例"""
    global _global_brain
    if _global_brain is None:
        _global_brain = AgentBrain(base_dir=base_dir, event_log=event_log)
    return _global_brain


# ═══════════════════════════════════════════════════════════════
#  CLI 测试
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from event_log import EventLog

    log = EventLog()
    brain = AgentBrain(event_log=log)

    # 模拟一天的使用
    print("─── 设置状态 ───")
    brain.set_intent("验证 v6.1 盘中评分准确性")
    brain.update_status("market_environment.py", "v6.1已部署")
    brain.update_status("realtime_fetcher.py", "待验证AKShare连通")
    brain.update_status("backtest_v6.py", "权重已同步v6.1")
    brain.record_decision("趋势权重 0.30→0.25", "MA滞后导致节后首日误判58分")
    brain.record_decision("情绪权重 0.25→0.30", "涨跌比更实时")
    brain.set_next_step("明天盘中验证AKShare", "收盘后对比盘中vs收盘评分")

    print("─── 学习 ───")
    brain.learn_from_output("market_env", "环境评分",
                            "市场环境: 74/100 (一般) - 谨慎交易")
    brain.learn_from_output("trade_signals", "信号扫描",
                            "阶梯突破: 000858.SZ 突破168.5, 601012.SH 突破22.3\n无其他信号")

    print("─── 上下文生成 ───")
    ctx = brain.get_context_for_prompt("环境评分 信号扫描")
    print(ctx)
    print(f"\n上下文长度: {len(ctx)} 字符, 约 {len(ctx)//2} tokens")

    print()
    print(brain.diagnose())
