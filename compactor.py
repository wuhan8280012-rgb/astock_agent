"""
ContextCompactor — 上下文窗口自动压缩
=====================================
防止长分析链（全市场扫描→板块轮动→个股筛选→回测）爆 context window。

接入方式：
  在 router.py 的 _route_with_llm() 中，发给 LLM 前过一遍：
      from compactor import compactor
      messages = compactor.compact(messages)

配置：config.py 中 COMPACTION_MAX_TOKENS = 8000
"""

import re
from typing import Optional, Callable


# 配置
try:
    from config import COMPACTION_MAX_TOKENS
except ImportError:
    COMPACTION_MAX_TOKENS = 8000


class ContextCompactor:
    """
    上下文压缩器

    策略：
    1. 估算当前 messages 总 token 数
    2. 如果未超限，原样返回
    3. 如果超限，保留最近 N 条 + 对早期消息做摘要
    4. 如果有 LLM summarizer，用 LLM 摘要；否则用截断+标记

    特别处理：
    - system 消息永远保留（含 Router prompt、Brain 上下文）
    - 最近的 user/assistant 轮次保留完整
    - 中间的历史对话做压缩
    """

    def __init__(self, max_tokens: int = None,
                 summarizer: Optional[Callable] = None,
                 preserve_recent: int = 4):
        """
        Parameters
        ----------
        max_tokens : 上下文 token 上限（默认从 config 读取）
        summarizer : 摘要函数 (text: str) -> str，不传则用截断
        preserve_recent : 保留最近几条 user/assistant 消息
        """
        self.max_tokens = max_tokens or COMPACTION_MAX_TOKENS
        self.summarizer = summarizer
        self.preserve_recent = preserve_recent

    def compact(self, messages: list[dict],
                preserve_recent: int = None) -> list[dict]:
        """
        压缩消息列表

        Parameters
        ----------
        messages : [{"role": "system"|"user"|"assistant", "content": "..."}]
        preserve_recent : 覆盖默认的保留条数

        Returns
        -------
        压缩后的消息列表（可能更短）
        """
        if not messages:
            return messages

        total = self._total_tokens(messages)
        if total <= self.max_tokens:
            return messages

        keep_recent = preserve_recent or self.preserve_recent

        # 分离 system 消息和对话消息
        system_msgs = [m for m in messages if m.get("role") == "system"]
        chat_msgs = [m for m in messages if m.get("role") != "system"]

        if len(chat_msgs) <= keep_recent:
            # 对话太短不压缩，可能是 system prompt 太长
            return messages

        # 分割：需要压缩的早期对话 + 保留的近期对话
        early = chat_msgs[:-keep_recent]
        recent = chat_msgs[-keep_recent:]

        # 检查压缩后是否足够
        system_tokens = sum(self._estimate_tokens(m.get("content", "")) for m in system_msgs)
        recent_tokens = sum(self._estimate_tokens(m.get("content", "")) for m in recent)
        budget_for_summary = max(500, self.max_tokens - system_tokens - recent_tokens - 200)

        # 生成摘要
        summary = self._summarize(early, budget_for_summary)

        # 组装结果
        result = system_msgs.copy()
        if summary:
            result.append({
                "role": "user",
                "content": f"[以下是之前对话的摘要，供参考]\n{summary}"
            })
        result.extend(recent)

        return result

    def needs_compaction(self, messages: list[dict]) -> bool:
        """检查是否需要压缩（不执行压缩）"""
        return self._total_tokens(messages) > self.max_tokens

    def _summarize(self, messages: list[dict], max_chars: int) -> str:
        """对消息列表生成摘要"""
        # 拼接原文
        lines = []
        for m in messages:
            role = m.get("role", "?")
            content = m.get("content", "")
            lines.append(f"[{role}] {content}")
        full_text = "\n".join(lines)

        if self.summarizer:
            # 有 LLM summarizer，用它生成摘要
            try:
                return self.summarizer(full_text)[:max_chars]
            except Exception:
                pass

        # 无 summarizer，用智能截断
        return self._truncate_summary(full_text, max_chars)

    @staticmethod
    def _truncate_summary(text: str, max_chars: int) -> str:
        """
        智能截断：保留开头和结尾，中间标记省略

        比简单的 text[:N] 更好，因为保留了最近的上下文
        """
        if len(text) <= max_chars:
            return text

        # 开头 40%，结尾 40%，中间省略
        head_len = int(max_chars * 0.4)
        tail_len = int(max_chars * 0.4)
        omitted = len(text) - head_len - tail_len

        head = text[:head_len]
        tail = text[-tail_len:]

        # 在句子边界截断（避免截断到一半）
        head = head[:head.rfind("。") + 1] if "。" in head else head
        tail_start = tail.find("。")
        if tail_start > 0:
            tail = tail[tail_start + 1:]

        return f"{head}\n\n...（省略约 {omitted} 字符的历史对话）...\n\n{tail}"

    def _total_tokens(self, messages: list[dict]) -> int:
        return sum(
            self._estimate_tokens(m.get("content", ""))
            for m in messages
        )

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """粗略估算 token 数（中文约 1.5 字/token，英文约 4 字符/token）"""
        if not text:
            return 0
        cn = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other = len(text) - cn
        return int(cn / 1.5 + other / 4)


# ═══════════════════════════════════════════
# 全局实例
# ═══════════════════════════════════════════

compactor = ContextCompactor()
