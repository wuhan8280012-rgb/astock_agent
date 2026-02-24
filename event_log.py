"""
event_log.py — 全链路事件追溯（Context Engineering 重构）
========================================================
设计原则（来自 Agent Skills for Context Engineering）：

1. Append-Only JSONL — 永不丢失，写入即不可变
2. Observation Masking — 读取时按需压缩，原始数据保留在磁盘
3. Schema-First — 每个 JSONL 文件首行是 schema，agent 可自主理解结构
4. tokens-per-task — 不追求单次最小，追求任务完成总 token 最少

用法：
  log = EventLog()
  log.emit("env.scored", {"score": 74, "level": "一般"}, source="market_env")
  log.emit("signal.found", {"symbol": "000001", "type": "ladder"}, source="trade_signals")

  # 读取（自动 observation masking，只返回摘要）
  summary = log.get_summary(last_n=20)

  # 读取原始（需要详情时才展开）
  events = log.get_raw(event_type="signal.found", last_n=5)
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


# Schema 定义（JSONL 首行）
EVENT_SCHEMA = {
    "_schema": True,
    "fields": {
        "ts": "ISO timestamp",
        "type": "event type (dot-separated hierarchy)",
        "source": "module that emitted",
        "data": "event payload (dict)",
        "summary": "one-line human-readable summary",
    },
    "version": "2.0",
}


class EventLog:
    """
    Append-only event log with observation masking.

    写入：永远 append，不修改历史
    读取：默认返回 masked summary，需要时才展开原始数据
    存储：每日一个 JSONL 文件，首行是 schema
    """

    def __init__(self, base_dir: str = "data/event_log"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._today = datetime.now().strftime("%Y%m%d")
        self._file = self.base_dir / f"{self._today}.jsonl"
        self._ensure_schema()

    def _ensure_schema(self):
        """确保 JSONL 文件首行是 schema"""
        if not self._file.exists():
            with open(self._file, "w") as f:
                f.write(json.dumps(EVENT_SCHEMA, ensure_ascii=False) + "\n")

    def _rotate_if_needed(self):
        """跨日自动切换文件"""
        today = datetime.now().strftime("%Y%m%d")
        if today != self._today:
            self._today = today
            self._file = self.base_dir / f"{self._today}.jsonl"
            self._ensure_schema()

    # ─────────────────────────────────────────
    # 写入（Append-Only）
    # ─────────────────────────────────────────

    def emit(self, event_type: str, data: dict = None,
             source: str = "", summary: str = ""):
        """
        写入一条事件

        Parameters
        ----------
        event_type : 事件类型，用点分层级（如 "env.scored", "signal.found"）
        data : 事件数据
        source : 来源模块
        summary : 一行摘要（用于 observation masking）
        """
        self._rotate_if_needed()

        # 自动生成 summary
        if not summary:
            summary = self._auto_summary(event_type, data or {})

        event = {
            "ts": datetime.now().isoformat(),
            "type": event_type,
            "source": source,
            "data": data or {},
            "summary": summary,
        }

        with open(self._file, "a") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

        return event

    def _auto_summary(self, event_type: str, data: dict) -> str:
        """从事件数据自动生成一行摘要"""
        parts = []

        # 常见字段的摘要模板
        if "score" in data:
            parts.append(f"score={data['score']}")
        if "level" in data:
            parts.append(data["level"])
        if "symbol" in data:
            parts.append(data["symbol"])
        if "type" in data and event_type.startswith("signal"):
            parts.append(data["type"])
        if "action" in data:
            parts.append(data["action"])
        if "error" in data:
            parts.append(f"ERROR: {str(data['error'])[:50]}")

        return " | ".join(parts)

    # ─────────────────────────────────────────
    # 读取 — Observation Masking（默认只返回摘要）
    # ─────────────────────────────────────────

    def get_summary(self, last_n: int = 20, event_type: str = None,
                    date: str = None) -> str:
        """
        获取事件摘要（Observation Masking）

        只返回每条事件的 summary 字段，不展开 data。
        这是给 LLM/Brain 用的——用最少 token 传递最多信息。

        Returns
        -------
        str: 多行摘要文本
        """
        events = self._read_events(last_n=last_n, event_type=event_type, date=date)

        lines = []
        for e in events:
            ts = e["ts"][11:19]  # 只取时间 HH:MM:SS
            etype = e.get("type", "?")
            summary = str(e.get("summary", ""))[:80]
            if summary:
                lines.append(f"[{ts}] {etype} | {summary}")
            else:
                lines.append(f"[{ts}] {etype}")

        if not lines:
            return "(无事件)"

        return "\n".join(lines)

    def get_raw(self, last_n: int = 5, event_type: str = None,
                date: str = None) -> list:
        """
        获取原始事件（Progressive Disclosure — 需要详情时才调用）

        Returns
        -------
        list[dict]: 完整事件列表
        """
        return self._read_events(last_n=last_n, event_type=event_type, date=date)

    def get_context_block(self, last_n: int = 10) -> str:
        """
        给 Brain 的上下文块（结构化，固定格式）

        格式：
          ## 最近事件 (10条)
          [10:32:15] env.scored | score=74 | 一般
          [10:33:01] signal.found | 000001 | ladder
        """
        summary = self.get_summary(last_n=last_n)
        return f"## 最近事件 ({last_n}条)\n{summary}"

    # ─────────────────────────────────────────
    # 统计
    # ─────────────────────────────────────────

    def count(self, event_type: str = None, date: str = None) -> int:
        """统计事件数"""
        return len(self._read_events(last_n=99999, event_type=event_type, date=date))

    def list_dates(self) -> list:
        """列出所有有事件的日期"""
        return sorted([f.stem for f in self.base_dir.glob("*.jsonl")])

    # ─────────────────────────────────────────
    # 内部
    # ─────────────────────────────────────────

    def _read_events(self, last_n: int = 20, event_type: str = None,
                     date: str = None) -> list:
        """读取事件（跳过 schema 行）"""
        target_file = self.base_dir / f"{date or self._today}.jsonl"
        if not target_file.exists():
            return []

        events = []
        with open(target_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if obj.get("_schema"):
                        continue
                    if event_type and not obj.get("type", "").startswith(event_type):
                        continue
                    events.append(obj)
                except json.JSONDecodeError:
                    continue

        return events[-last_n:]


# ═══════════════════════════════════════════════════════════════
#  便捷函数
# ═══════════════════════════════════════════════════════════════

_global_log = None

def get_event_log(base_dir: str = "data/event_log") -> EventLog:
    """获取全局 EventLog 实例"""
    global _global_log
    if _global_log is None:
        _global_log = EventLog(base_dir)
    return _global_log


if __name__ == "__main__":
    log = EventLog()
    log.emit("system.startup", {"version": "v6.1"}, source="test")
    log.emit("env.scored", {"score": 74, "level": "一般", "source": "intraday"},
             source="market_env")
    log.emit("signal.found", {"symbol": "000858.SZ", "type": "ladder", "price": 168.5},
             source="trade_signals")
    log.emit("signal.found", {"symbol": "601012.SH", "type": "leader", "price": 22.3},
             source="trade_signals")
    log.emit("decision.skip", {"reason": "节后第一天，观察为主"},
             source="workflow")

    print("=== Observation Masking（摘要模式）===")
    print(log.get_summary())
    print()
    print("=== Context Block（给 Brain 用）===")
    print(log.get_context_block())
    print()
    print("=== Raw（展开详情）===")
    for e in log.get_raw(event_type="signal", last_n=2):
        print(json.dumps(e, ensure_ascii=False, indent=2))
