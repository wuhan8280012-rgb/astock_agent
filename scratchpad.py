"""
Scratchpad: 所有Skill运行的append-only日志

设计参考 Dexter:
  - JSONL格式，每行一条记录
  - 磁盘持久化，崩溃不丢数据
  - 支持按日期/Skill/最近N条查询
  - 用于复盘时回溯"Agent当时看到了什么数据"
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any


class Scratchpad:
    """Append-only 运行日志"""

    def __init__(self, path: str = "./knowledge/scratchpad.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, skill: str, output_data: Any = None,
            input_data: Any = None, metadata: Dict = None):
        """记录一次Skill运行"""
        entry = {
            "ts": datetime.now().isoformat(),
            "date": datetime.now().strftime("%Y%m%d"),
            "skill": skill,
        }
        if input_data is not None:
            entry["input"] = self._safe(input_data)
        if output_data is not None:
            entry["output"] = self._safe(output_data)
        if metadata:
            entry["meta"] = metadata

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def log_start(self, agent: str, trade_date: str = ""):
        self.log("_start", metadata={"agent": agent, "trade_date": trade_date})

    def log_end(self, agent: str, report_path: str = ""):
        self.log("_end", metadata={"agent": agent, "report": report_path})

    def query(self, skill: str = None, date: str = None, last_n: int = None) -> List[Dict]:
        """查询历史记录"""
        entries = self._read()
        if skill:
            entries = [e for e in entries if e.get("skill") == skill]
        if date:
            entries = [e for e in entries if e.get("date") == date]
        if last_n:
            entries = entries[-last_n:]
        return entries

    def get_latest_run(self) -> List[Dict]:
        """获取最近一次完整运行"""
        entries = self._read()
        for i in range(len(entries) - 1, -1, -1):
            if entries[i].get("skill") == "_start":
                return entries[i:]
        return entries[-20:]

    def stats(self) -> Dict:
        entries = self._read()
        skills = {}
        for e in entries:
            s = e.get("skill", "?")
            skills[s] = skills.get(s, 0) + 1
        return {"total": len(entries), "by_skill": skills}

    def _read(self) -> List[Dict]:
        if not self.path.exists():
            return []
        entries = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return entries

    def _safe(self, obj):
        """确保可序列化"""
        try:
            json.dumps(obj, default=str)
            return obj
        except (TypeError, ValueError):
            return str(obj)
