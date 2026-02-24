"""
知识库层: 经验复盘 + 历史事件匹配

功能:
  1. 存储历史市场事件及其后续走势
  2. 存储个人投资决策复盘
  3. 根据当前市场状态匹配历史相似情景
  4. 提供决策参考
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

try:
    from config import KNOWLEDGE_DB
except ImportError:
    KNOWLEDGE_DB = "./knowledge/kb.json"


class KnowledgeBase:
    """知识库管理器"""

    def __init__(self, db_path: str = None):
        self.db_path = Path(db_path or KNOWLEDGE_DB)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def _load(self) -> dict:
        if self.db_path.exists():
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "events": [],        # 历史市场事件
            "decisions": [],     # 投资决策记录
            "patterns": [],      # 市场模式库
            "lessons": [],       # 经验教训
        }

    def _save(self):
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    # ========================================================
    # 历史事件管理
    # ========================================================
    def add_event(self, event: dict):
        """
        添加历史事件
        event = {
            "date": "20250204",
            "title": "春节后开盘大跌",
            "description": "DeepSeek事件引发市场对AI竞争格局重估",
            "market_reaction": "创业板跌3%，科技股领跌",
            "indicators": {
                "sentiment": "恐慌",
                "north_flow": -80,
                "shibor_change": 0.05,
            },
            "subsequent_move": "3日后反弹，科技股V型反转",
            "lesson": "事件性冲击通常提供买入机会，关键看流动性是否受损",
            "tags": ["事件冲击", "科技股", "AI"],
        }
        """
        event["added_at"] = datetime.now().isoformat()
        self._data["events"].append(event)
        self._save()

    def add_decision(self, decision: dict):
        """
        记录投资决策
        decision = {
            "date": "20260220",
            "action": "买入",
            "target": "比亚迪 002594.SZ",
            "reason": "板块轮动信号+缩量整理突破",
            "position_pct": 0.08,
            "result": "",  # 后续填写
            "review": "",  # 复盘
        }
        """
        decision["added_at"] = datetime.now().isoformat()
        self._data["decisions"].append(decision)
        self._save()

    def add_pattern(self, pattern: dict):
        """
        添加市场模式
        pattern = {
            "name": "两融余额新高+北向流出→短期见顶",
            "conditions": {
                "margin_trend": "快速上升",
                "north_flow": "持续流出",
                "sentiment": "贪婪",
            },
            "typical_outcome": "1-2周内出现5%+回调",
            "occurrences": ["20210218", "20230731", "20240520"],
            "confidence": "中",
        }
        """
        pattern["added_at"] = datetime.now().isoformat()
        self._data["patterns"].append(pattern)
        self._save()

    def add_lesson(self, lesson: str, context: str = "", tags: List[str] = None):
        """添加经验教训"""
        self._data["lessons"].append({
            "lesson": lesson,
            "context": context,
            "tags": tags or [],
            "added_at": datetime.now().isoformat(),
        })
        self._save()

    # ========================================================
    # 查询与匹配
    # ========================================================
    def search_events(self, tags: List[str] = None, keyword: str = None) -> List[dict]:
        """搜索历史事件"""
        results = self._data["events"]
        if tags:
            results = [
                e for e in results
                if any(t in e.get("tags", []) for t in tags)
            ]
        if keyword:
            keyword = keyword.lower()
            results = [
                e for e in results
                if keyword in e.get("title", "").lower()
                or keyword in e.get("description", "").lower()
            ]
        return results

    def match_pattern(self, current_state: dict) -> List[dict]:
        """
        根据当前市场状态匹配历史模式

        current_state = {
            "sentiment": "贪婪",
            "margin_trend": "快速上升",
            "north_flow": "持续流出",
            "liquidity": "中性偏紧",
        }
        """
        matches = []
        for pattern in self._data["patterns"]:
            conditions = pattern.get("conditions", {})
            match_count = 0
            total = len(conditions)
            for key, expected in conditions.items():
                if current_state.get(key) == expected:
                    match_count += 1
            if total > 0 and match_count / total >= 0.5:
                matches.append({
                    **pattern,
                    "match_score": match_count / total,
                })
        return sorted(matches, key=lambda x: x["match_score"], reverse=True)

    def get_recent_decisions(self, n: int = 10) -> List[dict]:
        """获取最近N条决策记录"""
        return self._data["decisions"][-n:]

    def get_lessons(self, tags: List[str] = None) -> List[dict]:
        """获取经验教训"""
        if not tags:
            return self._data["lessons"]
        return [
            l for l in self._data["lessons"]
            if any(t in l.get("tags", []) for t in tags)
        ]

    def get_stats(self) -> dict:
        """获取知识库统计"""
        return {
            "events": len(self._data["events"]),
            "decisions": len(self._data["decisions"]),
            "patterns": len(self._data["patterns"]),
            "lessons": len(self._data["lessons"]),
        }


def init_default_knowledge():
    """初始化默认知识库（A股常见模式）"""
    kb = KnowledgeBase()

    if kb.get_stats()["patterns"] > 0:
        print("[KnowledgeBase] 知识库已存在，跳过初始化")
        return kb

    print("[KnowledgeBase] 初始化默认A股知识库...")

    # 添加常见模式
    kb.add_pattern({
        "name": "两融新高+北向流出→短期见顶",
        "conditions": {
            "sentiment": "极度贪婪",
            "margin_trend": "杠杆加速上升",
            "north_flow_trend": "温和流出",
        },
        "typical_outcome": "1-2周内回调5-10%",
        "occurrences": ["20210218", "20230731", "20240520"],
        "confidence": "高",
    })

    kb.add_pattern({
        "name": "极度缩量+恐慌→底部区域",
        "conditions": {
            "sentiment": "极度恐慌",
            "turnover_trend": "极度缩量",
            "margin_trend": "去杠杆",
        },
        "typical_outcome": "2-4周内见底反弹",
        "occurrences": ["20181019", "20200323", "20240205"],
        "confidence": "高",
    })

    kb.add_pattern({
        "name": "北向持续流入+板块分化→结构性行情",
        "conditions": {
            "north_flow_trend": "持续流入",
            "sentiment": "中性",
            "turnover_trend": "正常",
        },
        "typical_outcome": "核心资产/权重股领涨，小盘股震荡",
        "occurrences": ["20190301", "20201101", "20230901"],
        "confidence": "中",
    })

    kb.add_pattern({
        "name": "政策底→市场底 (A股经典模式)",
        "conditions": {
            "sentiment": "恐慌",
            "policy_signal": "积极",
        },
        "typical_outcome": "政策出台后市场还会惯性下跌1-3周，随后见底",
        "occurrences": ["20181019", "20200401", "20240924"],
        "confidence": "高",
    })

    # 添加经验教训
    kb.add_lesson(
        "A股T+1制度下追涨停要极其谨慎，封板强度（封单量/流通市值>5%）是关键指标",
        context="多次被炸板止损",
        tags=["T+1", "涨停", "风控"],
    )
    kb.add_lesson(
        "北向资金单日大幅流入不一定是抄底信号，需要连续3天以上才有参考意义",
        context="单日流入后次日反转的情况很多",
        tags=["北向资金", "信号验证"],
    )
    kb.add_lesson(
        "板块轮动信号通常滞后1-2天，等确认后再入场比追第一根大阳线更安全",
        context="轮动策略优化",
        tags=["板块轮动", "入场时机"],
    )
    kb.add_lesson(
        "财报季（1月/4月/7月/10月）前2周避免重仓单只股票，业绩雷防不胜防",
        context="某次财报暴雷导致大幅亏损",
        tags=["财报", "风控", "仓位管理"],
    )

    print(f"[KnowledgeBase] 初始化完成: {kb.get_stats()}")
    return kb


if __name__ == "__main__":
    kb = init_default_knowledge()
    print(f"\n知识库统计: {kb.get_stats()}")

    # 测试模式匹配
    current = {
        "sentiment": "极度贪婪",
        "margin_trend": "杠杆加速上升",
        "north_flow_trend": "温和流出",
    }
    matches = kb.match_pattern(current)
    print(f"\n当前状态匹配到 {len(matches)} 个历史模式:")
    for m in matches:
        print(f"  [{m['name']}] 匹配度:{m['match_score']:.0%} | 典型结果:{m['typical_outcome']}")
