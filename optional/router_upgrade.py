"""
router_upgrade.py — Router 升级：从硬编码 → ToolCenter 动态获取
================================================================

改动点极小：
1. Router system prompt 中的工具列表 → ToolCenter.get_router_prompt()
2. 关键词路由的匹配字典 → ToolCenter.match_by_keywords()
3. 其余逻辑（LLM 调用、回退机制、Eval 集成）完全不变

本文件提供：
- RouterUpgrade 类：包装你现有 Router，最小改动接入 ToolCenter
- 关键词路由的平滑迁移方案
- 与 EventLog 的可选集成
"""

import re
import time
from typing import Optional, Any

from tool_center import ToolCenter


class RouterUpgrade:
    """
    Router 升级适配器

    不替换你现有的 Router 逻辑，只替换工具列表的来源。
    你可以选择：
      (a) 直接替换现有 Router（用这个类）
      (b) 只取其中的方法，粘贴到你现有 Router 中
    """

    def __init__(self, llm_caller: callable = None,
                 event_log: Any = None,
                 use_llm_routing: bool = True,
                 fallback_to_keywords: bool = True):
        """
        Parameters
        ----------
        llm_caller : LLM 调用函数，签名 (messages: list[dict]) -> str
                     如果为 None，则只用关键词路由
        event_log : EventLog 实例（可选）
        use_llm_routing : 是否使用 LLM 路由（True=优先用 LLM）
        fallback_to_keywords : LLM 路由失败时是否回退到关键词
        """
        self.llm_caller = llm_caller
        self.event_log = event_log
        self.use_llm_routing = use_llm_routing
        self.fallback_to_keywords = fallback_to_keywords

    # ─────────────────────────────────────────
    # 主路由方法
    # ─────────────────────────────────────────

    def route(self, query: str, context: str = "") -> dict:
        """
        路由用户查询到合适的工具

        Returns
        -------
        {
            "tool": "sector_rotation",     # 选中的工具名
            "confidence": 0.85,            # 置信度
            "method": "llm" | "keyword",   # 路由方式
            "reasoning": "...",            # LLM 的推理过程（仅 LLM 模式）
            "alternatives": [...],         # 备选工具
        }
        """
        start = time.time()
        result = None

        # 1. 尝试 LLM 路由
        if self.use_llm_routing and self.llm_caller:
            try:
                result = self._route_by_llm(query, context)
            except Exception as e:
                print(f"⚠️ LLM 路由失败: {e}")
                result = None

        # 2. 回退到关键词路由
        if result is None and self.fallback_to_keywords:
            result = self._route_by_keywords(query)

        # 3. 默认回退
        if result is None:
            result = {
                "tool": "market_overview",  # 兜底：给一个通用工具
                "confidence": 0.1,
                "method": "default",
                "reasoning": "未能匹配到合适工具，使用默认",
                "alternatives": [],
            }

        # 记录路由决策
        duration = time.time() - start
        if self.event_log:
            try:
                self.event_log.emit("router.dispatch", {
                    "query": query[:200],
                    "tool": result["tool"],
                    "method": result["method"],
                    "confidence": result["confidence"],
                    "duration_sec": round(duration, 3),
                    "alternatives": result.get("alternatives", []),
                }, source="router")
            except Exception:
                pass

        return result

    # ─────────────────────────────────────────
    # LLM 路由
    # ─────────────────────────────────────────

    def _route_by_llm(self, query: str, context: str = "") -> Optional[dict]:
        """
        LLM 路由：让 LLM 选择最合适的工具

        你只需要改这里的 system prompt —— 工具列表从 ToolCenter 动态获取
        """
        # 【核心改动】工具列表从 ToolCenter 获取
        tool_prompt = ToolCenter.get_router_prompt()
        tool_names = ToolCenter.get_tool_names()

        system_prompt = f"""你是一个A股量化投资 AI Agent 的路由器。
你的任务是根据用户的问题，选择最合适的工具来处理。

{tool_prompt}

请分析用户的问题，返回你选择的工具。
格式要求：
TOOL: 工具名
CONFIDENCE: 0-1 的置信度
REASONING: 简述选择理由
ALTERNATIVES: 备选工具1, 备选工具2（如有）

注意：
- 只能选择上面列出的工具
- 如果问题涉及多个方面，选最核心的那个工具
- 如果不确定，选 market_overview 作为兜底"""

        messages = [
            {"role": "system", "content": system_prompt},
        ]
        if context:
            messages.append({"role": "system", "content": f"[上下文]\n{context}"})
        messages.append({"role": "user", "content": query})

        # 调用 LLM
        response = self.llm_caller(messages)

        # 解析响应
        return self._parse_llm_response(response, tool_names)

    def _parse_llm_response(self, response: str, valid_tools: list[str]) -> Optional[dict]:
        """解析 LLM 路由响应"""
        if not response:
            return None

        # 提取 TOOL 字段
        tool_match = re.search(r"TOOL:\s*(\S+)", response, re.IGNORECASE)
        if not tool_match:
            return None

        tool_name = tool_match.group(1).strip().lower()

        # 验证工具名有效
        if tool_name not in valid_tools:
            # 尝试模糊匹配
            for valid in valid_tools:
                if tool_name in valid or valid in tool_name:
                    tool_name = valid
                    break
            else:
                return None  # 无法匹配，触发回退

        # 提取置信度
        conf_match = re.search(r"CONFIDENCE:\s*([\d.]+)", response, re.IGNORECASE)
        confidence = float(conf_match.group(1)) if conf_match else 0.7

        # 提取推理
        reason_match = re.search(r"REASONING:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        reasoning = reason_match.group(1).strip() if reason_match else ""

        # 提取备选
        alt_match = re.search(r"ALTERNATIVES:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        alternatives = []
        if alt_match:
            alts = [a.strip().lower() for a in alt_match.group(1).split(",")]
            alternatives = [a for a in alts if a in valid_tools and a != tool_name]

        return {
            "tool": tool_name,
            "confidence": min(1.0, max(0.0, confidence)),
            "method": "llm",
            "reasoning": reasoning,
            "alternatives": alternatives,
        }

    # ─────────────────────────────────────────
    # 关键词路由
    # ─────────────────────────────────────────

    def _route_by_keywords(self, query: str) -> Optional[dict]:
        """
        关键词路由：基于 ToolCenter 注册的关键词匹配

        【核心改动】不再维护硬编码的关键词字典，
        而是用 ToolCenter.match_by_keywords() 动态匹配
        """
        matches = ToolCenter.match_by_keywords(query, top_k=3)

        if not matches:
            return None

        best_tool, best_score = matches[0]
        alternatives = [tool for tool, _ in matches[1:]]

        return {
            "tool": best_tool,
            "confidence": min(1.0, best_score),
            "method": "keyword",
            "reasoning": f"关键词匹配 (得分: {best_score:.2f})",
            "alternatives": alternatives,
        }

    # ─────────────────────────────────────────
    # 你现有 Router 的迁移辅助
    # ─────────────────────────────────────────

    @staticmethod
    def get_system_prompt_patch() -> str:
        """
        获取需要替换到你现有 Router system prompt 中的工具列表部分

        用法：
            # 在你现有的 router.py 中，找到硬编码的 skill 列表，替换为：
            tool_section = RouterUpgrade.get_system_prompt_patch()
            system_prompt = f"你是路由器...\\n{tool_section}\\n请选择工具..."
        """
        return ToolCenter.get_router_prompt()

    @staticmethod
    def migrate_keyword_dict(old_keyword_dict: dict) -> None:
        """
        迁移辅助：检查旧关键词字典中有没有遗漏的关键词

        用法：
            # 传入你现有的关键词字典
            old_dict = {
                "板块": "sector_rotation",
                "轮动": "sector_rotation",
                "情绪": "sentiment_analysis",
                ...
            }
            RouterUpgrade.migrate_keyword_dict(old_dict)
        """
        tc_map = ToolCenter.get_keyword_map()
        tc_keywords = set(tc_map.keys())
        old_keywords = set(k.lower() for k in old_keyword_dict.keys())

        missing_in_tc = old_keywords - tc_keywords
        new_in_tc = tc_keywords - old_keywords

        if missing_in_tc:
            print(f"⚠️ 以下关键词在旧字典中存在，但 ToolCenter 中缺失（需补充注册）:")
            for kw in sorted(missing_in_tc):
                old_tool = old_keyword_dict.get(kw, "?")
                print(f"   '{kw}' → {old_tool}")

        if new_in_tc:
            print(f"✅ ToolCenter 新增了以下关键词（旧字典中没有的）:")
            for kw in sorted(new_in_tc):
                tools = tc_map.get(kw, [])
                print(f"   '{kw}' → {tools}")

        if not missing_in_tc and not new_in_tc:
            print("✅ 关键词完全匹配，迁移无遗漏")


# ═══════════════════════════════════════════
# 最小改动迁移指南
# ═══════════════════════════════════════════

MIGRATION_GUIDE = """
╔══════════════════════════════════════════════════════════════╗
║            Router → ToolCenter 最小改动迁移指南              ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  改动 1: 系统启动时注册 skills                                ║
║  ─────────────────────────────────                           ║
║  # main.py / daily_agent.py 入口处加一行：                    ║
║  from skill_registry import register_all_skills               ║
║  register_all_skills()                                        ║
║                                                              ║
║  改动 2: Router system prompt 中的工具列表                    ║
║  ─────────────────────────────────                           ║
║  # 找到你硬编码的 skill 列表，替换为：                         ║
║  from tool_center import ToolCenter                           ║
║  tool_prompt = ToolCenter.get_router_prompt()                 ║
║  # 然后拼接到 system prompt 中                                ║
║                                                              ║
║  改动 3: 关键词路由的匹配逻辑（如果有的话）                    ║
║  ─────────────────────────────────                           ║
║  # 找到你的关键词字典和匹配逻辑，替换为：                      ║
║  matches = ToolCenter.match_by_keywords(query, top_k=3)      ║
║  if matches:                                                  ║
║      tool_name = matches[0][0]                                ║
║                                                              ║
║  就这三处。其余代码（LLM 调用、Eval、Scratchpad）完全不动。    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
