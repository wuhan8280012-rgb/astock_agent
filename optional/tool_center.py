"""
ToolCenter — 集中式工具注册中心
================================
让 Router 从硬编码 skill 列表 → 动态获取，新增 skill 只需注册即可。

设计原则：
1. 注册逻辑跟着 skill 走（每个 skill 模块自行注册，而非集中维护）
2. 提供装饰器 @tool() 和函数式 register() 两种注册方式
3. 导出给 Router LLM 的 prompt、关键词路由的匹配表、Eval 层的工具列表
4. 与 EventLog 可选集成（记录工具调用）
5. 零外部依赖，纯标准库

接入方式：
    在每个 skill 模块顶层:
        from tool_center import ToolCenter, tool

        @tool(name="sector_rotation", category="analysis",
              keywords=["板块", "轮动", "行业"],
              description="分析A股板块轮动状态")
        def analyze_sector_rotation(query: str = "", **kwargs) -> str:
            ...

    在 Router 中:
        from tool_center import ToolCenter
        prompt = ToolCenter.get_router_prompt()      # 替代硬编码列表
        keywords = ToolCenter.get_keyword_map()      # 替代硬编码关键词字典
"""

import time
import inspect
import re
from typing import Callable, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from functools import wraps


# ═══════════════════════════════════════════
# 工具元数据
# ═══════════════════════════════════════════

@dataclass
class ToolMeta:
    """工具元数据 — 描述一个已注册的 skill/工具"""
    name: str                          # 工具名（唯一标识）
    description: str                   # 自然语言描述（给 LLM 看）
    category: str = "analysis"         # analysis / trading / risk / brain / system
    keywords: list[str] = field(default_factory=list)   # 关键词（给关键词路由用）
    parameters: dict = field(default_factory=dict)       # 参数 schema（可选）
    examples: list[str] = field(default_factory=list)    # 示例查询（可选，帮助 Router 理解）
    priority: int = 0                  # 优先级（同时匹配多个工具时的排序）
    enabled: bool = True               # 是否启用
    version: str = "1.0"              # 版本号

    def to_llm_schema(self) -> dict:
        """导出为 LLM function calling 格式"""
        schema = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
            }
        }
        if self.parameters:
            schema["function"]["parameters"] = self.parameters
        return schema

    def to_router_description(self) -> str:
        """导出为 Router prompt 中的工具描述行"""
        line = f"- **{self.name}**: {self.description}"
        if self.examples:
            examples_str = "；".join(self.examples[:2])
            line += f"（示例：{examples_str}）"
        return line


# ═══════════════════════════════════════════
# ToolCenter 主类
# ═══════════════════════════════════════════

class ToolCenter:
    """
    集中式工具注册中心（单例模式）

    所有 skill 模块在此注册，Router / Eval / CronEngine 从此获取工具信息。

    核心方法：
    - register()             : 注册工具（函数式）
    - @tool()                : 注册工具（装饰器）
    - execute()              : 执行工具
    - get_router_prompt()    : 生成 Router LLM 的工具描述 prompt
    - get_keyword_map()      : 生成关键词路由的匹配字典
    - get_tool_names()       : 获取工具名列表
    - match_by_keywords()    : 关键词匹配（替代硬编码关键词路由）
    """

    _tools: dict[str, Callable] = {}
    _metadata: dict[str, ToolMeta] = {}
    _event_log: Optional[Any] = None  # 可选的 EventLog 实例

    # ─────────────────────────────────────────
    # 注册
    # ─────────────────────────────────────────

    @classmethod
    def register(cls, name: str, func: Callable, description: str,
                 category: str = "analysis", keywords: list[str] = None,
                 parameters: dict = None, examples: list[str] = None,
                 priority: int = 0, enabled: bool = True, version: str = "1.0"):
        """
        注册一个工具（函数式接口）

        Parameters
        ----------
        name : 工具唯一名称
        func : 执行函数
        description : 自然语言描述（给 LLM 看）
        category : 分类 (analysis / trading / risk / brain / system)
        keywords : 关键词列表（给关键词路由匹配用）
        parameters : 参数 JSON Schema（可选）
        examples : 示例查询（可选）
        priority : 优先级（数字越大越优先）
        enabled : 是否启用
        version : 版本号

        示例：
            ToolCenter.register(
                name="sector_rotation",
                func=analyze_sector_rotation,
                description="分析A股板块轮动：板块强弱排名、轮动方向、热点切换信号",
                category="analysis",
                keywords=["板块", "轮动", "行业", "热点", "切换", "sector"],
                examples=["当前哪个板块最强", "板块轮动到什么阶段了"],
            )
        """
        cls._tools[name] = func
        cls._metadata[name] = ToolMeta(
            name=name,
            description=description,
            category=category,
            keywords=keywords or [],
            parameters=parameters or {},
            examples=examples or [],
            priority=priority,
            enabled=enabled,
            version=version,
        )

    @classmethod
    def unregister(cls, name: str):
        """注销工具"""
        cls._tools.pop(name, None)
        cls._metadata.pop(name, None)

    @classmethod
    def set_event_log(cls, event_log):
        """设置 EventLog 实例（可选，用于记录工具调用）"""
        cls._event_log = event_log

    # ─────────────────────────────────────────
    # 执行
    # ─────────────────────────────────────────

    @classmethod
    def execute(cls, name: str, **kwargs) -> Any:
        """
        执行指定工具

        如果设置了 EventLog，会自动记录调用和结果。
        """
        if name not in cls._tools:
            available = ", ".join(cls.get_tool_names())
            raise ValueError(f"工具 '{name}' 未注册。可用工具: [{available}]")

        meta = cls._metadata[name]
        if not meta.enabled:
            raise ValueError(f"工具 '{name}' 已禁用")

        start = time.time()
        error = None
        result = None

        try:
            result = cls._tools[name](**kwargs)
        except Exception as e:
            error = str(e)
            raise
        finally:
            duration = time.time() - start
            # 记录到 EventLog
            if cls._event_log:
                try:
                    cls._event_log.emit("tool.executed", {
                        "tool": name,
                        "category": meta.category,
                        "kwargs_keys": list(kwargs.keys()),
                        "duration_sec": round(duration, 3),
                        "success": error is None,
                        "error": error,
                        "result_preview": str(result)[:300] if result else None,
                    }, source="tool_center")
                except Exception:
                    pass  # EventLog 写入失败不影响主流程

        return result

    @classmethod
    def has_tool(cls, name: str) -> bool:
        return name in cls._tools and cls._metadata[name].enabled

    # ─────────────────────────────────────────
    # Router LLM 集成
    # ─────────────────────────────────────────

    @classmethod
    def get_router_prompt(cls, categories: list[str] = None,
                          include_examples: bool = True) -> str:
        """
        生成给 Router LLM 的工具描述 prompt

        替代你 Router 中硬编码的 skill 列表。

        Parameters
        ----------
        categories : 只包含指定分类的工具（默认全部）
        include_examples : 是否包含示例查询

        Returns
        -------
        格式化的 prompt 文本，可直接拼接到 Router system prompt 中
        """
        # 按 category 分组，每组按 priority 降序
        grouped: dict[str, list[ToolMeta]] = defaultdict(list)
        for meta in cls._metadata.values():
            if not meta.enabled:
                continue
            if categories and meta.category not in categories:
                continue
            grouped[meta.category].append(meta)

        if not grouped:
            return "当前没有可用的分析工具。"

        # category 展示顺序
        cat_order = ["analysis", "trading", "risk", "brain", "system"]
        cat_labels = {
            "analysis": "📊 分析工具",
            "trading": "💰 交易工具",
            "risk": "🛡️ 风控工具",
            "brain": "🧠 认知工具",
            "system": "⚙️ 系统工具",
        }

        lines = ["以下是可用的工具，请根据用户问题选择最合适的工具：\n"]

        for cat in cat_order:
            tools = grouped.get(cat, [])
            if not tools:
                continue
            tools.sort(key=lambda m: -m.priority)
            label = cat_labels.get(cat, cat.upper())
            lines.append(f"### {label}")
            for meta in tools:
                lines.append(meta.to_router_description())
            lines.append("")

        # 额外的 category（未在预定义列表中的）
        for cat, tools in grouped.items():
            if cat not in cat_order:
                tools.sort(key=lambda m: -m.priority)
                lines.append(f"### {cat.upper()}")
                for meta in tools:
                    lines.append(meta.to_router_description())
                lines.append("")

        return "\n".join(lines)

    # ─────────────────────────────────────────
    # 关键词路由集成
    # ─────────────────────────────────────────

    @classmethod
    def get_keyword_map(cls) -> dict[str, list[str]]:
        """
        生成关键词 → 工具名的映射字典

        替代你 Router 中硬编码的关键词字典。

        Returns
        -------
        {关键词: [工具名1, 工具名2, ...]}

        示例：
            {"板块": ["sector_rotation"], "轮动": ["sector_rotation"],
             "情绪": ["sentiment_analysis"], "CANSLIM": ["canslim_screening"]}
        """
        keyword_to_tools: dict[str, list[str]] = defaultdict(list)
        for meta in cls._metadata.values():
            if not meta.enabled:
                continue
            for kw in meta.keywords:
                kw_lower = kw.lower()
                if meta.name not in keyword_to_tools[kw_lower]:
                    keyword_to_tools[kw_lower].append(meta.name)
        return dict(keyword_to_tools)

    @classmethod
    def match_by_keywords(cls, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        """
        关键词匹配：根据查询文本匹配最相关的工具

        替代你现有 Router 的关键词路由逻辑。
        返回 [(工具名, 匹配得分), ...] 按得分降序。

        Parameters
        ----------
        query : 用户查询文本
        top_k : 返回前 K 个匹配结果

        匹配算法：
        1. 统计每个工具的关键词在 query 中出现的次数
        2. 加入 priority 加权
        3. 返回得分最高的工具
        """
        query_lower = query.lower()
        scores: dict[str, float] = defaultdict(float)

        for meta in cls._metadata.values():
            if not meta.enabled:
                continue
            hit_count = 0
            for kw in meta.keywords:
                if kw.lower() in query_lower:
                    hit_count += 1
            if hit_count > 0:
                # 得分 = 命中关键词数 / 总关键词数 + priority 加权
                coverage = hit_count / max(len(meta.keywords), 1)
                scores[meta.name] = coverage + meta.priority * 0.01

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked[:top_k]

    # ─────────────────────────────────────────
    # Eval 层集成
    # ─────────────────────────────────────────

    @classmethod
    def get_tool_names(cls, category: str = None, enabled_only: bool = True) -> list[str]:
        """获取工具名列表（给 Eval 层遍历用）"""
        result = []
        for name, meta in cls._metadata.items():
            if enabled_only and not meta.enabled:
                continue
            if category and meta.category != category:
                continue
            result.append(name)
        return result

    @classmethod
    def get_tool_meta(cls, name: str) -> Optional[ToolMeta]:
        """获取工具元数据"""
        return cls._metadata.get(name)

    @classmethod
    def get_all_metadata(cls, category: str = None) -> list[ToolMeta]:
        """获取所有工具元数据"""
        metas = list(cls._metadata.values())
        if category:
            metas = [m for m in metas if m.category == category]
        return sorted(metas, key=lambda m: (m.category, -m.priority))

    @classmethod
    def get_tool_descriptions_for_eval(cls) -> dict[str, str]:
        """
        给 Eval 层用的 {工具名: 描述} 字典
        Eval 可以用这个来生成测试用例
        """
        return {
            name: meta.description
            for name, meta in cls._metadata.items()
            if meta.enabled
        }

    # ─────────────────────────────────────────
    # 启用/禁用
    # ─────────────────────────────────────────

    @classmethod
    def enable(cls, name: str):
        if name in cls._metadata:
            cls._metadata[name].enabled = True

    @classmethod
    def disable(cls, name: str):
        if name in cls._metadata:
            cls._metadata[name].enabled = False

    # ─────────────────────────────────────────
    # 统计与调试
    # ─────────────────────────────────────────

    @classmethod
    def summary(cls) -> dict:
        """工具注册统计"""
        from collections import Counter
        cats = Counter(m.category for m in cls._metadata.values() if m.enabled)
        total_keywords = sum(len(m.keywords) for m in cls._metadata.values() if m.enabled)
        return {
            "total_tools": len([m for m in cls._metadata.values() if m.enabled]),
            "disabled": len([m for m in cls._metadata.values() if not m.enabled]),
            "by_category": dict(cats),
            "total_keywords": total_keywords,
        }

    @classmethod
    def reset(cls):
        """清空所有注册（主要用于测试）"""
        cls._tools.clear()
        cls._metadata.clear()
        cls._event_log = None


# ═══════════════════════════════════════════
# 装饰器注册方式
# ═══════════════════════════════════════════

def tool(name: str, description: str, category: str = "analysis",
         keywords: list[str] = None, parameters: dict = None,
         examples: list[str] = None, priority: int = 0):
    """
    装饰器：注册函数为 ToolCenter 工具

    用法：
        @tool(
            name="sector_rotation",
            description="分析A股板块轮动状态",
            category="analysis",
            keywords=["板块", "轮动", "行业", "热点"],
            examples=["当前哪个板块最强", "板块轮动方向"],
        )
        def analyze_sector_rotation(query: str = "", **kwargs) -> str:
            ...
    """
    def decorator(func: Callable) -> Callable:
        ToolCenter.register(
            name=name,
            func=func,
            description=description,
            category=category,
            keywords=keywords,
            parameters=parameters,
            examples=examples,
            priority=priority,
        )

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    return decorator
