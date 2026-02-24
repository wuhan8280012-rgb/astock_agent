"""
llm_client.py — LLM 统一调用层
================================
支持 DeepSeek V3 / R1 双模型路由，兼容 OpenAI SDK 格式。

配置方式（三选一，优先级从高到低）：
  1. 环境变量: export DEEPSEEK_API_KEY=sk-xxx
  2. config.py: DEEPSEEK_API_KEY = "sk-xxx"
  3. 直接传参: LLMClient(api_key="sk-xxx")

用法：
  from llm_client import llm
  # 轻量任务（路由、分类、简单问答）
  result = llm.chat("今天市场情绪怎么样")
  # 深度分析（复盘、CANSLIM、板块轮动）
  result = llm.reason("请深度分析消费电子板块轮动...")
  # 带 system prompt
  result = llm.chat("...", system="你是A股量化投资AI Agent的路由器...")
  # 带完整 messages
  result = llm.call(messages=[...], model="deepseek-chat")

依赖：pip install openai
"""

import os
import time
from typing import Optional

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# ═══════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════

# DeepSeek API 配置
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# 模型名
MODEL_CHAT = "deepseek-chat"          # V3 — 轻量快速，适合路由/分类
MODEL_REASONER = "deepseek-reasoner"  # R1 — 深度推理，适合分析/复盘

# 默认参数
DEFAULT_MAX_TOKENS = 2000
DEFAULT_TEMPERATURE = 0.7

# 从 config.py 或环境变量读取 API Key
def _get_api_key() -> str:
    # 优先级 1: 环境变量
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if key:
        return key
    # 优先级 2: config.py
    try:
        from config import DEEPSEEK_API_KEY
        return DEEPSEEK_API_KEY
    except (ImportError, AttributeError):
        pass
    return ""


# ═══════════════════════════════════════════
# LLM 客户端
# ═══════════════════════════════════════════

class LLMClient:
    """
    DeepSeek LLM 统一调用客户端

    双模型路由：
    - chat()   → deepseek-chat (V3)    轻量任务
    - reason() → deepseek-reasoner (R1) 深度推理
    - call()   → 自定义模型和参数
    """

    def __init__(self, api_key: str = None, base_url: str = None):
        if not HAS_OPENAI:
            raise ImportError(
                "需要安装 openai 库: pip install openai\n"
                "DeepSeek API 兼容 OpenAI SDK 格式。"
            )

        self.api_key = api_key or _get_api_key()
        if not self.api_key:
            raise ValueError(
                "未找到 DeepSeek API Key。请通过以下方式之一配置：\n"
                "  1. 环境变量: export DEEPSEEK_API_KEY=sk-xxx\n"
                "  2. config.py: DEEPSEEK_API_KEY = 'sk-xxx'\n"
                "  3. 直接传参: LLMClient(api_key='sk-xxx')"
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url or DEEPSEEK_BASE_URL,
        )

        # 调用统计
        self._call_count = 0
        self._total_tokens = 0
        self._total_cost = 0.0

    # ─────────────────────────────────────────
    # 便捷方法
    # ─────────────────────────────────────────

    def chat(self, question: str, system: str = "",
             max_tokens: int = DEFAULT_MAX_TOKENS,
             temperature: float = DEFAULT_TEMPERATURE) -> str:
        """
        轻量对话（用 V3 模型）

        适用：路由分发、简单问答、分类、摘要
        成本：~$0.14/百万输入 token

        用法：
            result = llm.chat("今天市场情绪怎么样")
            result = llm.chat(question, system="你是路由器...")
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": question})

        return self.call(
            messages=messages,
            model=MODEL_CHAT,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def reason(self, question: str, system: str = "",
               max_tokens: int = 4000) -> str:
        """
        深度推理（用 R1 模型）

        适用：板块轮动分析、CANSLIM 深度筛选、盘后复盘、策略评估
        成本：~$0.55/百万输入 token
        注意：R1 会输出思维链（CoT），响应更慢但质量更高

        用法：
            result = llm.reason("深度分析消费电子板块当前轮动阶段...")
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": question})

        return self.call(
            messages=messages,
            model=MODEL_REASONER,
            max_tokens=max_tokens,
            temperature=0,  # R1 推理模式建议 temperature=0
        )

    # ─────────────────────────────────────────
    # 核心调用
    # ─────────────────────────────────────────

    def call(self, messages: list[dict], model: str = MODEL_CHAT,
             max_tokens: int = DEFAULT_MAX_TOKENS,
             temperature: float = DEFAULT_TEMPERATURE,
             retry: int = 2) -> str:
        """
        通用 LLM 调用

        Parameters
        ----------
        messages : [{"role": "system"|"user"|"assistant", "content": "..."}]
        model : "deepseek-chat" (V3) 或 "deepseek-reasoner" (R1)
        max_tokens : 最大输出 token 数
        temperature : 温度（R1 推理建议 0）
        retry : 失败重试次数

        Returns
        -------
        模型输出文本
        """
        last_error = None
        for attempt in range(retry + 1):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                # 统计
                self._call_count += 1
                usage = response.usage
                if usage:
                    self._total_tokens += (usage.prompt_tokens + usage.completion_tokens)
                    self._total_cost += self._estimate_cost(
                        model, usage.prompt_tokens, usage.completion_tokens
                    )

                # 提取文本
                content = response.choices[0].message.content
                return content.strip() if content else ""

            except Exception as e:
                last_error = e
                if attempt < retry:
                    wait = (attempt + 1) * 2
                    print(f"⚠️ LLM 调用失败 (尝试 {attempt+1}/{retry+1}): {e}")
                    print(f"   {wait}秒后重试...")
                    time.sleep(wait)

        raise ConnectionError(
            f"LLM 调用失败（已重试{retry}次）: {last_error}"
        )

    def call_for_router(self, messages: list[dict]) -> str:
        """
        给 Router 用的调用接口

        签名与你现有 Router 的 llm_caller 一致：
            (messages: list[dict]) -> str

        用法（在 router.py 中）：
            from llm_client import llm
            # 替换你原来的 LLM 调用
            response = llm.call_for_router(messages)
        """
        return self.call(
            messages=messages,
            model=MODEL_CHAT,  # 路由用轻量模型
            max_tokens=500,    # 路由输出很短
            temperature=0.3,   # 路由需要确定性
        )

    # ─────────────────────────────────────────
    # 统计
    # ─────────────────────────────────────────

    @staticmethod
    def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
        """估算本次调用成本（美元）"""
        if model == MODEL_REASONER:
            return input_tokens * 0.55 / 1_000_000 + output_tokens * 2.19 / 1_000_000
        else:
            return input_tokens * 0.14 / 1_000_000 + output_tokens * 0.28 / 1_000_000

    @property
    def stats(self) -> dict:
        return {
            "calls": self._call_count,
            "total_tokens": self._total_tokens,
            "estimated_cost_usd": round(self._total_cost, 6),
        }

    def print_stats(self):
        s = self.stats
        print(f"📊 LLM 统计: {s['calls']}次调用, "
              f"{s['total_tokens']}tokens, "
              f"≈${s['estimated_cost_usd']:.4f}")


# ═══════════════════════════════════════════
# 全局实例（懒加载）
# ═══════════════════════════════════════════

_llm_instance: Optional[LLMClient] = None


def get_llm(api_key: str = None) -> LLMClient:
    """获取全局 LLM 实例"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMClient(api_key=api_key)
    return _llm_instance

# 便捷别名
try:
    llm = get_llm()
except (ImportError, ValueError):
    llm = None  # openai 未安装或 key 未配置时不报错

