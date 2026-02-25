"""
debate.py — 多空辩论引擎 v1.0
================================
灵感来源: TradingAgents (UCLA/MIT, arXiv:2412.20138v7)
核心改造: 将论文的多 Agent 架构压缩为单 LLM 多角色 prompt，
         3次调用完成完整的多空对抗+裁决，成本 ≤ ¥0.03/次。
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class DebateResult:
    """辩论结果"""
    question: str
    bull_case: str = ""
    bear_case: str = ""
    verdict: str = ""
    action: str = "观望"  # "买入" / "持有" / "减仓" / "卖出" / "观望"
    confidence: int = 50
    risk_perspectives: Dict[str, str] = field(default_factory=dict)
    token_usage: Dict[str, int] = field(default_factory=dict)
    elapsed_sec: float = 0.0
    env_gate: str = ""     # "通过" / "警告" / "拦截"

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "bull_case": self.bull_case,
            "bear_case": self.bear_case,
            "verdict": self.verdict,
            "action": self.action,
            "confidence": self.confidence,
            "risk_perspectives": self.risk_perspectives,
            "env_gate": self.env_gate,
            "token_usage": self.token_usage,
            "elapsed_sec": round(self.elapsed_sec, 2),
        }

    def to_brief(self) -> str:
        """供 Brain 记录的压缩版"""
        return (
            f"[辩论] {self.action}(置信{self.confidence}%) "
            f"环境:{self.env_gate} | "
            f"多方要点:{self.bull_case[:60]}... | "
            f"空方要点:{self.bear_case[:60]}..."
        )


BULL_PROMPT = """你是一位A股看多分析师。你的职责是基于给定数据，尽力构建最强的看多论据。

## 当前市场数据
{context}

## 用户问题
{question}

## 你的任务
1. 找出数据中所有支持看多的信号
2. 构建一个有说服力的看多理由（3-5个要点）
3. 给出看多情景下的目标和逻辑

请用 JSON 格式回答:
{{"bull_points": ["要点1", "要点2", ...], "target_scenario": "看多情景描述", "confidence": 0-100}}

注意: 你只负责看多，不需要平衡观点。尽全力找出看多理由。"""

BEAR_PROMPT = """你是一位A股看空/风控分析师。你的职责是反驳看多论据，找出所有风险。

## 当前市场数据
{context}

## 用户问题
{question}

## 看多方观点
{bull_case}

## 你的任务
1. 逐条反驳看多要点中的薄弱环节
2. 找出数据中被看多方忽略的风险信号
3. 指出最大的下行风险

请用 JSON 格式回答:
{{"bear_points": ["风险1", "风险2", ...], "rebuttals": ["反驳1", ...], "worst_case": "最坏情景", "confidence": 0-100}}

注意: 你只负责看空和风险，不需要平衡观点。尽全力找出风险。"""

JUDGE_PROMPT = """你是投资委员会主席。你刚听完多空双方的辩论，需要做出最终裁决。

## 当前市场数据
{context}

## 市场环境评分
{env_brief}

## 用户问题
{question}

## 看多方论据
{bull_case}

## 看空方论据
{bear_case}

## 风控三视角
{risk_perspectives}

## 你的裁决任务
综合多空双方观点和风控意见，给出最终投资决策。

关键原则:
- 环境评分 < 60 时，除非有极强理由，否则倾向观望
- 永远把风控放在第一位
- 不确定时选择保守

请用 JSON 格式回答:
{{
  "action": "买入/持有/减仓/卖出/观望",
  "confidence": 0-100,
  "reasoning": "裁决理由（2-3句话）",
  "bull_accepted": ["采纳的多方观点"],
  "bear_accepted": ["采纳的空方观点"],
  "key_risk": "最需要关注的风险",
  "position_advice": "仓位建议"
}}"""

RISK_PERSPECTIVE_PROMPT = """基于以下风控数据，分别从三个角度给出简短意见（每个 1-2 句话）:

## 风控数据
{risk_brief}

请用 JSON 格式回答:
{{
  "aggressive": "进攻型观点: 机会在哪里，可以承受什么风险",
  "neutral": "均衡型观点: 当前仓位是否合理",
  "conservative": "防守型观点: 底线在哪里，什么情况必须减仓"
}}"""


class DebateEngine:
    """多空辩论引擎"""

    ENV_GATE_BLOCK = 50
    ENV_GATE_CAUTION = 60

    def __init__(self, llm_client, config: dict = None):
        self.llm = llm_client
        self.config = config or {}
        self.max_tokens_per_call = self.config.get("max_tokens", 500)

    def run(
        self,
        question: str,
        context_brief: str,
        env_score: int = 65,
        env_brief: str = "",
        risk_brief: str = "",
    ) -> DebateResult:
        t0 = time.time()
        total_tokens = {"input": 0, "output": 0}
        result = DebateResult(question=question)

        if env_score < self.ENV_GATE_BLOCK:
            result.env_gate = "拦截"
            result.action = "观望"
            result.confidence = 90
            result.verdict = (
                f"市场环境评分{env_score}/100，低于安全阈值{self.ENV_GATE_BLOCK}，"
                f"触发环境门控，暂停交易决策。"
            )
            result.bull_case = "环境门控拦截，未发起辩论"
            result.bear_case = "环境门控拦截，未发起辩论"
            result.elapsed_sec = time.time() - t0
            return result

        result.env_gate = "通过" if env_score >= self.ENV_GATE_CAUTION else "警告"

        bull_prompt = BULL_PROMPT.format(context=context_brief, question=question)
        bull_raw = self._call_llm(bull_prompt, total_tokens)
        bull_data = self._parse_json(bull_raw)
        result.bull_case = self._format_bull(bull_data)

        bear_prompt = BEAR_PROMPT.format(
            context=context_brief,
            question=question,
            bull_case=result.bull_case,
        )
        bear_raw = self._call_llm(bear_prompt, total_tokens)
        bear_data = self._parse_json(bear_raw)
        result.bear_case = self._format_bear(bear_data)

        risk_perspectives_str = "暂无详细风控数据"
        if risk_brief and risk_brief != "[风控] 数据不足":
            risk_prompt = RISK_PERSPECTIVE_PROMPT.format(risk_brief=risk_brief)
            risk_raw = self._call_llm(risk_prompt, total_tokens)
            risk_data = self._parse_json(risk_raw)
            if isinstance(risk_data, dict):
                result.risk_perspectives = risk_data
                risk_perspectives_str = json.dumps(risk_data, ensure_ascii=False)

        judge_prompt = JUDGE_PROMPT.format(
            context=context_brief,
            env_brief=env_brief or f"环境评分: {env_score}/100",
            question=question,
            bull_case=result.bull_case,
            bear_case=result.bear_case,
            risk_perspectives=risk_perspectives_str,
        )
        judge_raw = self._call_llm(judge_prompt, total_tokens)
        judge_data = self._parse_json(judge_raw)

        if isinstance(judge_data, dict):
            result.action = judge_data.get("action", "观望")
            result.confidence = int(judge_data.get("confidence", 50))
            result.verdict = judge_data.get("reasoning", judge_raw)
        else:
            result.action = "观望"
            result.confidence = 30
            result.verdict = f"裁决解析失败，原始回复: {judge_raw[:200]}"

        if result.env_gate == "警告":
            result.confidence = min(result.confidence, 60)
            result.verdict += f" [环境警告: 评分{env_score}偏低，已降低置信度]"

        result.token_usage = total_tokens
        result.elapsed_sec = time.time() - t0
        return result

    def run_quick(self, question: str, context_brief: str) -> str:
        prompt = f"基于以下A股市场数据，简要回答问题。\n\n{context_brief}\n\n问题: {question}"
        return self._call_llm(prompt, {})

    def _call_llm(self, prompt: str, token_counter: dict) -> str:
        try:
            if hasattr(self.llm, "chat"):
                resp = self.llm.chat(prompt, max_tokens=self.max_tokens_per_call)
            elif callable(self.llm):
                resp = self.llm(prompt)
            else:
                return '{"error": "LLM client interface not recognized"}'

            if isinstance(token_counter, dict):
                token_counter["input"] = token_counter.get("input", 0) + len(prompt) // 2
                token_counter["output"] = token_counter.get("output", 0) + len(str(resp)) // 2

            return str(resp).strip()
        except Exception as e:
            return f'{{"error": "{str(e)}"}}'

    def _parse_json(self, raw: str) -> Any:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            return text

    def _format_bull(self, data) -> str:
        if isinstance(data, dict):
            points = data.get("bull_points", [])
            target = data.get("target_scenario", "")
            conf = data.get("confidence", "?")
            lines = [f"看多置信度: {conf}%"]
            for i, p in enumerate(points, 1):
                lines.append(f"  {i}. {p}")
            if target:
                lines.append(f"  目标情景: {target}")
            return "\n".join(lines)
        return str(data)[:500]

    def _format_bear(self, data) -> str:
        if isinstance(data, dict):
            points = data.get("bear_points", [])
            rebuttals = data.get("rebuttals", [])
            worst = data.get("worst_case", "")
            conf = data.get("confidence", "?")
            lines = [f"看空置信度: {conf}%"]
            for i, p in enumerate(points, 1):
                lines.append(f"  风险{i}. {p}")
            if rebuttals:
                lines.append("  反驳:")
                for r in rebuttals:
                    lines.append(f"    - {r}")
            if worst:
                lines.append(f"  最坏情景: {worst}")
            return "\n".join(lines)
        return str(data)[:500]


def needs_debate(question: str, route_keys: list) -> bool:
    """判断问题是否需要走辩论流程"""
    import re

    action_keywords = [
        "买", "卖", "加仓", "减仓", "建仓", "清仓",
        "止损", "止盈", "入场", "出场", "调仓",
        "能不能", "该不该", "要不要", "值不值",
        "现在可以", "是否应该", "适合",
    ]
    for kw in action_keywords:
        if kw in question:
            return True

    decision_routes = {"stock_analysis", "risk_check", "trade_signal"}
    if set(route_keys) & decision_routes:
        return True

    if re.search(r"\d{6}\.[SZ][HZ]", question):
        return True

    return False
