"""
路由Agent: 交互式问答，自动选择Skill执行

参考 Dexter 的核心设计:
  "决策空间要小" — LLM只看5个选项，不暴露所有Skill细节
  "探索者和回答者分离" — 路由LLM选Skill，回答LLM写答案

两种模式:
  1. Claude API 模式: 用LLM理解问题并路由 (需要API Key)
  2. 关键词模式: 用规则匹配路由 (无需API，离线可用)

用法:
  python router.py "今天市场情绪怎么样"
  python router.py "电子板块还能追吗"
  python router.py "我的持仓有风险吗"
  python router.py --interactive     # 交互模式
"""

import sys
import os
import json
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 11 个 Skill 全注册（final_system 总入口）
try:
    from final_system import register_all_skills
    register_all_skills()
except Exception:
    try:
        from skill_registry import register_all_skills
        register_all_skills()
    except Exception:
        pass

from data_fetcher import get_fetcher
from scratchpad import Scratchpad
from knowledge_base import KnowledgeBase, init_default_knowledge
from debate import DebateEngine, needs_debate

try:
    from cron_and_tools import ToolCenter, ContextCompactor, CronEngine
except Exception:
    ToolCenter = None
    ContextCompactor = None
    CronEngine = None

# EventLog + Brain（可选）
_event_log = None
try:
    from event_log import EventLog
    _event_log = EventLog(log_dir="data/event_log/")
except Exception:
    pass
try:
    from brain import get_brain
    _brain = get_brain(event_log=_event_log)
except Exception:
    _brain = None

# Skills
from skills.sentiment import SentimentSkill
from skills.sector_rotation import SectorRotationSkill
from skills.sector_stage_filter import SectorStageFilter
from skills.macro_monitor import MacroSkill
from skills.risk_control import RiskControlSkill

try:
    from config import CLAUDE_API_KEY, CLAUDE_MODEL
except ImportError:
    CLAUDE_API_KEY = ""
    CLAUDE_MODEL = "claude-sonnet-4-20250514"

# DeepSeek（优先于 Claude）
try:
    from llm_client import get_llm
    _llm = get_llm()
except Exception:
    _llm = None


# ============================================================
# 路由表: 5个决策选项 (Dexter: 缩小决策空间)
# ============================================================
ROUTES = {
    "market_sentiment": {
        "name": "市场情绪+大盘方向",
        "description": "分析当前A股市场情绪、涨跌比、北向资金、成交额水平，给出仓位建议",
        "keywords": ["情绪", "大盘", "市场", "仓位", "今天", "行情", "恐慌", "贪婪",
                      "涨跌", "北向", "两融", "成交额", "缩量", "放量"],
        "skills": ["sentiment"],
    },
    "sector_analysis": {
        "name": "板块轮动+行业分析",
        "description": "分析哪些板块强势、哪些在缩量整理、轮动信号",
        "keywords": ["板块", "行业", "轮动", "强势", "电子", "计算机", "医药", "新能源",
                      "AI", "半导体", "消费", "金融", "周期", "哪个板块", "追"],
        "skills": ["sector_rotation"],
    },
    "sector_stage": {
        "name": "板块Stage联合过滤",
        "description": "板块相对强度+黑名单+底部/收紧/R:R过滤",
        "keywords": ["板块轮动", "板块分析", "强势板块", "资金流向", "选股", "底部", "大底", "Stage", "stage", "哪些板块", "板块排名", "轮动"],
        "skills": ["sector_rotation", "sector_stage"],
    },
    "macro_liquidity": {
        "name": "宏观流动性",
        "description": "分析SHIBOR利率、北向资金趋势、两融杠杆、成交额趋势，判断流动性环境",
        "keywords": ["宏观", "流动性", "利率", "SHIBOR", "资金面", "宽松", "收紧",
                      "央行", "MLF", "逆回购", "杠杆"],
        "skills": ["macro"],
    },
    "risk_check": {
        "name": "持仓风控诊断",
        "description": "检查当前持仓的止损、仓位集中度、板块暴露，给出风控建议",
        "keywords": ["持仓", "风控", "风险", "止损", "仓位", "减仓", "加仓",
                      "该不该卖", "该不该买", "亏损", "盈利", "危险"],
        "skills": ["sentiment", "risk"],
    },
    "stock_analysis": {
        "name": "个股价值分析",
        "description": "对个股做价值投资分析：ROE、负债率、现金流、成长性、估值、护城河六维评分",
        "keywords": ["分析", "怎么样", "能买吗", "价值", "ROE", "基本面", "茅台",
                      "估值", "财报", "护城河", "股票", "个股"],
        "skills": ["value"],
    },
    "knowledge_query": {
        "name": "历史模式+经验查询",
        "description": "从知识库中匹配历史相似行情，查找过去的经验教训",
        "keywords": ["历史", "之前", "上次", "经验", "模式", "类似", "见过",
                      "规律", "复盘", "教训"],
        "skills": ["knowledge"],
    },
}


class Router:
    """路由Agent"""

    def __init__(self, event_log=None, brain=None):
        self.fetcher = get_fetcher()
        self.sp = Scratchpad()
        self.kb = init_default_knowledge()
        self.llm = _llm
        self.use_llm = self._llm_available() or bool(CLAUDE_API_KEY)
        self.event_log = event_log if event_log is not None else _event_log
        self.brain = brain if brain is not None else _brain
        self.compactor = ContextCompactor(max_tokens=8000) if ContextCompactor else None
        self.debate_engine = DebateEngine(self.llm)

    def _llm_available(self) -> bool:
        if not self.llm:
            return False
        if hasattr(self.llm, "has_key"):
            try:
                return bool(self.llm.has_key())
            except Exception:
                return False
        return hasattr(self.llm, "chat")

    def answer(self, question: str) -> str:
        """增强版回答流程: 信息查询走快速路径，决策问题走辩论"""
        print(f"\n💬 问题: {question}")
        route_keys = self._route(question)
        route_names = []
        for k in route_keys:
            route_info = (ToolCenter.get_route(k) if ToolCenter else None) or ROUTES.get(k, {})
            route_names.append(route_info.get("name", k))
        print(f"🔀 路由: {' + '.join(route_names)}")

        # 写入 EventLog（若可用）
        try:
            log = self.event_log
            if not log:
                from event_log import EventLog
                log = EventLog(log_dir="data/event_log/")
            from event_log import EventType
            log.emit(EventType.ROUTER_DISPATCH, {
                "question": question,
                "route_keys": route_keys,
                "route_names": route_names,
            }, source="router")
        except Exception:
            pass

        skill_results = self._execute_skills(route_keys, question)
        brief_context = self._build_brief_context(skill_results)

        env_score = 65
        env_brief = ""
        if "market_environment" in skill_results:
            env_data = skill_results["market_environment"]
            if isinstance(env_data, dict):
                env_score = env_data.get("total_score", 65)
                env_brief = f"[环境] {env_score}/100({env_data.get('level', '?')})"

        if needs_debate(question, route_keys) and self._llm_available():
            risk_brief = ""
            risk_data = skill_results.get("risk")
            if isinstance(risk_data, dict) and "error" not in risk_data:
                total = float(risk_data.get("total_position", 0))
                limit = float(risk_data.get("max_position_limit", 0))
                alerts = risk_data.get("portfolio_alerts", [])
                risk_brief = f"[风控] 仓位{total*100:.0f}%/{limit*100:.0f}%上限 | 预警{len(alerts)}条"
            elif risk_data:
                risk_brief = str(risk_data)[:200]

            debate_result = self.debate_engine.run(
                question=question,
                context_brief=brief_context,
                env_score=env_score,
                env_brief=env_brief,
                risk_brief=risk_brief,
            )

            if self.brain:
                try:
                    self.brain.learn_from_output("debate", question, debate_result.to_dict())
                except Exception:
                    pass

            answer = self._format_debate_answer(question, debate_result, brief_context)
        else:
            if self.use_llm:
                answer = self.debate_engine.run_quick(question, brief_context)
            else:
                answer = self._answer_with_template(question, route_keys, skill_results)

        self.sp.log("router", output_data={
            "question": question,
            "routes": route_keys,
            "answer_preview": answer[:200],
        })

        return answer

    def _build_brief_context(self, skill_results: dict) -> str:
        """
        将所有 Skill 结果转为压缩上下文
        优先用 to_brief()，回退到 to_dict() 截断
        """
        parts = []
        for _, result in skill_results.items():
            if result is None:
                continue
            if hasattr(result, "to_brief"):
                parts.append(result.to_brief())
            elif isinstance(result, dict):
                if "total_score" in result and "scores" in result:
                    s = result["scores"]
                    parts.append(
                        f"[环境] {result['total_score']}/100({result.get('level','?')}) "
                        f"趋势{s.get('trend',0)} 情绪{s.get('sentiment',0)} "
                        f"量能{s.get('volume',0)} 板块{s.get('sector',0)}"
                    )
                else:
                    parts.append(json.dumps(result, ensure_ascii=False)[:200])
            else:
                parts.append(str(result)[:200])
        return "\n".join(parts)

    def _format_debate_answer(self, question: str, result, context: str) -> str:
        """格式化辩论结果为用户可读回答"""
        del question
        del context
        sections = []

        if result.env_gate == "拦截":
            sections.append(f"⛔ 环境门控: {result.verdict}")
            return "\n".join(sections)

        icon = {"买入": "🟢", "持有": "🔵", "减仓": "🟡", "卖出": "🔴", "观望": "⚪"}.get(result.action, "❓")
        sections.append(f"{icon} **决策: {result.action}** (置信度 {result.confidence}%)")
        sections.append(f"理由: {result.verdict}")
        sections.append(f"\n📈 看多要点:\n{result.bull_case}")
        sections.append(f"\n📉 看空要点:\n{result.bear_case}")

        if result.risk_perspectives:
            rp = result.risk_perspectives
            sections.append("\n🛡️ 风控三视角:")
            sections.append(f"  进攻: {rp.get('aggressive', 'N/A')}")
            sections.append(f"  均衡: {rp.get('neutral', 'N/A')}")
            sections.append(f"  防守: {rp.get('conservative', 'N/A')}")

        if result.env_gate == "警告":
            sections.append("\n⚠️ 环境偏弱，已降低置信度")

        t = result.token_usage
        sections.append(f"\n---\n🔧 辩论耗时{result.elapsed_sec:.1f}s tokens≈{t.get('input',0)+t.get('output',0)}")
        return "\n".join(sections)

    # ========================================================
    # 路由层
    # ========================================================
    def _route(self, question: str) -> List[str]:
        """根据问题选择路由"""
        if self.use_llm:
            return self._route_with_llm(question)
        return self._route_with_keywords(question)

    def _route_with_keywords(self, question: str) -> List[str]:
        """关键词匹配路由 (离线模式)，使用 ToolCenter 注册的路由表"""
        if not ToolCenter:
            return ["market_sentiment"]
        matches = ToolCenter.match_by_keywords(question, top_k=3)
        if matches:
            return matches
        return ["market_sentiment"]  # 默认

    def _route_with_llm(self, question: str) -> List[str]:
        """LLM路由 (DeepSeek 优先，否则 Claude)，工具列表来自 ToolCenter，注入 Brain 上下文"""
        try:
            tool_prompt = ToolCenter.get_router_prompt() if ToolCenter else ""
            brain_context = ""
            if self.brain:
                brain_context = self.brain.get_context_for_prompt(question)
            if brain_context:
                tool_prompt = tool_prompt + "\n\n" + brain_context

            content = f"""你是一个路由器。根据用户问题，选择1-2个最相关的分析模块。
只输出模块key，用逗号分隔，不要解释。

{tool_prompt}

问题: {question}
回答(只输出key):"""
            messages = [{"role": "user", "content": content}]

            if self._llm_available():
                text = self.llm.call_for_router(messages)
            else:
                import anthropic
                client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
                response = client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=100,
                    messages=[{"role": "user", "content": content}],
                )
                text = response.content[0].text.strip()

            keys = [k.strip() for k in text.split(",")]
            valid = [k for k in keys if (ToolCenter and ToolCenter.get_route(k)) or k in ROUTES]
            return valid if valid else ["market_sentiment"]

        except Exception as e:
            print(f"[Router] LLM路由失败: {e}，回退到关键词模式")
            return self._route_with_keywords(question)

    # ========================================================
    # 执行层
    # ========================================================
    def _execute_skills(self, route_keys: List[str], question: str = "") -> Dict:
        """执行路由选中的Skills（skill 列表来自 ToolCenter 或 ROUTES 回退）"""
        results = {}
        skills_to_run = set()

        for key in route_keys:
            r = (ToolCenter.get_route(key) if ToolCenter else None) or ROUTES.get(key)
            if r:
                for skill in r.get("skills", []):
                    skills_to_run.add(skill)

        for skill_name in skills_to_run:
            print(f"  ⚙️ 运行 {skill_name}...")
            try:
                if skill_name == "sentiment":
                    s = SentimentSkill()
                    r = s.analyze()
                    results["sentiment"] = r.to_dict()
                    self.sp.log("sentiment", output_data=r.to_dict())
                    if self.brain and r:
                        try:
                            self.brain.learn_from_output("sentiment", question, r.to_brief())
                        except Exception:
                            pass

                elif skill_name == "sector_rotation":
                    s = SectorRotationSkill()
                    r = s.analyze()
                    results["sector"] = r.to_dict()
                    self.sp.log("sector_rotation", output_data=r.to_dict())
                    if self.brain and r:
                        try:
                            self.brain.learn_from_output("sector_rotation", question, r.to_brief())
                        except Exception:
                            pass

                elif skill_name == "macro":
                    s = MacroSkill()
                    r = s.analyze()
                    results["macro"] = r.to_dict()
                    self.sp.log("macro", output_data=r.to_dict())
                    if self.brain and r:
                        try:
                            self.brain.learn_from_output("macro", question, r.to_brief())
                        except Exception:
                            pass

                elif skill_name == "sector_stage":
                    s = SectorStageFilter()
                    r = s.analyze(verbose=False)
                    results["sector_stage"] = r.to_dict()
                    self.sp.log("sector_stage", output_data=r.to_dict())
                    if self.brain and r:
                        try:
                            self.brain.learn_from_output("sector_stage", question, r.to_brief())
                        except Exception:
                            pass

                elif skill_name == "risk":
                    from pathlib import Path
                    pos_file = Path("./knowledge/positions.json")
                    if pos_file.exists():
                        with open(pos_file) as f:
                            positions = json.load(f)
                        sentiment_level = results.get("sentiment", {}).get("overall_level", "中性")
                        s = RiskControlSkill()
                        r = s.analyze(positions, sentiment_level, self.fetcher.get_latest_trade_date())
                        results["risk"] = r.to_dict()
                        self.sp.log("risk", output_data=r.to_dict())
                        if self.brain and r:
                            try:
                                self.brain.learn_from_output("risk", question, r.to_brief())
                            except Exception:
                                pass
                    else:
                        results["risk"] = {"error": "未配置持仓"}

                elif skill_name == "knowledge":
                    # 从其他结果中构建当前状态
                    state = {}
                    if "sentiment" in results:
                        state["sentiment"] = results["sentiment"].get("overall_level", "")
                    if "macro" in results:
                        state["liquidity"] = results["macro"].get("liquidity_level", "")
                    matches = self.kb.match_pattern(state)
                    lessons = self.kb.get_lessons()
                    results["knowledge"] = {
                        "pattern_matches": matches[:3],
                        "lessons": [l["lesson"] for l in lessons[:5]],
                    }
                    if self.brain:
                        self.brain.learn_from_output("knowledge", question, results["knowledge"])

                elif skill_name == "value":
                    from value_investor import ValueInvestorSkill
                    # 从问题中提取股票代码（简单匹配）
                    code = self._extract_stock_code(question)
                    if code:
                        v = ValueInvestorSkill(self.fetcher)
                        r = v.analyze(code)
                        r_dict = r if isinstance(r, dict) else getattr(r, "to_dict", lambda: r)()
                        results["value"] = r_dict
                        self.sp.log("value_investor", output_data=r_dict)
                        if self.brain and r:
                            self.brain.learn_from_output("value", question, r)
                    else:
                        results["value"] = {"error": "未识别到股票代码，请用格式如 600519.SH"}

                elif skill_name == "trade_signals":
                    from trade_signals import TradeSignalGenerator
                    from market_environment import MarketEnvironment
                    env = MarketEnvironment(self.fetcher)
                    gen = TradeSignalGenerator(
                        data_fetcher=self.fetcher,
                        market_env=env,
                        event_log=self.event_log,
                        brain=self.brain,
                    )
                    out = gen.scan_all()
                    summary = out.get("summary", str(out))
                    results["trade_signals"] = {"summary": summary, "signals": out.get("signals", [])}
                    self.sp.log("trade_signals", output_data=out)
                    if self.brain and out:
                        self.brain.learn_from_output("trade_signals", question, summary)

                elif skill_name == "market_environment":
                    from market_environment import MarketEnvironment
                    env = MarketEnvironment(self.fetcher)
                    out = env.evaluate()
                    results["market_environment"] = out
                    self.sp.log("market_environment", output_data=out)
                    if self.brain and out:
                        self.brain.learn_from_output("market_environment", question, out.get("summary", ""))

            except Exception as e:
                print(f"    ❌ {skill_name}失败: {e}")
                results[skill_name] = {"error": str(e)}

        return results

    # ========================================================
    # 回答层 (探索者和回答者分离)
    # ========================================================
    def _answer_with_template(self, question: str, route_keys: List[str],
                               results: Dict) -> str:
        """模板回答 (离线模式)"""
        parts = []
        parts.append(f"📊 **分析结果** ({self.fetcher.get_latest_trade_date()})\n")

        if "sentiment" in results and "error" not in results.get("sentiment", {}):
            s = results["sentiment"]
            parts.append(f"**市场情绪: {s.get('overall_level', '?')}** "
                         f"(得分 {s.get('overall_score', 0):+.1f})")
            parts.append(f"仓位建议: {s.get('suggested_position', '?')}")
            for sig in s.get("signals", []):
                parts.append(f"  • {sig['name']}: {sig['level']} — {sig['detail']}")
            parts.append("")

        if "sector" in results and "error" not in results.get("sector", {}):
            s = results["sector"]
            top = s.get("top_sectors", [])
            if top:
                names = [f"{t['name']}({t['ret_20d']:+.1f}%)" for t in top[:5]]
                parts.append(f"**强势板块:** {', '.join(names)}")
            for sig in s.get("rotation_signals", []):
                parts.append(f"  → {sig}")
            parts.append("")

        if "macro" in results and "error" not in results.get("macro", {}):
            m = results["macro"]
            parts.append(f"**流动性: {m.get('liquidity_level', '?')}**")
            parts.append(f"影响: {m.get('market_impact', '')}")
            parts.append("")

        if "sector_stage" in results and "error" not in results.get("sector_stage", {}):
            st = results["sector_stage"]
            parts.append(f"**Stage过滤:** {st.get('summary', '')}")
            leaders = st.get("leaders", [])
            if leaders:
                parts.append("  领涨: " + ", ".join([x.get("name", "") for x in leaders[:3]]))
            blacks = st.get("blacklisted", [])
            if blacks:
                parts.append("  黑名单: " + ", ".join([x.get("name", "") for x in blacks[:3]]))
            parts.append("")

        if "risk" in results:
            r = results["risk"]
            if "error" in r:
                parts.append(f"**风控:** {r['error']}")
            else:
                parts.append(f"**风控:** {r.get('summary', '')}")
                for a in r.get("actions", []):
                    parts.append(f"  {a}")
            parts.append("")

        if "value" in results:
            v = results["value"]
            if "error" in v:
                parts.append(f"**个股分析:** {v['error']}")
            else:
                parts.append(f"**💎 价值分析: {v.get('name','')} ({v.get('ts_code','')})** "
                             f"评级: {v.get('grade','?')} | 总分: {v.get('total_score',0):.1f}")
                if v.get("disqualified"):
                    parts.append(f"  ❌ {v.get('disqualify_reason','')}")
                else:
                    for d in v.get("dimensions", []):
                        parts.append(f"  • {d['name']}: {d['level']} ({d['score']:.1f}/{d.get('max_score', d['score']):.0f}) — {d['detail']}")
                    for p in v.get("investment_points", []):
                        parts.append(f"  {p}")
                    for r in v.get("risk_flags", []):
                        parts.append(f"  🚨 {r}")
            parts.append("")

        if "trade_signals" in results and "error" not in results.get("trade_signals", {}):
            t = results["trade_signals"]
            parts.append(f"**📊 交易信号:** {t.get('summary', '')}")
            for s in (t.get("signals") or [])[:10]:
                parts.append(f"  • {s.get('name','')} ({s.get('symbol','')}) {s.get('action','')} 得分{s.get('score',0)}")
            parts.append("")

        if "market_environment" in results and "error" not in results.get("market_environment", {}):
            m = results["market_environment"]
            parts.append(f"**🌡 市场环境:** {m.get('summary', m.get('advice', ''))}")
            parts.append("")

        if "knowledge" in results:
            k = results["knowledge"]
            matches = k.get("pattern_matches", [])
            if matches:
                parts.append("**历史模式匹配:**")
                for m in matches:
                    parts.append(f"  📌 {m['name']} (匹配{m['match_score']:.0%}) "
                                 f"→ {m['typical_outcome']}")
            parts.append("")

        return "\n".join(parts)

    def _extract_stock_code(self, question: str) -> Optional[str]:
        """从问题中提取股票代码"""
        import re
        # 匹配 600519.SH / 000858.SZ 格式
        m = re.search(r'(\d{6}\.[A-Z]{2})', question)
        if m:
            return m.group(1)

        # 匹配纯6位数字，自动补后缀
        m = re.search(r'(\d{6})', question)
        if m:
            code = m.group(1)
            if code.startswith(("6", "9")):
                return f"{code}.SH"
            elif code.startswith(("0", "3")):
                return f"{code}.SZ"
            elif code.startswith(("4", "8")):
                return f"{code}.BJ"

        # 匹配常见股票名
        name_map = {
            "茅台": "600519.SH", "贵州茅台": "600519.SH",
            "五粮液": "000858.SZ",
            "招商银行": "600036.SH", "招行": "600036.SH",
            "比亚迪": "002594.SZ",
            "宁德时代": "300750.SZ",
            "腾讯": None,  # 港股不支持
        }
        for name, code in name_map.items():
            if name in question and code:
                return code

        return None

    def _answer_with_llm(self, question: str, results: Dict) -> str:
        """LLM回答 (DeepSeek 优先，否则 Claude)；过长时用 Compactor 截断"""
        try:
            data_str = json.dumps(results, ensure_ascii=False, indent=2, default=str)
            if len(data_str) > 8000:
                data_str = data_str[:6000] + "\n\n...(数据过长已截断)"
            user_content = f"""你是A股投研助手。用户问了一个问题，以下是各分析模块的实时数据。
请直接回答用户的问题，要求：
1. 先给结论，再给依据
2. 用数据说话，不要空话
3. 如果涉及操作建议，必须附带风险提示
4. 控制在300字以内

用户问题: {question}

分析数据:
{data_str}"""

            if self._llm_available():
                reply = self.llm.chat(
                    user_content,
                    system="你是A股投研助手，根据数据简洁回答。",
                )
                return reply or ""
            import anthropic
            client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1500,
                messages=[{"role": "user", "content": user_content}],
            )
            return response.content[0].text

        except Exception as e:
            print(f"[Router] LLM回答失败: {e}，回退到模板")
            return self._answer_with_template(question, [], results)


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="A股智能问答")
    parser.add_argument("question", nargs="?", default=None, help="问题")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互模式")
    args = parser.parse_args()

    router = Router(event_log=_event_log, brain=_brain)
    try:
        from auto_journal import AutoJournal
        journal = AutoJournal(brain=_brain, event_log=_event_log)
        on_result = journal.on_cron_result
    except Exception:
        on_result = None
    cron = CronEngine(
        event_log=_event_log,
        brain=_brain,
        ai_executor=lambda prompt, tools: router.answer(prompt),
        on_result=on_result,
    )

    if args.interactive:
        print(f"\n{'='*50}")
        print(f"  🤖 A股智能问答 (输入 q 退出)")
        print(f"  示例:")
        print(f"    今天市场情绪怎么样？")
        print(f"    电子板块还能追吗？")
        print(f"    我的持仓有风险吗？")
        print(f"    有没有类似历史行情？")
        print(f"{'='*50}\n")

        while True:
            try:
                q = input("你: ").strip()
                if q.lower() in ["q", "quit", "exit", "退出"]:
                    print("再见！")
                    break
                if not q:
                    continue
                answer = router.answer(q)
                print(f"\n{answer}")
            except KeyboardInterrupt:
                print("\n再见！")
                break
            except Exception as e:
                print(f"出错: {e}")
    else:
        if not args.question:
            print("用法: python router.py '你的问题'")
            print("或:   python router.py -i  (交互模式)")
            return
        answer = router.answer(args.question)
        print(f"\n{answer}")


if __name__ == "__main__":
    main()
