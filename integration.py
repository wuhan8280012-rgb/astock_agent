"""
集成入口 — 将所有新模块串联到现有 Agent 系统
=============================================

这个文件演示如何将 EventLog、Brain、CronEngine、ToolCenter、ContextCompactor
集成到你现有的 A股 Agent 系统中（含 Router LLM、Scratchpad、Eval 层）。

核心改动点：
1. Router LLM → 从 ToolCenter 动态获取工具列表（而非硬编码）
2. Scratchpad → 通过 EventLog 兼容层平滑迁移
3. 每次 LLM 调用前 → 自动注入 Brain 认知上下文
4. 系统启动时 → 注册 CronEngine 定时任务
5. 长对话 → 自动触发 ContextCompactor 压缩
"""

from event_log import EventLog, EventType
from brain import AgentBrain
from cron_and_tools import CronEngine, ToolCenter, ContextCompactor


def create_agent_system():
    """
    系统组装入口 — 类似 OpenAlice 的 main.ts composition root

    在你现有代码的入口处调用此函数，初始化所有新模块
    """

    # ═══════════════════════════════════════
    # 1. 初始化基础设施
    # ═══════════════════════════════════════

    event_log = EventLog(log_dir="data/event_log/")
    brain = AgentBrain(brain_dir="data/brain/", event_log=event_log)
    compactor = ContextCompactor(max_tokens=8000)

    # ═══════════════════════════════════════
    # 2. 注册你现有的 Skills 到 ToolCenter
    # ═══════════════════════════════════════

    # 替代你 Router 中硬编码的 skill 列表
    # 每个 skill 模块只需要在自己的 __init__ 中调用 ToolCenter.register()

    # ---- 你的现有 skills ----
    ToolCenter.register(
        name="sentiment_analysis",
        func=None,  # 替换为你的实际函数: analyze_sentiment
        description="分析A股市场情绪：新闻舆情、资金流向、技术指标情绪综合评分",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "情绪分析的具体问题"},
            }
        },
        category="analysis",
    )

    ToolCenter.register(
        name="sector_rotation",
        func=None,  # 替换为: analyze_sector_rotation
        description="分析A股板块轮动：板块强弱排名、轮动方向、热点切换信号",
        parameters={
            "type": "object",
            "properties": {
                "focus_sectors": {"type": "array", "items": {"type": "string"},
                                  "description": "重点关注的板块列表（可选）"},
            }
        },
        category="analysis",
    )

    ToolCenter.register(
        name="canslim_screening",
        func=None,  # 替换为: run_canslim_screen
        description="CANSLIM选股：按C/A/N/S/L/I/M七因子筛选A股标的",
        parameters={
            "type": "object",
            "properties": {
                "min_score": {"type": "number", "description": "最低综合得分"},
            }
        },
        category="analysis",
    )

    ToolCenter.register(
        name="backtest",
        func=None,  # 替换为: run_backtest
        description="策略回测：对指定策略和标的进行历史回测",
        category="analysis",
    )

    # ---- 新增 skills（借助 Brain 模块） ----
    ToolCenter.register(
        name="brain_recall",
        func=lambda query_tags, top_k=5: brain.recall_for_prompt(query_tags, top_k=top_k),
        description="召回历史交易经验和市场洞察",
        parameters={
            "type": "object",
            "properties": {
                "query_tags": {"type": "array", "items": {"type": "string"}},
                "top_k": {"type": "integer", "default": 5},
            }
        },
        category="brain",
    )

    ToolCenter.register(
        name="brain_remember",
        func=lambda **kwargs: str(brain.remember(**kwargs)),
        description="存储新的市场洞察到长期记忆",
        category="brain",
    )

    ToolCenter.register(
        name="brain_emotion",
        func=lambda: brain.get_emotion_for_prompt(),
        description="获取当前市场情绪判断",
        category="brain",
    )

    ToolCenter.register(
        name="brain_commit",
        func=lambda **kwargs: str(brain.commit_decision(**kwargs)),
        description="提交交易决策（含完整推理链）",
        category="brain",
    )

    # ═══════════════════════════════════════
    # 3. 升级 Router LLM
    # ═══════════════════════════════════════

    def enhanced_router(user_query: str, conversation_history: list[dict] = None) -> str:
        """
        增强版 Router — 替代你现有的 Router LLM / 关键词路由

        改进点：
        1. 工具列表从 ToolCenter 动态获取（而非硬编码）
        2. 自动注入 Brain 认知上下文
        3. 自动压缩过长的上下文
        4. 通过 EventLog 记录路由决策
        """

        # a) 从 ToolCenter 获取工具描述
        tool_prompt = ToolCenter.get_router_prompt()

        # b) 从 Brain 获取认知上下文
        # 从 query 中提取关键词作为 tags（简化版）
        import re
        potential_tags = re.findall(r'[\u4e00-\u9fff]+', user_query)
        brain_context = brain.get_context_for_prompt(
            query_tags=potential_tags[:5],
            include_emotion=True,
            include_memories=True,
            include_recent_commits=True,
            memory_top_k=3,
            commit_limit=2,
        )

        # c) 构建 Router prompt
        router_system_prompt = f"""你是一个A股量化投资AI Agent的路由器。
根据用户的问题，选择最合适的工具来处理。

{tool_prompt}

{brain_context}

请分析用户问题，返回应该调用的工具名称和参数。
如果需要多个工具，按执行顺序列出。"""

        # d) 上下文压缩
        messages = [{"role": "system", "content": router_system_prompt}]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_query})

        messages = compactor.compact(messages, preserve_recent=5)

        # e) 调用 LLM（这里替换为你实际的 LLM 调用）
        # response = call_llm(messages)

        # f) 记录路由决策到 EventLog
        event_log.emit(EventType.ROUTER_DISPATCH, {
            "query": user_query,
            "brain_context_injected": bool(brain_context),
            "context_compressed": len(messages) < len(conversation_history or []) + 2,
            # "routed_to": response.tool_name,  # 解析 LLM 返回后填入
        }, source="router")

        # 返回 LLM 响应（这里是占位）
        return "router_response_placeholder"

    # ═══════════════════════════════════════
    # 4. 设置 EventLog 订阅（模块间联动）
    # ═══════════════════════════════════════

    # 买入信号 → 自动触发风控检查
    def on_buy_signal(event):
        symbol = event.payload.get("symbol", "")
        print(f"📈 买入信号触发风控检查: {symbol}")
        # risk_result = ToolCenter.execute("risk_check", symbol=symbol)

    event_log.subscribe(EventType.SIGNAL_BUY, on_buy_signal)

    # 风控告警 → 自动推送通知（未来接微信/钉钉）
    def on_risk_alert(event):
        print(f"🚨 风控告警: {event.payload}")
        # send_wechat_notification(event.payload)

    event_log.subscribe_pattern("risk.*", on_risk_alert)

    # 决策提交 → 自动存入 Brain 记忆
    def on_decision_committed(event):
        symbol = event.payload.get("symbol", "")
        action = event.payload.get("action", "")
        signals = event.payload.get("signals", [])
        print(f"📝 决策已提交: {action} {symbol} (信号: {signals})")

    event_log.subscribe(EventType.BRAIN_DECISION_COMMITTED, on_decision_committed)

    # ═══════════════════════════════════════
    # 5. 初始化 CronEngine
    # ═══════════════════════════════════════

    def ai_executor(prompt: str, tools: list[str]) -> str:
        """AI 执行器 — 替换为你实际的 LLM 调用"""
        # messages = [{"role": "user", "content": prompt}]
        # response = call_llm(messages, tools=tools)
        # return response.content
        return f"[模拟] 执行了需要 {tools} 的任务"

    cron = CronEngine(
        jobs_file="data/cron/jobs.json",
        event_log=event_log,
        brain=brain,
        ai_executor=ai_executor,
    )

    # ═══════════════════════════════════════
    # 6. Eval 层增强（与 EventLog 集成）
    # ═══════════════════════════════════════

    def enhanced_eval(skill_name: str, test_query: str, expected: str, actual: str) -> dict:
        """
        增强版 Eval — 将评估结果写入 EventLog

        你现有的 Eval 层逻辑不变，只是加了 EventLog 记录
        """
        # 你现有的 eval 逻辑
        accuracy = 0.0  # 替换为实际的准确率计算
        passed = accuracy > 0.7

        result = {
            "skill": skill_name,
            "accuracy": accuracy,
            "passed": passed,
            "test_query": test_query,
        }

        # 写入 EventLog
        event_log.log_eval(skill_name, accuracy, result)

        # 如果准确率下降，存入 Brain 作为教训
        if not passed:
            brain.remember(
                key=f"eval_fail_{skill_name}_{int(time.time())}",
                category="strategy_note",
                content=f"{skill_name} 准确率下降到 {accuracy:.0%}，需要检查和调优。"
                        f"测试查询: {test_query}",
                confidence=0.8,
                tags=[skill_name, "eval", "准确率下降"],
                source="eval_layer",
                expiry_days=30,
            )

        return result

    # ═══════════════════════════════════════
    # 7. 返回系统组件（供外部使用）
    # ═══════════════════════════════════════

    return {
        "event_log": event_log,
        "brain": brain,
        "cron": cron,
        "compactor": compactor,
        "router": enhanced_router,
        "eval": enhanced_eval,
    }


# ═══════════════════════════════════════════
# 典型工作流示例
# ═══════════════════════════════════════════

def example_trading_workflow():
    """
    一个完整的交易日工作流示例

    展示所有模块如何协同工作
    """
    import time

    system = create_agent_system()
    event_log = system["event_log"]
    brain = system["brain"]
    cron = system["cron"]

    print("=" * 60)
    print("交易日工作流示例")
    print("=" * 60)

    # ─── 09:00 盘前扫描 ───
    print("\n[09:00] 盘前扫描...")
    cron.execute_job("pre_market_scan", force=True)

    # ─── 09:35 开盘后分析 ───
    print("\n[09:35] 用户查询: '今天AI板块怎么样？'")

    # Router 自动注入 Brain 上下文
    # 包含：昨天的情绪判断 + 相关历史记忆 + 最近的交易决策
    # response = system["router"]("今天AI板块怎么样？")

    # 模拟分析结果 → 发射买入信号
    event_log.emit(EventType.SIGNAL_BUY, {
        "symbol": "002415.SZ",
        "name": "海康威视",
        "reason": "AI应用板块启动，技术面突破",
        "score": 82,
    }, source="sector_rotation")

    # ─── 提交交易决策 ───
    print("\n[09:40] 提交交易决策...")
    brain.commit_decision(
        action="BUY",
        symbol="002415.SZ",
        reasoning_chain=[
            "1. 板块轮动信号：AI应用板块今日领涨",
            "2. CANSLIM评分：82分，通过筛选",
            "3. 技术面：突破20日线，量比2.1",
            "4. Brain召回：上次板块轮动初期入场，5日内盈利+8%",
            "5. 情绪面：市场偏暖(0.65)，可以操作",
        ],
        signals_used=["sector_rotation", "canslim", "technical", "brain_recall"],
        confidence=0.78,
        risk_assessment={"stop_loss": "-5%", "target": "+12%", "position": "10%"},
        market_context={"上证": 3280, "成交额": "1.1万亿"},
    )

    # ─── 10:30 盘中风控 ───
    print("\n[10:30] 盘中风控检查...")
    cron.execute_job("risk_check", force=True)

    # ─── 11:35 午间复盘 ───
    print("\n[11:35] 午间复盘...")
    cron.execute_job("midday_review", force=True)

    # ─── 15:30 盘后复盘 ───
    print("\n[15:30] 盘后复盘...")
    cron.execute_job("post_market_review", force=True)

    # 更新情绪
    brain.update_emotion(
        greed_fear_index=0.68,
        trend_bias="cautious_bullish",
        volatility_feel="normal",
        sector_heat={"AI应用": 0.9, "消费电子": 0.55, "医药": 0.25},
        market_phase="markup",
        confidence=0.72,
        reasoning="AI应用板块全天强势，成交额放大，北向持续流入。"
                  "但板块内部开始分化，需关注明日是否延续。",
    )

    # 存储今日洞察
    brain.remember(
        key=f"insight_ai_app_{datetime.now().strftime('%Y%m%d')}",
        category="sector_insight",
        content="AI应用板块今日全面启动，龙头海康威视涨停。"
                "板块从AI算力向AI应用切换的信号确认。"
                "明日关注：是否有2板确认，以及是否有新的AI应用股跟风。",
        confidence=0.8,
        tags=["AI应用", "板块轮动", "龙头", "海康威视"],
        source="post_market_review",
    )

    # 写日志
    brain.write_daily_journal(
        f"""## 大盘
上证收3280，涨0.8%，成交额1.2万亿。

## 操作
- 买入 002415 海康威视 10%仓位，均价XX元

## 板块
AI应用全天领涨，消费电子跟涨，医药继续调整。

## 明日计划
- 关注AI应用是否延续
- 海康威视设置止损：跌破20日线
- 如果板块继续走强，考虑加仓AI应用方向
"""
    )

    # ─── 查看统计 ───
    print("\n[统计] 事件日志摘要:")
    summary = event_log.summary(hours=24)
    print(f"  总事件: {summary['total_events']}")
    print(f"  按类型: {json.dumps(summary['by_type'], ensure_ascii=False)}")

    print("\n✅ 交易日工作流完成")


if __name__ == "__main__":
    import json
    from datetime import datetime
    example_trading_workflow()
