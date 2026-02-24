"""
Brain + CronEngine 集成指南
============================

你现在的系统结构：
  router.py      → _route_with_llm() / _route_with_keywords() / _execute_skills()
  event_log.py   → EventLog（已接）
  skill_registry.py → register_all_skills()（已接）
  cron_and_tools.py → ToolCenter（已接）

本次接入 Brain + CronEngine，改动 4 处：

  改动 1: config.py 增加配置项
  改动 2: router.py 初始化 Brain，_route_with_llm 注入上下文
  改动 3: router.py _execute_skills 完成后可选存记忆
  改动 4: 系统入口初始化 CronEngine

下面是具体的代码补丁。
"""


# ═══════════════════════════════════════════════════════════
# 改动 1: config.py / config_template.py
# ═══════════════════════════════════════════════════════════

CONFIG_PATCH = """
# ---------- Brain ----------
USE_BRAIN = True                    # 是否启用 Brain 认知状态
BRAIN_DIR = "data/brain"            # Brain 数据目录

# ---------- CronEngine ----------
USE_CRON = True                     # 是否启用定时任务
CRON_DIR = "data/cron"              # CronEngine 数据目录
"""


# ═══════════════════════════════════════════════════════════
# 改动 2: router.py — 初始化 Brain + 注入上下文
# ═══════════════════════════════════════════════════════════

ROUTER_PATCH_INIT = """
# ---- router.py 顶部，在 from skill_registry import ... 之后 ----

# Brain 初始化
from brain import get_brain
try:
    from config import USE_BRAIN
except ImportError:
    USE_BRAIN = True

brain = get_brain(event_log=event_log) if USE_BRAIN else None
"""

ROUTER_PATCH_LLM = """
# ---- router.py → _route_with_llm() 方法内 ----
# 在构建 messages 发给 LLM 之前，加入以下代码：

# Brain 上下文注入
brain_context = ""
if brain:
    brain_context = brain.get_context_for_prompt(question)

# 然后在 system prompt 或 user message 中拼接 brain_context
# 例如在你现有的 tool_prompt 后面追加：
#   full_prompt = f"{tool_prompt}\\n\\n{brain_context}" if brain_context else tool_prompt
"""

ROUTER_PATCH_EXECUTE = """
# ---- router.py → _execute_skills() 完成后（可选） ----
# 在 skill 执行完毕、返回 result 之前，可选存储洞察到 Brain：

if brain and result:
    # skill_name 是当前执行的 skill 名称
    brain.learn_from_output(skill_name, question, result)
"""


# ═══════════════════════════════════════════════════════════
# 改动 3: 系统入口 — 初始化 CronEngine
# ═══════════════════════════════════════════════════════════

ENTRY_PATCH = """
# ---- 在你的系统入口（如 main.py / daily_agent.py）----

from cron_engine import CronEngine
from brain import get_brain

# 1. 获取已有的 EventLog 和 Brain
# event_log = ...（你已有的 EventLog 实例）
brain = get_brain(event_log=event_log)

# 2. 定义 AI 执行器（连接你的 Router）
def ai_executor(prompt: str) -> str:
    '''让 CronEngine 通过 Router 执行 AI 任务'''
    # 方式 A: 如果你的 Router 有类似 router.ask(question) 的方法
    # return router.ask(prompt)

    # 方式 B: 如果需要直接调用 LLM
    # from your_llm_module import call_llm
    # return call_llm([{"role": "user", "content": prompt}])

    # 方式 C: 如果用 router.py 的入口函数
    # return router.route_and_execute(prompt)
    pass

# 3. 初始化 CronEngine
cron = CronEngine(
    event_log=event_log,
    brain=brain,
    ai_executor=ai_executor,
    on_result=lambda job_id, result: print(f"[{job_id}] {result[:200]}"),
    # 未来可以把 on_result 改为微信/钉钉推送
)

# 4. 启动后台调度（需要 pip install schedule）
# cron.start()

# 或者：在你已有的定时逻辑中手动调用
# cron.execute("pre_market", force=True)    # 盘前扫描
# cron.execute("post_market", force=True)   # 盘后复盘
"""


# ═══════════════════════════════════════════════════════════
# 完整集成示例（可直接运行测试）
# ═══════════════════════════════════════════════════════════

def integration_test():
    """模拟完整的 Brain + CronEngine 集成流程"""
    import sys
    sys.path.insert(0, ".")

    from event_log import EventLog, EventType
    from brain import AgentBrain
    from cron_engine import CronEngine

    print("=" * 60)
    print("Brain + CronEngine 集成测试")
    print("=" * 60)

    # ── 初始化 ──
    event_log = EventLog(log_dir="/tmp/brain_cron_test/event_log/")
    brain = AgentBrain(brain_dir="/tmp/brain_cron_test/brain/", event_log=event_log)
    print("✅ EventLog + Brain 初始化")

    # ── 1. Brain: 存储记忆 ──
    print("\n--- 1. Brain 记忆 ---")
    brain.remember(
        "pattern_rotation_ai", "sector_insight",
        "2025年板块轮动路径：AI算力→AI应用→消费电子→半导体",
        confidence=0.8, tags=["板块轮动", "AI", "消费电子"],
        source="weekly_review",
    )
    brain.remember(
        "lesson_chase_202501", "trade_lesson",
        "AI板块连涨5日后追高买入，次日跌停。教训：板块末期不追高。",
        confidence=0.9, tags=["AI", "追高", "教训"],
        source="post_market",
    )
    print(f"  记忆数: {brain.memory_count}")

    # ── 2. Brain: 更新情绪 ──
    print("\n--- 2. 市场情绪 ---")
    brain.update_emotion(
        greed_fear_index=0.65, trend_bias="cautious_bullish",
        volatility_feel="normal",
        sector_heat={"AI算力": 0.85, "消费电子": 0.6, "医药": 0.3},
        market_phase="markup", confidence=0.7,
        reasoning="成交额放大，北向流入，但高位分歧",
    )
    print(f"  情绪: {brain.get_emotion().trend_bias}")

    # ── 3. Brain: 提交决策 ──
    print("\n--- 3. 决策提交 ---")
    commit = brain.commit_decision(
        action="BUY", symbol="002415.SZ",
        reasoning_chain=[
            "1. CANSLIM: C=88, A=82, N=新品",
            "2. 板块轮动: 消费电子轮入",
            "3. 技术: 突破前高, 量比2x",
        ],
        signals_used=["canslim", "sector_rotation", "technical"],
        confidence=0.78,
        risk_assessment={"stop_loss": "-5%", "target": "+15%"},
        market_context={"上证": 3250, "成交额": "1.2万亿"},
    )
    print(f"  commit_id: {commit.commit_id}")

    # ── 4. Brain: 生成 prompt 上下文 ──
    print("\n--- 4. Prompt 上下文（核心输出）---")
    ctx = brain.get_context_for_prompt("AI板块轮动还能追吗")
    print(ctx)

    # ── 5. Brain: 从 skill 输出自动学习 ──
    print("\n--- 5. 自动学习 ---")
    mem = brain.learn_from_output(
        "sector_rotation", "当前板块轮动方向",
        "当前主线AI算力开始分化，消费电子接力上涨，成交额放大。"
        "建议关注消费电子中的龙头标的，同时注意AI算力的高位风险。"
    )
    if mem:
        print(f"  自动存储: {mem.key} ({mem.category})")
    print(f"  记忆数: {brain.memory_count}")

    # ── 6. CronEngine ──
    print("\n--- 6. CronEngine ---")
    results = []

    def mock_executor(prompt: str) -> str:
        return f"[模拟] 执行完成 (prompt {len(prompt)} 字)"

    def on_result(job_id: str, result: str):
        results.append((job_id, result))

    cron = CronEngine(
        jobs_file="/tmp/brain_cron_test/cron/jobs.json",
        event_log=event_log,
        brain=brain,
        ai_executor=mock_executor,
        on_result=on_result,
    )

    print("  任务列表:")
    for j in cron.list_jobs():
        s = "✅" if j["enabled"] else "❌"
        print(f"    {s} {j['time']} | {j['name']:6s} | {j['priority']}")

    # 手动执行几个任务
    print("\n  手动执行:")
    cron.execute("pre_market", force=True)
    cron.execute("risk_check", force=True)
    cron.execute("post_market", force=True)
    for job_id, result in results:
        print(f"    {job_id}: {result[:60]}")

    # ── 7. 心跳 ──
    print("\n--- 7. 心跳 ---")
    hb = cron.heartbeat()
    print(f"  {json.dumps(hb, ensure_ascii=False)}")

    # ── 8. EventLog 统计 ──
    print("\n--- 8. EventLog 统计 ---")
    summary = event_log.summary(hours=1)
    print(f"  总事件: {summary['total_events']}")
    for t, c in sorted(summary["by_type"].items(), key=lambda x: -x[1]):
        print(f"    {t}: {c}")

    # ── 9. 复盘日志 ──
    print("\n--- 9. 复盘日志 ---")
    brain.write_journal("## 今日总结\nAI算力分化，消费电子接力。买入002415。")
    journal = brain.read_journal()
    print(f"  写入并读取: {len(journal)} 字符")

    print("\n" + "=" * 60)
    print("🎉 Brain + CronEngine 集成测试全部通过！")
    print("=" * 60)


if __name__ == "__main__":
    import json
    integration_test()
