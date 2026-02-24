# OpenAlice 架构 → A股量化投资 AI Agent 映射与优化建议

## 一、架构层级对照

### 1. Providers 层（AI 后端）

| OpenAlice | 你的系统 | 差距分析 |
|-----------|----------|----------|
| 双提供商：Claude Code CLI + Vercel AI SDK | Router LLM（Anthropic API）+ 关键词路由回退 | ⚠️ 你的系统只有单一 LLM 提供商，且回退机制是降级（关键词）而非平级切换 |
| `ProviderRouter` 运行时动态切换 | Router 分发查询到不同 skill | ✅ 路由概念相似，但粒度不同 |
| `ai-provider.json` 配置热切换 | 无热切换配置 | ❌ 缺少运行时配置切换能力 |

**🔧 优化建议：**
- **多模型路由**：引入多 LLM 后端（如 Claude 用于深度分析、DeepSeek/Qwen 用于高频轻量查询），根据任务复杂度动态选择，降低 API 成本
- **运行时配置切换**：用 JSON/YAML 配置文件控制 AI 提供商，无需重启即可切换模型

---

### 2. Core 层（引擎核心）

| OpenAlice 组件 | 你的系统对应 | 差距分析 |
|----------------|-------------|----------|
| `Engine` — 薄门面层 | 主入口脚本 | ⚠️ 你的系统可能缺少统一的调度门面 |
| `AgentCenter` — 集中式代理管理 | Router LLM 分发 | ✅ 概念类似 |
| `ToolCenter` — 集中式工具注册 | 各 skill 模块分散定义 | ⚠️ 缺少统一的工具注册中心 |
| `Session Store` — JSONL 会话持久化 | Scratchpad 执行日志 | ⚠️ Scratchpad 偏向日志，缺少多轮对话会话管理 |
| `EventLog` — 持久化追加事件日志 | Scratchpad（部分覆盖） | ⚠️ 你的日志更偏执行记录，缺乏事件驱动的订阅/恢复机制 |
| `ConnectorRegistry` — 通道追踪 | 无 | ❌ 缺少多通道接入能力 |
| `Compaction` — 上下文窗口自动压缩 | 无 | ❌ 长对话/长分析链时可能会爆 context window |

**🔧 优化建议：**

#### a) 统一工具注册中心（ToolCenter）
```python
# tool_center.py
class ToolCenter:
    """集中式工具注册，所有 skill 在此注册"""
    _tools: dict[str, callable] = {}
    _metadata: dict[str, ToolMeta] = {}

    @classmethod
    def register(cls, name: str, func: callable, meta: ToolMeta):
        cls._tools[name] = func
        cls._metadata[name] = meta

    @classmethod
    def get_tool_descriptions(cls) -> list[dict]:
        """导出给 LLM 的工具描述列表"""
        return [m.to_llm_schema() for m in cls._metadata.values()]

    @classmethod
    def execute(cls, name: str, **kwargs):
        return cls._tools[name](**kwargs)
```
好处：Router LLM 不再需要硬编码 skill 列表，新增能力只需注册即可。

#### b) 事件日志系统（EventLog）
```python
# event_log.py
class EventLog:
    """持久化追加事件日志，支持订阅和崩溃恢复"""
    def __init__(self, path="data/events.jsonl"):
        self.path = path
        self.subscribers = defaultdict(list)

    def emit(self, event_type: str, payload: dict):
        entry = {"ts": time.time(), "type": event_type, "data": payload}
        with open(self.path, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        for cb in self.subscribers.get(event_type, []):
            cb(entry)

    def subscribe(self, event_type: str, callback):
        self.subscribers[event_type].append(callback)

    def replay(self, since: float = 0) -> list[dict]:
        """崩溃恢复：重放指定时间点之后的事件"""
        ...
```
这将你的 Scratchpad 从单纯的日志升级为**事件驱动架构**，交易信号、风控告警、定时任务都可以通过事件串联。

#### c) 上下文压缩（Compaction）
当分析链很长（如扫描全市场 → 板块轮动 → 个股筛选 → 回测验证），对话上下文会迅速膨胀。引入自动摘要压缩：
```python
def compact_context(messages: list, max_tokens: int = 8000) -> list:
    """当上下文超限时，对早期消息自动摘要"""
    total = sum(count_tokens(m) for m in messages)
    if total <= max_tokens:
        return messages
    # 保留最近 N 条 + 对历史做摘要
    summary = llm_summarize(messages[:-5])
    return [{"role": "system", "content": f"历史摘要：{summary}"}] + messages[-5:]
```

---

### 3. Extensions 层（领域扩展）

| OpenAlice 扩展 | 你的系统对应 | 差距分析 |
|----------------|-------------|----------|
| `analysis-kit`（技术指标、新闻、沙箱） | 情绪分析 skill、板块轮动分析 | ✅ 你有，但可以更模块化 |
| `crypto-trading`（CCXT 交易执行） | 无实盘执行层 | ⚠️ 你的系统侧重分析和回测，缺少统一的订单执行层 |
| `securities-trading`（Alpaca 证券） | CANSLIM 回测系统 | ⚠️ 回测 ≠ 执行，缺少实盘桥接 |
| `brain`（认知状态、记忆、情绪） | 无 | ❌ 这是一个重要缺失 |
| `browser`（浏览器自动化） | 无 | ⚠️ 非核心，但对抓取 A 股资讯有用 |

**🔧 优化建议：**

#### a) 认知状态模块（Brain）— 最大的缺失
OpenAlice 的 Brain 模块维护了 agent 的持久记忆和情绪状态，这对量化 agent 非常关键：

```python
# brain.py
class AgentBrain:
    """Agent 认知状态管理"""

    def __init__(self, brain_dir="data/brain/"):
        self.memory_file = f"{brain_dir}/memory.jsonl"      # 长期记忆
        self.emotion_file = f"{brain_dir}/emotion.json"      # 市场情绪判断
        self.commit_file = f"{brain_dir}/commits.jsonl"      # 决策历史

    def remember(self, key: str, insight: str, confidence: float):
        """存储市场洞察到长期记忆"""
        # 例如："2025年1月板块轮动从AI转向消费" → 下次分析时可回溯

    def get_market_emotion(self) -> dict:
        """当前市场情绪状态"""
        return {"greed_fear": 0.6, "trend": "cautious_bullish", ...}

    def commit_decision(self, decision: dict):
        """记录每次交易决策的完整推理链"""
        # 类似 git commit，可以回溯为什么做了某个决定

    def recall(self, query: str, top_k: int = 5) -> list:
        """根据当前市场情况召回相关历史记忆"""
```

核心价值：
- **避免重复犯错**：记住"上次在XX板块高位追涨导致亏损"
- **积累市场直觉**：长期记忆中沉淀板块轮动规律
- **决策可追溯**：每个交易都有完整的推理链 commit

#### b) 统一订单执行层（类比 crypto-trading 的 Wallet 模型）
OpenAlice 的 git 式钱包（stage → commit → push）设计值得借鉴：

```python
# order_wallet.py
class OrderWallet:
    """A股订单管理，stage → commit → push 三阶段"""

    def stage(self, symbol: str, action: str, amount: int, reason: str):
        """暂存意向单（AI 分析完成，尚未确认）"""
        self.staged.append({...})

    def commit(self, order_id: str):
        """确认订单（风控检查通过）"""
        order = self.staged.pop(order_id)
        self.run_risk_checks(order)  # 持仓集中度、涨停板检查等
        self.committed.append(order)

    def push(self, broker="easytrader"):
        """推送到券商执行"""
        for order in self.committed:
            broker.submit(order)
        self.log_execution(self.committed)

    def diff(self) -> str:
        """显示当前暂存 vs 已确认 vs 已执行的差异"""
```

好处：在 AI 分析和实际执行之间增加了**人工/风控审核缓冲层**。

---

### 4. Tasks 层（后台任务）

| OpenAlice | 你的系统 | 差距分析 |
|-----------|----------|----------|
| `CronEngine` — 事件驱动定时任务 | 无 | ❌ 缺少定时任务引擎 |
| `Heartbeat` — 健康检查 + 结构化响应 | 无 | ❌ 缺少系统健康监控 |

**🔧 优化建议：**

```python
# cron_engine.py
class CronEngine:
    """A股场景的定时任务"""
    jobs = {
        "pre_market":   {"cron": "0 9 0 * * 1-5",  "task": "盘前扫描：隔夜消息、竞价异动"},
        "morning_scan": {"cron": "0 9 35 * * 1-5", "task": "早盘扫描：量价异动、板块强度"},
        "midday_review":{"cron": "0 11 35 * * 1-5", "task": "午间复盘：上午走势总结"},
        "closing_scan": {"cron": "0 14 50 * * 1-5", "task": "尾盘扫描：尾盘异动、明日布局"},
        "post_market":  {"cron": "0 15 30 * * 1-5", "task": "盘后复盘：全天总结、更新认知"},
        "weekly_review": {"cron": "0 20 0 * * 5",   "task": "周度复盘：板块轮动、策略评估"},
    }
```

这让你的 agent 变成一个**真正 24/7 运转的交易助手**，而不是被动等待查询。

---

### 5. Interfaces 层（交互接口）

| OpenAlice | 你的系统 | 差距分析 |
|-----------|----------|----------|
| Web UI（本地聊天） | 命令行交互 | ⚠️ 可视化不足 |
| Telegram Bot | 无 | ⚠️ 缺少移动端推送 |
| HTTP API | 无 | ❌ 缺少 API 层，难以与其他系统集成 |
| MCP Server | 无 | 可选 |

**🔧 优化建议：**
- 短期：用 FastAPI 加一层 HTTP API，方便后续接 Web UI 或微信推送
- 中期：接入微信/钉钉机器人，实现盘中实时推送告警

---

## 二、优化优先级排序

| 优先级 | 优化项 | 预期收益 | 工作量 |
|--------|--------|----------|--------|
| 🔴 P0 | **Brain 认知状态模块** | 让 agent 积累经验、避免重复犯错 | 中 |
| 🔴 P0 | **EventLog 事件驱动架构** | 将 Scratchpad 升级为事件总线，串联所有模块 | 中 |
| 🟠 P1 | **CronEngine 定时任务** | 从被动查询变为主动监控 | 小 |
| 🟠 P1 | **ToolCenter 统一工具注册** | 新增 skill 更方便，Router 更智能 | 小 |
| 🟡 P2 | **Context Compaction** | 避免长分析链爆 context window | 小 |
| 🟡 P2 | **多模型路由** | 降低 API 成本，轻量任务用便宜模型 | 中 |
| 🟢 P3 | **OrderWallet 订单执行层** | 为未来实盘做准备 | 大 |
| 🟢 P3 | **HTTP API + 消息推送** | 提升交互体验 | 中 |

---

## 三、你的系统已有的优势（OpenAlice 不具备的）

1. **Eval 层** — OpenAlice 没有 skill 准确性验证机制，你的 Eval 层是更严谨的设计
2. **专业回测体系** — CANSLIM 回测系统、板块轮动回测是 OpenAlice 没有的深度功能
3. **A 股特色逻辑** — 涨跌停、T+1、竞价规则等 OpenAlice 完全不涉及
4. **Scratchpad** — 执行日志的透明性已经很好，只需升级为事件驱动即可

---

## 四、推荐实施路线

```
Phase 1（1-2周）：基础设施升级
  ├── EventLog 替代/增强 Scratchpad
  ├── ToolCenter 统一工具注册
  └── Context Compaction

Phase 2（2-3周）：认知与自动化
  ├── Brain 认知状态模块
  ├── CronEngine 定时任务（盘前/盘中/盘后）
  └── 决策 commit 历史

Phase 3（3-4周）：执行与接口
  ├── OrderWallet 三阶段订单管理
  ├── FastAPI HTTP 层
  └── 微信/钉钉推送集成

Phase 4（持续优化）
  ├── 多模型路由（DeepSeek 处理轻量查询）
  ├── Web UI 仪表板
  └── 策略进化模式（agent 自我优化参数）
```
