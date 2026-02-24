"""
CronEngine — A股交易定时任务引擎
================================
灵感来源：OpenAlice 的 CronEngine + Heartbeat
适配场景：A股量化投资 AI Agent

核心能力：
1. 基于 A 股交易时间的定时任务调度
2. AI 驱动的任务执行（每个任务由 LLM 处理）
3. 与 EventLog 和 Brain 联动
4. 心跳健康检查
5. 支持自定义任务注册

让 Agent 从"被动问答"变为"主动盘前→盘中→盘后全流程运转"
"""

import time
import json
import threading
from datetime import datetime, timedelta
from typing import Callable, Optional, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum

try:
    import schedule
    HAS_SCHEDULE = True
except ImportError:
    HAS_SCHEDULE = False

# 可选依赖
try:
    from event_log import EventLog, EventType
    HAS_EVENT_LOG = True
except ImportError:
    HAS_EVENT_LOG = False

try:
    from brain import AgentBrain
    HAS_BRAIN = True
except ImportError:
    HAS_BRAIN = False


# ═══════════════════════════════════════════
# A股交易时间工具
# ═══════════════════════════════════════════

class AShareTradingCalendar:
    """A股交易日历工具"""

    @staticmethod
    def is_trading_day(dt: datetime = None) -> bool:
        """判断是否为交易日（简化版，不含节假日）"""
        if dt is None:
            dt = datetime.now()
        return dt.weekday() < 5  # 周一到周五

    @staticmethod
    def is_trading_hours(dt: datetime = None) -> bool:
        """是否在交易时间内"""
        if dt is None:
            dt = datetime.now()
        if not AShareTradingCalendar.is_trading_day(dt):
            return False
        t = dt.time()
        morning = datetime.strptime("09:30", "%H:%M").time() <= t <= datetime.strptime("11:30", "%H:%M").time()
        afternoon = datetime.strptime("13:00", "%H:%M").time() <= t <= datetime.strptime("15:00", "%H:%M").time()
        return morning or afternoon

    @staticmethod
    def is_pre_market(dt: datetime = None) -> bool:
        """是否在盘前（集合竞价前）"""
        if dt is None:
            dt = datetime.now()
        if not AShareTradingCalendar.is_trading_day(dt):
            return False
        t = dt.time()
        return datetime.strptime("08:30", "%H:%M").time() <= t < datetime.strptime("09:25", "%H:%M").time()

    @staticmethod
    def get_session(dt: datetime = None) -> str:
        """获取当前交易时段"""
        if dt is None:
            dt = datetime.now()
        if not AShareTradingCalendar.is_trading_day(dt):
            return "closed"
        t = dt.time()
        if t < datetime.strptime("09:15", "%H:%M").time():
            return "pre_market"
        if t < datetime.strptime("09:30", "%H:%M").time():
            return "auction"  # 集合竞价
        if t <= datetime.strptime("11:30", "%H:%M").time():
            return "morning"
        if t < datetime.strptime("13:00", "%H:%M").time():
            return "midday_break"
        if t <= datetime.strptime("14:57", "%H:%M").time():
            return "afternoon"
        if t <= datetime.strptime("15:00", "%H:%M").time():
            return "closing_auction"  # 尾盘集合竞价
        return "post_market"


# ═══════════════════════════════════════════
# 任务定义
# ═══════════════════════════════════════════

class TaskPriority(str, Enum):
    CRITICAL = "critical"   # 必须执行（如风控检查）
    HIGH = "high"           # 高优（如开盘扫描）
    NORMAL = "normal"       # 常规（如定时分析）
    LOW = "low"             # 低优（如周报生成）


@dataclass
class CronJob:
    """定时任务定义"""
    job_id: str                     # 任务ID
    name: str                       # 任务名称
    description: str                # 任务描述（会传给 LLM）
    schedule_time: str              # 执行时间 "HH:MM"
    weekdays: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # 0=周一
    priority: str = TaskPriority.NORMAL
    enabled: bool = True
    require_trading_day: bool = True  # 是否只在交易日执行

    # AI 执行相关
    prompt_template: str = ""       # 发给 LLM 的 prompt 模板
    tools_required: list[str] = field(default_factory=list)  # 需要的工具列表
    tags: list[str] = field(default_factory=list)  # 标签，用于 Brain 记忆召回

    # 回调（非 AI 任务可直接注册 Python 回调）
    callback: Optional[str] = None  # 注册的回调函数名

    last_run: Optional[float] = None
    last_result: Optional[str] = None


# ═══════════════════════════════════════════
# 预定义的 A 股交易任务
# ═══════════════════════════════════════════

DEFAULT_JOBS: list[dict] = [
    {
        "job_id": "pre_market_scan",
        "name": "盘前扫描",
        "description": "盘前准备：扫描隔夜重大消息、集合竞价异动、北向资金动向",
        "schedule_time": "09:00",
        "priority": "high",
        "prompt_template": """执行盘前扫描任务：
1. 检查隔夜重大新闻和公告（特别是持仓股相关）
2. 分析集合竞价数据（如有）：高开/低开个股
3. 查看北向资金预期动向
4. 检查持仓股是否有利空/利好消息
5. 输出今日操作计划建议

{brain_context}

请给出具体的今日关注清单和操作建议。""",
        "tools_required": ["sentiment_analysis", "sector_rotation"],
        "tags": ["盘前", "扫描", "新闻"],
    },
    {
        "job_id": "morning_momentum",
        "name": "早盘动量扫描",
        "description": "开盘30分钟后扫描量价异动和板块强度",
        "schedule_time": "10:00",
        "priority": "high",
        "prompt_template": """执行早盘动量扫描：
1. 扫描开盘30分钟量价异动个股
2. 分析今日板块强弱排名
3. 检查是否有板块突然启动的信号
4. 对比昨日预判是否准确
5. 是否需要调整今日操作计划

{brain_context}

重点关注异常放量和板块轮动信号。""",
        "tools_required": ["technical_analysis", "sector_rotation"],
        "tags": ["早盘", "动量", "板块"],
    },
    {
        "job_id": "midday_review",
        "name": "午间复盘",
        "description": "午休期间对上午走势做简要复盘",
        "schedule_time": "11:35",
        "priority": "normal",
        "prompt_template": """执行午间复盘：
1. 上午大盘走势总结（指数、成交额、涨跌比）
2. 上午板块表现排名
3. 持仓股上午表现
4. 下午需要关注的风险点
5. 是否需要在下午做调仓

{brain_context}

简明扼要即可。""",
        "tools_required": ["technical_analysis"],
        "tags": ["午盘", "复盘"],
    },
    {
        "job_id": "closing_scan",
        "name": "尾盘扫描",
        "description": "尾盘异动扫描和明日布局",
        "schedule_time": "14:45",
        "priority": "high",
        "prompt_template": """执行尾盘扫描：
1. 扫描尾盘异动（尾盘拉升/跳水的个股和板块）
2. 今日涨停板梳理（板块分布、连板高度）
3. 明日可能的方向预判
4. 是否有尾盘需要执行的操作

{brain_context}

重点关注尾盘资金动向和明日潜在机会。""",
        "tools_required": ["technical_analysis", "sector_rotation"],
        "tags": ["尾盘", "异动", "布局"],
    },
    {
        "job_id": "post_market_review",
        "name": "盘后复盘",
        "description": "全天复盘总结，更新 Brain 认知状态",
        "schedule_time": "15:30",
        "priority": "critical",
        "prompt_template": """执行盘后全面复盘：
1. 今日大盘总结（指数、成交额、涨跌比、北向资金）
2. 板块轮动分析（今日主线、明日可能切换方向）
3. 持仓股逐一点评，标记需要止损/止盈的
4. 今日操作回顾（做对了什么、做错了什么）
5. 更新市场情绪判断
6. 明日操作计划
7. 将今日重要洞察存入长期记忆

{brain_context}

这是一天中最重要的任务，请详细分析。""",
        "tools_required": ["sentiment_analysis", "sector_rotation", "canslim"],
        "tags": ["复盘", "总结", "板块轮动", "情绪"],
    },
    {
        "job_id": "weekly_review",
        "name": "周度复盘",
        "description": "每周五盘后的深度复盘",
        "schedule_time": "20:00",
        "weekdays": [4],  # 仅周五
        "priority": "critical",
        "prompt_template": """执行周度深度复盘：
1. 本周大盘走势回顾
2. 本周板块轮动路径总结
3. 本周所有交易决策回顾和胜率统计
4. 策略有效性评估（CANSLIM、板块轮动等各策略本周表现）
5. 需要调整的策略参数
6. 下周市场展望和操作计划
7. 更新长期记忆中的板块规律

{brain_context}
{win_rate_stats}

请做深度分析，这关系到策略的持续优化。""",
        "tools_required": ["sentiment_analysis", "sector_rotation", "canslim"],
        "tags": ["周报", "复盘", "策略评估"],
    },
    {
        "job_id": "risk_check",
        "name": "盘中风控检查",
        "description": "定时检查持仓风险",
        "schedule_time": "10:30",
        "priority": "critical",
        "prompt_template": """执行风控检查：
1. 检查各持仓股是否触及止损线
2. 检查总仓位是否超限
3. 检查单板块集中度是否超标
4. 检查是否有持仓股临近涨跌停
5. 检查账户整体回撤是否超限

如有风险，立即告警。""",
        "tools_required": ["risk_manager"],
        "tags": ["风控", "止损"],
    },
]


# ═══════════════════════════════════════════
# CronEngine 主类
# ═══════════════════════════════════════════

class CronEngine:
    """
    A 股交易定时任务引擎

    特性：
    - 基于 A 股交易日历的智能调度
    - 每个任务可关联 AI 执行（通过 prompt_template）
    - 与 EventLog 联动（事件驱动）
    - 与 Brain 联动（自动注入认知上下文）
    - 支持手动触发和动态注册
    """

    def __init__(self, jobs_file: str = "data/cron/jobs.json",
                 event_log: Optional[Any] = None,
                 brain: Optional[Any] = None,
                 ai_executor: Optional[Callable] = None,
                 on_result: Optional[Callable[[str, str], None]] = None):
        """
        Parameters
        ----------
        jobs_file : 任务配置文件路径
        event_log : EventLog 实例（可选）
        brain : AgentBrain 实例（可选）
        ai_executor : AI 执行函数，签名 (prompt: str, tools: list[str]) -> str
        on_result : 任务执行完成回调 (job_id: str, result: str) -> None，供 AutoJournal 等使用
        """
        self.jobs_file = Path(jobs_file)
        self.jobs_file.parent.mkdir(parents=True, exist_ok=True)
        self.event_log = event_log
        self.brain = brain
        self.ai_executor = ai_executor
        self.on_result = on_result
        self.calendar = AShareTradingCalendar()

        self._jobs: dict[str, CronJob] = {}
        self._callbacks: dict[str, Callable] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # 加载任务
        self._load_jobs()

    # ─────────────────────────────────────────
    # 任务管理
    # ─────────────────────────────────────────

    def register_job(self, job: CronJob):
        """注册一个定时任务"""
        self._jobs[job.job_id] = job
        self._save_jobs()

    def register_callback(self, name: str, callback: Callable):
        """注册一个 Python 回调（用于非 AI 任务）"""
        self._callbacks[name] = callback

    def enable_job(self, job_id: str):
        if job_id in self._jobs:
            self._jobs[job_id].enabled = True
            self._save_jobs()

    def disable_job(self, job_id: str):
        if job_id in self._jobs:
            self._jobs[job_id].enabled = False
            self._save_jobs()

    def list_jobs(self) -> list[dict]:
        """列出所有任务及状态"""
        result = []
        for job in self._jobs.values():
            result.append({
                "job_id": job.job_id,
                "name": job.name,
                "schedule": job.schedule_time,
                "weekdays": job.weekdays,
                "enabled": job.enabled,
                "priority": job.priority,
                "last_run": datetime.fromtimestamp(job.last_run).strftime("%Y-%m-%d %H:%M") if job.last_run else "从未执行",
            })
        return sorted(result, key=lambda x: x["schedule"])

    # ─────────────────────────────────────────
    # 执行引擎
    # ─────────────────────────────────────────

    def execute_job(self, job_id: str, force: bool = False) -> Optional[str]:
        """
        执行指定任务

        Parameters
        ----------
        job_id : 任务ID
        force : 是否强制执行（跳过交易日检查）
        """
        job = self._jobs.get(job_id)
        if not job:
            return f"任务 {job_id} 不存在"

        if not job.enabled and not force:
            return f"任务 {job.name} 已禁用"

        if job.require_trading_day and not self.calendar.is_trading_day() and not force:
            return f"今日非交易日，跳过 {job.name}"

        # 发射事件
        if self.event_log and HAS_EVENT_LOG:
            self.event_log.emit(EventType.SYSTEM_CRON_FIRE, {
                "job_id": job_id,
                "job_name": job.name,
                "priority": job.priority,
            }, source="cron")

        result = None
        start_time = time.time()

        try:
            if job.callback and job.callback in self._callbacks:
                # Python 回调模式
                result = self._callbacks[job.callback]()

            elif job.prompt_template and self.ai_executor:
                # AI 执行模式
                prompt = self._build_prompt(job)
                result = self.ai_executor(prompt, job.tools_required)

            else:
                result = f"任务 {job.name} 无执行器（缺少 callback 或 ai_executor）"

        except Exception as e:
            result = f"任务 {job.name} 执行失败: {str(e)}"
            if self.event_log and HAS_EVENT_LOG:
                self.event_log.emit(EventType.SYSTEM_ERROR, {
                    "job_id": job_id,
                    "error": str(e),
                }, source="cron")

        # 更新执行状态
        duration = time.time() - start_time
        job.last_run = time.time()
        job.last_result = str(result)[:1000] if result else None
        self._save_jobs()

        # 完成事件
        if self.event_log and HAS_EVENT_LOG:
            self.event_log.emit(EventType.SYSTEM_CRON_COMPLETE, {
                "job_id": job_id,
                "job_name": job.name,
                "duration_sec": round(duration, 2),
                "result_preview": str(result)[:300] if result else "",
            }, source="cron")

        # 盘后/复盘结果回调（如 AutoJournal）
        if self.on_result and result is not None:
            try:
                self.on_result(job_id, str(result))
            except Exception:
                pass

        return result

    def _build_prompt(self, job: CronJob) -> str:
        """构建 AI 执行的 prompt（自动注入 Brain 上下文）"""
        prompt = job.prompt_template

        # 注入 Brain 认知上下文
        brain_context = ""
        if self.brain and HAS_BRAIN:
            brain_context = self.brain.get_context_for_prompt(
                query_tags=job.tags,
                include_emotion=True,
                include_memories=True,
                include_recent_commits=True,
            )

        prompt = prompt.replace("{brain_context}", brain_context)

        # 注入胜率统计（周报用）
        win_rate_stats = ""
        if self.brain and HAS_BRAIN and "{win_rate_stats}" in prompt:
            stats = self.brain.get_win_rate(days_back=7)
            win_rate_stats = f"本周交易统计：总{stats['overall']['total']}笔，" \
                           f"胜率{stats['overall']['rate']:.0%}"
        prompt = prompt.replace("{win_rate_stats}", win_rate_stats)

        # 注入当前市场时段
        session = self.calendar.get_session()
        prompt = f"[当前时段: {session} | 时间: {datetime.now().strftime('%H:%M')}]\n\n" + prompt

        return prompt

    # ─────────────────────────────────────────
    # 调度器
    # ─────────────────────────────────────────

    def start(self):
        """启动定时调度器（后台线程）"""
        if self._running:
            return
        if not HAS_SCHEDULE:
            print("⚠️ 未安装 schedule 库，请运行: pip install schedule")
            print("   CronEngine 将仅支持手动触发模式")
            return

        self._running = True

        # 为每个任务设置 schedule
        for job in self._jobs.values():
            if not job.enabled:
                continue
            schedule.every().day.at(job.schedule_time).do(
                self._scheduled_run, job_id=job.job_id
            )

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print(f"CronEngine 已启动，注册了 {len([j for j in self._jobs.values() if j.enabled])} 个活跃任务")

    def stop(self):
        """停止调度器"""
        self._running = False
        if HAS_SCHEDULE:
            schedule.clear()
        if self._thread:
            self._thread.join(timeout=5)
        print("CronEngine 已停止")

    def _run_loop(self):
        """调度器主循环"""
        while self._running:
            if HAS_SCHEDULE:
                schedule.run_pending()
            time.sleep(30)

    def _scheduled_run(self, job_id: str):
        """被 schedule 调用的包装函数"""
        job = self._jobs.get(job_id)
        if not job or not job.enabled:
            return

        # 检查是否为指定工作日
        today_weekday = datetime.now().weekday()
        if today_weekday not in job.weekdays:
            return

        print(f"⏰ 执行定时任务: {job.name} ({job_id})")
        result = self.execute_job(job_id)
        if result:
            print(f"   结果: {str(result)[:200]}")

    # ─────────────────────────────────────────
    # 心跳（Heartbeat）
    # ─────────────────────────────────────────

    def heartbeat(self) -> dict:
        """
        系统心跳检查

        返回当前系统状态，用于健康监控
        """
        now = datetime.now()
        status = {
            "timestamp": now.isoformat(),
            "trading_day": self.calendar.is_trading_day(),
            "session": self.calendar.get_session(),
            "engine_running": self._running,
            "active_jobs": len([j for j in self._jobs.values() if j.enabled]),
            "total_jobs": len(self._jobs),
        }

        # 检查是否有错过的任务
        missed = []
        for job in self._jobs.values():
            if not job.enabled:
                continue
            if job.last_run:
                # 如果今天应该执行但还没执行
                sched_hour, sched_min = map(int, job.schedule_time.split(":"))
                sched_time = now.replace(hour=sched_hour, minute=sched_min, second=0)
                if now > sched_time and job.last_run < sched_time.timestamp():
                    missed.append(job.job_id)

        status["missed_jobs"] = missed

        if self.event_log and HAS_EVENT_LOG:
            self.event_log.emit(EventType.SYSTEM_HEARTBEAT, status, source="cron")

        return status

    # ─────────────────────────────────────────
    # 持久化
    # ─────────────────────────────────────────

    def _load_jobs(self):
        """加载任务配置"""
        if self.jobs_file.exists():
            try:
                with open(self.jobs_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for item in data:
                    item.pop("callback", None)  # callback 不持久化
                    self._jobs[item["job_id"]] = CronJob(**item)
            except (json.JSONDecodeError, TypeError) as e:
                print(f"加载任务配置失败: {e}，使用默认任务")
                self._load_defaults()
        else:
            self._load_defaults()

    def _load_defaults(self):
        """加载默认任务"""
        for item in DEFAULT_JOBS:
            self._jobs[item["job_id"]] = CronJob(**item)
        self._save_jobs()

    def _save_jobs(self):
        """保存任务配置"""
        data = []
        for job in self._jobs.values():
            d = asdict(job)
            d.pop("callback", None)
            d.pop("last_result", None)  # 不保存大段结果文本
            data.append(d)
        with open(self.jobs_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════
# ToolCenter — 统一工具注册
# ═══════════════════════════════════════════

@dataclass
class ToolMeta:
    """工具元数据"""
    name: str
    description: str
    parameters: dict = field(default_factory=dict)
    category: str = "general"  # analysis / trading / risk / brain / system

    def to_llm_schema(self) -> dict:
        """导出为 LLM function calling 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }


class ToolCenter:
    """
    集中式工具注册中心

    支持两种注册方式：
    - register(): 工具名+函数，用于执行
    - register_route(): 路由 key+描述+关键词+skills，用于 Router 路由与关键词匹配
    """

    _tools: dict[str, Callable] = {}
    _metadata: dict[str, ToolMeta] = {}
    _routes: dict = {}  # route_key -> {name, description, keywords, skills}

    @classmethod
    def register_route(cls, route_key: str, name: str, description: str,
                      keywords: list, skills: list):
        """注册路由（供 Router / final_system 使用）"""
        cls._routes[route_key] = {
            "name": name,
            "description": description,
            "keywords": keywords or [],
            "skills": skills or [],
        }

    @classmethod
    def get_route(cls, route_key: str) -> Optional[dict]:
        """获取路由信息"""
        return cls._routes.get(route_key)

    @classmethod
    def match_by_keywords(cls, query: str, top_k: int = 3) -> list:
        """按关键词匹配路由，返回 route_key 列表"""
        q = (query or "").strip()
        scores = []
        for rk, meta in cls._routes.items():
            kws = meta.get("keywords") or []
            score = sum(1 for kw in kws if kw in q)
            if score > 0:
                scores.append((rk, float(score)))
        scores.sort(key=lambda x: -x[1])
        return [s[0] for s in scores[:top_k]]

    @classmethod
    def register(cls, name: str, func: Callable, description: str,
                 parameters: dict = None, category: str = "general"):
        """
        注册一个工具

        示例：
            ToolCenter.register(
                name="sector_rotation_analysis",
                func=analyze_sector_rotation,
                description="分析当前A股板块轮动状态，返回板块强弱排名和轮动方向",
                parameters={"type": "object", "properties": {...}},
                category="analysis",
            )
        """
        cls._tools[name] = func
        cls._metadata[name] = ToolMeta(
            name=name,
            description=description,
            parameters=parameters or {},
            category=category,
        )

    @classmethod
    def execute(cls, name: str, **kwargs) -> Any:
        """执行指定工具"""
        if name not in cls._tools:
            raise ValueError(f"工具 '{name}' 未注册。可用工具: {list(cls._tools.keys())}")
        return cls._tools[name](**kwargs)

    @classmethod
    def get_tool_descriptions(cls, category: str = None) -> list[dict]:
        """导出工具描述列表（给 LLM 用）"""
        metas = cls._metadata.values()
        if category:
            metas = [m for m in metas if m.category == category]
        return [m.to_llm_schema() for m in metas]

    @classmethod
    def get_tool_names(cls, category: str = None) -> list[str]:
        """获取工具名列表"""
        if category:
            return [n for n, m in cls._metadata.items() if m.category == category]
        return list(cls._tools.keys())

    @classmethod
    def get_router_prompt(cls) -> str:
        """
        生成给 Router LLM 的工具描述 prompt。
        若已用 register_route 注册路由，优先用路由表；否则用 _metadata。
        """
        if cls._routes:
            lines = ["根据用户问题选择最相关的模块（只输出模块 key，逗号分隔）：\n"]
            for k, v in cls._routes.items():
                lines.append(f"- {k}: {v.get('name', k)} — {v.get('description', '')}")
            return "\n".join(lines)
        lines = ["以下是可用的分析工具："]
        for cat in ["analysis", "trading", "risk", "brain", "system"]:
            tools = [(n, m) for n, m in cls._metadata.items() if m.category == cat]
            if tools:
                lines.append(f"\n## {cat.upper()}")
                for name, meta in tools:
                    lines.append(f"- **{name}**: {meta.description}")
        return "\n".join(lines)

    @classmethod
    def summary(cls) -> dict:
        """工具注册统计"""
        from collections import Counter
        cats = Counter(m.category for m in cls._metadata.values())
        return {"total": len(cls._tools), "by_category": dict(cats)}


# ═══════════════════════════════════════════
# Context Compaction — 上下文压缩
# ═══════════════════════════════════════════

class ContextCompactor:
    """
    上下文窗口自动压缩

    当分析链过长（如全市场扫描→板块轮动→个股筛选→回测验证），
    自动对早期消息做摘要，避免超出 LLM context window。
    """

    def __init__(self, max_tokens: int = 8000,
                 summarizer: Optional[Callable] = None):
        """
        Parameters
        ----------
        max_tokens : 最大 token 数（近似值）
        summarizer : 摘要函数，签名 (text: str) -> str
                     如果不提供，使用简单截断
        """
        self.max_tokens = max_tokens
        self.summarizer = summarizer

    def compact(self, messages: list[dict], preserve_recent: int = 5) -> list[dict]:
        """
        压缩消息列表

        Parameters
        ----------
        messages : 消息列表 [{"role": "...", "content": "..."}]
        preserve_recent : 保留最近 N 条消息不压缩

        Returns
        -------
        压缩后的消息列表
        """
        total_tokens = sum(self._estimate_tokens(m.get("content", "")) for m in messages)

        if total_tokens <= self.max_tokens:
            return messages  # 不需要压缩

        # 分离：需要压缩的早期消息 + 保留的近期消息
        if len(messages) <= preserve_recent:
            return messages

        early = messages[:-preserve_recent]
        recent = messages[-preserve_recent:]

        # 对早期消息做摘要
        early_text = "\n\n".join(
            f"[{m.get('role', 'unknown')}]: {m.get('content', '')}"
            for m in early
        )

        if self.summarizer:
            summary = self.summarizer(early_text)
        else:
            # 简单截断
            summary = early_text[:2000] + "\n...(早期对话已压缩)"

        compressed = [
            {"role": "system", "content": f"[历史对话摘要]\n{summary}"}
        ] + recent

        return compressed

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """粗略估算 token 数（中文约 1.5 字/token）"""
        if not text:
            return 0
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)


# ═══════════════════════════════════════════
# 使用示例
# ═══════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("A股 Agent 基础设施演示")
    print("=" * 60)

    # 1. 初始化 EventLog
    print("\n1. EventLog 初始化")
    # event_log = EventLog(log_dir="data/event_log/")

    # 2. 初始化 Brain
    print("\n2. Brain 初始化")
    # brain = AgentBrain(brain_dir="data/brain/", event_log=event_log)

    # 3. 初始化 ToolCenter
    print("\n3. ToolCenter 注册工具")
    ToolCenter.register(
        name="sector_rotation_analysis",
        func=lambda: "板块轮动分析结果...",
        description="分析当前A股板块轮动状态，返回板块强弱排名和轮动方向",
        category="analysis",
    )
    ToolCenter.register(
        name="canslim_screening",
        func=lambda: "CANSLIM筛选结果...",
        description="使用CANSLIM方法筛选符合条件的A股标的",
        category="analysis",
    )
    ToolCenter.register(
        name="sentiment_analysis",
        func=lambda: "情绪分析结果...",
        description="分析市场情绪，包括新闻情绪、资金流向、技术指标情绪",
        category="analysis",
    )
    ToolCenter.register(
        name="risk_check",
        func=lambda: "风控检查结果...",
        description="检查当前持仓风险，包括止损线、集中度、回撤",
        category="risk",
    )

    print(f"   已注册工具: {ToolCenter.summary()}")
    print(f"   Router Prompt:\n{ToolCenter.get_router_prompt()}")

    # 4. CronEngine
    print("\n4. CronEngine 任务列表")
    cron = CronEngine(jobs_file="data/cron/jobs.json")
    for job in cron.list_jobs():
        status = "✅" if job["enabled"] else "❌"
        print(f"   {status} {job['schedule']} {job['name']} ({job['priority']})")

    # 5. 心跳检查
    print("\n5. 系统心跳")
    heartbeat = cron.heartbeat()
    print(f"   交易日: {heartbeat['trading_day']}")
    print(f"   时段: {heartbeat['session']}")
    print(f"   活跃任务: {heartbeat['active_jobs']}")

    # 6. Context Compaction
    print("\n6. Context Compaction 演示")
    compactor = ContextCompactor(max_tokens=500)
    messages = [
        {"role": "user", "content": "分析一下AI板块" * 50},
        {"role": "assistant", "content": "AI板块分析结果..." * 50},
        {"role": "user", "content": "那消费电子呢" * 50},
        {"role": "assistant", "content": "消费电子分析..." * 50},
        {"role": "user", "content": "综合推荐买什么"},
    ]
    compressed = compactor.compact(messages, preserve_recent=2)
    print(f"   压缩前: {len(messages)} 条消息")
    print(f"   压缩后: {len(compressed)} 条消息")

    print("\n✅ 所有模块演示完成")
