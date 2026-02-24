"""
CronEngine — A股交易定时任务引擎（生产版）
==========================================
适配你的系统架构：通过 Router 执行 AI 任务

集成方式：
  1. 在系统入口初始化 CronEngine，传入 router 实例
  2. 调用 cron.start() 启动后台调度
  3. 每个任务的 prompt 自动注入 Brain 上下文

配置：config.py 中 USE_CRON = True, CRON_DIR = "data/cron"
依赖：schedule（pip install schedule），event_log.py 和 brain.py（可选）
"""

import time
import json
import threading
from datetime import datetime, timedelta
from typing import Callable, Optional, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path

try:
    import schedule
    HAS_SCHEDULE = True
except ImportError:
    HAS_SCHEDULE = False

try:
    from event_log import EventLog, EventType
    HAS_EVENT_LOG = True
except ImportError:
    HAS_EVENT_LOG = False

# 配置
try:
    from config import USE_CRON, CRON_DIR
except ImportError:
    USE_CRON = True
    CRON_DIR = "data/cron"


# ═══════════════════════════════════════════
# A股交易日历
# ═══════════════════════════════════════════

class TradingCalendar:
    """A股交易时段判断"""

    # 如果需要精确节假日，可在此加载交易日历 CSV
    # 当前为简化版：周一到周五 = 交易日

    @staticmethod
    def is_trading_day(dt: datetime = None) -> bool:
        dt = dt or datetime.now()
        return dt.weekday() < 5

    @staticmethod
    def get_session(dt: datetime = None) -> str:
        """
        返回当前时段：
        pre_market / auction / morning / midday_break /
        afternoon / closing_auction / post_market / closed
        """
        dt = dt or datetime.now()
        if not TradingCalendar.is_trading_day(dt):
            return "closed"
        t = dt.hour * 100 + dt.minute
        if t < 915:    return "pre_market"
        if t < 930:    return "auction"
        if t <= 1130:  return "morning"
        if t < 1300:   return "midday_break"
        if t <= 1457:  return "afternoon"
        if t <= 1500:  return "closing_auction"
        return "post_market"


# ═══════════════════════════════════════════
# 任务定义
# ═══════════════════════════════════════════

@dataclass
class CronJob:
    job_id: str
    name: str
    description: str
    schedule_time: str                  # "HH:MM"
    weekdays: list[int] = field(default_factory=lambda: [0,1,2,3,4])
    priority: str = "normal"            # critical / high / normal / low
    enabled: bool = True
    require_trading_day: bool = True
    prompt: str = ""                    # 发给 Router/LLM 的 prompt
    tags: list[str] = field(default_factory=list)
    last_run: Optional[float] = None
    last_result: Optional[str] = None


# ═══════════════════════════════════════════
# 默认任务（A 股全天流程）
# ═══════════════════════════════════════════

DEFAULT_JOBS = [
    {
        "job_id": "pre_market",
        "name": "盘前扫描",
        "description": "隔夜消息、竞价异动、北向预期",
        "schedule_time": "09:00",
        "priority": "high",
        "prompt": (
            "执行盘前扫描：\n"
            "1. 隔夜重大新闻和公告（尤其持仓股相关）\n"
            "2. 集合竞价高开/低开异动\n"
            "3. 北向资金预期\n"
            "4. 今日操作计划\n"
            "{brain_context}"
        ),
        "tags": ["盘前", "新闻", "竞价"],
    },
    {
        "job_id": "morning_scan",
        "name": "早盘动量",
        "description": "开盘30分钟后量价异动和板块强度",
        "schedule_time": "10:00",
        "priority": "high",
        "prompt": (
            "执行早盘动量扫描：\n"
            "1. 量价异动个股\n"
            "2. 板块强弱排名\n"
            "3. 是否有板块突然启动\n"
            "4. 需要调整操作计划吗\n"
            "{brain_context}"
        ),
        "tags": ["早盘", "动量", "板块"],
    },
    {
        "job_id": "risk_check",
        "name": "盘中风控",
        "description": "持仓风险检查",
        "schedule_time": "10:30",
        "priority": "critical",
        "prompt": (
            "执行风控检查：\n"
            "1. 持仓股是否触及止损线\n"
            "2. 仓位是否超限\n"
            "3. 板块集中度检查\n"
            "4. 账户回撤检查\n"
            "如有风险立即告警。"
        ),
        "tags": ["风控", "止损"],
    },
    {
        "job_id": "midday_review",
        "name": "午间复盘",
        "description": "上午走势总结",
        "schedule_time": "11:35",
        "priority": "normal",
        "prompt": (
            "午间简要复盘：\n"
            "1. 上午大盘走势（指数、成交额、涨跌比）\n"
            "2. 板块表现\n"
            "3. 持仓股上午表现\n"
            "4. 下午关注点\n"
            "{brain_context}"
        ),
        "tags": ["午盘", "复盘"],
    },
    {
        "job_id": "closing_scan",
        "name": "尾盘扫描",
        "description": "尾盘异动和明日布局",
        "schedule_time": "14:45",
        "priority": "high",
        "prompt": (
            "尾盘扫描：\n"
            "1. 尾盘拉升/跳水个股和板块\n"
            "2. 今日涨停板梳理\n"
            "3. 明日方向预判\n"
            "{brain_context}"
        ),
        "tags": ["尾盘", "异动", "布局"],
    },
    {
        "job_id": "post_market",
        "name": "盘后复盘",
        "description": "全天复盘，更新认知状态",
        "schedule_time": "15:30",
        "priority": "critical",
        "prompt": (
            "盘后全面复盘：\n"
            "1. 今日大盘总结\n"
            "2. 板块轮动分析\n"
            "3. 持仓股逐一点评\n"
            "4. 今日操作回顾（对错分析）\n"
            "5. 更新市场情绪判断\n"
            "6. 明日操作计划\n"
            "7. 将重要洞察存入长期记忆\n"
            "{brain_context}"
        ),
        "tags": ["复盘", "总结", "板块轮动", "情绪"],
    },
    {
        "job_id": "weekly_review",
        "name": "周度复盘",
        "description": "每周五深度复盘",
        "schedule_time": "20:00",
        "weekdays": [4],
        "priority": "critical",
        "prompt": (
            "周度深度复盘：\n"
            "1. 本周大盘走势\n"
            "2. 板块轮动路径总结\n"
            "3. 本周交易回顾和胜率统计\n"
            "4. 策略有效性评估\n"
            "5. 需要调整的参数\n"
            "6. 下周展望\n"
            "{brain_context}\n{win_rate}"
        ),
        "tags": ["周报", "策略评估"],
    },
]


# ═══════════════════════════════════════════
# CronEngine 主类
# ═══════════════════════════════════════════

class CronEngine:
    """
    A 股定时任务引擎

    执行方式：
      (a) ai_executor: 传入你的 Router 实例或任何 (prompt: str) -> str 的函数
      (b) 手动触发: cron.execute("post_market", force=True)
    """

    def __init__(self, jobs_file: str = None,
                 event_log: Any = None,
                 brain: Any = None,
                 ai_executor: Callable = None,
                 on_result: Callable = None):
        """
        Parameters
        ----------
        jobs_file : 任务持久化路径
        event_log : EventLog 实例
        brain : AgentBrain 实例（用于注入上下文）
        ai_executor : AI 执行函数，签名 (prompt: str) -> str
                      典型用法：传入 lambda prompt: router.ask(prompt)
        on_result : 结果回调，签名 (job_id: str, result: str) -> None
                    用于推送到微信/钉钉/Telegram
        """
        self.jobs_file = Path(jobs_file or CRON_DIR) / "jobs.json"
        self.jobs_file.parent.mkdir(parents=True, exist_ok=True)
        self.event_log = event_log
        self.brain = brain
        self.ai_executor = ai_executor
        self.on_result = on_result
        self.calendar = TradingCalendar()

        self._jobs: dict[str, CronJob] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._load_jobs()

    # ─────────────────────────────────────────
    # 任务管理
    # ─────────────────────────────────────────

    def add_job(self, job: CronJob):
        self._jobs[job.job_id] = job
        self._save_jobs()

    def enable(self, job_id: str):
        if job_id in self._jobs:
            self._jobs[job_id].enabled = True
            self._save_jobs()

    def disable(self, job_id: str):
        if job_id in self._jobs:
            self._jobs[job_id].enabled = False
            self._save_jobs()

    def list_jobs(self) -> list[dict]:
        result = []
        for j in self._jobs.values():
            result.append({
                "job_id": j.job_id,
                "name": j.name,
                "time": j.schedule_time,
                "priority": j.priority,
                "enabled": j.enabled,
                "last_run": (
                    datetime.fromtimestamp(j.last_run).strftime("%m-%d %H:%M")
                    if j.last_run else "—"
                ),
            })
        return sorted(result, key=lambda x: x["time"])

    # ─────────────────────────────────────────
    # 执行
    # ─────────────────────────────────────────

    def execute(self, job_id: str, force: bool = False) -> Optional[str]:
        """
        执行指定任务

        force=True 跳过交易日和启用检查（手动触发时用）
        """
        job = self._jobs.get(job_id)
        if not job:
            return f"任务不存在: {job_id}"
        if not force and not job.enabled:
            return None
        if not force and job.require_trading_day and not self.calendar.is_trading_day():
            return None

        self._emit("system.cron_fire", {
            "job_id": job_id, "job_name": job.name, "priority": job.priority,
        })

        start = time.time()
        result = None
        try:
            prompt = self._build_prompt(job)
            if self.ai_executor:
                result = self.ai_executor(prompt)
            else:
                result = f"[无执行器] prompt 长度: {len(prompt)} 字符"
        except Exception as e:
            result = f"执行失败: {e}"
            self._emit("system.error", {"job_id": job_id, "error": str(e)})

        duration = time.time() - start
        job.last_run = time.time()
        job.last_result = str(result)[:2000] if result else None
        self._save_jobs()

        self._emit("system.cron_complete", {
            "job_id": job_id, "job_name": job.name,
            "duration_sec": round(duration, 2),
            "result_preview": str(result)[:300] if result else "",
        })

        # 结果回调（推送通知等）
        if self.on_result and result:
            try:
                self.on_result(job_id, result)
            except Exception:
                pass

        return result

    def execute_all_due(self):
        """执行当前时间点应该运行的所有任务（供外部定时器调用）"""
        now = datetime.now()
        if not self.calendar.is_trading_day(now):
            return

        current_time = now.strftime("%H:%M")
        today_weekday = now.weekday()

        for job in self._jobs.values():
            if not job.enabled:
                continue
            if today_weekday not in job.weekdays:
                continue
            if job.schedule_time != current_time:
                continue
            # 避免同一分钟重复执行
            if job.last_run:
                last = datetime.fromtimestamp(job.last_run)
                if last.date() == now.date() and last.strftime("%H:%M") == current_time:
                    continue

            print(f"⏰ [{current_time}] 执行: {job.name}")
            self.execute(job.job_id)

    def _build_prompt(self, job: CronJob) -> str:
        """构建 prompt，自动注入 Brain 上下文"""
        prompt = job.prompt

        # Brain 上下文
        brain_context = ""
        if self.brain:
            try:
                brain_context = self.brain.get_context_for_prompt(
                    question=" ".join(job.tags),
                    include_emotion=True,
                    include_memories=True,
                    include_recent_commits=True,
                )
            except Exception:
                pass
        prompt = prompt.replace("{brain_context}", brain_context)

        # 胜率统计（周报用）
        win_rate = ""
        if self.brain and "{win_rate}" in prompt:
            try:
                stats = self.brain.get_win_rate(days_back=7)
                o = stats["overall"]
                win_rate = f"本周: {o['total']}笔交易, 胜率{o['rate']:.0%}"
            except Exception:
                pass
        prompt = prompt.replace("{win_rate}", win_rate)

        # 时段标注
        session = self.calendar.get_session()
        header = f"[{datetime.now().strftime('%H:%M')} | {session}]"

        return f"{header}\n\n{prompt}"

    # ─────────────────────────────────────────
    # 调度器
    # ─────────────────────────────────────────

    def start(self):
        """启动后台定时调度"""
        if self._running:
            return
        if not HAS_SCHEDULE:
            print("⚠️ 未安装 schedule 库: pip install schedule")
            print("   CronEngine 仅支持手动触发: cron.execute('job_id', force=True)")
            return

        self._running = True

        for job in self._jobs.values():
            if job.enabled:
                schedule.every().day.at(job.schedule_time).do(
                    self._safe_execute, job_id=job.job_id
                )

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

        active = sum(1 for j in self._jobs.values() if j.enabled)
        print(f"✅ CronEngine 已启动 ({active} 个活跃任务)")

    def stop(self):
        self._running = False
        if HAS_SCHEDULE:
            schedule.clear()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self):
        while self._running:
            if HAS_SCHEDULE:
                schedule.run_pending()
            time.sleep(30)

    def _safe_execute(self, job_id: str):
        job = self._jobs.get(job_id)
        if not job or not job.enabled:
            return
        if datetime.now().weekday() not in job.weekdays:
            return
        if job.require_trading_day and not self.calendar.is_trading_day():
            return
        print(f"⏰ 执行: {job.name} ({job_id})")
        self.execute(job_id)

    # ─────────────────────────────────────────
    # 心跳
    # ─────────────────────────────────────────

    def heartbeat(self) -> dict:
        now = datetime.now()
        status = {
            "time": now.strftime("%Y-%m-%d %H:%M"),
            "trading_day": self.calendar.is_trading_day(),
            "session": self.calendar.get_session(),
            "running": self._running,
            "active_jobs": sum(1 for j in self._jobs.values() if j.enabled),
        }
        self._emit("system.heartbeat", status)
        return status

    # ─────────────────────────────────────────
    # 内部
    # ─────────────────────────────────────────

    def _emit(self, event_type: str, payload: dict):
        if self.event_log and HAS_EVENT_LOG:
            try:
                self.event_log.emit(event_type, payload, source="cron")
            except Exception:
                pass

    def _load_jobs(self):
        if self.jobs_file.exists():
            try:
                data = json.loads(self.jobs_file.read_text(encoding="utf-8"))
                for item in data:
                    self._jobs[item["job_id"]] = CronJob(**item)
                return
            except Exception:
                pass
        # 加载默认
        for item in DEFAULT_JOBS:
            self._jobs[item["job_id"]] = CronJob(**item)
        self._save_jobs()

    def _save_jobs(self):
        data = []
        for j in self._jobs.values():
            d = asdict(j)
            d.pop("last_result", None)
            data.append(d)
        self.jobs_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
