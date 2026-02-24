"""
auto_scheduler.py — CronEngine 自动调度守护进程
=================================================
运行后自动按 A 股交易时间表执行定时任务，无需手动触发。

启动方式：
  python3 auto_scheduler.py              # 前台运行（Ctrl+C 停止）
  python3 auto_scheduler.py --daemon     # 后台运行（nohup）
  python3 auto_scheduler.py --status     # 查看状态
  python3 auto_scheduler.py --run <id>   # 手动触发某个任务
  python3 auto_scheduler.py --list       # 列出所有任务

依赖：pip install schedule
"""

import os
import sys
import time
import signal
import json
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

# ═══════════════════════════════════════════
# 日志
# ═══════════════════════════════════════════

LOG_DIR = Path("data/scheduler")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "scheduler.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("scheduler")

# PID 文件（用于状态查询和停止）
PID_FILE = LOG_DIR / "scheduler.pid"
STATUS_FILE = LOG_DIR / "scheduler_status.json"


# ═══════════════════════════════════════════
# 初始化系统组件
# ═══════════════════════════════════════════

def init_system():
    """初始化所有系统组件，返回 (cron, brain, event_log)"""
    # EventLog
    try:
        from event_log import EventLog
        event_log = EventLog()
        log.info("✅ EventLog 已加载")
    except Exception as e:
        log.warning(f"⚠️ EventLog 加载失败: {e}")
        event_log = None

    # Brain
    try:
        from brain import get_brain
        brain = get_brain(event_log=event_log)
        mem = getattr(brain, "memory_count", None)
        log.info(f"✅ Brain 已加载" + (f" (记忆: {mem} 条)" if mem is not None else ""))
    except Exception as e:
        log.warning(f"⚠️ Brain 加载失败: {e}")
        brain = None

    # LLM Client
    try:
        from llm_client import get_llm
        llm = get_llm()
        log.info("✅ DeepSeek LLM 已连接")
    except Exception as e:
        log.warning(f"⚠️ LLM 加载失败: {e}")
        llm = None

    # AI Executor — 连接 Router 或直接用 LLM
    ai_executor = None
    if llm:
        # 深度任务用 R1，其他用 V3
        def smart_executor(prompt: str) -> str:
            deep_keywords = ["复盘", "深度分析", "策略评估", "周度", "总结"]
            if any(kw in prompt for kw in deep_keywords):
                log.info("  → 使用 DeepSeek R1 (深度推理)")
                return llm.reason(prompt)
            else:
                log.info("  → 使用 DeepSeek V3 (快速分析)")
                return llm.chat(prompt)
        ai_executor = smart_executor

    # AutoJournal
    on_result = None
    try:
        from auto_journal import AutoJournal
        journal = AutoJournal(brain=brain, event_log=event_log)
        on_result = journal.on_cron_result
        log.info("✅ AutoJournal 已加载")
    except Exception as e:
        log.warning(f"⚠️ AutoJournal 加载失败: {e}")

    # CronEngine
    try:
        from cron_and_tools import CronEngine
        cron = CronEngine(
            event_log=event_log,
            brain=brain,
            ai_executor=ai_executor,
            on_result=on_result,
        )
        log.info(f"✅ CronEngine 已加载 ({len(cron._jobs)} 个任务)")
    except ImportError:
        # 尝试独立版
        from cron_engine import CronEngine
        cron = CronEngine(
            event_log=event_log,
            brain=brain,
            ai_executor=ai_executor,
            on_result=on_result,
        )
        log.info(f"✅ CronEngine 已加载 ({len(cron._jobs)} 个任务)")

    return cron, brain, event_log


# ═══════════════════════════════════════════
# 调度主循环（不依赖 schedule 库）
# ═══════════════════════════════════════════

class Scheduler:
    """
    自动调度器

    每 30 秒检查一次是否有任务需要执行。
    不依赖 schedule 库，用简单的时间匹配。
    比 schedule 更可靠：重启后自动恢复，不会漏任务。
    """

    def __init__(self, cron, brain=None, event_log=None):
        self.cron = cron
        self.brain = brain
        self.event_log = event_log
        self._running = False
        self._executed_today = set()  # 今日已执行的任务，防止重复

    def start(self):
        """启动调度主循环"""
        self._running = True
        self._write_pid()
        self._update_status("running")

        log.info("=" * 50)
        log.info("🚀 自动调度器已启动")
        log.info(f"   PID: {os.getpid()}")
        log.info(f"   日志: {LOG_DIR / 'scheduler.log'}")
        log.info("=" * 50)

        # 注册信号处理
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        # 列出今日任务
        self._print_today_schedule()

        # 主循环
        try:
            while self._running:
                self._tick()
                time.sleep(30)  # 每 30 秒检查一次
        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()

    def stop(self):
        self._running = False

    # ─────────────────────────────────────────
    # 核心：每次 tick 检查并执行
    # ─────────────────────────────────────────

    def _tick(self):
        now = datetime.now()

        # 日期切换：清空今日已执行集合
        today = now.strftime("%Y-%m-%d")
        if not hasattr(self, '_current_date') or self._current_date != today:
            self._current_date = today
            self._executed_today.clear()
            if now.hour < 9:
                log.info(f"📅 新的一天: {today}")
                self._print_today_schedule()

        # 逐个检查任务
        current_hhmm = now.strftime("%H:%M")
        current_weekday = now.weekday()

        for job_id, job in self.cron._jobs.items():
            if not job.enabled:
                continue
            if job_id in self._executed_today:
                continue
            if current_weekday not in job.weekdays:
                continue

            # 交易日检查
            if job.require_trading_day:
                try:
                    if not self.cron.calendar.is_trading_day(now):
                        continue
                except Exception:
                    if current_weekday >= 5:
                        continue

            # 时间匹配（允许 ±1 分钟窗口）
            if self._time_match(current_hhmm, job.schedule_time):
                self._run_job(job_id, job)

        # 每小时更新一次状态文件
        if now.minute == 0 and now.second < 35:
            self._update_status("running")

    def _run_job(self, job_id: str, job):
        """执行一个任务"""
        self._executed_today.add(job_id)

        log.info("")
        log.info(f"⏰ ═══ 执行: {job.name} ({job_id}) ═══")
        log.info(f"   时间: {datetime.now().strftime('%H:%M:%S')}")
        log.info(f"   优先级: {job.priority}")

        start = time.time()
        try:
            # 使用 CronEngine 的执行方法
            if hasattr(self.cron, 'execute_job'):
                result = self.cron.execute_job(job_id, force=True)
            else:
                result = self.cron.execute(job_id, force=True)

            duration = time.time() - start
            result_preview = str(result)[:200] if result else "(空)"

            log.info(f"   ✅ 完成 ({duration:.1f}秒)")
            log.info(f"   结果: {result_preview}")

        except Exception as e:
            duration = time.time() - start
            log.error(f"   ❌ 失败 ({duration:.1f}秒): {e}")

        # 更新状态
        self._update_status("running", last_job=job_id, last_time=datetime.now().isoformat())

    @staticmethod
    def _time_match(current: str, target: str) -> bool:
        """时间匹配，允许 ±1 分钟窗口"""
        try:
            ch, cm = map(int, current.split(":"))
            th, tm = map(int, target.split(":"))
            diff = abs((ch * 60 + cm) - (th * 60 + tm))
            return diff <= 1
        except ValueError:
            return False

    # ─────────────────────────────────────────
    # 辅助
    # ─────────────────────────────────────────

    def _print_today_schedule(self):
        now = datetime.now()
        weekday = now.weekday()
        is_trading = weekday < 5  # 简化判断

        log.info(f"\n📋 {'交易日' if is_trading else '非交易日'}任务列表:")
        for job_id, job in sorted(self.cron._jobs.items(), key=lambda x: x[1].schedule_time):
            if not job.enabled:
                continue
            if weekday not in job.weekdays:
                continue
            if job.require_trading_day and not is_trading:
                continue
            status = "✅" if job_id not in self._executed_today else "☑️"
            log.info(f"  {status} {job.schedule_time} | {job.name} [{job.priority}]")
        log.info("")

    def _handle_signal(self, signum, frame):
        log.info(f"\n📛 收到信号 {signum}，正在停止...")
        self._running = False

    def _shutdown(self):
        log.info("🛑 调度器已停止")
        self._update_status("stopped")
        if PID_FILE.exists():
            PID_FILE.unlink()

    def _write_pid(self):
        PID_FILE.write_text(str(os.getpid()))

    def _update_status(self, state: str, **kwargs):
        status = {
            "state": state,
            "pid": os.getpid(),
            "updated_at": datetime.now().isoformat(),
            "today_executed": list(self._executed_today),
            **kwargs,
        }
        STATUS_FILE.write_text(json.dumps(status, ensure_ascii=False, indent=2))


# ═══════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════

def cmd_start(args):
    """启动调度器"""
    if PID_FILE.exists():
        pid = int(PID_FILE.read_text().strip())
        # 检查进程是否还活着
        try:
            os.kill(pid, 0)
            print(f"⚠️ 调度器已在运行 (PID: {pid})")
            print(f"   用 --status 查看状态，或 kill {pid} 停止")
            return
        except ProcessLookupError:
            PID_FILE.unlink()  # 旧 PID 文件，进程已死

    if args.daemon:
        # 后台运行
        print("🚀 以后台模式启动调度器...")
        print(f"   日志: {LOG_DIR / 'scheduler.log'}")
        print(f"   停止: kill $(cat {PID_FILE})")
        # fork 或 nohup
        if os.fork() > 0:
            sys.exit(0)
        os.setsid()

    cron, brain, event_log = init_system()
    scheduler = Scheduler(cron, brain, event_log)
    scheduler.start()


def cmd_status(args):
    """查看调度器状态"""
    if not STATUS_FILE.exists():
        print("调度器未运行（无状态文件）")
        return

    status = json.loads(STATUS_FILE.read_text())
    print(f"状态: {status.get('state', '?')}")
    print(f"PID:  {status.get('pid', '?')}")
    print(f"更新: {status.get('updated_at', '?')}")

    executed = status.get("today_executed", [])
    if executed:
        print(f"今日已执行: {', '.join(executed)}")
    else:
        print("今日已执行: (无)")

    last_job = status.get("last_job")
    if last_job:
        print(f"最近任务: {last_job} @ {status.get('last_time', '?')}")

    # 检查进程是否还活着
    pid = status.get("pid")
    if pid:
        try:
            os.kill(pid, 0)
            print(f"\n进程 {pid} 运行中 ✅")
        except ProcessLookupError:
            print(f"\n进程 {pid} 已不存在 ❌")


def cmd_list(args):
    """列出所有任务"""
    cron, _, _ = init_system()
    jobs = cron.list_jobs()
    print(f"\n共 {len(jobs)} 个任务:")
    print(f"{'ID':<25} {'时间':>5} {'名称':<10} {'优先级':<10} {'状态':<5} {'上次执行'}")
    print("-" * 80)
    for j in jobs:
        status = "✅" if j["enabled"] else "❌"
        print(f"{j['job_id']:<25} {j['time']:>5} {j['name']:<10} {j['priority']:<10} {status:<5} {j['last_run']}")


def cmd_run(args):
    """手动触发一个任务"""
    cron, _, _ = init_system()
    job_id = args.job_id
    log.info(f"手动触发: {job_id}")

    if hasattr(cron, 'execute_job'):
        result = cron.execute_job(job_id, force=True)
    else:
        result = cron.execute(job_id, force=True)

    if result:
        print(f"\n结果:\n{result}")
    else:
        print("(无结果)")


def main():
    parser = argparse.ArgumentParser(
        description="A股交易 CronEngine 自动调度器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python3 auto_scheduler.py              # 前台启动
  python3 auto_scheduler.py --daemon     # 后台启动
  python3 auto_scheduler.py --status     # 查看状态
  python3 auto_scheduler.py --list       # 列出任务
  python3 auto_scheduler.py --run post_market_review   # 手动触发
        """,
    )
    parser.add_argument("--daemon", "-d", action="store_true", help="后台运行")
    parser.add_argument("--status", "-s", action="store_true", help="查看状态")
    parser.add_argument("--list", "-l", action="store_true", help="列出任务")
    parser.add_argument("--run", "-r", dest="job_id", help="手动触发任务")

    args = parser.parse_args()

    if args.status:
        cmd_status(args)
    elif args.list:
        cmd_list(args)
    elif args.job_id:
        cmd_run(args)
    else:
        cmd_start(args)


if __name__ == "__main__":
    main()
