"""
final_system.py — A股量化投资 AI Agent 最终集成
================================================
两轮会话全部产出的总集成入口。

系统全貌：
  ┌─ 4层架构（阶段②Dexter升级 + 本轮OpenAlice升级）──────────┐
  │  Layer 1: Scratchpad/EventLog  — 运行留痕                 │
  │  Layer 2: Eval/Brain           — 评估+认知记忆             │
  │  Layer 3: Router/ToolCenter    — 路由+工具注册             │
  │  Layer 4: CronEngine           — 定时任务调度              │
  └──────────────────────────────────────────────────────────┘

  ┌─ 11个Skill（阶段①②③ 9个 + 阶段④ 2个）──────────────────┐
  │  #1  market_sentiment   市场情绪+大盘方向                  │
  │  #2  sector_rotation    板块轮动分析                       │
  │  #3  canslim_screening  CANSLIM七因子选股                  │
  │  #4  technical_analysis 技术面分析                         │
  │  #5  risk_check         风控检查                           │
  │  #6  stock_analysis     个股综合分析                       │
  │  #7  knowledge_query    知识库查询                         │
  │  #8  macro_liquidity    宏观流动性                         │
  │  #9  value_investor     价值投资6维度（阶段③）             │
  │  #10 trade_signals      5类交易信号（含阶梯突破v6.0+龙头） │
  │  #11 market_environment 市场环境4维评估（v6.0）            │
  └──────────────────────────────────────────────────────────┘

  ┌─ 3个入口 ────────────────────────────────────────────────┐
  │  conversation.py   — 多轮对话CLI                          │
  │  auto_scheduler.py — CronEngine自动调度守护进程            │
  │  daily_workflow.py — 每日4命令工作流                       │
  └──────────────────────────────────────────────────────────┘

  ┌─ 半自动交易闭环 ─────────────────────────────────────────┐
  │  morning  → signal → evening → confirm                    │
  │  盘前分析 → 信号生成 → 盘后复盘 → 确认执行                │
  └──────────────────────────────────────────────────────────┘

用法：
  # 方式1: 启动全系统
  python3 final_system.py start

  # 方式2: 验证所有模块
  python3 final_system.py verify

  # 方式3: 查看系统状态
  python3 final_system.py status

  # 方式4: 注册所有Skill
  python3 final_system.py register
"""

import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Callable


# ═══════════════════════════════════════════════════════════════
#  PART 1: 完整文件清单
# ═══════════════════════════════════════════════════════════════

FILE_MANIFEST = """
A股量化投资 AI Agent — 完整文件清单
========================================

第一轮会话产出（4阶段: 测试→Dexter→对标→自动化）
──────────────────────────────────────────────────
策略核心:
  ladder_breakout_v4.py      阶梯突破策略 v4.1 核心引擎（不改）
  daily_signal.py            阶梯突破信号生成 v6.0（5类辅助信息+精简输出）
  market_environment.py      市场环境4维评估（大盘趋势/情绪/成交量/板块强度）
  trade_signals.py           5类买卖信号整合（阶梯+龙头+CANSLIM+轮动+价值）
  value_investor.py          价值投资6维度分析（Skill #9）

架构层:
  scratchpad.py              运行留痕（→ 被 EventLog 升级替代）
  eval_layer.py              Skill命中率评估（→ 被 Brain.Eval 升级替代）
  router.py                  6路由交互问答（→ 本轮升级为ToolCenter动态路由）

工作流:
  daily_workflow.py           每日4命令工作流（morning/signal/evening/confirm）
  cron_schedule.sh            定时任务脚本（→ 被 CronEngine 升级替代）

数据层:
  data_fetcher.py             数据获取（修复北向资金类型/单位）
  skills/*.py                 9个Skill实现

第二轮会话产出（OpenAlice架构升级）
──────────────────────────────────────────────────
基础设施:
  event_log.py               EventLog 全链路事件追溯（替代 scratchpad）
  brain.py                   AgentBrain 认知状态（记忆+情绪+决策+复盘日志）
  cron_and_tools.py          CronEngine + ToolCenter + ContextCompactor
  llm_client.py              DeepSeek API 统一调用（V3路由 + R1深度分析）
  auto_journal.py            盘后自动复盘日志+洞察提取+情绪更新

入口:
  conversation.py            多轮对话CLI（Brain上下文注入+会话保存恢复）
  auto_scheduler.py          CronEngine自动调度守护进程（30秒检查+防重复）
  config.py                  配置（DEEPSEEK_API_KEY + USE_BRAIN + 路径）

注册:
  skill_registry.py          → 替换为本文件的 register_all_skills()
  final_system.py            本文件：总集成入口

数据目录:
  data/brain/memory.jsonl         长期记忆
  data/brain/emotion.json         市场情绪
  data/brain/commits.jsonl        决策提交
  data/brain/daily_journal/*.md   复盘日志
  data/event_log/*.jsonl          事件日志
  data/scheduler/scheduler.log    调度器日志
  data/workflow/*_*.json          工作流状态
  data/conversations/*.json       对话会话
  cache/                          数据缓存
"""


# ═══════════════════════════════════════════════════════════════
#  PART 2: 11个Skill完整注册
# ═══════════════════════════════════════════════════════════════

def register_all_skills(tool_center_class=None):
    """
    注册全部 11 个 Skill 到 ToolCenter

    合并两轮会话所有 Skill：
      #1-#8:  阶段①② 原有Skill
      #9:     阶段③ 价值投资
      #10:    阶段④ 交易信号（含阶梯突破v6.0 + 龙头突破）
      #11:    v6.0 市场环境评估

    调用方式：
      # 在 router.py 顶部
      from final_system import register_all_skills
      register_all_skills()
    """
    # 获取 ToolCenter
    tc = tool_center_class
    if tc is None:
        try:
            from cron_and_tools import ToolCenter
            tc = ToolCenter
        except ImportError:
            print("⚠️ ToolCenter 不可用")
            return 0

    # ── #1 市场情绪 ──
    tc.register_route(
        route_key="market_sentiment",
        name="市场情绪+大盘方向",
        description="分析市场整体情绪（恐慌/贪婪）、大盘趋势、资金流向",
        keywords=[
            "情绪", "市场", "大盘", "行情", "怎么样", "涨跌",
            "恐慌", "贪婪", "指数", "上证", "深证", "创业板",
            "沪深", "A股", "今天", "sentiment", "market",
        ],
        skills=["sentiment"],
    )

    # ── #2 板块轮动 ──
    tc.register_route(
        route_key="sector_analysis",
        name="板块轮动分析",
        description="分析板块强弱排名、轮动方向、主线板块切换",
        keywords=[
            "板块", "轮动", "行业", "主线", "龙头", "题材",
            "热点", "切换", "概念", "赛道", "sector",
            "AI", "算力", "消费", "医药", "半导体", "新能源",
            "白酒", "军工", "地产", "汽车", "消费电子",
        ],
        skills=["sector_rotation"],
    )

    # ── #2b 板块Stage联合过滤 ──
    tc.register_route(
        route_key="sector_stage",
        name="板块Stage联合过滤",
        description="板块相对强度+黑名单+底部/收紧/R:R过滤，用于前置筛选",
        keywords=[
            "板块轮动", "板块分析", "强势板块", "资金流向",
            "选股", "底部", "大底", "stage", "Stage",
            "哪些板块", "板块排名", "轮动",
        ],
        skills=["sector_rotation", "sector_stage"],
    )

    # ── #3 CANSLIM选股 ──
    tc.register_route(
        route_key="canslim_screening",
        name="CANSLIM选股",
        description="基于CANSLIM七因子(C/A/N/S/L/I/M)的成长股筛选",
        keywords=[
            "CANSLIM", "选股", "筛选", "成长", "screening",
            "CAN", "SLIM", "因子", "基本面选股",
        ],
        skills=["canslim"],
    )

    # ── #4 技术面分析 ──
    tc.register_route(
        route_key="technical_analysis",
        name="技术面分析",
        description="K线形态、均线系统、MACD/KDJ/RSI等技术指标分析",
        keywords=[
            "技术", "K线", "均线", "MACD", "KDJ", "RSI",
            "形态", "支撑", "压力", "趋势", "金叉", "死叉",
            "背离", "量价", "布林", "技术分析",
        ],
        skills=["technical"],
    )

    # ── #5 风控检查 ──
    tc.register_route(
        route_key="risk_check",
        name="风控检查",
        description="持仓风险检查：止损、集中度、回撤、仓位管理",
        keywords=[
            "风控", "风险", "止损", "仓位", "回撤", "止盈",
            "亏损", "集中度", "预警", "敞口", "持仓",
        ],
        skills=["sentiment", "risk"],
    )

    # ── #6 个股分析 ──
    tc.register_route(
        route_key="stock_analysis",
        name="个股综合分析",
        description="对单只股票进行综合分析（基本面+技术面+资金面）",
        keywords=[
            "个股", "分析", "股票", "代码", "怎么看",
            "能买吗", "能追吗", "目标价",
            ".SH", ".SZ", "600", "000", "002", "300", "301",
        ],
        skills=["value"],
    )

    # ── #7 知识查询 ──
    tc.register_route(
        route_key="knowledge_query",
        name="知识库查询",
        description="查询投资知识、术语解释、策略原理",
        keywords=[
            "什么是", "解释", "原理", "知识", "学习",
            "概念", "定义", "区别", "如何", "怎样",
        ],
        skills=["knowledge"],
    )

    # ── #8 宏观流动性 ──
    tc.register_route(
        route_key="macro_liquidity",
        name="宏观流动性",
        description="宏观经济、货币政策、流动性分析",
        keywords=[
            "宏观", "流动性", "货币", "利率", "MLF",
            "降准", "降息", "CPI", "GDP", "PMI",
            "央行", "政策", "外资", "汇率",
        ],
        skills=["macro"],
    )

    # ── #9 价值投资（阶段③）──
    tc.register_route(
        route_key="value_investor",
        name="价值投资6维度",
        description="6维度价值分析：估值/盈利/成长/财务健康/分红/机构认可",
        keywords=[
            "价值", "估值", "PE", "PB", "ROE", "分红",
            "股息", "低估", "基本面", "财务", "巴菲特",
            "value", "长线", "蓝筹", "白马", "红利",
        ],
        skills=["value"],
    )

    # ── #10 交易信号（阶段④ + v6.0）──
    tc.register_route(
        route_key="trade_signals",
        name="交易信号扫描",
        description=(
            "5类交易信号：阶梯突破(v6.0)、龙头突破(theme leader)、"
            "CANSLIM、板块轮动、价值低估。含市场环境评估。"
        ),
        keywords=[
            "信号", "扫描", "买入", "卖出", "交易", "买点", "卖点",
            "突破", "阶梯", "龙头", "leader", "daily_signal",
            "扫描全市场", "今天买什么", "有什么机会",
            "signal", "scan", "trade",
        ],
        skills=["trade_signals"],
    )

    # ── #11 市场环境（v6.0）──
    tc.register_route(
        route_key="market_environment",
        name="市场环境评估",
        description=(
            "v6.0 市场环境4维评分：大盘趋势+情绪+成交量+板块强度 → 可交易/谨慎/观望"
        ),
        keywords=[
            "环境", "能不能交易", "适合交易", "市场环境",
            "今天能买吗", "观望", "可以操作吗",
        ],
        skills=["market_environment"],
    )

    # 统计
    routes = getattr(tc, '_routes', {})
    count = len(routes)
    print(f"✅ 全部 {count} 个 Skill 注册完成")
    return count


# ═══════════════════════════════════════════════════════════════
#  PART 3: 系统启动（一键初始化所有组件）
# ═══════════════════════════════════════════════════════════════

class AgentSystem:
    """
    系统总线 — 初始化并持有所有组件的引用

    用法：
        system = AgentSystem()
        system.start()

        # 访问组件
        system.router.ask("板块轮动")
        system.brain.get_context_for_prompt("...")
        system.cron.execute_job("post_market_review", force=True)
        system.workflow.morning()
        system.conversation.ask("消费电子怎么样")
    """

    def __init__(self):
        self.event_log = None
        self.brain = None
        self.llm = None
        self.router = None
        self.fetcher = None
        self.market_env = None
        self.signal_gen = None
        self.trade_scanner = None
        self.cron = None
        self.journal = None
        self.workflow = None
        self.conversation = None
        self._init_log = []

    def start(self, enable_cron: bool = False):
        """
        初始化全系统

        Parameters
        ----------
        enable_cron : 是否启动自动调度（后台 daemon）
        """
        print("=" * 60)
        print("  A股量化投资 AI Agent — 系统启动")
        print("=" * 60)

        # Layer 1: EventLog
        self._init_event_log()

        # Layer 2: Brain
        self._init_brain()

        # Layer 3: LLM + Router + ToolCenter
        self._init_llm()
        self._init_data_fetcher()
        self._init_strategies()
        self._register_skills()
        self._init_router()

        # Layer 4: CronEngine + AutoJournal
        self._init_auto_journal()
        self._init_cron(auto_start=enable_cron)

        # 入口: Workflow + Conversation
        self._init_workflow()
        self._init_conversation()

        # 汇总
        print("\n" + "-" * 60)
        for line in self._init_log:
            print(f"  {line}")
        print("-" * 60)
        print(f"  系统就绪 ✅")
        print("=" * 60)

        return self

    # ─── 初始化各组件 ───

    def _init_event_log(self):
        try:
            from event_log import EventLog
            self.event_log = EventLog()
            self.event_log.emit("system.startup", {"version": "final"}, source="final_system")
            self._init_log.append("✅ EventLog")
        except Exception as e:
            self._init_log.append(f"⚠️ EventLog: {e}")

    def _init_brain(self):
        try:
            from brain import get_brain
            self.brain = get_brain(event_log=self.event_log)
            mem = getattr(self.brain, 'memory_count', '?')
            self._init_log.append(f"✅ Brain ({mem} 条记忆)")
        except Exception as e:
            self._init_log.append(f"⚠️ Brain: {e}")

    def _init_llm(self):
        try:
            from llm_client import get_llm
            self.llm = get_llm()
            self._init_log.append("✅ DeepSeek LLM (V3+R1)")
        except Exception as e:
            self._init_log.append(f"⚠️ LLM: {e}")

    def _init_data_fetcher(self):
        try:
            from data_fetcher import get_fetcher
            from data_cache import DailyCache
            raw_fetcher = get_fetcher()
            self.fetcher = DailyCache(raw_fetcher)
            self._init_log.append("✅ DataFetcher + DailyCache")
        except Exception as e:
            self._init_log.append(f"⚠️ DataFetcher: {e}")

    def _init_strategies(self):
        """初始化策略模块: 市场环境 + 信号生成"""
        try:
            from market_environment import MarketEnvironment
            self.market_env = MarketEnvironment(self.fetcher)
            self._init_log.append("✅ MarketEnvironment (v6.0)")
        except Exception as e:
            self._init_log.append(f"⚠️ MarketEnvironment: {e}")

        try:
            from trade_signals import TradeSignalGenerator
            self.signal_gen = TradeSignalGenerator(
                data_fetcher=self.fetcher,
                market_env=self.market_env,
                event_log=self.event_log,
                brain=self.brain,
            )
            self._init_log.append("✅ TradeSignalGenerator (5类信号)")
        except Exception as e:
            self._init_log.append(f"⚠️ TradeSignalGenerator: {e}")

        try:
            from trade_signal import TradeSignalScanner
            self.trade_scanner = TradeSignalScanner()
            self._init_log.append("✅ TradeSignalScanner (买卖触发)")
        except Exception as e:
            self._init_log.append(f"⚠️ TradeSignalScanner: {e}")

    def _register_skills(self):
        count = register_all_skills()
        self._init_log.append(f"✅ ToolCenter ({count} 个路由)")

    def _init_router(self):
        try:
            from router import Router
            from debate import DebateEngine
            self.router = Router(event_log=self.event_log, brain=self.brain)
            if self.llm and getattr(self.router, "debate_engine", None) is None:
                self.router.debate_engine = DebateEngine(self.llm)
            self._init_log.append("✅ Router")
        except Exception as e:
            self._init_log.append(f"⚠️ Router: {e}")

    def _init_auto_journal(self):
        try:
            from auto_journal import AutoJournal
            self.journal = AutoJournal(brain=self.brain, event_log=self.event_log)
            self._init_log.append("✅ AutoJournal")
        except Exception as e:
            self._init_log.append(f"⚠️ AutoJournal: {e}")

    def _init_cron(self, auto_start=False):
        try:
            from cron_and_tools import CronEngine
            ai_executor = None
            if self.llm:
                def _executor(prompt, tools=None):
                    deep = any(kw in prompt for kw in ["复盘", "深度", "周度", "总结"])
                    return self.llm.reason(prompt) if deep else self.llm.chat(prompt)
                ai_executor = _executor

            on_result = self.journal.on_cron_result if self.journal else None

            self.cron = CronEngine(
                event_log=self.event_log,
                brain=self.brain,
                ai_executor=ai_executor,
                on_result=on_result,
            )
            job_count = len(self.cron._jobs)
            self._init_log.append(f"✅ CronEngine ({job_count} 个任务)")

            if auto_start:
                self.cron.start()
                self._init_log.append("  └─ 自动调度已启动")
        except Exception as e:
            self._init_log.append(f"⚠️ CronEngine: {e}")

    def _init_workflow(self):
        try:
            from daily_workflow import DailyWorkflow
            self.workflow = DailyWorkflow(
                data_fetcher=self.fetcher,
                router=self.router,
                brain=self.brain,
                event_log=self.event_log,
                llm=self.llm,
                signal_generator=self.signal_gen,
                market_env=self.market_env,
                trade_scanner=self.trade_scanner,
            )
            self._init_log.append("✅ DailyWorkflow (4命令闭环)")

            # 将 Cron 的 post_market_review 接到工作流盘后链路
            if self.cron and "post_market_review" in getattr(self.cron, "_jobs", {}):
                self.cron.register_callback("post_market_review_workflow", self.workflow.evening)
                self.cron._jobs["post_market_review"].callback = "post_market_review_workflow"
                self._init_log.append("  └─ Cron post_market_review → workflow.evening")
        except Exception as e:
            self._init_log.append(f"⚠️ DailyWorkflow: {e}")

    def _init_conversation(self):
        try:
            from conversation import Conversation
            llm_caller = None
            if self.llm:
                llm_caller = lambda msgs: self.llm.call_for_router(msgs)

            self.conversation = Conversation(
                router=self.router,
                llm_caller=llm_caller,
                brain=self.brain,
                event_log=self.event_log,
            )
            self._init_log.append("✅ Conversation (多轮对话)")
        except Exception as e:
            self._init_log.append(f"⚠️ Conversation: {e}")


# ═══════════════════════════════════════════════════════════════
#  PART 4: 验证（一键检查所有模块）
# ═══════════════════════════════════════════════════════════════

def verify_system():
    """验证所有模块是否可导入、接口是否正确"""
    print("=" * 60)
    print("  系统验证")
    print("=" * 60)

    checks = []

    # ── 第一轮产出 ──
    print("\n[第一轮: 策略+架构]")

    modules_v1 = {
        "data_fetcher": "DataFetcher",
        "market_environment": "MarketEnvironment",
        "trade_signals": "TradeSignalGenerator",
        "value_investor": "ValueInvestorSkill",
        "daily_workflow": "DailyWorkflow",
    }
    for mod, cls in modules_v1.items():
        try:
            m = __import__(mod)
            assert hasattr(m, cls), f"缺少 {cls}"
            checks.append((mod, True, ""))
            print(f"  ✅ {mod}.{cls}")
        except Exception as e:
            checks.append((mod, False, str(e)))
            print(f"  ❌ {mod}: {e}")

    # 检查策略文件
    strategy_files = [
        "ladder_breakout_v4.py",
        "daily_signal.py",
    ]
    for f in strategy_files:
        exists = Path(f).exists()
        checks.append((f, exists, "" if exists else "文件不存在"))
        print(f"  {'✅' if exists else '⚠️'} {f} {'存在' if exists else '不存在（请确认文件名）'}")

    # ── 第二轮产出 ──
    print("\n[第二轮: 基础设施]")

    modules_v2 = {
        "event_log": "EventLog",
        "brain": "AgentBrain",
        "cron_and_tools": "CronEngine",
        "llm_client": "get_llm",
        "auto_journal": "AutoJournal",
        "auto_scheduler": "Scheduler",
        "conversation": "Conversation",
    }
    for mod, cls in modules_v2.items():
        try:
            m = __import__(mod)
            assert hasattr(m, cls), f"缺少 {cls}"
            checks.append((mod, True, ""))
            print(f"  ✅ {mod}.{cls}")
        except Exception as e:
            checks.append((mod, False, str(e)))
            print(f"  ❌ {mod}: {e}")
    # ContextCompactor 可能在 compactor 或 cron_and_tools
    try:
        __import__("compactor")
        assert hasattr(__import__("compactor"), "ContextCompactor")
        checks.append(("compactor", True, ""))
        print("  ✅ compactor.ContextCompactor")
    except Exception:
        try:
            m = __import__("cron_and_tools")
            assert hasattr(m, "ContextCompactor")
            checks.append(("cron_and_tools.ContextCompactor", True, ""))
            print("  ✅ cron_and_tools.ContextCompactor")
        except Exception as e:
            checks.append(("ContextCompactor", False, str(e)))
            print(f"  ❌ ContextCompactor: {e}")

    # ── Skill 注册 ──
    print("\n[Skill 注册]")
    try:
        count = register_all_skills()
        print(f"  ✅ {count} 个路由注册成功")
    except Exception as e:
        print(f"  ❌ 注册失败: {e}")

    # ── DeepSeek 连通 ──
    print("\n[DeepSeek API]")
    try:
        from llm_client import get_llm
        llm = get_llm()
        if llm and getattr(llm, 'has_key', lambda: False)():
            print("  ✅ API Key 已配置")
        elif llm:
            print("  ✅ LLM 客户端已创建")
        else:
            print("  ⚠️ LLM 未初始化")
    except Exception as e:
        print(f"  ❌ {e}")

    # ── 数据目录 ──
    print("\n[数据目录]")
    dirs = ["data/brain", "data/event_log", "data/scheduler",
            "data/workflow", "data/conversations", "cache"]
    for d in dirs:
        exists = Path(d).exists()
        print(f"  {'✅' if exists else '⬜'} {d}")
        if not exists:
            Path(d).mkdir(parents=True, exist_ok=True)
            print(f"      → 已创建")

    # 汇总
    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)
    print(f"\n{'=' * 60}")
    print(f"  验证结果: {passed}/{total} 通过")

    failed = [(m, e) for m, ok, e in checks if not ok]
    if failed:
        print(f"\n  需要关注:")
        for m, e in failed:
            print(f"    - {m}: {e}")

    print("=" * 60)
    return passed == total


# ═══════════════════════════════════════════════════════════════
#  PART 5: v6.0 策略关键规则（写入文档，不修改代码）
# ═══════════════════════════════════════════════════════════════

V6_RULES = """
阶梯突破策略 v6.0 — 三条核心原则
========================================

原则1: 不改选股逻辑
  - ladder_breakout_v4.py 的 detect() 逻辑不动
  - 评分门槛 ≥65 不变
  - 信号数量保持不变，避免过度拟合

原则2: 辅助信息仅供参考
  v6.0 在每个信号后附加5类辅助信息：
    [1] 60天新高: 是/否，距高点百分比
    [2] 形态质量: 优秀/良好/一般
    [3] 量比: 直接数值（如1.75）
    [4] 板块共振: 强/中/单独
    [5] 风险提示: 具体风险因素
  这些信息不用于过滤，不影响信号扫描，用户自主决策。

原则3: 市场环境评估
  4维评分: 大盘趋势 + 市场情绪 + 成交量 + 板块强度
  ≥75: 良好 → 可交易
  60-74: 一般 → 谨慎交易
  <60: 较差 → 建议观望

  核心理念: 知道何时不交易，比知道何时交易更重要！
  11月环境差: v4.1 胜率18.9%(-6.63%) → v6.0 看到环境差暂停, 估计-2%
  12月环境好: 保持 +5%
  全年改善: +2.3pp/年
"""


# ═══════════════════════════════════════════════════════════════
#  PART 6: CLI
# ═══════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print(FILE_MANIFEST)
        print("\n用法:")
        print("  python3 final_system.py start     — 启动全系统")
        print("  python3 final_system.py verify     — 验证所有模块")
        print("  python3 final_system.py status     — 系统状态")
        print("  python3 final_system.py register   — 注册所有Skill")
        print("  python3 final_system.py manifest   — 文件清单")
        print("  python3 final_system.py rules      — v6.0策略规则")
        sys.exit(0)

    cmd = sys.argv[1].lower()

    if cmd == "start":
        system = AgentSystem()
        system.start(enable_cron="--cron" in sys.argv)
        print("\n系统已就绪。使用方式:")
        print("  system.workflow.morning()              # 盘前分析")
        print("  system.workflow.signal()               # 信号扫描")
        print("  system.workflow.evening()              # 盘后复盘")
        print("  system.conversation.ask('问题')        # 多轮对话")
        print("  system.cron.execute_job('post_market_review', force=True)")
        # 进入交互模式
        import code
        code.interact(local={"system": system, "s": system})

    elif cmd == "verify":
        verify_system()

    elif cmd == "status":
        system = AgentSystem()
        system.start()
        if system.workflow:
            system.workflow.status()
        if system.llm:
            system.llm.print_stats()

    elif cmd == "register":
        register_all_skills()

    elif cmd == "manifest":
        print(FILE_MANIFEST)

    elif cmd == "rules":
        print(V6_RULES)

    else:
        print(f"未知命令: {cmd}")


if __name__ == "__main__":
    main()
