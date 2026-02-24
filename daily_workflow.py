"""
daily_workflow.py — 每日 4 命令工作流（Context Engineering 重构）
================================================================
重构原则（来自 Agent Skills for Context Engineering）：

1. Sub-Agent Partitioning
   每个命令（morning/signal/evening/confirm）独立运行，
   只向 Brain 提交结论，不传递中间过程。

2. Observation Masking
   信号扫描 300 只股票 → 只保留命中的 5-10 只摘要
   环境评分详情 → 只存一行 "74/100 (一般) [intraday]"
   复盘长文 → 提取洞察写入记忆

3. Structured Compaction
   Brain 4 段固定结构（意图/状态/决策/下一步），
   每个命令负责更新自己相关的段。

4. Progressive Disclosure
   Brain.get_context_for_prompt() 分 3 层返回，
   不把全部历史塞进 LLM。

5. Tokens-per-task
   用 EventLog 记录全链路事件（append-only JSONL），
   Brain 只从中读摘要（observation masking）。

用法：
  python3 daily_workflow.py morning     # 盘前
  python3 daily_workflow.py signal      # 信号
  python3 daily_workflow.py evening     # 盘后
  python3 daily_workflow.py confirm     # 确认
  python3 daily_workflow.py full        # 全部按顺序执行
  python3 daily_workflow.py status      # 今日状态
"""

import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


class DailyWorkflow:
    """
    每日工作流管理器（Context Engineering 版）

    每个命令是一个 Sub-Agent：
      独立上下文 → 独立处理 → 只提交结论到 Brain/EventLog
    """

    def __init__(self, data_fetcher=None, router=None, brain=None,
                 event_log=None, llm=None, signal_generator=None,
                 market_env=None, news_service=None, notifier=None):
        self.fetcher = data_fetcher
        self.router = router
        self.brain = brain
        self.event_log = event_log
        self.llm = llm
        self.signal_gen = signal_generator
        self.market_env = market_env
        self.news_service = news_service
        self.notifier = notifier

        self.status_dir = Path("data/workflow")
        self.status_dir.mkdir(parents=True, exist_ok=True)
        self.dashboard_dir = Path("data/dashboard")
        self.dashboard_dir.mkdir(parents=True, exist_ok=True)
        self._today = datetime.now().strftime("%Y-%m-%d")

    # ═══════════════════════════════════════════
    # 实时数据自动获取 (v6.1)
    # ═══════════════════════════════════════════

    def _is_trading_hours(self) -> bool:
        """判断当前是否在交易时段（9:15-15:05）"""
        now = datetime.now()
        if now.weekday() >= 5:
            return False
        hour_min = now.hour * 100 + now.minute
        return 915 <= hour_min <= 1505

    def _get_intraday_auto(self) -> Optional[dict]:
        """盘中自动获取实时数据"""
        if not self._is_trading_hours():
            return None
        try:
            from realtime_fetcher import get_intraday
            data = get_intraday(source="auto", verbose=True)
            if data and data.get("up_count"):
                return data
        except ImportError:
            print("  ⚠️ realtime_fetcher 未安装，盘中数据跳过")
        except Exception as e:
            print(f"  ⚠️ 实时数据获取失败: {e}")
        return None

    def _evaluate_env_smart(self) -> Optional[dict]:
        """智能环境评估（盘中用实时，盘后用 Tushare）"""
        if not self.market_env:
            return None
        today = datetime.now().strftime("%Y%m%d")
        intraday = self._get_intraday_auto()
        try:
            return self.market_env.evaluate(
                date=today, intraday=intraday, diagnose=True)
        except TypeError:
            try:
                return self.market_env.evaluate(date=today)
            except Exception:
                return None
        except Exception as e:
            print(f"  ⚠️ 环境评估失败: {e}")
            return None

    # ═══════════════════════════════════════════
    # 命令 1: 盘前分析 (Sub-Agent)
    # ═══════════════════════════════════════════

    def morning(self) -> str:
        """
        盘前分析（建议 08:30 运行）

        Sub-Agent: 独立评估 → 只向 Brain 提交结论
        """
        self._log_step("morning", "start")
        print("\n" + "=" * 50)
        print("☀️ 盘前分析")
        print("=" * 50)

        # Brain: 设置今日意图
        if self.brain:
            self.brain.set_intent("盘前分析 → 判断今日是否适合交易")

        sections = []

        # ── Sub-task 1: 环境评估 ──
        env_result = self._evaluate_env_smart()
        if env_result:
            report = self.market_env.format_report(env_result)
            sections.append(report)

            # Observation Masking: 只存一行结论到 Brain
            if self.brain:
                self.brain.update_status(
                    "环境评分",
                    f"{env_result['total_score']}/100 ({env_result['level']}) [{env_result.get('source', '?')}]"
                )

        # ── Sub-task 2: Brain 相关记忆 ──
        brain_ctx = ""
        if self.brain:
            brain_ctx = self.brain.get_context_for_prompt("盘前 消息 北向 计划")
            if brain_ctx:
                sections.append(brain_ctx)

        # ── Sub-task 3: LLM/Router 分析 ──
        analysis = self._run_analysis(
            "执行盘前分析：\n"
            "1. 隔夜重大新闻和政策\n"
            "2. 持仓股相关公告\n"
            "3. 北向资金预期\n"
            "4. 今日操作计划\n",
            brain_ctx
        )
        if analysis:
            sections.append(analysis)
            # Observation Masking: 学习结论不存原文
            if self.brain:
                self.brain.learn_from_output("morning", "盘前分析", analysis)

        result = "\n\n".join(sections) if sections else "无分析结果"

        # Brain: 设置下一步
        if self.brain:
            self.brain.set_next_step("运行 signal 命令扫描信号")

        self._log_step("morning", "done", preview=result[:200])
        self._save_status("morning", result)
        print(result)
        return result

    # ═══════════════════════════════════════════
    # 命令 2: 信号生成 (Sub-Agent)
    # ═══════════════════════════════════════════

    def signal(self) -> str:
        """
        信号生成（建议 09:45-10:00 或 15:00 后运行）

        Sub-Agent 模式：
        1. 独立评估环境
        2. 独立扫描信号
        3. Observation Masking: 300 只 → 只保留命中摘要
        4. 向 Brain 提交: 环境分 + 命中信号列表
        """
        self._log_step("signal", "start")
        print("\n" + "=" * 50)
        print("📊 信号扫描")
        print("=" * 50)

        # Brain: 更新意图
        if self.brain:
            self.brain.set_intent("信号扫描 → 寻找交易机会")

        # ── Step 1: 环境评估 ──
        env_result = self._evaluate_env_smart()
        env_score = 65  # 默认
        if env_result:
            env_score = env_result.get("total_score", 65)
            # 完整报告输出
            print(f"\n{self.market_env.format_report(env_result)}")

            if env_score < 60:
                print(f"\n  ⚠️ 环境较差({env_score}/100)，信号仅供观察，建议不操作")
            elif env_score < 75:
                print(f"  📋 环境一般({env_score}/100)，谨慎对待信号，轻仓试探")
            else:
                print(f"  ✅ 环境良好({env_score}/100)，信号可正常执行")
            print()

            # Observation Masking → Brain
            if self.brain:
                self.brain.update_status(
                    "环境评分",
                    f"{env_score}/100 ({env_result['level']}) [{env_result.get('source', '?')}]"
                )

        # ── Step 2: 信号扫描 ──
        if self.signal_gen:
            try:
                raw_result = self.signal_gen.scan_all()
                signals = raw_result.get("signals", [])

                # ── 新闻舆情验证（可选）──
                if self.news_service and signals:
                    print("  📰 搜索信号股票新闻...")
                    signals = self.news_service.enrich_signals(signals)
                    raw_result["signals"] = signals

                output = self.signal_gen.format_dashboard(raw_result, env_score)

                # Observation Masking: 从 300 只扫描结果中只提取命中信号
                hit_count = len(signals)
                hit_summary = self._mask_signal_output(signals)

                # Brain: 只存命中摘要
                if self.brain:
                    self.brain.learn_from_output(
                        "trade_signals", "信号扫描", hit_summary)
                    self.brain.update_status(
                        "今日信号", f"{hit_count}个命中")

                    # 记录决策建议
                    if env_score < 60:
                        self.brain.record_decision(
                            f"环境{env_score}分，{hit_count}个信号仅观察",
                            "环境较差不操作")
                    elif hit_count > 0:
                        self.brain.record_decision(
                            f"发现{hit_count}个信号，环境{env_score}分",
                            "待 confirm 审核")

                self._save_status("signal", output, data=raw_result)
                self._save_dashboard(output)
                print(output)

                # ── 推送（可选）──
                if self.notifier:
                    push_result = self.notifier.push(output, title=f"📊 {self._today} 信号仪表盘")
                    pushed = [k for k, v in push_result.items() if v]
                    if pushed:
                        print(f"  📱 已推送: {', '.join(pushed)}")

                print("\n" + "=" * 50)
                print("  ✅ signal 扫描完成")
                print("=" * 50)
                return output

            except Exception as e:
                msg = f"信号扫描失败: {e}"
                print(msg)
                return msg
        else:
            # 无独立 signal generator，通过 Router
            if self.router:
                try:
                    if hasattr(self.router, 'answer'):
                        result = self.router.answer(
                            "执行全量信号扫描，包括阶梯突破、龙头突破、CANSLIM、板块轮动、价值低估")
                    else:
                        result = "Router 无可用接口"
                    output = str(result)
                    self._save_status("signal", output)
                    print(output)
                    print("\n" + "=" * 50)
                    print("  ✅ signal 扫描完成")
                    print("=" * 50)
                    return output
                except Exception as e:
                    msg = f"Router 信号扫描失败: {e}"
                    print(msg)
                    return msg

        # Brain: 设置下一步
        if self.brain:
            self.brain.set_next_step("收盘后运行 evening 复盘", "审核信号运行 confirm")

        return "无信号生成器"

    def _mask_signal_output(self, signals: list) -> str:
        """
        Observation Masking: 将信号列表压缩为一行摘要

        输入: [{symbol, type, score, ...}, ...]（可能很长）
        输出: "命中3只: 000858.SZ(阶梯82), 601012.SH(龙头75), ..."
        """
        if not signals:
            return "无命中信号"

        parts = []
        for s in signals[:10]:  # 最多显示 10 只
            sym = s.get("symbol", "?")
            sig_type = s.get("type", "?")[:2]
            score = s.get("score", "?")
            parts.append(f"{sym}({sig_type}{score})")

        summary = f"命中{len(signals)}只: {', '.join(parts)}"
        if len(signals) > 10:
            summary += f" ...等{len(signals)}只"
        return summary

    # ═══════════════════════════════════════════
    # 命令 3: 盘后更新 (Sub-Agent)
    # ═══════════════════════════════════════════

    def evening(self) -> str:
        """
        盘后更新（建议 15:30 后运行）

        Sub-Agent 模式:
        1. 收集今日所有 Sub-Agent 结论
        2. LLM 深度复盘
        3. Observation Masking: 提取洞察到 Brain 记忆
        4. 更新情绪 + 写日志
        """
        self._log_step("evening", "start")
        print("\n" + "=" * 50)
        print("🌙 盘后复盘")
        print("=" * 50)

        # Brain: 更新意图
        if self.brain:
            self.brain.set_intent("盘后复盘 → 总结今日 + 规划明日")

        # ── 收集今日各 Sub-Agent 的结论（不是原始数据）──
        signal_summary = self._load_status("signal") or "未扫描"
        morning_summary = self._load_status("morning") or "未分析"

        # Brain 上下文（只取相关记忆）
        brain_ctx = ""
        if self.brain:
            brain_ctx = self.brain.get_context_for_prompt("复盘 总结 板块 操作回顾")

        # ── 收盘后重新评估环境（纯 Tushare，对比盘中）──
        env_post = None
        if self.market_env:
            try:
                today = datetime.now().strftime("%Y%m%d")
                env_post = self.market_env.evaluate(date=today, diagnose=True)
                if env_post:
                    report = self.market_env.format_report(env_post)
                    print(f"\n收盘后环境评分:\n{report}\n")

                    if self.brain:
                        self.brain.update_status(
                            "收盘评分",
                            f"{env_post['total_score']}/100 ({env_post['level']}) [tushare]"
                        )
            except Exception:
                pass

        # ── LLM 复盘 ──
        prompt = (
            "盘后全面复盘：\n"
            "1. 今日大盘总结（指数、成交额、涨跌比）\n"
            "2. 板块轮动分析\n"
            "3. 持仓股逐一点评\n"
            "4. 今日操作回顾（对错分析）\n"
            "5. 更新市场情绪判断（贪婪恐惧 0-1）\n"
            "6. 明日操作计划\n"
            "7. 提炼1-2条可复用的交易教训\n"
        )
        if brain_ctx:
            prompt += f"\n{brain_ctx}"
        if signal_summary:
            prompt += f"\n\n今日信号摘要:\n{signal_summary[:500]}"

        result = self._run_analysis(prompt, "")
        if not result:
            result = "未配置 LLM/Router，无法复盘"

        # ── Brain: Observation Masking — 提取洞察 ──
        if self.brain and result:
            # 写日志（完整保存，但 Brain 只学摘要）
            try:
                self.brain.write_journal(result)
            except Exception:
                pass

            # 学习（Observation Masking: 从长文提取一行洞察）
            try:
                self.brain.learn_from_output("evening", "盘后复盘", result)
            except Exception:
                pass

            # 设置明日下一步
            self.brain.set_next_step("明早运行 morning 盘前分析")

        self._log_step("evening", "done", preview=result[:200])
        self._save_status("evening", result)
        print(result)

        if self.notifier:
            push_result = self.notifier.push(result, title=f"🌙 {self._today} 盘后复盘")
            pushed = [k for k, v in push_result.items() if v]
            if pushed:
                print(f"  📱 已推送: {', '.join(pushed)}")
        return result

    # ═══════════════════════════════════════════
    # 命令 4: 确认执行
    # ═══════════════════════════════════════════

    def confirm(self, actions: list = None) -> str:
        """
        确认执行（人工审核后运行）

        Parameters
        ----------
        actions : [{"action": "BUY", "symbol": "600219.SH", "reason": "阶梯突破"}, ...]
        """
        self._log_step("confirm", "start")
        print("\n" + "=" * 50)
        print("✅ 确认执行")
        print("=" * 50)

        if not actions:
            # 从今日信号中提取待确认的
            signal_data = self._load_status("signal", raw=True)
            if signal_data and isinstance(signal_data, dict):
                signals = signal_data.get("signals", [])
                if signals:
                    print(f"今日共 {len(signals)} 个信号待审核:")
                    for i, sig in enumerate(signals[:10], 1):
                        print(f"  [{i}] {sig.get('action', '?')} {sig.get('name', '?')} "
                              f"({sig.get('symbol', '?')}) 评分{sig.get('score', '?')}")
                    print("\n请指定要执行的操作，或调用:")
                    print("  workflow.confirm([{'action': 'BUY', 'symbol': '600219.SH', 'reason': '...'}])")
                    return "等待确认"
                else:
                    print("今日无信号")
                    return "无信号"
            else:
                print("今日未生成信号，请先运行 signal")
                return "未生成信号"

        # 获取当前环境评分
        env_score = 0
        if self.brain:
            status = self.brain._state.get("status", {})
            env_str = status.get("环境评分", "")
            try:
                env_score = int(env_str.split("/")[0])
            except Exception:
                pass

        # 执行确认
        committed = []
        for act in actions:
            symbol = act.get("symbol", "")
            action = act.get("action", "WATCH")
            reason = act.get("reason", "")

            print(f"\n  → {action} {symbol}: {reason}")

            # Brain: 记录决策
            if self.brain:
                try:
                    commit = self.brain.commit_decision(
                        symbol=symbol,
                        action=action,
                        reason=reason,
                        env_score=env_score,
                    )
                    committed.append(symbol)
                    print(f"    ✅ 已记录")
                except Exception as e:
                    print(f"    ⚠️ Brain 记录失败: {e}")

            # EventLog
            if self.event_log:
                try:
                    self.event_log.emit(
                        "order.confirmed",
                        {"symbol": symbol, "action": action,
                         "reason": reason, "env_score": env_score},
                        source="daily_workflow",
                    )
                except Exception:
                    pass

        result = f"已确认 {len(committed)} 笔操作: {', '.join(committed)}"

        # Brain: 更新状态
        if self.brain:
            self.brain.update_status("今日执行", result)
            self.brain.set_next_step("盘后运行 evening 复盘评估执行效果")

        self._save_status("confirm", result)
        print(f"\n{result}")
        return result

    # ═══════════════════════════════════════════
    # 状态 & 全量执行
    # ═══════════════════════════════════════════

    def status(self) -> str:
        """今日工作流状态"""
        print(f"\n📋 工作流状态 ({self._today})")
        print("-" * 40)

        steps = ["morning", "signal", "evening", "confirm"]
        for step in steps:
            status_file = self.status_dir / f"{self._today}_{step}.json"
            if status_file.exists():
                data = json.loads(status_file.read_text(encoding="utf-8"))
                ts = data.get("timestamp", "?")
                preview = data.get("preview", "")[:60]
                print(f"  ✅ {step:10s} | {ts} | {preview}")
            else:
                print(f"  ⬜ {step:10s} | 未执行")

        # Brain 结构化状态（Structured Compaction）
        if self.brain:
            print(f"\n  Brain:")
            state = self.brain._state
            print(f"    意图: {state.get('intent', '(空)')}")
            status_items = state.get("status", {})
            for k, v in list(status_items.items())[:5]:
                print(f"    {k}: {v}")
            next_steps = state.get("next_steps", [])
            if next_steps:
                print(f"    下一步: {'; '.join(next_steps)}")
            print(f"    记忆: {self.brain.memory_count} 条")

            emotion = self.brain.get_emotion()
            if emotion:
                print(f"    情绪: {emotion.get('level', '?')} ({emotion.get('score', '?')})")

        return "status_displayed"

    def full(self) -> str:
        """按顺序执行全部 4 步"""
        print("\n🔄 执行完整工作流...")
        self.morning()
        print("\n⏳ 等待 3 秒...")
        time.sleep(3)
        self.signal()
        print("\n⏳ 等待 3 秒...")
        time.sleep(3)
        self.evening()
        print("\n⚠️ confirm 需要人工审核，跳过自动执行")
        self.status()
        return "full_workflow_done"

    # ─────────────────────────────────────────
    # 内部工具
    # ─────────────────────────────────────────

    def _run_analysis(self, prompt: str, brain_ctx: str = "") -> str:
        """统一的 LLM/Router 分析入口"""
        if brain_ctx:
            prompt += f"\n{brain_ctx}"

        if self.llm:
            try:
                return self.llm.chat(prompt)
            except Exception as e:
                return f"LLM 失败: {e}"

        if self.router:
            try:
                if hasattr(self.router, 'answer'):
                    return str(self.router.answer(prompt))
                elif hasattr(self.router, 'route_and_execute'):
                    return str(self.router.route_and_execute(prompt))
            except Exception as e:
                return f"Router 失败: {e}"

        return ""

    def _save_status(self, step: str, output: str, data: dict = None):
        def _json_default(obj):
            # 兼容 numpy/pandas 标量
            if hasattr(obj, "item"):
                try:
                    return obj.item()
                except Exception:
                    pass
            return str(obj)

        status = {
            "step": step,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "preview": output[:500] if output else "",
            "data": data,
        }
        path = self.status_dir / f"{self._today}_{step}.json"
        path.write_text(
            json.dumps(status, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )

    def _load_status(self, step: str, raw: bool = False):
        path = self.status_dir / f"{self._today}_{step}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("data") if raw else data.get("preview", "")
        except Exception:
            return None

    def _log_step(self, step: str, state: str, **kwargs):
        if self.event_log:
            try:
                self.event_log.emit("workflow.step", {
                    "step": step, "state": state, **kwargs,
                }, source="daily_workflow")
            except Exception:
                pass

    def _save_dashboard(self, content: str):
        date_tag = datetime.now().strftime("%Y%m%d")
        path = self.dashboard_dir / f"{date_tag}.txt"
        path.write_text(content, encoding="utf-8")


# ═══════════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════════

def main():
    """CLI 入口，自动初始化所有组件"""
    if len(sys.argv) < 2:
        print("用法: python3 daily_workflow.py <command>")
        print("  morning  — 盘前分析")
        print("  signal   — 信号生成")
        print("  evening  — 盘后复盘")
        print("  confirm  — 确认执行")
        print("  full     — 全部执行")
        print("  status   — 今日状态")
        sys.exit(1)

    cmd = sys.argv[1].lower()

    # 初始化（Progressive Disclosure: 按需加载）
    event_log = None
    brain = None
    llm = None
    router = None
    fetcher = None
    market_env = None
    signal_gen = None
    news = None
    notifier = None

    try:
        from event_log import EventLog
        event_log = EventLog()
    except Exception:
        pass

    try:
        from brain import get_brain
        brain = get_brain(event_log=event_log)
    except Exception:
        pass

    try:
        from llm_client import get_llm
        llm = get_llm()
    except Exception:
        pass

    try:
        from router import Router
        router = Router()
    except Exception:
        pass

    try:
        from data_fetcher import DataFetcher
        fetcher = DataFetcher()
    except Exception:
        pass

    try:
        from market_environment import MarketEnvironment
        market_env = MarketEnvironment(fetcher)
    except Exception:
        pass

    try:
        from trade_signals import TradeSignalGenerator
        signal_gen = TradeSignalGenerator(
            data_fetcher=fetcher,
            market_env=market_env,
            event_log=event_log,
            brain=brain,
        )
    except Exception:
        pass

    try:
        from news_service import NewsService
        news = NewsService()
    except Exception:
        news = None

    try:
        from notification import Notifier
        notifier = Notifier()
    except Exception:
        notifier = None

    workflow = DailyWorkflow(
        data_fetcher=fetcher,
        router=router,
        brain=brain,
        event_log=event_log,
        llm=llm,
        signal_generator=signal_gen,
        market_env=market_env,
        news_service=news,
        notifier=notifier,
    )

    commands = {
        "morning": workflow.morning,
        "signal": workflow.signal,
        "evening": workflow.evening,
        "confirm": workflow.confirm,
        "full": workflow.full,
        "status": workflow.status,
    }

    func = commands.get(cmd)
    if func:
        func()
    else:
        print(f"未知命令: {cmd}")
        print(f"可用: {', '.join(commands.keys())}")


if __name__ == "__main__":
    main()
