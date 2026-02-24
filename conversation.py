"""
conversation.py — 多轮对话管理器
=================================
支持连续追问，自动管理上下文窗口，集成 Brain 记忆。

用法：
  # 交互式 CLI
  python3 conversation.py

  # 程序化调用
  from conversation import Conversation
  conv = Conversation(router=router, brain=brain)
  answer1 = conv.ask("当前板块轮动方向")
  answer2 = conv.ask("那消费电子具体怎么样")  # 自动带上上一轮上下文
  answer3 = conv.ask("帮我选几只标的")         # 继续追问

特性：
  - 自动保持对话历史，支持连续追问
  - Brain 上下文在每轮自动注入（根据当前问题召回相关记忆）
  - Compactor 自动压缩长对话，不会爆 context window
  - 每轮对话记录到 EventLog
  - 支持会话保存/恢复
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Callable

# 可选依赖
try:
    from compactor import ContextCompactor
    HAS_COMPACTOR = True
except ImportError:
    try:
        from cron_and_tools import ContextCompactor
        HAS_COMPACTOR = True
    except ImportError:
        HAS_COMPACTOR = False

try:
    from event_log import EventLog, EventType
    HAS_EVENT_LOG = True
except ImportError:
    HAS_EVENT_LOG = False


# ═══════════════════════════════════════════
# 对话管理器
# ═══════════════════════════════════════════

class Conversation:
    """
    多轮对话管理器

    核心逻辑：
    1. 每轮 ask() 时，把历史对话 + Brain 上下文 + 新问题 组装成 messages
    2. 如果超长，用 Compactor 压缩早期对话
    3. 发给 LLM，获取回答
    4. 将本轮 Q&A 追加到历史
    5. 可选：Brain.learn_from_output() 沉淀洞察

    与 Router 集成：
      方式 A: 传入 router 实例，用 Router 的路由+执行能力
      方式 B: 传入 llm_caller，直接对话（不走路由）
    """

    def __init__(self,
                 router: Any = None,
                 llm_caller: Callable = None,
                 brain: Any = None,
                 event_log: Any = None,
                 system_prompt: str = None,
                 max_tokens: int = 8000,
                 max_history: int = 20,
                 session_dir: str = "data/conversations"):
        """
        Parameters
        ----------
        router : Router 实例（有 route_and_execute 或类似方法）
        llm_caller : LLM 调用函数 (messages: list[dict]) -> str
                     与 router 二选一，都传则优先 router
        brain : AgentBrain 实例
        event_log : EventLog 实例
        system_prompt : 系统提示词（不传则用默认）
        max_tokens : 上下文 token 上限
        max_history : 最大保留的对话轮数（超过则压缩）
        session_dir : 会话持久化目录
        """
        self.router = router
        self.llm_caller = llm_caller
        self.brain = brain
        self.event_log = event_log
        self.max_history = max_history

        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # 对话历史
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        self.messages: list[dict] = []  # [{"role": "user"|"assistant", "content": "..."}]
        self.turn_count = 0

        # 系统提示词
        self.system_prompt = system_prompt or self._default_system_prompt()

        # Compactor
        self.compactor = ContextCompactor(max_tokens=max_tokens) if HAS_COMPACTOR else None

    # ─────────────────────────────────────────
    # 核心：提问
    # ─────────────────────────────────────────

    def ask(self, question: str, use_router: bool = True) -> str:
        """
        向 Agent 提问（自动带上历史上下文）

        Parameters
        ----------
        question : 用户问题
        use_router : True=通过 Router 路由执行, False=直接 LLM 对话

        Returns
        -------
        Agent 回答文本
        """
        self.turn_count += 1
        start_time = time.time()

        # 1. 获取 Brain 上下文（根据当前问题）
        brain_context = ""
        if self.brain:
            try:
                brain_context = self.brain.get_context_for_prompt(question)
            except Exception:
                pass

        # 2. 组装 messages
        full_messages = self._build_messages(question, brain_context)

        # 3. 压缩（如果需要）
        if self.compactor:
            full_messages = self.compactor.compact(full_messages)

        # 4. 获取回答
        answer = ""
        method = "unknown"

        if use_router and self.router:
            answer, method = self._ask_via_router(question, brain_context)
        elif self.llm_caller:
            # full_messages 为 list[dict]（role/content），与 call_for_router 接口一致
            answer = self.llm_caller(full_messages)
            method = "llm_direct"
        else:
            answer = "未配置 Router 或 LLM，无法回答。"
            method = "none"

        # 5. 记录到历史
        self.messages.append({"role": "user", "content": question})
        self.messages.append({"role": "assistant", "content": answer})

        # 6. 裁剪历史（保留最近 N 轮）
        if len(self.messages) > self.max_history * 2:
            self.messages = self.messages[-(self.max_history * 2):]

        # 7. Brain 学习（可选）
        if self.brain and answer and len(answer) > 50:
            try:
                self.brain.learn_from_output("conversation", question, answer)
            except Exception:
                pass

        # 8. EventLog 记录
        duration = time.time() - start_time
        self._log_turn(question, answer, method, duration)

        return answer

    def _ask_via_router(self, question: str, brain_context: str) -> tuple[str, str]:
        """通过 Router 路由并执行"""
        try:
            # 尝试不同的 Router 接口
            if hasattr(self.router, 'route_and_execute'):
                result = self.router.route_and_execute(question)
                return str(result), "router"
            elif hasattr(self.router, 'ask'):
                result = self.router.ask(question)
                return str(result), "router"
            elif hasattr(self.router, 'route'):
                # 先路由再执行
                route_result = self.router.route(question)
                if hasattr(self.router, 'execute'):
                    result = self.router.execute(route_result)
                    return str(result), "router"
                return str(route_result), "router"
            else:
                # Router 没有可识别的接口，回退到 LLM
                if self.llm_caller:
                    msgs = self._build_messages(question, brain_context)
                    return self.llm_caller(msgs), "llm_fallback"
                return "Router 接口不兼容", "error"
        except Exception as e:
            # Router 执行失败，回退到 LLM
            if self.llm_caller:
                try:
                    msgs = self._build_messages(question, brain_context)
                    return self.llm_caller(msgs), "llm_fallback"
                except Exception:
                    pass
            return f"执行失败: {e}", "error"

    # ─────────────────────────────────────────
    # 消息组装
    # ─────────────────────────────────────────

    def _build_messages(self, question: str, brain_context: str = "") -> list[dict]:
        """组装完整的 messages 列表"""
        msgs = []

        # System prompt
        system = self.system_prompt
        if brain_context:
            system += f"\n\n{brain_context}"
        msgs.append({"role": "system", "content": system})

        # 历史对话
        for m in self.messages:
            msgs.append(m.copy())

        # 当前问题
        msgs.append({"role": "user", "content": question})

        return msgs

    # ─────────────────────────────────────────
    # 会话管理
    # ─────────────────────────────────────────

    def reset(self):
        """清空对话历史，开始新会话"""
        self.messages.clear()
        self.turn_count = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]

    def save(self) -> str:
        """保存会话到文件，返回文件路径"""
        path = self.session_dir / f"{self.session_id}.json"
        data = {
            "session_id": self.session_id,
            "created_at": datetime.now().isoformat(),
            "turn_count": self.turn_count,
            "messages": self.messages,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

    def load(self, session_id: str) -> bool:
        """从文件恢复会话"""
        path = self.session_dir / f"{session_id}.json"
        if not path.exists():
            return False
        data = json.loads(path.read_text(encoding="utf-8"))
        self.session_id = data["session_id"]
        self.turn_count = data.get("turn_count", 0)
        self.messages = data.get("messages", [])
        return True

    def list_sessions(self) -> list[dict]:
        """列出所有保存的会话"""
        sessions = []
        for f in sorted(self.session_dir.glob("*.json"), reverse=True):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                sessions.append({
                    "session_id": data["session_id"],
                    "created_at": data.get("created_at", "?"),
                    "turns": data.get("turn_count", 0),
                    "preview": data["messages"][0]["content"][:50] if data.get("messages") else "",
                })
            except Exception:
                continue
        return sessions[:20]

    @property
    def history_summary(self) -> str:
        """对话历史摘要"""
        if not self.messages:
            return "(空对话)"
        lines = []
        for i in range(0, len(self.messages), 2):
            q = self.messages[i]["content"][:40] if i < len(self.messages) else "?"
            a = self.messages[i+1]["content"][:40] if i+1 < len(self.messages) else "?"
            lines.append(f"  Q{i//2+1}: {q}...")
        return "\n".join(lines)

    # ─────────────────────────────────────────
    # 内部
    # ─────────────────────────────────────────

    @staticmethod
    def _default_system_prompt() -> str:
        return (
            "你是一个专业的A股量化投资AI Agent。\n"
            "你擅长市场情绪分析、板块轮动研判、CANSLIM选股、技术面分析和风险管控。\n"
            "回答要简洁、有数据支撑、给出明确的操作建议。\n"
            "如果用户在追问之前的话题，请结合之前的对话上下文回答。"
        )

    def _log_turn(self, question: str, answer: str, method: str, duration: float):
        if self.event_log and HAS_EVENT_LOG:
            try:
                self.event_log.emit("conversation.turn", {
                    "session_id": self.session_id,
                    "turn": self.turn_count,
                    "question_preview": question[:100],
                    "answer_length": len(answer),
                    "method": method,
                    "duration_sec": round(duration, 2),
                }, source="conversation")
            except Exception:
                pass


# ═══════════════════════════════════════════
# 交互式 CLI
# ═══════════════════════════════════════════

def interactive_cli():
    """交互式多轮对话 CLI"""
    print("=" * 50)
    print("  A股量化投资 AI Agent — 多轮对话")
    print("=" * 50)
    print("  输入问题开始对话，支持连续追问。")
    print("  命令: /new 新会话 | /save 保存 | /history 查看历史")
    print("        /load <id> 恢复 | /sessions 列出会话")
    print("        /stats 统计 | /quit 退出")
    print("=" * 50)

    # 初始化组件
    brain = None
    event_log = None
    llm_caller = None
    router = None

    try:
        from event_log import EventLog
        event_log = EventLog()
    except Exception:
        pass

    try:
        from brain import get_brain
        brain = get_brain(event_log=event_log)
        mem = getattr(brain, "memory_count", None)
        print(f"  Brain: {mem} 条记忆" if mem is not None else "  Brain: 已加载")
    except Exception:
        print("  Brain: 未加载")

    # 尝试加载 Router
    try:
        from router import Router
        _router_inst = Router()
        # 包装成 ask 函数
        if hasattr(_router_inst, 'route_and_execute'):
            router = _router_inst
            print("  Router: 已加载 (路由+执行模式)")
        elif hasattr(_router_inst, 'ask'):
            router = _router_inst
            print("  Router: 已加载 (ask 模式)")
    except Exception:
        print("  Router: 未加载")

    # 尝试加载 LLM（作为 Router 不可用时的回退）
    try:
        from llm_client import get_llm
        _llm = get_llm()

        def llm_caller(messages):
            return _llm.call_for_router(messages)

        print("  LLM: DeepSeek 已连接")
    except Exception:
        print("  LLM: 未加载")

    if not router and not llm_caller:
        print("\n⚠️ 未加载 Router 和 LLM，无法对话。")
        print("   请确保 llm_client.py 和 config.py 配置正确。")
        return

    # 创建对话
    conv = Conversation(
        router=router,
        llm_caller=llm_caller,
        brain=brain,
        event_log=event_log,
    )

    print(f"\n会话 ID: {conv.session_id}\n")

    while True:
        try:
            question = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not question:
            continue

        # 命令处理
        if question.startswith("/"):
            cmd = question.lower().split()
            if cmd[0] == "/quit" or cmd[0] == "/exit" or cmd[0] == "/q":
                print("再见!")
                break
            elif cmd[0] == "/new":
                conv.reset()
                print(f"🆕 新会话: {conv.session_id}\n")
                continue
            elif cmd[0] == "/save":
                path = conv.save()
                print(f"💾 已保存: {path}\n")
                continue
            elif cmd[0] == "/history":
                print(f"📜 对话历史 ({conv.turn_count} 轮):")
                print(conv.history_summary)
                print()
                continue
            elif cmd[0] == "/sessions":
                sessions = conv.list_sessions()
                if sessions:
                    for s in sessions[:10]:
                        print(f"  {s['session_id']} ({s['turns']}轮) {s['preview']}")
                else:
                    print("  无保存的会话")
                print()
                continue
            elif cmd[0] == "/load" and len(cmd) > 1:
                if conv.load(cmd[1]):
                    print(f"📂 已恢复会话: {conv.session_id} ({conv.turn_count} 轮)\n")
                else:
                    print(f"❌ 未找到会话: {cmd[1]}\n")
                continue
            elif cmd[0] == "/stats":
                try:
                    from llm_client import get_llm
                    get_llm().print_stats()
                except Exception:
                    pass
                print(f"对话轮数: {conv.turn_count}")
                print(f"历史消息: {len(conv.messages)} 条\n")
                continue
            else:
                print(f"未知命令: {cmd[0]}\n")
                continue

        # 正常对话
        print("思考中...", end="", flush=True)
        start = time.time()
        answer = conv.ask(question)
        duration = time.time() - start
        print(f"\r            \r", end="")  # 清除 "思考中..."

        print(f"Agent> {answer}")
        print(f"  [{duration:.1f}秒 | 第{conv.turn_count}轮]\n")


# ═══════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════

if __name__ == "__main__":
    interactive_cli()
