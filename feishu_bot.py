#!/usr/bin/env python3
"""
feishu_bot.py — 飞书 Bot × OpenClaw 双向集成
端口 8765  |  POST /feishu/events  |  GET /healthz
"""
import os, sys, json, re, time, asyncio, subprocess
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse, Response
import httpx

ASTOCK_DIR = os.path.expanduser("~/project/astock_agent")
sys.path.insert(0, ASTOCK_DIR)

try:
    from config import FEISHU_APP_ID, FEISHU_APP_SECRET, FEISHU_VERIFICATION_TOKEN
except ImportError:
    FEISHU_APP_ID     = os.environ.get("FEISHU_APP_ID", "")
    FEISHU_APP_SECRET = os.environ.get("FEISHU_APP_SECRET", "")
    FEISHU_VERIFICATION_TOKEN = os.environ.get("FEISHU_VERIFICATION_TOKEN", "")

PORT = 8765
FEISHU_API = "https://open.feishu.cn/open-apis"

# 命名常量，替代魔法数字
SUBPROCESS_TIMEOUT_OPENCLAW = 300
SUBPROCESS_TIMEOUT_DAILY    = 180
HTTP_TIMEOUT_TOKEN           = 10
HTTP_TIMEOUT_SEND            = 15
CARD_CONTENT_TRUNCATE_AT     = 2900

OPENCLAW_SCRIPT = os.path.join(ASTOCK_DIR, "openclaw_wrapper.py")
DAILY_SCRIPT    = os.path.join(ASTOCK_DIR, "daily_agent.py")
_executor = ThreadPoolExecutor(max_workers=3)

_token_cache: dict = {"token": "", "expires_at": 0.0}

_seen_events: dict[str, float] = {}
DEDUP_TTL = 60  # 秒

def _is_duplicate(event_id: str) -> bool:
    now = time.time()
    expired = [k for k, v in _seen_events.items() if now - v > DEDUP_TTL]
    for k in expired:
        del _seen_events[k]
    if event_id in _seen_events:
        return True
    _seen_events[event_id] = now
    return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    _executor.shutdown(wait=False, cancel_futures=True)
    print("[Bot] executor 已关闭")


app = FastAPI(title="OpenClaw Feishu Bot", lifespan=lifespan)

def _route_message(text: str) -> tuple[str, str]:
    """返回 (title, reply_text)"""
    if re.match(r"^\d{6}(\.[A-Z]{2})?$", text):
        return _call_openclaw(text)
    if re.match(r"^(/daily|日报)", text, re.IGNORECASE):
        return _call_daily()
    return _call_router(text)


def _clean_env() -> dict:
    """返回清除了畸形代理变量的环境变量副本"""
    env = os.environ.copy()
    for k in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        env.pop(k, None)
    return env


def _call_openclaw(code: str) -> tuple[str, str]:
    try:
        r = subprocess.run(
            [sys.executable, OPENCLAW_SCRIPT, code],
            capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_OPENCLAW,
            cwd=ASTOCK_DIR, env=_clean_env(),
        )
        out = r.stdout.strip() or r.stderr.strip() or "无输出"
        return f"{code} 分析", out
    except subprocess.TimeoutExpired:
        return f"{code} 分析", f"❌ 分析超时（>{SUBPROCESS_TIMEOUT_OPENCLAW}s）"
    except Exception as e:
        return f"{code} 分析", f"❌ 调用失败: {e}"


def _call_daily() -> tuple[str, str]:
    try:
        r = subprocess.run(
            [sys.executable, DAILY_SCRIPT],
            capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_DAILY,
            cwd=ASTOCK_DIR, env=_clean_env(),
        )
        out = r.stdout.strip() or r.stderr.strip() or "日报生成完成"
        return "每日前瞻报告", out
    except subprocess.TimeoutExpired:
        return "每日前瞻报告", f"❌ 生成超时（>{SUBPROCESS_TIMEOUT_DAILY}s）"
    except Exception as e:
        return "每日前瞻报告", f"❌ 调用失败: {e}"


_LALA_INTRO = (
    "Lala。wuhan 的 A 股量化助手，INTJ。\n\n"
    "职责：分析个股/市场、找你方案里的漏洞、推动决策落地。\n"
    "原则：结论先行，数据说话，不确定标置信度，发现你错了直接说。\n\n"
    "发股票代码（如 000001.SH）开始分析，或直接问问题。"
)

_IDENTITY_PATTERNS = re.compile(
    r"^(你是谁|你叫什么|介绍(一下)?自己|你是什么|who are you|what are you)", re.IGNORECASE
)


def _call_router(question: str) -> tuple[str, str]:
    # 身份问题直接返回，不走 LLM（防止模型暴露底层身份）
    if _IDENTITY_PATTERNS.match(question.strip()):
        return "Lala", _LALA_INTRO

    try:
        from openclaw_os import OpenClawOS
        os_instance = OpenClawOS()
        result = os_instance.handle(question, context={"question": question})
        routing = result.get("routing")
        execution = result.get("execution", {})

        zone = routing.zone.value if routing else "unknown"
        role = routing.primary_role.name if routing else "unknown"
        confidence = f"{routing.confidence:.0%}" if routing else "?"

        exec_status = execution.get("status", "unknown")
        if exec_status == "success":
            answer = execution.get("result", "")
            title = "Lala"
        elif exec_status in ("no_skill", "multi_match"):
            from router import Router
            answer = Router().answer(question)
            title = "Lala"
        else:
            reason = execution.get("reason", exec_status)
            answer = f"执行受阻: {reason}"
            title = "Lala"

        answer = f"{answer}\n\n---\n*路由: {zone}/{role}  置信度: {confidence}*"
        return title, answer
    except Exception as e:
        try:
            from router import Router
            answer = Router().answer(question)
            return "Lala", answer
        except Exception as e2:
            return "Lala", f"❌ 路由失败: {e}"


async def _get_token() -> str:
    """获取 tenant_access_token（带 60s 提前刷新）"""
    now = time.time()
    if _token_cache["token"] and now < _token_cache["expires_at"] - 60:
        return _token_cache["token"]
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_TOKEN, trust_env=False) as c:
        r = await c.post(
            f"{FEISHU_API}/auth/v3/tenant_access_token/internal",
            json={"app_id": FEISHU_APP_ID, "app_secret": FEISHU_APP_SECRET},
        )
    data = r.json()
    token = data.get("tenant_access_token", "")
    if not token:
        raise RuntimeError(f"飞书 token 为空，响应: {data}")
    _token_cache["token"]      = token
    _token_cache["expires_at"] = now + data.get("expire", 7200)
    return token


async def send_card(
    chat_id: str, title: str, content: str,
    elapsed: float = 0.0, error: bool = False,
):
    """构造并发送飞书交互卡片"""
    token    = await _get_token()
    template = "red" if error else "blue"
    icon     = "❌" if error else "🤖"

    if len(content) > CARD_CONTENT_TRUNCATE_AT:
        content = content[:CARD_CONTENT_TRUNCATE_AT] + "\n\n…（内容过长已截断）"

    note = f"⏱ {elapsed:.1f}s  |  OpenClaw"

    card = {
        "header": {
            "title":    {"tag": "plain_text", "content": f"{icon} OpenClaw — {title}"},
            "template": template,
        },
        "elements": [
            {"tag": "div", "text": {"tag": "lark_md", "content": content}},
            {"tag": "hr"},
            {"tag": "note", "elements": [{"tag": "plain_text", "content": note}]},
        ],
    }

    payload = {
        "receive_id": chat_id,
        "msg_type":   "interactive",
        "content":    json.dumps(card, ensure_ascii=False),
    }

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SEND, trust_env=False) as c:
        resp = await c.post(
            f"{FEISHU_API}/im/v1/messages?receive_id_type=chat_id",
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
        )
    data = resp.json()
    if data.get("code") != 0:
        print(f"[Bot] 发送卡片失败 code={data.get('code')} msg={data.get('msg')}")
    else:
        print(f"[Bot] 卡片已发送 chat={chat_id} elapsed={elapsed:.1f}s")


async def _handle_message(chat_id: str, text: str):
    t0 = time.time()
    loop = asyncio.get_running_loop()
    title, reply, error_flag = "错误", "内部异常", True
    try:
        title, reply = await loop.run_in_executor(_executor, _route_message, text)
        error_flag = False
    except Exception as e:
        print(f"[Bot] _route_message 异常: {e}")
        reply = f"处理请求时发生错误: {e}"

    elapsed = time.time() - t0
    try:
        await send_card(chat_id, title, reply, elapsed, error=error_flag)
    except Exception as e:
        print(f"[Bot] send_card 最终失败，无法通知用户: {e}")

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "service": "feishu-bot", "port": PORT}

@app.post("/feishu/events")
async def feishu_events(request: Request, background_tasks: BackgroundTasks):
    body = await request.body()
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return JSONResponse({"code": 0, "msg": "invalid json"})

    # ── Challenge 校验（飞书配置事件订阅 URL 时触发一次）──
    if payload.get("type") == "url_verification":
        challenge = payload.get("challenge", "")
        token     = payload.get("token", "")
        if FEISHU_VERIFICATION_TOKEN and token != FEISHU_VERIFICATION_TOKEN:
            return Response(status_code=403, content="invalid token")
        return JSONResponse({"challenge": challenge})

    # ── Token 校验（事件推送，非 challenge）──
    header      = payload.get("header", {})
    event_token = header.get("token", "")
    if FEISHU_VERIFICATION_TOKEN and event_token != FEISHU_VERIFICATION_TOKEN:
        return Response(status_code=403, content="invalid token")

    # ── event_id 去重 ──
    event_id = header.get("event_id", "")
    if event_id and _is_duplicate(event_id):
        return JSONResponse({"code": 0, "msg": "duplicate"})

    # ── 消息事件 ──
    event_type = header.get("event_type", "")
    if event_type == "im.message.receive_v1":
        event    = payload.get("event", {})
        message  = event.get("message", {})
        msg_type = message.get("message_type", "")
        chat_id  = message.get("chat_id", "")

        if msg_type == "text" and chat_id:
            try:
                text = json.loads(message.get("content", "{}")).get("text", "").strip()
            except Exception:
                text = ""
            if text:
                background_tasks.add_task(_handle_message, chat_id, text)
                return JSONResponse({"code": 0, "msg": "ok"})

    return JSONResponse({"code": 0, "msg": "ok"})

# ─────────────────────────────────────────
# 同步推送（供非 async 上下文调用，如后台线程）
# ─────────────────────────────────────────

_FEISHU_DEFAULT_CHAT_ID: str = os.environ.get("FEISHU_DEFAULT_CHAT_ID", "")

try:
    from config import FEISHU_DEFAULT_CHAT_ID as _cfg_chat  # type: ignore
    if _cfg_chat:
        _FEISHU_DEFAULT_CHAT_ID = _cfg_chat
except (ImportError, AttributeError):
    pass


def push_message_sync(
    title: str,
    content: str,
    chat_id: str = "",
    error: bool = False,
) -> bool:
    """
    同步阻塞推送飞书消息（在普通线程中调用，自行管理 event loop）。

    Returns True on success, False on failure（不抛异常）。
    """
    _id = chat_id or _FEISHU_DEFAULT_CHAT_ID
    if not _id:
        print("[Bot] push_message_sync: 未配置 chat_id，跳过推送", file=sys.stderr)
        return False
    try:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                send_card(_id, title, content, elapsed=0.0, error=error)
            )
        finally:
            loop.close()
        return True
    except RuntimeError as e:
        # 通常是在已有运行中的 event loop 内调用（如 Jupyter / FastAPI 上下文）
        print(
            f"[Bot] push_message_sync RuntimeError（可能存在运行中的 event loop，"
            f"请改用 await send_card(...)）: {e}",
            file=sys.stderr,
        )
        return False
    except Exception as e:
        import traceback
        print(f"[Bot] push_message_sync 失败: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return False


if __name__ == "__main__":
    import uvicorn
    print(f"🤖 OpenClaw Feishu Bot  port={PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
