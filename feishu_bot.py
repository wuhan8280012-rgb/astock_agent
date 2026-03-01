#!/usr/bin/env python3
"""
feishu_bot.py — 飞书 Bot × OpenClaw 双向集成
端口 8765  |  POST /feishu/events  |  GET /healthz
"""
import os, sys, json, re, time, asyncio
from concurrent.futures import ThreadPoolExecutor
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

app = FastAPI(title="OpenClaw Feishu Bot")

def _route_message(text: str) -> tuple[str, str]:
    """返回 (title, reply_text)"""
    if re.match(r"^\d{6}(\.[A-Z]{2})?$", text):
        return _call_openclaw(text)
    if re.match(r"^(/daily|日报)", text, re.IGNORECASE):
        return _call_daily()
    return _call_router(text)


def _call_openclaw(code: str) -> tuple[str, str]:
    import subprocess
    try:
        r = subprocess.run(
            [sys.executable, OPENCLAW_SCRIPT, code],
            capture_output=True, text=True, timeout=120, cwd=ASTOCK_DIR,
        )
        out = r.stdout.strip() or r.stderr.strip() or "无输出"
        return f"{code} 分析", out
    except subprocess.TimeoutExpired:
        return f"{code} 分析", "❌ 分析超时（>120s）"
    except Exception as e:
        return f"{code} 分析", f"❌ 调用失败: {e}"


def _call_daily() -> tuple[str, str]:
    import subprocess
    try:
        r = subprocess.run(
            [sys.executable, DAILY_SCRIPT],
            capture_output=True, text=True, timeout=180, cwd=ASTOCK_DIR,
        )
        out = r.stdout.strip() or "日报生成完成"
        return "每日前瞻报告", out[:3000]
    except subprocess.TimeoutExpired:
        return "每日前瞻报告", "❌ 生成超时（>180s）"
    except Exception as e:
        return "每日前瞻报告", f"❌ 调用失败: {e}"


def _call_router(question: str) -> tuple[str, str]:
    try:
        from router import Router
        answer = Router().answer(question)
        return "智能问答", answer
    except Exception as e:
        return "智能问答", f"❌ 路由失败: {e}"


async def _get_token() -> str:
    """获取 tenant_access_token（带 60s 提前刷新）"""
    now = time.time()
    if _token_cache["token"] and now < _token_cache["expires_at"] - 60:
        return _token_cache["token"]
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.post(
            f"{FEISHU_API}/auth/v3/tenant_access_token/internal",
            json={"app_id": FEISHU_APP_ID, "app_secret": FEISHU_APP_SECRET},
        )
    data = r.json()
    _token_cache["token"]      = data.get("tenant_access_token", "")
    _token_cache["expires_at"] = now + data.get("expire", 7200)
    return _token_cache["token"]


async def send_card(
    chat_id: str, title: str, content: str,
    elapsed: float = 0.0, error: bool = False,
):
    """构造并发送飞书交互卡片"""
    token    = await _get_token()
    template = "red" if error else "blue"
    icon     = "❌" if error else "🤖"

    if len(content) > 3000:
        content = content[:2900] + "\n\n…（内容过长已截断）"

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

    async with httpx.AsyncClient(timeout=15) as c:
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
    t0   = time.time()
    loop = asyncio.get_event_loop()
    try:
        title, reply = await loop.run_in_executor(_executor, _route_message, text)
        elapsed = time.time() - t0
        await send_card(chat_id, title, reply, elapsed)
    except Exception as e:
        elapsed = time.time() - t0
        await send_card(chat_id, "错误", str(e), elapsed, error=True)

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "service": "feishu-bot", "port": PORT}

@app.post("/feishu/events")
async def feishu_events(request: Request, background_tasks: BackgroundTasks):
    body    = await request.body()
    payload = json.loads(body)

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

if __name__ == "__main__":
    import uvicorn
    print(f"🤖 OpenClaw Feishu Bot  port={PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
