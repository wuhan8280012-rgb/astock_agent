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

    return JSONResponse({"code": 0, "msg": "ok"})

if __name__ == "__main__":
    import uvicorn
    print(f"🤖 OpenClaw Feishu Bot  port={PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
