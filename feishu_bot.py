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

app = FastAPI(title="OpenClaw Feishu Bot")

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "service": "feishu-bot", "port": PORT}

if __name__ == "__main__":
    import uvicorn
    print(f"🤖 OpenClaw Feishu Bot  port={PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
