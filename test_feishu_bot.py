#!/usr/bin/env python3
"""
test_feishu_bot.py — feishu_bot 本地回归测试

用法（先确保 feishu_bot 服务已启动）：
  python3 feishu_bot.py &
  sleep 2
  python3 test_feishu_bot.py
"""
import json
import time
import sys
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:8765"


def _get(path: str) -> tuple[int, dict]:
    try:
        req = urllib.request.Request(f"{BASE}{path}")
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, {}
    except Exception as e:
        print(f"  连接失败: {e}")
        return 0, {}


def _post(path: str, body: dict) -> tuple[int, dict]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE}{path}", data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, {}
    except Exception as e:
        print(f"  连接失败: {e}")
        return 0, {}


def test_healthz():
    status, body = _get("/healthz")
    assert status == 200, f"healthz 返回 {status}"
    assert body.get("status") == "ok", f"body={body}"
    print("✅ GET /healthz")


def test_challenge():
    status, body = _post(
        "/feishu/events",
        {"type": "url_verification", "challenge": "abc123", "token": ""},
    )
    assert status == 200, f"challenge 返回 {status}"
    assert body.get("challenge") == "abc123", f"body={body}"
    print("✅ POST /feishu/events - challenge 校验")


def test_invalid_token():
    import os
    token_env = os.environ.get("FEISHU_VERIFICATION_TOKEN", "")
    if not token_env:
        print("⏭  FEISHU_VERIFICATION_TOKEN 未设置，跳过 token 校验测试")
        return
    status, _ = _post(
        "/feishu/events",
        {
            "schema": "2.0",
            "header": {"token": "wrong_token", "event_type": "x", "event_id": "t1"},
            "event": {},
        },
    )
    assert status == 403, f"错误 token 期望 403，实际 {status}"
    print("✅ POST /feishu/events - 错误 token → 403")


def test_invalid_json():
    req = urllib.request.Request(
        f"{BASE}/feishu/events",
        data=b"not-valid-json",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            status = r.status
            body = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        status = e.code
        body = {}
    assert status == 200, f"非法 JSON 期望 200，实际 {status}"
    assert body.get("msg") == "invalid json", f"body={body}"
    print("✅ POST /feishu/events - 非法 JSON → 200")


def test_duplicate():
    payload = {
        "schema": "2.0",
        "header": {
            "token": "",
            "event_type": "im.message.receive_v1",
            "event_id": f"dup-{int(time.time())}",
        },
        "event": {
            "message": {
                "chat_id": "oc_test",
                "message_type": "text",
                "content": json.dumps({"text": "hello"}),
            }
        },
    }
    _, b1 = _post("/feishu/events", payload)
    _, b2 = _post("/feishu/events", payload)
    assert b1.get("msg") != "duplicate", f"首次不应去重，但 b1={b1}"
    assert b2.get("msg") == "duplicate", f"重复事件应返回 duplicate，但 b2={b2}"
    print("✅ POST /feishu/events - event_id 去重")


def test_message_route():
    """发送 mock 消息，验证服务立即返回 200（不校验卡片推送）"""
    for text in ["600519", "/daily", "今天大盘怎样"]:
        eid = f"ev-{int(time.time() * 1000)}-{text[:4]}"
        payload = {
            "schema": "2.0",
            "header": {
                "token": "",
                "event_type": "im.message.receive_v1",
                "event_id": eid,
            },
            "event": {
                "message": {
                    "chat_id": "oc_test",
                    "message_type": "text",
                    "content": json.dumps({"text": text}),
                }
            },
        }
        status, body = _post("/feishu/events", payload)
        assert status == 200, f"消息 '{text}' 返回 {status}"
        assert body.get("msg") == "ok", f"body={body}"
        print(f"✅ 消息路由: '{text}' → 立即 200")
        time.sleep(0.3)


if __name__ == "__main__":
    print(f"🔗 连接 {BASE}")
    print()
    failed = False
    for fn in [test_healthz, test_challenge, test_invalid_token,
               test_invalid_json, test_duplicate, test_message_route]:
        try:
            fn()
        except AssertionError as e:
            print(f"❌ {fn.__name__}: {e}")
            failed = True
        except Exception as e:
            print(f"❌ {fn.__name__}: 未预期异常 {e}")
            failed = True

    print()
    if failed:
        print("❌ 部分测试失败")
        sys.exit(1)
    else:
        print("🎉 所有本地回归测试通过")
