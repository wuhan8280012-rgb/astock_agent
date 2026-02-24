"""
notification.py — 多渠道消息推送
================================
支持企业微信、飞书、Telegram、自定义 Webhook。
配了哪个推哪个，可同时推多个。

配置（环境变量）:
  WECHAT_WEBHOOK_URL=https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx
  FEISHU_WEBHOOK_URL=https://open.feishu.cn/open-apis/bot/v2/hook/xxx
  TELEGRAM_BOT_TOKEN=xxx
  TELEGRAM_CHAT_ID=xxx
"""

import os
from typing import Dict, List

import requests


class Notifier:
    def __init__(self):
        self.wechat_url = os.getenv("WECHAT_WEBHOOK_URL", "").strip()
        self.feishu_url = os.getenv("FEISHU_WEBHOOK_URL", "").strip()
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

        states = {
            "企业微信": bool(self.wechat_url),
            "飞书": bool(self.feishu_url),
            "Telegram": bool(self.telegram_token and self.telegram_chat_id),
        }
        msg = " ".join([f"{k}{'✅' if v else '❌'}" for k, v in states.items()])
        print(f"📱 推送渠道: {msg}")

    def available_channels(self) -> List[str]:
        channels = []
        if self.wechat_url:
            channels.append("wechat")
        if self.feishu_url:
            channels.append("feishu")
        if self.telegram_token and self.telegram_chat_id:
            channels.append("telegram")
        return channels

    def push(self, content: str, title: str = None) -> Dict[str, bool]:
        body = content
        if title:
            body = f"{title}\n\n{content}"
        result = {"wechat": False, "feishu": False, "telegram": False}
        if self.wechat_url:
            result["wechat"] = self._push_wechat(body)
        if self.feishu_url:
            result["feishu"] = self._push_feishu(body)
        if self.telegram_token and self.telegram_chat_id:
            result["telegram"] = self._push_telegram(body)
        return result

    def _truncate(self, text: str, limit: int = 4096) -> str:
        if len(text) <= limit:
            return text
        suffix = "\n\n...完整报告见本地文件"
        return text[: max(0, limit - len(suffix))] + suffix

    def _push_wechat(self, content: str) -> bool:
        try:
            resp = requests.post(
                self.wechat_url,
                json={
                    "msgtype": "markdown",
                    "markdown": {"content": self._truncate(content, 4096)},
                },
                timeout=10,
            )
            return resp.ok
        except Exception:
            return False

    def _push_feishu(self, content: str) -> bool:
        try:
            resp = requests.post(
                self.feishu_url,
                json={
                    "msg_type": "text",
                    "content": {"text": self._truncate(content, 4096)},
                },
                timeout=10,
            )
            return resp.ok
        except Exception:
            return False

    def _push_telegram(self, content: str) -> bool:
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            resp = requests.post(
                url,
                json={
                    "chat_id": self.telegram_chat_id,
                    "text": self._truncate(content, 4096),
                    "parse_mode": "Markdown",
                },
                timeout=10,
            )
            return resp.ok
        except Exception:
            return False
