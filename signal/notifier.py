#!/usr/bin/env python3
"""
交易信号推送模块

支持渠道:
  1. 邮件 (SMTP) — 支持QQ邮箱/163邮箱/Gmail
  2. 钉钉机器人 (Webhook)
  3. 企业微信机器人 (Webhook)
  4. Server酱 (微信推送)
  5. Bark (iOS推送)
  6. 本地文件输出 (保底方案)

配置方式: 环境变量 或 notify_config.json
"""

from __future__ import annotations

import json
import os
import smtplib
import ssl
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError


# ══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class NotifyConfig:
    """推送配置"""
    # 邮件
    email_enabled: bool = False
    smtp_host: str = ""           # smtp.qq.com / smtp.163.com / smtp.gmail.com
    smtp_port: int = 465          # 465=SSL, 587=TLS
    smtp_user: str = ""           # 发件人邮箱
    smtp_password: str = ""       # 授权码 (非登录密码)
    email_to: list[str] = field(default_factory=list)  # 收件人列表

    # 钉钉
    dingtalk_enabled: bool = False
    dingtalk_webhook: str = ""    # https://oapi.dingtalk.com/robot/send?access_token=xxx
    dingtalk_secret: str = ""     # 加签密钥 (可选)

    # 企业微信
    wecom_enabled: bool = False
    wecom_webhook: str = ""       # https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx

    # Server酱 (微信推送)
    serverchan_enabled: bool = False
    serverchan_key: str = ""      # https://sctapi.ftqq.com/{key}.send

    # Bark (iOS)
    bark_enabled: bool = False
    bark_url: str = ""            # https://api.day.app/{key}/

    @classmethod
    def from_env(cls) -> "NotifyConfig":
        """从环境变量加载配置"""
        cfg = cls()

        # 邮件
        if os.getenv("SMTP_HOST"):
            cfg.email_enabled = True
            cfg.smtp_host = os.getenv("SMTP_HOST", "")
            cfg.smtp_port = int(os.getenv("SMTP_PORT", "465"))
            cfg.smtp_user = os.getenv("SMTP_USER", "")
            cfg.smtp_password = os.getenv("SMTP_PASSWORD", "")
            cfg.email_to = os.getenv("EMAIL_TO", "").split(",")

        # 钉钉
        if os.getenv("DINGTALK_WEBHOOK"):
            cfg.dingtalk_enabled = True
            cfg.dingtalk_webhook = os.getenv("DINGTALK_WEBHOOK", "")
            cfg.dingtalk_secret = os.getenv("DINGTALK_SECRET", "")

        # 企业微信
        if os.getenv("WECOM_WEBHOOK"):
            cfg.wecom_enabled = True
            cfg.wecom_webhook = os.getenv("WECOM_WEBHOOK", "")

        # Server酱
        if os.getenv("SERVERCHAN_KEY"):
            cfg.serverchan_enabled = True
            cfg.serverchan_key = os.getenv("SERVERCHAN_KEY", "")

        # Bark
        if os.getenv("BARK_URL"):
            cfg.bark_enabled = True
            cfg.bark_url = os.getenv("BARK_URL", "")

        return cfg

    @classmethod
    def from_json(cls, path: str | Path) -> "NotifyConfig":
        """从JSON文件加载配置"""
        data = json.loads(Path(path).read_text("utf-8"))
        cfg = cls()
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    def save(self, path: str | Path):
        """保存配置到JSON"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "email_enabled": self.email_enabled,
            "smtp_host": self.smtp_host,
            "smtp_port": self.smtp_port,
            "smtp_user": self.smtp_user,
            "smtp_password": self.smtp_password,
            "email_to": self.email_to,
            "dingtalk_enabled": self.dingtalk_enabled,
            "dingtalk_webhook": self.dingtalk_webhook,
            "dingtalk_secret": self.dingtalk_secret,
            "wecom_enabled": self.wecom_enabled,
            "wecom_webhook": self.wecom_webhook,
            "serverchan_enabled": self.serverchan_enabled,
            "serverchan_key": self.serverchan_key,
            "bark_enabled": self.bark_enabled,
            "bark_url": self.bark_url,
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


# ══════════════════════════════════════════════════════════════════════════════
#  Notifier
# ══════════════════════════════════════════════════════════════════════════════

class Notifier:
    """多渠道消息推送器"""

    def __init__(self, config: NotifyConfig = None):
        self.config = config or NotifyConfig.from_env()
        self.results: list[dict] = []

    def send(self, title: str, content: str, html_content: str = None) -> list[dict]:
        """
        推送消息到所有已启用的渠道。

        Args:
            title: 消息标题
            content: 纯文本内容
            html_content: HTML内容 (仅邮件使用)

        Returns:
            每个渠道的发送结果
        """
        self.results = []

        if self.config.email_enabled:
            self._send_email(title, content, html_content)

        if self.config.dingtalk_enabled:
            self._send_dingtalk(title, content)

        if self.config.wecom_enabled:
            self._send_wecom(title, content)

        if self.config.serverchan_enabled:
            self._send_serverchan(title, content)

        if self.config.bark_enabled:
            self._send_bark(title, content)

        # 如果没有任何渠道启用, 输出到控制台
        if not any([
            self.config.email_enabled,
            self.config.dingtalk_enabled,
            self.config.wecom_enabled,
            self.config.serverchan_enabled,
            self.config.bark_enabled,
        ]):
            print("[推送] 未配置任何推送渠道, 信号仅输出到控制台和文件")
            self.results.append({"channel": "console", "success": True})

        return self.results

    # ── 邮件 ──

    def _send_email(self, title: str, content: str, html_content: str = None):
        """通过SMTP发送邮件"""
        cfg = self.config
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = title
            msg["From"] = cfg.smtp_user
            msg["To"] = ", ".join(cfg.email_to)

            # 纯文本
            msg.attach(MIMEText(content, "plain", "utf-8"))

            # HTML (如果有)
            if html_content:
                msg.attach(MIMEText(html_content, "html", "utf-8"))

            # 发送
            context = ssl.create_default_context()
            if cfg.smtp_port == 465:
                with smtplib.SMTP_SSL(cfg.smtp_host, cfg.smtp_port, context=context) as server:
                    server.login(cfg.smtp_user, cfg.smtp_password)
                    server.sendmail(cfg.smtp_user, cfg.email_to, msg.as_string())
            else:
                with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port) as server:
                    server.starttls(context=context)
                    server.login(cfg.smtp_user, cfg.smtp_password)
                    server.sendmail(cfg.smtp_user, cfg.email_to, msg.as_string())

            print(f"[推送] 邮件发送成功 -> {cfg.email_to}")
            self.results.append({"channel": "email", "success": True, "to": cfg.email_to})

        except Exception as e:
            print(f"[推送] 邮件发送失败: {e}")
            traceback.print_exc()
            self.results.append({"channel": "email", "success": False, "error": str(e)})

    # ── 钉钉机器人 ──

    def _send_dingtalk(self, title: str, content: str):
        """通过钉钉Webhook推送"""
        cfg = self.config
        try:
            webhook_url = cfg.dingtalk_webhook

            # 加签 (如果配置了secret)
            if cfg.dingtalk_secret:
                import hashlib
                import hmac
                import base64
                import time
                from urllib.parse import quote_plus

                timestamp = str(round(time.time() * 1000))
                string_to_sign = f"{timestamp}\n{cfg.dingtalk_secret}"
                hmac_code = hmac.new(
                    cfg.dingtalk_secret.encode("utf-8"),
                    string_to_sign.encode("utf-8"),
                    digestmod=hashlib.sha256,
                ).digest()
                sign = quote_plus(base64.b64encode(hmac_code))
                webhook_url += f"&timestamp={timestamp}&sign={sign}"

            # 钉钉 markdown 消息
            payload = {
                "msgtype": "markdown",
                "markdown": {
                    "title": title,
                    "text": f"### {title}\n\n```\n{content}\n```",
                },
            }

            data = json.dumps(payload).encode("utf-8")
            req = Request(webhook_url, data=data, headers={"Content-Type": "application/json"})
            resp = urlopen(req, timeout=10)
            result = json.loads(resp.read())

            if result.get("errcode") == 0:
                print("[推送] 钉钉发送成功")
                self.results.append({"channel": "dingtalk", "success": True})
            else:
                print(f"[推送] 钉钉发送失败: {result}")
                self.results.append({"channel": "dingtalk", "success": False, "error": str(result)})

        except Exception as e:
            print(f"[推送] 钉钉发送失败: {e}")
            self.results.append({"channel": "dingtalk", "success": False, "error": str(e)})

    # ── 企业微信 ──

    def _send_wecom(self, title: str, content: str):
        """通过企业微信Webhook推送"""
        cfg = self.config
        try:
            # 企业微信 markdown 限制 4096 字节
            text = f"### {title}\n\n{content}"
            if len(text.encode("utf-8")) > 4000:
                text = text[:2000] + "\n...(内容过长已截断)"

            payload = {
                "msgtype": "markdown",
                "markdown": {"content": text},
            }

            data = json.dumps(payload).encode("utf-8")
            req = Request(cfg.wecom_webhook, data=data, headers={"Content-Type": "application/json"})
            resp = urlopen(req, timeout=10)
            result = json.loads(resp.read())

            if result.get("errcode") == 0:
                print("[推送] 企业微信发送成功")
                self.results.append({"channel": "wecom", "success": True})
            else:
                print(f"[推送] 企业微信发送失败: {result}")
                self.results.append({"channel": "wecom", "success": False, "error": str(result)})

        except Exception as e:
            print(f"[推送] 企业微信发送失败: {e}")
            self.results.append({"channel": "wecom", "success": False, "error": str(e)})

    # ── Server酱 ──

    def _send_serverchan(self, title: str, content: str):
        """通过Server酱推送到微信"""
        cfg = self.config
        try:
            url = f"https://sctapi.ftqq.com/{cfg.serverchan_key}.send"
            payload = f"title={title}&desp={content}".encode("utf-8")
            req = Request(url, data=payload, headers={"Content-Type": "application/x-www-form-urlencoded"})
            resp = urlopen(req, timeout=10)
            result = json.loads(resp.read())

            if result.get("code") == 0:
                print("[推送] Server酱发送成功")
                self.results.append({"channel": "serverchan", "success": True})
            else:
                print(f"[推送] Server酱发送失败: {result}")
                self.results.append({"channel": "serverchan", "success": False, "error": str(result)})

        except Exception as e:
            print(f"[推送] Server酱发送失败: {e}")
            self.results.append({"channel": "serverchan", "success": False, "error": str(e)})

    # ── Bark ──

    def _send_bark(self, title: str, content: str):
        """通过Bark推送到iOS"""
        cfg = self.config
        try:
            from urllib.parse import quote
            # Bark URL格式: https://api.day.app/{key}/{title}/{content}
            # 截取前200字做推送
            short = content[:200].replace("\n", " ")
            url = f"{cfg.bark_url.rstrip('/')}/{quote(title)}/{quote(short)}?group=量化交易"
            req = Request(url, method="GET")
            resp = urlopen(req, timeout=10)
            result = json.loads(resp.read())

            if result.get("code") == 200:
                print("[推送] Bark发送成功")
                self.results.append({"channel": "bark", "success": True})
            else:
                print(f"[推送] Bark发送失败: {result}")
                self.results.append({"channel": "bark", "success": False, "error": str(result)})

        except Exception as e:
            print(f"[推送] Bark发送失败: {e}")
            self.results.append({"channel": "bark", "success": False, "error": str(e)})


# ══════════════════════════════════════════════════════════════════════════════
#  HTML格式化 (用于邮件)
# ══════════════════════════════════════════════════════════════════════════════

def format_signal_html(result) -> str:
    """将SignalResult格式化为HTML邮件内容"""
    from signal.signal_generator import SignalResult

    html = []
    html.append("""
    <html><head><style>
    body { font-family: 'Microsoft YaHei', Arial, sans-serif; padding: 20px; background: #f5f5f5; }
    .container { max-width: 700px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
    h1 { color: #333; font-size: 20px; border-bottom: 2px solid #e74c3c; padding-bottom: 8px; }
    h2 { color: #555; font-size: 16px; margin-top: 20px; }
    .regime { padding: 8px 16px; border-radius: 4px; display: inline-block; font-weight: bold; margin: 8px 0; }
    .regime-HALT { background: #e74c3c; color: white; }
    .regime-DEFENSIVE { background: #f39c12; color: white; }
    .regime-RUN { background: #27ae60; color: white; }
    .regime-STRONG_RUN { background: #2ecc71; color: white; }
    table { border-collapse: collapse; width: 100%; margin: 10px 0; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 13px; }
    th { background: #f8f8f8; }
    .buy { color: #e74c3c; font-weight: bold; }
    .sell { color: #27ae60; font-weight: bold; }
    .stop { color: #e74c3c; font-weight: bold; background: #fff3cd; }
    .hold { color: #666; }
    .positive { color: #e74c3c; }
    .negative { color: #27ae60; }
    .footer { color: #999; font-size: 12px; margin-top: 20px; border-top: 1px solid #eee; padding-top: 10px; }
    </style></head><body><div class="container">
    """)

    html.append(f"<h1>动量轮动策略 · 每日信号 {result.trade_date}</h1>")
    html.append(f'<div class="regime regime-{result.market_regime}">市场状态: {result.market_regime}</div>')
    html.append(f"<p>账户总值: <b>¥{result.total_value:,.2f}</b> | 现金: ¥{result.cash:,.2f}</p>")

    # 消息层: 宏观风险 + 舆情
    if getattr(result, "macro_risk", None) and result.macro_risk.get("risk_level", "NORMAL") != "NORMAL":
        risk = result.macro_risk
        risk_color = {"CRITICAL": "#e74c3c", "ELEVATED": "#f39c12"}.get(risk["risk_level"], "#999")
        html.append(f'<div style="background:{risk_color}15;border-left:4px solid {risk_color};padding:10px;margin:10px 0;border-radius:4px;">')
        html.append(f'<b style="color:{risk_color};">⚠️ 宏观风险: {risk["risk_level"]}</b>')
        if risk.get("nearby_events"):
            html.append("<ul style='margin:5px 0;padding-left:20px;'>")
            for evt in risk["nearby_events"][:5]:
                icon = {"high": "🔴", "medium": "🟡"}.get(evt.get("impact", ""), "⚪")
                html.append(f'<li>{icon} {evt.get("distance","")}: {evt.get("name","")}</li>')
            html.append("</ul>")
        if risk.get("quiet_period"):
            html.append(f'<p style="color:{risk_color};font-weight:bold;">📛 静默期: 建议不做主动调仓, 仓位折扣至 {risk.get("position_discount",1):.0%}</p>')
        html.append("</div>")

    if getattr(result, "sentiment", None) and result.sentiment.get("level", "NEUTRAL") != "NEUTRAL":
        sent = result.sentiment
        sent_color = "#e74c3c" if "FEAR" in sent["level"] else "#27ae60"
        html.append(f'<div style="background:{sent_color}10;border-left:4px solid {sent_color};padding:10px;margin:10px 0;border-radius:4px;">')
        html.append(f'<b>📰 舆情: {sent["level"]} (得分: {sent.get("sentiment_score",0):+.2f})</b>')
        if sent.get("top_fear_words"):
            html.append(f'<br>恐慌热词: {", ".join(sent["top_fear_words"][:3])}')
        if sent.get("top_greed_words"):
            html.append(f'<br>乐观热词: {", ".join(sent["top_greed_words"][:3])}')
        html.append("</div>")

    # 交易信号表
    action_signals = [s for s in result.signals if s.action != "HOLD"]
    if action_signals:
        html.append("<h2>交易信号</h2>")
        html.append("<table><tr><th>操作</th><th>代码</th><th>名称</th><th>股数</th><th>价格</th><th>金额</th><th>说明</th></tr>")
        for s in action_signals:
            cls = {"BUY": "buy", "SELL": "sell", "STOP_LOSS": "stop"}.get(s.action, "hold")
            action_cn = {"BUY": "买入", "SELL": "卖出", "STOP_LOSS": "⚠止损"}.get(s.action, s.action)
            html.append(f'<tr class="{cls}"><td>{action_cn}</td><td>{s.ts_code}</td><td>{s.name}</td>'
                       f'<td>{s.target_shares}</td><td>¥{s.current_price:.2f}</td>'
                       f'<td>¥{s.target_amount:,.0f}</td><td>{s.reason}</td></tr>')
        html.append("</table>")
    else:
        html.append("<h2>今日无操作</h2>")

    # 当前持仓
    if result.current_holdings:
        html.append("<h2>当前持仓</h2>")
        html.append("<table><tr><th>代码</th><th>名称</th><th>股数</th><th>成本</th><th>现价</th><th>盈亏</th><th>市值</th></tr>")
        for h in result.current_holdings:
            pnl_cls = "positive" if h["pnl_pct"] >= 0 else "negative"
            html.append(f'<tr><td>{h["ts_code"]}</td><td>{h["name"]}</td><td>{h["shares"]}</td>'
                       f'<td>¥{h["cost_price"]:.2f}</td><td>¥{h["current_price"]:.2f}</td>'
                       f'<td class="{pnl_cls}">{h["pnl_pct"]:+.2f}%</td>'
                       f'<td>¥{h["market_value"]:,.0f}</td></tr>')
        html.append("</table>")

    # 动量TOP10
    html.append("<h2>动量排行 TOP 10</h2>")
    html.append("<table><tr><th>#</th><th>代码</th><th>名称</th><th>得分</th><th>价格</th><th>5日涨幅</th></tr>")
    for s in result.top_momentum[:10]:
        html.append(f'<tr><td>{s["rank"]}</td><td>{s["ts_code"]}</td><td>{s["name"]}</td>'
                   f'<td>{s["score"]:+.4f}</td><td>¥{s["close"]:.2f}</td>'
                   f'<td class="{"positive" if s["pct_chg_5d"]>=0 else "negative"}">{s["pct_chg_5d"]:+.1%}</td></tr>')
    html.append("</table>")

    html.append(f'<div class="footer">⚠️ 以上为策略信号, 请人工确认后在广发易淘金APP执行下单<br>'
               f'生成时间: {result.generated_at}</div>')
    html.append("</div></body></html>")

    return "\n".join(html)
