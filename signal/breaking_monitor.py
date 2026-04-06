#!/usr/bin/env python3
"""
突发事件监听器 — 消息层第三层 (黑天鹅防御)

功能:
  1. 多源快讯抓取: tushare新闻 / 新浪财经 / cls.cn(财联社) RSS
  2. 地缘政治/金融危机关键词实时匹配
  3. 风险分级: BLACKSWAN(立即清仓) / CRISIS(暂停交易) / WARNING(收紧仓位)
  4. 触发后立即通过已配置渠道推送告警
  5. 可作为独立守护进程运行, 也可被信号生成器调用

解决的问题:
  - macro_calendar.py 只覆盖已知日程, 无法应对美伊开战等突发事件
  - sentiment_scanner.py 是批量扫描, 不是实时监听
  - 本模块填补 "盘中突发黑天鹅" 这个防御缺口

设计原则:
  - 零外部依赖 (仅标准库 + 可选tushare)
  - 可降级运行: 任何数据源不可用不影响其他源
  - 告警去重: 同一事件30分钟内不重复推送
  - 守护模式: 支持 --daemon 每N分钟轮询
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError


# ══════════════════════════════════════════════════════════════════════════════
#  黑天鹅关键词库 (分级)
# ══════════════════════════════════════════════════════════════════════════════

# BLACKSWAN级 — 立即全仓位撤出信号
BLACKSWAN_PATTERNS: list[tuple[str, str]] = [
    # 地缘战争 (间距放宽到20字符, 兼容各种句式)
    (r"(美国|美军|以色列).{0,20}(开战|宣战|军事打击|空袭|轰炸).{0,20}(伊朗|朝鲜|台湾|台海|俄罗斯)",
     "地缘军事冲突"),
    (r"(伊朗|朝鲜|俄罗斯).{0,20}(开战|宣战|军事打击|空袭|轰炸|反击).{0,20}(美国|以色列|北约)",
     "地缘军事冲突"),
    (r"(美军|美国|以色列).{0,20}(伊朗|朝鲜|台湾|叙利亚).{0,20}(空袭|轰炸|打击|开战|入侵)",
     "地缘军事冲突"),
    (r"(中东|台海|朝鲜半岛).{0,10}(战争|全面冲突|军事对抗)", "地缘军事冲突"),
    (r"(台海|台湾海峡).{0,15}(开火|封锁|军事行动|战争)", "台海危机"),
    (r"(核武器|核弹|核攻击|核战)", "核威胁"),
    (r"第三次世界大战", "全球战争风险"),

    # 金融系统崩溃
    (r"(雷曼|系统性).{0,5}(崩溃|危机|破产)", "系统性金融危机"),
    (r"(美国|全球).{0,5}(债务违约|国债违约)", "主权债务违约"),
    (r"(美联储|央行).{0,5}紧急(加息|降息|会议)", "央行紧急行动"),
    (r"(A股|股市).{0,5}(熔断|全面停牌|暂停交易)", "市场熔断"),

    # 重大灾难
    (r"(全球|大规模).{0,5}(疫情|瘟疫).{0,5}(爆发|蔓延|失控)", "全球疫情"),
]

# CRISIS级 — 暂停主动交易, 仅保留止损
CRISIS_PATTERNS: list[tuple[str, str]] = [
    # 地缘升级
    (r"(美国|欧盟|中国).{0,10}(制裁|禁运|脱钩|断交)", "国际制裁升级"),
    (r"(中美|中欧).{0,5}(贸易战|关税战).{0,5}(升级|全面)", "贸易战升级"),
    (r"(伊朗|朝鲜|俄乌).{0,10}(导弹|袭击|入侵)", "地缘冲突"),
    (r"(石油|原油).{0,5}(暴涨|飙升).{0,5}(超过|突破)", "油价暴涨冲击"),

    # 金融市场异动
    (r"(美股|道琼斯|纳斯达克|标普).{0,5}(暴跌|熔断|崩盘)", "美股暴跌"),
    (r"(日经|欧洲股市|亚太).{0,5}(暴跌|崩盘)", "全球股市暴跌"),
    (r"(人民币|汇率).{0,5}(暴跌|大幅贬值|跌破)", "汇率风险"),
    (r"(恒大|碧桂园|万科|大型房企).{0,5}(破产|清盘|违约)", "房企暴雷"),
    (r"(银行|券商|保险).{0,5}(暴雷|挤兑|破产)", "金融机构危机"),

    # 政策突变
    (r"(印花税|证券交易税).{0,5}(上调|大幅提高)", "交易税突变"),
    (r"(IPO|融资).{0,5}(暂停|全面叫停)", "资本市场政策突变"),
    (r"(外资|QFII|北向).{0,5}(全面撤出|暂停|禁止)", "外资政策突变"),
]

# WARNING级 — 收紧仓位, 提高警惕
WARNING_PATTERNS: list[tuple[str, str]] = [
    (r"(美国|欧洲|全球).{0,10}(通胀|CPI).{0,5}(超预期|飙升|创新高)", "通胀超预期"),
    (r"(失业率|非农).{0,5}(飙升|暴涨|远超预期)", "就业数据恶化"),
    (r"(美债|国债)收益率.{0,5}(飙升|突破|创新高)", "美债收益率冲击"),
    (r"(大宗商品|黄金|铜).{0,5}(暴跌|暴涨|崩盘)", "大宗商品异动"),
    (r"(加密货币|比特币).{0,5}(暴跌|崩盘)", "加密市场崩盘"),
    (r"(监管|证监会).{0,5}(严查|叫停|处罚).{0,5}(量化|私募|基金)", "量化监管风险"),
    (r"(北向|外资).{0,5}(大幅净卖出|净流出).{0,5}(超过|突破).{0,5}(百亿|200亿)", "外资大幅流出"),
    (r"千股跌停", "市场恐慌"),
    (r"(上证|沪指|大盘).{0,5}(暴跌|大跌).{0,5}(超过|突破).{0,5}[4-9]%", "大盘暴跌"),
]


# ══════════════════════════════════════════════════════════════════════════════
#  数据结构
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BreakingAlert:
    """突发事件告警"""
    timestamp: str
    level: str            # BLACKSWAN / CRISIS / WARNING
    category: str         # 事件类别
    headline: str         # 触发的新闻标题
    matched_pattern: str  # 匹配到的模式描述
    source: str           # 数据来源
    action: str           # 建议动作

    @property
    def urgency_score(self) -> int:
        return {"BLACKSWAN": 3, "CRISIS": 2, "WARNING": 1}.get(self.level, 0)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MonitorResult:
    """监听结果"""
    scan_time: str
    alerts: list[BreakingAlert] = field(default_factory=list)
    highest_level: str = "NORMAL"
    position_override: float = 1.0     # 仓位覆写: 0=清仓, 0.5=半仓, 1.0=不干预
    should_halt_trading: bool = False   # 是否应暂停交易
    emergency_text: str = ""           # 紧急推送文本
    news_scanned: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ══════════════════════════════════════════════════════════════════════════════
#  新闻源抓取
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_tushare_flash(token: str, limit: int = 50) -> list[dict]:
    """通过tushare获取最新快讯"""
    if not token:
        return []
    try:
        import tushare as ts
        pro = ts.pro_api(token)
        # 尝试拉取最近几小时的新闻
        now = datetime.now()
        start = (now - timedelta(hours=4)).strftime("%Y-%m-%d %H:%M:%S")
        end = now.strftime("%Y-%m-%d %H:%M:%S")

        df = pro.news(src="sina", start_date=start, end_date=end)
        if df is not None and not df.empty:
            results = []
            for _, row in df.head(limit).iterrows():
                results.append({
                    "title": str(row.get("title", "")),
                    "content": str(row.get("content", ""))[:200],
                    "source": "tushare/sina",
                    "time": str(row.get("datetime", "")),
                })
            return results
    except Exception as e:
        print(f"[快讯监听] tushare快讯拉取失败: {e}")
    return []


def _fetch_cls_telegraph() -> list[dict]:
    """从财联社电报API获取快讯 (公开接口, 无需token)"""
    try:
        url = "https://www.cls.cn/nodeapi/updateTelegraph"
        req = Request(url, headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.cls.cn/telegraph",
        })
        resp = urlopen(req, timeout=10)
        data = json.loads(resp.read().decode("utf-8"))

        results = []
        items = data.get("data", {}).get("roll_data", [])
        for item in items[:50]:
            title = item.get("title", "") or item.get("brief", "") or ""
            content = item.get("content", "") or item.get("brief", "") or ""
            # 去除HTML标签
            content = re.sub(r"<[^>]+>", "", content)
            results.append({
                "title": title,
                "content": content[:200],
                "source": "cls",
                "time": datetime.fromtimestamp(
                    item.get("ctime", 0)
                ).strftime("%Y-%m-%d %H:%M:%S") if item.get("ctime") else "",
            })
        return results
    except Exception as e:
        print(f"[快讯监听] 财联社电报拉取失败: {e}")
    return []


def _fetch_eastmoney_news() -> list[dict]:
    """从东方财富获取快讯 (公开接口)"""
    try:
        url = ("https://push2ex.eastmoney.com/getAllNewsCount?"
               "ut=6d2ffaa6a585d612ede28e2c7e3e362a&type=0")
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urlopen(req, timeout=10)
        # 东财接口返回结构可能变化, 做好容错
        data = json.loads(resp.read().decode("utf-8"))
        # 如果有数据, 尝试解析
        results = []
        items = data.get("data", []) if isinstance(data.get("data"), list) else []
        for item in items[:30]:
            title = str(item.get("title", ""))
            if title:
                results.append({
                    "title": title,
                    "content": "",
                    "source": "eastmoney",
                    "time": str(item.get("showtime", "")),
                })
        return results
    except Exception:
        pass  # 东财接口不稳定, 静默失败
    return []


# ══════════════════════════════════════════════════════════════════════════════
#  监听引擎
# ══════════════════════════════════════════════════════════════════════════════

class BreakingMonitor:
    """
    突发事件监听器。

    用法:
        monitor = BreakingMonitor(tushare_token="xxx")
        result = monitor.scan()
        if result.should_halt_trading:
            print("紧急: 暂停交易!")

    守护模式:
        monitor.run_daemon(interval_minutes=5)
    """

    # 告警去重窗口 (秒)
    DEDUP_WINDOW = 1800  # 30分钟

    # 级别对应的仓位覆写
    POSITION_OVERRIDE = {
        "BLACKSWAN": 0.0,   # 清仓
        "CRISIS": 0.3,      # 仅保留30%
        "WARNING": 0.7,     # 收紧至70%
    }

    ACTION_MAP = {
        "BLACKSWAN": "立即清仓! 全部卖出, 保留现金, 等待事态明朗",
        "CRISIS": "暂停所有主动交易, 仅保留止损, 考虑减仓至30%",
        "WARNING": "收紧仓位至70%, 提高止损线, 暂缓新建仓",
    }

    def __init__(self, tushare_token: str = "", alert_log_path: str = ""):
        self.tushare_token = tushare_token
        self.alert_log_path = Path(alert_log_path) if alert_log_path else None
        self._seen_hashes: dict[str, float] = {}  # hash -> timestamp

    def _match_patterns(self, text: str,
                        patterns: list[tuple[str, str]],
                        level: str) -> Optional[tuple[str, str]]:
        """匹配关键词模式"""
        for pattern, category in patterns:
            if re.search(pattern, text):
                return (level, category)
        return None

    def _classify(self, headline: str, content: str = "") -> Optional[tuple[str, str]]:
        """对单条新闻进行风险分类"""
        text = f"{headline} {content}"

        # 按严重度依次匹配
        result = self._match_patterns(text, BLACKSWAN_PATTERNS, "BLACKSWAN")
        if result:
            return result

        result = self._match_patterns(text, CRISIS_PATTERNS, "CRISIS")
        if result:
            return result

        result = self._match_patterns(text, WARNING_PATTERNS, "WARNING")
        if result:
            return result

        return None

    def _dedup_key(self, level: str, category: str) -> str:
        """生成去重key"""
        raw = f"{level}:{category}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _is_duplicate(self, level: str, category: str) -> bool:
        """检查是否在去重窗口内"""
        key = self._dedup_key(level, category)
        now = time.time()
        if key in self._seen_hashes:
            if now - self._seen_hashes[key] < self.DEDUP_WINDOW:
                return True
        self._seen_hashes[key] = now
        return False

    def scan(self, headlines: list[dict] = None) -> MonitorResult:
        """
        执行一次扫描。

        Args:
            headlines: 手动传入新闻列表 (跳过网络抓取)
                       格式: [{"title": "...", "content": "...", "source": "..."}]
        Returns:
            MonitorResult
        """
        now = datetime.now().isoformat()

        # 1. 获取新闻
        if headlines is None:
            headlines = []
            # 多源并行 (实际串行, 但每个有超时)
            sources = [
                ("tushare", lambda: _fetch_tushare_flash(self.tushare_token)),
                ("cls", _fetch_cls_telegraph),
                ("eastmoney", _fetch_eastmoney_news),
            ]
            for name, fetcher in sources:
                try:
                    items = fetcher()
                    if items:
                        headlines.extend(items)
                        print(f"[快讯监听] {name}: 获取 {len(items)} 条")
                except Exception as e:
                    print(f"[快讯监听] {name} 失败: {e}")

        if not headlines:
            return MonitorResult(scan_time=now, news_scanned=0)

        # 2. 逐条分类
        alerts = []
        for item in headlines:
            title = item.get("title", "")
            content = item.get("content", "")
            source = item.get("source", "unknown")

            result = self._classify(title, content)
            if result is None:
                continue

            level, category = result

            # 去重
            if self._is_duplicate(level, category):
                continue

            alerts.append(BreakingAlert(
                timestamp=item.get("time", now),
                level=level,
                category=category,
                headline=title[:100],
                matched_pattern=category,
                source=source,
                action=self.ACTION_MAP.get(level, ""),
            ))

        # 3. 确定最高风险级别
        if not alerts:
            return MonitorResult(scan_time=now, news_scanned=len(headlines))

        alerts.sort(key=lambda a: a.urgency_score, reverse=True)
        highest = alerts[0].level

        pos_override = self.POSITION_OVERRIDE.get(highest, 1.0)
        halt = highest in ("BLACKSWAN", "CRISIS")

        # 4. 构建告警文本
        level_icons = {
            "BLACKSWAN": "🚨🚨🚨",
            "CRISIS": "🚨🚨",
            "WARNING": "⚠️",
        }
        lines = [f"{level_icons.get(highest, '⚠️')} 突发事件告警 [{highest}]", ""]

        for alert in alerts:
            icon = level_icons.get(alert.level, "⚠️")
            lines.append(f"{icon} [{alert.level}] {alert.category}")
            lines.append(f"   {alert.headline}")
            lines.append(f"   来源: {alert.source} | {alert.timestamp}")
            lines.append(f"   建议: {alert.action}")
            lines.append("")

        if halt:
            lines.append("═" * 50)
            lines.append(f"📛 紧急指令: 暂停所有交易, 仓位覆写至 {pos_override:.0%}")
            lines.append("═" * 50)

        emergency_text = "\n".join(lines)

        # 5. 记录日志
        if self.alert_log_path:
            try:
                self.alert_log_path.parent.mkdir(parents=True, exist_ok=True)
                log_entry = {
                    "scan_time": now,
                    "highest_level": highest,
                    "alerts": [a.to_dict() for a in alerts],
                }
                # 追加写入
                existing = []
                if self.alert_log_path.exists():
                    try:
                        existing = json.loads(
                            self.alert_log_path.read_text("utf-8")
                        )
                    except Exception:
                        existing = []
                existing.append(log_entry)
                self.alert_log_path.write_text(
                    json.dumps(existing, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as e:
                print(f"[快讯监听] 日志写入失败: {e}")

        return MonitorResult(
            scan_time=now,
            alerts=alerts,
            highest_level=highest,
            position_override=pos_override,
            should_halt_trading=halt,
            emergency_text=emergency_text,
            news_scanned=len(headlines),
        )

    def run_daemon(self, interval_minutes: int = 5,
                   notify_callback=None, max_iterations: int = 0):
        """
        守护模式: 每隔N分钟轮询一次。

        Args:
            interval_minutes: 轮询间隔 (分钟)
            notify_callback: 告警回调 fn(title, content) → 用notifier推送
            max_iterations: 最大轮询次数 (0=无限)
        """
        print(f"[快讯监听] 守护模式启动, 间隔 {interval_minutes} 分钟")
        print(f"[快讯监听] Ctrl+C 退出")
        print()

        iteration = 0
        while True:
            iteration += 1
            if max_iterations > 0 and iteration > max_iterations:
                print("[快讯监听] 达到最大轮询次数, 退出")
                break

            try:
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"[{ts}] 第 {iteration} 次扫描...")

                result = self.scan()

                if result.alerts:
                    print(result.emergency_text)

                    # 回调推送
                    if notify_callback and result.highest_level != "NORMAL":
                        try:
                            title = f"🚨 突发事件 [{result.highest_level}]"
                            notify_callback(title, result.emergency_text)
                        except Exception as e:
                            print(f"[快讯监听] 推送失败: {e}")
                else:
                    print(f"  扫描 {result.news_scanned} 条, 无异常")

            except KeyboardInterrupt:
                print("\n[快讯监听] 用户中断, 退出")
                break
            except Exception as e:
                print(f"[快讯监听] 扫描异常: {e}")

            if max_iterations > 0 and iteration >= max_iterations:
                break

            time.sleep(interval_minutes * 60)

    def format_opus_context(self, result: MonitorResult) -> str:
        """为Opus反思生成突发事件上下文"""
        if not result.alerts:
            return ""

        lines = [f"突发事件告警 (扫描时间 {result.scan_time}):"]
        for alert in result.alerts:
            lines.append(f"  [{alert.level}] {alert.category}: {alert.headline}")
        lines.append(f"风险等级: {result.highest_level}")
        lines.append(f"仓位覆写: {result.position_override:.0%}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="突发事件监听器")
    parser.add_argument("--token", type=str, help="Tushare API token")
    parser.add_argument("--daemon", action="store_true", help="守护模式")
    parser.add_argument("--interval", type=int, default=5, help="守护轮询间隔(分钟)")
    parser.add_argument("--test", action="store_true", help="模拟黑天鹅测试")
    parser.add_argument("--log", type=str, default="", help="告警日志路径")
    args = parser.parse_args()

    monitor = BreakingMonitor(
        tushare_token=args.token or "",
        alert_log_path=args.log,
    )

    if args.test:
        print("=== 模拟黑天鹅测试 ===\n")
        test_news = [
            {"title": "突发: 美军对伊朗发动大规模空袭, 中东局势急剧升级",
             "content": "据多家外媒报道, 美军战机于今日凌晨对伊朗军事设施发动空袭", "source": "test"},
            {"title": "美股期货暴跌, 道琼斯期指跌超1000点",
             "content": "受中东冲突升级影响, 美股三大期指全线暴跌", "source": "test"},
            {"title": "国际油价暴涨15%, 布伦特原油突破120美元",
             "content": "中东战争风险推动原油暴涨", "source": "test"},
            {"title": "北向资金大幅净卖出超过200亿, 外资避险情绪激增",
             "content": "地缘冲突引发外资恐慌性撤离A股", "source": "test"},
            {"title": "央行紧急声明维护金融市场稳定",
             "content": "针对国际局势变化, 央行表示将维护市场流动性", "source": "test"},
        ]
        result = monitor.scan(headlines=test_news)
        print(result.emergency_text)
        print(f"\n扫描 {result.news_scanned} 条 | 告警 {len(result.alerts)} 条")
        print(f"最高级别: {result.highest_level}")
        print(f"仓位覆写: {result.position_override:.0%}")
        print(f"暂停交易: {result.should_halt_trading}")

    elif args.daemon:
        monitor.run_daemon(interval_minutes=args.interval)
    else:
        result = monitor.scan()
        if result.alerts:
            print(result.emergency_text)
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 扫描 {result.news_scanned} 条快讯, 未检测到异常")
