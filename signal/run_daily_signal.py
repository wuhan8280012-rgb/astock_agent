#!/usr/bin/env python3
"""
每日交易信号 — 一键入口脚本

使用方式:
  1. 在线模式 (推荐, 自动拉取最新行情):
     python signal/run_daily_signal.py

  2. 离线模式 (使用CSV文件):
     python signal/run_daily_signal.py --csv data/csi1000_market_bundle.csv

  3. 指定日期分析:
     python signal/run_daily_signal.py --date 20260320

  4. 手动确认执行后更新持仓:
     python signal/run_daily_signal.py --confirm 20260320

  5. 盘前复检 (次日开盘前验证前日信号):
     python signal/run_daily_signal.py --precheck

定时运行 (crontab):
  # 每个交易日 15:30 收盘后生成信号
  30 15 * * 1-5 cd /path/to/stock_agent/new && python signal/run_daily_signal.py >> logs/daily_signal.log 2>&1
  # 每个交易日 08:50 盘前复检 (开盘前10分钟)
  50 8 * * 1-5 cd /path/to/stock_agent/new && python signal/run_daily_signal.py --precheck >> logs/precheck.log 2>&1
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from signal.signal_generator import SignalGenerator, SignalConfig, SignalResult
from signal.notifier import Notifier, NotifyConfig, format_signal_html


def load_config() -> tuple[SignalConfig, NotifyConfig]:
    """加载所有配置"""
    sig_cfg = SignalConfig()
    notify_cfg = NotifyConfig()

    # 1. 从 .env 文件加载
    env_path = PROJECT_ROOT / "config" / ".env"
    if env_path.exists():
        for line in env_path.read_text().strip().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

    # Tushare token
    sig_cfg.tushare_token = os.environ.get("TUSHARE_TOKEN", "")

    # 2. 从 notify_config.json 加载推送配置
    notify_json = PROJECT_ROOT / "config" / "notify_config.json"
    if notify_json.exists():
        notify_cfg = NotifyConfig.from_json(notify_json)

    # 3. 用环境变量覆盖 JSON，避免敏感 webhook 落在配置文件里
    env_notify = NotifyConfig.from_env()
    for key, value in env_notify.__dict__.items():
        default_value = getattr(NotifyConfig(), key)
        if value != default_value:
            setattr(notify_cfg, key, value)

    return sig_cfg, notify_cfg


def generate_and_push(sig_cfg: SignalConfig, notify_cfg: NotifyConfig,
                      as_of_date: str = None) -> SignalResult:
    """生成信号并推送"""

    # 生成信号
    gen = SignalGenerator(sig_cfg)
    result = gen.generate(as_of_date=as_of_date)

    if not result.success:
        print(f"\n[错误] 信号生成失败: {result.error_message}")
        # 推送错误通知
        notifier = Notifier(notify_cfg)
        notifier.send(
            title=f"⚠️ 动量策略信号异常 {datetime.now().strftime('%m-%d')}",
            content=f"信号生成失败: {result.error_message}",
        )
        return result

    # 打印信号
    print(result.summary_text)

    # 推送
    notifier = Notifier(notify_cfg)
    html = format_signal_html(result)
    notifier.send(
        title=f"动量轮动信号 {result.trade_date} [{result.market_regime}]",
        content=result.summary_text,
        html_content=html,
    )

    # 保存信号到日志
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"signal_{result.trade_date}.txt"
    log_file.write_text(result.summary_text, encoding="utf-8")
    print(f"\n[日志] 信号报告已保存: {log_file}")

    return result


def _fetch_overseas_markets() -> dict:
    """
    盘前抓取外围市场隔夜表现。

    数据源: 东方财富全球指数接口 (公开, 无需token)
    关注标的:
      - 富时A50期货 (最直接的A股领先指标)
      - 标普500 / 纳斯达克 (美股隔夜)
      - 恒生指数 (港股夜盘)

    Returns:
        {
            "success": bool,
            "indicators": [{"name": "富时A50", "pct_chg": -3.2, "source": "..."}],
            "worst_drop": float,     # 最大跌幅
            "alert_level": str,      # "NORMAL" / "WARNING" / "CRITICAL"
            "summary": str,
        }
    """
    from urllib.request import Request, urlopen
    import re

    result = {
        "success": False,
        "indicators": [],
        "worst_drop": 0.0,
        "alert_level": "NORMAL",
        "summary": "",
    }

    # 东方财富全球指数实时行情
    targets = {
        "富时A50期货": "107.CNFUTR",     # 新加坡A50期货
        "标普500": "100.SPX",
        "纳斯达克": "100.NDX",
        "恒生指数": "100.HSI",
        "日经225": "100.N225",
    }

    # 尝试从东财全球指数页面抓数据
    try:
        # 东财全球指数API (公开)
        secids = ",".join(targets.values())
        url = (
            f"https://push2.eastmoney.com/api/qt/ulist.np/get?"
            f"fltt=2&secids={secids}&"
            f"fields=f2,f3,f4,f12,f14&ut=fa5fd1943c7b386f172d6893dbbd1"
        )
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urlopen(req, timeout=8)
        data = json.loads(resp.read().decode("utf-8"))

        items = data.get("data", {}).get("diff", [])
        if items:
            result["success"] = True
            for item in items:
                name = str(item.get("f14", ""))
                pct = item.get("f3")  # 涨跌幅%
                if pct is not None:
                    pct = float(pct)
                    result["indicators"].append({
                        "name": name,
                        "pct_chg": pct,
                    })
    except Exception as e:
        print(f"[外围] 东财全球指数接口失败: {e}")

    # 备选: 从新浪财经获取A50期货
    if not result["success"]:
        try:
            url = "https://hq.sinajs.cn/list=CHA50CFD"
            req = Request(url, headers={
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://finance.sina.com.cn",
            })
            resp = urlopen(req, timeout=8)
            text = resp.read().decode("gbk")
            # 解析: var hq_str_CHA50CFD="...";
            match = re.search(r'"([^"]+)"', text)
            if match:
                fields = match.group(1).split(",")
                if len(fields) > 8:
                    current = float(fields[0])
                    prev_close = float(fields[7])
                    if prev_close > 0:
                        pct = (current / prev_close - 1) * 100
                        result["success"] = True
                        result["indicators"].append({
                            "name": "富时A50期货(新浪)",
                            "pct_chg": round(pct, 2),
                        })
        except Exception as e:
            print(f"[外围] 新浪A50接口失败: {e}")

    # 分析结果
    if result["indicators"]:
        drops = [i["pct_chg"] for i in result["indicators"] if i["pct_chg"] < 0]
        result["worst_drop"] = min(drops) if drops else 0.0

        parts = []
        for ind in result["indicators"]:
            icon = "🔴" if ind["pct_chg"] < -1 else ("🟡" if ind["pct_chg"] < 0 else "🟢")
            parts.append(f"{icon} {ind['name']} {ind['pct_chg']:+.2f}%")
        result["summary"] = " | ".join(parts)

        # 判定级别
        # A50跌超3% 或 两个以上主要指数跌超2% → CRITICAL
        a50_drop = next((i["pct_chg"] for i in result["indicators"]
                         if "A50" in i["name"] or "a50" in i["name"].lower()), 0)
        big_drops = [i for i in result["indicators"] if i["pct_chg"] < -2]

        if a50_drop < -3 or len(big_drops) >= 2:
            result["alert_level"] = "CRITICAL"
        elif a50_drop < -1.5 or len(big_drops) >= 1:
            result["alert_level"] = "WARNING"

    return result


def precheck_before_open(sig_cfg: SignalConfig, notify_cfg: NotifyConfig):
    """
    盘前复检 — 在执行前日信号前, 扫描盘前可获取的增量信息。

    核心认知:
      盘前8:50无法获取当日A股行情 (还没开盘), 所以不能重算regime。
      但可以获取两类增量信息:
        1. 外围市场隔夜表现 (A50期货/美股/港股) — 对A股开盘有强领先性
        2. 突发新闻 (财联社/东财/tushare) — 凌晨/隔夜的重大事件

    判定逻辑:
      ABORT (废弃信号):
        - 突发事件达到 BLACKSWAN/CRISIS 级别
        - A50期货跌超3% 或 多个外围指数跌超2%
      DOWNGRADE (降级警告):
        - 突发事件达到 WARNING 级别
        - A50期货跌超1.5% 或 有外围指数跌超2%
        - 当日处于宏观事件窗口期
      CONFIRMED (放行):
        - 以上均无异常

    执行时机: crontab 设置为 08:50 (开盘前10分钟)
    """

    # 1. 找到最近一份待执行的信号文件
    signal_dir = PROJECT_ROOT / "data" / "signals"
    if not signal_dir.exists():
        print("[复检] 无信号目录")
        return

    signal_files = sorted(signal_dir.glob("signal_*.json"), reverse=True)
    if not signal_files:
        print("[复检] 无历史信号文件")
        return

    latest_signal_path = signal_files[0]
    signal_data = json.loads(latest_signal_path.read_text("utf-8"))
    signal_date = signal_data.get("trade_date", "")
    signal_regime = signal_data.get("market_regime", "")
    buy_signals = [s for s in signal_data.get("signals", []) if s.get("action") == "BUY"]

    # 已经复检过的不重复
    if signal_data.get("precheck_result"):
        prev = signal_data["precheck_result"]
        print(f"[复检] {signal_date} 已复检过: {prev.get('action')} @ {prev.get('checked_at', '')[:16]}")
        return

    if not buy_signals:
        print(f"[复检] {signal_date} 信号无买入操作, 无需复检")
        return

    buy_names = ", ".join(s["name"] for s in buy_signals)
    buy_amount = sum(s.get("target_amount", 0) for s in buy_signals)
    print(f"[复检] 待执行信号: {signal_date} [{signal_regime}], {len(buy_signals)}笔买入")
    print(f"[复检] 标的: {buy_names}, 金额约 ¥{buy_amount:,.0f}")

    # ── 信息收集 ──

    # 2. 外围市场隔夜表现
    print("\n[复检] 扫描外围市场...")
    overseas = _fetch_overseas_markets()
    if overseas["success"]:
        print(f"  {overseas['summary']}")
        print(f"  最大跌幅: {overseas['worst_drop']:+.2f}% | 级别: {overseas['alert_level']}")
    else:
        print("  外围数据获取失败 (非致命)")

    # 3. 突发新闻扫描
    print("\n[复检] 扫描突发新闻...")
    breaking_halt = False
    breaking_crisis = False
    breaking_warning = False
    breaking_text = ""
    breaking_alerts = []

    if sig_cfg.enable_breaking_monitor:
        try:
            from signal.breaking_monitor import BreakingMonitor
            monitor = BreakingMonitor(
                tushare_token=sig_cfg.tushare_token,
                alert_log_path=str(PROJECT_ROOT / "data" / "breaking_alerts.json"),
            )
            br_result = monitor.scan()
            if br_result.alerts:
                breaking_alerts = br_result.alerts
                breaking_text = br_result.emergency_text
                level = br_result.highest_level
                if level == "BLACKSWAN":
                    breaking_halt = True
                elif level == "CRISIS":
                    breaking_crisis = True
                elif level == "WARNING":
                    breaking_warning = True
                print(f"  发现 {len(br_result.alerts)} 条告警, 最高级别: {level}")
                print(f"  {breaking_text[:200]}")
            else:
                print("  未发现突发事件")
        except Exception as e:
            print(f"  突发事件扫描失败 (非致命): {e}")
    else:
        print("  突发事件监听未启用")

    # 4. 宏观事件日历检查 (今天是否有重大事件)
    print("\n[复检] 检查宏观日历...")
    macro_quiet = False
    macro_text = ""
    today_str = datetime.now().strftime("%Y%m%d")
    if sig_cfg.enable_macro_calendar:
        try:
            from signal.macro_calendar import MacroCalendar
            cal = MacroCalendar()
            assessment = cal.assess(today_str)
            if assessment.quiet_period:
                macro_quiet = True
                macro_text = assessment.alert_text
                print(f"  {macro_text[:150]}")
            else:
                print(f"  风险级别: {assessment.risk_level} (无静默期)")
        except Exception as e:
            print(f"  宏观日历检查失败 (非致命): {e}")
    else:
        print("  宏观日历未启用")

    # ── 综合判定 ──

    should_abort = False
    should_downgrade = False
    reasons = []

    # 突发事件: BLACKSWAN/CRISIS → 废弃
    if breaking_halt:
        should_abort = True
        reasons.append(f"突发BLACKSWAN: {breaking_text[:80]}")
    elif breaking_crisis:
        should_abort = True
        reasons.append(f"突发CRISIS: {breaking_text[:80]}")

    # 外围市场: CRITICAL → 废弃, WARNING → 降级
    if overseas["alert_level"] == "CRITICAL":
        should_abort = True
        reasons.append(f"外围暴跌: {overseas['summary']}")
    elif overseas["alert_level"] == "WARNING":
        should_downgrade = True
        reasons.append(f"外围下跌: {overseas['summary']}")

    # 突发WARNING + 宏观静默 → 降级
    if breaking_warning:
        should_downgrade = True
        reasons.append(f"突发WARNING: {breaking_text[:60]}")
    if macro_quiet:
        should_downgrade = True
        reasons.append(f"宏观静默期: {macro_text[:60]}")

    # ── 输出结果 ──

    notifier = Notifier(notify_cfg)
    check_time = datetime.now().isoformat()

    if should_abort:
        abort_msg = (
            f"🚫 盘前复检: 废弃 {signal_date} 买入信号\n\n"
            f"原始信号: {signal_regime}, 买入 {buy_names}\n"
            f"废弃原因:\n" +
            "\n".join(f"  · {r}" for r in reasons) +
            f"\n\n⚠️ 今日不执行任何买入操作!"
        )
        print(f"\n{'='*60}")
        print(abort_msg)
        print(f"{'='*60}")

        notifier.send(title=f"🚫 买入信号废弃", content=abort_msg)

        signal_data["precheck_result"] = {
            "checked_at": check_time,
            "action": "ABORT",
            "reasons": reasons,
            "overseas": overseas if overseas["success"] else None,
            "breaking_count": len(breaking_alerts),
        }

    elif should_downgrade:
        warn_msg = (
            f"⚠️ 盘前复检: 信号降级警告\n\n"
            f"原始信号: {signal_regime}, 买入 {buy_names}\n"
            f"风险提示:\n" +
            "\n".join(f"  · {r}" for r in reasons) +
            f"\n\n建议: 减半执行或观望, 谨慎操作"
        )
        print(f"\n{'='*60}")
        print(warn_msg)
        print(f"{'='*60}")

        notifier.send(title=f"⚠️ 信号降级警告", content=warn_msg)

        signal_data["precheck_result"] = {
            "checked_at": check_time,
            "action": "DOWNGRADE",
            "reasons": reasons,
            "overseas": overseas if overseas["success"] else None,
        }

    else:
        overseas_note = overseas["summary"] if overseas["success"] else "外围数据未获取"
        ok_msg = (
            f"✅ 盘前复检通过\n\n"
            f"信号日期: {signal_date} [{signal_regime}]\n"
            f"外围市场: {overseas_note}\n"
            f"突发事件: 无\n"
            f"可执行买入: {buy_names}\n\n"
            f"请在开盘后执行"
        )
        print(f"\n{'='*60}")
        print(ok_msg)
        print(f"{'='*60}")

        notifier.send(title=f"✅ 复检通过, 可执行买入", content=ok_msg)

        signal_data["precheck_result"] = {
            "checked_at": check_time,
            "action": "CONFIRMED",
            "overseas": overseas if overseas["success"] else None,
        }

    latest_signal_path.write_text(
        json.dumps(signal_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[复检] 结果已写入信号文件")


def confirm_trades(date: str):
    """手动确认今日交易已执行, 更新持仓状态"""
    signal_file = PROJECT_ROOT / "data" / "signals" / f"signal_{date}.json"
    if not signal_file.exists():
        print(f"[错误] 未找到 {date} 的信号文件")
        return

    data = json.loads(signal_file.read_text("utf-8"))
    signals = data.get("signals", [])

    action_signals = [s for s in signals if s["action"] != "HOLD"]
    if not action_signals:
        print(f"[信息] {date} 无需执行的交易")
        return

    print(f"\n确认 {date} 的交易执行:")
    for i, s in enumerate(action_signals):
        action_cn = {"BUY": "买入", "SELL": "卖出", "STOP_LOSS": "止损"}.get(s["action"], s["action"])
        print(f"  {i+1}. {action_cn} {s['name']}({s['ts_code']}) {s['target_shares']}股 @¥{s['current_price']:.2f}")

    # 加载持仓状态
    from signal.signal_generator import PortfolioState, TradeSignal

    portfolio_path = PROJECT_ROOT / "data" / "portfolio_state.json"
    portfolio = PortfolioState(portfolio_path)

    confirm = input("\n是否全部确认执行? (y/n/逐条确认c): ").strip().lower()

    if confirm == "y":
        for s in action_signals:
            sig = TradeSignal(**s)
            # 用信号价格作为执行价 (实际可输入真实成交价)
            exec_price = s["current_price"]
            portfolio.apply_trade(sig, exec_price)
            action_cn = {"BUY": "买入", "SELL": "卖出", "STOP_LOSS": "止损"}.get(s["action"], s["action"])
            print(f"  ✓ {action_cn} {s['name']} {s['target_shares']}股 @¥{exec_price:.2f}")

        portfolio.data["last_rebalance_date"] = date
        portfolio.data["rebalance_count"] = portfolio.data.get("rebalance_count", 0) + 1
        portfolio.save()
        print(f"\n[持仓] 已更新, 总资产: ¥{portfolio.total_value:,.2f}")

    elif confirm == "c":
        for s in action_signals:
            action_cn = {"BUY": "买入", "SELL": "卖出", "STOP_LOSS": "止损"}.get(s["action"], s["action"])
            ans = input(f"  {action_cn} {s['name']} {s['target_shares']}股? (y/n, 或输入实际成交价): ").strip()
            if ans.lower() == "n":
                print(f"    跳过")
                continue

            try:
                exec_price = float(ans) if ans.replace(".", "").isdigit() else s["current_price"]
            except ValueError:
                exec_price = s["current_price"]

            sig = TradeSignal(**s)
            portfolio.apply_trade(sig, exec_price)
            print(f"    ✓ 已确认 @¥{exec_price:.2f}")

        portfolio.data["last_rebalance_date"] = date
        portfolio.data["rebalance_count"] = portfolio.data.get("rebalance_count", 0) + 1
        portfolio.save()
        print(f"\n[持仓] 已更新, 总资产: ¥{portfolio.total_value:,.2f}")
    else:
        print("[取消] 未执行任何交易")


def init_portfolio():
    """初始化持仓状态 (首次使用)"""
    portfolio_path = PROJECT_ROOT / "data" / "portfolio_state.json"
    if portfolio_path.exists():
        print(f"[持仓] 已存在: {portfolio_path}")
        data = json.loads(portfolio_path.read_text("utf-8"))
        print(f"  总资产: ¥{data.get('cash', 0):,.2f}")
        print(f"  持仓数: {len(data.get('positions', {}))}")
        ans = input("是否重新初始化? (y/n): ").strip().lower()
        if ans != "y":
            return

    capital = input("请输入初始资金 (默认100万): ").strip()
    try:
        capital = float(capital) if capital else 1_000_000.0
    except ValueError:
        capital = 1_000_000.0

    from signal.signal_generator import PortfolioState
    state = PortfolioState(portfolio_path)
    state.data["initial_capital"] = capital
    state.data["cash"] = capital
    state.data["positions"] = {}
    state.save()
    print(f"[持仓] 已初始化, 初始资金: ¥{capital:,.2f}")
    print(f"  文件: {portfolio_path}")


def show_status():
    """显示当前持仓状态"""
    portfolio_path = PROJECT_ROOT / "data" / "portfolio_state.json"
    if not portfolio_path.exists():
        print("[持仓] 未初始化, 请先运行: python signal/run_daily_signal.py --init")
        return

    data = json.loads(portfolio_path.read_text("utf-8"))
    positions = data.get("positions", {})
    cash = data.get("cash", 0)
    pos_value = sum(p.get("shares", 0) * p.get("current_price", p.get("cost_price", 0)) for p in positions.values())
    total = cash + pos_value

    print(f"\n{'='*50}")
    print(f"  持仓状态  (更新: {data.get('updated_at', 'N/A')})")
    print(f"{'='*50}")
    print(f"  总资产: ¥{total:,.2f}")
    print(f"  现金:   ¥{cash:,.2f} ({cash/total*100:.1f}%)" if total > 0 else f"  现金:   ¥{cash:,.2f}")
    print(f"  持仓:   ¥{pos_value:,.2f} ({pos_value/total*100:.1f}%)" if total > 0 else f"  持仓:   ¥{pos_value:,.2f}")
    print(f"  标的数: {len(positions)}")
    print(f"  调仓次数: {data.get('rebalance_count', 0)}")
    print(f"  上次调仓: {data.get('last_rebalance_date', 'N/A')}")

    if positions:
        print(f"\n  {'代码':<12s} {'名称':<8s} {'股数':>6s} {'成本':>8s} {'现价':>8s} {'盈亏':>8s} {'市值':>10s}")
        print(f"  {'-'*70}")
        for code, pos in positions.items():
            cost = pos.get("cost_price", 0)
            current = pos.get("current_price", cost)
            shares = pos.get("shares", 0)
            pnl = (current / cost - 1) * 100 if cost > 0 else 0
            mv = shares * current
            pnl_str = f"{pnl:+.2f}%"
            print(f"  {code:<12s} {pos.get('name',''):<8s} {shares:>6d} {cost:>8.2f} {current:>8.2f} {pnl_str:>8s} {mv:>10,.0f}")

    # 最近交易记录
    history = data.get("trade_history", [])[-10:]
    if history:
        print(f"\n  最近交易:")
        for h in history:
            action_cn = {"BUY": "买入", "SELL": "卖出", "STOP_LOSS": "止损"}.get(h["action"], h["action"])
            print(f"    {h['date']} {action_cn} {h['name']} {h['shares']}股 @¥{h['price']:.2f}")

    print()


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="动量轮动策略 · 每日信号生成与推送",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--date", type=str, help="指定分析日期 (YYYYMMDD)")
    parser.add_argument("--csv", type=str, help="使用CSV文件 (离线模式)")
    parser.add_argument("--confirm", type=str, metavar="DATE", help="确认指定日期的交易已执行")
    parser.add_argument("--init", action="store_true", help="初始化持仓状态")
    parser.add_argument("--status", action="store_true", help="查看当前持仓状态")
    parser.add_argument("--token", type=str, help="Tushare API token (覆盖.env)")
    parser.add_argument("--test-notify", action="store_true", help="测试推送通道")
    parser.add_argument("--precheck", action="store_true", help="盘前复检: 用最新数据验证前日信号是否仍可执行")
    parser.add_argument("--monitor", action="store_true", help="启动突发事件守护监听 (盘中实时)")
    parser.add_argument("--monitor-interval", type=int, default=5, help="监听间隔(分钟, 默认5)")
    parser.add_argument("--check-evolution", action="store_true", help="检查是否需要触发Opus进化")
    parser.add_argument("--evolution-status", action="store_true", help="查看进化触发器状态")

    args = parser.parse_args()

    # 初始化持仓
    if args.init:
        init_portfolio()
        return

    # 查看状态
    if args.status:
        show_status()
        return

    # 确认交易
    if args.confirm:
        confirm_trades(args.confirm)
        return

    # 测试推送
    if args.test_notify:
        _, notify_cfg = load_config()
        notifier = Notifier(notify_cfg)
        results = notifier.send(
            title="[测试] 动量轮动策略推送测试",
            content="如果你收到这条消息, 说明推送通道配置正确。\n\n测试时间: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        for r in results:
            status = "✓" if r.get("success") else "✗"
            print(f"  {status} {r.get('channel')}")
        return

    # 盘前复检
    if args.precheck:
        sig_cfg, notify_cfg = load_config()
        if args.csv:
            sig_cfg.data_csv_path = args.csv
        if args.token:
            sig_cfg.tushare_token = args.token
        precheck_before_open(sig_cfg, notify_cfg)
        return

    # 突发事件守护模式
    if args.monitor:
        sig_cfg, notify_cfg = load_config()
        if args.token:
            sig_cfg.tushare_token = args.token

        from signal.breaking_monitor import BreakingMonitor
        monitor = BreakingMonitor(
            tushare_token=sig_cfg.tushare_token,
            alert_log_path=str(PROJECT_ROOT / "data" / "breaking_alerts.json"),
        )
        notifier = Notifier(notify_cfg)

        def on_alert(title, content):
            notifier.send(title=title, content=content)

        monitor.run_daemon(
            interval_minutes=args.monitor_interval,
            notify_callback=on_alert,
        )
        return

    # 进化触发检查
    if args.check_evolution or args.evolution_status:
        from signal.evolution_trigger import EvolutionTrigger
        trigger = EvolutionTrigger(data_dir=str(PROJECT_ROOT / "data"))

        if args.evolution_status:
            print(trigger.format_status_text())
        else:
            should, reason = trigger.check()
            if should:
                print(f"🟢 需要触发进化: {reason}")
                print(f"  执行: python scripts/run_evolution_test.py")
            else:
                print(f"⚪ 无需进化: {reason}")
        return

    # 生成信号并推送
    sig_cfg, notify_cfg = load_config()
    if args.csv:
        sig_cfg.data_csv_path = args.csv
    if args.token:
        sig_cfg.tushare_token = args.token

    # 检查是否有tushare token或CSV
    if not sig_cfg.tushare_token and not sig_cfg.data_csv_path:
        print("[警告] 未配置 TUSHARE_TOKEN 且未指定 --csv 文件")
        print("  请在 config/.env 中配置 TUSHARE_TOKEN")
        print("  或使用 --csv 参数指定CSV数据文件")
        return

    # 检查持仓是否初始化
    portfolio_path = PROJECT_ROOT / "data" / "portfolio_state.json"
    if not portfolio_path.exists():
        print("[提示] 首次使用, 需要初始化持仓状态")
        init_portfolio()

    result = generate_and_push(sig_cfg, notify_cfg, as_of_date=args.date)

    if result.success:
        # 周五自动记录周度绩效快照 (用于进化触发判断)
        try:
            from datetime import date as dt_date
            trade_dt = datetime.strptime(result.trade_date, "%Y%m%d")
            if trade_dt.weekday() == 4:  # 周五
                from signal.evolution_trigger import EvolutionTrigger, WeeklySnapshot
                trigger = EvolutionTrigger(data_dir=str(PROJECT_ROOT / "data"))

                # 计算周收益率 (简单近似: 与上一快照对比)
                recent = trigger.tracker.get_recent(1)
                last_value = recent[0]["portfolio_value"] if recent else result.total_value
                weekly_ret = (result.total_value / last_value - 1) * 100 if last_value > 0 else 0

                snapshot = WeeklySnapshot(
                    week_end=result.trade_date,
                    portfolio_value=result.total_value,
                    weekly_return=round(weekly_ret, 2),
                    weekly_sharpe=round(trigger.tracker.calc_rolling_sharpe(), 2),
                    max_drawdown=round(trigger.tracker.calc_rolling_drawdown(), 2),
                    market_regime=result.market_regime,
                    trade_count=sum(1 for s in result.signals if s.action != "HOLD"),
                )
                trigger.tracker.record_week(snapshot)
                print(f"[进化] 周度快照已记录: 收益{weekly_ret:+.2f}% | 夏普{snapshot.weekly_sharpe:.2f}")

                # 自动检查是否需要触发进化
                breaking_level = None
                if result.breaking_alerts and result.breaking_alerts.get("highest_level"):
                    breaking_level = result.breaking_alerts["highest_level"]

                should, reason = trigger.check(
                    current_regime=result.market_regime,
                    breaking_level=breaking_level,
                )
                if should:
                    print(f"[进化] 🟢 触发条件满足: {reason}")
                    print(f"[进化] 请运行: python scripts/run_evolution_test.py")
        except Exception as e:
            print(f"[进化] 快照记录失败 (非致命): {e}")

        # 提示下一步
        action_count = sum(1 for s in result.signals if s.action != "HOLD")
        if action_count > 0:
            print(f"\n[下一步] 在广发易淘金APP执行上述 {action_count} 笔交易")
            print(f"  执行完毕后运行: python signal/run_daily_signal.py --confirm {result.trade_date}")
        else:
            print(f"\n[提示] 今日无需操作")


if __name__ == "__main__":
    main()
