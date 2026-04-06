#!/usr/bin/env python3
"""
交易确认工具 — 将信号执行结果写入持仓状态

使用方式:

  1. 确认买入 (按信号全部执行):
     python scripts/confirm_trades.py --apply-signal 20260327

  2. 手动录入单笔交易:
     python scripts/confirm_trades.py --buy 600726.SH --shares 22400 --price 6.68
     python scripts/confirm_trades.py --sell 600726.SH --price 7.10

  3. 查看当前持仓:
     python scripts/confirm_trades.py --show

  4. 从CSV批量导入 (支持广发易淘金导出格式):
     python scripts/confirm_trades.py --import trades.csv

  5. 重置持仓 (清空):
     python scripts/confirm_trades.py --reset
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_PORTFOLIO_PATH = PROJECT_ROOT / "data" / "portfolio_state.json"
STRATEGY_PORTFOLIO_PATHS = {
    "f": PROJECT_ROOT / "data" / "f_strategy_portfolio_state.json",
}


def default_portfolio() -> dict:
    return {
        "initial_capital": 1_000_000.0,
        "cash": 1_000_000.0,
        "positions": {},
        "last_rebalance_date": "",
        "rebalance_count": 0,
        "trade_history": [],
        "updated_at": "",
    }


def resolve_portfolio_path(portfolio_path: str | Path | None = None, strategy: str = "") -> Path:
    if portfolio_path:
        return Path(portfolio_path)
    strategy_key = strategy.strip().lower()
    if strategy_key in STRATEGY_PORTFOLIO_PATHS:
        return STRATEGY_PORTFOLIO_PATHS[strategy_key]
    return DEFAULT_PORTFOLIO_PATH


def load_portfolio(portfolio_path: str | Path | None = None, strategy: str = "") -> dict:
    resolved_path = resolve_portfolio_path(portfolio_path, strategy=strategy)
    if resolved_path.exists():
        return json.loads(resolved_path.read_text("utf-8"))
    return default_portfolio()


def save_portfolio(data: dict, portfolio_path: str | Path | None = None, strategy: str = ""):
    resolved_path = resolve_portfolio_path(portfolio_path, strategy=strategy)
    data["updated_at"] = datetime.now().isoformat()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
    print(f"[持仓] 已保存: {resolved_path}")


def show_portfolio(data: dict, portfolio_path: str | Path | None = None, strategy: str = ""):
    resolved_path = resolve_portfolio_path(portfolio_path, strategy=strategy)
    positions = data.get("positions", {})
    cash = data.get("cash", 0)
    total_pos_value = sum(
        p.get("shares", 0) * p.get("current_price", p.get("cost_price", 0))
        for p in positions.values()
    )
    total = cash + total_pos_value
    updated = data.get("updated_at", "未知")
    last_rb = data.get("last_rebalance_date", "未设置")

    print(f"\n{'='*60}")
    print(f"  持仓状态  (更新于: {updated})")
    print(f"  状态文件: {resolved_path}")
    print(f"  上次调仓: {last_rb}")
    print(f"{'='*60}")
    print(f"  总资产: ¥{total:,.2f}")
    print(f"  现金:   ¥{cash:,.2f}  ({cash/total*100:.1f}%)" if total > 0 else f"  现金:   ¥{cash:,.2f}")
    print(f"  持仓:   ¥{total_pos_value:,.2f}  ({total_pos_value/total*100:.1f}%)" if total > 0 else f"  持仓:   ¥{total_pos_value:,.2f}")

    if positions:
        print(f"\n  {'代码':<12s} {'名称':<10s} {'股数':>7s} {'成本':>8s} {'现价':>8s} {'市值':>12s} {'盈亏':>8s}")
        print(f"  {'─'*75}")
        for code, pos in positions.items():
            shares = pos.get("shares", 0)
            cost = pos.get("cost_price", 0)
            cur = pos.get("current_price", cost)
            mv = shares * cur
            pnl_pct = (cur / cost - 1) * 100 if cost > 0 else 0
            name = pos.get("name", "?")
            print(f"  {code:<12s} {name:<10s} {shares:>7d} {cost:>8.2f} {cur:>8.2f} ¥{mv:>10,.0f} {pnl_pct:>+7.2f}%")
    else:
        print("\n  (空仓)")

    print(f"\n  交易历史: {len(data.get('trade_history', []))} 笔")
    print(f"{'='*60}\n")


def resolve_signal_path(signal_date: str, signal_path: str | Path | None = None, strategy: str = "") -> Path | None:
    if signal_path:
        candidate = Path(signal_path)
        return candidate if candidate.exists() else None

    strategy_key = strategy.strip().lower()
    signal_paths = [
        PROJECT_ROOT / "data" / "signals" / f"signal_{signal_date}.json",
        PROJECT_ROOT / "data" / f"signal_{signal_date}.json",
        PROJECT_ROOT / "data" / "signals" / "latest_signal.json",
        PROJECT_ROOT / "data" / "latest_signal.json",
    ]

    if strategy_key == "f":
        signal_paths = [
            PROJECT_ROOT / "data" / "signals" / f"f_signal_{signal_date}.json",
            PROJECT_ROOT / "data" / "signals" / "latest_f_signal.json",
            *signal_paths,
        ]

    for signal_file in signal_paths:
        if signal_file.exists():
            return signal_file
    return None


def normalize_signal(sig: dict) -> dict | None:
    action = str(sig.get("action", "")).upper().strip()
    if action == "STOP_LOSS":
        action = "SELL"
    if action in ("", "HOLD"):
        return None

    code = str(sig.get("ts_code", "")).strip()
    if not code:
        return None

    try:
        shares = int(float(sig.get("target_shares", sig.get("shares", 0)) or 0))
    except (TypeError, ValueError):
        shares = 0

    try:
        price = float(sig.get("current_price", sig.get("price", 0)) or 0.0)
    except (TypeError, ValueError):
        price = 0.0

    return {
        "action": action,
        "ts_code": code,
        "name": sig.get("name", code),
        "shares": shares,
        "price": price,
        "reason": str(sig.get("reason", "")),
    }


def apply_signal(
    data: dict,
    signal_date: str,
    signal_path: str | Path | None = None,
    portfolio_path: str | Path | None = None,
    strategy: str = "",
):
    """读取信号结果, 按信号内容更新持仓"""
    resolved_signal_path = resolve_signal_path(signal_date, signal_path=signal_path, strategy=strategy)
    signal_data = None
    if resolved_signal_path:
        signal_data = json.loads(resolved_signal_path.read_text("utf-8"))
        print(f"[信号] 读取: {resolved_signal_path}")

    if not signal_data:
        print("[错误] 未找到信号文件")
        print("  请使用 --signal-file 指定, 或用 --buy/--sell 手动录入")
        return

    normalized_signals = []
    for raw_signal in signal_data.get("signals", []):
        normalized = normalize_signal(raw_signal)
        if normalized:
            normalized_signals.append(normalized)

    is_rebalance_day = bool(signal_data.get("is_rebalance_day"))
    if not is_rebalance_day:
        is_rebalance_day = any(sig["action"] == "BUY" for sig in normalized_signals)

    if not normalized_signals:
        print("[信号] 文件中无可执行交易指令")
        if is_rebalance_day:
            data["last_rebalance_date"] = signal_date
            data["rebalance_count"] = data.get("rebalance_count", 0) + 1
            print(f"[调仓] 无成交，但已记录调仓日: {signal_date}")
            save_portfolio(data, portfolio_path=portfolio_path, strategy=strategy)
        return

    executed = 0
    for sig in normalized_signals:
        action = sig.get("action", "")
        code = sig.get("ts_code", "")
        name = sig.get("name", code)
        shares = sig.get("shares", 0)
        price = sig.get("price", 0.0)

        if action == "BUY" and shares > 0 and price > 0:
            exec_price = price * 1.002  # 含滑点
            cost = shares * exec_price
            commission = max(cost * 0.0003, 5)
            total_cost = cost + commission
            if total_cost > data["cash"]:
                print(f"  跳过买入 {name}({code})，现金不足: 需要 ¥{total_cost:,.0f}, 可用 ¥{data['cash']:,.0f}")
                continue
            data["cash"] -= (cost + commission)
            if code in data["positions"]:
                pos = data["positions"][code]
                old_shares = pos.get("shares", 0)
                new_shares = old_shares + shares
                if new_shares <= 0:
                    continue
                pos["cost_price"] = round(
                    (pos.get("cost_price", 0.0) * old_shares + exec_price * shares) / new_shares,
                    4,
                )
                pos["shares"] = new_shares
                pos["peak_price"] = max(pos.get("peak_price", 0.0), price)
                pos["current_price"] = price
                pos["name"] = name
            else:
                data["positions"][code] = {
                    "shares": shares,
                    "cost_price": round(exec_price, 4),
                    "entry_date": signal_date,
                    "peak_price": price,
                    "current_price": price,
                    "name": name,
                }
            data["trade_history"].append({
                "date": signal_date, "action": "BUY", "ts_code": code,
                "name": name, "shares": shares, "price": round(exec_price, 4),
            })
            print(f"  买入 {name}({code}) {shares}股 × ¥{exec_price:.2f} = ¥{cost:,.0f}")
            executed += 1

        elif action == "SELL" and code in data["positions"] and price > 0:
            pos = data["positions"][code]
            sell_shares = pos["shares"]
            proceeds = sell_shares * price
            commission = max(proceeds * 0.0003, 5)
            stamp_tax = proceeds * 0.001
            data["cash"] += proceeds - commission - stamp_tax
            data["trade_history"].append({
                "date": signal_date, "action": "SELL", "ts_code": code,
                "name": name, "shares": sell_shares, "price": price,
            })
            del data["positions"][code]
            print(f"  卖出 {name}({code}) {sell_shares}股 × ¥{price:.2f} = ¥{proceeds:,.0f}")
            executed += 1
        elif action == "SELL":
            print(f"  跳过卖出 {name}({code})，当前未持有或价格无效")

    if is_rebalance_day:
        data["last_rebalance_date"] = signal_date
        data["rebalance_count"] = data.get("rebalance_count", 0) + 1

    print(f"\n[完成] 执行 {executed} 笔交易, 现金余额: ¥{data['cash']:,.2f}")
    save_portfolio(data, portfolio_path=portfolio_path, strategy=strategy)


def manual_buy(
    data: dict,
    code: str,
    shares: int,
    price: float,
    name: str = "",
    portfolio_path: str | Path | None = None,
    strategy: str = "",
):
    exec_price = price * 1.002
    cost = shares * exec_price
    commission = max(cost * 0.0003, 5)
    total_cost = cost + commission

    if total_cost > data["cash"]:
        print(f"[错误] 现金不足: 需要 ¥{total_cost:,.0f}, 可用 ¥{data['cash']:,.0f}")
        return

    data["cash"] -= total_cost
    today = datetime.now().strftime("%Y%m%d")
    if code in data["positions"]:
        pos = data["positions"][code]
        old_shares = pos.get("shares", 0)
        new_shares = old_shares + shares
        pos["cost_price"] = round(
            (pos.get("cost_price", 0.0) * old_shares + exec_price * shares) / new_shares,
            4,
        )
        pos["shares"] = new_shares
        pos["peak_price"] = max(pos.get("peak_price", 0.0), price)
        pos["current_price"] = price
        pos["name"] = name or code
    else:
        data["positions"][code] = {
            "shares": shares,
            "cost_price": round(exec_price, 4),
            "entry_date": today,
            "peak_price": price,
            "current_price": price,
            "name": name or code,
        }
    data["trade_history"].append({
        "date": today, "action": "BUY", "ts_code": code,
        "name": name or code, "shares": shares, "price": round(exec_price, 4),
    })
    print(f"  买入 {name or code}({code}) {shares}股 × ¥{exec_price:.2f} = ¥{cost:,.0f}")
    save_portfolio(data, portfolio_path=portfolio_path, strategy=strategy)


def manual_sell(
    data: dict,
    code: str,
    price: float,
    portfolio_path: str | Path | None = None,
    strategy: str = "",
):
    if code not in data["positions"]:
        print(f"[错误] 未持有 {code}")
        return

    pos = data["positions"][code]
    shares = pos["shares"]
    proceeds = shares * price
    commission = max(proceeds * 0.0003, 5)
    stamp_tax = proceeds * 0.001
    data["cash"] += proceeds - commission - stamp_tax

    today = datetime.now().strftime("%Y%m%d")
    data["trade_history"].append({
        "date": today, "action": "SELL", "ts_code": code,
        "name": pos.get("name", code), "shares": shares, "price": price,
    })
    pnl = (price / pos["cost_price"] - 1) * 100 if pos["cost_price"] > 0 else 0
    print(f"  卖出 {pos.get('name', code)}({code}) {shares}股 × ¥{price:.2f} = ¥{proceeds:,.0f} ({pnl:+.2f}%)")
    del data["positions"][code]
    save_portfolio(data, portfolio_path=portfolio_path, strategy=strategy)


def import_csv(
    data: dict,
    csv_path: str,
    portfolio_path: str | Path | None = None,
    strategy: str = "",
):
    """从CSV导入交易记录 (格式: action,ts_code,name,shares,price)"""
    import csv
    p = Path(csv_path)
    if not p.exists():
        print(f"[错误] 文件不存在: {csv_path}")
        return

    count = 0
    with open(p, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            action = row.get("action", "").upper()
            code = row.get("ts_code", "")
            name = row.get("name", code)
            shares = int(row.get("shares", 0))
            price = float(row.get("price", 0))

            if action == "BUY" and shares > 0 and price > 0:
                manual_buy(data, code, shares, price, name, portfolio_path=portfolio_path, strategy=strategy)
                count += 1
            elif action == "SELL" and price > 0:
                manual_sell(data, code, price, portfolio_path=portfolio_path, strategy=strategy)
                count += 1

    print(f"\n[完成] 导入 {count} 笔交易")


def reset_portfolio(portfolio_path: str | Path | None = None, strategy: str = ""):
    data = default_portfolio()
    save_portfolio(data, portfolio_path=portfolio_path, strategy=strategy)
    print("[重置] 持仓已清空")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="交易确认工具")
    parser.add_argument("--show", action="store_true", help="显示当前持仓")
    parser.add_argument("--apply-signal", type=str, metavar="DATE", help="按信号执行 (YYYYMMDD)")
    parser.add_argument("--signal-file", type=str, help="显式指定信号文件路径")
    parser.add_argument("--strategy", type=str, default="", help="策略命名空间，例如 f")
    parser.add_argument("--portfolio-path", type=str, help="显式指定持仓状态文件路径")
    parser.add_argument("--buy", type=str, metavar="CODE", help="手动买入 (需 --shares, --price)")
    parser.add_argument("--sell", type=str, metavar="CODE", help="手动卖出 (需 --price)")
    parser.add_argument("--shares", type=int, help="买入股数")
    parser.add_argument("--price", type=float, help="成交价格")
    parser.add_argument("--name", type=str, default="", help="股票名称")
    parser.add_argument("--import", dest="import_csv", type=str, metavar="CSV", help="从CSV导入")
    parser.add_argument("--reset", action="store_true", help="重置持仓")
    parser.add_argument("--set-rebalance-date", type=str, metavar="DATE", help="设置上次调仓日期")
    args = parser.parse_args()

    portfolio_path = resolve_portfolio_path(args.portfolio_path, strategy=args.strategy)
    data = load_portfolio(portfolio_path=portfolio_path, strategy=args.strategy)

    if args.reset:
        reset_portfolio(portfolio_path=portfolio_path, strategy=args.strategy)
    elif args.show:
        show_portfolio(data, portfolio_path=portfolio_path, strategy=args.strategy)
    elif args.apply_signal:
        apply_signal(
            data,
            args.apply_signal,
            signal_path=args.signal_file,
            portfolio_path=portfolio_path,
            strategy=args.strategy,
        )
    elif args.buy:
        if not args.shares or not args.price:
            print("[错误] --buy 需要同时指定 --shares 和 --price")
        else:
            manual_buy(
                data,
                args.buy,
                args.shares,
                args.price,
                args.name,
                portfolio_path=portfolio_path,
                strategy=args.strategy,
            )
    elif args.sell:
        if not args.price:
            print("[错误] --sell 需要指定 --price")
        else:
            manual_sell(data, args.sell, args.price, portfolio_path=portfolio_path, strategy=args.strategy)
    elif args.import_csv:
        import_csv(data, args.import_csv, portfolio_path=portfolio_path, strategy=args.strategy)
    elif args.set_rebalance_date:
        data["last_rebalance_date"] = args.set_rebalance_date
        save_portfolio(data, portfolio_path=portfolio_path, strategy=args.strategy)
        print(f"[设置] 上次调仓日期: {args.set_rebalance_date}")
    else:
        show_portfolio(data, portfolio_path=portfolio_path, strategy=args.strategy)
