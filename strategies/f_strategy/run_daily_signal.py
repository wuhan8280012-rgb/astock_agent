#!/usr/bin/env python3
"""Daily executable signal generator for F + strength_transition_coef >= -0.1."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.confirm_trades import load_portfolio, save_portfolio, show_portfolio  # noqa: E402
from scripts.confirm_trades import apply_signal as confirm_signal  # noqa: E402
from strategies.f_strategy.live_data import extend_dataset_with_live_data  # noqa: E402
from strategies.f_strategy.run_backtest import (  # noqa: E402
    DATASETS,
    enrich_industry_strength20_features,
    load_dataset,
    load_local_trend_index_df,
    make_filtered_backtest,
)


BACKTEST_SCRIPT = PROJECT_ROOT / "scripts" / "backtest_strategies.py"
NOTIFIER_SCRIPT = PROJECT_ROOT / "signal" / "notifier.py"
SIGNAL_DIR = PROJECT_ROOT / "data" / "signals"
NOTIFY_CONFIG_PATH = PROJECT_ROOT / "config" / "notify_config.json"
F_PORTFOLIO_PATH = PROJECT_ROOT / "data" / "f_strategy_portfolio_state.json"
LATEST_F_SIGNAL_PATH = SIGNAL_DIR / "latest_f_signal.json"
LOCAL_TREND_INDEX_PATH = PROJECT_ROOT / "data" / "market_index_000001sh_5y.csv"


def load_backtest_module():
    spec = importlib.util.spec_from_file_location("backtest_strategies", BACKTEST_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_notifier_module():
    spec = importlib.util.spec_from_file_location("project_signal_notifier", NOTIFIER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_notifier_module = load_notifier_module()
NotifyConfig = _notifier_module.NotifyConfig
Notifier = _notifier_module.Notifier


def init_portfolio():
    data = {
        "initial_capital": 1_000_000.0,
        "cash": 1_000_000.0,
        "positions": {},
        "last_rebalance_date": "",
        "rebalance_count": 0,
        "trade_history": [],
        "updated_at": datetime.now().isoformat(),
    }
    save_portfolio(data, portfolio_path=F_PORTFOLIO_PATH)
    print(f"[初始化] F 策略持仓已初始化: {F_PORTFOLIO_PATH}")


def load_notify_config() -> NotifyConfig:
    cfg = NotifyConfig()
    if NOTIFY_CONFIG_PATH.exists():
        cfg = NotifyConfig.from_json(NOTIFY_CONFIG_PATH)
    env_cfg = NotifyConfig.from_env()
    if env_cfg.wecom_webhook:
        cfg.wecom_enabled = True
        cfg.wecom_webhook = env_cfg.wecom_webhook
    return cfg


def render_signal_text(result: dict) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("  F 策略 · 每日执行信号")
    lines.append(f"  日期: {result['trade_date']}")
    lines.append("=" * 60)
    lines.append(f"市场状态: {result['trend_state']} | 最大仓位: {result['max_position_pct']:.0%}")
    lines.append(f"账户总值: ¥{result['portfolio_value']:,.2f} | 现金: ¥{result['cash']:,.2f}")
    lines.append(
        f"调仓状态: {'调仓日' if result['is_rebalance_day'] else '非调仓日'} | 距上次调仓: {result['days_since_rebalance']} 个交易日"
    )
    lines.append("")
    lines.append("TOP 15 候选:")
    for i, item in enumerate(result["top_candidates"][:15], start=1):
        lines.append(
            f"  {i:>2}. {item['name']:<10} {item['ts_code']} "
            f"rank={item['rank']:>2} price=¥{item['price']:.2f}"
        )

    lines.append("")
    if result["holding_rank_report"]:
        lines.append("持仓排名日报:")
        for item in result["holding_rank_report"]:
            delta = item["delta_text"]
            lines.append(
                f"  - {item['name']}({item['ts_code']}) 今日#{item['rank_today']} | {delta}"
            )
        lines.append("")

    if result["signals"]:
        lines.append("执行建议:")
        for sig in result["signals"]:
            detail = f"{sig['action']} {sig['name']}({sig['ts_code']})"
            if sig.get("shares", 0):
                detail += f" {sig['shares']}股"
            if sig.get("price", 0):
                detail += f" @ ¥{sig['price']:.4f}"
            detail += f" | {sig['reason']}"
            lines.append(f"  - {detail}")
    else:
        lines.append("执行建议:")
        lines.append("  - 今日无操作，继续持有。")

    lines.append("=" * 60)
    lines.append("确认执行后可用: python3 strategies/f_strategy/run_daily_signal.py --confirm YYYYMMDD")
    lines.append("=" * 60)
    return "\n".join(lines)


def normalize_execution_signals(result: dict) -> dict:
    normalized_signals = []
    base_portfolio_value = max(float(result.get("portfolio_value", 0.0)), 1.0)

    for sig in result.get("signals", []):
        shares = int(sig.get("shares", 0) or 0)
        price = float(sig.get("price", 0.0) or 0.0)
        target_amount = round(shares * price, 4)
        enriched = {
            **sig,
            "current_price": price,
            "target_shares": shares,
            "target_amount": target_amount,
            "target_weight": round(target_amount / base_portfolio_value, 6) if sig.get("action") == "BUY" else 0.0,
            "momentum_score": float(sig.get("momentum_score", 0.0) or 0.0),
            "rank": int(sig.get("rank", 0) or 0),
            "urgency": "HIGH" if "STOP_LOSS" in str(sig.get("reason", "")) else "NORMAL",
        }
        normalized_signals.append(enriched)

    return {
        **result,
        "signal_namespace": "f_strategy",
        "signal_schema": "trade_signal_v1",
        "portfolio_path": str(F_PORTFOLIO_PATH),
        "execution_timing": "signal_close",
        "signals": normalized_signals,
    }


def build_holding_rank_report(trade_date: str, trade_dates: list[str], positions: dict, rank_map_today: dict, name_map: dict, bt) -> list[dict]:
    if not positions:
        return []
    trade_idx = trade_dates.index(trade_date)
    prev_date = trade_dates[trade_idx - 1] if trade_idx > 0 else None
    prev_rank_map = {}
    if prev_date:
        prev_scores = bt._score_universe(prev_date)
        prev_rank_map = {code: i + 1 for i, (code, _) in enumerate(prev_scores)}

    report = []
    for code, pos in positions.items():
        rank_today = int(rank_map_today.get(code, 999))
        rank_prev = prev_rank_map.get(code)
        if rank_prev is None:
            delta = None
            delta_text = "前一日无排名"
        else:
            delta = rank_today - int(rank_prev)
            if delta > 0:
                delta_text = f"较前一日落后 {delta} 名"
            elif delta < 0:
                delta_text = f"较前一日提升 {abs(delta)} 名"
            else:
                delta_text = "较前一日持平"
        report.append(
            {
                "ts_code": code,
                "name": pos.get("name", name_map.get(code, code)),
                "rank_today": rank_today,
                "rank_prev": int(rank_prev) if rank_prev is not None else None,
                "delta": delta,
                "delta_text": delta_text,
            }
        )
    report.sort(key=lambda x: x["rank_today"])
    return report


def render_rank_report_markdown(result: dict) -> str:
    lines = []
    lines.append(f"日期: `{result['trade_date']}`")
    lines.append(f"市场状态: `{result['trend_state']}`  仓位上限: `{result['max_position_pct']:.0%}`")
    lines.append("")
    if result["holding_rank_report"]:
        lines.append("持仓股排名变化:")
        for item in result["holding_rank_report"]:
            lines.append(
                f"- `{item['name']} {item['ts_code']}` 今日 `#{item['rank_today']}`，{item['delta_text']}"
            )
    else:
        lines.append("当前空仓，无持仓排名变化。")
    if result["signals"]:
        lines.append("")
        lines.append("今日执行建议:")
        for sig in result["signals"]:
            lines.append(
                f"- `{sig['action']} {sig['name']} {sig['ts_code']}` {sig.get('shares', 0)}股，{sig['reason']}"
            )
    return "\n".join(lines)


def render_manual_execution_markdown(result: dict) -> str:
    signals = result.get("signals", [])
    sell_signals = [sig for sig in signals if sig.get("action") == "SELL"]
    buy_signals = [sig for sig in signals if sig.get("action") == "BUY"]

    lines = []
    lines.append(
        f"> 日期: `{result['trade_date']}`  市场: `{result['trend_state']}`  仓位上限: `{result['max_position_pct']:.0%}`"
    )
    lines.append(
        f"> 账户总值: `¥{result['portfolio_value']:,.0f}`  现金: `¥{result['cash']:,.0f}`"
    )
    lines.append(
        f"> 调仓状态: `{'调仓日' if result['is_rebalance_day'] else '非调仓日'}`  距上次调仓: `{result['days_since_rebalance']}` 个交易日"
    )

    if sell_signals:
        lines.append("")
        lines.append("**卖出指令**")
        for sig in sell_signals:
            lines.append(
                f"- `{sig['name']} {sig['ts_code']}` 卖出 `{sig.get('shares', 0)}股` @ `¥{sig.get('price', 0):.4f}`"
                f"  原因: `{sig.get('reason', '')}`"
            )

    if buy_signals:
        lines.append("")
        lines.append("**买入指令**")
        for sig in buy_signals:
            lines.append(
                f"- `{sig['name']} {sig['ts_code']}` 买入 `{sig.get('shares', 0)}股` @ `¥{sig.get('price', 0):.4f}`"
                f"  原因: `{sig.get('reason', '')}`"
            )

    if not sell_signals and not buy_signals:
        lines.append("")
        lines.append("**今日结论**")
        lines.append("- 无需下单，继续持有。")

    lines.append("")
    lines.append("**执行步骤**")
    lines.append("- 在券商 APP 手动下单，优先执行卖出。")
    lines.append(f"- 成交后执行 `python3 strategies/f_strategy/run_daily_signal.py --confirm {result['trade_date']}`")
    return "\n".join(lines)


def send_manual_execution_notification(result: dict):
    cfg = load_notify_config()
    if not (cfg.wecom_enabled and cfg.wecom_webhook):
        return
    notifier = Notifier(cfg)
    title = f"F策略手动执行提醒 {result['trade_date']} [{result['trend_state']}]"
    content = render_manual_execution_markdown(result)
    notifier.send(title, content)


def send_test_notification():
    cfg = load_notify_config()
    if not (cfg.wecom_enabled and cfg.wecom_webhook):
        raise ValueError("未配置企业微信 webhook，请先设置 WECOM_WEBHOOK 或 notify_config.json")
    notifier = Notifier(cfg)
    notifier.send(
        "F策略企业微信测试",
        "\n".join(
            [
                "> 这是一条 F 策略手动执行提醒测试消息。",
                "",
                "**预期流程**",
                "- 每个交易日收盘后自动生成信号",
                "- 企业微信收到买卖清单",
                "- 你手动下单",
                "- 成交后执行 `python3 strategies/f_strategy/run_daily_signal.py --confirm YYYYMMDD`",
            ]
        ),
    )


def resolve_default_trade_date(trade_dates: list[str], allow_stale_data: bool) -> str:
    latest_trade_date = trade_dates[-1]
    staleness_days = (datetime.now().date() - datetime.strptime(latest_trade_date, "%Y%m%d").date()).days
    if not allow_stale_data and staleness_days > 3:
        raise ValueError(
            f"样本最新交易日为 {latest_trade_date}，距离今天已 {staleness_days} 天。"
            "实盘模式请先更新数据，或显式传入 --date / --allow-stale-data。"
        )
    return latest_trade_date


def load_runtime_dataset(module, requested_trade_date: str | None, allow_stale_data: bool, disable_live_update: bool) -> tuple:
    daily, idx, basic, trade_dates = load_dataset(
        DATASETS["csi1000_5y"],
        module,
        trend_index_loader=lambda dates, _: load_local_trend_index_df(LOCAL_TREND_INDEX_PATH, dates),
    )
    live_update = {
        "enabled": False,
        "used_live_data": False,
        "last_local_trade_date": trade_dates[-1],
        "requested_trade_date": requested_trade_date or trade_dates[-1],
        "latest_trade_date": trade_dates[-1],
        "fetched_trade_dates": [],
        "message": "未启用在线补数。",
    }
    if not disable_live_update:
        daily, idx, basic, trade_dates, live_update = extend_dataset_with_live_data(
            daily,
            idx,
            basic,
            trade_dates,
            enrich_industry_strength20_features=enrich_industry_strength20_features,
            requested_trade_date=requested_trade_date,
        )

    if requested_trade_date:
        trade_date = str(requested_trade_date).strip()
        if trade_date not in trade_dates:
            raise ValueError(
                f"trade_date {trade_date} 不在可用数据集中，当前最新可用交易日为 {trade_dates[-1]}"
            )
    else:
        trade_date = resolve_default_trade_date(trade_dates, allow_stale_data=allow_stale_data)

    return daily, idx, basic, trade_dates, trade_date, live_update


def generate_signal(trade_date: str, daily: pd.DataFrame, idx: pd.DataFrame, basic: pd.DataFrame, trade_dates: list[str], live_update: dict | None = None) -> dict:
    if trade_date not in trade_dates:
        raise ValueError(f"trade_date {trade_date} 不在数据集中")

    module = load_backtest_module()
    cfg = [s for s in module.get_strategies() if s.name == "F_三因子+趋势过滤"][0]
    backtest_cls = make_filtered_backtest(module, min_transition_coef=-0.1)
    bt = backtest_cls(cfg, daily, idx, basic, trade_dates)

    portfolio = load_portfolio(portfolio_path=F_PORTFOLIO_PATH)
    positions = portfolio.get("positions", {})
    cash = float(portfolio.get("cash", 0.0))

    prices = bt._get_prices(trade_date)
    for code, pos in positions.items():
        if code in prices:
            pos["current_price"] = float(prices[code])

    portfolio_value = cash + sum(
        pos.get("shares", 0) * pos.get("current_price", pos.get("cost_price", 0))
        for pos in positions.values()
    )

    trend_state = bt._get_trend_state(trade_date)
    max_position_pct = bt._get_trend_position(trade_date)
    scores = bt._score_universe(trade_date)
    target_codes = [code for code, _ in scores[: cfg.top_n]]
    buffer_codes = {code for code, _ in scores[: int(cfg.top_n * cfg.hold_buffer_ratio)]}
    rank_map = {code: i + 1 for i, (code, _) in enumerate(scores)}
    name_map = {code: str(row.get("name", code)) for code, row in bt._basic_map.items()}
    holding_rank_report = build_holding_rank_report(trade_date, trade_dates, positions, rank_map, name_map, bt)

    last_rebalance_date = portfolio.get("last_rebalance_date", "")
    days_since_rebalance = 999
    if last_rebalance_date and last_rebalance_date in trade_dates:
        days_since_rebalance = trade_dates.index(trade_date) - trade_dates.index(last_rebalance_date)
    is_rebalance_day = days_since_rebalance >= cfg.rebalance_interval or last_rebalance_date == ""

    simulated_cash = cash
    simulated_positions = {
        code: {
            **pos,
            "current_price": pos.get("current_price", pos.get("cost_price", 0)),
        }
        for code, pos in positions.items()
    }
    signals = []

    # Daily stop-loss
    for code in list(simulated_positions.keys()):
        pos = simulated_positions[code]
        price = pos.get("current_price", pos.get("cost_price", 0))
        if pos.get("cost_price", 0) > 0:
            pnl = price / pos["cost_price"] - 1
            if pnl <= cfg.stop_loss_pct:
                proceeds = pos["shares"] * price * (1 - cfg.slippage)
                simulated_cash += proceeds - proceeds * (cfg.commission + cfg.stamp_tax)
                signals.append(
                    {
                        "action": "SELL",
                        "ts_code": code,
                        "name": pos.get("name", name_map.get(code, code)),
                        "shares": int(pos["shares"]),
                        "price": round(price * (1 - cfg.slippage), 4),
                        "reason": f"STOP_LOSS({pnl * 100:+.2f}%)",
                    }
                )
                del simulated_positions[code]

    if is_rebalance_day:
        for code in list(simulated_positions.keys()):
            if code not in buffer_codes:
                price = simulated_positions[code].get("current_price", simulated_positions[code]["cost_price"])
                proceeds = simulated_positions[code]["shares"] * price * (1 - cfg.slippage)
                simulated_cash += proceeds - proceeds * (cfg.commission + cfg.stamp_tax)
                signals.append(
                    {
                        "action": "SELL",
                        "ts_code": code,
                        "name": simulated_positions[code].get("name", name_map.get(code, code)),
                        "shares": int(simulated_positions[code]["shares"]),
                        "price": round(price * (1 - cfg.slippage), 4),
                        "reason": f"OUT_OF_BUFFER(rank={rank_map.get(code, 999)})",
                    }
                )
                del simulated_positions[code]

        portfolio_value_now = simulated_cash + sum(
            pos["shares"] * pos.get("current_price", pos.get("cost_price", 0))
            for pos in simulated_positions.values()
        )
        current_position_value = sum(
            pos["shares"] * pos.get("current_price", pos.get("cost_price", 0))
            for pos in simulated_positions.values()
        )
        max_equity = portfolio_value_now * max_position_pct
        available_for_equity = max_equity - current_position_value
        available_cash = min(simulated_cash, available_for_equity) if available_for_equity > 0 else 0.0
        buy_slots = cfg.top_n - len(simulated_positions)

        for code in target_codes:
            if buy_slots <= 0 or available_cash < 10000:
                break
            if code in simulated_positions or code not in prices or prices[code] <= 0:
                continue
            buy_price = float(prices[code]) * (1 + cfg.slippage)
            target_amount = min(portfolio_value_now * cfg.max_single_weight, available_cash * 0.95)
            shares = int(target_amount / buy_price / 100) * 100
            if shares < 100:
                continue
            gross = shares * buy_price
            cost = gross * cfg.commission
            simulated_cash -= gross + cost
            available_cash -= gross + cost
            signals.append(
                {
                    "action": "BUY",
                    "ts_code": code,
                    "name": name_map.get(code, code),
                    "shares": int(shares),
                    "price": round(buy_price, 4),
                    "reason": f"TOP_RANK(rank={rank_map.get(code, 999)})",
                }
            )
            simulated_positions[code] = {
                "shares": shares,
                "cost_price": buy_price,
                "current_price": float(prices[code]),
                "entry_date": trade_date,
                "name": name_map.get(code, code),
            }
            buy_slots -= 1

    top_candidates = []
    for code, rank in scores[:15]:
        top_candidates.append(
            {
                "ts_code": code,
                "name": name_map.get(code, code),
                "rank": int(rank_map[code]),
                "price": float(prices.get(code, 0.0)),
            }
        )

    result = {
        "strategy": "F + strength_transition_coef >= -0.1",
        "trade_date": trade_date,
        "trend_state": trend_state,
        "max_position_pct": max_position_pct,
        "portfolio_value": round(portfolio_value, 2),
        "cash": round(cash, 2),
        "days_since_rebalance": int(days_since_rebalance if days_since_rebalance != 999 else cfg.rebalance_interval),
        "is_rebalance_day": bool(is_rebalance_day),
        "signals": signals,
        "holding_rank_report": holding_rank_report,
        "top_candidates": top_candidates,
        "current_positions": {
            code: {
                "name": pos.get("name", name_map.get(code, code)),
                "shares": pos.get("shares", 0),
                "cost_price": round(pos.get("cost_price", 0.0), 4),
                "current_price": round(pos.get("current_price", pos.get("cost_price", 0.0)), 4),
                "rank": int(rank_map.get(code, 999)),
            }
            for code, pos in positions.items()
        },
        "generated_at": datetime.now().isoformat(),
        "live_update": live_update or {},
    }
    return normalize_execution_signals(result)


def main():
    parser = argparse.ArgumentParser(description="F 策略每日可执行信号")
    parser.add_argument("--date", type=str, help="指定交易日 YYYYMMDD，默认取样本最后一个交易日")
    parser.add_argument("--init", action="store_true", help="初始化 F 策略持仓文件")
    parser.add_argument("--status", action="store_true", help="显示当前持仓状态")
    parser.add_argument("--confirm", type=str, metavar="DATE", help="按 F 信号确认执行 (YYYYMMDD)")
    parser.add_argument("--test-notify", action="store_true", help="测试企业微信提醒")
    parser.add_argument("--allow-stale-data", action="store_true", help="允许使用超过 3 天未更新的样本")
    parser.add_argument("--disable-live-update", action="store_true", help="禁用在线补数，只使用本地样本")
    args = parser.parse_args()

    if args.init:
        init_portfolio()
        return

    if args.status:
        show_portfolio(load_portfolio(portfolio_path=F_PORTFOLIO_PATH), portfolio_path=F_PORTFOLIO_PATH)
        return

    if args.test_notify:
        send_test_notification()
        return

    if args.confirm:
        data = load_portfolio(portfolio_path=F_PORTFOLIO_PATH)
        confirm_signal(
            data,
            args.confirm,
            portfolio_path=F_PORTFOLIO_PATH,
            strategy="f",
        )
        show_portfolio(load_portfolio(portfolio_path=F_PORTFOLIO_PATH), portfolio_path=F_PORTFOLIO_PATH)
        return

    module = load_backtest_module()
    daily, idx, basic, trade_dates, trade_date, live_update = load_runtime_dataset(
        module,
        requested_trade_date=args.date,
        allow_stale_data=args.allow_stale_data,
        disable_live_update=args.disable_live_update,
    )
    if live_update.get("message"):
        print(f"[数据] {live_update['message']}")
    result = generate_signal(trade_date, daily=daily, idx=idx, basic=basic, trade_dates=trade_dates, live_update=live_update)
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SIGNAL_DIR / f"f_signal_{trade_date}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    LATEST_F_SIGNAL_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(render_signal_text(result))
    print(f"[信号] 已保存: {out_path}")
    print(f"[信号] 最新快照: {LATEST_F_SIGNAL_PATH}")
    send_manual_execution_notification(result)


if __name__ == "__main__":
    main()
