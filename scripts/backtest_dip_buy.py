#!/usr/bin/env python3
"""
回测对比: 回撤买入 vs 开盘买入

实盘复盘: 周一建仓日10只股票7只高开(均+1.12%), 8只当日收跌(均-3.46%)
如果不追开盘, 而是挂限价单等回撤, 能否降低建仓成本?

用日线OHLC模拟限价单:
  如果当日 low <= 限价, 视为成交, 成交价 = 限价
  如果当日 low > 限价, 未成交, 延续到下一天
  超过 deadline 天数未成交, 按当日收盘价市价买入 (放弃等待)

测试模式:
  A) open_buy     — 基准: 开盘价买入 (现有逻辑)
  B) close_buy    — 信号日收盘价挂限价单 (不追高开)
  C) dip_1pct     — 信号日收盘价 × 0.99 挂限价单 (等1%回撤)
  D) dip_2pct     — 信号日收盘价 × 0.98 挂限价单 (等2%回撤)
  E) dip_3pct     — 信号日收盘价 × 0.97 挂限价单 (等3%回撤)
  F) vwap_approx  — 用 (high+low+close)/3 近似VWAP买入

每种模式的 deadline = 3天 (超时按收盘价市价买入, 避免永远接不住)
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import signal as _builtin_signal
_saved_signal = sys.modules.pop("signal")
sys.modules.pop("signal.signal_generator", None)
from signal.signal_generator import SignalConfig, SignalGenerator
sys.modules["_builtin_signal"] = _saved_signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_v2_vs_v3 import load_all_data, BacktestEngine, Position, DailyRecord


class DipBuyBacktestEngine(BacktestEngine):
    """回撤买入回测引擎"""

    def __init__(self, data: dict, buy_mode: str = "open_buy",
                 dip_pct: float = 0.0, deadline_days: int = 3,
                 initial_capital: float = 1_000_000):
        super().__init__("v3.1", data, initial_capital)
        self.buy_mode = buy_mode
        self.dip_pct = dip_pct           # 回撤幅度 (0.01 = 1%)
        self.deadline_days = deadline_days
        # 挂单簿: {ts_code: {limit_price, signal_date, signal_idx, name, score, deadline_idx}}
        self.pending_orders = {}
        self.fill_stats = {"filled_at_limit": 0, "filled_at_market": 0, "total": 0}

    def _get_ohlc(self, ts_code: str, date: str) -> dict:
        """获取OHLC"""
        df = self.data["stock_data"].get(ts_code)
        if df is None:
            return None
        row = df[df["trade_date"] == date]
        if row.empty:
            return None
        r = row.iloc[0]
        return {
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
        }

    def _try_fill_pending(self, date: str, date_idx: int) -> int:
        """尝试成交挂单"""
        trade_count = 0
        cfg = self.config
        total_value = self._total_value(date)

        filled_codes = []
        for code, order in list(self.pending_orders.items()):
            ohlc = self._get_ohlc(code, date)
            if ohlc is None:
                continue

            filled = False
            fill_price = 0
            fill_reason = ""

            # 检查是否超过deadline
            past_deadline = (date_idx - order["signal_idx"]) >= self.deadline_days

            if past_deadline:
                # 超时: 按收盘价市价买入
                fill_price = ohlc["close"]
                filled = True
                fill_reason = f"超时市价买入(第{date_idx - order['signal_idx']}天)"
                self.fill_stats["filled_at_market"] += 1
            elif ohlc["low"] <= order["limit_price"]:
                # 限价单触发
                fill_price = order["limit_price"]
                filled = True
                fill_reason = f"限价单成交({date_idx - order['signal_idx']}天后)"
                self.fill_stats["filled_at_limit"] += 1

            if filled and fill_price > 0:
                exec_price = fill_price * (1 + cfg.slippage_pct)
                single_weight = cfg.max_total_position / cfg.top_n
                target_value = min(total_value * single_weight, self.cash * 0.95)

                current_pos_value = sum(
                    p.shares * self._get_price(p.ts_code, date)
                    for p in self.positions.values()
                )
                max_pos = cfg.max_total_position
                if total_value > 0 and (current_pos_value + target_value) / total_value > max_pos:
                    target_value = max(0, total_value * max_pos - current_pos_value)

                shares = int(target_value / exec_price / 100) * 100
                if shares >= 100 and shares * exec_price >= cfg.min_position_amount:
                    cost = shares * exec_price
                    commission = max(cost * cfg.commission_rate, 5)
                    self.cash -= (cost + commission)
                    self.positions[code] = Position(
                        ts_code=code,
                        name=order["name"],
                        shares=shares,
                        cost_price=exec_price,
                        entry_date=date,
                        peak_price=ohlc["high"],
                    )
                    trade_count += 1
                    self.trade_log.append({
                        "date": date, "action": "BUY", "code": code,
                        "name": order["name"], "shares": shares,
                        "price": exec_price,
                        "limit": order["limit_price"],
                        "reason": fill_reason,
                    })
                    self.fill_stats["total"] += 1
                filled_codes.append(code)

        for code in filled_codes:
            del self.pending_orders[code]

        return trade_count

    def _execute_rebalance(self, date: str, scores: list[dict], regime: str):
        """执行调仓 (回撤买入版)"""
        cfg = self.config
        total_value = self._total_value(date)
        trade_count = 0

        if regime in ("HALT", "WAIT"):
            max_position = 0.0
        elif regime == "DEFENSIVE":
            max_position = 0.5 * cfg.max_total_position
        else:
            max_position = cfg.max_total_position

        # HALT/WAIT清仓 + 清空挂单
        if regime in ("HALT", "WAIT"):
            self.pending_orders.clear()
            reason = "HALT清仓" if regime == "HALT" else "WAIT空仓"
            for code in list(self.positions.keys()):
                pos = self.positions[code]
                sell_price = self._get_open_price(code, date)
                proceeds = pos.shares * sell_price
                commission = max(proceeds * cfg.commission_rate, 5)
                stamp_tax = proceeds * cfg.stamp_tax_rate
                self.cash += proceeds - commission - stamp_tax
                self.trade_log.append({
                    "date": date, "action": "SELL", "code": code,
                    "name": pos.name, "shares": pos.shares, "price": sell_price,
                    "reason": reason,
                })
                del self.positions[code]
                trade_count += 1
            return trade_count

        buffer_codes = [s["ts_code"] for s in scores[:int(cfg.top_n * cfg.hold_buffer_ratio)]]

        # 卖出
        for code in list(self.positions.keys()):
            if code not in buffer_codes:
                pos = self.positions[code]
                sell_price = self._get_open_price(code, date)
                proceeds = pos.shares * sell_price
                commission = max(proceeds * cfg.commission_rate, 5)
                stamp_tax = proceeds * cfg.stamp_tax_rate
                self.cash += proceeds - commission - stamp_tax
                self.trade_log.append({
                    "date": date, "action": "SELL", "code": code,
                    "name": pos.name, "shares": pos.shares, "price": sell_price,
                    "reason": "出缓冲带",
                })
                del self.positions[code]
                trade_count += 1

        # 清空旧挂单 (新信号覆盖)
        self.pending_orders.clear()

        # 买入 / 挂单
        hold_count = len(self.positions)
        buy_slots = cfg.top_n - hold_count

        dates = self.data["trade_dates"]
        date_idx = dates.index(date) if date in dates else 0

        scan_range = min(len(scores), cfg.top_n * 3)
        for s in scores[:scan_range]:
            code = s["ts_code"]
            if code in self.positions:
                continue
            if buy_slots <= 0:
                break

            ref_close = s["close"]  # 信号日收盘价
            if ref_close <= 0:
                continue

            if self.buy_mode == "open_buy":
                # 原始逻辑: 开盘价买入
                buy_price = self._get_open_price(code, date)
                if buy_price <= 0:
                    continue
                # 缺口保护
                if cfg.open_gap_limit > 0:
                    gap = buy_price / ref_close - 1
                    if gap > cfg.open_gap_limit:
                        continue

                exec_price = buy_price * (1 + cfg.slippage_pct)
                single_weight = max_position / cfg.top_n
                target_value = min(total_value * single_weight, self.cash * 0.95)
                current_pos_value = sum(
                    p.shares * self._get_price(p.ts_code, date)
                    for p in self.positions.values()
                )
                if total_value > 0 and (current_pos_value + target_value) / total_value > max_position:
                    target_value = max(0, total_value * max_position - current_pos_value)

                shares = int(target_value / exec_price / 100) * 100
                if shares >= 100 and shares * exec_price >= cfg.min_position_amount:
                    cost = shares * exec_price
                    commission = max(cost * cfg.commission_rate, 5)
                    self.cash -= (cost + commission)
                    self.positions[code] = Position(
                        ts_code=code, name=s["name"], shares=shares,
                        cost_price=exec_price, entry_date=date,
                        peak_price=buy_price,
                    )
                    buy_slots -= 1
                    trade_count += 1
                    self.trade_log.append({
                        "date": date, "action": "BUY", "code": code,
                        "name": s["name"], "shares": shares, "price": exec_price,
                        "reason": f"排名#{scores.index(s)+1} 开盘买入",
                    })
                    self.fill_stats["total"] += 1

            elif self.buy_mode == "vwap":
                # VWAP近似: (H+L+C)/3
                ohlc = self._get_ohlc(code, date)
                if ohlc is None:
                    continue
                vwap = (ohlc["high"] + ohlc["low"] + ohlc["close"]) / 3
                exec_price = vwap * (1 + cfg.slippage_pct)
                single_weight = max_position / cfg.top_n
                target_value = min(total_value * single_weight, self.cash * 0.95)
                current_pos_value = sum(
                    p.shares * self._get_price(p.ts_code, date)
                    for p in self.positions.values()
                )
                if total_value > 0 and (current_pos_value + target_value) / total_value > max_position:
                    target_value = max(0, total_value * max_position - current_pos_value)

                shares = int(target_value / exec_price / 100) * 100
                if shares >= 100 and shares * exec_price >= cfg.min_position_amount:
                    cost = shares * exec_price
                    commission = max(cost * cfg.commission_rate, 5)
                    self.cash -= (cost + commission)
                    self.positions[code] = Position(
                        ts_code=code, name=s["name"], shares=shares,
                        cost_price=exec_price, entry_date=date,
                        peak_price=ohlc["high"],
                    )
                    buy_slots -= 1
                    trade_count += 1
                    self.trade_log.append({
                        "date": date, "action": "BUY", "code": code,
                        "name": s["name"], "shares": shares, "price": exec_price,
                        "reason": f"排名#{scores.index(s)+1} VWAP买入",
                    })
                    self.fill_stats["total"] += 1

            else:
                # 限价单模式: 挂单等回撤
                limit_price = ref_close * (1 - self.dip_pct)
                ohlc = self._get_ohlc(code, date)

                if ohlc and ohlc["low"] <= limit_price:
                    # 当日就能成交
                    exec_price = limit_price * (1 + cfg.slippage_pct)
                    single_weight = max_position / cfg.top_n
                    target_value = min(total_value * single_weight, self.cash * 0.95)
                    current_pos_value = sum(
                        p.shares * self._get_price(p.ts_code, date)
                        for p in self.positions.values()
                    )
                    if total_value > 0 and (current_pos_value + target_value) / total_value > max_position:
                        target_value = max(0, total_value * max_position - current_pos_value)

                    shares = int(target_value / exec_price / 100) * 100
                    if shares >= 100 and shares * exec_price >= cfg.min_position_amount:
                        cost = shares * exec_price
                        commission = max(cost * cfg.commission_rate, 5)
                        self.cash -= (cost + commission)
                        self.positions[code] = Position(
                            ts_code=code, name=s["name"], shares=shares,
                            cost_price=exec_price, entry_date=date,
                            peak_price=ohlc["high"],
                        )
                        trade_count += 1
                        self.trade_log.append({
                            "date": date, "action": "BUY", "code": code,
                            "name": s["name"], "shares": shares, "price": exec_price,
                            "reason": f"排名#{scores.index(s)+1} 限价当日成交",
                        })
                        self.fill_stats["filled_at_limit"] += 1
                        self.fill_stats["total"] += 1
                else:
                    # 挂单等待
                    self.pending_orders[code] = {
                        "limit_price": limit_price,
                        "signal_date": date,
                        "signal_idx": date_idx,
                        "name": s["name"],
                        "score": s["score"],
                        "deadline_idx": date_idx + self.deadline_days,
                    }

                buy_slots -= 1

        return trade_count

    def run(self, start_date: str, end_date: str):
        """运行回测 (含挂单成交检查)"""
        dates = self.data["trade_dates"]
        bt_dates = [d for d in dates if start_date <= d <= end_date]

        mode_label = {
            "open_buy": "开盘买入",
            "vwap": "VWAP买入",
            "limit": f"限价回撤{self.dip_pct:.0%}",
        }.get(self.buy_mode, self.buy_mode)
        print(f"\n[回测] v3.1 ({mode_label}): {bt_dates[0]} ~ {bt_dates[-1]}, {len(bt_dates)} 交易日")

        for i, date in enumerate(bt_dates):
            regime = self._check_regime(date)
            trade_count = 0

            # 止损
            trade_count += self._check_stop_loss(date)

            # 检查挂单成交
            if self.pending_orders:
                trade_count += self._try_fill_pending(date, dates.index(date))

            # HALT
            if regime in ("HALT", "WAIT") and self.positions:
                scores = self._score_stocks(date, regime)
                trade_count += self._execute_rebalance(date, scores, regime)
                self._halt_liquidated = True
                self.rebalance_dates.append(date)
            elif regime not in ("HALT", "WAIT"):
                halt_recovery = getattr(self, "_halt_liquidated", False) and not self.positions
                days_since = i - self.last_rebalance_idx
                if (halt_recovery
                        or days_since >= self.config.rebalance_interval_days
                        or self.last_rebalance_idx < 0):
                    scores = self._score_stocks(date, regime)
                    trade_count += self._execute_rebalance(date, scores, regime)
                    self.last_rebalance_idx = i
                    self.rebalance_dates.append(date)
                    if halt_recovery:
                        self._halt_liquidated = False

            for pos in self.positions.values():
                p = self._get_price(pos.ts_code, date)
                pos.peak_price = max(pos.peak_price, p)

            total = self._total_value(date)
            pos_value = total - self.cash
            self.daily_records.append(DailyRecord(
                date=date, total_value=total, cash=self.cash,
                position_value=pos_value,
                position_ratio=pos_value / total if total > 0 else 0,
                regime=regime, num_positions=len(self.positions),
                trade_count=trade_count,
            ))

        stats = self.fill_stats
        print(f"[回测] 完成: 终值=¥{self.daily_records[-1].total_value:,.0f}, "
              f"调仓{len(self.rebalance_dates)}次, 交易{len(self.trade_log)}笔")
        if stats["total"] > 0:
            limit_pct = stats["filled_at_limit"] / stats["total"] * 100 if stats["total"] > 0 else 0
            market_pct = stats["filled_at_market"] / stats["total"] * 100 if stats["total"] > 0 else 0
            print(f"[回测] 成交统计: 限价成交{stats['filled_at_limit']}笔({limit_pct:.0f}%) "
                  f"超时市价{stats['filled_at_market']}笔({market_pct:.0f}%)")


def main():
    csv_lhb = PROJECT_ROOT / "data" / "csi1000_market_bundle_300d_lhb.csv"
    csv_300d = PROJECT_ROOT / "data" / "csi1000_market_bundle_300d.csv"
    csv_path = str(csv_lhb) if csv_lhb.exists() else str(csv_300d)
    data = load_all_data(csv_path)

    start_date = "20250206"
    end_date = "20260324"
    bt_dates = [d for d in data["trade_dates"] if start_date <= d <= end_date]

    # 测试配置
    configs = [
        ("A_开盘买入",     "open_buy", 0.00, 3),
        ("B_收盘限价",     "limit",    0.00, 3),   # 信号日收盘价挂单
        ("C_回撤1%",      "limit",    0.01, 3),
        ("D_回撤2%",      "limit",    0.02, 3),
        ("E_回撤3%",      "limit",    0.03, 3),
        ("F_VWAP买入",    "vwap",     0.00, 3),
    ]

    engines = {}
    metrics = {}

    for label, mode, dip, deadline in configs:
        bt = DipBuyBacktestEngine(data, buy_mode=mode, dip_pct=dip, deadline_days=deadline)
        bt.run(start_date, end_date)
        engines[label] = bt
        metrics[label] = bt.calc_metrics()

    # 基准
    idx = data["index_data"]
    idx_bt = idx[(idx["trade_date"] >= start_date) & (idx["trade_date"] <= end_date)]
    idx_return = float(idx_bt.iloc[-1]["close"]) / float(idx_bt.iloc[0]["close"]) - 1 if not idx_bt.empty else 0

    n_days = len(bt_dates)
    print("\n" + "=" * 110)
    print("  回测报告: 回撤买入策略对比 (v3.1 龙虎榜增强)")
    print(f"  区间: {start_date} ~ {end_date}  |  {n_days}个交易日  |  初始资金: ¥1,000,000")
    print(f"  规则: 限价单未成交超过3天 → 按收盘价市价买入")
    print("=" * 110)

    labels = [c[0] for c in configs]

    print(f"\n{'指标':<20s}", end="")
    for lb in labels:
        print(f" {lb:>14s}", end="")
    print(f" {'基准':>10s}")
    print("─" * (20 + 14 * len(labels) + 12))

    rows = [
        ("累计收益率", "total_return", ".2%"),
        ("年化收益率", "annual_return", ".2%"),
        ("年化波动率", "annual_vol", ".2%"),
        ("夏普比率", "sharpe", ".2f"),
        ("最大回撤", "max_drawdown", ".2%"),
        ("日胜率", "daily_win_rate", ".1%"),
        ("总交易笔数", "total_trades", "d"),
        ("调仓次数", "rebalance_count", "d"),
    ]

    for name, key, fmt in rows:
        print(f"{name:<18s}", end="")
        for lb in labels:
            val = metrics[lb][key]
            if fmt == "d":
                print(f" {val:>14d}", end="")
            else:
                print(f" {val:>14{fmt}}", end="")
        if key == "total_return":
            print(f" {idx_return:>10.2%}")
        else:
            print(f" {'—':>10s}")

    # 期末净值
    print(f"{'期末净值':<18s}", end="")
    for lb in labels:
        print(f" ¥{metrics[lb]['final_value']:>12,.0f}", end="")
    print(f" {'—':>10s}")

    # 成交统计
    print(f"\n{'─'*90}")
    print(f"  限价单成交统计")
    print(f"{'─'*90}")
    for lb in labels:
        bt = engines[lb]
        s = bt.fill_stats
        if s["total"] > 0:
            limit_r = s["filled_at_limit"] / s["total"] * 100
            market_r = s["filled_at_market"] / s["total"] * 100
            print(f"  {lb:<14s} 总{s['total']}笔  限价成交{s['filled_at_limit']}笔({limit_r:.0f}%)  超时市价{s['filled_at_market']}笔({market_r:.0f}%)")
        else:
            print(f"  {lb:<14s} 总{len(bt.trade_log)}笔 (直接买入)")

    # 净值采样
    print(f"\n{'─'*110}")
    print("  每日净值曲线 (每10天)")
    print(f"{'─'*110}")
    print(f"  {'日期':<12s}", end="")
    for lb in labels:
        short = lb.split("_")[1] if "_" in lb else lb
        print(f" {short:>10s}", end="")
    print(f" {'基准':>10s}")

    idx_dict = dict(zip(idx_bt["trade_date"], idx_bt["close"]))
    idx_base = float(idx_bt.iloc[0]["close"]) if not idx_bt.empty else 1
    all_records = {lb: engines[lb].daily_records for lb in labels}
    cap = 1_000_000

    for i in range(len(all_records[labels[0]])):
        if i % 10 != 0 and i != len(all_records[labels[0]]) - 1:
            continue
        d = all_records[labels[0]][i].date
        print(f"  {d:<12s}", end="")
        for lb in labels:
            nav = all_records[lb][i].total_value / cap
            print(f" {nav:>9.4f}", end="")
        idx_nav = float(idx_dict.get(d, idx_base)) / idx_base
        print(f"  {idx_nav:>9.4f}")

    # 结论
    best = max(metrics.items(), key=lambda x: x[1]["sharpe"])
    print(f"\n{'='*90}")
    print(f"  结论")
    print(f"{'='*90}")
    for lb in labels:
        m = metrics[lb]
        print(f"  {lb:<14s}: 年化{m['annual_return']:>7.2%}  夏普{m['sharpe']:>5.2f}  回撤{m['max_drawdown']:>7.2%}")
    print(f"\n  最优: {best[0]}  夏普={best[1]['sharpe']:.2f}  年化={best[1]['annual_return']:.2%}")

    # 保存
    result = {"backtest_period": f"{start_date}~{end_date}", "trading_days": n_days}
    for lb in labels:
        result[lb] = metrics[lb]
        result[lb]["fill_stats"] = engines[lb].fill_stats
    out_path = PROJECT_ROOT / "data" / "backtest_dip_buy.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    print(f"\n[保存] {out_path}")


if __name__ == "__main__":
    main()
