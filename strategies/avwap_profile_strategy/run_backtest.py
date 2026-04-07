#!/usr/bin/env python3
"""Weekly AVWAP + Volume Profile + Sentiment Breakout/Failure backtest."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKTEST_SCRIPT = PROJECT_ROOT / "scripts" / "backtest_strategies.py"
F_STRATEGY_SCRIPT = PROJECT_ROOT / "strategies" / "f_strategy" / "run_backtest.py"

DEFAULT_TREND_INDEX_CODE = "000001.SH"
DATASETS = {
    "csi1000_5y": PROJECT_ROOT / "data_exports" / "tushare_20210329_20260327_csi1000_5y" / "csi1000_market_bundle_5y.csv",
    "csi1000_5y_pit": PROJECT_ROOT / "data_exports" / "tushare_20210329_20260327_csi1000_5y_pit" / "csi1000_market_bundle_5y_pit.csv",
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_dataset_with_flow_signals(data_path: Path, bt_module, trend_index_code: str = DEFAULT_TREND_INDEX_CODE):
    f_module = load_module(F_STRATEGY_SCRIPT, "f_strategy_run_backtest")
    daily, idx, basic, trade_dates = f_module.load_dataset(
        data_path,
        bt_module,
        trend_index_code=trend_index_code,
    )

    extra_cols = [
        "data_type",
        "ts_code",
        "trade_date",
        "smart_money_intensity",
        "net_mf_intensity",
    ]
    extra = pd.read_csv(data_path, usecols=extra_cols, low_memory=False)
    extra = extra[extra["data_type"] == "daily"][["ts_code", "trade_date", "smart_money_intensity", "net_mf_intensity"]].copy()
    extra["trade_date"] = extra["trade_date"].astype(str)
    for col in ["smart_money_intensity", "net_mf_intensity"]:
        extra[col] = pd.to_numeric(extra[col], errors="coerce")

    daily = daily.merge(extra, on=["ts_code", "trade_date"], how="left")
    return daily, idx, basic, trade_dates


def anchored_vwap(window: pd.DataFrame) -> float:
    weights = pd.to_numeric(window["amount"], errors="coerce").fillna(0.0).astype(float).values
    prices = pd.to_numeric(window["close"], errors="coerce").astype(float).values
    total = float(weights.sum())
    if total <= 0:
        return float(np.nanmean(prices)) if len(prices) else np.nan
    return float(np.nansum(prices * weights) / total)


def fixed_range_volume_profile(window: pd.DataFrame, bins: int = 24) -> dict | None:
    if window.empty:
        return None

    lows = pd.to_numeric(window["low"], errors="coerce")
    highs = pd.to_numeric(window["high"], errors="coerce")
    closes = pd.to_numeric(window["close"], errors="coerce")
    weights = pd.to_numeric(window["amount"], errors="coerce").fillna(0.0)

    price_min = float(lows.min())
    price_max = float(highs.max())
    if not np.isfinite(price_min) or not np.isfinite(price_max) or price_max <= price_min:
        return None

    edges = np.linspace(price_min, price_max, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    bucket = np.clip(np.digitize(closes.fillna(price_min).values, edges) - 1, 0, bins - 1)
    volume_by_bin = np.bincount(bucket, weights=weights.values, minlength=bins)
    total_volume = float(volume_by_bin.sum())
    if total_volume <= 0:
        return None

    poc_idx = int(np.argmax(volume_by_bin))
    ranked = np.argsort(volume_by_bin)[::-1]
    selected = []
    acc = 0.0
    for idx in ranked:
        selected.append(int(idx))
        acc += float(volume_by_bin[idx])
        if acc >= total_volume * 0.7:
            break
    selected.sort()
    return {
        "poc": float(centers[poc_idx]),
        "val": float(centers[selected[0]]),
        "vah": float(centers[selected[-1]]),
        "price_min": price_min,
        "price_max": price_max,
    }


def period_key(series: pd.Series, cycle: str) -> pd.Series:
    dt = pd.to_datetime(series.astype(str))
    if cycle == "monthly":
        return dt.dt.strftime("%Y-%m")
    return dt.dt.strftime("%Y-%W")


def build_rebalance_dates(trade_dates: list[str], cycle: str) -> set[str]:
    df = pd.DataFrame({"trade_date": trade_dates})
    if cycle == "biweekly":
        df["period_key"] = period_key(df["trade_date"], "weekly")
        weekly_last = df.groupby("period_key", sort=False)["trade_date"].last().tolist()
        return set(weekly_last[::2])
    df["period_key"] = period_key(df["trade_date"], cycle)
    return set(df.groupby("period_key", sort=False)["trade_date"].last().tolist())


def calc_ma_angle_deg(series: pd.Series, ma_window: int, scale: float = 0.02) -> float:
    ma = series.rolling(ma_window).mean()
    if len(ma.dropna()) < 2:
        return np.nan
    current = float(ma.iloc[-1])
    prev = float(ma.iloc[-2])
    if not np.isfinite(current) or not np.isfinite(prev) or prev <= 0:
        return np.nan
    slope = (current / prev - 1.0) / scale
    return float(np.degrees(np.arctan(slope)))


def clip_score(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(np.clip(value, low, high))


def get_strategy_config(bt_module, holding_cycle: str, **overrides):
    interval = 20 if holding_cycle == "monthly" else (10 if holding_cycle == "biweekly" else 5)
    cfg_kwargs = dict(
        name=f"AVWAPProfile_{holding_cycle}",
        top_n=6,
        hold_buffer_ratio=1.20,
        max_single_weight=0.18,
        rebalance_interval=interval,
        stop_loss_pct=-0.12,
        use_halt=False,
        use_trend_filter=False,
        commission=0.0003,
        stamp_tax=0.001,
        slippage=0.003,
        min_amount_20d=3e8,
        min_price=5.0,
        min_list_days=250,
    )
    cfg_kwargs.update(overrides)
    return bt_module.BacktestConfig(**cfg_kwargs)


def market_filter_params(mode: str) -> tuple[int, float, float]:
    if mode == "ma60":
        return 60, 5, 0.0
    if mode == "ma120":
        return 120, 10, 0.0
    return 0, 0.0, -999.0


def make_avwap_profile_backtest(
    bt_module,
    holding_cycle: str,
    breakout_volume_mult: float = 1.5,
    pullback_volume_frac: float = 0.8,
    daily_failure_exit: bool = False,
    market_filter_mode: str = "off",
    *,
    balance_periods_override: int | None = None,
    recent_breakout_lookback_override: int | None = None,
    range_pct_max: float = 0.32,
    value_area_width_pct_max: float = 0.18,
    poc_distance_max: float = 0.05,
    angle_deg_min: float = 1.0,
    angle_deg_max: float = 20.0,
    market_filter_half_position: bool = False,
):
    cycle_for_data = "weekly" if holding_cycle == "biweekly" else holding_cycle
    ma_window = 4 if cycle_for_data == "monthly" else 8
    balance_periods = balance_periods_override if balance_periods_override is not None else (6 if cycle_for_data == "monthly" else 8)
    recent_breakout_lookback = recent_breakout_lookback_override if recent_breakout_lookback_override is not None else (2 if cycle_for_data == "monthly" else 4)
    market_ma_days, market_angle_span, market_angle_floor = market_filter_params(market_filter_mode)

    class AVWAPProfileBacktest(bt_module.Backtest):
        def __init__(self, cfg, daily, idx, basic, trade_dates):
            super().__init__(cfg, daily, idx, basic, trade_dates)
            self.holding_cycle = holding_cycle
            self._rebalance_dates = build_rebalance_dates(trade_dates, holding_cycle)
            self._cycle_for_data = cycle_for_data
            self._period_data = {
                code: self._build_period_frame(df.reset_index())
                for code, df in self._stock_data.items()
            }
            self._score_diag = []
            self._candidate_meta_by_date: dict[str, dict[str, dict]] = {}

        def _build_period_frame(self, stock_daily: pd.DataFrame) -> pd.DataFrame:
            df = stock_daily.copy()
            df["period_key"] = period_key(df["trade_date"], self._cycle_for_data)
            grouped = (
                df.groupby("period_key", sort=True)
                .agg(
                    start_date=("trade_date", "first"),
                    end_date=("trade_date", "last"),
                    open=("open", "first"),
                    high=("high", "max"),
                    low=("low", "min"),
                    close=("close", "last"),
                    amount=("amount", "sum"),
                    smart_money_intensity=("smart_money_intensity", "mean"),
                    net_mf_intensity=("net_mf_intensity", "mean"),
                )
                .reset_index(drop=True)
            )
            grouped.index = grouped["end_date"].astype(str)
            return grouped

        def _balance_context(self, hist: pd.DataFrame, period_df: pd.DataFrame, breakout_loc: int) -> dict | None:
            if breakout_loc < balance_periods:
                return None

            balance_period_df = period_df.iloc[breakout_loc - balance_periods:breakout_loc]
            if len(balance_period_df) < balance_periods:
                return None

            start_date = str(balance_period_df["start_date"].iloc[0])
            end_date = str(balance_period_df["end_date"].iloc[-1])
            balance_daily = hist[(hist.index >= start_date) & (hist.index <= end_date)].copy()
            if len(balance_daily) < balance_periods * 5 // 2:
                return None

            profile = fixed_range_volume_profile(balance_daily)
            if profile is None:
                return None

            avwap = anchored_vwap(balance_daily)
            if not np.isfinite(avwap) or avwap <= 0:
                return None

            balance_high = float(pd.to_numeric(balance_daily["high"], errors="coerce").max())
            balance_low = float(pd.to_numeric(balance_daily["low"], errors="coerce").min())
            range_pct = (balance_high - balance_low) / avwap if avwap > 0 else np.nan
            value_area_width_pct = (profile["vah"] - profile["val"]) / avwap if avwap > 0 else np.nan
            poc_distance = abs(profile["poc"] / avwap - 1.0) if avwap > 0 else np.nan
            if (
                not np.isfinite(range_pct)
                or not np.isfinite(value_area_width_pct)
                or not np.isfinite(poc_distance)
                or range_pct > range_pct_max
                or value_area_width_pct > value_area_width_pct_max
                or poc_distance > poc_distance_max
            ):
                return None

            breakout_level = max(balance_high, profile["vah"], profile["poc"], avwap)
            return {
                "start_date": start_date,
                "end_date": end_date,
                "avwap": avwap,
                "poc": profile["poc"],
                "vah": profile["vah"],
                "val": profile["val"],
                "balance_high": balance_high,
                "balance_low": balance_low,
                "breakout_level": breakout_level,
            }

        def _valid_breakout(self, hist: pd.DataFrame, period_df: pd.DataFrame, breakout_loc: int) -> dict | None:
            if breakout_loc <= 0:
                return None
            context = self._balance_context(hist, period_df, breakout_loc)
            if context is None:
                return None

            breakout_bar = period_df.iloc[breakout_loc]
            balance_period_df = period_df.iloc[breakout_loc - balance_periods:breakout_loc]
            prev_amount_mean = float(balance_period_df["amount"].mean()) * 1000.0
            breakout_amount = float(breakout_bar["amount"]) * 1000.0
            breakout_close = float(breakout_bar["close"])
            breakout_valid = (
                breakout_close > context["breakout_level"] * 1.01
                and breakout_amount > prev_amount_mean * breakout_volume_mult
            )
            if not breakout_valid:
                return None

            context.update(
                {
                    "breakout_end_date": str(breakout_bar["end_date"]),
                    "breakout_close": breakout_close,
                    "breakout_amount": breakout_amount,
                    "breakout_amount_ratio": breakout_amount / prev_amount_mean if prev_amount_mean > 0 else np.nan,
                }
            )
            return context

        def _latest_breakout_context(self, hist: pd.DataFrame, period_df: pd.DataFrame, current_loc: int) -> dict | None:
            start = max(balance_periods, current_loc - recent_breakout_lookback)
            for loc in range(current_loc, start - 1, -1):
                breakout = self._valid_breakout(hist, period_df, loc)
                if breakout is not None:
                    return breakout
            return None

        def _market_filter_pass(self, date: str) -> bool:
            if market_filter_mode == "off":
                return True
            hist = self._idx_series[self._idx_series.index <= date]
            if len(hist) < market_ma_days + int(market_angle_span):
                return False
            ma = hist.rolling(market_ma_days).mean()
            ma_now = float(ma.iloc[-1])
            ma_prev = float(ma.iloc[-1 - int(market_angle_span)])
            current = float(hist.iloc[-1])
            if not np.isfinite(ma_now) or not np.isfinite(ma_prev) or ma_now <= 0 or ma_prev <= 0:
                return False
            angle = calc_ma_angle_deg(hist, ma_window=market_ma_days, scale=0.01)
            return current > ma_now and angle > market_angle_floor and ma_now >= ma_prev

        def _should_rebalance(self, date: str) -> bool:
            return date in self._rebalance_dates

        def _append_diag(self, date: str, count: int) -> None:
            self._score_diag.append({"date": date, "candidate_count": int(count)})

        def _diagnostics_summary(self) -> dict:
            if not self._score_diag:
                return {
                    "candidate_count_mean": 0.0,
                    "candidate_count_median": 0.0,
                    "candidate_count_min": 0,
                    "candidate_count_max": 0,
                    "enough_candidates_ratio": 0.0,
                }
            counts = pd.Series([item["candidate_count"] for item in self._score_diag], dtype=float)
            return {
                "candidate_count_mean": round(float(counts.mean()), 2),
                "candidate_count_median": round(float(counts.median()), 2),
                "candidate_count_min": int(counts.min()),
                "candidate_count_max": int(counts.max()),
                "enough_candidates_ratio": round(float((counts >= self.cfg.top_n).mean()), 4),
            }

        def _score_universe(self, date: str) -> list:
            cfg = self.cfg
            candidates = []
            self._candidate_meta_by_date[date] = {}

            if date not in self.trade_dates:
                self._append_diag(date, 0)
                return []

            for code, data in self._stock_data.items():
                if date not in data.index:
                    continue
                row = data.loc[date]
                close = float(row["close"])
                if not np.isfinite(close) or close < cfg.min_price:
                    continue

                info = self._basic_map.get(code)
                if info is not None:
                    name = str(info.get("name", ""))
                    if "ST" in name.upper():
                        continue
                    list_date = str(info.get("list_date", ""))
                    if list_date and len(list_date) >= 8:
                        try:
                            ld = pd.Timestamp(list_date[:8])
                            cd = pd.Timestamp(date)
                            if (cd - ld).days < cfg.min_list_days:
                                continue
                        except Exception:
                            pass

                hist = data[data.index <= date]
                if len(hist) < max(100, balance_periods * 5 + 20):
                    continue
                avg_amount_20d = float(hist.tail(20)["amount"].mean()) * 1000.0
                if avg_amount_20d < cfg.min_amount_20d:
                    continue

                period_df = self._period_data.get(code)
                if period_df is None or date not in period_df.index:
                    continue
                period_hist = period_df[period_df.index <= date]
                if len(period_hist) < max(ma_window + 2, balance_periods + 2):
                    continue

                weekly_close = period_hist["close"].astype(float)
                angle_deg = calc_ma_angle_deg(weekly_close, ma_window=ma_window)
                if not np.isfinite(angle_deg) or angle_deg < angle_deg_min or angle_deg > angle_deg_max:
                    continue
                ma_now = weekly_close.rolling(ma_window).mean().iloc[-1]
                ma_prev3 = weekly_close.rolling(ma_window).mean().iloc[-3] if len(weekly_close) >= ma_window + 2 else np.nan
                if not np.isfinite(ma_now) or not np.isfinite(ma_prev3) or ma_now <= ma_prev3:
                    continue

                current_loc = len(period_hist) - 1
                current_bar = period_hist.iloc[-1]

                latest_breakout = self._latest_breakout_context(hist, period_hist, current_loc)
                if latest_breakout is None:
                    continue

                anchor = max(latest_breakout["poc"], latest_breakout["avwap"], latest_breakout["vah"])
                failure = close <= anchor * 0.995
                if failure:
                    continue

                current_amount = float(current_bar["amount"]) * 1000.0
                smart_sent = float(current_bar.get("smart_money_intensity", np.nan))
                net_sent = float(current_bar.get("net_mf_intensity", np.nan))
                if np.isnan(smart_sent):
                    smart_sent = float(pd.to_numeric(hist.tail(5)["smart_money_intensity"], errors="coerce").mean())
                if np.isnan(net_sent):
                    net_sent = float(pd.to_numeric(hist.tail(5)["net_mf_intensity"], errors="coerce").mean())
                sentiment_raw = float(np.nanmean([smart_sent, net_sent]))
                if not np.isfinite(sentiment_raw):
                    sentiment_raw = 0.0

                entry_type = None
                setup_score = 0.0
                volume_quality = 0.0
                proximity_quality = 0.0

                is_current_breakout = latest_breakout["breakout_end_date"] == date
                if is_current_breakout:
                    if sentiment_raw <= 0:
                        continue
                    entry_type = "breakout"
                    setup_score = clip_score((close / latest_breakout["breakout_level"] - 1.0) / 0.10)
                    volume_quality = clip_score(latest_breakout["breakout_amount_ratio"] / 2.0)
                    proximity_quality = clip_score(1.0 - min(abs(close / latest_breakout["breakout_level"] - 1.03), 0.10) / 0.10)
                else:
                    pullback_ok = (
                        close > anchor * 1.005
                        and close <= latest_breakout["breakout_close"] * 1.08
                        and float(current_bar["low"]) <= latest_breakout["breakout_level"] * 1.03
                        and current_amount < latest_breakout["breakout_amount"] * pullback_volume_frac
                    )
                    if pullback_ok and sentiment_raw > -0.01:
                        entry_type = "pullback"
                        setup_score = clip_score(1.0 - min(abs(close / latest_breakout["breakout_level"] - 1.02), 0.08) / 0.08)
                        volume_quality = clip_score(latest_breakout["breakout_amount"] / max(current_amount, 1.0) / 2.0)
                        proximity_quality = clip_score(1.0 - min(abs(close / anchor - 1.01), 0.08) / 0.08)
                    else:
                        trend_hold = (
                            close > latest_breakout["breakout_level"] * 1.04
                            and close > anchor * 1.02
                            and sentiment_raw > -0.02
                        )
                        if not trend_hold:
                            continue
                        entry_type = "hold"
                        setup_score = clip_score((close / latest_breakout["breakout_level"] - 1.0) / 0.15)
                        volume_quality = clip_score(sentiment_raw + 0.5)
                        proximity_quality = clip_score(1.0 - min(abs(close / anchor - 1.06), 0.12) / 0.12)

                angle_quality = clip_score(1.0 - min(abs(angle_deg - 8.0), 8.0) / 8.0)
                sentiment_quality = clip_score((sentiment_raw + 0.10) / 0.20)
                entry_bonus = {"pullback": 1.0, "breakout": 0.8, "hold": 0.5}[entry_type]

                candidates.append(
                    (
                        code,
                        entry_bonus,
                        angle_quality,
                        volume_quality,
                        sentiment_quality,
                        proximity_quality,
                        setup_score,
                    )
                )
                self._candidate_meta_by_date[date][code] = {
                    "entry_type": entry_type,
                    "anchor": anchor,
                    "failure_level": anchor * 0.995,
                    "breakout_level": latest_breakout["breakout_level"],
                    "breakout_end_date": latest_breakout["breakout_end_date"],
                    "breakout_amount": latest_breakout["breakout_amount"],
                    "breakout_amount_ratio": latest_breakout["breakout_amount_ratio"],
                    "sentiment_raw": sentiment_raw,
                    "angle_deg": angle_deg,
                }

            if not candidates:
                self._append_diag(date, 0)
                return []

            df = pd.DataFrame(
                candidates,
                columns=[
                    "code",
                    "entry_bonus",
                    "angle_quality",
                    "volume_quality",
                    "sentiment_quality",
                    "proximity_quality",
                    "setup_score",
                ],
            )
            for col in [
                "entry_bonus",
                "angle_quality",
                "volume_quality",
                "sentiment_quality",
                "proximity_quality",
                "setup_score",
            ]:
                df[f"{col}_rank"] = df[col].rank(ascending=False)

            df["composite_rank"] = (
                0.25 * df["entry_bonus_rank"]
                + 0.20 * df["angle_quality_rank"]
                + 0.20 * df["volume_quality_rank"]
                + 0.15 * df["sentiment_quality_rank"]
                + 0.10 * df["proximity_quality_rank"]
                + 0.10 * df["setup_score_rank"]
            )
            df = df.sort_values("composite_rank").reset_index(drop=True)
            self._append_diag(date, len(df))
            return list(zip(df["code"], df["composite_rank"]))

        def run(self, start_offset: int = 250, include_daily: bool = False, end_date: str | None = None) -> dict:
            cfg = self.cfg
            dates = self.trade_dates
            if start_offset >= len(dates):
                return {"error": "数据不足"}
            if end_date is not None and end_date not in dates:
                return {"error": f"end_date 不在交易日历中: {end_date}"}
            end_idx = dates.index(end_date) if end_date is not None else len(dates) - 1
            if end_idx < start_offset:
                return {"error": "回测结束日早于起始偏移"}

            start_date = dates[start_offset]
            final_date = dates[end_idx]
            capital = 1_000_000.0
            cash = capital
            positions = {}
            nav_history = []
            trade_count = 0
            rebalance_count = 0
            self._score_diag = []

            for i in range(start_offset, end_idx + 1):
                date = dates[i]
                prices = self._get_prices(date)

                if cfg.stop_loss_pct > -0.99:
                    for code in list(positions.keys()):
                        if code not in prices:
                            continue
                        pnl = prices[code] / positions[code]["cost_price"] - 1.0
                        if pnl <= cfg.stop_loss_pct:
                            sell_price = prices[code] * (1.0 - cfg.slippage)
                            proceeds = positions[code]["shares"] * sell_price
                            cost = proceeds * (cfg.commission + cfg.stamp_tax)
                            cash += proceeds - cost
                            trade_count += 1
                            del positions[code]

                if daily_failure_exit:
                    for code in list(positions.keys()):
                        if code not in prices:
                            continue
                        failure_level = float(positions[code].get("failure_level", np.nan))
                        if np.isfinite(failure_level) and prices[code] <= failure_level:
                            sell_price = prices[code] * (1.0 - cfg.slippage)
                            proceeds = positions[code]["shares"] * sell_price
                            cost = proceeds * (cfg.commission + cfg.stamp_tax)
                            cash += proceeds - cost
                            trade_count += 1
                            del positions[code]

                should_rebalance = self._should_rebalance(date)
                if should_rebalance:
                    scores = self._score_universe(date)
                    market_ok = self._market_filter_pass(date)
                    if market_ok:
                        target_count = min(cfg.top_n, len(scores))
                    elif market_filter_half_position:
                        target_count = min(max(1, cfg.top_n // 2), len(scores))
                    else:
                        target_count = 0
                    target_codes = [item[0] for item in scores[:target_count]]
                    buffer_count = min(
                        len(scores),
                        max(target_count, int(np.ceil(target_count * cfg.hold_buffer_ratio))),
                    )
                    buffer_codes = set(item[0] for item in scores[:buffer_count])

                    for code in list(positions.keys()):
                        if code not in buffer_codes:
                            if code in prices:
                                sell_price = prices[code] * (1.0 - cfg.slippage)
                                proceeds = positions[code]["shares"] * sell_price
                                cost = proceeds * (cfg.commission + cfg.stamp_tax)
                                cash += proceeds - cost
                                trade_count += 1
                            del positions[code]

                    if target_count > 0:
                        portfolio_value_now = cash + sum(
                            pos["shares"] * prices.get(code, pos.get("current_price", pos["cost_price"]))
                            for code, pos in positions.items()
                        )
                        available_cash = cash
                        buy_slots = target_count - len(positions)
                        target_weight = min(cfg.max_single_weight, 0.95 / target_count)
                        for code in target_codes:
                            if buy_slots <= 0 or available_cash < 10000:
                                break
                            if code in positions or code not in prices or prices[code] <= 0:
                                continue
                            buy_price = prices[code] * (1.0 + cfg.slippage)
                            target_amount = min(portfolio_value_now * target_weight, available_cash * 0.95)
                            shares = int(target_amount / buy_price / 100) * 100
                            if shares < 100:
                                continue
                            amount = shares * buy_price
                            cost = amount * cfg.commission
                            cash -= amount + cost
                            available_cash -= amount + cost
                            positions[code] = {
                                "shares": shares,
                                "cost_price": buy_price,
                                "entry_date": date,
                                "current_price": prices[code],
                            }
                            positions[code].update(self._candidate_meta_by_date.get(date, {}).get(code, {}))
                            trade_count += 1
                            buy_slots -= 1
                    rebalance_count += 1

                final_value = cash
                position_value = 0.0
                for code, pos in positions.items():
                    px = prices.get(code, pos.get("current_price", pos["cost_price"]))
                    pos["current_price"] = px
                    market_value = pos["shares"] * px
                    final_value += market_value
                    position_value += market_value

                nav_history.append(
                    {
                        "date": date,
                        "nav": final_value,
                        "cash": cash,
                        "position_value": position_value,
                        "position_pct": position_value / final_value if final_value > 0 else 0.0,
                        "max_position_pct": 1.0,
                        "trend_state": self.holding_cycle.upper(),
                        "idx_close": float(self._idx_series.get(date, np.nan)),
                    }
                )

            result = self._build_result(
                nav_history=nav_history,
                capital=capital,
                start_date=start_date,
                end_date=final_date,
                trade_count=trade_count,
                rebalance_count=rebalance_count,
                include_daily=include_daily,
            )
            result["holding_cycle"] = self.holding_cycle
            result.update(self._diagnostics_summary())
            result["breakout_volume_mult"] = breakout_volume_mult
            result["pullback_volume_frac"] = pullback_volume_frac
            result["daily_failure_exit"] = daily_failure_exit
            result["market_filter_mode"] = market_filter_mode
            result["balance_periods"] = balance_periods
            result["recent_breakout_lookback"] = recent_breakout_lookback
            result["range_pct_max"] = range_pct_max
            result["value_area_width_pct_max"] = value_area_width_pct_max
            result["poc_distance_max"] = poc_distance_max
            result["angle_deg_min"] = angle_deg_min
            result["angle_deg_max"] = angle_deg_max
            result["market_filter_half_position"] = market_filter_half_position
            return result

    return AVWAPProfileBacktest


def main():
    parser = argparse.ArgumentParser(description="Weekly/Monthly AVWAP profile breakout backtest")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="csi1000_5y_pit")
    parser.add_argument("--holding-cycle", choices=["weekly", "biweekly", "monthly"], default="weekly")
    parser.add_argument("--recent-days", type=int, default=100)
    parser.add_argument("--trend-index-code", type=str, default=DEFAULT_TREND_INDEX_CODE)
    parser.add_argument("--breakout-volume-mult", type=float, default=1.5)
    parser.add_argument("--pullback-volume-frac", type=float, default=0.8)
    parser.add_argument("--daily-failure-exit", action="store_true")
    parser.add_argument("--market-filter-mode", choices=["off", "ma60", "ma120"], default="off")
    # Candidate pool tuning
    parser.add_argument("--balance-periods", type=int, default=None, help="Override balance periods (default: auto by cycle)")
    parser.add_argument("--recent-breakout-lookback", type=int, default=None, help="Override breakout lookback (default: auto by cycle)")
    parser.add_argument("--range-pct-max", type=float, default=0.32, help="Max balance range amplitude")
    parser.add_argument("--value-area-width-pct-max", type=float, default=0.18, help="Max value area width")
    parser.add_argument("--poc-distance-max", type=float, default=0.05, help="Max POC-AVWAP distance")
    parser.add_argument("--angle-deg-min", type=float, default=1.0, help="Min MA angle degrees")
    parser.add_argument("--angle-deg-max", type=float, default=20.0, help="Max MA angle degrees")
    # Market filter tuning
    parser.add_argument("--market-filter-half-position", action="store_true", help="Use half position when market filter fails")
    # Portfolio construction
    parser.add_argument("--top-n", type=int, default=6, help="Number of holdings")
    parser.add_argument("--hold-buffer-ratio", type=float, default=1.20)
    parser.add_argument("--max-single-weight", type=float, default=0.18)
    parser.add_argument("--stop-loss-pct", type=float, default=-0.12)
    parser.add_argument("--min-amount-20d", type=float, default=3e8)
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--min-list-days", type=int, default=250)
    args = parser.parse_args()

    bt_module = load_module(BACKTEST_SCRIPT, "backtest_strategies")
    daily, idx, basic, trade_dates = load_dataset_with_flow_signals(
        DATASETS[args.dataset],
        bt_module,
        trend_index_code=args.trend_index_code,
    )

    cfg = get_strategy_config(
        bt_module,
        args.holding_cycle,
        top_n=args.top_n,
        hold_buffer_ratio=args.hold_buffer_ratio,
        max_single_weight=args.max_single_weight,
        stop_loss_pct=args.stop_loss_pct,
        min_amount_20d=args.min_amount_20d,
        min_price=args.min_price,
        min_list_days=args.min_list_days,
    )
    strategy_cls = make_avwap_profile_backtest(
        bt_module,
        args.holding_cycle,
        breakout_volume_mult=args.breakout_volume_mult,
        pullback_volume_frac=args.pullback_volume_frac,
        daily_failure_exit=args.daily_failure_exit,
        market_filter_mode=args.market_filter_mode,
        balance_periods_override=args.balance_periods,
        recent_breakout_lookback_override=args.recent_breakout_lookback,
        range_pct_max=args.range_pct_max,
        value_area_width_pct_max=args.value_area_width_pct_max,
        poc_distance_max=args.poc_distance_max,
        angle_deg_min=args.angle_deg_min,
        angle_deg_max=args.angle_deg_max,
        market_filter_half_position=args.market_filter_half_position,
    )
    bt = strategy_cls(cfg, daily, idx, basic, trade_dates)

    start_offset = max(0, len(trade_dates) - args.recent_days)
    start_date = trade_dates[start_offset]
    t0 = time.time()
    result = bt.run(start_offset=start_offset)
    result["elapsed_sec"] = round(time.time() - t0, 1)
    result["recent_days"] = args.recent_days
    result["recent_start_date"] = start_date
    result["trend_filter_index_code"] = args.trend_index_code

    suffix = f"{args.dataset}_{args.holding_cycle}_{args.recent_days}d_bv{args.breakout_volume_mult:g}_pv{args.pullback_volume_frac:g}_{args.market_filter_mode}"
    if args.daily_failure_exit:
        suffix += "_dailyexit"
    out_path = PROJECT_ROOT / "backtest" / f"strategy_avwap_profile_{suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "data_file": str(DATASETS[args.dataset]),
        "result": result,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"WROTE {out_path}")


if __name__ == "__main__":
    main()
