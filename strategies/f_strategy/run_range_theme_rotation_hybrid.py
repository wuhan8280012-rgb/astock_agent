#!/usr/bin/env python3
"""Hybrid strategy: current F in BULL, dedicated theme-rotation strategy in RANGE, cash in BEAR."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_backtest import (
    DATASETS,
    DEFAULT_TREND_INDEX_CODE,
    calc_strength_transition_coef,
    load_backtest_module,
    load_dataset,
    make_filtered_backtest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = PROJECT_ROOT / "backtest" / "strategy_f_range_theme_rotation_hybrid_csi1000_5y.json"


def build_features(daily: pd.DataFrame, basic: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy()
    df = df.merge(basic[["ts_code", "name", "list_date"]], on="ts_code", how="left")
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    g = df.groupby("ts_code", group_keys=False)
    df["ret5"] = g["close"].pct_change(5)
    df["ret10"] = g["close"].pct_change(10)
    df["ret20"] = g["close"].pct_change(20)
    df["ret60"] = g["close"].pct_change(60)
    df["ma5"] = g["close"].transform(lambda s: s.rolling(5).mean())
    df["ma10"] = g["close"].transform(lambda s: s.rolling(10).mean())
    df["ma20"] = g["close"].transform(lambda s: s.rolling(20).mean())
    df["ma60"] = g["close"].transform(lambda s: s.rolling(60).mean())
    df["vol5_avg"] = g["amount"].transform(lambda s: s.rolling(5).mean())
    df["vol10_avg"] = g["amount"].transform(lambda s: s.rolling(10).mean())
    df["ret20_vol"] = g["close"].transform(lambda s: s.pct_change().rolling(20).std())
    df["ret60_vol"] = g["close"].transform(lambda s: s.pct_change().rolling(60).std())
    df["high_3_prev"] = g["high"].transform(lambda s: s.shift(1).rolling(3).max())
    df["high_10"] = g["high"].transform(lambda s: s.rolling(10).max())
    df["high_10_prev"] = g["high"].transform(lambda s: s.shift(1).rolling(10).max())
    df["close_below_ma10_2d"] = g.apply(
        lambda x: (x["close"] < x["ma10"]).astype(int).rolling(2).sum().reset_index(level=0, drop=True)
    ).values
    df["above_ma20"] = (df["close"] > df["ma20"]).astype(float)
    df["ret5_pos"] = (df["ret5"] > 0).astype(float)
    df["dist_ma20"] = df["close"] / df["ma20"] - 1.0
    df["dist_ma10"] = df["close"] / df["ma10"] - 1.0
    df["vol_ratio5"] = df["amount"] / df["vol5_avg"]
    df["vol_ratio10"] = df["amount"] / df["vol10_avg"]
    body = (df["close"] - df["open"]).abs()
    upper_shadow = df["high"] - df[["open", "close"]].max(axis=1)
    df["upper_shadow_ratio"] = np.where(df["close"] > 0, upper_shadow / df["close"], 0.0)
    df["body_ratio"] = np.where(df["close"] > 0, body / df["close"], 0.0)

    # Transition coef uses MA20 slope today vs yesterday.
    prev_ma20_angle = g["ma20_angle_deg"].shift(1)
    df["strength_transition_coef"] = [
        calc_strength_transition_coef(a0, a1)
        if pd.notna(a0) and pd.notna(a1)
        else np.nan
        for a0, a1 in zip(df["ma20_angle_deg"], prev_ma20_angle)
    ]

    # Listing days.
    list_date = pd.to_datetime(df["list_date"], format="%Y%m%d", errors="coerce")
    trade_date = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
    df["list_days"] = (trade_date - list_date).dt.days
    return df


def compute_sector_daily(features: pd.DataFrame) -> pd.DataFrame:
    elig = features.copy()
    elig = elig[elig["ths_theme_primary"].fillna("") != ""].copy()
    grp = elig.groupby(["trade_date", "ths_theme_primary"])
    sector = grp.agg(
        member_count=("ts_code", "nunique"),
        rs5=("ret5", "mean"),
        rs10=("ret10", "mean"),
        breadth_ma20=("above_ma20", "mean"),
        breadth_ret5=("ret5_pos", "mean"),
        ret3=("pct_chg", lambda s: pd.to_numeric(s, errors="coerce").tail(3).mean() / 100.0),
        vol_recent=("amount", lambda s: pd.to_numeric(s, errors="coerce").tail(3).mean()),
        vol_base=("amount", lambda s: pd.to_numeric(s, errors="coerce").tail(10).head(7).mean()),
        overheat_ret2=("pct_chg", lambda s: pd.to_numeric(s, errors="coerce").tail(2).sum() / 100.0),
    ).reset_index()

    sector["rs_5_10_raw"] = 0.5 * sector["rs5"].fillna(0.0) + 0.5 * sector["rs10"].fillna(0.0)
    sector["breadth_raw"] = 0.5 * sector["breadth_ma20"].fillna(0.0) + 0.5 * sector["breadth_ret5"].fillna(0.0)
    sector["vol_ratio"] = sector["vol_recent"] / sector["vol_base"]
    sector["pullback_quality_raw"] = (
        1.0
        - sector["ret3"].clip(lower=-0.08, upper=0.05).abs() / 0.08
        - (sector["vol_ratio"].fillna(1.0) - 1.0).clip(lower=0) * 0.3
    ).clip(lower=0.0, upper=1.0)
    sector["overheat_penalty_raw"] = (
        sector["overheat_ret2"].clip(lower=0.0) / 0.08
        + (sector["vol_ratio"].fillna(1.0) - 1.3).clip(lower=0.0) / 1.0
    ).clip(lower=0.0, upper=1.0) / 2.0

    sector = sector.sort_values(["ths_theme_primary", "trade_date"]).reset_index(drop=True)
    by_theme = sector.groupby("ths_theme_primary", group_keys=False)
    sector["rs_persistence_pct"] = by_theme["rs_5_10_raw"].transform(
        lambda s: s.rolling(10).apply(lambda x: (pd.Series(x).rank(pct=True).iloc[-1] >= 0.8), raw=False)
    ).fillna(0.0)

    # Cross-sectional percentile scores by date.
    for col in ["rs_5_10_raw", "breadth_raw", "rs_persistence_pct", "pullback_quality_raw"]:
        sector[col + "_pct"] = sector.groupby("trade_date")[col].rank(pct=True)
    sector["overheat_penalty_pct"] = sector.groupby("trade_date")["overheat_penalty_raw"].rank(pct=True)
    sector["sector_score"] = (
        0.35 * sector["rs_5_10_raw_pct"]
        + 0.25 * sector["breadth_raw_pct"]
        + 0.20 * sector["rs_persistence_pct_pct"]
        + 0.10 * sector["pullback_quality_raw_pct"]
        - 0.10 * sector["overheat_penalty_pct"]
    )
    return sector


class RangeThemeRotationHybrid:
    def __init__(self, base_bt, features: pd.DataFrame, sector_daily: pd.DataFrame):
        self.base = base_bt
        self.cfg = base_bt.cfg
        self.daily = base_bt.daily
        self.idx = base_bt.idx
        self.basic = base_bt.basic
        self.trade_dates = base_bt.trade_dates
        self._stock_data = base_bt._stock_data
        self._idx_series = base_bt._idx_series
        self._get_prices = base_bt._get_prices
        self._get_trend_position = base_bt._get_trend_position
        self._get_trend_state = base_bt._get_trend_state
        self._score_universe = base_bt._score_universe
        self.features = features
        self.sector_daily = sector_daily
        self.feature_by_date = {d: df for d, df in features.groupby("trade_date")}
        self.sector_by_date = {d: df for d, df in sector_daily.groupby("trade_date")}

    def _market_breadth_cap(self, date: str) -> float:
        day = self.feature_by_date.get(date)
        if day is None or day.empty:
            return 0.40
        breadth = ((day["pct_chg"] > 0).mean() + day["above_ma20"].mean()) / 2.0
        return 0.25 if breadth < 0.45 else 0.40

    def _range_candidates(self, date: str) -> list[tuple[str, str, float]]:
        day = self.feature_by_date.get(date)
        sec = self.sector_by_date.get(date)
        if day is None or sec is None or day.empty or sec.empty:
            return []

        pool = day.copy()
        pool = pool[
            (pool["close"] >= self.cfg.min_price)
            & (pool["amount_ma20"] * 1000 >= self.cfg.min_amount_20d)
            & (pool["strength_transition_coef"] >= -0.1)
            & (pool["list_days"] >= self.cfg.min_list_days)
            & (pool["pct_chg"] < 9.5)
            & (pool["ths_theme_primary"].fillna("") != "")
        ].copy()
        if pool.empty:
            return []

        top_sectors = sec.sort_values("sector_score", ascending=False).head(5)["ths_theme_primary"].tolist()
        pool = pool[pool["ths_theme_primary"].isin(top_sectors)].copy()
        if pool.empty:
            return []

        # Sector rank for stop monitoring.
        sector_rank = {
            theme: rank + 1
            for rank, theme in enumerate(sec.sort_values("sector_score", ascending=False)["ths_theme_primary"].tolist())
        }
        pool["sector_rank"] = pool["ths_theme_primary"].map(sector_rank)

        # Trigger conditions.
        trigger = (
            (pool["close"] > pool["high"].shift(1).fillna(-np.inf))
            | (pool["close"] > pool["high_3_prev"].fillna(-np.inf))
        )
        vol_ok = (pool["vol_ratio5"] >= 1.2) & (pool["vol_ratio5"] <= 2.0)
        dist_ok = pool["dist_ma20"].abs() <= 0.06
        pool = pool[trigger & vol_ok & dist_ok].copy()
        if pool.empty:
            return []

        # Stock score.
        pool["stock_rs_raw"] = 0.5 * pool["ret5"].fillna(0.0) + 0.5 * pool["ret10"].fillna(0.0)
        pool["pullback_raw"] = (
            1.0
            - (pool["high_10_prev"] / pool["close"] - 1.0).clip(lower=0.0, upper=0.12) / 0.12
            - pool["dist_ma20"].clip(lower=0.0).fillna(0.0) / 0.06
        ).clip(lower=0.0, upper=1.0)
        pool["trend_health_raw"] = (
            (pool["close"] > pool["ma20"]).astype(float)
            + (pool["close"] > pool["ma60"]).astype(float)
            + (pool["ma20_angle_deg"] >= 0).astype(float)
        ) / 3.0
        pool["volume_quality_raw"] = (
            1.0
            - (pool["vol_ratio10"] - 1.4).abs().clip(upper=1.4) / 1.4
            - pool["upper_shadow_ratio"].fillna(0.0).clip(lower=0.0, upper=0.06) / 0.06 * 0.5
        ).clip(lower=0.0, upper=1.0)
        pool["liquidity_raw"] = np.log((pool["amount_ma20"] * 1000).clip(lower=1.0))
        pool["low_gap_risk_raw"] = (
            1.0
            - pool["dist_ma20"].abs().clip(upper=0.1) / 0.1
            - pool["upper_shadow_ratio"].fillna(0.0).clip(upper=0.06) / 0.06 * 0.5
        ).clip(lower=0.0, upper=1.0)

        for col in [
            "stock_rs_raw",
            "pullback_raw",
            "trend_health_raw",
            "volume_quality_raw",
            "liquidity_raw",
            "low_gap_risk_raw",
        ]:
            pool[col + "_pct"] = pool[col].rank(pct=True)

        pool["stock_score"] = (
            0.30 * pool["stock_rs_raw_pct"]
            + 0.25 * pool["pullback_raw_pct"]
            + 0.15 * pool["trend_health_raw_pct"]
            + 0.15 * pool["volume_quality_raw_pct"]
            + 0.10 * pool["liquidity_raw_pct"]
            + 0.05 * pool["low_gap_risk_raw_pct"]
        )
        pool = pool.sort_values(["sector_rank", "stock_score"], ascending=[True, False])
        return list(pool[["ts_code", "ths_theme_primary", "stock_score"]].itertuples(index=False, name=None))

    def run(self, start_offset: int = 250, include_daily: bool = False) -> dict:
        cfg = self.cfg
        dates = self.trade_dates
        capital = 1_000_000.0
        cash = capital
        positions: dict[str, dict] = {}
        nav_history = []
        trade_count = 0
        rebalance_count = 0
        last_bull_rebalance_idx = start_offset - cfg.rebalance_interval

        start_date = dates[start_offset]
        print("\n" + "=" * 60)
        print("回测: F基线 + RANGE板块轮动子策略")
        print(f"区间: {start_date} ~ {dates[-1]} ({len(dates) - start_offset} 交易日)")
        print("=" * 60)

        for i in range(start_offset, len(dates)):
            date = dates[i]
            prices = self._get_prices(date)
            trend_state = self._get_trend_state(date)

            # Mark current prices.
            for code, pos in list(positions.items()):
                if code in prices:
                    pos["current_price"] = prices[code]

            # Daily exits.
            for code in list(positions.keys()):
                pos = positions[code]
                if code not in prices:
                    continue
                row = self._stock_data[code].loc[date]
                pnl = prices[code] / pos["cost_price"] - 1.0
                exit_reason = None

                if trend_state == "BULL":
                    if pnl <= cfg.stop_loss_pct:
                        exit_reason = "bull_stop"
                elif trend_state == "RANGE":
                    sector_rank = 999
                    sec = self.sector_by_date.get(date)
                    if sec is not None and not sec.empty:
                        sr = sec.sort_values("sector_score", ascending=False).reset_index(drop=True)
                        sector_rank_map = {theme: rank + 1 for rank, theme in enumerate(sr["ths_theme_primary"].tolist())}
                        sector_rank = sector_rank_map.get(pos.get("theme", row.get("ths_theme_primary", "")), 999)

                    if pnl <= -0.045:
                        exit_reason = "range_hard_stop"
                    elif (
                        (pd.notna(row.get("ma10")) and row["close"] < row["ma10"] and row.get("vol_ratio5", 0) > 1.0)
                        or row.get("close_below_ma10_2d", 0) >= 2
                    ):
                        exit_reason = "range_trend_stop"
                    elif sector_rank > 10:
                        exit_reason = "range_sector_stop"
                    elif pos.get("holding_days", 0) >= 5 and pnl < 0.02:
                        exit_reason = "range_time_stop"
                    elif pos.get("holding_days", 0) >= 8:
                        exit_reason = "range_time_out"
                    elif (not pos.get("tp_half_done")) and pnl >= 0.08:
                        sell_price = prices[code] * (1 - cfg.slippage)
                        sell_shares = int(pos["shares"] / 2 / 100) * 100
                        if sell_shares >= 100:
                            proceeds = sell_shares * sell_price
                            cost = proceeds * (cfg.commission + cfg.stamp_tax)
                            cash += proceeds - cost
                            pos["shares"] -= sell_shares
                            pos["tp_half_done"] = True
                            trade_count += 1
                    elif pnl >= 0.12:
                        exit_reason = "range_take_profit"
                else:
                    exit_reason = "bear_exit"

                if exit_reason and code in positions:
                    sell_price = prices[code] * (1 - cfg.slippage)
                    proceeds = positions[code]["shares"] * sell_price
                    cost = proceeds * (cfg.commission + cfg.stamp_tax)
                    cash += proceeds - cost
                    trade_count += 1
                    del positions[code]

            # Increment holding days for survivors.
            for pos in positions.values():
                pos["holding_days"] = pos.get("holding_days", 0) + 1

            if trend_state == "BULL":
                should_rebalance = (i - last_bull_rebalance_idx) >= cfg.rebalance_interval
                if should_rebalance:
                    scores = self._score_universe(date)
                    if len(scores) >= cfg.top_n:
                        target_codes = [s[0] for s in scores[: cfg.top_n]]
                        buffer_codes = set(s[0] for s in scores[: int(cfg.top_n * cfg.hold_buffer_ratio)])
                        for code in list(positions.keys()):
                            if code not in buffer_codes:
                                if code in prices:
                                    sell_price = prices[code] * (1 - cfg.slippage)
                                    proceeds = positions[code]["shares"] * sell_price
                                    cost = proceeds * (cfg.commission + cfg.stamp_tax)
                                    cash += proceeds - cost
                                    trade_count += 1
                                del positions[code]

                        portfolio_value = cash + sum(
                            pos["shares"] * prices.get(code, pos.get("current_price", pos["cost_price"]))
                            for code, pos in positions.items()
                        )
                        current_position_value = sum(
                            pos["shares"] * prices.get(code, pos.get("current_price", pos["cost_price"]))
                            for code, pos in positions.items()
                        )
                        max_equity = portfolio_value
                        available_cash = min(cash, max_equity - current_position_value)
                        buy_slots = cfg.top_n - len(positions)
                        for code in target_codes:
                            if buy_slots <= 0 or available_cash < 10000:
                                break
                            if code in positions or code not in prices:
                                continue
                            buy_price = prices[code] * (1 + cfg.slippage)
                            target_amount = min(portfolio_value * cfg.max_single_weight, available_cash * 0.95)
                            shares = int(target_amount / buy_price / 100) * 100
                            if shares >= 100:
                                amount = shares * buy_price
                                cost = amount * cfg.commission
                                cash -= amount + cost
                                available_cash -= amount + cost
                                positions[code] = {
                                    "shares": shares,
                                    "cost_price": buy_price,
                                    "entry_date": date,
                                    "current_price": prices[code],
                                    "holding_days": 0,
                                    "theme": self._stock_data[code].loc[date].get("ths_theme_primary", ""),
                                    "tp_half_done": False,
                                }
                                trade_count += 1
                                buy_slots -= 1
                        last_bull_rebalance_idx = i
                        rebalance_count += 1

            elif trend_state == "RANGE":
                candidate_list = self._range_candidates(date)
                if candidate_list:
                    max_position_pct = self._market_breadth_cap(date)
                    portfolio_value = cash + sum(
                        pos["shares"] * prices.get(code, pos.get("current_price", pos["cost_price"]))
                        for code, pos in positions.items()
                    )
                    current_position_value = sum(
                        pos["shares"] * prices.get(code, pos.get("current_price", pos["cost_price"]))
                        for code, pos in positions.items()
                    )
                    max_equity = portfolio_value * max_position_pct
                    available_cash = min(cash, max_equity - current_position_value)
                    if available_cash > 10000:
                        sector_exposure = {}
                        for code, pos in positions.items():
                            theme = pos.get("theme", "")
                            sector_exposure[theme] = sector_exposure.get(theme, 0.0) + pos["shares"] * prices.get(code, pos.get("current_price", pos["cost_price"]))

                        max_names = 8
                        for code, theme, _score in candidate_list:
                            if len(positions) >= max_names or available_cash < 10000:
                                break
                            if code in positions or code not in prices:
                                continue
                            buy_price = prices[code] * (1 + cfg.slippage)
                            single_cap = portfolio_value * 0.05
                            sector_cap_remain = portfolio_value * 0.15 - sector_exposure.get(theme, 0.0)
                            target_amount = min(single_cap, sector_cap_remain, available_cash * 0.95)
                            shares = int(target_amount / buy_price / 100) * 100
                            if shares >= 100 and target_amount > 0:
                                amount = shares * buy_price
                                cost = amount * cfg.commission
                                cash -= amount + cost
                                available_cash -= amount + cost
                                sector_exposure[theme] = sector_exposure.get(theme, 0.0) + amount
                                positions[code] = {
                                    "shares": shares,
                                    "cost_price": buy_price,
                                    "entry_date": date,
                                    "current_price": prices[code],
                                    "holding_days": 0,
                                    "theme": theme,
                                    "tp_half_done": False,
                                }
                                trade_count += 1

            # BEAR = do nothing after exits (cash regime)

            final_value = cash + sum(
                pos["shares"] * prices.get(code, pos.get("current_price", pos["cost_price"]))
                for code, pos in positions.items()
            )
            position_value = sum(
                pos["shares"] * prices.get(code, pos.get("current_price", pos["cost_price"]))
                for code, pos in positions.items()
            )
            nav_history.append(
                {
                    "date": date,
                    "nav": final_value,
                    "cash": cash,
                    "position_value": position_value,
                    "position_pct": position_value / final_value if final_value > 0 else 0.0,
                    "trend_state": trend_state,
                    "idx_close": float(self._idx_series.get(date, np.nan)),
                }
            )

        nav_df = pd.DataFrame(nav_history)
        nav_df["daily_return"] = nav_df["nav"].pct_change().fillna(0.0)
        total_return = (nav_df["nav"].iloc[-1] / capital - 1.0) * 100.0
        years = len(nav_df) / 252.0
        annual_return = ((nav_df["nav"].iloc[-1] / capital) ** (1.0 / years) - 1.0) * 100.0
        annual_vol = nav_df["daily_return"].std() * np.sqrt(252.0) * 100.0
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0.0
        cummax = nav_df["nav"].cummax()
        drawdown = (nav_df["nav"] - cummax) / cummax
        max_dd = drawdown.min() * 100.0
        max_dd_date = nav_df.loc[drawdown.idxmin(), "date"]
        idx_start = self._idx_series.get(start_date, None)
        idx_end = self._idx_series.get(dates[-1], None)
        benchmark_return = ((idx_end / idx_start) - 1.0) * 100.0 if idx_start and idx_end else 0.0
        nav_df["year"] = nav_df["date"].str[:4]
        yearly = {
            year: round((grp["nav"].iloc[-1] / grp["nav"].iloc[0] - 1.0) * 100.0, 2)
            for year, grp in nav_df.groupby("year")
            if len(grp) > 10
        }
        result = {
            "name": "F基线 + RANGE板块轮动子策略",
            "total_return_pct": round(total_return, 2),
            "annual_return_pct": round(annual_return, 2),
            "annual_vol_pct": round(annual_vol, 2),
            "sharpe": round(sharpe, 2),
            "calmar": round(annual_return / abs(max_dd), 2) if max_dd != 0 else 0.0,
            "max_drawdown_pct": round(max_dd, 2),
            "max_dd_date": max_dd_date,
            "total_trades": trade_count,
            "rebalance_count": rebalance_count,
            "final_value": round(nav_df["nav"].iloc[-1], 2),
            "benchmark_return_pct": round(benchmark_return, 2),
            "excess_return_pct": round(total_return - benchmark_return, 2),
            "yearly_returns": yearly,
            "trading_days": len(nav_df),
        }
        if include_daily:
            result["daily_records"] = nav_df.to_dict(orient="records")
        print(f"结果: 总收益={total_return:.2f}%, 年化={annual_return:.2f}%, 夏普={sharpe:.2f}, 最大回撤={max_dd:.2f}%")
        return result


def summarize_regimes(daily_df: pd.DataFrame) -> dict:
    out = {}
    for regime in ["BULL", "RANGE", "BEAR"]:
        grp = daily_df[daily_df["trend_state"] == regime].copy()
        rets = grp["daily_return"].fillna(0.0)
        total_return = (1.0 + rets).prod() - 1.0 if len(grp) else 0.0
        annual_return = ((1.0 + total_return) ** (252.0 / len(grp)) - 1.0) * 100.0 if len(grp) else 0.0
        out[regime] = {
            "days": int(len(grp)),
            "total_return_pct": round(total_return * 100.0, 2),
            "annual_return_pct": round(annual_return, 2),
            "avg_position_pct": round(grp["position_pct"].mean() * 100.0, 2) if len(grp) else 0.0,
        }
    return out


def main():
    module = load_backtest_module()
    data_path = DATASETS["csi1000_5y"]
    daily, idx, basic, trade_dates = load_dataset(data_path, module, trend_index_code=DEFAULT_TREND_INDEX_CODE)
    raw = pd.read_csv(data_path, usecols=["data_type", "ts_code", "trade_date", "ma10_angle_deg", "ma20_angle_deg", "amount_ma20", "ths_theme_primary"], low_memory=False)
    extra = raw[raw["data_type"] == "daily"][["ts_code", "trade_date", "ma10_angle_deg", "ma20_angle_deg", "amount_ma20", "ths_theme_primary"]].copy()
    extra["trade_date"] = extra["trade_date"].astype(str)
    for col in ["ma10_angle_deg", "ma20_angle_deg", "amount_ma20"]:
        extra[col] = pd.to_numeric(extra[col], errors="coerce")
    daily = daily.drop(columns=[c for c in ["ma10_angle_deg", "ma20_angle_deg", "amount_ma20", "ths_theme_primary"] if c in daily.columns])
    daily = daily.merge(extra, on=["ts_code", "trade_date"], how="left")

    basic = basic.merge(
        raw[raw["data_type"] == "stock_basic"][["ts_code", "ths_theme_primary"]].drop_duplicates("ts_code"),
        on="ts_code",
        how="left",
    )

    features = build_features(daily, basic)
    sector_daily = compute_sector_daily(features)

    cfg = next(s for s in module.get_strategies() if s.name == "F_三因子+趋势过滤")
    base_cls = make_filtered_backtest(module, min_transition_coef=-0.1)

    t0 = time.time()
    baseline = base_cls(cfg, daily, idx, basic[["ts_code", "name", "industry", "list_date"]], trade_dates).run(start_offset=250, include_daily=True)
    hybrid = RangeThemeRotationHybrid(
        base_cls(cfg, daily, idx, basic[["ts_code", "name", "industry", "list_date"]], trade_dates),
        features,
        sector_daily,
    ).run(start_offset=250, include_daily=True)
    elapsed = round(time.time() - t0, 2)

    baseline_daily = pd.DataFrame(baseline["daily_records"])
    hybrid_daily = pd.DataFrame(hybrid["daily_records"])
    payload = {
        "data_file": str(data_path),
        "trend_filter_index_code": DEFAULT_TREND_INDEX_CODE,
        "range_strategy_note": "RANGE uses THS primary theme rotation with daily review, 40%/25% cap, 4.5% hard stop, time stop, sector stop and staged take-profit.",
        "baseline": {
            k: baseline[k]
            for k in [
                "name",
                "total_return_pct",
                "annual_return_pct",
                "annual_vol_pct",
                "sharpe",
                "calmar",
                "max_drawdown_pct",
                "max_dd_date",
                "total_trades",
                "rebalance_count",
                "final_value",
                "benchmark_return_pct",
                "excess_return_pct",
                "yearly_returns",
                "trading_days",
            ]
        },
        "baseline_regime_summary": summarize_regimes(baseline_daily),
        "hybrid": {
            k: hybrid[k]
            for k in [
                "name",
                "total_return_pct",
                "annual_return_pct",
                "annual_vol_pct",
                "sharpe",
                "calmar",
                "max_drawdown_pct",
                "max_dd_date",
                "total_trades",
                "rebalance_count",
                "final_value",
                "benchmark_return_pct",
                "excess_return_pct",
                "yearly_returns",
                "trading_days",
            ]
        },
        "hybrid_regime_summary": summarize_regimes(hybrid_daily),
        "elapsed_sec": elapsed,
    }
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT_PATH), "hybrid": payload["hybrid"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
