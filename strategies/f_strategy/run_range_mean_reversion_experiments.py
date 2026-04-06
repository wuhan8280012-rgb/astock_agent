#!/usr/bin/env python3
"""
Experiment: mean-reversion sub-strategy for RANGE market regime.

Previous range experiments only changed scoring (which stocks to pick)
but kept the same rebalance interval, position cap, and stop-loss.
All four variants performed worse than baseline (-35% to -40% annual in
RANGE vs -20.5% baseline).

This experiment changes **four dimensions** during RANGE:
  1. Scoring: Bollinger Band + RSI mean-reversion instead of momentum
     - Prefer stocks whose close is below the lower Bollinger Band (oversold)
     - RSI(14) < 35 confirms oversold entry; penalize RSI > 65
     - Low 20-day volatility preferred (stability filter)
  2. Rebalance frequency: 10 days instead of 20 during RANGE
  3. Position cap: 30% max equity during RANGE (vs 50% baseline)
  4. Stop-loss: -4% per-stock during RANGE (vs -15% baseline)

BULL and BEAR regimes remain on the official F baseline logic.
"""

import json
import time
from copy import deepcopy
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
OUT_PATH = (
    PROJECT_ROOT
    / "backtest"
    / "strategy_f_range_mean_reversion_experiments_csi1000_5y.json"
)


# ── helpers ──────────────────────────────────────────────────────


def summarize_regimes(daily_df: pd.DataFrame) -> tuple[dict, dict]:
    regime_summary = {}
    annual_matrix = {}
    for regime in ["BULL", "RANGE", "BEAR"]:
        grp = daily_df[daily_df["trend_state"] == regime].copy()
        if grp.empty:
            regime_summary[regime] = {
                "days": 0,
                "total_return_pct": 0.0,
                "annual_return_pct": 0.0,
                "avg_position_pct": 0.0,
            }
            annual_matrix[regime] = 0.0
            continue

        rets = grp["daily_return"].fillna(0.0)
        total_return = (1.0 + rets).prod() - 1.0
        annual_return = (
            ((1.0 + total_return) ** (252.0 / len(grp)) - 1.0) * 100.0
            if len(grp) > 0
            else 0.0
        )
        regime_summary[regime] = {
            "days": int(len(grp)),
            "total_return_pct": round(total_return * 100.0, 2),
            "annual_return_pct": round(annual_return, 2),
            "avg_position_pct": round(grp["position_pct"].mean() * 100.0, 2),
        }
        annual_matrix[regime] = round(annual_return, 2)
    return regime_summary, annual_matrix


# ── mean-reversion backtest builder ──────────────────────────────


def make_range_mean_reversion_backtest(
    module,
    *,
    range_rebalance_interval: int = 10,
    range_max_position_pct: float = 0.30,
    range_stop_loss_pct: float = -0.04,
    bb_period: int = 20,
    bb_std_mult: float = 2.0,
    rsi_period: int = 14,
    rsi_oversold: float = 35.0,
    rsi_overbought: float = 65.0,
    vol_weight: float = 0.40,
):
    """Build a Backtest subclass that uses mean-reversion scoring in RANGE.

    During RANGE the backtest also:
    - Uses a shorter rebalance interval (*range_rebalance_interval*).
    - Caps total equity at *range_max_position_pct*.
    - Applies a tighter per-stock stop-loss (*range_stop_loss_pct*).
    """

    Base = make_filtered_backtest(module, min_transition_coef=-0.1)

    class MeanReversionRangeBacktest(Base):

        # ── common filter (reused from substrategy experiments) ──

        def _passes_common_filters(self, code, data, date):
            cfg = self.cfg
            if date not in data.index:
                return None
            row = data.loc[date]

            close = row["close"]
            if pd.isna(close) or close < cfg.min_price:
                return None

            info = self._basic_map.get(code)
            if info is not None:
                name = str(info.get("name", ""))
                if "ST" in name.upper():
                    return None
                list_date = str(info.get("list_date", ""))
                if list_date and len(list_date) >= 8:
                    try:
                        from datetime import datetime

                        ld = datetime.strptime(list_date[:8], "%Y%m%d")
                        cd = datetime.strptime(date, "%Y%m%d")
                        if (cd - ld).days < cfg.min_list_days:
                            return None
                    except Exception:
                        pass

            hist = data[data.index <= date]
            if len(hist) < 125:
                return None

            avg_amount = hist.tail(20)["amount"].mean() * 1000
            if avg_amount < cfg.min_amount_20d:
                return None

            if row["pct_chg"] >= 9.5:
                return None

            if "ma20_angle_deg" not in hist.columns:
                return None
            angles = hist["ma20_angle_deg"].tail(2).values.astype(float)
            if len(angles) < 2 or np.isnan(angles[-1]) or np.isnan(angles[-2]):
                return None
            coef = calc_strength_transition_coef(
                float(angles[-1]), float(angles[-2])
            )
            if coef < -0.1:
                return None

            closes = hist["close"].values.astype(float)
            return row, hist, closes

        # ── Bollinger / RSI helpers ──────────────────────────────

        @staticmethod
        def _bollinger(closes, period, std_mult):
            """Return (middle, upper, lower) bands using *period* & *std_mult*."""
            if len(closes) < period:
                return np.nan, np.nan, np.nan
            window = closes[-period:]
            mid = float(np.mean(window))
            std = float(np.std(window, ddof=1))
            return mid, mid + std_mult * std, mid - std_mult * std

        @staticmethod
        def _rsi(closes, period):
            """Wilder-smoothed RSI over the last *period* bars."""
            if len(closes) < period + 1:
                return np.nan
            deltas = np.diff(closes[-(period + 1):])
            gains = np.where(deltas > 0, deltas, 0.0)
            losses = np.where(deltas < 0, -deltas, 0.0)
            avg_gain = float(np.mean(gains))
            avg_loss = float(np.mean(losses))
            if avg_loss == 0:
                return 100.0
            rs = avg_gain / avg_loss
            return 100.0 - 100.0 / (1.0 + rs)

        # ── mean-reversion scoring ───────────────────────────────

        def _score_range_mean_reversion(self, date):
            """Score universe using Bollinger Band position + RSI + low-vol."""
            scores = []
            for code, data in self._stock_data.items():
                passed = self._passes_common_filters(code, data, date)
                if passed is None:
                    continue
                row, hist, closes = passed

                # Bollinger Band position  (how far below lower band)
                mid, upper, lower = self._bollinger(
                    closes, bb_period, bb_std_mult
                )
                if np.isnan(mid) or mid == 0:
                    continue
                # Normalised distance: negative = below lower band (oversold)
                bb_position = (closes[-1] - mid) / (mid - lower) if (mid - lower) > 0 else 0.0
                # We want stocks that are *below* mid → low bb_position
                # Invert so that more-oversold = higher score
                bb_score = -bb_position

                # RSI filter
                rsi = self._rsi(closes, rsi_period)
                if np.isnan(rsi):
                    continue
                # Prefer RSI < oversold threshold; penalise overbought
                if rsi > rsi_overbought:
                    continue  # skip overbought stocks entirely
                rsi_score = (rsi_oversold - rsi) / rsi_oversold  # positive when RSI < threshold

                # 20-day volatility (lower = better)
                rets20 = np.diff(closes[-21:]) / closes[-21:-1] if len(closes) >= 21 else np.array([])
                vol20 = float(np.std(rets20)) if len(rets20) > 0 else 0.0
                vol_component = -vol20

                scores.append(
                    {
                        "code": code,
                        "bb": bb_score,
                        "rsi": rsi_score,
                        "vol": vol_component,
                    }
                )

            if not scores:
                return []

            df = pd.DataFrame(scores)
            # Weighted rank composite: bb + rsi jointly act as 'momentum' axis
            mr_weight = 1.0  # mean-reversion signal (bb + rsi combined)
            v_weight = vol_weight

            # Combine bb and rsi into a single mean-reversion score
            df["mr"] = 0.6 * df["bb"] + 0.4 * df["rsi"]

            total_w = mr_weight + v_weight
            df["mr_rank"] = df["mr"].rank(ascending=False)
            df["vol_rank"] = df["vol"].rank(ascending=False)
            df["composite_rank"] = (
                (mr_weight / total_w) * df["mr_rank"]
                + (v_weight / total_w) * df["vol_rank"]
            )
            df = df.sort_values("composite_rank").reset_index(drop=True)
            return list(zip(df["code"], df["composite_rank"]))

        # ── override _score_universe to switch in RANGE ──────────

        def _score_universe(self, date):
            trend_state = (
                self._get_trend_state(date)
                if self.cfg.use_trend_filter
                else "BULL"
            )
            if trend_state != "RANGE":
                return super()._score_universe(date)
            return self._score_range_mean_reversion(date)

        # ── override run() to inject per-regime parameters ───────

        def run(self, start_offset=250, include_daily=False):
            """Run the back-test with regime-aware rebalance interval,
            position cap, and stop-loss."""
            cfg = self.cfg
            dates = self.trade_dates
            if start_offset >= len(dates):
                return {"error": "数据不足"}

            start_date = dates[start_offset]
            print(
                f"\n{'='*60}\n回测: {cfg.name} (MeanReversion RANGE)\n"
                f"区间: {start_date} ~ {dates[-1]} "
                f"({len(dates) - start_offset} 交易日)\n{'='*60}"
            )

            capital = 1_000_000.0
            cash = capital
            positions = {}
            nav_history = []
            trade_count = 0
            rebalance_count = 0
            last_rebalance_idx = start_offset - cfg.rebalance_interval

            for i in range(start_offset, len(dates)):
                date = dates[i]
                trend_state = (
                    self._get_trend_state(date)
                    if cfg.use_trend_filter
                    else "BULL"
                )

                # ── per-regime parameters ────────────────────────
                if trend_state == "RANGE":
                    active_rebalance_interval = range_rebalance_interval
                    active_stop_loss = range_stop_loss_pct
                else:
                    active_rebalance_interval = cfg.rebalance_interval
                    active_stop_loss = cfg.stop_loss_pct

                # 1. Update position prices
                prices = self._get_prices(date)
                portfolio_value = cash
                for code, pos in list(positions.items()):
                    if code in prices:
                        pos["current_price"] = prices[code]
                        portfolio_value += pos["shares"] * prices[code]
                    else:
                        portfolio_value += pos["shares"] * pos.get(
                            "current_price", pos["cost_price"]
                        )

                # 2. Per-stock stop-loss (daily, regime-aware threshold)
                if active_stop_loss > -0.99:
                    for code in list(positions.keys()):
                        pos = positions[code]
                        if code in prices:
                            pnl = prices[code] / pos["cost_price"] - 1
                            if pnl <= active_stop_loss:
                                sell_price = prices[code] * (1 - cfg.slippage)
                                proceeds = pos["shares"] * sell_price
                                cost = proceeds * (
                                    cfg.commission + cfg.stamp_tax
                                )
                                cash += proceeds - cost
                                trade_count += 1
                                del positions[code]

                # 3. Rebalance
                should_rebalance = (
                    i - last_rebalance_idx
                ) >= active_rebalance_interval
                max_position_pct = 1.0

                if should_rebalance:
                    halt = False
                    if cfg.use_halt:
                        halt = self._check_halt(date)
                        if halt:
                            for code in list(positions.keys()):
                                if code in prices:
                                    sell_price = prices[code] * (
                                        1 - cfg.slippage
                                    )
                                    proceeds = (
                                        positions[code]["shares"] * sell_price
                                    )
                                    cost = proceeds * (
                                        cfg.commission + cfg.stamp_tax
                                    )
                                    cash += proceeds - cost
                                    trade_count += 1
                            positions = {}
                            last_rebalance_idx = i
                            rebalance_count += 1

                    if not halt:
                        # Trend filter → position cap
                        if cfg.use_trend_filter:
                            max_position_pct = self._get_trend_position(date)
                        # Override cap for RANGE to the tighter limit
                        if trend_state == "RANGE":
                            max_position_pct = min(
                                max_position_pct, range_max_position_pct
                            )

                        scores = self._score_universe(date)
                        if len(scores) >= cfg.top_n:
                            target_codes = [
                                s[0] for s in scores[: cfg.top_n]
                            ]
                            buffer_codes = set(
                                s[0]
                                for s in scores[
                                    : int(cfg.top_n * cfg.hold_buffer_ratio)
                                ]
                            )

                            # Sell positions outside buffer
                            for code in list(positions.keys()):
                                if code not in buffer_codes:
                                    if code in prices:
                                        sell_price = prices[code] * (
                                            1 - cfg.slippage
                                        )
                                        proceeds = (
                                            positions[code]["shares"]
                                            * sell_price
                                        )
                                        cost = proceeds * (
                                            cfg.commission + cfg.stamp_tax
                                        )
                                        cash += proceeds - cost
                                        trade_count += 1
                                    del positions[code]

                            # Available equity
                            portfolio_value_now = cash + sum(
                                pos["shares"]
                                * prices.get(
                                    c,
                                    pos.get(
                                        "current_price", pos["cost_price"]
                                    ),
                                )
                                for c, pos in positions.items()
                            )
                            current_position_value = sum(
                                pos["shares"]
                                * prices.get(
                                    c,
                                    pos.get(
                                        "current_price", pos["cost_price"]
                                    ),
                                )
                                for c, pos in positions.items()
                            )
                            max_equity = (
                                portfolio_value_now * max_position_pct
                            )
                            available_for_equity = (
                                max_equity - current_position_value
                            )
                            available_cash = (
                                min(cash, available_for_equity)
                                if available_for_equity > 0
                                else 0
                            )

                            # Buy new positions
                            hold_count = len(positions)
                            buy_slots = cfg.top_n - hold_count

                            for code in target_codes:
                                if buy_slots <= 0 or available_cash < 10000:
                                    break
                                if code in positions:
                                    continue
                                if code not in prices or prices[code] <= 0:
                                    continue

                                buy_price = prices[code] * (
                                    1 + cfg.slippage
                                )
                                target_amount = min(
                                    portfolio_value_now
                                    * cfg.max_single_weight,
                                    available_cash * 0.95,
                                )
                                shares = (
                                    int(target_amount / buy_price / 100) * 100
                                )
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
                                    }
                                    trade_count += 1
                                    buy_slots -= 1

                            last_rebalance_idx = i
                            rebalance_count += 1

                # 4. Record NAV
                final_value = cash + sum(
                    pos["shares"]
                    * prices.get(
                        c, pos.get("current_price", pos["cost_price"])
                    )
                    for c, pos in positions.items()
                )
                position_value = sum(
                    pos["shares"]
                    * prices.get(
                        c, pos.get("current_price", pos["cost_price"])
                    )
                    for c, pos in positions.items()
                )
                nav_history.append(
                    {
                        "date": date,
                        "nav": final_value,
                        "cash": cash,
                        "position_value": position_value,
                        "position_pct": (
                            position_value / final_value
                            if final_value > 0
                            else 0.0
                        ),
                        "max_position_pct": max_position_pct,
                        "trend_state": trend_state,
                        "idx_close": float(
                            self._idx_series.get(date, np.nan)
                        ),
                    }
                )

            # ── compute metrics (same as base) ───────────────────
            nav_df = pd.DataFrame(nav_history)
            nav_df["daily_return"] = nav_df["nav"].pct_change()
            total_return = (nav_df["nav"].iloc[-1] / capital - 1) * 100
            days = len(nav_df)
            years = days / 252
            annual_return = (
                (nav_df["nav"].iloc[-1] / capital) ** (1 / years) - 1
            ) * 100
            annual_vol = nav_df["daily_return"].std() * np.sqrt(252) * 100
            sharpe = annual_return / annual_vol if annual_vol > 0 else 0

            cummax = nav_df["nav"].cummax()
            drawdown = (nav_df["nav"] - cummax) / cummax
            max_dd = drawdown.min() * 100
            max_dd_date = (
                nav_df.loc[drawdown.idxmin(), "date"]
                if len(nav_df) > 0
                else ""
            )

            idx_start = self._idx_series.get(start_date, None)
            idx_end = self._idx_series.get(dates[-1], None)
            benchmark_return = (
                ((idx_end / idx_start) - 1) * 100
                if idx_start and idx_end
                else 0
            )

            nav_df["year"] = nav_df["date"].str[:4]
            yearly = {}
            for year, grp in nav_df.groupby("year"):
                if len(grp) > 10:
                    yr = (grp["nav"].iloc[-1] / grp["nav"].iloc[0] - 1) * 100
                    yearly[year] = round(yr, 2)

            calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

            result = {
                "name": cfg.name,
                "total_return_pct": round(total_return, 2),
                "annual_return_pct": round(annual_return, 2),
                "annual_vol_pct": round(annual_vol, 2),
                "sharpe": round(sharpe, 2),
                "calmar": round(calmar, 2),
                "max_drawdown_pct": round(max_dd, 2),
                "max_dd_date": max_dd_date,
                "total_trades": trade_count,
                "rebalance_count": rebalance_count,
                "final_value": round(nav_df["nav"].iloc[-1], 2),
                "benchmark_return_pct": round(benchmark_return, 2),
                "excess_return_pct": round(
                    total_return - benchmark_return, 2
                ),
                "yearly_returns": yearly,
                "trading_days": days,
            }

            if include_daily:
                daily_export = nav_df.copy()
                daily_export["daily_return"] = daily_export[
                    "daily_return"
                ].fillna(0.0)
                result["daily_records"] = daily_export.to_dict(
                    orient="records"
                )

            print(
                f"\n结果: 总收益={total_return:.2f}%, 年化={annual_return:.2f}%, "
                f"夏普={sharpe:.2f}, 最大回撤={max_dd:.2f}%"
            )
            return result

    return MeanReversionRangeBacktest


# ── experiment runner ────────────────────────────────────────────


def run_variant(module, daily, idx, basic, trade_dates, label, bt_cls):
    cfg = next(
        s for s in module.get_strategies() if s.name == "F_三因子+趋势过滤"
    )
    bt = bt_cls(cfg, daily, idx, basic, trade_dates)
    t0 = time.time()
    result = bt.run(start_offset=250, include_daily=True)
    elapsed = round(time.time() - t0, 2)
    daily_df = pd.DataFrame(result["daily_records"])
    regime_summary, regime_annual_matrix = summarize_regimes(daily_df)
    cleaned = {
        k: result[k]
        for k in [
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
    }
    return {
        "experiment": label,
        "result": cleaned,
        "regime_summary": regime_summary,
        "regime_annual_matrix": regime_annual_matrix,
        "elapsed_sec": elapsed,
    }


# ── main ─────────────────────────────────────────────────────────


def main():
    module = load_backtest_module()
    data_path = DATASETS["csi1000_5y"]
    daily, idx, basic, trade_dates = load_dataset(
        data_path, module, trend_index_code=DEFAULT_TREND_INDEX_CODE
    )

    experiments = [
        # 0. Baseline (unchanged F strategy)
        (
            "baseline",
            make_filtered_backtest(module, min_transition_coef=-0.1),
            "Current main F baseline (rebal=20d, range_cap=50%, sl=-15%)",
        ),
        # 1. Mean-reversion: default parameters
        (
            "mr_default",
            make_range_mean_reversion_backtest(
                module,
                range_rebalance_interval=10,
                range_max_position_pct=0.30,
                range_stop_loss_pct=-0.04,
            ),
            "RANGE mean-reversion: BB(20,2)+RSI(14), rebal=10d, cap=30%, sl=-4%",
        ),
        # 2. Mean-reversion: wider Bollinger (less frequent trigger)
        (
            "mr_bb_wide",
            make_range_mean_reversion_backtest(
                module,
                range_rebalance_interval=10,
                range_max_position_pct=0.30,
                range_stop_loss_pct=-0.04,
                bb_std_mult=2.5,
            ),
            "RANGE mean-reversion: BB(20,2.5)+RSI(14), rebal=10d, cap=30%, sl=-4%",
        ),
        # 3. Mean-reversion: higher position cap (40%)
        (
            "mr_cap40",
            make_range_mean_reversion_backtest(
                module,
                range_rebalance_interval=10,
                range_max_position_pct=0.40,
                range_stop_loss_pct=-0.04,
            ),
            "RANGE mean-reversion: BB(20,2)+RSI(14), rebal=10d, cap=40%, sl=-4%",
        ),
        # 4. Mean-reversion: looser stop-loss (-8%)
        (
            "mr_sl8",
            make_range_mean_reversion_backtest(
                module,
                range_rebalance_interval=10,
                range_max_position_pct=0.30,
                range_stop_loss_pct=-0.08,
            ),
            "RANGE mean-reversion: BB(20,2)+RSI(14), rebal=10d, cap=30%, sl=-8%",
        ),
        # 5. Mean-reversion: even lower cap (20%) – quasi cash-like
        (
            "mr_cap20",
            make_range_mean_reversion_backtest(
                module,
                range_rebalance_interval=10,
                range_max_position_pct=0.20,
                range_stop_loss_pct=-0.04,
            ),
            "RANGE mean-reversion: BB(20,2)+RSI(14), rebal=10d, cap=20%, sl=-4%",
        ),
        # 6. Pure cash in RANGE (Direction A reference point)
        (
            "mr_cash_only",
            make_range_mean_reversion_backtest(
                module,
                range_rebalance_interval=10,
                range_max_position_pct=0.0,
                range_stop_loss_pct=-0.04,
            ),
            "RANGE: 100% cash (Direction A reference)",
        ),
    ]

    runs = []
    for label, cls, desc in experiments:
        print(f"\nRUN {label}", flush=True)
        run = run_variant(module, daily, idx, basic, trade_dates, label, cls)
        run["description"] = desc
        runs.append(run)
        r = run["result"]
        rr = run["regime_summary"].get("RANGE", {})
        print(
            f"DONE {label}  total={r['total_return_pct']}  "
            f"annual={r['annual_return_pct']}  sharpe={r['sharpe']}  "
            f"mdd={r['max_drawdown_pct']}  "
            f"RANGE_annual={rr.get('annual_return_pct', 'N/A')}",
            flush=True,
        )

    # ── comparison table ─────────────────────────────────────────
    print("\n" + "=" * 100)
    print("COMPARISON TABLE")
    print("=" * 100)
    header = (
        f"{'Experiment':<20} {'Total%':>8} {'Annual%':>8} {'Sharpe':>7} "
        f"{'MDD%':>7} {'RANGE Ann%':>10} {'RANGE Pos%':>10} {'Trades':>7}"
    )
    print(header)
    print("-" * 100)
    for run in runs:
        r = run["result"]
        rr = run["regime_summary"].get("RANGE", {})
        print(
            f"{run['experiment']:<20} "
            f"{r['total_return_pct']:>8.1f} "
            f"{r['annual_return_pct']:>8.1f} "
            f"{r['sharpe']:>7.2f} "
            f"{r['max_drawdown_pct']:>7.1f} "
            f"{rr.get('annual_return_pct', 0):>10.1f} "
            f"{rr.get('avg_position_pct', 0):>10.1f} "
            f"{r['total_trades']:>7}"
        )

    # ── save results ─────────────────────────────────────────────
    payload = {
        "data_file": str(data_path),
        "trend_filter_index_code": DEFAULT_TREND_INDEX_CODE,
        "strategy": "F + strength_transition_coef >= -0.1",
        "note": (
            "Mean-reversion experiment for RANGE regime. "
            "BULL/BEAR remain on the current official F baseline. "
            "Changes 4 dimensions in RANGE: scoring (Bollinger+RSI), "
            "rebalance interval (10d), position cap (20-40%), stop-loss (-4/-8%)."
        ),
        "experiments": runs,
    }
    OUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(OUT_PATH)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
