#!/usr/bin/env python3
"""
RS + MA20 Angle Strategy (龙头策略 v2) backtest runner.

Core thesis: combine industry-relative strength (RS) with MA20 angle
transition coefficient to identify leaders that are *accelerating*
their outperformance.

Architecture:
  - Inherits ``Backtest`` from ``scripts/backtest_strategies.py``
  - Overrides ``_score_universe()`` with 4-factor ranking:
      RS vs industry (40%), transition_coef (30%),
      RS new-high (15%), 60d momentum (15%)
  - Overrides ``run()`` to add emergency exit on coef < -0.5
  - Hard filters: transition_coef ≥ 0, RS > 1.0, ma20_angle_deg > 0

Usage:
    python strategies/rs_angle_strategy/run_backtest.py [--dataset csi1000_5y]
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Import leader factors from sibling package
LEADER_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "leader_strategy"

import sys
sys.path.insert(0, str(LEADER_STRATEGY_DIR))

from leader_factors import (
    calc_industry_relative_strength,
    is_rs_new_high,
)

sys.path.pop(0)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKTEST_SCRIPT = PROJECT_ROOT / "scripts" / "backtest_strategies.py"
F_STRATEGY_SCRIPT = PROJECT_ROOT / "strategies" / "f_strategy" / "run_backtest.py"

DATASETS = {
    "csi1000_5y": PROJECT_ROOT / "data_exports" / "tushare_20210329_20260327_csi1000_5y" / "csi1000_market_bundle_5y.csv",
    "csi1000_5y_pit": PROJECT_ROOT / "data_exports" / "tushare_20210329_20260327_csi1000_5y_pit" / "csi1000_market_bundle_5y_pit.csv",
}


def load_backtest_module():
    spec = importlib.util.spec_from_file_location("backtest_strategies", BACKTEST_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_f_strategy_module():
    spec = importlib.util.spec_from_file_location("f_run_backtest", F_STRATEGY_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ══════════════════════════════════════════════════════════════════
#  Transition coefficient — reuse the F strategy formula
# ══════════════════════════════════════════════════════════════════

def calc_strength_transition_coef(a0: float, a1: float) -> float:
    """Combine current MA20 angle strength with angle acceleration.

    Parameters
    ----------
    a0 : float
        Current day's ma20_angle_deg.
    a1 : float
        Previous day's ma20_angle_deg.

    Returns
    -------
    float
        Transition coefficient in [-1, 1].
        > 0 means strengthening trend, < 0 means weakening.
    """
    base = np.tanh(a0 / 10.0)
    turn = np.tanh((a0 - a1) / 5.0)
    return float(np.clip(0.7 * base + 0.3 * turn, -1.0, 1.0))


# ══════════════════════════════════════════════════════════════════
#  RS + Angle Strategy Config
# ══════════════════════════════════════════════════════════════════

def get_rs_angle_config(bt_module):
    """Return a BacktestConfig tuned for the RS + MA20 Angle strategy."""
    return bt_module.BacktestConfig(
        name="RSAngle_龙头v2_RS40%+Coef30%",
        # Momentum — 60d only, as a baseline quality check
        momentum_days=[60],
        momentum_weights=[1.0],
        subtract_short_momentum=False,
        # Pure trend-following — no reversal or regime
        use_reversal_factor=False,
        use_regime_switch=False,
        # No volatility or size in base config — handled in _score_universe
        use_volatility_factor=False,
        use_size_factor=False,
        # Portfolio: 10 concentrated, 12% max, 半月 (10d) rebalance
        top_n=10,
        hold_buffer_ratio=1.3,       # Top 13 buffer band
        max_single_weight=0.12,
        rebalance_interval=10,       # 半月调仓 — faster than leader's 20d
        # Risk control
        stop_loss_pct=-0.15,
        use_halt=False,
        use_trend_filter=True,
        trend_ma_days=200,
        trend_reduce_pct=0.3,        # 30% position when below trend
        # Cost
        commission=0.0003,
        stamp_tax=0.001,
        slippage=0.003,              # 0.3% for concentrated positions
        # Filters — tight liquidity requirement
        min_amount_20d=3e8,          # 20d avg amount >= 3亿
        min_price=5.0,
        min_list_days=250,
    )


# ══════════════════════════════════════════════════════════════════
#  RS + Angle Backtest (override _score_universe + run)
# ══════════════════════════════════════════════════════════════════

# Emergency exit threshold: if transition_coef drops below this,
# sell immediately without waiting for rebalance day.
EMERGENCY_EXIT_COEF = -0.5


def make_rs_angle_backtest(bt_module):
    """Create a Backtest subclass with RS + MA20 Angle scoring."""

    class RSAngleBacktest(bt_module.Backtest):
        def __init__(self, cfg, daily, idx, basic, trade_dates):
            super().__init__(cfg, daily, idx, basic, trade_dates)
            # Pre-compute per-industry 60d returns for RS calculation
            self._industry_ret60 = self._build_industry_ret_lookup(daily, lookback=60)

        # ── Industry return helpers (same as LeaderBacktest) ──

        def _build_industry_ret_lookup(self, daily: pd.DataFrame, lookback: int = 60) -> dict:
            """Build {trade_date -> {industry_name -> ret_Nd}} lookup."""
            if "sw_l1_name" not in daily.columns:
                return {}

            ind = (
                daily.dropna(subset=["sw_l1_name", "close"])
                .groupby(["trade_date", "sw_l1_name"], as_index=False)["close"]
                .mean()
                .sort_values(["sw_l1_name", "trade_date"])
            )
            ind[f"ret{lookback}"] = ind.groupby("sw_l1_name")["close"].transform(
                lambda s: s / s.shift(lookback) - 1
            )

            result = {}
            for _, row in ind.dropna(subset=[f"ret{lookback}"]).iterrows():
                d = str(row["trade_date"])
                if d not in result:
                    result[d] = {}
                result[d][row["sw_l1_name"]] = float(row[f"ret{lookback}"])
            return result

        def _get_industry_ret(self, date: str, industry: str) -> float:
            """Get industry Nd return for a given date."""
            date_map = self._industry_ret60.get(date, {})
            return date_map.get(industry, np.nan)

        # ── Transition coefficient for a stock on a given date ──

        def _calc_stock_transition_coef(self, hist: pd.DataFrame) -> float:
            """Calculate transition_coef from the last 2 days of ma20_angle_deg."""
            if "ma20_angle_deg" not in hist.columns:
                return np.nan
            angles = hist["ma20_angle_deg"].tail(2).values.astype(float)
            if len(angles) < 2 or np.isnan(angles[-1]) or np.isnan(angles[-2]):
                return np.nan
            return calc_strength_transition_coef(float(angles[-1]), float(angles[-2]))

        # ── Core scoring ──

        def _score_universe(self, date: str) -> list:
            """Score stocks using RS + MA20 Angle factors.

            Hard filters (all must pass):
              - transition_coef >= 0
              - RS vs industry > 1.0
              - ma20_angle_deg > 0

            Ranking factors:
              - RS vs industry:    40%
              - transition_coef:   30%
              - RS new-high:       15%
              - 60d abs momentum:  15%
            """
            cfg = self.cfg
            if date not in self.trade_dates:
                return []

            scores = []
            for code, data in self._stock_data.items():
                if date not in data.index:
                    continue
                row = data.loc[date]

                # ── Basic filters ──
                close = row["close"]
                if pd.isna(close) or close < cfg.min_price:
                    continue

                info = self._basic_map.get(code)
                industry = None
                if info is not None:
                    name = str(info.get("name", ""))
                    if "ST" in name.upper():
                        continue
                    list_date = str(info.get("list_date", ""))
                    if list_date and len(list_date) >= 8:
                        try:
                            from datetime import datetime
                            ld = datetime.strptime(list_date[:8], "%Y%m%d")
                            cd = datetime.strptime(date, "%Y%m%d")
                            if (cd - ld).days < cfg.min_list_days:
                                continue
                        except Exception:
                            pass
                    industry = str(info.get("industry", ""))

                hist = data[data.index <= date]
                if len(hist) < 125:  # need 120d for RS new-high + buffer
                    continue

                # Liquidity filter
                avg_amount = hist.tail(20)["amount"].mean() * 1000
                if avg_amount < cfg.min_amount_20d:
                    continue

                # No limit-up buys
                if row["pct_chg"] >= 9.5:
                    continue

                # ── Hard filter 1: ma20_angle_deg > 0 ──
                angle = row.get("ma20_angle_deg", np.nan)
                if pd.isna(angle) or angle <= 0:
                    continue

                # ── Hard filter 2: transition_coef >= 0 ──
                coef = self._calc_stock_transition_coef(hist)
                if np.isnan(coef) or coef < 0:
                    continue

                closes = hist["close"].values.astype(float)

                # ── Hard filter 3: RS vs industry > 1.0 ──
                rs_value = None
                rs_new_high = False
                ind_name = row.get("sw_l1_name", industry)
                if ind_name:
                    ind_ret = self._get_industry_ret(date, ind_name)
                    if not np.isnan(ind_ret):
                        rs_value = calc_industry_relative_strength(
                            closes, ind_ret, lookback=60
                        )
                        if rs_value is not None:
                            # Build industry returns series for new-high check
                            ind_rets_series = pd.Series({
                                d: self._get_industry_ret(d, ind_name)
                                for d in hist.index[-125:]
                                if not np.isnan(self._get_industry_ret(d, ind_name))
                            })
                            if len(ind_rets_series) > 10:
                                rs_new_high = is_rs_new_high(
                                    hist, ind_rets_series,
                                    rs_lookback=60, high_lookback=120,
                                )

                # Hard filter: RS must exceed 1.0 (outperforming industry)
                if rs_value is None or rs_value <= 1.0:
                    continue

                # ── 60d absolute momentum ──
                if len(closes) < 61:
                    continue
                mom_60d = closes[-1] / closes[-61] - 1

                scores.append((
                    code,
                    rs_value,
                    coef,
                    1.0 if rs_new_high else 0.0,
                    mom_60d,
                ))

            if not scores:
                return []

            # ── Rank composition ──
            df = pd.DataFrame(scores, columns=[
                "code", "rs_vs_industry", "transition_coef",
                "rs_new_high", "mom_60d",
            ])

            # Per-factor rank (lower rank = better)
            df["rs_rank"] = df["rs_vs_industry"].rank(ascending=False)
            df["coef_rank"] = df["transition_coef"].rank(ascending=False)
            df["rs_nh_rank"] = df["rs_new_high"].rank(ascending=False)
            df["mom_rank"] = df["mom_60d"].rank(ascending=False)

            # Weighted composite rank
            # RS 40%, transition_coef 30%, RS new-high 15%, momentum 15%
            w_rs = 0.40
            w_coef = 0.30
            w_nh = 0.15
            w_mom = 0.15

            df["composite_rank"] = (
                w_rs * df["rs_rank"]
                + w_coef * df["coef_rank"]
                + w_nh * df["rs_nh_rank"]
                + w_mom * df["mom_rank"]
            )

            df = df.sort_values("composite_rank").reset_index(drop=True)
            return list(zip(df["code"], df["composite_rank"]))

        # ── Override run() for emergency exit logic ──

        def run(self, start_offset: int = 250, include_daily: bool = False) -> dict:
            """Run backtest with emergency exit on coef < -0.5.

            Between rebalance days, if any held stock's transition_coef
            drops below EMERGENCY_EXIT_COEF, it is sold immediately.
            """
            cfg = self.cfg
            dates = self.trade_dates
            if start_offset >= len(dates):
                return {"error": "数据不足"}

            start_date = dates[start_offset]
            print(f"\n{'='*60}")
            print(f"回测: {cfg.name}")
            print(f"区间: {start_date} ~ {dates[-1]} ({len(dates) - start_offset} 交易日)")
            print(f"参数: top_n={cfg.top_n}, 调仓间隔={cfg.rebalance_interval}d, "
                  f"止损={cfg.stop_loss_pct:.0%}, 紧急退出coef<{EMERGENCY_EXIT_COEF}")
            print(f"{'='*60}")

            capital = 1_000_000.0
            cash = capital
            positions = {}
            nav_history = []
            trade_count = 0
            rebalance_count = 0
            emergency_exit_count = 0
            last_rebalance_idx = start_offset - cfg.rebalance_interval

            for i in range(start_offset, len(dates)):
                date = dates[i]

                # 1. Update position prices
                prices = self._get_prices(date)
                portfolio_value = cash
                for code, pos in list(positions.items()):
                    if code in prices:
                        pos["current_price"] = prices[code]
                        portfolio_value += pos["shares"] * prices[code]
                    else:
                        portfolio_value += pos["shares"] * pos.get("current_price", pos["cost_price"])

                # 2. Individual stop-loss check (daily)
                if cfg.stop_loss_pct > -0.99:
                    for code in list(positions.keys()):
                        pos = positions[code]
                        if code in prices:
                            pnl = prices[code] / pos["cost_price"] - 1
                            if pnl <= cfg.stop_loss_pct:
                                sell_price = prices[code] * (1 - cfg.slippage)
                                proceeds = pos["shares"] * sell_price
                                cost = proceeds * (cfg.commission + cfg.stamp_tax)
                                cash += proceeds - cost
                                trade_count += 1
                                del positions[code]

                # 2b. Emergency exit: coef < -0.5
                for code in list(positions.keys()):
                    if code not in self._stock_data:
                        continue
                    stock_data = self._stock_data[code]
                    hist = stock_data[stock_data.index <= date]
                    coef = self._calc_stock_transition_coef(hist)
                    if not np.isnan(coef) and coef < EMERGENCY_EXIT_COEF:
                        if code in prices:
                            sell_price = prices[code] * (1 - cfg.slippage)
                            proceeds = positions[code]["shares"] * sell_price
                            cost = proceeds * (cfg.commission + cfg.stamp_tax)
                            cash += proceeds - cost
                            trade_count += 1
                            emergency_exit_count += 1
                            del positions[code]

                # 3. Rebalance check
                should_rebalance = (i - last_rebalance_idx) >= cfg.rebalance_interval
                max_position_pct = 1.0

                if should_rebalance:
                    # Trend filter
                    if cfg.use_trend_filter:
                        max_position_pct = self._get_trend_position(date)

                    # Score universe
                    scores = self._score_universe(date)
                    if len(scores) >= cfg.top_n:
                        target_codes = [s[0] for s in scores[:cfg.top_n]]
                        buffer_codes = set(
                            s[0] for s in scores[:int(cfg.top_n * cfg.hold_buffer_ratio)]
                        )

                        # Sell: not in buffer band
                        for code in list(positions.keys()):
                            if code not in buffer_codes:
                                if code in prices:
                                    sell_price = prices[code] * (1 - cfg.slippage)
                                    proceeds = positions[code]["shares"] * sell_price
                                    cost = proceeds * (cfg.commission + cfg.stamp_tax)
                                    cash += proceeds - cost
                                    trade_count += 1
                                del positions[code]

                        # Calculate available buying power
                        portfolio_value_now = cash + sum(
                            pos["shares"] * prices.get(code, pos.get("current_price", pos["cost_price"]))
                            for code, pos in positions.items()
                        )
                        current_position_value = sum(
                            pos["shares"] * prices.get(code, pos.get("current_price", pos["cost_price"]))
                            for code, pos in positions.items()
                        )
                        max_equity = portfolio_value_now * max_position_pct
                        available_for_equity = max_equity - current_position_value
                        available_cash = min(cash, available_for_equity) if available_for_equity > 0 else 0

                        # Buy: in target but not held
                        hold_count = len(positions)
                        buy_slots = cfg.top_n - hold_count

                        for code in target_codes:
                            if buy_slots <= 0 or available_cash < 10000:
                                break
                            if code in positions:
                                continue
                            if code not in prices or prices[code] <= 0:
                                continue

                            buy_price = prices[code] * (1 + cfg.slippage)
                            target_amount = min(
                                portfolio_value_now * cfg.max_single_weight,
                                available_cash * 0.95,
                            )
                            shares = int(target_amount / buy_price / 100) * 100
                            if shares >= 100:
                                amount = shares * buy_price
                                cost = amount * cfg.commission
                                cash -= (amount + cost)
                                available_cash -= (amount + cost)
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
                    pos["shares"] * prices.get(code, pos.get("current_price", pos["cost_price"]))
                    for code, pos in positions.items()
                )
                position_value = sum(
                    pos["shares"] * prices.get(code, pos.get("current_price", pos["cost_price"]))
                    for code, pos in positions.items()
                )
                trend_state = self._get_trend_state(date) if cfg.use_trend_filter else "FULL"
                nav_history.append({
                    "date": date,
                    "nav": final_value,
                    "cash": cash,
                    "position_value": position_value,
                    "position_pct": position_value / final_value if final_value > 0 else 0.0,
                    "max_position_pct": max_position_pct,
                    "trend_state": trend_state,
                    "idx_close": float(self._idx_series.get(date, np.nan)),
                })

            # Compute performance metrics
            nav_df = pd.DataFrame(nav_history)
            nav_df["daily_return"] = nav_df["nav"].pct_change()
            total_return = (nav_df["nav"].iloc[-1] / capital - 1) * 100
            days = len(nav_df)
            years = days / 252
            annual_return = ((nav_df["nav"].iloc[-1] / capital) ** (1 / years) - 1) * 100
            annual_vol = nav_df["daily_return"].std() * np.sqrt(252) * 100
            sharpe = annual_return / annual_vol if annual_vol > 0 else 0

            cummax = nav_df["nav"].cummax()
            drawdown = (nav_df["nav"] - cummax) / cummax
            max_dd = drawdown.min() * 100
            max_dd_date = nav_df.loc[drawdown.idxmin(), "date"] if len(nav_df) > 0 else ""

            idx_start = self._idx_series.get(start_date, None)
            idx_end = self._idx_series.get(dates[-1], None)
            benchmark_return = ((idx_end / idx_start) - 1) * 100 if idx_start and idx_end else 0

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
                "emergency_exit_count": emergency_exit_count,
                "final_value": round(nav_df["nav"].iloc[-1], 2),
                "benchmark_return_pct": round(benchmark_return, 2),
                "excess_return_pct": round(total_return - benchmark_return, 2),
                "yearly_returns": yearly,
                "trading_days": days,
            }

            if include_daily:
                daily_export = nav_df.copy()
                daily_export["daily_return"] = daily_export["daily_return"].fillna(0.0)
                result["daily_records"] = daily_export.to_dict(orient="records")

            print(f"\n结果: 总收益={total_return:.2f}%, 年化={annual_return:.2f}%, "
                  f"夏普={sharpe:.2f}, 最大回撤={max_dd:.2f}%")
            print(f"交易次数={trade_count}, 调仓次数={rebalance_count}, "
                  f"紧急退出={emergency_exit_count}")
            print(f"分年度: {yearly}")
            print(f"基准: {benchmark_return:.2f}%, 超额: {total_return - benchmark_return:.2f}%")

            return result

    return RSAngleBacktest


# ══════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RS + MA20 Angle (龙头v2) strategy backtest")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="csi1000_5y")
    parser.add_argument("--trend-index-code", type=str, default="000001.SH")
    args = parser.parse_args()

    data_path = DATASETS[args.dataset]
    if not data_path.exists():
        gz_path = data_path.with_suffix(data_path.suffix + ".gz")
        if gz_path.exists():
            print(f"Decompressing {gz_path} ...")
            import gzip
            import shutil
            with gzip.open(gz_path, "rb") as f_in, open(data_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        else:
            print(f"ERROR: Dataset not found: {data_path}")
            print(f"       (also checked {gz_path})")
            print(f"       Run the data export pipeline first.")
            return

    bt_module = load_backtest_module()
    f_module = load_f_strategy_module()

    # Reuse F strategy's data loader
    daily, idx, basic, trade_dates = f_module.load_dataset(
        data_path, bt_module, trend_index_code=args.trend_index_code,
    )

    cfg = get_rs_angle_config(bt_module)
    RSAngleBT = make_rs_angle_backtest(bt_module)

    print(f"\n{'='*60}")
    print(f"RS + MA20 Angle Strategy Backtest")
    print(f"Dataset: {args.dataset} ({len(trade_dates)} trade days)")
    print(f"{'='*60}")

    t0 = time.time()
    bt = RSAngleBT(cfg, daily, idx, basic, trade_dates)
    result = bt.run(start_offset=250)
    result["elapsed_sec"] = round(time.time() - t0, 1)
    result["trend_filter_index_code"] = args.trend_index_code

    # Save
    suffix = f"_{args.dataset}"
    out_path = PROJECT_ROOT / "backtest" / f"strategy_rs_angle{suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"data_file": str(data_path), "result": result}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nWROTE {out_path}")


if __name__ == "__main__":
    main()
