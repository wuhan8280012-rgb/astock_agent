#!/usr/bin/env python3
"""
Leader Strategy (龙头策略) backtest runner.

Core thesis: detect institutional accumulation via capital-flow proxies
(absorption score, RS vs industry) rather than waiting for lagging
earnings reports. Revenue growth serves as a *hold* confirmation, not
an entry signal.

Architecture:
  - Inherits the full ``Backtest`` engine from ``scripts/backtest_strategies.py``
  - Overrides ``_score_universe()`` with leader-specific factor scoring
  - Reuses data loading from ``strategies/f_strategy/run_backtest.py``

Usage:
    python strategies/leader_strategy/run_backtest.py [--dataset csi1000_5y]
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from leader_factors import (
    calc_absorption_score,
    calc_industry_relative_strength,
    is_rs_new_high,
)

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
#  Leader Strategy Config
# ══════════════════════════════════════════════════════════════════

def get_leader_config(bt_module):
    """Return a BacktestConfig tuned for the Leader strategy."""
    return bt_module.BacktestConfig(
        name="Leader_RS+吸筹+趋势",
        # Momentum — 60d only, as the main ranking signal
        momentum_days=[60],
        momentum_weights=[1.0],
        subtract_short_momentum=False,
        # No reversal, no regime switch — leader strategy is pure trend-following
        use_reversal_factor=False,
        use_regime_switch=False,
        # Low-volatility as a mild quality filter (not dominant)
        use_volatility_factor=True,
        volatility_days=60,
        volatility_weight=0.15,
        # No size factor — leaders can be any cap
        use_size_factor=False,
        # Portfolio: 10 concentrated holdings, 12% max single
        top_n=10,
        hold_buffer_ratio=1.5,
        max_single_weight=0.12,
        rebalance_interval=20,
        # Risk control
        stop_loss_pct=-0.15,
        use_halt=False,
        use_trend_filter=True,
        trend_ma_days=200,
        trend_reduce_pct=0.3,    # 30% position when below trend — more aggressive reduction
        # Cost
        commission=0.0003,
        stamp_tax=0.001,
        slippage=0.003,          # 0.3% for concentrated leader positions
        # Filters — tighter than F strategy
        min_amount_20d=3e8,      # 20d avg amount >= 3亿 (vs F's 1亿)
        min_price=5.0,           # no penny stocks
        min_list_days=250,
    )


# ══════════════════════════════════════════════════════════════════
#  Leader Backtest (override _score_universe)
# ══════════════════════════════════════════════════════════════════

def make_leader_backtest(bt_module):
    """Create a Backtest subclass with leader-factor scoring."""

    class LeaderBacktest(bt_module.Backtest):
        def __init__(self, cfg, daily, idx, basic, trade_dates):
            super().__init__(cfg, daily, idx, basic, trade_dates)
            # Pre-compute per-industry 60d returns for RS calculation
            self._industry_ret60 = self._build_industry_ret_lookup(daily, lookback=60)

        def _build_industry_ret_lookup(self, daily: pd.DataFrame, lookback: int = 60) -> dict:
            """Build {trade_date -> {industry_name -> ret_Nd}} lookup.

            Uses the same logic as enrich_industry_strength20_features but
            with a configurable lookback window.
            """
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

        def _score_universe(self, date: str) -> list:
            """Score stocks using leader factors: RS, absorption, momentum."""
            cfg = self.cfg
            if date not in self.trade_dates:
                return []

            scores = []
            for code, data in self._stock_data.items():
                if date not in data.index:
                    continue
                row = data.loc[date]

                # ── Basic filters (same as base engine) ──
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
                if len(hist) < 125:  # need 120d for RS new-high check + buffer
                    continue

                # Liquidity filter (tighter than F)
                avg_amount = hist.tail(20)["amount"].mean() * 1000
                if avg_amount < cfg.min_amount_20d:
                    continue

                # No limit-up buys
                if row["pct_chg"] >= 9.5:
                    continue

                closes = hist["close"].values.astype(float)

                # ── Factor 1: 60d Momentum ──
                mom_60d = 0.0
                if len(closes) >= 61:
                    mom_60d = closes[-1] / closes[-61] - 1
                else:
                    continue

                # ── Factor 2: Absorption Score (放量不跌) ──
                absorption = calc_absorption_score(hist)

                # ── Factor 3: RS vs Industry ──
                rs_value = None
                rs_new_high = False
                ind_name = row.get("sw_l1_name", industry)
                if ind_name:
                    ind_ret = self._get_industry_ret(date, ind_name)
                    if not np.isnan(ind_ret):
                        rs_value = calc_industry_relative_strength(
                            closes, ind_ret, lookback=60
                        )
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

                # ── Factor 4: Low Volatility (mild) ──
                vol_component = 0.0
                if cfg.use_volatility_factor and len(closes) >= cfg.volatility_days + 1:
                    rets = np.diff(closes[-cfg.volatility_days - 1:]) / closes[-cfg.volatility_days - 1:-1]
                    vol = np.std(rets)
                    if vol > 0:
                        vol_component = -vol

                scores.append((
                    code,
                    mom_60d,
                    absorption,
                    rs_value if rs_value is not None else 0.0,
                    1.0 if rs_new_high else 0.0,
                    vol_component,
                ))

            if not scores:
                return []

            # ── Rank composition ──
            df = pd.DataFrame(scores, columns=[
                "code", "mom_60d", "absorption", "rs_vs_industry",
                "rs_new_high", "vol",
            ])

            # Per-factor rank (lower rank = better)
            df["mom_rank"] = df["mom_60d"].rank(ascending=False)
            df["absorption_rank"] = df["absorption"].rank(ascending=False)
            df["rs_rank"] = df["rs_vs_industry"].rank(ascending=False)
            df["rs_nh_rank"] = df["rs_new_high"].rank(ascending=False)
            df["vol_rank"] = df["vol"].rank(ascending=False)

            # Weighted composite rank
            # Weights: momentum 30%, absorption 20%, RS 25%, RS new-high 10%, vol 15%
            w_mom = 0.30
            w_abs = 0.20
            w_rs = 0.25
            w_nh = 0.10
            w_vol = cfg.volatility_weight if cfg.use_volatility_factor else 0.0
            # Normalize so weights sum to 1
            total_w = w_mom + w_abs + w_rs + w_nh + w_vol
            df["composite_rank"] = (
                (w_mom / total_w) * df["mom_rank"]
                + (w_abs / total_w) * df["absorption_rank"]
                + (w_rs / total_w) * df["rs_rank"]
                + (w_nh / total_w) * df["rs_nh_rank"]
                + (w_vol / total_w) * df["vol_rank"]
            )

            df = df.sort_values("composite_rank").reset_index(drop=True)
            return list(zip(df["code"], df["composite_rank"]))

    return LeaderBacktest


# ══════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Leader (龙头) strategy backtest")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="csi1000_5y")
    parser.add_argument("--trend-index-code", type=str, default="000001.SH")
    args = parser.parse_args()

    data_path = DATASETS[args.dataset]
    if not data_path.exists():
        # Try compressed version
        gz_path = data_path.with_suffix(data_path.suffix + ".gz")
        if gz_path.exists():
            print(f"Decompressing {gz_path} ...")
            import gzip, shutil
            with gzip.open(gz_path, "rb") as f_in, open(data_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        else:
            print(f"ERROR: Dataset not found: {data_path}")
            print(f"       (also checked {gz_path})")
            print(f"       Run the data export pipeline first.")
            return

    bt_module = load_backtest_module()
    f_module = load_f_strategy_module()

    # Reuse F strategy's data loader (handles adj_factor, industry features, etc.)
    daily, idx, basic, trade_dates = f_module.load_dataset(
        data_path, bt_module, trend_index_code=args.trend_index_code,
    )

    cfg = get_leader_config(bt_module)
    LeaderBT = make_leader_backtest(bt_module)

    print(f"\n{'='*60}")
    print(f"Leader Strategy Backtest")
    print(f"Dataset: {args.dataset} ({len(trade_dates)} trade days)")
    print(f"{'='*60}")

    t0 = time.time()
    bt = LeaderBT(cfg, daily, idx, basic, trade_dates)
    result = bt.run(start_offset=250)
    result["elapsed_sec"] = round(time.time() - t0, 1)
    result["trend_filter_index_code"] = args.trend_index_code

    # Save
    suffix = f"_{args.dataset}"
    out_path = PROJECT_ROOT / "backtest" / f"strategy_leader{suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"data_file": str(data_path), "result": result}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nWROTE {out_path}")


if __name__ == "__main__":
    main()
