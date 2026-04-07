#!/usr/bin/env python3
"""
Trend Initiation Strategy (趋势启动策略) backtest runner.

Core thesis: buy stocks emerging from a consolidation/shakeout phase
where the transition coefficient is **accelerating** and momentum is
picking up — but the stock has NOT yet topped.

Four orthogonal signals:
  1. Volatility contraction (振仓后压缩): vol_10d / vol_60d < 1
  2. Coef acceleration (coef 加速): coef_today - coef_5d_ago > 0
  3. Momentum acceleration (动量加速): ret_20d > ret_prev_20d
  4. Headroom (未到顶): stock is below its 60d high

Architecture:
  - Inherits ``Backtest`` from ``scripts/backtest_strategies.py``
  - Overrides ``_score_universe()`` only — no emergency exit
  - Uses base class ``run()`` for clean trade structure
  - NO hard filters on trend signals — only basic quality filters

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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKTEST_SCRIPT = PROJECT_ROOT / "scripts" / "backtest_strategies.py"
F_STRATEGY_SCRIPT = PROJECT_ROOT / "strategies" / "f_strategy" / "run_backtest.py"

DATASETS = {
    "csi1000_5y": PROJECT_ROOT / "data_exports" / "tushare_20210329_20260327_csi1000_5y" / "csi1000_market_bundle_5y.csv",       # static sample (has survivorship bias)
    "csi1000_5y_pit": PROJECT_ROOT / "data_exports" / "tushare_20210329_20260327_csi1000_5y_pit" / "csi1000_market_bundle_5y_pit.csv",  # point-in-time (bias-free)
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
#  Trend Initiation Config
# ══════════════════════════════════════════════════════════════════

def get_trend_init_config(bt_module):
    """Return a BacktestConfig tuned for the Trend Initiation strategy."""
    return bt_module.BacktestConfig(
        name="TrendInit_趋势启动_振仓后coef加速",
        # Momentum — used by base class but scoring is done in _score_universe
        momentum_days=[60],
        momentum_weights=[1.0],
        subtract_short_momentum=False,
        # No reversal/regime — pure trend initiation
        use_reversal_factor=False,
        use_regime_switch=False,
        use_volatility_factor=False,
        use_size_factor=False,
        # Portfolio: 15 holdings, diversified, 20d rebalance
        top_n=15,
        hold_buffer_ratio=1.5,       # Top 22 buffer — less churn
        max_single_weight=0.08,      # 8% max — less concentrated
        rebalance_interval=20,       # Monthly rebalance — half the old cost
        # Risk control
        stop_loss_pct=-0.20,         # Wider stop — give trend time to develop
        use_halt=False,
        use_trend_filter=True,
        trend_ma_days=200,
        trend_reduce_pct=0.5,        # 50% in range (not 0% or 30%)
        # Cost
        commission=0.0003,
        stamp_tax=0.001,
        slippage=0.002,              # Lower — less concentrated buying
        # Filters — slightly looser to keep candidate pool large
        min_amount_20d=2e8,          # 20d avg amount >= 2亿
        min_price=5.0,
        min_list_days=250,
    )


# ══════════════════════════════════════════════════════════════════
#  Trend Initiation Backtest (override _score_universe only)
# ══════════════════════════════════════════════════════════════════

# Minimum history length: 60d for vol_60d + 7d for coef_delta + buffer.
MIN_HISTORY_DAYS = 65


def make_trend_init_backtest(bt_module):
    """Create a Backtest subclass with Trend Initiation scoring.

    Key difference from RS+Angle v2:
      - NO hard filters on trend signals (no coef≥0, RS>1, angle>0)
      - NO emergency exit — uses base class run()
      - NO RS/industry dependency — simpler, larger candidate pool
      - 4-factor scoring: coef_delta + vol_contraction + mom_accel + headroom
    """

    class TrendInitBacktest(bt_module.Backtest):

        def _score_universe(self, date: str) -> list:
            """Score stocks for trend initiation after consolidation.

            NO hard filters on signal dimensions — only basic quality
            filters (ST, price, listing days, liquidity, limit-up).

            Ranking factors:
              - coef_delta_5d:     30%  — coef acceleration (coef加速)
              - vol_contraction:   25%  — prior consolidation (振仓后压缩)
              - mom_accel:         25%  — momentum acceleration (动量加速)
              - headroom:          20%  — below 60d high (未到顶)
            """
            cfg = self.cfg
            if date not in self.trade_dates:
                return []

            scores = []
            for code, data in self._stock_data.items():
                if date not in data.index:
                    continue
                row = data.loc[date]

                # ── Basic quality filters (keep pool large) ──
                close = row["close"]
                if pd.isna(close) or close < cfg.min_price:
                    continue

                info = self._basic_map.get(code)
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

                hist = data[data.index <= date]
                if len(hist) < MIN_HISTORY_DAYS:
                    continue

                # Liquidity filter
                avg_amount = hist.tail(20)["amount"].mean() * 1000
                if avg_amount < cfg.min_amount_20d:
                    continue

                # No limit-up buys
                if row["pct_chg"] >= 9.5:
                    continue

                # ─────────────────────────────────────────────
                # Signal 1: Coef delta (5d) — trend acceleration
                # ─────────────────────────────────────────────
                if "ma20_angle_deg" not in hist.columns:
                    continue
                angles = hist["ma20_angle_deg"].tail(7).values.astype(float)
                if len(angles) < 7 or np.any(np.isnan(angles)):
                    continue
                coef_today = calc_strength_transition_coef(
                    float(angles[-1]), float(angles[-2])
                )
                coef_5d_ago = calc_strength_transition_coef(
                    float(angles[-6]), float(angles[-7])
                )
                coef_delta = coef_today - coef_5d_ago

                # ─────────────────────────────────────────────
                # Signal 2: Volatility contraction — prior consolidation
                #   vol_10d / vol_60d < 1 means recent calm after movement
                # ─────────────────────────────────────────────
                pct_chg_vals = hist["pct_chg"].values.astype(float)
                if len(pct_chg_vals) < 61:
                    continue
                vol_10d = np.nanstd(pct_chg_vals[-10:])
                vol_60d = np.nanstd(pct_chg_vals[-60:])
                if vol_60d < 1e-8:
                    continue
                vol_contraction = vol_10d / vol_60d

                # ─────────────────────────────────────────────
                # Signal 3: Momentum acceleration — ret_20d vs ret_prev_20d
                #   Positive means momentum is speeding up
                # ─────────────────────────────────────────────
                closes = hist["close"].values.astype(float)
                if len(closes) < 41:
                    continue
                ret_20d = closes[-1] / closes[-21] - 1
                ret_prev_20d = closes[-21] / closes[-41] - 1
                mom_accel = ret_20d - ret_prev_20d

                # ─────────────────────────────────────────────
                # Signal 4: Headroom — distance from 60d high
                #   Below 60d high = more room to run before topping
                # ─────────────────────────────────────────────
                max_60d = float(np.nanmax(closes[-60:]))
                headroom = closes[-1] / max_60d if max_60d > 0 else 1.0

                scores.append((
                    code, coef_delta, vol_contraction, mom_accel, headroom,
                ))

            if not scores:
                return []

            # ── Rank composition ──
            df = pd.DataFrame(scores, columns=[
                "code", "coef_delta", "vol_contraction", "mom_accel", "headroom",
            ])

            # Per-factor rank (lower rank number = better)
            df["cd_rank"] = df["coef_delta"].rank(ascending=False)       # higher delta = better
            df["vc_rank"] = df["vol_contraction"].rank(ascending=True)   # lower ratio = more consolidation
            df["ma_rank"] = df["mom_accel"].rank(ascending=False)        # higher accel = better
            df["hr_rank"] = df["headroom"].rank(ascending=True)          # further from high = more room

            # Weighted composite rank
            # coef_delta 30%, vol_contraction 25%, mom_accel 25%, headroom 20%
            df["composite_rank"] = (
                0.30 * df["cd_rank"]
                + 0.25 * df["vc_rank"]
                + 0.25 * df["ma_rank"]
                + 0.20 * df["hr_rank"]
            )

            df = df.sort_values("composite_rank").reset_index(drop=True)
            return list(zip(df["code"], df["composite_rank"]))

    return TrendInitBacktest


# ══════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Trend Initiation (趋势启动) strategy backtest")
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

    cfg = get_trend_init_config(bt_module)
    TrendInitBT = make_trend_init_backtest(bt_module)

    print(f"\n{'='*60}")
    print(f"Trend Initiation Strategy Backtest (趋势启动)")
    print(f"Dataset: {args.dataset} ({len(trade_dates)} trade days)")
    print(f"{'='*60}")

    t0 = time.time()
    bt = TrendInitBT(cfg, daily, idx, basic, trade_dates)
    result = bt.run(start_offset=250)
    result["elapsed_sec"] = round(time.time() - t0, 1)
    result["trend_filter_index_code"] = args.trend_index_code

    # Save
    suffix = f"_{args.dataset}"
    out_path = PROJECT_ROOT / "backtest" / f"strategy_trend_init{suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"data_file": str(data_path), "result": result}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nWROTE {out_path}")


if __name__ == "__main__":
    main()
