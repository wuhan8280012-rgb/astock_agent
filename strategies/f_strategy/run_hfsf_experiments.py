#!/usr/bin/env python3
"""Exploratory HFSF experiments on top of the current F strategy.

Note: HFSF fields in the 5y bundle are latest-available snapshots, not true
historical-as-of fundamentals. Results here are exploratory and should not be
treated as production-grade long-horizon evidence.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKTEST_SCRIPT = PROJECT_ROOT / "scripts" / "backtest_strategies.py"
DATA_PATH = PROJECT_ROOT / "data_exports" / "tushare_20210329_20260327_csi1000_5y" / "csi1000_market_bundle_5y.csv"
OUT_PATH = PROJECT_ROOT / "backtest" / "strategy_f_hfsf_experiments_csi1000_5y.json"
TREND_INDEX_CODE = "000001.SH"


def load_module():
    spec = importlib.util.spec_from_file_location("backtest_strategies", BACKTEST_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_dataset(module):
    from run_backtest import load_dataset as load_f_dataset  # type: ignore

    daily, idx, basic, trade_dates = load_f_dataset(DATA_PATH, module, trend_index_code=TREND_INDEX_CODE)
    raw = pd.read_csv(DATA_PATH, low_memory=False)
    extra_cols = ["ts_code", "trade_date", "hfsf_score", "hfsf_signal"]
    extra = raw[raw["data_type"] == "daily"][extra_cols].copy()
    extra["trade_date"] = extra["trade_date"].astype(str)
    extra["hfsf_score"] = pd.to_numeric(extra["hfsf_score"], errors="coerce")
    daily = daily.merge(extra, on=["ts_code", "trade_date"], how="left")
    return daily, idx, basic, trade_dates


def calc_strength_transition_coef(a0: float, a1: float) -> float:
    base = np.tanh(a0 / 10.0)
    turn = np.tanh((a0 - a1) / 5.0)
    return float(np.clip(0.7 * base + 0.3 * turn, -1.0, 1.0))


def make_hfsf_backtest(module, gate_quantile=None, min_hfsf_score=None, hfsf_rank_weight: float = 0.0):
    class HFSFBacktest(module.Backtest):
        def __init__(self, cfg, daily, idx, basic, trade_dates):
            super().__init__(cfg, daily, idx, basic, trade_dates)
            stock_scores = (
                daily[["ts_code", "hfsf_score"]]
                .dropna()
                .drop_duplicates("ts_code")
                .set_index("ts_code")["hfsf_score"]
            )
            self._hfsf_map = stock_scores.to_dict()
            vals = pd.Series(list(self._hfsf_map.values())).dropna()
            self._gate_cut = float(vals.quantile(gate_quantile)) if gate_quantile is not None and not vals.empty else None

        def _score_universe(self, date: str):
            cfg = self.cfg
            scores = []

            for code, data in self._stock_data.items():
                if date not in data.index:
                    continue
                row = data.loc[date]

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
                if len(hist) < max(cfg.momentum_days[0], 60) + 5:
                    continue

                avg_amount = hist.tail(20)["amount"].mean() * 1000
                if avg_amount < cfg.min_amount_20d:
                    continue

                if row["pct_chg"] >= 9.5:
                    continue

                if "ma20_angle_deg" not in hist.columns:
                    continue
                angles = hist["ma20_angle_deg"].tail(2).values.astype(float)
                if len(angles) < 2 or np.isnan(angles[-1]) or np.isnan(angles[-2]):
                    continue
                if calc_strength_transition_coef(float(angles[-1]), float(angles[-2])) < -0.1:
                    continue

                hfsf_score = self._hfsf_map.get(code, np.nan)
                if min_hfsf_score is not None and (pd.isna(hfsf_score) or hfsf_score < min_hfsf_score):
                    continue
                if self._gate_cut is not None and (pd.isna(hfsf_score) or hfsf_score < self._gate_cut):
                    continue

                closes = hist["close"].values.astype(float)
                mom_score = 0.0
                for lb, w in zip(cfg.momentum_days, cfg.momentum_weights):
                    if len(closes) >= lb + 1:
                        ret = closes[-1] / closes[-(lb + 1)] - 1
                        mom_score += ret * w
                    else:
                        mom_score = np.nan
                        break
                if np.isnan(mom_score):
                    continue

                vol_component = 0.0
                if cfg.use_volatility_factor and len(closes) >= cfg.volatility_days + 1:
                    rets = np.diff(closes[-cfg.volatility_days - 1:]) / closes[-cfg.volatility_days - 1:-1]
                    vol = np.std(rets)
                    if vol > 0:
                        vol_component = -vol

                size_component = 0.0
                if cfg.use_size_factor:
                    circ_mv = row.get("circ_mv", None)
                    if circ_mv and not pd.isna(circ_mv) and circ_mv > 0:
                        size_component = -np.log(circ_mv)

                scores.append((code, mom_score, vol_component, size_component, hfsf_score))

            if not scores:
                return []

            df = pd.DataFrame(scores, columns=["code", "mom", "vol", "size", "hfsf"])
            df["mom_rank"] = df["mom"].rank(ascending=False)
            df["vol_rank"] = df["vol"].rank(ascending=False)
            df["size_rank"] = df["size"].rank(ascending=False)
            df["hfsf_rank"] = df["hfsf"].rank(ascending=False, na_option="bottom")

            mom_weight = 1.0
            vol_w = cfg.volatility_weight if cfg.use_volatility_factor else 0.0
            size_w = cfg.size_weight if cfg.use_size_factor else 0.0
            total_w = mom_weight + vol_w + size_w + hfsf_rank_weight

            df["composite_rank"] = (
                (mom_weight / total_w) * df["mom_rank"] +
                (vol_w / total_w) * df["vol_rank"] +
                (size_w / total_w) * df["size_rank"] +
                (hfsf_rank_weight / total_w) * df["hfsf_rank"]
            )
            df = df.sort_values("composite_rank").reset_index(drop=True)
            return list(zip(df["code"], df["composite_rank"]))

    return HFSFBacktest


def run_experiment(name: str, backtest_cls, module, daily, idx, basic, trade_dates):
    cfg = [s for s in module.get_strategies() if s.name == "F_三因子+趋势过滤"][0]
    bt = backtest_cls(cfg, daily, idx, basic, trade_dates)
    result = bt.run(start_offset=250)
    return {
        "name": name,
        "total_return_pct": result["total_return_pct"],
        "annual_return_pct": result["annual_return_pct"],
        "sharpe": result["sharpe"],
        "max_drawdown_pct": result["max_drawdown_pct"],
        "benchmark_return_pct": result["benchmark_return_pct"],
        "excess_return_pct": result["excess_return_pct"],
        "total_trades": result["total_trades"],
        "rebalance_count": result["rebalance_count"],
    }


def main():
    module = load_module()
    daily, idx, basic, trade_dates = load_dataset(module)

    experiments = [
        ("baseline", make_hfsf_backtest(module, gate_quantile=None, min_hfsf_score=None, hfsf_rank_weight=0.0)),
        ("hfsf_gate_top50pct", make_hfsf_backtest(module, gate_quantile=0.50, min_hfsf_score=None, hfsf_rank_weight=0.0)),
        ("hfsf_gate_top20pct", make_hfsf_backtest(module, gate_quantile=0.80, min_hfsf_score=None, hfsf_rank_weight=0.0)),
        ("hfsf_rank_boost_0_15", make_hfsf_backtest(module, gate_quantile=None, min_hfsf_score=None, hfsf_rank_weight=0.15)),
    ]

    results = [run_experiment(name, cls, module, daily, idx, basic, trade_dates) for name, cls in experiments]
    payload = {
        "note": "Exploratory only. HFSF uses latest snapshot fields and has forward-looking bias over 5y.",
        "trend_filter_index_code": TREND_INDEX_CODE,
        "data_file": str(DATA_PATH),
        "results": results,
    }
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"WROTE {OUT_PATH}")


if __name__ == "__main__":
    main()
