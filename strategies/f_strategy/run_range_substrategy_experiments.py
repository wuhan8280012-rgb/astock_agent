#!/usr/bin/env python3
"""Test dedicated RANGE sub-strategies on top of the current F baseline."""

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
OUT_PATH = PROJECT_ROOT / "backtest" / "strategy_f_range_substrategy_experiments_csi1000_5y.json"


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
        annual_return = ((1.0 + total_return) ** (252.0 / len(grp)) - 1.0) * 100.0 if len(grp) > 0 else 0.0
        regime_summary[regime] = {
            "days": int(len(grp)),
            "total_return_pct": round(total_return * 100.0, 2),
            "annual_return_pct": round(annual_return, 2),
            "avg_position_pct": round(grp["position_pct"].mean() * 100.0, 2),
        }
        annual_matrix[regime] = round(annual_return, 2)
    return regime_summary, annual_matrix


def make_range_substrategy_backtest(module, variant: str):
    Base = make_filtered_backtest(module, min_transition_coef=-0.1)

    class RangeBacktest(Base):
        def _passes_common_filters(self, code: str, data: pd.DataFrame, date: str):
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
            coef = calc_strength_transition_coef(float(angles[-1]), float(angles[-2]))
            if coef < -0.1:
                return None

            closes = hist["close"].values.astype(float)
            return row, hist, closes

        def _rank_from_components(self, df: pd.DataFrame):
            if df.empty:
                return []
            total_w = df.attrs["mom_weight"] + df.attrs["vol_weight"] + df.attrs["size_weight"]
            df["mom_rank"] = df["mom"].rank(ascending=False)
            df["vol_rank"] = df["vol"].rank(ascending=False)
            df["size_rank"] = df["size"].rank(ascending=False)
            df["composite_rank"] = (
                (df.attrs["mom_weight"] / total_w) * df["mom_rank"] +
                (df.attrs["vol_weight"] / total_w) * df["vol_rank"] +
                (df.attrs["size_weight"] / total_w) * df["size_rank"]
            )
            df = df.sort_values("composite_rank").reset_index(drop=True)
            return list(zip(df["code"], df["composite_rank"]))

        def _score_range_universe(self, date: str):
            scores = []
            for code, data in self._stock_data.items():
                passed = self._passes_common_filters(code, data, date)
                if passed is None:
                    continue
                row, hist, closes = passed

                vol60 = 0.0
                rets60 = np.diff(closes[-61:]) / closes[-61:-1]
                if len(rets60) > 0:
                    vol60 = np.std(rets60)
                vol20 = 0.0
                rets20 = np.diff(closes[-21:]) / closes[-21:-1]
                if len(rets20) > 0:
                    vol20 = np.std(rets20)

                circ_mv = row.get("circ_mv", np.nan)
                size_component = -np.log(circ_mv) if pd.notna(circ_mv) and circ_mv > 0 else 0.0

                if variant == "range_vol_defensive":
                    mom = closes[-1] / closes[-61] - 1
                    vol_component = -vol60
                    size = 0.0
                    scores.append({"code": code, "mom": mom, "vol": vol_component, "size": size})
                    continue

                if variant == "range_skip_short":
                    mom60 = closes[-1] / closes[-61] - 1
                    short20 = closes[-1] / closes[-21] - 1
                    mom = mom60 - short20
                    vol_component = -vol60
                    size = 0.0
                    scores.append({"code": code, "mom": mom, "vol": vol_component, "size": size})
                    continue

                if variant == "range_reversal_lowvol":
                    rev20 = -(closes[-1] / closes[-21] - 1)
                    vol_component = -vol20
                    size = 0.0
                    scores.append({"code": code, "mom": rev20, "vol": vol_component, "size": size})
                    continue

                if variant == "range_dual_momentum":
                    mom20 = closes[-1] / closes[-21] - 1
                    mom60 = closes[-1] / closes[-61] - 1
                    mom = 0.4 * mom20 + 0.6 * mom60
                    vol_component = -vol20
                    size = 0.0
                    scores.append({"code": code, "mom": mom, "vol": vol_component, "size": size})
                    continue

                raise ValueError(f"Unknown variant: {variant}")

            if not scores:
                return []

            if variant == "range_vol_defensive":
                mom_weight, vol_weight, size_weight = 1.0, 0.60, 0.0
            elif variant == "range_skip_short":
                mom_weight, vol_weight, size_weight = 1.0, 0.50, 0.0
            elif variant == "range_reversal_lowvol":
                mom_weight, vol_weight, size_weight = 1.0, 0.80, 0.0
            elif variant == "range_dual_momentum":
                mom_weight, vol_weight, size_weight = 1.0, 0.50, 0.0
            else:
                mom_weight, vol_weight, size_weight = 1.0, 0.25, 0.2

            df = pd.DataFrame(scores)
            df.attrs["mom_weight"] = mom_weight
            df.attrs["vol_weight"] = vol_weight
            df.attrs["size_weight"] = size_weight
            return self._rank_from_components(df)

        def _score_universe(self, date: str):
            trend_state = self._get_trend_state(date) if self.cfg.use_trend_filter else "BULL"
            if trend_state != "RANGE":
                return super()._score_universe(date)
            return self._score_range_universe(date)

    return RangeBacktest


def run_variant(module, daily, idx, basic, trade_dates, label: str, bt_cls):
    cfg = next(s for s in module.get_strategies() if s.name == "F_三因子+趋势过滤")
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


def main():
    module = load_backtest_module()
    data_path = DATASETS["csi1000_5y"]
    daily, idx, basic, trade_dates = load_dataset(data_path, module, trend_index_code=DEFAULT_TREND_INDEX_CODE)

    experiments = [
        ("baseline", make_filtered_backtest(module, min_transition_coef=-0.1), "Current main F baseline"),
        ("range_vol_defensive", make_range_substrategy_backtest(module, "range_vol_defensive"), "RANGE: 60d momentum + stronger low-vol, no size"),
        ("range_skip_short", make_range_substrategy_backtest(module, "range_skip_short"), "RANGE: 60d minus 20d hot money + low-vol"),
        ("range_reversal_lowvol", make_range_substrategy_backtest(module, "range_reversal_lowvol"), "RANGE: 20d reversal + low-vol"),
        ("range_dual_momentum", make_range_substrategy_backtest(module, "range_dual_momentum"), "RANGE: 20/60 composite momentum + low-vol"),
    ]

    runs = []
    for label, cls, desc in experiments:
        print(f"RUN {label}", flush=True)
        run = run_variant(module, daily, idx, basic, trade_dates, label, cls)
        run["description"] = desc
        runs.append(run)
        r = run["result"]
        print(f"DONE {label} total={r['total_return_pct']} annual={r['annual_return_pct']} sharpe={r['sharpe']} mdd={r['max_drawdown_pct']}", flush=True)

    payload = {
        "data_file": str(data_path),
        "trend_filter_index_code": DEFAULT_TREND_INDEX_CODE,
        "strategy": "F + strength_transition_coef >= -0.1",
        "note": "Only RANGE scoring logic changes; BULL/BEAR remain on the current official F baseline.",
        "experiments": runs,
    }
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT_PATH)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
