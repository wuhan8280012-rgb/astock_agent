#!/usr/bin/env python3
"""Experiments for MA20 strength-transition coefficient on F strategy."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/wuhan/project/stock_agent/new")

from strategies.f_strategy.run_backtest import DATASETS, load_backtest_module, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = PROJECT_ROOT / "backtest" / "strategy_f_transition_coef_experiments_csi1000_5y.json"


def transition_coef(a0: float, a1: float) -> float:
    base = np.tanh(a0 / 10.0)
    turn = np.tanh((a0 - a1) / 5.0)
    coef = 0.7 * base + 0.3 * turn
    return float(np.clip(coef, -1.0, 1.0))


def make_transition_backtest(module, mode: str):
    class TransitionBacktest(module.Backtest):
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

                closes = hist["close"].values.astype(float)
                mom_score = 0.0
                for lb, w in zip(cfg.momentum_days, cfg.momentum_weights):
                    if len(closes) >= lb + 1:
                        mom_score += (closes[-1] / closes[-(lb + 1)] - 1) * w
                    else:
                        mom_score = np.nan
                        break
                if np.isnan(mom_score):
                    continue

                vol_component = 0.0
                if cfg.use_volatility_factor and len(closes) >= cfg.volatility_days + 1:
                    rets = np.diff(closes[-cfg.volatility_days - 1 :]) / closes[-cfg.volatility_days - 1 : -1]
                    vol = np.std(rets)
                    if vol > 0:
                        vol_component = -vol

                size_component = 0.0
                if cfg.use_size_factor:
                    circ_mv = row.get("circ_mv", None)
                    if circ_mv and not pd.isna(circ_mv) and circ_mv > 0:
                        size_component = -np.log(circ_mv)

                angles = hist["ma20_angle_deg"].tail(2).values.astype(float) if "ma20_angle_deg" in hist.columns else np.array([])
                if len(angles) < 2 or np.isnan(angles[-1]) or np.isnan(angles[-2]):
                    continue
                coef = transition_coef(float(angles[-1]), float(angles[-2]))

                if mode in {"gate", "gate_boost"} and coef < 0:
                    continue

                scores.append((code, mom_score, vol_component, size_component, coef))

            if not scores:
                return []

            df = pd.DataFrame(scores, columns=["code", "mom", "vol", "size", "coef"])
            df["mom_rank"] = df["mom"].rank(ascending=False)
            df["vol_rank"] = df["vol"].rank(ascending=False)
            df["size_rank"] = df["size"].rank(ascending=False)
            df["coef_rank"] = df["coef"].rank(ascending=False)

            mom_weight = 1.0
            vol_w = cfg.volatility_weight if cfg.use_volatility_factor else 0
            size_w = cfg.size_weight if cfg.use_size_factor else 0
            coef_w = 0.2 if mode in {"boost", "gate_boost"} else 0.0
            total_w = mom_weight + vol_w + size_w + coef_w

            df["composite_rank"] = (
                (mom_weight / total_w) * df["mom_rank"]
                + (vol_w / total_w) * df["vol_rank"]
                + (size_w / total_w) * df["size_rank"]
                + (coef_w / total_w) * df["coef_rank"]
            )
            df = df.sort_values("composite_rank").reset_index(drop=True)
            return list(zip(df["code"], df["composite_rank"]))

    return TransitionBacktest


def main():
    module = load_backtest_module()
    daily, idx, basic, trade_dates = load_dataset(DATASETS["csi1000_5y"], module)
    cfg = [s for s in module.get_strategies() if s.name == "F_三因子+趋势过滤"][0]

    experiments = [
        ("baseline_ma20_ge_0", None, 0.0),
        ("transition_gate", "gate", None),
        ("transition_boost", "boost", None),
        ("transition_gate_boost", "gate_boost", None),
    ]

    results = []
    for name, mode, min_angle in experiments:
        if mode is None:
            bt_cls = make_angle_filtered_backtest(module, min_angle)
        else:
            bt_cls = make_transition_backtest(module, mode)
        t0 = time.time()
        bt = bt_cls(cfg, daily, idx, basic, trade_dates)
        result = bt.run(start_offset=250)
        result["elapsed_sec"] = round(time.time() - t0, 1)
        results.append({"experiment": name, **result})
        print(name, result["annual_return_pct"], result["sharpe"], result["max_drawdown_pct"], flush=True)

    OUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WROTE {OUT_PATH}")


if __name__ == "__main__":
    from strategies.f_strategy.run_backtest import make_angle_filtered_backtest

    main()
