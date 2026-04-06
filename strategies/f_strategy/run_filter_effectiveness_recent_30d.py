#!/usr/bin/env python3
"""Test candidate filters on top of current F strategy over recent 30 trading days."""

from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tushare as ts

from run_backtest import DATASETS, calc_strength_transition_coef, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKTEST_SCRIPT = PROJECT_ROOT / "scripts" / "backtest_strategies.py"
OUT_PATH = PROJECT_ROOT / "backtest" / "strategy_f_filter_effectiveness_recent_30d.json"
TREND_INDEX_CODE = "000001.SH"
LATEST_PERIODS = [
    "20250930",
    "20250630",
    "20250331",
    "20241231",
    "20240930",
    "20240630",
    "20240331",
]


def load_module():
    spec = importlib.util.spec_from_file_location("backtest_strategies", BACKTEST_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_token() -> str:
    env_path = PROJECT_ROOT / "config" / ".env"
    if os.environ.get("TUSHARE_TOKEN"):
        return os.environ["TUSHARE_TOKEN"].strip()
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("TUSHARE_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("TUSHARE_TOKEN not found")


def fetch_with_retry(fetcher, **kwargs):
    delay = 0.5
    last_err = None
    for _ in range(5):
        try:
            return fetcher(**kwargs)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"API failed {fetcher} {kwargs}: {last_err}")


def fetch_latest_fundamental_maps(pool_codes: set[str]) -> pd.DataFrame:
    token = load_token()
    ts.set_token(token)
    pro = ts.pro_api(token)

    frames = []
    for period in LATEST_PERIODS:
        df = fetch_with_retry(
            pro.fina_indicator_vip,
            period=period,
            fields="ts_code,ann_date,end_date,q_sales_yoy,netprofit_yoy",
        )
        if df is None or df.empty:
            continue
        df["ts_code"] = df["ts_code"].astype(str)
        df = df[df["ts_code"].isin(pool_codes)].copy()
        frames.append(df)
        time.sleep(0.1)
    latest = pd.concat(frames, ignore_index=True)
    latest["ann_date"] = latest["ann_date"].fillna("").astype(str)
    latest["end_date"] = latest["end_date"].fillna("").astype(str)
    latest = latest.sort_values(["ts_code", "ann_date", "end_date"], ascending=[True, False, False])
    latest = latest.drop_duplicates("ts_code", keep="first")
    latest["q_sales_yoy"] = pd.to_numeric(latest["q_sales_yoy"], errors="coerce")
    latest["netprofit_yoy"] = pd.to_numeric(latest["netprofit_yoy"], errors="coerce")

    annual_frames = []
    for period in ["20241231", "20231231"]:
        df = fetch_with_retry(pro.income_vip, period=period, fields="ts_code,end_date,n_income_attr_p")
        if df is None or df.empty:
            continue
        df["ts_code"] = df["ts_code"].astype(str)
        df = df[df["ts_code"].isin(pool_codes)].copy()
        annual_frames.append(df)
        time.sleep(0.1)
    annual = pd.concat(annual_frames, ignore_index=True)
    annual["n_income_attr_p"] = pd.to_numeric(annual["n_income_attr_p"], errors="coerce")
    annual = annual.pivot_table(index="ts_code", columns="end_date", values="n_income_attr_p", aggfunc="first").reset_index()
    annual["netprofit_pos_2y"] = (
        annual.get("20241231", pd.Series(index=annual.index)).gt(0)
        & annual.get("20231231", pd.Series(index=annual.index)).gt(0)
    )

    merged = latest.merge(annual[["ts_code", "netprofit_pos_2y"]], on="ts_code", how="left")
    merged["netprofit_pos_2y"] = merged["netprofit_pos_2y"].fillna(False)
    return merged[["ts_code", "q_sales_yoy", "netprofit_yoy", "netprofit_pos_2y"]]


def enrich_daily_features(daily: pd.DataFrame, idx: pd.DataFrame, raw_bundle: pd.DataFrame) -> pd.DataFrame:
    extra = raw_bundle[raw_bundle["data_type"] == "daily"][["ts_code", "trade_date", "sw_l1_name", "hfsf_roe"]].copy()
    extra["trade_date"] = extra["trade_date"].astype(str)
    extra["hfsf_roe"] = pd.to_numeric(extra["hfsf_roe"], errors="coerce")
    daily = daily.merge(extra, on=["ts_code", "trade_date"], how="left")

    pool_codes = set(daily["ts_code"].unique())
    funda = fetch_latest_fundamental_maps(pool_codes)
    daily = daily.merge(funda, on="ts_code", how="left")

    daily = daily.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    grouped = daily.groupby("ts_code", group_keys=False)
    daily["ret20"] = grouped["close"].transform(lambda s: s / s.shift(20) - 1)
    daily["ret60"] = grouped["close"].transform(lambda s: s / s.shift(60) - 1)
    daily["ma50"] = grouped["close"].transform(lambda s: s.rolling(50).mean())
    daily["ma150"] = grouped["close"].transform(lambda s: s.rolling(150).mean())
    daily["high252"] = grouped["close"].transform(lambda s: s.rolling(252).max())
    roll_max_126 = grouped["close"].transform(lambda s: s.rolling(126).max())
    roll_min_126 = grouped["close"].transform(lambda s: s.rolling(126).min())
    daily["amp126"] = roll_max_126 / roll_min_126 - 1
    daily["rs60_pct"] = daily.groupby("trade_date")["ret60"].rank(pct=True)

    ind = (
        daily.dropna(subset=["sw_l1_name"])
        .groupby(["trade_date", "sw_l1_name"], as_index=False)["close"]
        .mean()
        .sort_values(["sw_l1_name", "trade_date"])
    )
    ind["ind_ret20"] = ind.groupby("sw_l1_name")["close"].transform(lambda s: s / s.shift(20) - 1)
    ind["ind_ret60"] = ind.groupby("sw_l1_name")["close"].transform(lambda s: s / s.shift(60) - 1)
    ind = ind.rename(columns={"close": "ind_close"})
    daily = daily.merge(ind[["trade_date", "sw_l1_name", "ind_ret20", "ind_ret60"]], on=["trade_date", "sw_l1_name"], how="left")

    idx = idx.sort_values("trade_date").reset_index(drop=True)
    idx["idx_ret20"] = idx["idx_close"] / idx["idx_close"].shift(20) - 1
    idx["idx_ret60"] = idx["idx_close"] / idx["idx_close"].shift(60) - 1
    daily = daily.merge(idx[["trade_date", "idx_ret20", "idx_ret60"]], on="trade_date", how="left")
    return daily


def make_filter_backtest(module, filter_name: str | None):
    class FilterBacktest(module.Backtest):
        def _passes_extra_filter(self, row, hist) -> bool:
            if filter_name is None:
                return True
            if filter_name == "quarterly_revenue_yoy_gt_20":
                return pd.notna(row.get("q_sales_yoy")) and float(row.get("q_sales_yoy")) > 20
            if filter_name == "quarterly_netprofit_yoy_gt_25":
                return pd.notna(row.get("netprofit_yoy")) and float(row.get("netprofit_yoy")) > 25
            if filter_name == "two_year_netprofit_positive":
                return bool(row.get("netprofit_pos_2y"))
            if filter_name == "roe_gt_10":
                return pd.notna(row.get("hfsf_roe")) and float(row.get("hfsf_roe")) > 10
            if filter_name == "rs60_top20pct":
                return pd.notna(row.get("rs60_pct")) and float(row.get("rs60_pct")) >= 0.8
            if filter_name == "industry_20_60_gt_market":
                return (
                    pd.notna(row.get("ind_ret20")) and pd.notna(row.get("ind_ret60"))
                    and pd.notna(row.get("idx_ret20")) and pd.notna(row.get("idx_ret60"))
                    and float(row.get("ind_ret20")) > float(row.get("idx_ret20"))
                    and float(row.get("ind_ret60")) > float(row.get("idx_ret60"))
                )
            if filter_name == "price_above_ma50_ma150":
                return (
                    pd.notna(row.get("ma50")) and pd.notna(row.get("ma150"))
                    and float(row["close"]) > float(row["ma50"])
                    and float(row["close"]) > float(row["ma150"])
                )
            if filter_name == "within_15pct_of_52w_high":
                return pd.notna(row.get("high252")) and float(row["close"]) >= 0.85 * float(row["high252"])
            if filter_name == "consolidation_amp_15_35":
                return pd.notna(row.get("amp126")) and 0.15 <= float(row["amp126"]) <= 0.35
            return True

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
                if len(hist) < max(cfg.momentum_days[0], 150) + 5:
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
                coef = calc_strength_transition_coef(float(angles[-1]), float(angles[-2]))
                if coef < -0.1:
                    continue

                if not self._passes_extra_filter(row, hist):
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

                scores.append((code, mom_score, vol_component, size_component))

            if not scores:
                return []

            df = pd.DataFrame(scores, columns=["code", "mom", "vol", "size"])
            df["mom_rank"] = df["mom"].rank(ascending=False)
            df["vol_rank"] = df["vol"].rank(ascending=False)
            df["size_rank"] = df["size"].rank(ascending=False)
            total_w = 1.0 + cfg.volatility_weight + cfg.size_weight
            df["composite_rank"] = (
                (1.0 / total_w) * df["mom_rank"]
                + (cfg.volatility_weight / total_w) * df["vol_rank"]
                + (cfg.size_weight / total_w) * df["size_rank"]
            )
            df = df.sort_values("composite_rank").reset_index(drop=True)
            return list(zip(df["code"], df["composite_rank"]))

    return FilterBacktest


def summarize(name: str, result: dict, baseline: dict | None = None) -> dict:
    item = {
        "name": name,
        "total_return_pct": result["total_return_pct"],
        "annual_return_pct": result["annual_return_pct"],
        "sharpe": result["sharpe"],
        "max_drawdown_pct": result["max_drawdown_pct"],
        "excess_return_pct": result["excess_return_pct"],
        "total_trades": result["total_trades"],
    }
    if baseline:
        item["delta_total_return_pct"] = round(result["total_return_pct"] - baseline["total_return_pct"], 2)
        item["delta_sharpe"] = round(result["sharpe"] - baseline["sharpe"], 2)
        item["delta_max_drawdown_pct"] = round(result["max_drawdown_pct"] - baseline["max_drawdown_pct"], 2)
    return item


def main():
    module = load_module()
    raw_bundle = pd.read_csv(DATASETS["csi1000_5y"], low_memory=False)
    daily, idx, basic, trade_dates = load_dataset(DATASETS["csi1000_5y"], module, trend_index_code=TREND_INDEX_CODE)
    daily = enrich_daily_features(daily, idx, raw_bundle)

    cfg = [s for s in module.get_strategies() if s.name == "F_三因子+趋势过滤"][0]
    start_offset = len(trade_dates) - 30

    filters = [
        None,
        "quarterly_revenue_yoy_gt_20",
        "quarterly_netprofit_yoy_gt_25",
        "two_year_netprofit_positive",
        "roe_gt_10",
        "rs60_top20pct",
        "industry_20_60_gt_market",
        "price_above_ma50_ma150",
        "within_15pct_of_52w_high",
        "consolidation_amp_15_35",
    ]

    results = []
    baseline_result = None
    for filter_name in filters:
        bt_cls = make_filter_backtest(module, filter_name)
        bt = bt_cls(cfg, daily.copy(), idx, basic, trade_dates)
        result = bt.run(start_offset=start_offset)
        if filter_name is None:
            baseline_result = result
            results.append(summarize("baseline", result))
        else:
            results.append(summarize(filter_name, result, baseline=baseline_result))

    payload = {
        "note": "Quarterly growth/ROE filters use latest-available snapshots, not historical-as-of financials.",
        "window_start_date": trade_dates[start_offset],
        "window_end_date": trade_dates[-1],
        "trend_filter_index_code": TREND_INDEX_CODE,
        "results": results,
    }
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"WROTE {OUT_PATH}")


if __name__ == "__main__":
    main()
