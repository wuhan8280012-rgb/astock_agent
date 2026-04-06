#!/usr/bin/env python3
"""Enrich the CSI1000 5y bundle with Henry Fundamental Safety Factor fields."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tushare as ts


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / "config" / ".env"
BUNDLE_PATH = PROJECT_ROOT / "data_exports" / "tushare_20210329_20260327_csi1000_5y" / "csi1000_market_bundle_5y.csv"
CACHE_PATH = PROJECT_ROOT / "data" / "hfsf_snapshot_cache.csv"
SUMMARY_PATH = PROJECT_ROOT / "data" / "hfsf_enrichment_summary.json"

LATEST_PERIODS = [
    "20250930",
    "20250630",
    "20250331",
    "20241231",
    "20240930",
    "20240630",
    "20240331",
]
ANNUAL_PERIODS = ["20241231", "20231231", "20221231", "20211231"]


def load_token() -> str:
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("TUSHARE_TOKEN="):
            token = line.split("=", 1)[1].strip()
            if token:
                return token
    raise RuntimeError("TUSHARE_TOKEN not found in .env")


def normalize_code(val: object) -> str:
    if pd.isna(val):
        return ""
    text = str(val).strip()
    return text[:-2] if text.endswith(".0") else text


def load_pool_context(bundle_path: Path) -> tuple[pd.DataFrame, str, list[str]]:
    cols = [
        "data_type",
        "ts_code",
        "trade_date",
        "sw_l1_name",
        "sw_l2_name",
        "sw_l3_name",
    ]
    raw = pd.read_csv(bundle_path, usecols=cols, low_memory=False)
    raw["ts_code"] = raw["ts_code"].map(normalize_code)
    stock_basic = raw[raw["data_type"] == "stock_basic"][["ts_code", "sw_l1_name", "sw_l2_name", "sw_l3_name"]].drop_duplicates("ts_code")
    stock_basic["peer_group"] = (
        stock_basic["sw_l3_name"]
        .fillna("")
        .replace("", pd.NA)
        .fillna(stock_basic["sw_l2_name"].fillna("").replace("", pd.NA))
        .fillna(stock_basic["sw_l1_name"].fillna("").replace("", pd.NA))
        .fillna("")
    )
    daily_dates = raw.loc[raw["data_type"] == "daily", "trade_date"].dropna().astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)
    latest_trade_date = daily_dates.max()
    month_end_dates = sorted(daily_dates.groupby(daily_dates.str.slice(0, 6)).max().tolist())
    return stock_basic, latest_trade_date, month_end_dates


def fetch_with_retry(fetcher, **kwargs) -> pd.DataFrame:
    delay = 0.5
    last_err = None
    for _ in range(5):
        try:
            df = fetcher(**kwargs)
            return df if df is not None else pd.DataFrame()
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"API call failed for {fetcher}: {kwargs} -> {last_err}")


def fetch_latest_period_snapshot(pro, api_name: str, periods: list[str], fields: str, pool_codes: set[str]) -> pd.DataFrame:
    frames = []
    fetcher = getattr(pro, api_name)
    for period in periods:
        df = fetch_with_retry(fetcher, period=period, fields=fields)
        if df.empty:
            continue
        df["ts_code"] = df["ts_code"].map(normalize_code)
        df = df[df["ts_code"].isin(pool_codes)].copy()
        if df.empty:
            continue
        frames.append(df)
        time.sleep(0.12)
    if not frames:
        return pd.DataFrame(columns=fields.split(","))
    merged = pd.concat(frames, ignore_index=True)
    for col in ["ann_date", "f_ann_date", "end_date"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna("").astype(str)
    sort_cols = ["ts_code"]
    ascending = [True]
    for col in ["ann_date", "end_date"]:
        if col in merged.columns:
            sort_cols.append(col)
            ascending.append(False)
    merged = merged.sort_values(sort_cols, ascending=ascending)
    return merged.drop_duplicates("ts_code", keep="first")


def fetch_annual_history(pro, api_name: str, periods: list[str], fields: str, pool_codes: set[str]) -> pd.DataFrame:
    frames = []
    fetcher = getattr(pro, api_name)
    for period in periods:
        df = fetch_with_retry(fetcher, period=period, fields=fields)
        if df.empty:
            continue
        df["ts_code"] = df["ts_code"].map(normalize_code)
        df = df[df["ts_code"].isin(pool_codes)].copy()
        if df.empty:
            continue
        frames.append(df)
        time.sleep(0.12)
    if not frames:
        return pd.DataFrame(columns=fields.split(","))
    return pd.concat(frames, ignore_index=True)


def fetch_latest_daily_basic(pro, latest_trade_date: str, pool_codes: set[str]) -> pd.DataFrame:
    df = fetch_with_retry(
        pro.daily_basic,
        trade_date=latest_trade_date,
        fields="ts_code,trade_date,close,pe_ttm,dv_ttm,total_mv",
    )
    df["ts_code"] = df["ts_code"].map(normalize_code)
    return df[df["ts_code"].isin(pool_codes)].copy()


def fetch_monthly_pe_history(pro, month_end_dates: list[str], pool_codes: set[str]) -> pd.DataFrame:
    frames = []
    total = len(month_end_dates)
    for idx, trade_date in enumerate(month_end_dates, start=1):
        df = fetch_with_retry(pro.daily_basic, trade_date=trade_date, fields="ts_code,trade_date,pe_ttm")
        if df.empty:
            continue
        df["ts_code"] = df["ts_code"].map(normalize_code)
        df = df[df["ts_code"].isin(pool_codes)].copy()
        if not df.empty:
            frames.append(df)
        if idx % 12 == 0 or idx == total:
            print(f"monthly PE progress {idx}/{total}", flush=True)
        time.sleep(0.08)
    if not frames:
        return pd.DataFrame(columns=["ts_code", "trade_date", "pe_ttm"])
    return pd.concat(frames, ignore_index=True)


def compute_cagr(series: pd.Series) -> float:
    values = [float(x) for x in series if pd.notna(x)]
    if len(values) < 2:
        return math.nan
    first = values[0]
    last = values[-1]
    periods = len(values) - 1
    if first <= 0 or last <= 0 or periods <= 0:
        return math.nan
    return (last / first) ** (1 / periods) - 1


def zscore(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    std = vals.std(skipna=True)
    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=series.index)
    return (vals - vals.mean(skipna=True)) / std


def build_hfsf_snapshot(stock_basic: pd.DataFrame, latest_trade_date: str, month_end_dates: list[str], pro) -> pd.DataFrame:
    pool_codes = set(stock_basic["ts_code"])

    latest_fina = fetch_latest_period_snapshot(
        pro,
        "fina_indicator_vip",
        LATEST_PERIODS,
        "ts_code,ann_date,end_date,roe,roic,grossprofit_margin,profit_dedt,fcff,debt_to_assets",
        pool_codes,
    )
    latest_income = fetch_latest_period_snapshot(
        pro,
        "income_vip",
        LATEST_PERIODS,
        "ts_code,ann_date,end_date,revenue,n_income_attr_p",
        pool_codes,
    )
    latest_cashflow = fetch_latest_period_snapshot(
        pro,
        "cashflow_vip",
        LATEST_PERIODS,
        "ts_code,ann_date,end_date,n_cashflow_act,free_cashflow,c_pay_acq_const_fiolta",
        pool_codes,
    )
    latest_bs = fetch_latest_period_snapshot(
        pro,
        "balancesheet_vip",
        LATEST_PERIODS,
        "ts_code,ann_date,end_date,total_assets,total_liab",
        pool_codes,
    )
    latest_daily_basic = fetch_latest_daily_basic(pro, latest_trade_date, pool_codes)

    annual_fina = fetch_annual_history(
        pro,
        "fina_indicator_vip",
        ANNUAL_PERIODS,
        "ts_code,end_date,grossprofit_margin",
        pool_codes,
    )
    annual_income = fetch_annual_history(
        pro,
        "income_vip",
        ANNUAL_PERIODS,
        "ts_code,end_date,revenue,n_income_attr_p",
        pool_codes,
    )
    annual_cf = fetch_annual_history(
        pro,
        "cashflow_vip",
        ANNUAL_PERIODS,
        "ts_code,end_date,free_cashflow,n_cashflow_act,c_pay_acq_const_fiolta",
        pool_codes,
    )
    monthly_pe = fetch_monthly_pe_history(pro, month_end_dates, pool_codes)

    snapshot = stock_basic[["ts_code", "peer_group"]].copy()
    for df, renames in [
        (
            latest_fina,
            {
                "ann_date": "hfsf_latest_ann_date",
                "end_date": "hfsf_latest_report_period",
                "roe": "hfsf_roe",
                "roic": "hfsf_roic",
                "grossprofit_margin": "hfsf_gross_margin_pct",
                "profit_dedt": "hfsf_profit_dedt",
                "fcff": "hfsf_fcff",
                "debt_to_assets": "hfsf_debt_to_assets_pct",
            },
        ),
        (
            latest_income,
            {
                "revenue": "hfsf_revenue",
                "n_income_attr_p": "hfsf_net_profit",
            },
        ),
        (
            latest_cashflow,
            {
                "n_cashflow_act": "hfsf_ocf",
                "free_cashflow": "hfsf_free_cashflow",
                "c_pay_acq_const_fiolta": "hfsf_capex",
            },
        ),
        (
            latest_bs,
            {
                "total_assets": "hfsf_total_assets",
                "total_liab": "hfsf_total_liab",
            },
        ),
        (
            latest_daily_basic,
            {
                "close": "hfsf_close",
                "pe_ttm": "hfsf_pe_ttm",
                "dv_ttm": "hfsf_dividend_yield_pct",
                "total_mv": "hfsf_total_mv_10k",
            },
        ),
    ]:
        keep_cols = ["ts_code"] + [c for c in renames if c in df.columns]
        tmp = df[keep_cols].rename(columns=renames)
        snapshot = snapshot.merge(tmp, on="ts_code", how="left")

    for col in [
        "hfsf_roe",
        "hfsf_roic",
        "hfsf_gross_margin_pct",
        "hfsf_profit_dedt",
        "hfsf_fcff",
        "hfsf_debt_to_assets_pct",
        "hfsf_revenue",
        "hfsf_net_profit",
        "hfsf_ocf",
        "hfsf_free_cashflow",
        "hfsf_capex",
        "hfsf_total_assets",
        "hfsf_total_liab",
        "hfsf_close",
        "hfsf_pe_ttm",
        "hfsf_dividend_yield_pct",
        "hfsf_total_mv_10k",
    ]:
        if col in snapshot.columns:
            snapshot[col] = pd.to_numeric(snapshot[col], errors="coerce")

    snapshot["hfsf_dividend_yield"] = snapshot["hfsf_dividend_yield_pct"] / 100.0
    snapshot["hfsf_pe_inverse"] = 1.0 / snapshot["hfsf_pe_ttm"].replace(0, np.nan)
    snapshot["hfsf_market_cap"] = snapshot["hfsf_total_mv_10k"] * 10000.0
    snapshot["hfsf_free_cashflow"] = snapshot["hfsf_free_cashflow"].where(
        snapshot["hfsf_free_cashflow"].notna(),
        snapshot["hfsf_ocf"] - snapshot["hfsf_capex"],
    )
    snapshot["hfsf_fcf_yield"] = snapshot["hfsf_free_cashflow"] / snapshot["hfsf_market_cap"]
    snapshot["hfsf_profit_dedt_ratio"] = snapshot["hfsf_profit_dedt"] / snapshot["hfsf_net_profit"].replace(0, np.nan)
    snapshot["hfsf_ocf_np_ratio"] = snapshot["hfsf_ocf"] / snapshot["hfsf_net_profit"].replace(0, np.nan)
    snapshot["hfsf_fcf_to_assets"] = snapshot["hfsf_free_cashflow"] / snapshot["hfsf_total_assets"].replace(0, np.nan)

    if not monthly_pe.empty:
        monthly_pe["pe_ttm"] = pd.to_numeric(monthly_pe["pe_ttm"], errors="coerce")
        pe_avg = monthly_pe.groupby("ts_code", as_index=False)["pe_ttm"].mean().rename(columns={"pe_ttm": "hfsf_pe_ttm_5y_monthly_avg"})
        snapshot = snapshot.merge(pe_avg, on="ts_code", how="left")
        snapshot["hfsf_pe_ttm_vs_5y_avg"] = snapshot["hfsf_pe_ttm"] / snapshot["hfsf_pe_ttm_5y_monthly_avg"].replace(0, np.nan)
    else:
        snapshot["hfsf_pe_ttm_5y_monthly_avg"] = np.nan
        snapshot["hfsf_pe_ttm_vs_5y_avg"] = np.nan

    if not annual_fina.empty:
        annual_fina["grossprofit_margin"] = pd.to_numeric(annual_fina["grossprofit_margin"], errors="coerce")
        gm_std = annual_fina.groupby("ts_code")["grossprofit_margin"].std().reset_index().rename(columns={"grossprofit_margin": "hfsf_gross_margin_3y_std"})
        snapshot = snapshot.merge(gm_std, on="ts_code", how="left")
    else:
        snapshot["hfsf_gross_margin_3y_std"] = np.nan

    if not annual_income.empty:
        annual_income["revenue"] = pd.to_numeric(annual_income["revenue"], errors="coerce")
        rev_cagr = annual_income.sort_values(["ts_code", "end_date"]).groupby("ts_code")["revenue"].apply(compute_cagr).reset_index().rename(columns={"revenue": "hfsf_revenue_3y_cagr"})
        snapshot = snapshot.merge(rev_cagr, on="ts_code", how="left")
    else:
        snapshot["hfsf_revenue_3y_cagr"] = np.nan

    if not annual_cf.empty:
        annual_cf["free_cashflow"] = pd.to_numeric(annual_cf["free_cashflow"], errors="coerce")
        annual_cf["c_pay_acq_const_fiolta"] = pd.to_numeric(annual_cf.get("c_pay_acq_const_fiolta"), errors="coerce")
        annual_cf["n_cashflow_act"] = pd.to_numeric(annual_cf.get("n_cashflow_act"), errors="coerce")
        annual_cf["free_cashflow"] = annual_cf["free_cashflow"].where(
            annual_cf["free_cashflow"].notna(),
            annual_cf["n_cashflow_act"] - annual_cf["c_pay_acq_const_fiolta"],
        )
        fcf_cagr = annual_cf.sort_values(["ts_code", "end_date"]).groupby("ts_code")["free_cashflow"].apply(compute_cagr).reset_index().rename(columns={"free_cashflow": "hfsf_fcf_3y_cagr"})
        snapshot = snapshot.merge(fcf_cagr, on="ts_code", how="left")
    else:
        snapshot["hfsf_fcf_3y_cagr"] = np.nan

    peer_pe = snapshot.groupby("peer_group", as_index=False)["hfsf_pe_ttm"].mean().rename(columns={"hfsf_pe_ttm": "hfsf_pe_ttm_peer_avg"})
    peer_gm = snapshot.groupby("peer_group", as_index=False)["hfsf_gross_margin_pct"].mean().rename(columns={"hfsf_gross_margin_pct": "hfsf_gross_margin_peer_avg"})
    peer_fcf_assets = snapshot.groupby("peer_group", as_index=False)["hfsf_fcf_to_assets"].mean().rename(columns={"hfsf_fcf_to_assets": "hfsf_fcf_to_assets_peer_avg"})
    snapshot = snapshot.merge(peer_pe, on="peer_group", how="left")
    snapshot = snapshot.merge(peer_gm, on="peer_group", how="left")
    snapshot = snapshot.merge(peer_fcf_assets, on="peer_group", how="left")
    snapshot["hfsf_pe_ttm_vs_peer"] = snapshot["hfsf_pe_ttm"] / snapshot["hfsf_pe_ttm_peer_avg"].replace(0, np.nan)
    snapshot["hfsf_gross_margin_vs_peer"] = snapshot["hfsf_gross_margin_pct"] - snapshot["hfsf_gross_margin_peer_avg"]

    asset_component = zscore(snapshot["hfsf_fcf_yield"].fillna(0) + snapshot["hfsf_dividend_yield"].fillna(0))
    valuation_component = zscore(snapshot["hfsf_pe_inverse"].fillna(0) - snapshot["hfsf_pe_ttm_vs_5y_avg"].fillna(1) - snapshot["hfsf_pe_ttm_vs_peer"].fillna(1))
    quality_component = zscore(snapshot["hfsf_roe"].fillna(0) + snapshot["hfsf_profit_dedt_ratio"].fillna(0))
    moat_component = zscore(snapshot["hfsf_roic"].fillna(0) + snapshot["hfsf_gross_margin_vs_peer"].fillna(0))
    cashflow_component = zscore(snapshot["hfsf_ocf_np_ratio"].fillna(0) + snapshot["hfsf_fcf_to_assets"].fillna(0))
    snapshot["hfsf_score"] = (
        0.25 * asset_component
        + 0.20 * valuation_component
        + 0.20 * quality_component
        + 0.20 * moat_component
        + 0.15 * cashflow_component
    )
    snapshot["hfsf_signal"] = np.where(
        snapshot["hfsf_score"] > 1.5,
        "strong_buy",
        np.where(snapshot["hfsf_score"] >= snapshot["hfsf_score"].quantile(0.8), "top_20pct", ""),
    )

    return snapshot


def enrich_bundle(bundle_path: Path, snapshot: pd.DataFrame) -> None:
    raw = pd.read_csv(bundle_path, low_memory=False)
    raw["ts_code"] = raw["ts_code"].map(normalize_code)

    new_cols = [c for c in snapshot.columns if c != "ts_code"]
    for col in new_cols:
        if col in raw.columns:
            raw = raw.drop(columns=[col])

    raw = raw.merge(snapshot, on="ts_code", how="left")
    mask = raw["data_type"].isin(["index_daily", "trade_cal"])
    text_cols = [c for c in new_cols if raw[c].dtype == object]
    num_cols = [c for c in new_cols if c not in text_cols]
    if text_cols:
        raw.loc[mask, text_cols] = ""
    if num_cols:
        raw.loc[mask, num_cols] = pd.NA

    cols = raw.columns.tolist()
    for col in new_cols:
        cols.remove(col)
    insert_at = cols.index("latest_roe") + 1 if "latest_roe" in cols else len(cols)
    cols[insert_at:insert_at] = new_cols
    raw = raw[cols]
    raw.to_csv(bundle_path, index=False)


def main() -> None:
    token = load_token()
    ts.set_token(token)
    pro = ts.pro_api(token)

    stock_basic, latest_trade_date, month_end_dates = load_pool_context(BUNDLE_PATH)
    snapshot = build_hfsf_snapshot(stock_basic, latest_trade_date, month_end_dates, pro)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_csv(CACHE_PATH, index=False)

    enrich_bundle(BUNDLE_PATH, snapshot)

    summary = {
        "bundle": str(BUNDLE_PATH),
        "cache": str(CACHE_PATH),
        "latest_trade_date": latest_trade_date,
        "stock_count": int(snapshot["ts_code"].nunique()),
        "fields_added": [c for c in snapshot.columns if c != "ts_code"],
        "non_null_counts": {c: int(snapshot[c].notna().sum()) for c in snapshot.columns if c != "ts_code"},
        "sample": snapshot.head(10).to_dict(orient="records"),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
