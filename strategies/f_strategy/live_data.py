#!/usr/bin/env python3
"""Incremental live-data extension for the F strategy runtime."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / "config" / ".env"


def load_env_token() -> str | None:
    if os.environ.get("TUSHARE_TOKEN"):
        token = os.environ["TUSHARE_TOKEN"].strip()
        if token:
            return token
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            if line.startswith("TUSHARE_TOKEN="):
                token = line.split("=", 1)[1].strip()
                if token:
                    return token
    return None


def _normalize_trade_date(value: str | None) -> str:
    if value is None:
        return datetime.now().strftime("%Y%m%d")
    text = str(value).strip().replace("-", "")
    if len(text) != 8 or not text.isdigit():
        raise ValueError(f"非法交易日格式: {value}")
    return text


def _merge_basic_info(basic: pd.DataFrame, fresh_basic: pd.DataFrame) -> pd.DataFrame:
    if fresh_basic is None or fresh_basic.empty:
        return basic

    keep_cols = ["ts_code", "name", "industry", "list_date"]
    fresh = fresh_basic[keep_cols].copy()
    fresh["ts_code"] = fresh["ts_code"].astype(str)
    fresh["list_date"] = fresh["list_date"].astype(str)

    current = basic[keep_cols].copy()
    current["ts_code"] = current["ts_code"].astype(str)
    current["list_date"] = current["list_date"].astype(str)

    merged = fresh.drop_duplicates("ts_code").set_index("ts_code").combine_first(
        current.drop_duplicates("ts_code").set_index("ts_code")
    )
    return merged.reset_index().sort_values("ts_code").reset_index(drop=True)


def _prepare_incremental_daily(
    daily_fresh: pd.DataFrame,
    daily_basic_fresh: pd.DataFrame,
    adj_fresh: pd.DataFrame,
    existing_daily: pd.DataFrame,
) -> pd.DataFrame:
    if daily_fresh is None or daily_fresh.empty:
        return pd.DataFrame(columns=existing_daily.columns)

    keep_cols = ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount", "pct_chg"]
    daily_new = daily_fresh[keep_cols].copy()
    daily_new["ts_code"] = daily_new["ts_code"].astype(str)
    daily_new["trade_date"] = daily_new["trade_date"].astype(str)

    for col in ["open", "high", "low", "close", "vol", "amount", "pct_chg"]:
        daily_new[col] = pd.to_numeric(daily_new[col], errors="coerce")

    if adj_fresh is not None and not adj_fresh.empty:
        adj = adj_fresh[["ts_code", "trade_date", "adj_factor"]].copy()
        adj["ts_code"] = adj["ts_code"].astype(str)
        adj["trade_date"] = adj["trade_date"].astype(str)
        adj["adj_factor"] = pd.to_numeric(adj["adj_factor"], errors="coerce")
        daily_new = daily_new.merge(adj, on=["ts_code", "trade_date"], how="left")
    else:
        daily_new["adj_factor"] = np.nan

    if daily_basic_fresh is not None and not daily_basic_fresh.empty:
        db = daily_basic_fresh[["ts_code", "trade_date", "total_mv", "circ_mv"]].copy()
        db["ts_code"] = db["ts_code"].astype(str)
        db["trade_date"] = db["trade_date"].astype(str)
        for col in ["total_mv", "circ_mv"]:
            db[col] = pd.to_numeric(db[col], errors="coerce")
        daily_new = daily_new.merge(db, on=["ts_code", "trade_date"], how="left")
    else:
        daily_new["total_mv"] = np.nan
        daily_new["circ_mv"] = np.nan

    daily_new["adj_factor"] = daily_new["adj_factor"].fillna(1.0)
    daily_new["adj_close"] = daily_new["close"] * daily_new["adj_factor"]

    for col in ["sw_l1_name", "sw_l1_ret20", "market_ret20", "sw_l1_excess20", "sw_l1_strength20_vs_market"]:
        if col in existing_daily.columns:
            daily_new[col] = np.nan

    if "sw_l1_name" in existing_daily.columns:
        sw_map = (
            existing_daily.dropna(subset=["sw_l1_name"])
            .sort_values(["ts_code", "trade_date"])
            .groupby("ts_code")["sw_l1_name"]
            .last()
            .to_dict()
        )
        daily_new["sw_l1_name"] = daily_new["ts_code"].map(sw_map)

    return daily_new


def _recompute_runtime_features(daily: pd.DataFrame, recompute_from_date: str | None = None) -> pd.DataFrame:
    daily = daily.sort_values(["ts_code", "trade_date"]).reset_index(drop=True).copy()
    update_mask = pd.Series(True, index=daily.index)
    if recompute_from_date:
        update_mask = daily["trade_date"].astype(str) > str(recompute_from_date)

    grouped_close = daily.groupby("ts_code", sort=False)["close"]
    ma10 = grouped_close.transform(lambda s: s.rolling(10, min_periods=10).mean())
    ma20 = grouped_close.transform(lambda s: s.rolling(20, min_periods=20).mean())
    ma10_prev = ma10.groupby(daily["ts_code"], sort=False).shift(1)
    ma20_prev = ma20.groupby(daily["ts_code"], sort=False).shift(1)

    amount_ma20 = daily.groupby("ts_code", sort=False)["amount"].transform(
        lambda s: s.rolling(20, min_periods=20).mean()
    )
    px_ma10_ratio = daily["close"] / ma10
    px_ma20_ratio = daily["close"] / ma20
    ma10_angle_deg = np.degrees(np.arctan((ma10 / ma10_prev - 1.0) * 100.0))
    ma20_angle_deg = np.degrees(np.arctan((ma20 / ma20_prev - 1.0) * 100.0))

    for col in ["px_ma10_ratio", "px_ma20_ratio", "ma10_angle_deg", "ma20_angle_deg", "amount_ma20"]:
        if col not in daily.columns:
            daily[col] = np.nan

    daily.loc[update_mask, "px_ma10_ratio"] = px_ma10_ratio.loc[update_mask]
    daily.loc[update_mask, "px_ma20_ratio"] = px_ma20_ratio.loc[update_mask]
    daily.loc[update_mask, "ma10_angle_deg"] = ma10_angle_deg.loc[update_mask]
    daily.loc[update_mask, "ma20_angle_deg"] = ma20_angle_deg.loc[update_mask]
    daily.loc[update_mask, "amount_ma20"] = amount_ma20.loc[update_mask]
    return daily


def _query_market_slice_by_dates(client, api_name: str, trade_dates: list[str], fields: str, pool_codes: list[str]) -> pd.DataFrame:
    if not trade_dates:
        return pd.DataFrame()

    pool = set(pool_codes)
    frames = []
    for trade_date in trade_dates:
        df = client.query_by_date(api_name, trade_date, fields=fields)
        if df is None or df.empty:
            continue
        df = df.copy()
        if "ts_code" in df.columns:
            df["ts_code"] = df["ts_code"].astype(str)
            df = df[df["ts_code"].isin(pool)]
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def extend_dataset_with_live_data(
    daily: pd.DataFrame,
    idx: pd.DataFrame,
    basic: pd.DataFrame,
    trade_dates: list[str],
    enrich_industry_strength20_features,
    requested_trade_date: str | None = None,
    trend_index_code: str = "000001.SH",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], dict]:
    last_local_trade_date = trade_dates[-1]
    requested_trade_date = _normalize_trade_date(requested_trade_date)
    report = {
        "enabled": False,
        "used_live_data": False,
        "last_local_trade_date": last_local_trade_date,
        "requested_trade_date": requested_trade_date,
        "latest_trade_date": last_local_trade_date,
        "fetched_trade_dates": [],
        "message": "",
    }

    if requested_trade_date <= last_local_trade_date:
        report["message"] = "请求日期不晚于本地样本，无需在线补数。"
        return daily, idx, basic, trade_dates, report

    token = load_env_token()
    if not token:
        report["message"] = "未配置 TUSHARE_TOKEN，无法在线补数。"
        return daily, idx, basic, trade_dates, report

    report["enabled"] = True

    try:
        from data_pipeline.tushare_client import TushareClient
    except Exception as exc:
        report["message"] = f"Tushare client 不可用: {exc}"
        return daily, idx, basic, trade_dates, report

    try:
        client = TushareClient(token=token)
        cal_start = (datetime.strptime(last_local_trade_date, "%Y%m%d") - timedelta(days=7)).strftime("%Y%m%d")
        open_dates = client.get_trade_calendar(cal_start, requested_trade_date)
        fetch_dates = [d for d in open_dates if d > last_local_trade_date]
        if not fetch_dates:
            report["message"] = "交易日历中没有需要补的开放日。"
            return daily, idx, basic, trade_dates, report

        pool_codes = basic["ts_code"].dropna().astype(str).unique().tolist()
        daily_fresh = _query_market_slice_by_dates(
            client,
            "daily",
            fetch_dates,
            fields="ts_code,trade_date,open,high,low,close,vol,amount,pct_chg",
            pool_codes=pool_codes,
        )
        if daily_fresh is None or daily_fresh.empty:
            report["message"] = "在线接口未返回新的个股日线。"
            return daily, idx, basic, trade_dates, report

        daily_fresh["trade_date"] = daily_fresh["trade_date"].astype(str)
        available_dates = sorted(d for d in daily_fresh["trade_date"].unique().tolist() if d > last_local_trade_date)
        if not available_dates:
            report["message"] = "在线接口未返回晚于本地样本的新交易日。"
            return daily, idx, basic, trade_dates, report

        daily_fresh = daily_fresh[daily_fresh["trade_date"].isin(available_dates)].copy()
        daily_basic_fresh = _query_market_slice_by_dates(
            client,
            "daily_basic",
            available_dates,
            fields="ts_code,trade_date,total_mv,circ_mv",
            pool_codes=pool_codes,
        )
        adj_fresh = _query_market_slice_by_dates(
            client,
            "adj_factor",
            available_dates,
            fields="ts_code,trade_date,adj_factor",
            pool_codes=pool_codes,
        )
        basic_fresh = client.stock_basic(fields="ts_code,name,industry,market,list_date,list_status")
        trend_idx_fresh = client.index_daily(
            ts_code=trend_index_code,
            start_date=available_dates[0],
            end_date=available_dates[-1],
            fields="ts_code,trade_date,close,pct_chg",
        )
    except Exception as exc:
        report["message"] = f"在线补数失败: {exc}"
        return daily, idx, basic, trade_dates, report

    daily_new = _prepare_incremental_daily(daily_fresh, daily_basic_fresh, adj_fresh, daily)
    if daily_new.empty:
        report["message"] = "在线补数完成，但增量行为空。"
        return daily, idx, basic, trade_dates, report

    combined_daily = pd.concat([daily, daily_new], ignore_index=True)
    combined_daily = combined_daily.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
    combined_daily = _recompute_runtime_features(combined_daily, recompute_from_date=last_local_trade_date)

    combined_idx = idx.copy()
    if trend_idx_fresh is not None and not trend_idx_fresh.empty:
        idx_new = trend_idx_fresh[["trade_date", "close", "pct_chg"]].copy()
        idx_new.columns = ["trade_date", "idx_close", "idx_pct_chg"]
        idx_new["trade_date"] = idx_new["trade_date"].astype(str)
        for col in ["idx_close", "idx_pct_chg"]:
            idx_new[col] = pd.to_numeric(idx_new[col], errors="coerce")
        idx_new = idx_new[idx_new["trade_date"].isin(available_dates)].copy()
        combined_idx = pd.concat([combined_idx, idx_new], ignore_index=True)
        combined_idx = combined_idx.drop_duplicates(subset=["trade_date"], keep="last")
        combined_idx = combined_idx.sort_values("trade_date").reset_index(drop=True)

    combined_basic = _merge_basic_info(basic, basic_fresh)
    feature_cols = ["sw_l1_ret20", "market_ret20", "sw_l1_excess20", "sw_l1_strength20_vs_market"]
    drop_cols = [col for col in feature_cols if col in combined_daily.columns]
    if drop_cols:
        combined_daily = combined_daily.drop(columns=drop_cols)
    combined_daily = enrich_industry_strength20_features(combined_daily, combined_idx)

    combined_trade_dates = sorted(set(combined_daily["trade_date"].astype(str)) & set(combined_idx["trade_date"].astype(str)))
    if combined_trade_dates:
        latest_trade_date = combined_trade_dates[-1]
        combined_daily = combined_daily[combined_daily["trade_date"].isin(combined_trade_dates)].copy()
        combined_idx = combined_idx[combined_idx["trade_date"].isin(combined_trade_dates)].copy()
    else:
        latest_trade_date = last_local_trade_date

    report.update(
        {
            "used_live_data": True,
            "latest_trade_date": latest_trade_date,
            "fetched_trade_dates": available_dates,
            "message": f"已在线补数 {available_dates[0]} ~ {available_dates[-1]}。",
        }
    )
    return (
        combined_daily.reset_index(drop=True),
        combined_idx.reset_index(drop=True),
        combined_basic.reset_index(drop=True),
        combined_trade_dates,
        report,
    )
