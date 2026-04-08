#!/usr/bin/env python3
"""Point-in-time index constituent snapshots backed by Tushare index_weight."""

from __future__ import annotations

import calendar
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / "config" / ".env"
CACHE_DIR = PROJECT_ROOT / "data" / "pit_constituents"


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


def _month_key(value: str) -> str:
    return str(value)[:6]


def _month_window(year: int, month: int) -> tuple[str, str]:
    start = datetime(year, month, 1)
    end = datetime(year, month, calendar.monthrange(year, month)[1])
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")


def _iter_month_keys(start_date: str, end_date: str) -> list[str]:
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    months = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        months.append(f"{y:04d}{m:02d}")
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1
    return months


def default_cache_path(index_code: str) -> Path:
    return CACHE_DIR / f"index_weight_{index_code.replace('.', '_')}.csv"


def _fetch_month_snapshot(index_code: str, month_key: str, token: str) -> pd.DataFrame:
    try:
        import tushare as ts
    except Exception as exc:
        raise RuntimeError(f"tushare 不可用: {exc}") from exc

    year = int(month_key[:4])
    month = int(month_key[4:6])
    start_date, end_date = _month_window(year, month)
    ts.set_token(token)
    pro = ts.pro_api(token)
    df = pro.index_weight(
        index_code=index_code,
        start_date=start_date,
        end_date=end_date,
        fields="index_code,con_code,trade_date,weight",
    )
    if df is None or df.empty:
        return pd.DataFrame(columns=["index_code", "con_code", "trade_date", "weight"])
    df = df.copy()
    df["index_code"] = df["index_code"].astype(str)
    df["con_code"] = df["con_code"].astype(str)
    df["trade_date"] = df["trade_date"].astype(str)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    return df[["index_code", "con_code", "trade_date", "weight"]]


def load_or_fetch_index_weight_history(
    index_code: str,
    start_date: str,
    end_date: str,
    cache_path: Path | None = None,
    refresh: bool = False,
    token: str | None = None,
) -> pd.DataFrame:
    cache = Path(cache_path) if cache_path else default_cache_path(index_code)
    cache.parent.mkdir(parents=True, exist_ok=True)

    existing = pd.DataFrame(columns=["index_code", "con_code", "trade_date", "weight"])
    if cache.exists() and not refresh:
        existing = pd.read_csv(cache)
        if not existing.empty:
            existing["index_code"] = existing["index_code"].astype(str)
            existing["con_code"] = existing["con_code"].astype(str)
            existing["trade_date"] = existing["trade_date"].astype(str)
            existing["weight"] = pd.to_numeric(existing["weight"], errors="coerce")

    fetch_start = (datetime.strptime(start_date, "%Y%m%d") - timedelta(days=40)).strftime("%Y%m%d")
    month_keys = _iter_month_keys(fetch_start, end_date)
    existing_months = set(existing["trade_date"].astype(str).map(_month_key)) if not existing.empty else set()
    missing_months = month_keys if refresh else [month for month in month_keys if month not in existing_months]

    if missing_months:
        api_token = token or load_env_token()
        if not api_token:
            raise RuntimeError("未配置 TUSHARE_TOKEN，无法拉取 PIT 成分股")
        frames = [existing] if not existing.empty else []
        for month_key in missing_months:
            frames.append(_fetch_month_snapshot(index_code=index_code, month_key=month_key, token=api_token))
        merged = pd.concat(frames, ignore_index=True) if frames else existing
        merged = merged.drop_duplicates(subset=["index_code", "con_code", "trade_date"], keep="last")
        merged = merged.sort_values(["trade_date", "con_code"]).reset_index(drop=True)
        merged.to_csv(cache, index=False)
        existing = merged

    if existing.empty:
        return existing

    lower_bound = (datetime.strptime(start_date, "%Y%m%d") - timedelta(days=40)).strftime("%Y%m%d")
    filtered = existing[
        (existing["trade_date"].astype(str) >= lower_bound) & (existing["trade_date"].astype(str) <= str(end_date))
    ].copy()
    return filtered.sort_values(["trade_date", "con_code"]).reset_index(drop=True)


@dataclass
class PitUniverse:
    index_code: str
    constituents_by_date: dict[str, set[str]]
    snapshot_date_by_trade_date: dict[str, str]
    snapshot_count: int
    constituent_count_max: int
    unique_codes: int
    cache_path: str
    earliest_snapshot_date: str
    latest_snapshot_date: str
    fallback_to_earliest_days: int


def build_pit_universe(index_weights: pd.DataFrame, trade_dates: list[str], index_code: str, cache_path: Path | None = None) -> PitUniverse:
    if index_weights.empty:
        raise ValueError("index_weights 为空，无法构建 PIT 成分股")

    snapshot_dates = sorted(index_weights["trade_date"].astype(str).unique())
    snapshot_codes = {
        trade_date: set(group["con_code"].astype(str))
        for trade_date, group in index_weights.groupby(index_weights["trade_date"].astype(str))
    }

    constituents_by_date: dict[str, set[str]] = {}
    snapshot_by_trade_date: dict[str, str] = {}
    snapshot_idx = 0
    fallback_days = 0

    for trade_date in sorted(str(d) for d in trade_dates):
        while snapshot_idx + 1 < len(snapshot_dates) and snapshot_dates[snapshot_idx + 1] <= trade_date:
            snapshot_idx += 1

        if trade_date < snapshot_dates[0]:
            ref_snapshot = snapshot_dates[0]
            fallback_days += 1
        else:
            ref_snapshot = snapshot_dates[snapshot_idx]

        constituents_by_date[trade_date] = snapshot_codes[ref_snapshot]
        snapshot_by_trade_date[trade_date] = ref_snapshot

    return PitUniverse(
        index_code=index_code,
        constituents_by_date=constituents_by_date,
        snapshot_date_by_trade_date=snapshot_by_trade_date,
        snapshot_count=len(snapshot_dates),
        constituent_count_max=max(len(v) for v in snapshot_codes.values()),
        unique_codes=int(index_weights["con_code"].astype(str).nunique()),
        cache_path=str(cache_path or default_cache_path(index_code)),
        earliest_snapshot_date=snapshot_dates[0],
        latest_snapshot_date=snapshot_dates[-1],
        fallback_to_earliest_days=fallback_days,
    )


def load_or_fetch_pit_universe(
    index_code: str,
    trade_dates: list[str],
    refresh: bool = False,
    cache_path: Path | None = None,
    token: str | None = None,
) -> PitUniverse:
    if not trade_dates:
        raise ValueError("trade_dates 为空")
    history = load_or_fetch_index_weight_history(
        index_code=index_code,
        start_date=str(trade_dates[0]),
        end_date=str(trade_dates[-1]),
        cache_path=cache_path,
        refresh=refresh,
        token=token,
    )
    return build_pit_universe(history, trade_dates=trade_dates, index_code=index_code, cache_path=cache_path)
