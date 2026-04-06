#!/usr/bin/env python3
"""Enrich the CSI1000 5y bundle with 20d SW L1 industry strength vs market."""

from __future__ import annotations

from pathlib import Path
import os

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = PROJECT_ROOT / "data_exports" / "tushare_20210329_20260327_csi1000_5y" / "csi1000_market_bundle_5y.csv"
MARKET_INDEX_PATH = PROJECT_ROOT / "data" / "market_index_000001sh_5y.csv"
TMP_PATH = BUNDLE_PATH.with_suffix(".industry_strength20.tmp.csv")
CHUNKSIZE = 120_000


def build_strength_frame(raw: pd.DataFrame) -> pd.DataFrame:
    daily = raw.loc[raw["data_type"] == "daily", ["trade_date", "ts_code", "close", "sw_l1_name"]].copy()
    daily["trade_date"] = daily["trade_date"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)
    daily["close"] = pd.to_numeric(daily["close"], errors="coerce")
    daily = daily.dropna(subset=["close", "sw_l1_name"])
    daily = daily.sort_values(["sw_l1_name", "trade_date"]).reset_index(drop=True)

    ind = (
        daily.groupby(["trade_date", "sw_l1_name"], as_index=False)["close"]
        .mean()
        .sort_values(["sw_l1_name", "trade_date"])
        .reset_index(drop=True)
    )
    ind["sw_l1_ret20"] = ind.groupby("sw_l1_name")["close"].transform(lambda s: s / s.shift(20) - 1)

    idx = pd.read_csv(MARKET_INDEX_PATH)
    idx["trade_date"] = idx["trade_date"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)
    idx["close"] = pd.to_numeric(idx["close"], errors="coerce")
    idx = idx.sort_values("trade_date").reset_index(drop=True)
    idx["market_ret20"] = idx["close"] / idx["close"].shift(20) - 1
    idx = idx[["trade_date", "market_ret20"]]

    ind = ind.merge(idx, on="trade_date", how="left")
    ind["sw_l1_strength20_vs_market"] = ind["sw_l1_ret20"] / ind["market_ret20"]
    ind.loc[ind["market_ret20"].abs() < 1e-12, "sw_l1_strength20_vs_market"] = pd.NA
    return ind[["trade_date", "sw_l1_name", "sw_l1_ret20", "market_ret20", "sw_l1_strength20_vs_market"]]


def enrich_bundle() -> dict:
    raw_for_strength = pd.read_csv(BUNDLE_PATH, low_memory=False)
    strength = build_strength_frame(raw_for_strength)
    del raw_for_strength

    strength["trade_date"] = strength["trade_date"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)
    strength["sw_l1_name"] = strength["sw_l1_name"].astype(str)
    strength = strength.set_index(["trade_date", "sw_l1_name"])

    daily_rows = 0
    daily_non_null = 0
    sample = None
    first = True

    for chunk in pd.read_csv(BUNDLE_PATH, low_memory=False, chunksize=CHUNKSIZE):
        for col in ["sw_l1_ret20", "market_ret20", "sw_l1_strength20_vs_market"]:
            if col in chunk.columns:
                chunk = chunk.drop(columns=[col])

        chunk["trade_date"] = chunk["trade_date"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)
        chunk["sw_l1_name"] = chunk["sw_l1_name"].fillna("").astype(str)

        joined = chunk.join(
            strength,
            on=["trade_date", "sw_l1_name"],
            how="left",
        )
        mask = joined["data_type"].isin(["index_daily", "trade_cal"])
        joined.loc[mask, ["sw_l1_ret20", "market_ret20", "sw_l1_strength20_vs_market"]] = pd.NA

        cols = joined.columns.tolist()
        for col in ["sw_l1_ret20", "market_ret20", "sw_l1_strength20_vs_market"]:
            cols.remove(col)
        insert_at = cols.index("sw_l1_name") + 1 if "sw_l1_name" in cols else len(cols)
        cols[insert_at:insert_at] = ["sw_l1_ret20", "market_ret20", "sw_l1_strength20_vs_market"]
        joined = joined[cols]

        daily_mask = joined["data_type"] == "daily"
        daily_rows += int(daily_mask.sum())
        daily_non_null += int(joined.loc[daily_mask, "sw_l1_strength20_vs_market"].notna().sum())
        if sample is None:
            sample = (
                joined.loc[daily_mask, ["ts_code", "trade_date", "sw_l1_name", "sw_l1_ret20", "market_ret20", "sw_l1_strength20_vs_market"]]
                .dropna(subset=["sw_l1_strength20_vs_market"])
                .head(10)
            )

        joined.to_csv(TMP_PATH, index=False, mode="w" if first else "a", header=first)
        first = False

    os.replace(TMP_PATH, BUNDLE_PATH)

    return {
        "bundle": str(BUNDLE_PATH),
        "field": "sw_l1_strength20_vs_market",
        "daily_non_null": daily_non_null,
        "daily_rows": daily_rows,
        "sample": [] if sample is None else sample.to_dict(orient="records"),
    }


def main() -> None:
    result = enrich_bundle()
    print(result)


if __name__ == "__main__":
    main()
