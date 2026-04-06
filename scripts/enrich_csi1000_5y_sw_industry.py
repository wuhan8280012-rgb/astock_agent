#!/usr/bin/env python3
"""Enrich the CSI1000 5y bundle with Shenwan 2021 industry hierarchy."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import tushare as ts


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / "config" / ".env"
BUNDLE_PATH = PROJECT_ROOT / "data_exports" / "tushare_20210329_20260327_csi1000_5y" / "csi1000_market_bundle_5y.csv"
CACHE_PATH = PROJECT_ROOT / "data" / "sw2021_industry_cache.json"


def load_token() -> str:
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("TUSHARE_TOKEN="):
            token = line.split("=", 1)[1].strip()
            if token:
                return token
    raise RuntimeError("TUSHARE_TOKEN not found")


def load_latest_trade_date(bundle_path: Path) -> str:
    raw = pd.read_csv(bundle_path, usecols=["data_type", "trade_date"], low_memory=False)
    daily_dates = raw.loc[raw["data_type"] == "daily", "trade_date"].dropna().astype(str)
    daily_dates = daily_dates.str.replace(r"\.0$", "", regex=True).str.zfill(8)
    return daily_dates.max()


def fetch_classify_tables(pro) -> tuple[pd.DataFrame, dict[str, dict]]:
    dfs = []
    for level in ["L1", "L2", "L3"]:
        df = pro.index_classify(src="SW2021", level=level)
        if df is None or df.empty:
            raise RuntimeError(f"index_classify empty for {level}")
        dfs.append(df.copy())
    classify = pd.concat(dfs, ignore_index=True)
    classify["industry_code"] = classify["industry_code"].astype(str)
    classify["parent_code"] = classify["parent_code"].astype(str)
    mapping = {
        row["industry_code"]: {
            "industry_name": row["industry_name"],
            "level": row["level"],
            "parent_code": row["parent_code"],
            "index_code": row["index_code"],
        }
        for _, row in classify.iterrows()
    }
    return classify, mapping


def fetch_active_membership(pro, classify: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
    rows = []
    total = len(classify)
    for i, row in classify.iterrows():
        index_code = str(row["index_code"]).strip()
        industry_code = str(row["industry_code"]).strip()
        industry_name = str(row["industry_name"]).strip()
        level = str(row["level"]).strip()
        delay = 0.25
        last_err = None
        df = None
        for _ in range(5):
            try:
                df = pro.index_member(index_code=index_code, fields="index_code,con_code,in_date,out_date,is_new")
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(delay)
                delay *= 2
        if df is None:
            raise RuntimeError(f"index_member failed for {index_code} {industry_name}: {last_err}")
        if df.empty:
            continue
        tmp = df.copy()
        tmp["con_code"] = tmp["con_code"].astype(str)
        tmp["in_date"] = tmp["in_date"].fillna("").astype(str)
        tmp["out_date"] = tmp["out_date"].fillna("").astype(str)
        tmp = tmp[(tmp["in_date"] == "") | (tmp["in_date"] <= as_of_date)]
        tmp = tmp[(tmp["out_date"] == "") | (tmp["out_date"] > as_of_date)]
        if tmp.empty:
            continue
        tmp["industry_code"] = industry_code
        tmp["industry_name"] = industry_name
        tmp["level"] = level
        rows.append(tmp[["con_code", "industry_code", "industry_name", "level", "in_date", "out_date"]])
        if (i + 1) % 50 == 0 or i + 1 == total:
            print(f"progress {i + 1}/{total} industries", flush=True)
        time.sleep(0.10)

    if not rows:
        raise RuntimeError("No active SW membership rows fetched")
    members = pd.concat(rows, ignore_index=True)
    return members


def pick_latest_active(members: pd.DataFrame, level: str) -> pd.DataFrame:
    df = members[members["level"] == level].copy()
    if df.empty:
        return df
    df["in_date_sort"] = pd.to_numeric(df["in_date"].replace("", "0"), errors="coerce").fillna(0)
    df = df.sort_values(["con_code", "in_date_sort", "industry_code"], ascending=[True, False, True])
    df = df.drop_duplicates(subset=["con_code"], keep="first")
    return df[["con_code", "industry_code", "industry_name"]].rename(
        columns={
            "con_code": "ts_code",
            "industry_code": f"sw_{level.lower()}_code",
            "industry_name": f"sw_{level.lower()}_name",
        }
    )


def build_sw_mapping(members: pd.DataFrame, classify_map: dict[str, dict]) -> pd.DataFrame:
    l1 = pick_latest_active(members, "L1")
    l2 = pick_latest_active(members, "L2")
    l3 = pick_latest_active(members, "L3")

    all_codes = sorted(set(l1["ts_code"]) | set(l2["ts_code"]) | set(l3["ts_code"]))
    merged = pd.DataFrame({"ts_code": all_codes})
    for df in [l1, l2, l3]:
        if not df.empty:
            merged = merged.merge(df, on="ts_code", how="left")

    # Backfill parents from more specific levels when needed.
    def parent_of(code: str) -> str:
        if not code or code not in classify_map:
            return ""
        return str(classify_map[code]["parent_code"])

    def name_of(code: str) -> str:
        if not code or code not in classify_map:
            return ""
        return str(classify_map[code]["industry_name"])

    sw_l2_codes = []
    sw_l1_codes = []
    sw_l2_names = []
    sw_l1_names = []

    for _, row in merged.iterrows():
        l3_code = row.get("sw_l3_code")
        l2_code = row.get("sw_l2_code")
        l1_code = row.get("sw_l1_code")

        if pd.notna(l3_code) and str(l3_code):
            l2_code = str(l2_code) if pd.notna(l2_code) and str(l2_code) else parent_of(str(l3_code))
        if pd.notna(l2_code) and str(l2_code):
            l1_code = str(l1_code) if pd.notna(l1_code) and str(l1_code) else parent_of(str(l2_code))

        l2_code = str(l2_code) if pd.notna(l2_code) and str(l2_code) != "nan" else ""
        l1_code = str(l1_code) if pd.notna(l1_code) and str(l1_code) != "nan" else ""
        sw_l2_codes.append(l2_code)
        sw_l1_codes.append(l1_code)
        sw_l2_names.append(name_of(l2_code))
        sw_l1_names.append(name_of(l1_code))

    merged["sw_l2_code"] = sw_l2_codes
    merged["sw_l2_name"] = sw_l2_names
    merged["sw_l1_code"] = sw_l1_codes
    merged["sw_l1_name"] = sw_l1_names
    return merged[
        [
            "ts_code",
            "sw_l1_code",
            "sw_l1_name",
            "sw_l2_code",
            "sw_l2_name",
            "sw_l3_code",
            "sw_l3_name",
        ]
    ]


def enrich_bundle(bundle_path: Path, sw_df: pd.DataFrame) -> None:
    raw = pd.read_csv(bundle_path, low_memory=False)
    for col in ["sw_l1_code", "sw_l1_name", "sw_l2_code", "sw_l2_name", "sw_l3_code", "sw_l3_name"]:
        if col in raw.columns:
            raw = raw.drop(columns=[col])

    raw = raw.merge(sw_df, on="ts_code", how="left")
    raw.loc[raw["data_type"].isin(["index_daily", "trade_cal"]), ["sw_l1_code", "sw_l1_name", "sw_l2_code", "sw_l2_name", "sw_l3_code", "sw_l3_name"]] = ""

    cols = raw.columns.tolist()
    for col in ["sw_l1_code", "sw_l1_name", "sw_l2_code", "sw_l2_name", "sw_l3_code", "sw_l3_name"]:
        cols.remove(col)
    insert_at = cols.index("industry") + 1 if "industry" in cols else len(cols)
    cols[insert_at:insert_at] = ["sw_l1_code", "sw_l1_name", "sw_l2_code", "sw_l2_name", "sw_l3_code", "sw_l3_name"]
    raw = raw[cols]
    raw.to_csv(bundle_path, index=False)


def main():
    token = load_token()
    ts.set_token(token)
    pro = ts.pro_api(token)
    as_of_date = load_latest_trade_date(BUNDLE_PATH)
    classify, classify_map = fetch_classify_tables(pro)
    members = fetch_active_membership(pro, classify, as_of_date)
    sw_df = build_sw_mapping(members, classify_map)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(
            {
                "as_of_date": as_of_date,
                "industry_count": len(classify),
                "stock_count": len(sw_df),
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "rows": sw_df.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    enrich_bundle(BUNDLE_PATH, sw_df)
    print(
        json.dumps(
            {
                "bundle": str(BUNDLE_PATH),
                "cache": str(CACHE_PATH),
                "as_of_date": as_of_date,
                "industry_count": len(classify),
                "stock_count": len(sw_df),
                "sample": sw_df.head(10).to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
