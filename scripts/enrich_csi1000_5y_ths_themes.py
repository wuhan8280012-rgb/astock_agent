#!/usr/bin/env python3
"""Enrich the CSI1000 5y bundle with Tonghuashun concept/theme fields."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import tushare as ts


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / "config" / ".env"
BUNDLE_PATH = PROJECT_ROOT / "data_exports" / "tushare_20210329_20260327_csi1000_5y" / "csi1000_market_bundle_5y.csv"
CACHE_PATH = PROJECT_ROOT / "data" / "ths_theme_membership_cache.json"


def load_token() -> str:
    if not ENV_PATH.exists():
        raise FileNotFoundError(f"Missing env file: {ENV_PATH}")
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("TUSHARE_TOKEN="):
            token = line.split("=", 1)[1].strip()
            if token:
                return token
    raise RuntimeError("TUSHARE_TOKEN not found in .env")


def fetch_concepts(pro) -> pd.DataFrame:
    idx = pro.ths_index(exchange="A", type="N", fields="ts_code,name,count")
    if idx is None or idx.empty:
        raise RuntimeError("ths_index returned empty")
    idx = idx[idx["ts_code"].astype(str).str.startswith(("885", "886"))].copy()
    idx["count"] = pd.to_numeric(idx["count"], errors="coerce")
    idx = idx.dropna(subset=["ts_code", "name"]).reset_index(drop=True)
    return idx


def fetch_membership(pro, concept_df: pd.DataFrame, pool_codes: set[str]) -> dict[str, list[dict]]:
    membership: dict[str, list[dict]] = defaultdict(list)
    total = len(concept_df)

    for i, row in concept_df.iterrows():
        concept_code = str(row["ts_code"]).strip()
        concept_name = str(row["name"]).strip()
        concept_count = float(row["count"]) if pd.notna(row["count"]) else None

        delay = 0.25
        last_err = None
        members = None
        for _ in range(5):
            try:
                members = pro.ths_member(ts_code=concept_code, fields="ts_code,con_code,in_date,out_date,is_new")
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(delay)
                delay *= 2
        if members is None:
            raise RuntimeError(f"ths_member failed for {concept_code} {concept_name}: {last_err}")

        if not members.empty and "con_code" in members.columns:
            codes = members["con_code"].dropna().astype(str).tolist()
            for code in codes:
                if code in pool_codes:
                    membership[code].append(
                        {
                            "theme": concept_name,
                            "theme_code": concept_code,
                            "theme_member_count": concept_count,
                        }
                    )

        if (i + 1) % 50 == 0 or i + 1 == total:
            print(f"progress {i + 1}/{total} concepts, mapped stocks={len(membership)}", flush=True)
        time.sleep(0.15)

    return membership


def choose_primary_theme(items: list[dict]) -> str:
    if not items:
        return ""
    # Prefer the more specific concept (smaller member count), then shorter name.
    ordered = sorted(
        items,
        key=lambda x: (
            float("inf") if x["theme_member_count"] is None else x["theme_member_count"],
            len(x["theme"]),
            x["theme"],
        ),
    )
    return ordered[0]["theme"]


def build_theme_fields(membership: dict[str, list[dict]]) -> pd.DataFrame:
    rows = []
    for ts_code, items in membership.items():
        unique = {}
        for item in items:
            unique[item["theme"]] = item
        ordered = sorted(
            unique.values(),
            key=lambda x: (
                float("inf") if x["theme_member_count"] is None else x["theme_member_count"],
                x["theme"],
            ),
        )
        theme_names = [x["theme"] for x in ordered]
        rows.append(
            {
                "ts_code": ts_code,
                "ths_theme_primary": choose_primary_theme(ordered),
                "ths_theme_count": len(theme_names),
                "ths_themes": "|".join(theme_names),
            }
        )
    return pd.DataFrame(rows)


def enrich_bundle(bundle_path: Path, theme_df: pd.DataFrame) -> None:
    raw = pd.read_csv(bundle_path, low_memory=False)
    for col in ["ths_theme_primary", "ths_theme_count", "ths_themes"]:
        if col in raw.columns:
            raw = raw.drop(columns=[col])

    raw = raw.merge(theme_df, on="ts_code", how="left")
    raw["ths_theme_count"] = pd.to_numeric(raw["ths_theme_count"], errors="coerce")
    raw.loc[raw["data_type"].isin(["index_daily", "trade_cal"]), ["ths_theme_primary", "ths_themes"]] = ""
    raw.loc[raw["data_type"].isin(["index_daily", "trade_cal"]), "ths_theme_count"] = pd.NA

    cols = raw.columns.tolist()
    for col in ["ths_theme_primary", "ths_theme_count", "ths_themes"]:
        cols.remove(col)
    insert_at = cols.index("industry") + 1 if "industry" in cols else len(cols)
    cols[insert_at:insert_at] = ["ths_theme_primary", "ths_theme_count", "ths_themes"]
    raw = raw[cols]
    raw.to_csv(bundle_path, index=False)


def main():
    token = load_token()
    ts.set_token(token)
    pro = ts.pro_api(token)

    raw = pd.read_csv(BUNDLE_PATH, usecols=["data_type", "ts_code"], low_memory=False)
    pool_codes = set(raw[raw["data_type"] == "stock_basic"]["ts_code"].dropna().astype(str).tolist())

    concept_df = fetch_concepts(pro)
    membership = fetch_membership(pro, concept_df, pool_codes)
    theme_df = build_theme_fields(membership)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(
            {
                "concept_count": len(concept_df),
                "stock_count": len(theme_df),
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "rows": theme_df.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    enrich_bundle(BUNDLE_PATH, theme_df)

    print(json.dumps(
        {
            "bundle": str(BUNDLE_PATH),
            "cache": str(CACHE_PATH),
            "concept_count": len(concept_df),
            "stock_count": len(theme_df),
            "sample": theme_df.head(10).to_dict(orient="records"),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
