"""
data_cache.py — 本地数据缓存层
================================
"""

import os
import time
import json
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd
import numpy as np


CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
DAILY_DIR = os.path.join(CACHE_DIR, "daily")
FUND_DIR = os.path.join(CACHE_DIR, "fundamental")
META_DIR = os.path.join(CACHE_DIR, "meta")
MANIFEST_PATH = os.path.join(META_DIR, "cache_manifest.json")

CACHE_CONFIG = {
    "keep_days": 120,
    "meta_refresh_days": 7,
    "fundamental_refresh_days": 1,
    "financial_refresh_quarter": True,
    "batch_size": 80,
    "adj_factor": True,
}


class DataCache:
    """本地数据缓存管理器"""

    def __init__(self, fetcher=None):
        self.fetcher = fetcher
        self.pro = fetcher.pro if fetcher and hasattr(fetcher, "pro") else None
        self._ensure_dirs()
        self._manifest = self._load_manifest()

        self._mem_daily = {}
        self._mem_name = {}
        self._mem_basic = None
        self._mem_daily_basic = {}
        self._mem_daily_basic_df = {}
        self._mem_latest_fin_df = None

    def _ensure_dirs(self):
        for d in [CACHE_DIR, DAILY_DIR, FUND_DIR, META_DIR]:
            os.makedirs(d, exist_ok=True)

    def _table_exists(self, path: str) -> bool:
        return os.path.exists(path) or os.path.exists(path + ".pkl")

    def _write_table(self, path: str, df: pd.DataFrame):
        try:
            df.to_parquet(path, index=False)
            return
        except Exception:
            pass
        df.to_pickle(path + ".pkl")

    def _read_table(self, path: str) -> pd.DataFrame:
        if os.path.exists(path):
            try:
                return pd.read_parquet(path)
            except Exception:
                pass
        if os.path.exists(path + ".pkl"):
            return pd.read_pickle(path + ".pkl")
        raise FileNotFoundError(path)

    def _load_manifest(self) -> dict:
        try:
            if os.path.exists(MANIFEST_PATH):
                with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {
            "last_daily": "",
            "last_meta": "",
            "last_fund": "",
            "daily_dates": [],
        }

    def _save_manifest(self):
        try:
            with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
                json.dump(self._manifest, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ===== 日线 =====

    def update_daily(self, date: str, stock_list: list = None):
        if not self.pro:
            print("  ⚠️ 无 Tushare 连接，跳过日线更新")
            return

        parquet_path = os.path.join(DAILY_DIR, f"{date}.parquet")
        if self._table_exists(parquet_path):
            print(f"  📦 日线缓存已存在: {date}")
            return

        print(f"  📥 拉取全市场日线: {date} ...")
        t0 = time.time()

        try:
            if hasattr(self.fetcher, "_throttle"):
                self.fetcher._throttle()
            df = self.pro.daily(trade_date=date)
            if df is None or df.empty:
                print(f"  ⚠️ {date} 无日线数据（非交易日？）")
                return

            if CACHE_CONFIG["adj_factor"] and hasattr(self.pro, "adj_factor"):
                if hasattr(self.fetcher, "_throttle"):
                    self.fetcher._throttle()
                adj = self.pro.adj_factor(trade_date=date)
                if adj is not None and not adj.empty:
                    df = df.merge(
                        adj[["ts_code", "trade_date", "adj_factor"]],
                        on=["ts_code", "trade_date"],
                        how="left",
                    )

            self._write_table(parquet_path, df)

            dates = self._manifest.get("daily_dates", [])
            if date not in dates:
                dates.append(date)
                dates.sort()
            self._manifest["daily_dates"] = dates
            self._manifest["last_daily"] = date
            self._save_manifest()

            dt = time.time() - t0
            print(f"  ✅ 日线缓存写入: {len(df)}条 ({dt:.1f}s)")
        except Exception as e:
            print(f"  ❌ 日线更新失败: {e}")

    def update_daily_range(self, start_date: str, end_date: str):
        if not self.pro:
            return
        try:
            if hasattr(self.fetcher, "_throttle"):
                self.fetcher._throttle()
            cal = self.pro.trade_cal(
                start_date=start_date,
                end_date=end_date,
                is_open="1",
                fields="cal_date",
            )
            if cal is None or cal.empty:
                return
            trade_dates = sorted(cal["cal_date"].tolist())
        except Exception as e:
            print(f"  ❌ 获取交易日历失败: {e}")
            return

        print(f"  📅 需更新 {len(trade_dates)} 个交易日 ({start_date}~{end_date})")
        for i, td in enumerate(trade_dates):
            self.update_daily(td)
            if (i + 1) % 10 == 0:
                print(f"    ...已更新 {i+1}/{len(trade_dates)}")
            time.sleep(0.1)

    def get_daily(self, ts_code: str, end_date: str = None, days: int = 60) -> Optional[pd.DataFrame]:
        mem_key = (ts_code, end_date, days)
        if mem_key in self._mem_daily:
            return self._mem_daily[mem_key]

        dates = self._manifest.get("daily_dates", [])
        if end_date:
            dates = [d for d in dates if d <= end_date]

        target_dates = dates[-days:] if len(dates) >= days else dates
        if not target_dates:
            return self._fallback_remote(ts_code, end_date, days)

        frames = []
        for d in target_dates:
            path = os.path.join(DAILY_DIR, f"{d}.parquet")
            if not self._table_exists(path):
                continue
            try:
                day_df = self._read_table(path)
                stock_row = day_df[day_df["ts_code"] == ts_code]
                if not stock_row.empty:
                    frames.append(stock_row)
            except Exception:
                continue

        if not frames:
            return self._fallback_remote(ts_code, end_date, days)

        df = pd.concat(frames, ignore_index=True)
        df = df.sort_values("trade_date").reset_index(drop=True)

        if "adj_factor" in df.columns:
            latest_factor = df["adj_factor"].iloc[-1]
            if pd.notna(latest_factor) and latest_factor > 0:
                for col in ["open", "high", "low", "close"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce") * pd.to_numeric(
                            df["adj_factor"], errors="coerce"
                        ) / float(latest_factor)

        if "vol" not in df.columns and "volume" in df.columns:
            df["vol"] = df["volume"]

        self._mem_daily[mem_key] = df
        return df

    def get_daily_batch(self, ts_codes: list, end_date: str = None, days: int = 60) -> dict:
        out = {}
        for code in ts_codes:
            df = self.get_daily(code, end_date=end_date, days=days)
            if df is not None and not df.empty:
                out[code] = df
        return out

    def _fallback_remote(self, ts_code, end_date, days):
        if not self.fetcher:
            return None
        try:
            for method in ["get_daily", "get_stock_daily", "get_k_data"]:
                if hasattr(self.fetcher, method):
                    result = getattr(self.fetcher, method)(
                        ts_code, days=days, end_date=end_date
                    )
                    return result
        except Exception:
            pass
        return None

    # ===== 基本面 =====

    def update_fundamental(self, date: str):
        if not self.pro:
            return

        basic_path = os.path.join(FUND_DIR, f"daily_basic_{date}.parquet")
        if not self._table_exists(basic_path):
            print(f"  📥 拉取日度基本面: {date} ...")
            try:
                if hasattr(self.fetcher, "_throttle"):
                    self.fetcher._throttle()
                df = self.pro.daily_basic(trade_date=date)
                if df is not None and not df.empty:
                    self._write_table(basic_path, df)
                    self._manifest["last_fund"] = date
                    self._save_manifest()
                    print(f"  ✅ 日度基本面: {len(df)} 条")
            except Exception as e:
                print(f"  ⚠️ 日度基本面失败: {e}")

        self._update_financial_if_needed(date)

    def _update_financial_if_needed(self, date: str):
        day = date[-2:]
        if day != "01":
            return

        year = int(date[:4])
        month = int(date[4:6])
        if month >= 11:
            period = f"{year}0930"
        elif month >= 9:
            period = f"{year}0630"
        elif month >= 5:
            period = f"{year}0331"
        else:
            period = f"{year-1}1231"

        fin_path = os.path.join(FUND_DIR, f"financial_{period}.parquet")
        if self._table_exists(fin_path):
            return

        print(f"  📥 拉取季度财报: {period} ...")
        try:
            if hasattr(self.fetcher, "_throttle"):
                self.fetcher._throttle()
            income = self.pro.income(
                period=period,
                fields="ts_code,revenue,n_income,n_income_attr_p",
            )
            if hasattr(self.fetcher, "_throttle"):
                self.fetcher._throttle()
            fina = self.pro.fina_indicator(
                period=period,
                fields="ts_code,roe,roa,grossprofit_margin,netprofit_yoy,or_yoy",
            )
            if income is not None and fina is not None:
                merged = income.merge(fina, on="ts_code", how="outer")
                merged["period"] = period
                self._write_table(fin_path, merged)
                print(f"  ✅ 季度财报: {len(merged)} 条")
        except Exception as e:
            print(f"  ⚠️ 季度财报失败: {e}")

    def get_fundamental(self, ts_code: str, date: str) -> dict:
        cache_key = f"{ts_code}_{date}"
        if cache_key in self._mem_daily_basic:
            return self._mem_daily_basic[cache_key]

        result = {}
        basic_path = os.path.join(FUND_DIR, f"daily_basic_{date}.parquet")
        if self._table_exists(basic_path):
            try:
                if date not in self._mem_daily_basic_df:
                    self._mem_daily_basic_df[date] = self._read_table(basic_path)
                df = self._mem_daily_basic_df.get(date, pd.DataFrame())
                row = df[df["ts_code"] == ts_code]
                if not row.empty:
                    r = row.iloc[0]
                    result.update({
                        "pe_ttm": float(r.get("pe_ttm", 0) or 0),
                        "pb": float(r.get("pb", 0) or 0),
                        "turnover_rate": float(r.get("turnover_rate", 0) or 0),
                        "circ_mv": float(r.get("circ_mv", 0) or 0),
                        "total_mv": float(r.get("total_mv", 0) or 0),
                    })
            except Exception:
                pass

        fin_files = sorted([f for f in os.listdir(FUND_DIR) if f.startswith("financial_")])
        if fin_files:
            latest_fin = os.path.join(FUND_DIR, fin_files[-1])
            try:
                if self._mem_latest_fin_df is None:
                    self._mem_latest_fin_df = self._read_table(latest_fin)
                fdf = self._mem_latest_fin_df
                row = fdf[fdf["ts_code"] == ts_code]
                if not row.empty:
                    r = row.iloc[0]
                    result.update({
                        "roe": float(r.get("roe", 0) or 0),
                        "revenue_yoy": float(r.get("or_yoy", 0) or 0),
                        "profit_yoy": float(r.get("netprofit_yoy", 0) or 0),
                        "gross_margin": float(r.get("grossprofit_margin", 0) or 0),
                    })
            except Exception:
                pass

        self._mem_daily_basic[cache_key] = result
        return result

    def get_fundamental_batch(self, date: str) -> pd.DataFrame:
        basic_path = os.path.join(FUND_DIR, f"daily_basic_{date}.parquet")
        if self._table_exists(basic_path):
            try:
                return self._read_table(basic_path)
            except Exception:
                pass
        if self.pro:
            try:
                if hasattr(self.fetcher, "_throttle"):
                    self.fetcher._throttle()
                return self.pro.daily_basic(trade_date=date)
            except Exception:
                pass
        return pd.DataFrame()

    # ===== 元数据 =====

    def update_meta(self, force: bool = False):
        meta_path = os.path.join(META_DIR, "stock_basic.parquet")
        last_meta = self._manifest.get("last_meta", "")
        if not force and last_meta:
            try:
                last_dt = datetime.strptime(last_meta, "%Y%m%d")
                if (datetime.now() - last_dt).days < CACHE_CONFIG["meta_refresh_days"]:
                    return
            except Exception:
                pass

        if not self.pro:
            return

        print("  📥 更新股票基本信息 ...")
        try:
            if hasattr(self.fetcher, "_throttle"):
                self.fetcher._throttle()
            df = self.pro.stock_basic(
                list_status="L",
                fields="ts_code,name,industry,market,list_date,is_hs",
            )
            if df is not None and not df.empty:
                self._write_table(meta_path, df)
                self._manifest["last_meta"] = datetime.now().strftime("%Y%m%d")
                self._save_manifest()
                print(f"  ✅ 股票基本信息: {len(df)} 只")
        except Exception as e:
            print(f"  ⚠️ 元数据更新失败: {e}")

    def get_stock_name(self, ts_code: str) -> str:
        if not self._mem_name:
            self._load_name_map()
        return self._mem_name.get(ts_code, ts_code)

    def get_stock_basic(self) -> pd.DataFrame:
        if self._mem_basic is not None:
            return self._mem_basic

        meta_path = os.path.join(META_DIR, "stock_basic.parquet")
        if self._table_exists(meta_path):
            try:
                self._mem_basic = self._read_table(meta_path)
                return self._mem_basic
            except Exception:
                pass

        self.update_meta(force=True)
        if self._table_exists(meta_path):
            try:
                self._mem_basic = self._read_table(meta_path)
                return self._mem_basic
            except Exception:
                pass

        return pd.DataFrame()

    def _load_name_map(self):
        basic = self.get_stock_basic()
        if basic is not None and not basic.empty:
            self._mem_name = dict(zip(basic["ts_code"], basic["name"]))

    # ===== 维护 =====

    def cleanup_old(self):
        keep = CACHE_CONFIG["keep_days"]
        dates = self._manifest.get("daily_dates", [])
        if len(dates) <= keep:
            return

        to_remove = dates[:-keep]
        for d in to_remove:
            p1 = os.path.join(DAILY_DIR, f"{d}.parquet")
            p2 = os.path.join(FUND_DIR, f"daily_basic_{d}.parquet")
            try:
                if os.path.exists(p1):
                    os.remove(p1)
                if os.path.exists(p1 + ".pkl"):
                    os.remove(p1 + ".pkl")
                if os.path.exists(p2):
                    os.remove(p2)
                if os.path.exists(p2 + ".pkl"):
                    os.remove(p2 + ".pkl")
            except Exception:
                pass

        self._manifest["daily_dates"] = dates[-keep:]
        self._save_manifest()
        print(f"  🗑️ 清理 {len(to_remove)} 天过期缓存")

    def get_cache_stats(self) -> dict:
        dates = self._manifest.get("daily_dates", [])
        return {
            "daily_dates": len(dates),
            "date_range": f"{dates[0]}~{dates[-1]}" if dates else "空",
            "last_daily": self._manifest.get("last_daily", ""),
            "last_meta": self._manifest.get("last_meta", ""),
            "mem_daily_entries": len(self._mem_daily),
            "mem_name_entries": len(self._mem_name),
        }

    def daily_update(self, date: str):
        print("=" * 50)
        print(f"  📦 数据缓存更新: {date}")
        print("=" * 50)
        t0 = time.time()
        self.update_meta()
        self.update_daily(date)
        self.update_fundamental(date)
        self.cleanup_old()
        dt = time.time() - t0
        stats = self.get_cache_stats()
        print(
            f"  ✅ 缓存更新完成 ({dt:.1f}s) | 日线{stats['daily_dates']}天 | 名称{stats['mem_name_entries']}只"
        )
        print("=" * 50)

    def ensure_history(self, date: str, days: int = 120):
        dates = self._manifest.get("daily_dates", [])
        if len(dates) >= int(days * 0.8):
            return
        end_dt = datetime.strptime(date, "%Y%m%d")
        start_dt = end_dt - timedelta(days=int(days * 1.5))
        start_str = start_dt.strftime("%Y%m%d")
        print(f"  📦 首次运行，回填历史数据 {start_str}~{date} ...")
        self.update_daily_range(start_str, date)


class DailyCache:
    """单日数据缓存，包装 DataFetcher（向后兼容）"""

    def __init__(self, fetcher):
        self.fetcher = fetcher
        self._cache = {}
        self._cache_date = ""

    def _make_key(self, method: str, *args, **kwargs) -> str:
        parts = [method] + [str(a) for a in args]
        parts += [f"{k}={v}" for k, v in sorted(kwargs.items())]
        return "|".join(parts)

    def _check_date(self):
        today = datetime.now().strftime("%Y%m%d")
        if today != self._cache_date:
            self._cache.clear()
            self._cache_date = today

    def _cached_call(self, method_name: str, *args, **kwargs) -> Any:
        self._check_date()
        key = self._make_key(method_name, *args, **kwargs)
        if key in self._cache:
            return self._cache[key]
        method = getattr(self.fetcher, method_name)
        result = method(*args, **kwargs)
        self._cache[key] = result
        return result

    def get_north_flow(self, days=10, **kw):
        return self._cached_call("get_north_flow", days=days, **kw)

    def get_margin_data(self, days=10, **kw):
        return self._cached_call("get_margin_data", days=days, **kw)

    def get_index_daily(self, index_code, days=30, **kw):
        return self._cached_call("get_index_daily", index_code, days=days, **kw)

    def get_market_breadth(self, **kw):
        return self._cached_call("get_market_breadth", **kw)

    def get_limit_list(self, **kw):
        return self._cached_call("get_limit_list", **kw)

    def get_shibor(self, days=30, **kw):
        return self._cached_call("get_shibor", days=days, **kw)

    def get_all_sector_performance(self, days=120, **kw):
        return self._cached_call("get_all_sector_performance", days=days, **kw)

    def get_stock_daily(self, ts_code, days=60, **kw):
        return self._cached_call("get_stock_daily", ts_code, days=days, **kw)

    def get_latest_trade_date(self, **kw):
        return self._cached_call("get_latest_trade_date", **kw)

    def __getattr__(self, name):
        return getattr(self.fetcher, name)

    @property
    def cache_stats(self) -> dict:
        return {
            "date": self._cache_date,
            "entries": len(self._cache),
            "keys": list(self._cache.keys())[:10],
        }
