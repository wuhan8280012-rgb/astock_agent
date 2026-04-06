"""
Unified Tushare API wrapper.
All external modules MUST use this client — direct tushare.pro_api() calls are forbidden.

Responsibilities:
1. Rate limiting (token bucket shared across all concurrent calls)
2. Retry with backoff
3. Data integrity validation (nulls, duplicates, outliers)
"""

import asyncio
import hashlib
import json as json_mod
import sqlite3
import threading
import time as time_module
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
import tushare as ts
try:
    from loguru import logger
except ImportError:  # pragma: no cover - fallback for lean runtime envs
    import logging

    logger = logging.getLogger(__name__)

from config.settings import TUSHARE_TOKEN, TUSHARE_RATE_LIMIT_PER_MINUTE, TUSHARE_BATCH_SIZE
from data_pipeline.clock import set_trade_calendar


class TushareRequestError(Exception):
    """Raised when all retries for a Tushare request are exhausted."""
    pass


# APIs that support comma-separated ts_code for batch queries
BATCH_CAPABLE_APIS = {
    "daily",           # 日线行情
    "adj_factor",      # 复权因子
    "moneyflow",       # 个股资金流向
    "margin_detail",   # 融资融券明细
    "stk_limit",       # 涨跌停价格
    "suspend_d",       # 停复牌信息
}

# APIs known to be unavailable or unsupported — skip without sending requests
DISABLED_APIS = {"anns"}


class TushareClient:
    """
    Singleton Tushare client with rate limiting and data validation.

    Rate limiting:
    - Global token bucket, capacity based on Tushare point tier
    - Default conservative: 80 calls/minute (for 2000-point accounts)
    - All async gather calls share the same semaphore instance

    Retry:
    - Max 3 retries, 10s interval
    - All failures raise TushareRequestError

    Singleton:
    - Only one instance is created per (token, rate_limit) pair.
    - All modules share the same client, ensuring rate limiting works globally.
    """

    _instances: dict[tuple, "TushareClient"] = {}
    _instance_lock = threading.Lock()

    def __new__(cls, token: str = TUSHARE_TOKEN, rate_limit: int = TUSHARE_RATE_LIMIT_PER_MINUTE):
        key = (token, rate_limit)
        with cls._instance_lock:
            if key not in cls._instances:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instances[key] = instance
            return cls._instances[key]

    def __init__(self, token: str = TUSHARE_TOKEN, rate_limit: int = TUSHARE_RATE_LIMIT_PER_MINUTE):
        if self._initialized:
            return
        self._api = ts.pro_api(token)
        self._rate_limit = rate_limit
        self._min_interval = 60.0 / rate_limit
        self._last_call_time = 0.0
        self._trade_cal_cache: Optional[List[str]] = None
        self._sync_lock = threading.Lock()
        self._cache_hits = 0
        self._cache_misses = 0
        # Async rate limiting: semaphore + lock shared across all gather calls
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._async_lock: Optional[asyncio.Lock] = None
        self._initialized = True

    def _ensure_async_primitives(self):
        """Lazily initialize async primitives (must be called inside an event loop)."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(1)
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()

    @staticmethod
    def _normalize_fields(fields):
        if fields is None:
            return None
        if isinstance(fields, (list, tuple, set)):
            fields = ",".join(str(x) for x in fields)
        fields = str(fields)
        fields = fields.replace("turnover,", "turnover_rate,").replace(",turnover", ",turnover_rate")
        if fields == "turnover":
            fields = "turnover_rate"
        return fields

    def _wait_for_rate_limit(self):
        """Synchronous rate limiting via sleep."""
        with self._sync_lock:
            now = time_module.time()
            elapsed = now - self._last_call_time
            if elapsed < self._min_interval:
                time_module.sleep(self._min_interval - elapsed)
            self._last_call_time = time_module.time()

    def query(self, api_name: str, max_retries: int = 3, retry_interval: int = 10,
              enable_cache: bool = False, cache_dir: str = None, **kwargs) -> pd.DataFrame:
        """
        Unified entry point for all Tushare API calls (synchronous).

        Input:
            api_name: Tushare API name (e.g. 'daily', 'fina_indicator')
            max_retries: max retry attempts (default 3)
            retry_interval: seconds between retries (default 10)
            enable_cache: if True, cache results to parquet (backtest use only)
            cache_dir: directory for cache files (required if enable_cache=True)
            **kwargs: passed directly to the Tushare API

        Output:
            pd.DataFrame with validated data

        Raises:
            TushareRequestError if all retries fail
        """
        if api_name in DISABLED_APIS:
            logger.debug(f"Skipping disabled API: {api_name}")
            return pd.DataFrame()

        # Cache check
        cache_path = None
        if enable_cache and cache_dir:
            cache_path = self._get_cache_path(cache_dir, api_name, kwargs)
            if cache_path.exists():
                try:
                    self._cache_hits += 1
                    return pd.read_parquet(cache_path)
                except Exception:
                    pass  # corrupted cache, re-fetch
            self._cache_misses += 1

        last_error = None
        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit()
                api_func = getattr(self._api, api_name)
                df = api_func(**kwargs)
                if df is None:
                    df = pd.DataFrame()
                df = self._validate(df, api_name)
                # Cache write
                if cache_path and not df.empty:
                    try:
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        df.to_parquet(cache_path, index=False)
                    except Exception:
                        pass  # non-critical
                return df
            except TushareRequestError:
                raise
            except Exception as e:
                last_error = e
                err_msg = str(e)
                if any(kw in err_msg for kw in ("每分钟最多访问", "exceed", "rate limit", "频率")):
                    wait = 60
                    logger.warning(f"Tushare rate limit hit for {api_name}, waiting {wait}s before retry "
                                   f"(attempt {attempt + 1}/{max_retries})")
                else:
                    wait = retry_interval
                    logger.warning(f"Tushare request failed (attempt {attempt + 1}/{max_retries}): "
                                 f"{api_name}({kwargs}) -> {e}")
                if attempt < max_retries - 1:
                    time_module.sleep(wait)

        raise TushareRequestError(
            f"All {max_retries} retries failed for {api_name}({kwargs}): {last_error}"
        )

    def stock_basic(self, **kwargs) -> pd.DataFrame:
        kwargs = dict(kwargs)
        kwargs["fields"] = self._normalize_fields(kwargs.get("fields")) or "ts_code,name,industry,exchange,list_date"
        return self.query("stock_basic", **kwargs)

    def daily_basic(self, **kwargs) -> pd.DataFrame:
        kwargs = dict(kwargs)
        kwargs["fields"] = self._normalize_fields(kwargs.get("fields")) or "ts_code,trade_date,total_mv,turnover_rate"
        return self.query("daily_basic", **kwargs)

    def index_daily(self, **kwargs) -> pd.DataFrame:
        kwargs = dict(kwargs)
        kwargs["fields"] = self._normalize_fields(kwargs.get("fields")) or "ts_code,trade_date,close,pct_chg"
        return self.query("index_daily", **kwargs)

    def suspend_d(self, **kwargs) -> pd.DataFrame:
        kwargs = dict(kwargs)
        kwargs["fields"] = self._normalize_fields(kwargs.get("fields")) or "ts_code,suspend_date,resume_date"
        return self.query("suspend_d", **kwargs)

    def daily(self, **kwargs) -> pd.DataFrame:
        kwargs = dict(kwargs)
        fields = self._normalize_fields(kwargs.pop("fields", None)) or "ts_code,trade_date,close,vol"
        ts_code = str(kwargs.pop("ts_code", "") or "")
        start_date = kwargs.pop("start_date", None)
        end_date = kwargs.pop("end_date", None)

        if "," in ts_code:
            codes = [code.strip() for code in ts_code.split(",") if code.strip()]
            if not end_date:
                raise ValueError("daily() batch mode requires end_date")
            if not start_date:
                end_dt = datetime.strptime(end_date, "%Y%m%d")
                start_date = (end_dt - timedelta(days=400)).strftime("%Y%m%d")
            return self.batch_query("daily", codes, start_date, end_date, fields=fields)

        if ts_code:
            if start_date is None and end_date is not None:
                end_dt = datetime.strptime(end_date, "%Y%m%d")
                start_date = (end_dt - timedelta(days=400)).strftime("%Y%m%d")
            return self.query("daily", ts_code=ts_code, start_date=start_date, end_date=end_date, fields=fields)

        return self.query("daily", start_date=start_date, end_date=end_date, fields=fields)

    async def async_query(self, api_name: str, max_retries: int = 3, retry_interval: int = 10, **kwargs) -> pd.DataFrame:
        """
        Async entry point for Tushare API calls. Uses asyncio.Semaphore for
        rate limiting across concurrent gather calls.

        All agents' asyncio.gather parallel calls share the same semaphore instance.

        Input/Output/Raises: same as query()
        """
        self._ensure_async_primitives()

        last_error = None
        for attempt in range(max_retries):
            try:
                async with self._semaphore:
                    # Rate limit within the semaphore
                    async with self._async_lock:
                        now = time_module.time()
                        elapsed = now - self._last_call_time
                        if elapsed < self._min_interval:
                            await asyncio.sleep(self._min_interval - elapsed)
                        self._last_call_time = time_module.time()

                    # Run sync Tushare call in executor to avoid blocking event loop
                    loop = asyncio.get_event_loop()
                    api_func = getattr(self._api, api_name)
                    df = await loop.run_in_executor(None, lambda: api_func(**kwargs))

                    if df is None:
                        df = pd.DataFrame()
                    df = self._validate(df, api_name)
                    return df
            except TushareRequestError:
                raise
            except Exception as e:
                last_error = e
                err_msg = str(e)
                if any(kw in err_msg for kw in ("每分钟最多访问", "exceed", "rate limit", "频率")):
                    wait = 60
                    logger.warning(f"Tushare rate limit hit for {api_name}, waiting {wait}s before retry "
                                   f"(attempt {attempt + 1}/{max_retries})")
                else:
                    wait = retry_interval
                    logger.warning(f"Tushare async request failed (attempt {attempt + 1}/{max_retries}): "
                                 f"{api_name}({kwargs}) -> {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(wait)

        raise TushareRequestError(
            f"All {max_retries} retries failed for {api_name}({kwargs}): {last_error}"
        )

    def batch_query(self, api_name: str, stock_codes: List[str],
                    start_date: str, end_date: str, **kwargs) -> pd.DataFrame:
        """
        Batch query multiple stocks. For BATCH_CAPABLE_APIS, sends comma-separated
        ts_code in chunks of TUSHARE_BATCH_SIZE. For unsupported APIs, falls back
        to sequential per-stock queries.

        Returns a single merged DataFrame with all stocks' data.
        """
        if not stock_codes:
            return pd.DataFrame()

        if api_name not in BATCH_CAPABLE_APIS:
            logger.debug(f"batch_query: {api_name} not batch-capable, falling back to sequential")
            frames = []
            for code in stock_codes:
                try:
                    df = self.query(api_name, ts_code=code,
                                    start_date=start_date, end_date=end_date, **kwargs)
                    if not df.empty:
                        frames.append(df)
                except TushareRequestError as e:
                    logger.warning(f"batch_query sequential fallback failed for {code}: {e}")
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        # Batch-capable: chunk stock_codes and query with comma-separated ts_code
        batch_size = TUSHARE_BATCH_SIZE
        frames = []
        for i in range(0, len(stock_codes), batch_size):
            chunk = stock_codes[i:i + batch_size]
            ts_code_str = ",".join(chunk)
            try:
                df = self.query(api_name, ts_code=ts_code_str,
                                start_date=start_date, end_date=end_date, **kwargs)
                if not df.empty:
                    frames.append(df)
            except TushareRequestError as e:
                logger.warning(f"batch_query chunk failed for {api_name} "
                             f"({len(chunk)} stocks): {e}")

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def query_by_date(self, api_name: str, trade_date: str,
                      enable_cache: bool = False, cache_dir: str = None,
                      **kwargs) -> pd.DataFrame:
        """
        Query full-market data for a single trade date.
        No ts_code param — returns all stocks for that date.
        Supported: daily, daily_basic, adj_factor, suspend_d, stk_limit.
        """
        return self.query(api_name, trade_date=trade_date,
                          enable_cache=enable_cache, cache_dir=cache_dir, **kwargs)

    def query_by_period(self, api_name: str, period: str,
                        enable_cache: bool = False, cache_dir: str = None,
                        **kwargs) -> pd.DataFrame:
        """
        Query full-market financial data for a report period.
        Supported: fina_indicator, balancesheet, cashflow.
        period format: YYYYMMDD (e.g. 20240331 for 2024Q1).
        """
        return self.query(api_name, period=period,
                          enable_cache=enable_cache, cache_dir=cache_dir, **kwargs)

    def cache_stats(self) -> dict:
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total) if total else 0.0
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": round(hit_rate, 4),
        }

    @staticmethod
    def _get_cache_path(cache_dir: str, api_name: str, params: dict) -> Path:
        """Generate deterministic cache file path from API name + params."""
        sorted_params = json_mod.dumps(params, sort_keys=True, default=str)
        key = hashlib.md5(sorted_params.encode()).hexdigest()
        return Path(cache_dir) / api_name / f"{key}.parquet"

    def _validate(self, df: pd.DataFrame, api_name: str) -> pd.DataFrame:
        """
        Data integrity validation, executed after every request.

        Checks:
        - Empty result warning (for unexpected cases)
        - Price fields (open/high/low/close) must not be NaN — drop those rows
        - Outlier detection: single-day change > ±22% triggers WARNING (not dropped)
        - Duplicate detection: dedup by (ts_code, trade_date), keep last
        """
        if df.empty:
            logger.warning(f"Tushare {api_name} returned empty DataFrame")
            return df

        # Price field null check
        price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
        if price_cols:
            null_mask = df[price_cols].isnull().any(axis=1)
            if null_mask.any():
                dropped = null_mask.sum()
                logger.warning(f"Tushare {api_name}: dropping {dropped} rows with null price fields")
                df = df[~null_mask].copy()

        # Outlier detection (pct_chg > ±22%)
        if "pct_chg" in df.columns:
            outliers = df["pct_chg"].abs() > 22
            if outliers.any():
                outlier_codes = df.loc[outliers, "ts_code"].tolist() if "ts_code" in df.columns else []
                logger.warning(f"Tushare {api_name}: {outliers.sum()} rows with |pct_chg| > 22%: {outlier_codes[:5]}")

        # Duplicate detection
        dedup_cols = []
        if "ts_code" in df.columns:
            dedup_cols.append("ts_code")
        if "trade_date" in df.columns:
            dedup_cols.append("trade_date")
        if len(dedup_cols) == 2:
            before = len(df)
            df = df.drop_duplicates(subset=dedup_cols, keep="last")
            after = len(df)
            if before > after:
                logger.warning(f"Tushare {api_name}: removed {before - after} duplicate rows")

        return df

    def query_concept_cached(self, api_name: str, cache_db_path: str = None, **kwargs) -> pd.DataFrame:
        """Query concept/ths_member data with SQLite caching.

        Cache is refreshed every Monday; other days return cached data.
        Supported api_names: concept, concept_detail, ths_member.
        """
        if api_name not in ("concept", "concept_detail", "ths_member"):
            return self.query(api_name, **kwargs)

        if cache_db_path is None:
            cache_db_path = str(Path(__file__).parent.parent / "data" / "concept_cache.db")

        Path(cache_db_path).parent.mkdir(parents=True, exist_ok=True)

        cache_key = f"{api_name}:{json_mod.dumps(kwargs, sort_keys=True, default=str)}"

        conn = sqlite3.connect(cache_db_path)
        conn.execute("""CREATE TABLE IF NOT EXISTS concept_cache (
            cache_key TEXT PRIMARY KEY,
            data TEXT,
            updated_at TEXT
        )""")

        # Check if cache is fresh (updated this week, and today is not Monday or already updated today)
        row = conn.execute(
            "SELECT data, updated_at FROM concept_cache WHERE cache_key = ?",
            (cache_key,)
        ).fetchone()

        now = datetime.now()
        if row is not None:
            updated_at = datetime.fromisoformat(row[1])
            # Refresh on Monday, or if cache is older than 7 days
            is_monday = now.weekday() == 0
            same_day = updated_at.date() == now.date()
            age_days = (now - updated_at).days
            if not (is_monday and not same_day) and age_days < 7:
                try:
                    df = pd.read_json(row[0], orient="records")
                    logger.debug(f"Concept cache hit: {api_name} ({len(df)} rows)")
                    conn.close()
                    return df
                except Exception:
                    pass  # corrupted cache, re-fetch

        # Fetch fresh data
        try:
            df = self.query(api_name, **kwargs)
            if not df.empty:
                data_json = df.to_json(orient="records", force_ascii=False)
                conn.execute(
                    "INSERT OR REPLACE INTO concept_cache (cache_key, data, updated_at) VALUES (?, ?, ?)",
                    (cache_key, data_json, now.isoformat())
                )
                conn.commit()
                logger.info(f"Concept cache updated: {api_name} ({len(df)} rows)")
            conn.close()
            return df
        except Exception as e:
            # On fetch failure, return stale cache if available
            if row is not None:
                try:
                    df = pd.read_json(row[0], orient="records")
                    logger.warning(f"Concept fetch failed, using stale cache: {api_name} -> {e}")
                    conn.close()
                    return df
                except Exception:
                    pass
            conn.close()
            raise

    def get_trade_calendar(self, start: str, end: str) -> List[str]:
        """
        Return list of trade dates (YYYYMMDD strings) between start and end.
        Results are cached in memory (queried once per day).

        Also updates the clock module's trade calendar cache.

        Input:
            start: YYYYMMDD
            end: YYYYMMDD

        Output:
            List of YYYYMMDD strings for trade dates
        """
        if self._trade_cal_cache is not None:
            filtered = [d for d in self._trade_cal_cache if start <= d <= end]
            if filtered:
                return filtered

        df = self.query("trade_cal", exchange="SSE", start_date=start, end_date=end)
        if df is None or df.empty or "is_open" not in df.columns or "cal_date" not in df.columns:
            # Transient empty responses happen occasionally; fall back to in-memory cache when possible.
            if self._trade_cal_cache is not None:
                filtered = [d for d in self._trade_cal_cache if start <= d <= end]
                if filtered:
                    return filtered
            return []
        trade_dates = df[df["is_open"] == 1]["cal_date"].tolist()
        trade_dates = sorted(trade_dates)

        # Update internal cache and clock module
        if self._trade_cal_cache is None:
            self._trade_cal_cache = trade_dates
        else:
            existing = set(self._trade_cal_cache)
            for d in trade_dates:
                if d not in existing:
                    self._trade_cal_cache.append(d)
            self._trade_cal_cache.sort()

        set_trade_calendar(self._trade_cal_cache)
        return trade_dates

    def is_today_trade_date(self, today: str) -> bool:
        """Check if today is a trade date using cached calendar."""
        cal = self.get_trade_calendar(today, today)
        return len(cal) > 0
