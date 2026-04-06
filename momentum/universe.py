"""
Universe filtering module for momentum rotation strategy.
Filters A-share stocks based on market cap, turnover, and other criteria.
"""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from loguru import logger

from data_pipeline.tushare_client import TushareClient
from momentum.config import MomentumConfig


def filter_universe(
    client: TushareClient,
    as_of: str,
    config: MomentumConfig,
) -> pd.DataFrame:
    """
    Filter A-share universe based on config criteria.

    Args:
        client: TushareClient instance
        as_of: date string (YYYYMMDD format)
        config: MomentumConfig instance

    Returns:
        DataFrame with columns: ts_code, name, industry, market_cap, avg_turnover
        Sorted by market cap descending.
    """
    logger.info(
        f"Filtering A-share universe as of {as_of} "
        f"(min_cap={config.universe_min_market_cap}B, "
        f"min_turnover={config.universe_min_avg_turnover}M)"
    )

    # Get all A-share stocks
    try:
        stock_basic = client.stock_basic(list_status="L", exchange="")
    except Exception as e:
        logger.error(f"Failed to fetch stock_basic: {e}")
        raise

    if stock_basic.empty:
        logger.warning("No stocks found in stock_basic")
        return pd.DataFrame()

    logger.debug(f"Fetched {len(stock_basic)} total stocks")

    # Filter by exchange (A-shares are Shanghai and Shenzhen)
    stock_basic = stock_basic[stock_basic["exchange"].isin(["SSE", "SZSE"])].copy()
    logger.debug(f"After exchange filter: {len(stock_basic)} stocks")

    # Exclude ST stocks if configured
    if config.universe_exclude_st:
        initial_count = len(stock_basic)
        stock_basic = stock_basic[~stock_basic["name"].str.contains("ST", na=False)].copy()
        excluded_st = initial_count - len(stock_basic)
        logger.debug(f"Excluded {excluded_st} ST stocks, remaining: {len(stock_basic)}")

    # Exclude new listings if configured
    if config.universe_exclude_new_days > 0:
        try:
            as_of_dt = datetime.strptime(as_of, "%Y%m%d")
            cutoff_date = as_of_dt - timedelta(days=config.universe_exclude_new_days)

            # list_date is in YYYYMMDD format
            stock_basic["list_date"] = pd.to_datetime(
                stock_basic["list_date"], format="%Y%m%d", errors="coerce"
            )
            initial_count = len(stock_basic)
            stock_basic = stock_basic[stock_basic["list_date"] <= cutoff_date].copy()
            excluded_new = initial_count - len(stock_basic)
            logger.debug(
                f"Excluded {excluded_new} new listings (<{config.universe_exclude_new_days} days), "
                f"remaining: {len(stock_basic)}"
            )
        except Exception as e:
            logger.warning(f"Failed to filter by listing date: {e}")

    if stock_basic.empty:
        logger.warning("No stocks remaining after exchange/ST/age filters")
        return pd.DataFrame()

    # Fetch market cap and turnover data for filtered stocks
    ts_codes = stock_basic["ts_code"].tolist()
    logger.debug(f"Fetching market data for {len(ts_codes)} stocks")

    # Get market cap from daily_basic (Tushare's liquidity/valuation data)
    try:
        daily_basic = client.daily_basic(
            trade_date=as_of,
            fields=["ts_code", "trade_date", "total_mv", "turnover"],
        )
    except Exception as e:
        logger.error(f"Failed to fetch daily_basic: {e}")
        raise

    if daily_basic.empty:
        logger.warning(f"No daily_basic data for {as_of}")
        return pd.DataFrame()

    logger.debug(f"Fetched daily_basic data for {len(daily_basic)} stocks")

    # Merge with stock_basic
    merged = stock_basic.merge(daily_basic, on="ts_code", how="inner")

    if merged.empty:
        logger.warning("No matches between stock_basic and daily_basic")
        return pd.DataFrame()

    logger.debug(f"After merge with daily_basic: {len(merged)} stocks")

    # Calculate market cap in billions CNY (total_mv is in billions)
    # Filter by market cap
    merged["market_cap"] = pd.to_numeric(merged["total_mv"], errors="coerce")
    initial_count = len(merged)
    merged = merged[merged["market_cap"] >= config.universe_min_market_cap].copy()
    excluded_cap = initial_count - len(merged)
    logger.debug(
        f"Excluded {excluded_cap} stocks by market cap (<{config.universe_min_market_cap}B), "
        f"remaining: {len(merged)}"
    )

    if merged.empty:
        logger.warning(f"No stocks meet market cap filter (>{config.universe_min_market_cap}B)")
        return pd.DataFrame()

    # Filter by average daily turnover
    # turnover from daily_basic is already in %
    # We need to calculate turnover in millions CNY using price and volume
    # For now, use turnover rate as proxy and fetch daily data to compute
    merged["turnover_rate"] = pd.to_numeric(merged.get("turnover", 0), errors="coerce")

    # Fetch recent daily data to compute average turnover
    try:
        avg_turnover_data = _compute_avg_turnover(client, merged["ts_code"].tolist(), as_of)
        merged = merged.merge(avg_turnover_data, on="ts_code", how="inner")
    except Exception as e:
        logger.warning(f"Failed to compute average turnover: {e}. Using turnover rate as fallback.")
        merged["avg_turnover"] = merged["turnover_rate"]

    # Filter by minimum average turnover
    initial_count = len(merged)
    merged = merged[merged["avg_turnover"] >= config.universe_min_avg_turnover].copy()
    excluded_turnover = initial_count - len(merged)
    logger.debug(
        f"Excluded {excluded_turnover} stocks by avg turnover (<{config.universe_min_avg_turnover}M), "
        f"remaining: {len(merged)}"
    )

    if merged.empty:
        logger.warning(f"No stocks meet turnover filter (>{config.universe_min_avg_turnover}M)")
        return pd.DataFrame()

    # Exclude suspended stocks
    try:
        suspended = _get_suspended_stocks(client, as_of)
        initial_count = len(merged)
        merged = merged[~merged["ts_code"].isin(suspended)].copy()
        excluded_suspended = initial_count - len(merged)
        logger.debug(f"Excluded {excluded_suspended} suspended stocks, remaining: {len(merged)}")
    except Exception as e:
        logger.warning(f"Failed to filter suspended stocks: {e}")

    # Select and rename output columns
    result = merged[["ts_code", "name", "industry", "market_cap", "avg_turnover"]].copy()
    result = result.drop_duplicates(subset=["ts_code"])
    result = result.sort_values("market_cap", ascending=False).reset_index(drop=True)

    logger.info(f"Filtered universe: {len(result)} stocks remaining")
    logger.debug(f"Top 10 by market cap: {result.head(10)['ts_code'].tolist()}")

    return result


def _compute_avg_turnover(client: TushareClient, ts_codes: list, as_of: str, lookback: int = 20) -> pd.DataFrame:
    """
    Compute average daily turnover (in millions CNY) over lookback period.

    Args:
        client: TushareClient instance
        ts_codes: list of ts_code strings
        as_of: reference date (YYYYMMDD)
        lookback: number of days to look back

    Returns:
        DataFrame with ts_code and avg_turnover columns
    """
    logger.debug(f"Computing average turnover for {len(ts_codes)} stocks (lookback={lookback}d)")

    try:
        # Fetch daily data with vol and close
        daily_data = client.daily(
            ts_code=",".join(ts_codes),
            start_date=None,  # Let Tushare handle date range based on trade_date
            end_date=as_of,
            fields=["ts_code", "trade_date", "close", "vol"],
        )
    except Exception as e:
        logger.error(f"Failed to fetch daily data for turnover: {e}")
        raise

    if daily_data.empty:
        raise ValueError("No daily data returned for turnover calculation")

    # Convert to numeric
    daily_data["close"] = pd.to_numeric(daily_data["close"], errors="coerce")
    daily_data["vol"] = pd.to_numeric(daily_data["vol"], errors="coerce")

    # Turnover in CNY = vol * close (vol is in shares, close in CNY per share)
    daily_data["turnover_cny"] = daily_data["vol"] * daily_data["close"] / 1_000_000  # Convert to millions

    # Group by ts_code and calculate average
    avg_turnover = daily_data.groupby("ts_code")["turnover_cny"].mean().reset_index()
    avg_turnover.columns = ["ts_code", "avg_turnover"]

    logger.debug(f"Computed avg turnover for {len(avg_turnover)} stocks")

    return avg_turnover


def _get_suspended_stocks(client: TushareClient, as_of: str) -> set:
    """
    Get set of suspended stocks as of given date.

    Args:
        client: TushareClient instance
        as_of: date string (YYYYMMDD format)

    Returns:
        Set of ts_code strings for suspended stocks
    """
    try:
        suspend_data = client.suspend_d(trade_date=as_of, fields=["ts_code", "suspend_date", "resume_date"])
    except Exception as e:
        logger.warning(f"Failed to fetch suspend data: {e}")
        return set()

    if suspend_data.empty:
        return set()

    if "suspend_date" not in suspend_data.columns or "resume_date" not in suspend_data.columns:
        return set(pd.Series(suspend_data.get("ts_code", [])).dropna().astype(str).tolist())

    as_of_dt = datetime.strptime(as_of, "%Y%m%d")
    suspend_data["suspend_date"] = pd.to_datetime(suspend_data["suspend_date"], format="%Y%m%d", errors="coerce")
    suspend_data["resume_date"] = pd.to_datetime(suspend_data["resume_date"], format="%Y%m%d", errors="coerce")

    # Stock is suspended if suspend_date <= as_of and resume_date > as_of
    suspended = suspend_data[
        (suspend_data["suspend_date"] <= as_of_dt) & (suspend_data["resume_date"] > as_of_dt)
    ]

    return set(suspended["ts_code"].tolist())
