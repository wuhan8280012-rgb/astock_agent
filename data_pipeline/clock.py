"""
Time gating module — the most critical module in the system.
ALL data fetching MUST go through this module to prevent lookahead bias.

Data availability rules (hardcoded, not configurable):
- daily_price:    available after T 16:00
- margin_balance: available after T+1 09:00
- top_list:       available after T 18:00
- announcement:   conservative — only announcements before T 09:00 count for T
- etf_flow:       available after T 17:00
- index_daily:    available after T 16:00 (same as daily_price)
"""

from datetime import datetime, time, timedelta
from typing import List, Optional

import pandas as pd

# Trade calendar cache (populated by tushare_client)
_trade_calendar: Optional[List[str]] = None


def set_trade_calendar(dates: List[str]):
    """Set the trade calendar cache. Called by tushare_client on init."""
    global _trade_calendar
    _trade_calendar = sorted(dates)


def get_trade_calendar() -> List[str]:
    """Get the cached trade calendar."""
    if _trade_calendar is None:
        raise RuntimeError("Trade calendar not initialized. Call set_trade_calendar() first.")
    return _trade_calendar


def is_trade_date(date_str: str) -> bool:
    """Check if a date string (YYYYMMDD) is a trade date."""
    cal = get_trade_calendar()
    return date_str in cal


def prev_trade_date(date_str: str) -> str:
    """Get the previous trade date before date_str (YYYYMMDD)."""
    cal = get_trade_calendar()
    for d in reversed(cal):
        if d < date_str:
            return d
    raise ValueError(f"No trade date found before {date_str}")


def _to_yyyymmdd(dt: datetime) -> str:
    return dt.strftime("%Y%m%d")


# Data availability rules: (hour, minute) after which T-day data is available
_AVAILABILITY_RULES = {
    "daily_price":    {"same_day_after": time(16, 0)},
    "index_daily":    {"same_day_after": time(16, 0)},
    "margin_balance": {"next_day_after": time(9, 0)},
    "top_list":       {"same_day_after": time(18, 0)},
    "announcement":   {"same_day_before": time(9, 0)},
    "etf_flow":       {"same_day_after": time(17, 0)},
}


def get_data_available_at(data_type: str, as_of: datetime) -> str:
    """
    Return the latest data date (YYYYMMDD) truly available at as_of time.

    If the data for the current day isn't available yet, rolls back to the
    previous trade date. If that date is not a trade date, continues rolling back.

    Input:
        data_type: one of the keys in _AVAILABILITY_RULES
        as_of: the current datetime to evaluate against

    Output:
        YYYYMMDD string of the latest available data date

    Raises:
        ValueError if data_type is unknown
    """
    if data_type not in _AVAILABILITY_RULES:
        raise ValueError(f"Unknown data_type: {data_type}. Valid types: {list(_AVAILABILITY_RULES.keys())}")

    rule = _AVAILABILITY_RULES[data_type]
    today_str = _to_yyyymmdd(as_of)
    current_time = as_of.time()

    if "same_day_after" in rule:
        # T-day data available after the specified time on T
        threshold = rule["same_day_after"]
        if current_time >= threshold and is_trade_date(today_str):
            return today_str
        else:
            return _find_prev_available_trade_date(today_str)

    elif "next_day_after" in rule:
        # T-day data available after specified time on T+1
        # So at as_of, the latest available is T-1 data (if as_of time >= threshold)
        # or T-2 data (if as_of time < threshold)
        threshold = rule["next_day_after"]
        if current_time >= threshold:
            # We can get yesterday's data (the previous trade date before today)
            return _find_prev_available_trade_date(today_str)
        else:
            # We can only get the day before yesterday's data
            prev = _find_prev_available_trade_date(today_str)
            return _find_prev_available_trade_date(prev)

    elif "same_day_before" in rule:
        # Only announcements before the threshold on T count for T
        # After the threshold, T-day announcements are considered for next trade date,
        # so the latest available data rolls back to the previous trade date.
        threshold = rule["same_day_before"]
        if current_time < threshold and is_trade_date(today_str):
            return today_str
        else:
            return _find_prev_available_trade_date(today_str)

    raise ValueError(f"Invalid rule configuration for {data_type}")


def _find_prev_available_trade_date(date_str: str) -> str:
    """Find the most recent trade date strictly before date_str."""
    cal = get_trade_calendar()
    for d in reversed(cal):
        if d < date_str:
            return d
    raise ValueError(f"No trade date found before {date_str} in calendar")


def assert_no_lookahead(data_df: pd.DataFrame, as_of: datetime, date_col: str, data_type: str = "daily_price"):
    """
    Check that no rows in data_df have dates beyond what's available at as_of.

    This function MUST be called after every data query — no exceptions.

    Input:
        data_df: the queried DataFrame
        as_of: the current evaluation time
        date_col: the column name containing date strings (YYYYMMDD format)
        data_type: the type of data for availability lookup

    Raises:
        LookaheadError if any future data is detected
    """
    if data_df.empty:
        return

    available_date = get_data_available_at(data_type, as_of)

    # Normalize date column to string for comparison
    dates = data_df[date_col].astype(str).str.replace("-", "")
    future_mask = dates > available_date
    if future_mask.any():
        future_dates = dates[future_mask].unique().tolist()
        raise LookaheadError(
            f"Lookahead bias detected! Data type '{data_type}' at as_of={as_of}: "
            f"available up to {available_date}, but found dates: {future_dates}"
        )


class LookaheadError(Exception):
    """Raised when lookahead bias is detected in data."""
    pass
