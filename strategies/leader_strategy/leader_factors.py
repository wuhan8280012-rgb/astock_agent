#!/usr/bin/env python3
"""
Capital-flow and relative-strength factors for the Leader (龙头) strategy.

These factors are designed to detect institutional accumulation *before*
earnings reports confirm the revenue acceleration — addressing the core
A-share timing problem: "炒预期, 卖事实".

All helpers operate on the pre-grouped ``_stock_data`` dict that the backtest
engine already builds (``{ts_code -> DataFrame indexed by trade_date}``), so
they integrate into ``_score_universe()`` with zero data-pipeline changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════
#  1. Absorption Score — "放量不跌" (volume surge without price drop)
# ══════════════════════════════════════════════════════════════════

def calc_absorption_score(
    hist: pd.DataFrame,
    *,
    turnover_col: str = "amount",
    pct_chg_col: str = "pct_chg",
    ma_window: int = 20,
    min_ratio: float = 2.0,
    max_drop_pct: float = -1.0,
    consecutive_days: int = 3,
) -> float:
    """Detect institutional accumulation via high turnover + flat/rising price.

    Returns a score in [0, 1]:
      - 1.0 if the stock shows sustained high-volume-no-drop over the last
        ``consecutive_days`` trading days.
      - Fractional values for partial signals.
      - 0.0 if no signal.

    Parameters
    ----------
    hist : pd.DataFrame
        Historical daily data for one stock, indexed by trade_date, ascending.
        Must contain ``turnover_col`` and ``pct_chg_col``.
    turnover_col : str
        Column used as turnover/volume proxy.  Default ``amount``.
    ma_window : int
        Moving-average window for baseline turnover.
    min_ratio : float
        Minimum ratio of today's turnover to MA for a "surge" day.
    max_drop_pct : float
        Price change floor (%) — days with ``pct_chg < max_drop_pct`` are
        *not* absorption days (could be panic selling).
    consecutive_days : int
        How many of the most-recent N days must be absorption days for a
        full score.
    """
    if len(hist) < ma_window + consecutive_days:
        return 0.0

    amt = hist[turnover_col].values.astype(float)
    pct = hist[pct_chg_col].values.astype(float)

    # rolling MA of turnover
    ma = pd.Series(amt).rolling(ma_window).mean().values

    # check last ``consecutive_days`` trading days
    check_window = min(consecutive_days + 2, len(amt))  # a bit wider for robustness
    hits = 0
    for offset in range(1, check_window + 1):
        idx = -offset
        if np.isnan(ma[idx]) or ma[idx] <= 0:
            continue
        ratio = amt[idx] / ma[idx]
        drop_ok = pct[idx] >= max_drop_pct
        if ratio >= min_ratio and drop_ok:
            hits += 1

    return min(hits / consecutive_days, 1.0)


# ══════════════════════════════════════════════════════════════════
#  2. Relative Strength vs Industry (RS)
# ══════════════════════════════════════════════════════════════════

def calc_industry_relative_strength(
    stock_closes: np.ndarray,
    industry_ret: float,
    lookback: int = 60,
) -> float | None:
    """Return RS = stock_return(lookback) / industry_return(lookback).

    A value > 1.0 means the stock is outperforming its sector.
    Returns ``None`` if data is insufficient or industry return is ~zero.
    """
    if len(stock_closes) < lookback + 1:
        return None
    stock_ret = stock_closes[-1] / stock_closes[-(lookback + 1)] - 1
    if abs(industry_ret) < 1e-8:
        return None
    return (1 + stock_ret) / (1 + industry_ret)


def is_rs_new_high(
    hist: pd.DataFrame,
    industry_rets_series: pd.Series,
    rs_lookback: int = 60,
    high_lookback: int = 120,
) -> bool:
    """Check if current RS is at a ``high_lookback``-day high.

    Parameters
    ----------
    hist : pd.DataFrame
        Stock daily data, indexed by trade_date, ascending.
    industry_rets_series : pd.Series
        Per-date industry returns over ``rs_lookback`` days.
        Indexed by trade_date.
    """
    if len(hist) < high_lookback + 1:
        return False

    closes = hist["close"].values.astype(float)
    dates = hist.index.values

    # Compute RS for each day in the window
    rs_values = []
    check_dates = dates[-(high_lookback + 1):]
    for i in range(rs_lookback, len(check_dates)):
        d = check_dates[i]
        ind_ret = industry_rets_series.get(d, np.nan)
        if np.isnan(ind_ret):
            continue
        # find position in closes array
        pos = len(closes) - (len(check_dates) - i)
        if pos < rs_lookback:
            continue
        stock_ret = closes[pos] / closes[pos - rs_lookback] - 1
        if abs(ind_ret) < 1e-8:
            continue
        rs_values.append((1 + stock_ret) / (1 + ind_ret))

    if len(rs_values) < 2:
        return False

    current_rs = rs_values[-1]
    prior_max = max(rs_values[:-1])
    return current_rs >= prior_max


# ══════════════════════════════════════════════════════════════════
#  3. Composite Leader Score
# ══════════════════════════════════════════════════════════════════

def calc_leader_composite(
    *,
    momentum_60d: float,
    absorption_score: float,
    rs_vs_industry: float | None,
    rs_is_new_high: bool,
    weights: dict | None = None,
) -> float:
    """Combine factors into a single ranking score (higher = better).

    Default weights:
      - momentum_60d:      0.35
      - absorption_score:  0.25
      - rs_vs_industry:    0.30
      - rs_new_high_bonus: 0.10  (binary)

    All inputs are raw values; ranking is done by the caller.
    """
    w = weights or {
        "momentum": 0.35,
        "absorption": 0.25,
        "rs": 0.30,
        "rs_new_high": 0.10,
    }
    score = w["momentum"] * momentum_60d
    score += w["absorption"] * absorption_score
    if rs_vs_industry is not None:
        score += w["rs"] * rs_vs_industry
    if rs_is_new_high:
        score += w["rs_new_high"]
    return score
