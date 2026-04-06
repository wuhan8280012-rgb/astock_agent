"""
Momentum score calculation module.
Implements simple, risk-adjusted, and composite momentum calculations.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

from data_pipeline.tushare_client import TushareClient
from momentum.config import MomentumConfig


def calc_simple_momentum(prices: pd.Series, lookback: int) -> float:
    """
    Calculate simple momentum (total return) over lookback period.

    Args:
        prices: Series of prices (must be sorted ascending by date)
        lookback: number of periods to look back

    Returns:
        Simple return over the period (can be negative)
    """
    if len(prices) < lookback + 1:
        return 0.0

    start_price = prices.iloc[-lookback - 1]
    end_price = prices.iloc[-1]

    if start_price <= 0:
        return 0.0

    return (end_price - start_price) / start_price


def calc_risk_adjusted_momentum(prices: pd.Series, lookback: int) -> float:
    """
    Calculate risk-adjusted momentum (Sharpe ratio proxy).
    Returns / Volatility over the period.

    Args:
        prices: Series of prices (must be sorted ascending by date)
        lookback: number of periods to look back

    Returns:
        Risk-adjusted momentum score
    """
    if len(prices) < lookback + 1:
        return 0.0

    # Calculate period returns
    period_prices = prices.iloc[-lookback:]
    daily_returns = period_prices.pct_change().dropna()

    if len(daily_returns) < 2 or daily_returns.std() == 0:
        return 0.0

    # Total return over period
    total_return = (prices.iloc[-1] - prices.iloc[-lookback - 1]) / prices.iloc[-lookback - 1]

    # Annualized volatility (assuming 252 trading days)
    vol = daily_returns.std() * np.sqrt(252)

    if vol == 0:
        return 0.0

    # Risk-adjusted return
    return total_return / vol


def calc_composite_momentum(
    prices_dict: Dict[str, pd.Series],
    config: MomentumConfig,
    ts_code: str,
) -> float:
    """
    Calculate composite momentum score using weighted combination of multiple lookbacks.
    Optionally applies volatility penalty.

    Args:
        prices_dict: dict with ts_code as key, prices (pd.Series) as value
        config: MomentumConfig instance
        ts_code: stock code to calculate momentum for

    Returns:
        Composite momentum score
    """
    if ts_code not in prices_dict:
        return 0.0

    prices = prices_dict[ts_code]

    if prices.empty or len(prices) < max(config.lookback_days) + 1:
        return 0.0

    momentum_score = 0.0

    # Calculate weighted momentum across lookback periods
    for lookback, weight in zip(config.lookback_days, config.lookback_weights):
        if config.momentum_type == "simple":
            mom = calc_simple_momentum(prices, lookback)
        elif config.momentum_type == "risk_adjusted":
            mom = calc_risk_adjusted_momentum(prices, lookback)
        else:  # composite
            mom = calc_simple_momentum(prices, lookback)

        momentum_score += mom * weight

    # Apply volatility penalty if configured
    if config.volatility_penalty > 0:
        daily_returns = prices.pct_change().dropna()
        if len(daily_returns) > 1:
            vol = daily_returns.std()
            # Higher volatility reduces score
            vol_penalty = 1.0 - (config.volatility_penalty * min(vol, 0.1) / 0.1)
            vol_penalty = max(vol_penalty, 0.5)  # Don't penalize too much
            momentum_score *= vol_penalty

    return momentum_score


def rank_by_momentum(
    universe: pd.DataFrame,
    client: TushareClient,
    as_of: str,
    config: MomentumConfig,
) -> pd.DataFrame:
    """
    Rank stocks in universe by momentum score.

    Args:
        universe: DataFrame with at least ts_code column
        client: TushareClient instance
        as_of: reference date (YYYYMMDD format)
        config: MomentumConfig instance

    Returns:
        DataFrame with momentum scores and ranks, sorted by momentum descending
    """
    if universe.empty:
        logger.warning("Empty universe provided to rank_by_momentum")
        return pd.DataFrame()

    ts_codes = universe["ts_code"].tolist()
    logger.info(f"Calculating momentum for {len(ts_codes)} stocks")

    # Fetch price data for all stocks
    try:
        prices_dict = _fetch_prices(client, ts_codes, as_of, config)
    except Exception as e:
        logger.error(f"Failed to fetch prices for momentum calculation: {e}")
        raise

    if not prices_dict:
        logger.warning("No price data fetched for momentum calculation")
        return pd.DataFrame()

    # Calculate momentum for each stock
    momentum_scores = []
    for ts_code in ts_codes:
        try:
            momentum = calc_composite_momentum(prices_dict, config, ts_code)
            momentum_scores.append({"ts_code": ts_code, "momentum_score": momentum})
        except Exception as e:
            logger.debug(f"Failed to calculate momentum for {ts_code}: {e}")
            momentum_scores.append({"ts_code": ts_code, "momentum_score": 0.0})

    momentum_df = pd.DataFrame(momentum_scores)

    if momentum_df.empty:
        logger.warning("No momentum scores calculated")
        return pd.DataFrame()

    # Merge with original universe data
    result = universe.merge(momentum_df, on="ts_code", how="left")

    # Rank by momentum score (higher is better)
    result["momentum_rank"] = result["momentum_score"].rank(ascending=False, method="min")

    # Sort by momentum score descending
    result = result.sort_values("momentum_score", ascending=False).reset_index(drop=True)

    logger.info(
        f"Ranked {len(result)} stocks by momentum. "
        f"Top: {result.iloc[0]['ts_code']} (score={result.iloc[0]['momentum_score']:.4f})"
    )

    return result


def _fetch_prices(
    client: TushareClient,
    ts_codes: list,
    as_of: str,
    config: MomentumConfig,
    lookback_buffer: int = 20,
) -> Dict[str, pd.Series]:
    """
    Fetch daily price data for stocks.

    Args:
        client: TushareClient instance
        ts_codes: list of ts_code strings
        as_of: reference date (YYYYMMDD format)
        config: MomentumConfig instance
        lookback_buffer: additional days to fetch for robustness

    Returns:
        Dict mapping ts_code to pd.Series of closing prices
    """
    # Calculate required lookback (max lookback_days + buffer)
    max_lookback = max(config.lookback_days) + lookback_buffer
    logger.debug(f"Fetching {max_lookback} days of price data for {len(ts_codes)} stocks")

    try:
        daily_data = client.daily(
            ts_code=",".join(ts_codes),
            end_date=as_of,
            fields=["ts_code", "trade_date", "close"],
        )
    except Exception as e:
        logger.error(f"Failed to fetch daily data: {e}")
        raise

    if daily_data.empty:
        logger.warning(f"No daily data returned for {len(ts_codes)} stocks")
        return {}

    # Convert close to numeric
    daily_data["close"] = pd.to_numeric(daily_data["close"], errors="coerce")
    daily_data = daily_data.dropna(subset=["close"])

    # Convert trade_date to datetime
    daily_data["trade_date"] = pd.to_datetime(daily_data["trade_date"], format="%Y%m%d", errors="coerce")

    # Sort by ts_code and trade_date
    daily_data = daily_data.sort_values(["ts_code", "trade_date"])

    # Create prices_dict: ts_code -> Series of closing prices
    prices_dict = {}
    for ts_code, group in daily_data.groupby("ts_code"):
        # Sort by date and reset index
        prices = group.set_index("trade_date")["close"].sort_index()
        if len(prices) > max(config.lookback_days):
            # Keep only recent data
            prices = prices.iloc[-max_lookback:]
            prices_dict[ts_code] = prices

    logger.debug(f"Fetched price data for {len(prices_dict)}/{len(ts_codes)} stocks")

    return prices_dict
