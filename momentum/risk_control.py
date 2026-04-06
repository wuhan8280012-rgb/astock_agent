"""
Risk management and control module for momentum rotation strategy.
Handles regime filters, stop losses, position limits, and circuit breakers.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from data_pipeline.tushare_client import TushareClient
from momentum.config import MomentumConfig


@dataclass
class RegimeInfo:
    """Information about current market regime."""

    regime: str  # "HALT" | "DEFENSIVE" | "NEUTRAL" | "RUN" | "STRONG_RUN"
    regime_date: str
    cash_pct: float
    max_positions: Optional[int]
    signal_strength: float  # 0-100 indicator of confidence


def check_regime_filter(
    client: TushareClient,
    as_of: str,
    config: MomentumConfig,
) -> RegimeInfo:
    """
    Check market regime and determine appropriate cash/position limits.

    Args:
        client: TushareClient instance
        as_of: reference date (YYYYMMDD format)
        config: MomentumConfig instance

    Returns:
        RegimeInfo with regime classification and policy adjustments
    """
    if not config.regime_filter_enabled:
        return RegimeInfo(
            regime="NEUTRAL",
            regime_date=as_of,
            cash_pct=0.0,
            max_positions=None,
            signal_strength=50.0,
        )

    logger.debug(f"Checking market regime as of {as_of}")

    try:
        # Fetch recent market breadth data (CSI 300 as proxy for regime)
        market_data = _fetch_market_breadth(client, as_of)
    except Exception as e:
        logger.warning(f"Failed to fetch market breadth data: {e}, defaulting to NEUTRAL")
        return RegimeInfo(
            regime="NEUTRAL",
            regime_date=as_of,
            cash_pct=0.0,
            max_positions=None,
            signal_strength=50.0,
        )

    # Classify regime based on breadth and momentum indicators
    regime, strength = _classify_regime(market_data)

    # Determine cash and position limits based on regime
    if regime == "HALT":
        cash_pct = config.regime_halt_cash_pct
        max_positions = 0
    elif regime == "DEFENSIVE":
        cash_pct = config.regime_defensive_cash_pct
        max_positions = max(config.top_n // 2, 1)
    else:  # RUN or STRONG_RUN
        cash_pct = 0.0
        max_positions = None

    logger.info(
        f"Market regime: {regime} (strength={strength:.1f}, "
        f"cash_pct={cash_pct*100:.1f}%, max_positions={max_positions})"
    )

    return RegimeInfo(
        regime=regime,
        regime_date=as_of,
        cash_pct=cash_pct,
        max_positions=max_positions,
        signal_strength=strength,
    )


def apply_stop_loss(
    holdings: List[dict],
    current_prices: dict,
    config: MomentumConfig,
) -> tuple[List[dict], List[str]]:
    """
    Check trailing stop loss and identify positions to liquidate.

    Args:
        holdings: list of dicts with keys: ts_code, shares, entry_price
        current_prices: dict mapping ts_code to current close price
        config: MomentumConfig instance

    Returns:
        Tuple of (remaining_holdings, stopped_out_codes)
    """
    remaining = []
    stopped_out = []

    for holding in holdings:
        ts_code = holding["ts_code"]
        entry_price = holding.get("entry_price", 0)
        current_price = current_prices.get(ts_code, 0)

        if entry_price <= 0 or current_price <= 0:
            remaining.append(holding)
            continue

        # Calculate loss percentage
        loss_pct = (current_price - entry_price) / entry_price

        if loss_pct <= config.stop_loss_pct:
            # Stop loss triggered
            stopped_out.append(ts_code)
            logger.info(
                f"Stop loss triggered for {ts_code}: "
                f"entry={entry_price:.2f}, current={current_price:.2f}, "
                f"loss={loss_pct*100:.2f}%"
            )
        else:
            remaining.append(holding)

    return remaining, stopped_out


def apply_position_limits(
    target_weights: dict,
    config: MomentumConfig,
    regime_info: RegimeInfo,
) -> dict:
    """
    Apply position weight limits based on config and regime.

    Args:
        target_weights: dict mapping ts_code to target weight
        config: MomentumConfig instance
        regime_info: RegimeInfo with regime-based limits

    Returns:
        Adjusted weights dict
    """
    adjusted = {}

    # Apply max single weight limit
    for ts_code, weight in target_weights.items():
        capped_weight = min(weight, config.max_single_weight)
        adjusted[ts_code] = capped_weight

    # Apply regime-based position limit
    if regime_info.max_positions and len(adjusted) > regime_info.max_positions:
        # Keep only top positions
        sorted_positions = sorted(
            adjusted.items(), key=lambda x: x[1], reverse=True
        )[:regime_info.max_positions]
        adjusted = {ts_code: weight for ts_code, weight in sorted_positions}

    # Add cash component based on regime
    total_weight = sum(adjusted.values())
    if total_weight > 0 and regime_info.cash_pct > 0:
        # Scale down equity weights to accommodate cash
        scale_factor = 1.0 - regime_info.cash_pct
        adjusted = {
            ts_code: weight * scale_factor for ts_code, weight in adjusted.items()
        }
        adjusted["CASH"] = regime_info.cash_pct

    # Normalize if needed
    total_weight = sum(v for k, v in adjusted.items() if k != "CASH")
    if total_weight > 0 and total_weight != 1.0:
        for ts_code in adjusted:
            if ts_code != "CASH":
                adjusted[ts_code] /= total_weight

    return adjusted


def check_drawdown(
    portfolio_value: float,
    peak_value: float,
    max_drawdown_threshold: float = -0.20,
) -> tuple[bool, float]:
    """
    Check if portfolio drawdown exceeds maximum allowed.

    Args:
        portfolio_value: current portfolio value
        peak_value: historical peak value
        max_drawdown_threshold: maximum allowed drawdown (e.g., -0.20 = -20%)

    Returns:
        Tuple of (is_ok, current_drawdown)
    """
    if peak_value <= 0:
        return True, 0.0

    drawdown = (portfolio_value - peak_value) / peak_value

    is_ok = drawdown >= max_drawdown_threshold

    if not is_ok:
        logger.warning(
            f"Drawdown circuit breaker triggered: "
            f"current={portfolio_value:.2f}, peak={peak_value:.2f}, "
            f"drawdown={drawdown*100:.2f}%"
        )

    return is_ok, drawdown


def _fetch_market_breadth(client: TushareClient, as_of: str) -> dict:
    """
    Fetch market breadth indicators.

    Args:
        client: TushareClient instance
        as_of: reference date (YYYYMMDD format)

    Returns:
        Dict with breadth metrics
    """
    try:
        # Fetch CSI 300 index as proxy for broad market
        index_daily = client.index_daily(
            ts_code="000300.SH",
            end_date=as_of,
            fields=["trade_date", "close", "pct_chg"],
        )
    except Exception as e:
        logger.debug(f"Failed to fetch index data: {e}")
        return {}

    if index_daily.empty:
        return {}

    # Simple breadth: positive if index up, negative if down
    latest = index_daily.iloc[0]
    pct_chg = float(latest.get("pct_chg", 0))

    # Calculate 5-day breadth (positive close count)
    index_daily_recent = index_daily.head(5)
    positive_days = (
        pd.to_numeric(index_daily_recent["pct_chg"], errors="coerce") > 0
    ).sum()

    breadth_ratio = positive_days / max(len(index_daily_recent), 1)

    return {
        "index_pct_chg": pct_chg,
        "breadth_ratio": breadth_ratio,
        "positive_days": positive_days,
    }


def _classify_regime(market_data: dict) -> tuple[str, float]:
    """
    Classify market regime based on breadth and momentum.

    Args:
        market_data: dict with market metrics

    Returns:
        Tuple of (regime_name, signal_strength_0_100)
    """
    if not market_data:
        return "NEUTRAL", 50.0

    pct_chg = market_data.get("index_pct_chg", 0)
    breadth_ratio = market_data.get("breadth_ratio", 0.5)

    # Simple regime classification
    if pct_chg < -2.0:
        regime = "HALT"
        strength = abs(pct_chg) * 10
    elif pct_chg < -0.5:
        regime = "DEFENSIVE"
        strength = abs(pct_chg) * 10
    elif pct_chg > 2.0 and breadth_ratio > 0.6:
        regime = "STRONG_RUN"
        strength = pct_chg * 10
    elif pct_chg > 0.5:
        regime = "RUN"
        strength = pct_chg * 10
    else:
        regime = "NEUTRAL"
        strength = 50.0

    # Clamp strength to 0-100
    strength = max(0.0, min(100.0, strength))

    return regime, strength
