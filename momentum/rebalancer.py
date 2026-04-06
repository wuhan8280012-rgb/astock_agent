"""
Rebalancing logic for momentum rotation strategy.
Determines what trades to make based on current holdings and target portfolio.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd
from loguru import logger

from momentum.config import MomentumConfig


@dataclass
class Trade:
    """Represents a single buy or sell trade."""

    ts_code: str
    action: str  # "buy" or "sell"
    reason: str
    target_weight: Optional[float] = None


@dataclass
class RebalanceResult:
    """Results of a rebalance calculation."""

    buys: List[Trade] = field(default_factory=list)
    sells: List[Trade] = field(default_factory=list)
    holds: List[Trade] = field(default_factory=list)
    turnover_pct: float = 0.0
    estimated_cost: float = 0.0
    new_weights: dict = field(default_factory=dict)
    rebalance_required: bool = False
    rebalance_reason: str = ""


def compute_rebalance(
    current_holdings: List[dict],
    target_ranking: pd.DataFrame,
    config: MomentumConfig,
    total_capital: float,
    current_prices: dict,
) -> RebalanceResult:
    """
    Compute rebalance trades based on current holdings and target portfolio.

    Args:
        current_holdings: list of dicts with keys: ts_code, shares, entry_price, entry_date
        target_ranking: DataFrame with ranking and momentum scores
        config: MomentumConfig instance
        total_capital: total portfolio capital in CNY
        current_prices: dict mapping ts_code to current close price

    Returns:
        RebalanceResult with computed trades
    """
    result = RebalanceResult()

    if target_ranking.empty:
        logger.warning("Empty target ranking provided to compute_rebalance")
        return result

    # Get current holdings as dict for quick lookup
    current_holdings_dict = {h["ts_code"]: h for h in current_holdings}
    current_positions = set(current_holdings_dict.keys())

    # Get target positions (top N by momentum)
    target_positions = set(target_ranking.head(config.top_n)["ts_code"].tolist())

    logger.info(
        f"Current positions: {len(current_positions)}, "
        f"Target positions: {len(target_positions)}, "
        f"Capital: {total_capital:.2f} CNY"
    )

    # Calculate current portfolio value
    current_value = sum(
        h.get("shares", 0) * current_prices.get(h["ts_code"], 0) for h in current_holdings
    )

    # Determine actions
    to_hold = current_positions & target_positions
    to_sell = current_positions - target_positions
    to_buy = target_positions - current_positions

    logger.debug(
        f"Actions: hold {len(to_hold)}, sell {len(to_sell)}, buy {len(to_buy)}"
    )

    # Calculate target weights (equal weight within top N)
    target_weight = 1.0 / config.top_n if config.top_n > 0 else 0.0

    # Enforce max single weight
    target_weight = min(target_weight, config.max_single_weight)

    # Check if rebalance is needed (using rank change threshold)
    rank_changes = _compute_rank_changes(current_holdings_dict, target_ranking, config)

    rebalance_needed = (len(to_buy) > 0 or len(to_sell) > 0) and any(
        change > config.rebalance_threshold for change in rank_changes.values()
    )

    # Add sells (exits from top N)
    for ts_code in to_sell:
        holding = current_holdings_dict[ts_code]
        current_rank = _find_rank(target_ranking, ts_code)
        reason = f"Dropped out of top {config.top_n} (rank {current_rank})"
        result.sells.append(Trade(ts_code=ts_code, action="sell", reason=reason))

    # Add holds
    for ts_code in to_hold:
        holding = current_holdings_dict[ts_code]
        current_rank = _find_rank(target_ranking, ts_code)
        reason = f"Holding (rank {current_rank})"
        result.holds.append(Trade(ts_code=ts_code, action="hold", reason=reason, target_weight=target_weight))

    # Add buys (entries into top N)
    for ts_code in to_buy:
        current_rank = _find_rank(target_ranking, ts_code)
        reason = f"Entered top {config.top_n} (rank {current_rank})"
        result.buys.append(Trade(ts_code=ts_code, action="buy", reason=reason, target_weight=target_weight))

    # Calculate turnover and costs
    sell_value = sum(
        current_holdings_dict[ts_code].get("shares", 0) * current_prices.get(ts_code, 0)
        for ts_code in to_sell
    )
    turnover = (sell_value / max(current_value, total_capital)) * 100 if current_value > 0 else 0

    # Estimate transaction costs (0.1% per trade for typical A-share commissions + fees)
    num_trades = len(to_sell) + len(to_buy)
    transaction_cost_rate = 0.001  # 0.1%
    estimated_cost = (current_value * transaction_cost_rate * num_trades) if current_value > 0 else 0

    result.turnover_pct = turnover
    result.estimated_cost = estimated_cost
    result.rebalance_required = rebalance_needed

    if rebalance_needed:
        result.rebalance_reason = (
            f"Momentum changes detected: {len(to_buy)} buys, {len(to_sell)} sells, "
            f"turnover {turnover:.1f}%"
        )
    else:
        result.rebalance_reason = "No significant rank changes, holding current portfolio"

    # Calculate new weights
    result.new_weights = {ts_code: target_weight for ts_code in target_positions}

    logger.info(
        f"Rebalance result: buys={len(result.buys)}, sells={len(result.sells)}, "
        f"holds={len(result.holds)}, turnover={turnover:.1f}%, "
        f"cost={estimated_cost:.2f} CNY, required={rebalance_needed}"
    )

    return result


def _compute_rank_changes(
    current_holdings: dict,
    target_ranking: pd.DataFrame,
    config: MomentumConfig,
) -> dict:
    """
    Compute rank changes for each current holding.

    Args:
        current_holdings: dict mapping ts_code to holding dict
        target_ranking: DataFrame with momentum_rank column
        config: MomentumConfig instance

    Returns:
        Dict mapping ts_code to rank change (as percentage of top_n)
    """
    changes = {}

    for ts_code in current_holdings.keys():
        current_rank = current_holdings[ts_code].get("momentum_rank", float("inf"))
        new_rank = _find_rank(target_ranking, ts_code)

        if current_rank == float("inf"):
            # Was not ranked before, treat as large change
            changes[ts_code] = 1.0
        else:
            rank_change = abs(new_rank - current_rank) / max(config.top_n, 1)
            changes[ts_code] = rank_change

    return changes


def _find_rank(target_ranking: pd.DataFrame, ts_code: str) -> int:
    """
    Find rank of a stock in target ranking.

    Args:
        target_ranking: DataFrame with momentum_rank column
        ts_code: stock code to find

    Returns:
        Rank number (1-indexed), or float('inf') if not found
    """
    matches = target_ranking[target_ranking["ts_code"] == ts_code]
    if matches.empty:
        return float("inf")
    return int(matches.iloc[0]["momentum_rank"])


def apply_position_limits(weights: dict, config: MomentumConfig) -> dict:
    """
    Apply position weight limits to portfolio weights.

    Args:
        weights: dict mapping ts_code to weight
        config: MomentumConfig instance

    Returns:
        Adjusted weights dict
    """
    adjusted = {}
    total_weight = 0.0

    # Cap individual positions
    for ts_code, weight in weights.items():
        capped_weight = min(weight, config.max_single_weight)
        adjusted[ts_code] = capped_weight
        total_weight += capped_weight

    # Normalize to sum to 1.0 if needed
    if total_weight > 0:
        for ts_code in adjusted:
            adjusted[ts_code] /= total_weight

    return adjusted
