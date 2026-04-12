#!/usr/bin/env python3
"""
Ultra Rotation Strategy – 6-factor scoring engine.

Factor model (all via soft cross-sectional rank blending):
  1. Multi-period momentum composite  (weight: configurable)
  2. Momentum acceleration            (weight: configurable)
  3. Volume surge                      (weight: configurable)
  4. Low volatility                    (weight: configurable)
  5. MA20 angle trend                  (weight: configurable)
  6. Industry relative strength        (weight: configurable)

Design principles (lessons from prior strategies):
  * NO hard filters on signal dimensions – use soft rank-weighting only
  * At most 1 hard buy filter (transition_coef >= threshold)
  * Keep candidate pool >= 100 stocks
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from strategies.f_strategy.scoring import (
    ScoreFilters,
    calc_angle_trend_metrics,
    calc_strength_transition_coef,
)


# ── dataclass for ultra-rotation factor weights ──────────────────

@dataclass(frozen=True)
class UltraFactorWeights:
    """Weights for the 6-factor model (must sum > 0)."""

    momentum_composite: float = 0.25
    momentum_accel: float = 0.20
    volume_surge: float = 0.15
    low_volatility: float = 0.15
    angle_trend: float = 0.15
    industry_rs: float = 0.10


# ── helper: minimum history required ────────────────────────────

def _min_history(lookbacks: list[int], angle_days: int) -> int:
    return max(*lookbacks, angle_days, 60) + 5


# ── filter stage (mirrors F-strategy logic) ─────────────────────

def _passes_filters(
    code: str,
    date: str,
    row: pd.Series,
    hist: pd.DataFrame,
    info: Any,
    *,
    min_price: float,
    min_amount_20d: float,
    min_list_days: int,
    min_history_rows: int,
    filters: ScoreFilters,
) -> bool:
    close = row["close"]
    if pd.isna(close) or close < min_price:
        return False

    # ST filter
    if info is not None:
        name = str(info.get("name", ""))
        if "ST" in name.upper():
            return False
        list_date = str(info.get("list_date", ""))
        if list_date and len(list_date) >= 8:
            try:
                ld = datetime.strptime(list_date[:8], "%Y%m%d")
                cd = datetime.strptime(date, "%Y%m%d")
                if (cd - ld).days < min_list_days:
                    return False
            except Exception:
                pass

    if len(hist) < min_history_rows:
        return False

    # Liquidity
    avg_amount = hist.tail(20)["amount"].mean() * 1000
    if avg_amount < min_amount_20d:
        return False

    # Limit-up filter (can't buy)
    if row["pct_chg"] >= 9.5:
        return False

    # Transition-coef hard filter (only 1 hard filter – proven effective)
    if filters.min_transition_coef is not None:
        if "ma20_angle_deg" not in hist.columns:
            return False
        angles = hist["ma20_angle_deg"].tail(2).values.astype(float)
        if len(angles) < 2 or np.isnan(angles[-1]) or np.isnan(angles[-2]):
            return False
        coef = calc_strength_transition_coef(float(angles[-1]), float(angles[-2]))
        if coef < filters.min_transition_coef:
            return False

    return True


# ── main scoring function ───────────────────────────────────────

def score_universe_ultra(
    stock_data: dict[str, pd.DataFrame],
    basic_map: dict[str, Any],
    date: str,
    *,
    momentum_lookbacks: list[int] | None = None,
    momentum_lookback_weights: list[float] | None = None,
    accel_short_days: int = 5,
    accel_long_days: int = 20,
    vol_surge_short_days: int = 5,
    vol_surge_long_days: int = 20,
    volatility_days: int = 20,
    angle_trend_days: int = 10,
    angle_trend_slope_weight: float = 0.6,
    angle_trend_persistence_weight: float = 0.4,
    factor_weights: UltraFactorWeights | None = None,
    min_price: float = 3.0,
    min_amount_20d: float = 1e8,
    min_list_days: int = 250,
    filters: ScoreFilters | None = None,
    allowed_codes: set[str] | None = None,
) -> list[tuple[str, float]]:
    """Score the universe using 6 orthogonal factors via cross-sectional rank blending.

    Returns [(code, composite_rank), ...] sorted ascending (lower = better).
    """
    if momentum_lookbacks is None:
        momentum_lookbacks = [5, 10, 20, 60]
    if momentum_lookback_weights is None:
        momentum_lookback_weights = [0.15, 0.25, 0.35, 0.25]
    if factor_weights is None:
        factor_weights = UltraFactorWeights()

    active_filters = filters or ScoreFilters()
    min_hist = _min_history(momentum_lookbacks + [accel_long_days, vol_surge_long_days, volatility_days], angle_trend_days)

    records: list[dict] = []
    allowed = set(allowed_codes) if allowed_codes else None

    for code, data in stock_data.items():
        if allowed is not None and code not in allowed:
            continue
        if date not in data.index:
            continue

        row = data.loc[date]
        hist = data[data.index <= date]
        info = basic_map.get(code)

        if not _passes_filters(
            code, date, row, hist, info,
            min_price=min_price,
            min_amount_20d=min_amount_20d,
            min_list_days=min_list_days,
            min_history_rows=min_hist,
            filters=active_filters,
        ):
            continue

        closes = hist["close"].values.astype(float)
        if len(closes) < max(momentum_lookbacks) + 1:
            continue

        # ── Factor 1: Multi-period momentum composite ──
        mom_composite = 0.0
        valid = True
        for lb, w in zip(momentum_lookbacks, momentum_lookback_weights):
            if len(closes) < int(lb) + 1:
                valid = False
                break
            ret = closes[-1] / closes[-(int(lb) + 1)] - 1
            mom_composite += ret * w
        if not valid:
            continue

        # ── Factor 2: Momentum acceleration ──
        # Short-term excess return over long-term daily rate
        if len(closes) < accel_long_days + 1:
            continue
        ret_short = closes[-1] / closes[-(accel_short_days + 1)] - 1
        ret_long = closes[-1] / closes[-(accel_long_days + 1)] - 1
        daily_long_rate = ret_long / accel_long_days if accel_long_days > 0 else 0.0
        expected_short = daily_long_rate * accel_short_days
        mom_accel = ret_short - expected_short

        # ── Factor 3: Volume surge ──
        if "vol" in hist.columns:
            vols = hist["vol"].values.astype(float)
        elif "amount" in hist.columns:
            vols = hist["amount"].values.astype(float)
        else:
            continue

        if len(vols) < vol_surge_long_days:
            continue
        avg_short_vol = np.mean(vols[-vol_surge_short_days:])
        avg_long_vol = np.mean(vols[-vol_surge_long_days:])
        vol_surge = (avg_short_vol / avg_long_vol - 1) if avg_long_vol > 0 else 0.0

        # ── Factor 4: Low volatility ──
        if len(closes) < volatility_days + 1:
            continue
        rets = np.diff(closes[-(volatility_days + 1):]) / closes[-(volatility_days + 1):-1]
        low_vol = -float(np.std(rets))

        # ── Factor 5: MA20 angle trend ──
        angle_score = 0.0
        if "ma20_angle_deg" in hist.columns:
            recent_angles = hist["ma20_angle_deg"].tail(angle_trend_days)
            _, _, a_score = calc_angle_trend_metrics(
                recent_angles,
                slope_weight=angle_trend_slope_weight,
                persistence_weight=angle_trend_persistence_weight,
            )
            if pd.notna(a_score):
                angle_score = float(a_score)
            else:
                angle_score = 0.0

        # ── Factor 6: Industry relative strength ──
        industry_rs = 0.0
        rs_val = row.get("sw_l1_strength20_vs_market", np.nan)
        if pd.notna(rs_val):
            industry_rs = float(rs_val)
        else:
            # Use excess return (difference) to avoid division-by-near-zero instability
            ind_ret = row.get("sw_l1_ret20", np.nan)
            mkt_ret = row.get("market_ret20", np.nan)
            if pd.notna(ind_ret) and pd.notna(mkt_ret):
                industry_rs = float(ind_ret) - float(mkt_ret)

        records.append({
            "code": code,
            "mom_composite": mom_composite,
            "mom_accel": mom_accel,
            "vol_surge": vol_surge,
            "low_vol": low_vol,
            "angle_trend": angle_score,
            "industry_rs": industry_rs,
        })

    if not records:
        return []

    df = pd.DataFrame(records)

    # Cross-sectional ranking (higher value = lower rank number = better)
    df["r_mom"] = df["mom_composite"].rank(ascending=False, method="average")
    df["r_accel"] = df["mom_accel"].rank(ascending=False, method="average")
    df["r_vol"] = df["vol_surge"].rank(ascending=False, method="average")
    df["r_lowvol"] = df["low_vol"].rank(ascending=False, method="average")
    df["r_angle"] = df["angle_trend"].rank(ascending=False, method="average")
    df["r_indrs"] = df["industry_rs"].rank(ascending=False, method="average")

    w = factor_weights
    total_w = (
        w.momentum_composite
        + w.momentum_accel
        + w.volume_surge
        + w.low_volatility
        + w.angle_trend
        + w.industry_rs
    )
    if total_w <= 0:
        total_w = 1.0

    df["composite_rank"] = (
        (w.momentum_composite / total_w) * df["r_mom"]
        + (w.momentum_accel / total_w) * df["r_accel"]
        + (w.volume_surge / total_w) * df["r_vol"]
        + (w.low_volatility / total_w) * df["r_lowvol"]
        + (w.angle_trend / total_w) * df["r_angle"]
        + (w.industry_rs / total_w) * df["r_indrs"]
    )

    df = df.sort_values("composite_rank").reset_index(drop=True)
    return list(zip(df["code"], df["composite_rank"]))
