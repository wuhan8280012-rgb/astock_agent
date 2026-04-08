#!/usr/bin/env python3
"""Shared F-strategy scoring logic for backtest and live signal generation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScoreFilters:
    min_ma20_angle: float | None = None
    min_transition_coef: float | None = None
    min_industry_strength20_vs_market: float | None = None


def calc_strength_transition_coef(a0: float, a1: float) -> float:
    base = np.tanh(a0 / 10.0)
    turn = np.tanh((a0 - a1) / 5.0)
    return float(np.clip(0.7 * base + 0.3 * turn, -1.0, 1.0))


def calc_angle_trend_metrics(
    angle_series: pd.Series | np.ndarray,
    slope_weight: float = 0.6,
    persistence_weight: float = 0.4,
) -> tuple[float, float, float]:
    """Quantify MA20 angle trend by combining slope and persistence.

    slope:
        Linear-regression slope of the recent ma20_angle_deg series.
    persistence:
        Blend of "angle stays above zero" and "angle keeps improving".
    score:
        Bounded composite for cross-sectional ranking.
    """
    angles = pd.Series(angle_series, dtype="float64").dropna().to_numpy()
    if len(angles) < 3:
        return np.nan, np.nan, np.nan

    x = np.arange(len(angles), dtype=float)
    slope = float(np.polyfit(x, angles, 1)[0])
    positive_angle_ratio = float((angles > 0).mean())
    diffs = np.diff(angles)
    improving_ratio = float((diffs > 0).mean()) if len(diffs) else np.nan
    persistence = 0.5 * positive_angle_ratio + 0.5 * improving_ratio
    score = slope_weight * np.tanh(slope / 1.5) + persistence_weight * (2.0 * persistence - 1.0)
    return slope, float(persistence), float(score)


def _min_history_required(cfg: Any) -> int:
    lookbacks = [60]
    momentum_days = getattr(cfg, "momentum_days", None)
    if momentum_days:
        lookbacks.extend(int(lb) for lb in momentum_days)
    if getattr(cfg, "use_reversal_factor", False):
        lookbacks.append(int(getattr(cfg, "reversal_days", 20)))
    if getattr(cfg, "use_volatility_factor", False):
        lookbacks.append(int(getattr(cfg, "volatility_days", 60)))
    if getattr(cfg, "subtract_short_momentum", False):
        lookbacks.append(int(getattr(cfg, "short_momentum_days", 20)))
    if getattr(cfg, "use_angle_trend_factor", False):
        lookbacks.append(int(getattr(cfg, "angle_trend_days", 10)))
    return max(lookbacks) + 5


def _passes_filters(
    code: str,
    date: str,
    row: pd.Series,
    hist: pd.DataFrame,
    info: Any,
    cfg: Any,
    filters: ScoreFilters,
) -> bool:
    close = row["close"]
    if pd.isna(close) or close < cfg.min_price:
        return False

    if info is not None:
        name = str(info.get("name", ""))
        if "ST" in name.upper():
            return False
        list_date = str(info.get("list_date", ""))
        if list_date and len(list_date) >= 8:
            try:
                ld = datetime.strptime(list_date[:8], "%Y%m%d")
                cd = datetime.strptime(date, "%Y%m%d")
                if (cd - ld).days < cfg.min_list_days:
                    return False
            except Exception:
                pass

    if len(hist) < _min_history_required(cfg):
        return False

    avg_amount = hist.tail(20)["amount"].mean() * 1000
    if avg_amount < cfg.min_amount_20d:
        return False

    if row["pct_chg"] >= 9.5:
        return False

    if filters.min_ma20_angle is not None:
        angle = row.get("ma20_angle_deg", np.nan)
        if pd.isna(angle) or float(angle) < filters.min_ma20_angle:
            return False

    if filters.min_transition_coef is not None:
        if "ma20_angle_deg" not in hist.columns:
            return False
        angles = hist["ma20_angle_deg"].tail(2).values.astype(float)
        if len(angles) < 2 or np.isnan(angles[-1]) or np.isnan(angles[-2]):
            return False
        coef = calc_strength_transition_coef(float(angles[-1]), float(angles[-2]))
        if coef < filters.min_transition_coef:
            return False

    if filters.min_industry_strength20_vs_market is not None:
        ratio = row.get("sw_l1_strength20_vs_market", np.nan)
        if pd.isna(ratio) or float(ratio) < filters.min_industry_strength20_vs_market:
            return False

    return True


def score_universe(
    stock_data: dict[str, pd.DataFrame],
    basic_map: dict[str, Any],
    cfg: Any,
    date: str,
    filters: ScoreFilters | None = None,
    allowed_codes: set[str] | None = None,
) -> list[tuple[str, float]]:
    """Score the stock universe and return composite ranks in ascending order."""
    active_filters = filters or ScoreFilters()
    scores: list[tuple[str, float, float, float, float]] = []
    allowed = set(allowed_codes) if allowed_codes else None

    for code, data in stock_data.items():
        if allowed is not None and code not in allowed:
            continue
        if date not in data.index:
            continue

        row = data.loc[date]
        hist = data[data.index <= date]
        info = basic_map.get(code)
        if not _passes_filters(code, date, row, hist, info, cfg, active_filters):
            continue

        closes = hist["close"].values.astype(float)
        factor_score = 0.0

        if getattr(cfg, "use_reversal_factor", False):
            rev_lb = int(getattr(cfg, "reversal_days", 20))
            if len(closes) < rev_lb + 1:
                continue
            past_ret = closes[-1] / closes[-(rev_lb + 1)] - 1
            factor_score = -past_ret
        else:
            momentum_days = getattr(cfg, "momentum_days", [60])
            momentum_weights = getattr(cfg, "momentum_weights", [1.0])
            valid = True
            for lb, weight in zip(momentum_days, momentum_weights):
                if len(closes) < int(lb) + 1:
                    valid = False
                    break
                ret = closes[-1] / closes[-(int(lb) + 1)] - 1
                factor_score += ret * float(weight)
            if not valid:
                continue

            if getattr(cfg, "subtract_short_momentum", False):
                short_days = int(getattr(cfg, "short_momentum_days", 20))
                if len(closes) >= short_days + 1:
                    short_ret = closes[-1] / closes[-(short_days + 1)] - 1
                    factor_score -= short_ret

        vol_component = 0.0
        if getattr(cfg, "use_volatility_factor", False):
            vol_days = int(getattr(cfg, "volatility_days", 60))
            if len(closes) >= vol_days + 1:
                rets = np.diff(closes[-vol_days - 1:]) / closes[-vol_days - 1:-1]
                vol = np.std(rets)
                if vol > 0:
                    vol_component = -vol

        size_component = 0.0
        if getattr(cfg, "use_size_factor", False):
            circ_mv = row.get("circ_mv", None)
            if circ_mv and not pd.isna(circ_mv) and float(circ_mv) > 0:
                size_component = -np.log(float(circ_mv))

        angle_component = 0.0
        if getattr(cfg, "use_angle_trend_factor", False):
            angle_days = int(getattr(cfg, "angle_trend_days", 10))
            if "ma20_angle_deg" not in hist.columns:
                continue
            recent_angles = hist["ma20_angle_deg"].tail(angle_days)
            _, _, angle_component = calc_angle_trend_metrics(
                recent_angles,
                slope_weight=float(getattr(cfg, "angle_trend_slope_weight", 0.6)),
                persistence_weight=float(getattr(cfg, "angle_trend_persistence_weight", 0.4)),
            )
            if pd.isna(angle_component):
                continue

        scores.append((code, factor_score, vol_component, size_component, angle_component))

    if not scores:
        return []

    df = pd.DataFrame(scores, columns=["code", "factor", "vol", "size", "angle"])
    df["factor_rank"] = df["factor"].rank(ascending=False)
    df["vol_rank"] = df["vol"].rank(ascending=False)
    df["size_rank"] = df["size"].rank(ascending=False)
    df["angle_rank"] = df["angle"].rank(ascending=False)

    factor_weight = float(getattr(cfg, "reversal_weight", 1.0)) if getattr(cfg, "use_reversal_factor", False) else 1.0
    vol_weight = float(getattr(cfg, "volatility_weight", 0.0)) if getattr(cfg, "use_volatility_factor", False) else 0.0
    size_weight = float(getattr(cfg, "size_weight", 0.0)) if getattr(cfg, "use_size_factor", False) else 0.0
    angle_weight = float(getattr(cfg, "angle_trend_weight", 0.0)) if getattr(cfg, "use_angle_trend_factor", False) else 0.0
    total_weight = factor_weight + vol_weight + size_weight + angle_weight
    if total_weight <= 0:
        total_weight = 1.0

    df["composite_rank"] = (
        (factor_weight / total_weight) * df["factor_rank"]
        + (vol_weight / total_weight) * df["vol_rank"]
        + (size_weight / total_weight) * df["size_rank"]
        + (angle_weight / total_weight) * df["angle_rank"]
    )
    df = df.sort_values("composite_rank").reset_index(drop=True)
    return list(zip(df["code"], df["composite_rank"]))
