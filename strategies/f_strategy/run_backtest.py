#!/usr/bin/env python3
"""Run the standalone F strategy backtest on the CSI 1000 5y bundle."""

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKTEST_SCRIPT = PROJECT_ROOT / "scripts" / "backtest_strategies.py"
ENV_PATH = PROJECT_ROOT / "config" / ".env"
DEFAULT_TREND_INDEX_CODE = "000001.SH"
DEFAULT_TREND_INDEX_NAME = "上证指数"

DATASETS = {
    "csi1000_5y": PROJECT_ROOT / "data_exports" / "tushare_20210329_20260327_csi1000_5y" / "csi1000_market_bundle_5y.csv",
}


def optional_float(value: str):
    if value is None:
        return None
    if str(value).strip().lower() in {"none", "null", "off"}:
        return None
    return float(value)


def load_backtest_module():
    spec = importlib.util.spec_from_file_location("backtest_strategies", BACKTEST_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_env_token() -> str | None:
    if os.environ.get("TUSHARE_TOKEN"):
        return os.environ["TUSHARE_TOKEN"].strip()
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            if line.startswith("TUSHARE_TOKEN="):
                token = line.split("=", 1)[1].strip()
                if token:
                    return token
    return None


def fetch_trend_index_df(trade_dates: list[str], trend_index_code: str) -> pd.DataFrame | None:
    token = load_env_token()
    if not token:
        return None
    try:
        import tushare as ts
    except Exception:
        return None

    try:
        ts.set_token(token)
        pro = ts.pro_api(token)
        idx = pro.index_daily(
            ts_code=trend_index_code,
            start_date=trade_dates[0],
            end_date=trade_dates[-1],
        )
    except Exception:
        return None

    if idx is None or idx.empty:
        return None

    idx = idx[["trade_date", "close", "pct_chg"]].copy()
    idx.columns = ["trade_date", "idx_close", "idx_pct_chg"]
    idx["trade_date"] = idx["trade_date"].astype(str)
    for col in ["idx_close", "idx_pct_chg"]:
        idx[col] = pd.to_numeric(idx[col], errors="coerce")
    idx = idx.sort_values("trade_date").reset_index(drop=True)
    idx = idx[idx["trade_date"].isin(trade_dates)].copy().reset_index(drop=True)
    return idx


def load_local_trend_index_df(index_path: Path | str, trade_dates: list[str]) -> pd.DataFrame:
    idx = pd.read_csv(index_path)
    idx["trade_date"] = idx["trade_date"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)
    close_col = "close" if "close" in idx.columns else "idx_close"
    pct_col = "pct_chg" if "pct_chg" in idx.columns else "idx_pct_chg"
    idx = idx[["trade_date", close_col, pct_col]].copy()
    idx.columns = ["trade_date", "idx_close", "idx_pct_chg"]
    for col in ["idx_close", "idx_pct_chg"]:
        idx[col] = pd.to_numeric(idx[col], errors="coerce")
    idx = idx[idx["trade_date"].isin(trade_dates)].sort_values("trade_date").reset_index(drop=True)
    return idx


def load_dataset(
    data_path: Path,
    module,
    trend_index_code: str = DEFAULT_TREND_INDEX_CODE,
    trend_index_loader=None,
):
    raw = pd.read_csv(data_path, low_memory=False)

    daily_cols = ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount", "pct_chg"]
    if "circ_mv" in raw.columns:
        daily_cols.append("circ_mv")
    if "ma20_angle_deg" in raw.columns:
        daily_cols.append("ma20_angle_deg")
    if "sw_l1_name" in raw.columns:
        daily_cols.append("sw_l1_name")
    if "sw_l1_ret20" in raw.columns:
        daily_cols.append("sw_l1_ret20")
    if "market_ret20" in raw.columns:
        daily_cols.append("market_ret20")
    if "sw_l1_excess20" in raw.columns:
        daily_cols.append("sw_l1_excess20")
    if "sw_l1_strength20_vs_market" in raw.columns:
        daily_cols.append("sw_l1_strength20_vs_market")

    daily = raw[raw["data_type"] == "daily"][daily_cols].copy()
    for col in [c for c in daily.columns if c not in {"ts_code", "trade_date", "sw_l1_name"}]:
        daily[col] = pd.to_numeric(daily[col], errors="coerce")
    daily["trade_date"] = daily["trade_date"].astype(str)
    daily = daily.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    if "adj_factor" in raw.columns and (raw["data_type"] == "adj_factor").any():
        adj = raw[raw["data_type"] == "adj_factor"][["ts_code", "trade_date", "adj_factor"]].copy()
        adj["adj_factor"] = pd.to_numeric(adj["adj_factor"], errors="coerce")
        adj["trade_date"] = adj["trade_date"].astype(str)
        daily = daily.merge(adj, on=["ts_code", "trade_date"], how="left")
        daily["adj_close"] = daily["close"] * daily["adj_factor"]
    else:
        daily["adj_factor"] = 1.0
        daily["adj_close"] = daily["close"]

    if "daily_basic" in set(raw["data_type"].dropna().astype(str)) and {"total_mv", "circ_mv"}.issubset(raw.columns):
        db = raw[raw["data_type"] == "daily_basic"][["ts_code", "trade_date", "total_mv", "circ_mv"]].copy()
        db["trade_date"] = db["trade_date"].astype(str)
        for col in ["total_mv", "circ_mv"]:
            db[col] = pd.to_numeric(db[col], errors="coerce")
        daily = daily.merge(db, on=["ts_code", "trade_date"], how="left", suffixes=("", "_db"))
        if "circ_mv_db" in daily.columns:
            daily["circ_mv"] = daily["circ_mv_db"].combine_first(daily.get("circ_mv"))
            daily = daily.drop(columns=["circ_mv_db"])
        if "total_mv_db" in daily.columns:
            daily["total_mv"] = daily["total_mv_db"]
            daily = daily.drop(columns=["total_mv_db"])
    else:
        if "float_market_value" in raw.columns:
            fm = raw[raw["data_type"] == "daily"][["ts_code", "trade_date", "float_market_value"]].copy()
            fm["trade_date"] = fm["trade_date"].astype(str)
            fm["float_market_value"] = pd.to_numeric(fm["float_market_value"], errors="coerce")
            daily = daily.merge(fm, on=["ts_code", "trade_date"], how="left")
            daily["circ_mv"] = daily["float_market_value"]
            daily = daily.drop(columns=["float_market_value"])
        daily["total_mv"] = np.nan

    idx = raw[raw["data_type"] == "index_daily"][["trade_date", "close", "pct_chg"]].copy()
    idx.columns = ["trade_date", "idx_close", "idx_pct_chg"]
    for col in ["idx_close", "idx_pct_chg"]:
        idx[col] = pd.to_numeric(idx[col], errors="coerce")
    idx["trade_date"] = idx["trade_date"].astype(str)
    idx = idx.sort_values("trade_date").reset_index(drop=True)

    basic = raw[raw["data_type"] == "stock_basic"][["ts_code", "name", "industry", "list_date"]].copy()
    basic["list_date"] = basic["list_date"].astype(str)

    trade_dates = sorted(daily["trade_date"].unique())
    if trend_index_loader is not None:
        fetched_idx = trend_index_loader(trade_dates, trend_index_code)
    else:
        fetched_idx = fetch_trend_index_df(trade_dates, trend_index_code)
    if fetched_idx is not None and not fetched_idx.empty:
        idx = fetched_idx

    daily = enrich_industry_strength20_features(daily, idx)
    return daily, idx, basic, trade_dates


def calc_strength_transition_coef(a0: float, a1: float) -> float:
    base = np.tanh(a0 / 10.0)
    turn = np.tanh((a0 - a1) / 5.0)
    return float(np.clip(0.7 * base + 0.3 * turn, -1.0, 1.0))


def enrich_industry_strength20_features(daily: pd.DataFrame, idx: pd.DataFrame) -> pd.DataFrame:
    if "sw_l1_name" not in daily.columns:
        return daily

    feature_cols = ["sw_l1_ret20", "market_ret20", "sw_l1_excess20", "sw_l1_strength20_vs_market"]
    if all(col in daily.columns and daily[col].notna().any() for col in feature_cols):
        return daily
    existing = [col for col in feature_cols if col in daily.columns]
    if existing:
        daily = daily.drop(columns=existing)

    ind = (
        daily.dropna(subset=["sw_l1_name", "close"])
        .groupby(["trade_date", "sw_l1_name"], as_index=False)["close"]
        .mean()
        .sort_values(["sw_l1_name", "trade_date"])
    )
    ind["sw_l1_ret20"] = ind.groupby("sw_l1_name")["close"].transform(lambda s: s / s.shift(20) - 1)

    idx_ret = idx[["trade_date", "idx_close"]].copy().sort_values("trade_date").reset_index(drop=True)
    idx_ret["idx_ret20"] = idx_ret["idx_close"] / idx_ret["idx_close"].shift(20) - 1

    features = ind.merge(idx_ret[["trade_date", "idx_ret20"]], on="trade_date", how="left")
    features["market_ret20"] = features["idx_ret20"]
    features["sw_l1_excess20"] = features["sw_l1_ret20"] - features["market_ret20"]
    features["sw_l1_strength20_vs_market"] = features["sw_l1_ret20"] / features["market_ret20"]
    features.loc[features["market_ret20"].abs() < 1e-12, "sw_l1_strength20_vs_market"] = np.nan

    features = features[
        ["trade_date", "sw_l1_name", "sw_l1_ret20", "market_ret20", "sw_l1_excess20", "sw_l1_strength20_vs_market"]
    ]
    return daily.merge(features, on=["trade_date", "sw_l1_name"], how="left")


def make_filtered_backtest(
    module,
    min_ma20_angle=None,
    min_transition_coef=None,
    min_industry_strength20_vs_market=None,
):
    if (
        min_ma20_angle is None
        and min_transition_coef is None
        and min_industry_strength20_vs_market is None
    ):
        return module.Backtest

    class FilteredBacktest(module.Backtest):
        def __init__(self, cfg, daily, idx, basic, trade_dates):
            super().__init__(cfg, daily, idx, basic, trade_dates)

        def _score_universe(self, date: str):
            cfg = self.cfg
            scores = []
            for code, data in self._stock_data.items():
                if date not in data.index:
                    continue
                row = data.loc[date]

                close = row["close"]
                if pd.isna(close) or close < cfg.min_price:
                    continue

                info = self._basic_map.get(code)
                if info is not None:
                    name = str(info.get("name", ""))
                    if "ST" in name.upper():
                        continue
                    list_date = str(info.get("list_date", ""))
                    if list_date and len(list_date) >= 8:
                        try:
                            from datetime import datetime
                            ld = datetime.strptime(list_date[:8], "%Y%m%d")
                            cd = datetime.strptime(date, "%Y%m%d")
                            if (cd - ld).days < cfg.min_list_days:
                                continue
                        except Exception:
                            pass

                hist = data[data.index <= date]
                if len(hist) < max(cfg.momentum_days[0], 60) + 5:
                    continue

                avg_amount = hist.tail(20)["amount"].mean() * 1000
                if avg_amount < cfg.min_amount_20d:
                    continue

                if row["pct_chg"] >= 9.5:
                    continue

                if min_ma20_angle is not None:
                    angle = row.get("ma20_angle_deg", np.nan)
                    if pd.isna(angle) or angle < min_ma20_angle:
                        continue

                if min_transition_coef is not None:
                    if "ma20_angle_deg" not in hist.columns:
                        continue
                    angles = hist["ma20_angle_deg"].tail(2).values.astype(float)
                    if len(angles) < 2 or np.isnan(angles[-1]) or np.isnan(angles[-2]):
                        continue
                    coef = calc_strength_transition_coef(float(angles[-1]), float(angles[-2]))
                    if coef < min_transition_coef:
                        continue

                if min_industry_strength20_vs_market is not None:
                    ratio = row.get("sw_l1_strength20_vs_market", np.nan)
                    if pd.isna(ratio) or float(ratio) < min_industry_strength20_vs_market:
                        continue

                closes = hist["close"].values.astype(float)
                mom_score = 0.0
                for lb, w in zip(cfg.momentum_days, cfg.momentum_weights):
                    if len(closes) >= lb + 1:
                        ret = closes[-1] / closes[-(lb + 1)] - 1
                        mom_score += ret * w
                    else:
                        mom_score = np.nan
                        break
                if np.isnan(mom_score):
                    continue

                if cfg.subtract_short_momentum and len(closes) >= cfg.short_momentum_days + 1:
                    short_ret = closes[-1] / closes[-(cfg.short_momentum_days + 1)] - 1
                    mom_score -= short_ret

                vol_component = 0.0
                if cfg.use_volatility_factor and len(closes) >= cfg.volatility_days + 1:
                    rets = np.diff(closes[-cfg.volatility_days - 1:]) / closes[-cfg.volatility_days - 1:-1]
                    vol = np.std(rets)
                    if vol > 0:
                        vol_component = -vol

                size_component = 0.0
                if cfg.use_size_factor:
                    circ_mv = row.get("circ_mv", None)
                    if circ_mv and not pd.isna(circ_mv) and circ_mv > 0:
                        size_component = -np.log(circ_mv)

                scores.append((code, mom_score, vol_component, size_component))

            if not scores:
                return []

            df = pd.DataFrame(scores, columns=["code", "mom", "vol", "size"])
            df["mom_rank"] = df["mom"].rank(ascending=False)
            df["vol_rank"] = df["vol"].rank(ascending=False)
            df["size_rank"] = df["size"].rank(ascending=False)

            mom_weight = 1.0
            vol_w = cfg.volatility_weight if cfg.use_volatility_factor else 0
            size_w = cfg.size_weight if cfg.use_size_factor else 0
            total_w = mom_weight + vol_w + size_w

            df["composite_rank"] = (
                (mom_weight / total_w) * df["mom_rank"] +
                (vol_w / total_w) * df["vol_rank"] +
                (size_w / total_w) * df["size_rank"]
            )
            df = df.sort_values("composite_rank").reset_index(drop=True)
            return list(zip(df["code"], df["composite_rank"]))

    return FilteredBacktest


def make_angle_filtered_backtest(module, min_ma20_angle):
    return make_filtered_backtest(module, min_ma20_angle=min_ma20_angle, min_transition_coef=None)


def make_regime_industry_excess_rank_boost_backtest(
    module,
    bonus_weight: float,
    active_regimes: tuple[str, ...] = ("RANGE", "BEAR"),
    min_ma20_angle=None,
    min_transition_coef=None,
    min_industry_strength20_vs_market=None,
):
    Base = make_filtered_backtest(
        module,
        min_ma20_angle=min_ma20_angle,
        min_transition_coef=min_transition_coef,
        min_industry_strength20_vs_market=min_industry_strength20_vs_market,
    )
    normalized_regimes = tuple(reg.upper() for reg in active_regimes if str(reg).strip())

    class RegimeIndustryExcessRankBoostBacktest(Base):
        def _score_universe(self, date: str):
            base_scores = super()._score_universe(date)
            if not base_scores or bonus_weight is None or bonus_weight <= 0:
                return base_scores

            trend_state = self._get_trend_state(date) if self.cfg.use_trend_filter else "BULL"
            if trend_state not in normalized_regimes:
                return base_scores

            rows = []
            for base_rank, (code, _) in enumerate(base_scores, start=1):
                data = self._stock_data.get(code)
                if data is None or date not in data.index:
                    continue
                row = data.loc[date]
                excess = row.get("sw_l1_excess20", np.nan)
                if pd.isna(excess):
                    ind_ret20 = row.get("sw_l1_ret20", np.nan)
                    market_ret20 = row.get("market_ret20", np.nan)
                    if pd.notna(ind_ret20) and pd.notna(market_ret20):
                        excess = float(ind_ret20) - float(market_ret20)
                rows.append(
                    {
                        "code": code,
                        "base_rank": float(base_rank),
                        "industry_excess20": float(excess) if pd.notna(excess) else np.nan,
                    }
                )

            if not rows:
                return base_scores

            df = pd.DataFrame(rows)
            if df["industry_excess20"].notna().sum() == 0:
                return base_scores

            df["industry_excess_rank"] = df["industry_excess20"].rank(
                ascending=False,
                method="average",
                na_option="bottom",
            )
            df["boosted_rank"] = df["base_rank"] + bonus_weight * df["industry_excess_rank"]
            df = df.sort_values(["boosted_rank", "base_rank"]).reset_index(drop=True)
            return list(zip(df["code"], df["boosted_rank"]))

    return RegimeIndustryExcessRankBoostBacktest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="csi1000_5y")
    parser.add_argument("--min-ma20-angle", type=optional_float, default=None)
    parser.add_argument("--min-transition-coef", type=optional_float, default=-0.1)
    parser.add_argument("--min-industry-strength20-vs-market", type=optional_float, default=None)
    parser.add_argument("--industry-excess-rank-bonus-weight", type=optional_float, default=None)
    parser.add_argument("--industry-excess-active-regimes", type=str, default="RANGE,BEAR")
    parser.add_argument("--trend-index-code", type=str, default=DEFAULT_TREND_INDEX_CODE)
    args = parser.parse_args()

    module = load_backtest_module()
    data_path = DATASETS[args.dataset]
    daily, idx, basic, trade_dates = load_dataset(data_path, module, trend_index_code=args.trend_index_code)

    cfg = [s for s in module.get_strategies() if s.name == "F_三因子+趋势过滤"][0]
    active_regimes = tuple(part.strip().upper() for part in args.industry_excess_active_regimes.split(",") if part.strip())
    if args.industry_excess_rank_bonus_weight is not None and args.industry_excess_rank_bonus_weight > 0:
        backtest_cls = make_regime_industry_excess_rank_boost_backtest(
            module,
            bonus_weight=args.industry_excess_rank_bonus_weight,
            active_regimes=active_regimes,
            min_ma20_angle=args.min_ma20_angle,
            min_transition_coef=args.min_transition_coef,
            min_industry_strength20_vs_market=args.min_industry_strength20_vs_market,
        )
    else:
        backtest_cls = make_filtered_backtest(
            module,
            min_ma20_angle=args.min_ma20_angle,
            min_transition_coef=args.min_transition_coef,
            min_industry_strength20_vs_market=args.min_industry_strength20_vs_market,
        )

    t0 = time.time()
    bt = backtest_cls(cfg, daily, idx, basic, trade_dates)
    result = bt.run(start_offset=250)
    result["elapsed_sec"] = round(time.time() - t0, 1)
    result["trend_filter_index_code"] = args.trend_index_code
    if args.industry_excess_rank_bonus_weight is not None and args.industry_excess_rank_bonus_weight > 0:
        result["industry_excess_rank_bonus_weight"] = args.industry_excess_rank_bonus_weight
        result["industry_excess_active_regimes"] = list(active_regimes)

    suffix = f"_{args.dataset}"
    if args.min_transition_coef is not None:
        suffix += f"_transition_coef_ge_{str(args.min_transition_coef).replace('.', '_').replace('-', 'neg_')}"
    if args.min_ma20_angle is not None:
        suffix += f"_ma20_angle_ge_{str(args.min_ma20_angle).replace('.', '_')}"
    if args.min_industry_strength20_vs_market is not None:
        suffix += (
            "_industry_strength20_vs_market_ge_"
            f"{str(args.min_industry_strength20_vs_market).replace('.', '_').replace('-', 'neg_')}"
        )
    if args.industry_excess_rank_bonus_weight is not None and args.industry_excess_rank_bonus_weight > 0:
        regimes_suffix = "_".join(active_regimes).lower() if active_regimes else "none"
        suffix += (
            "_industry_excess20_rank_bonus_"
            f"{str(args.industry_excess_rank_bonus_weight).replace('.', '_').replace('-', 'neg_')}"
            f"_{regimes_suffix}"
        )
    out_path = PROJECT_ROOT / "backtest" / f"strategy_f{suffix}.json"
    out_path.write_text(
        json.dumps({"data_file": str(data_path), "result": result}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"WROTE {out_path}")


if __name__ == "__main__":
    main()
