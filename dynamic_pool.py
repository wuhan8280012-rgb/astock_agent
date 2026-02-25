"""
dynamic_pool.py — 动态股票池构建器
===================================
每日盘后根据市场状态动态生成当天的股票池。

过滤条件（按顺序收窄）:
  1. 非ST、非退市、非停牌
  2. 上市 >= 60天
  3. 流通市值 >= 50亿
  4. 收盘 > MA10
  5. MA10角度 > MA30角度（5日）
  6. MA10 > MA30
  7. 近20日涨幅 > 0%
  8. 近5日平均换手率 > 1%
  9. 当日量比 > 0.8
"""

import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


POOL_CONFIG = {
    # 基本面
    "min_circ_mv": int(os.getenv("POOL_MIN_MV", "500000")),  # 万元，50亿
    "exclude_st": True,
    "min_list_days": int(os.getenv("POOL_MIN_LIST_DAYS", "60")),
    # 均线
    "ma_short": 10,
    "ma_long": 30,
    "ma_angle_lookback": 5,
    # 趋势
    "min_pct_20d": 0.0,
    # 流动性
    "min_turnover_5d": 1.0,
    "min_vol_ratio": 0.8,
    # 数据
    "need_days": 60,
    "batch_size": 80,
    "api_sleep": 0.15,
}


class DynamicPool:
    """动态股票池构建器"""

    def __init__(self, data_fetcher=None, data_cache=None):
        self.fetcher = data_fetcher
        self.cache = data_cache
        self.pro = None
        if data_fetcher is not None and hasattr(data_fetcher, "pro"):
            self.pro = data_fetcher.pro
        elif data_cache is not None and hasattr(data_cache, "pro"):
            self.pro = data_cache.pro

    def build(self, date: str, verbose: bool = True) -> dict:
        t0 = time.time()
        funnel = {}

        if verbose:
            print(f"\n  🏊 构建动态股票池 ({date})")
            print("     条件: 市值≥50亿 | MA10角度>MA30角度 | 收盘>MA10 | 多头排列")
            print("           20日涨幅>0% | 5日换手>1% | 量比>0.8 | 非ST | 上市≥60天")

        basic = self._get_stock_basic()
        if basic is None or basic.empty:
            print("  ❌ 无法获取股票基本信息")
            return self._empty_result()
        funnel["total"] = len(basic)

        # 非ST
        if POOL_CONFIG["exclude_st"] and "name" in basic.columns:
            basic = basic[~basic["name"].str.contains("ST", case=False, na=False)]
        funnel["after_st"] = len(basic)

        # 上市 >= 60天
        if "list_date" in basic.columns:
            min_date = (
                datetime.strptime(date, "%Y%m%d") - timedelta(days=POOL_CONFIG["min_list_days"])
            ).strftime("%Y%m%d")
            basic = basic[basic["list_date"] <= min_date]
        funnel["after_new"] = len(basic)

        # 市值过滤
        mv_qualified = self._filter_by_market_value(basic["ts_code"].tolist(), date)
        basic = basic[basic["ts_code"].isin(mv_qualified)]
        funnel["after_mv"] = len(basic)

        if verbose:
            print(
                f"     基础过滤: 全市场{funnel['total']} → 去ST{funnel['after_st']}"
                f" → 去次新{funnel['after_new']} → 市值≥50亿{funnel['after_mv']}"
            )

        candidates = basic["ts_code"].tolist()
        passed, tech_stats = self._filter_technical(candidates, date, verbose)
        funnel.update(tech_stats)
        funnel["final"] = len(passed)

        by_sector = {}
        if "industry" in basic.columns:
            sector_map = dict(zip(basic["ts_code"], basic["industry"]))
            for code in passed:
                by_sector.setdefault(sector_map.get(code, "未知"), []).append(code)

        pool_size = len(passed)
        breadth = self._assess_breadth(pool_size)
        dt = time.time() - t0

        if verbose:
            print(f"  ✅ 动态池构建完成 ({dt:.1f}s)")
            print(f"     📊 池子: {pool_size} 只 | 市场宽度: {breadth}")
            print(
                f"     📊 漏斗: {funnel['total']}→{funnel['after_mv']}"
                f"→MA过滤{funnel.get('after_ma', '?')}"
                f"→趋势{funnel.get('after_trend', '?')}"
                f"→流动性{funnel.get('after_liquidity', '?')}"
                f"→最终{pool_size}"
            )
            if by_sector:
                top5 = sorted(by_sector.items(), key=lambda x: -len(x[1]))[:5]
                sectors_str = " | ".join(f"{sec}:{len(codes)}只" for sec, codes in top5)
                print(f"     📊 板块分布 Top5: {sectors_str}")

        return {
            "pool": passed,
            "pool_size": pool_size,
            "market_breadth": breadth,
            "by_sector": by_sector,
            "filter_funnel": funnel,
            "build_time": round(dt, 1),
        }

    def build_with_sectors(self, date: str) -> dict:
        """兼容旧调用，返回与 build 一致结构。"""
        return self.build(date, verbose=True)

    def _get_stock_basic(self) -> pd.DataFrame:
        if self.cache is not None and hasattr(self.cache, "get_stock_basic"):
            try:
                df = self.cache.get_stock_basic()
                if df is not None and not df.empty:
                    return df
            except Exception:
                pass

        if self.pro is not None:
            try:
                return self.pro.stock_basic(
                    list_status="L",
                    fields="ts_code,name,industry,market,list_date",
                )
            except Exception as e:
                print(f"  ⚠️ 获取股票基本信息失败: {e}")
        return pd.DataFrame()

    def _filter_by_market_value(self, codes: list, date: str) -> list:
        min_mv = POOL_CONFIG["min_circ_mv"]

        if self.cache is not None and hasattr(self.cache, "get_fundamental_batch"):
            try:
                df = self.cache.get_fundamental_batch(date)
                if df is not None and not df.empty and "circ_mv" in df.columns:
                    return df[df["circ_mv"] >= min_mv]["ts_code"].tolist()
            except Exception:
                pass

        if self.pro is not None:
            try:
                df = self.pro.daily_basic(trade_date=date, fields="ts_code,circ_mv,turnover_rate")
                if df is not None and not df.empty:
                    return df[df["circ_mv"] >= min_mv]["ts_code"].tolist()
            except Exception as e:
                print(f"  ⚠️ 市值数据获取失败: {e}")

        return codes

    def _filter_technical(self, candidates: list, date: str, verbose: bool) -> tuple:
        passed = []
        stats = {
            "tech_checked": 0,
            "after_ma": 0,
            "after_trend": 0,
            "after_liquidity": 0,
        }

        turnover_map = {}
        vol_ratio_map = {}
        if self.pro is not None:
            try:
                db = self.pro.daily_basic(trade_date=date, fields="ts_code,turnover_rate,vol_ratio")
                if db is not None and not db.empty:
                    if "turnover_rate" in db.columns:
                        turnover_map = dict(zip(db["ts_code"], db["turnover_rate"].fillna(0)))
                    if "vol_ratio" in db.columns:
                        vol_ratio_map = dict(zip(db["ts_code"], db["vol_ratio"].fillna(0)))
            except Exception:
                pass

        daily_data = self._batch_load_daily(candidates, date)
        total = len(candidates)
        t0 = time.time()
        lookback = POOL_CONFIG["ma_angle_lookback"]
        need_days = POOL_CONFIG["need_days"]

        for idx, code in enumerate(candidates, 1):
            if verbose and idx % 200 == 0:
                elapsed = time.time() - t0
                print(f"     ...技术面过滤 {idx}/{total} 通过{len(passed)} ({elapsed:.1f}s)")

            df = daily_data.get(code)
            if df is None or len(df) < need_days:
                continue

            try:
                close = pd.to_numeric(df["close"], errors="coerce").dropna().values
                if len(close) < need_days:
                    continue
                stats["tech_checked"] += 1
                current = float(close[-1])

                ma10_now = float(np.mean(close[-10:]))
                if current <= ma10_now:
                    continue

                ma10_prev = float(np.mean(close[-(10 + lookback):-lookback]))
                ma30_now = float(np.mean(close[-30:]))
                ma30_prev = float(np.mean(close[-(30 + lookback):-lookback]))
                ma10_angle = ((ma10_now - ma10_prev) / ma10_prev * 100) if ma10_prev > 0 else 0
                ma30_angle = ((ma30_now - ma30_prev) / ma30_prev * 100) if ma30_prev > 0 else 0
                if ma10_angle <= ma30_angle:
                    continue
                if ma10_now <= ma30_now:
                    continue
                stats["after_ma"] += 1

                if len(close) < 20:
                    continue
                pct_20d = (current / float(close[-20]) - 1) * 100
                if pct_20d <= POOL_CONFIG["min_pct_20d"]:
                    continue
                stats["after_trend"] += 1

                turnover = float(turnover_map.get(code, 0) or 0)
                if "turnover_rate" in df.columns:
                    tr_5d = pd.to_numeric(df["turnover_rate"].tail(5), errors="coerce").mean()
                    if not np.isnan(tr_5d):
                        turnover = float(tr_5d)
                if turnover < POOL_CONFIG["min_turnover_5d"]:
                    continue

                vol_ratio = float(vol_ratio_map.get(code, 0) or 0)
                if vol_ratio <= 0 and "vol" in df.columns and len(df) >= 21:
                    today_vol = float(df["vol"].iloc[-1] or 0)
                    avg_vol = float(df["vol"].iloc[-21:-1].mean() or 0)
                    vol_ratio = today_vol / avg_vol if avg_vol > 0 else 0
                if vol_ratio < POOL_CONFIG["min_vol_ratio"]:
                    continue

                stats["after_liquidity"] += 1
                passed.append(code)
            except Exception:
                continue

        return passed, stats

    def _batch_load_daily(self, codes: list, date: str) -> dict:
        result = {}
        need_days = POOL_CONFIG["need_days"]

        if self.cache is not None and hasattr(self.cache, "get_daily"):
            for code in codes:
                try:
                    df = self.cache.get_daily(code, end_date=date, days=need_days)
                    if df is not None and not df.empty:
                        result[code] = df
                except Exception:
                    continue
            if len(result) >= int(len(codes) * 0.8):
                return result

        if self.pro is None:
            return result

        end_dt = datetime.strptime(date, "%Y%m%d")
        start_dt = end_dt - timedelta(days=int(need_days * 1.5))
        start_str = start_dt.strftime("%Y%m%d")

        uncached = [c for c in codes if c not in result]
        batch_size = POOL_CONFIG["batch_size"]
        print(f"     📦 批量加载 {len(uncached)} 只日线 ({start_str}~{date}) ...")
        t0 = time.time()

        for i in range(0, len(uncached), batch_size):
            batch = uncached[i:i + batch_size]
            codes_str = ",".join(batch)
            try:
                df = self.pro.daily(ts_code=codes_str, start_date=start_str, end_date=date)
                if df is None or df.empty:
                    continue

                try:
                    adj = self.pro.adj_factor(ts_code=codes_str, start_date=start_str, end_date=date)
                    if adj is not None and not adj.empty:
                        df = df.merge(
                            adj[["ts_code", "trade_date", "adj_factor"]],
                            on=["ts_code", "trade_date"],
                            how="left",
                        )
                        latest = df.groupby("ts_code")["adj_factor"].transform("last")
                        for col in ("open", "high", "low", "close"):
                            if col in df.columns:
                                df[col] = df[col] * df["adj_factor"] / latest
                        df.drop(columns=["adj_factor"], inplace=True, errors="ignore")
                except Exception:
                    pass

                for code, group in df.groupby("ts_code"):
                    result[code] = group.sort_values("trade_date").reset_index(drop=True)
            except Exception as e:
                if i == 0:
                    print(f"     ⚠️ 批量加载失败: {e}")
                continue

            time.sleep(POOL_CONFIG["api_sleep"])

        dt = time.time() - t0
        print(f"     ✅ 加载完成: {len(result)}只 ({dt:.1f}s)")
        return result

    def _assess_breadth(self, pool_size: int) -> str:
        if pool_size > 600:
            return "强势 🟢"
        if pool_size > 400:
            return "偏强 🟢"
        if pool_size > 250:
            return "中性 🟡"
        if pool_size > 100:
            return "偏弱 🟡"
        return "弱势 🔴"

    def _empty_result(self) -> dict:
        return {
            "pool": [],
            "pool_size": 0,
            "market_breadth": "无数据",
            "by_sector": {},
            "filter_funnel": {},
            "build_time": 0.0,
        }


if __name__ == "__main__":
    import sys

    from data_fetcher import get_fetcher

    run_date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y%m%d")
    fetcher = get_fetcher()
    pool = DynamicPool(data_fetcher=fetcher)
    out = pool.build(run_date, verbose=True)
    print("\n" + "=" * 60)
    print(f"  股票池: {out['pool_size']} 只")
    print(f"  市场宽度: {out['market_breadth']}")
    print(f"  构建耗时: {out['build_time']}s")
    if out["pool"]:
        print(f"  前20只: {', '.join(out['pool'][:20])}")
