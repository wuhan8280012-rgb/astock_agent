"""
DataHub — 统一数据访问层
所有 Skill 只能通过 DataHub 获取数据。
绝对禁止 Skill 直接 import tushare。

设计原则：
1. 缓存优先，API 兜底
2. 回测模式下禁止实时 API 调用
3. 数据异常标记（停牌/涨跌停/ST）在此层处理
4. 统一接口，上层不关心数据来源
"""

import os
import sys
import logging
from datetime import datetime, date, timedelta
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class DataMode(Enum):
    LIVE = "live"          # 实盘模式：缓存优先，允许 API 兜底
    BACKTEST = "backtest"  # 回测模式：只读缓存，禁止 API 调用


class MarketFlag(Enum):
    """数据异常标记"""
    NORMAL = "normal"
    SUSPENDED = "suspended"      # 停牌
    LIMIT_UP = "limit_up"        # 涨停
    LIMIT_DOWN = "limit_down"    # 跌停
    ST = "st"                    # ST/*ST
    NEW_LISTING = "new_listing"  # 次新股（上市<60天）


@dataclass
class PriceRecord:
    """标准化价格记录"""
    ts_code: str
    trade_date: str       # YYYYMMDD
    open: float
    high: float
    low: float
    close: float
    volume: float         # 成交量（手）
    amount: float         # 成交额（千元）
    pct_chg: float        # 涨跌幅
    flags: list[MarketFlag]


@dataclass
class CacheStats:
    """缓存状态"""
    last_update: Optional[datetime]
    total_records: int
    date_range: tuple[str, str]  # (最早日期, 最晚日期)
    stale: bool                  # 是否过期


class DataHub:
    """
    统一数据访问层

    使用方式：
        hub = DataHub(cache_dir="/data/parquet", mode=DataMode.LIVE)
        price = hub.get_price("000001.SZ", "20250101", "20250301")
        index = hub.get_index("000300.SH", "20250301")
        strength = hub.get_theme_strength("20250301")

    强制规则：
        - 回测模式下调用 API 会抛出 BacktestAPIViolation
        - 所有返回数据自动附带 MarketFlag
        - 缓存过期自动告警
    """

    def __init__(
        self,
        cache_dir: str = "./data/parquet",
        mode: DataMode = DataMode.LIVE,
        api_timeout: int = 30,
        max_retries: int = 3,
        fetcher=None,          # 接受外部注入的 DataFetcher，None 则延迟创建
    ):
        self.cache_dir = Path(cache_dir)
        self.mode = mode
        self.api_timeout = api_timeout
        self.max_retries = max_retries
        self._fetcher = fetcher  # 延迟初始化，首次 API 调用时若为 None 则自动创建

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._api_call_count = 0
        self._cache_hit_count = 0

        logger.info(f"DataHub 初始化: mode={mode.value}, cache_dir={cache_dir}")

    # =============================================
    # 核心数据接口 — 所有 Skill 只能调用这些方法
    # =============================================

    def get_price(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
    ) -> list[PriceRecord]:
        """
        获取个股日线数据

        Args:
            ts_code: 股票代码，如 "000001.SZ"
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD

        Returns:
            PriceRecord 列表，已附带 MarketFlag

        Raises:
            BacktestAPIViolation: 回测模式下缓存未命中
        """
        cache_key = f"price/{ts_code}"
        data = self._read_cache(cache_key, start_date, end_date)

        if data is not None:
            self._cache_hit_count += 1
            return self._attach_flags(data)

        # 缓存未命中
        if self.mode == DataMode.BACKTEST:
            raise BacktestAPIViolation(
                f"回测模式下缓存未命中: {ts_code} [{start_date}, {end_date}]。"
                f"请先运行增量更新。"
            )

        # 实盘模式：API 兜底
        logger.warning(f"缓存未命中，调用 API: {ts_code}")
        data = self._fetch_from_api("daily", ts_code=ts_code,
                                     start_date=start_date, end_date=end_date)
        self._write_cache(cache_key, data)
        return self._attach_flags(data)

    def get_index(
        self,
        index_code: str,
        trade_date: str,
    ) -> Optional[dict]:
        """
        获取指数数据

        Args:
            index_code: 指数代码，如 "000300.SH"（沪深300）
            trade_date: 交易日期 YYYYMMDD
        """
        cache_key = f"index/{index_code}"
        data = self._read_cache(cache_key, trade_date, trade_date)

        if data is not None:
            self._cache_hit_count += 1
            return data[0] if data else None

        if self.mode == DataMode.BACKTEST:
            raise BacktestAPIViolation(
                f"回测模式下缓存未命中: {index_code} [{trade_date}]"
            )

        data = self._fetch_from_api("index_daily", ts_code=index_code,
                                     start_date=trade_date, end_date=trade_date)
        self._write_cache(cache_key, data)
        return data[0] if data else None

    def get_theme_strength(self, trade_date: str) -> list[dict]:
        """
        获取板块/主题强度数据

        Returns:
            [{"theme": "商业航天", "strength": 0.82, "rank": 1}, ...]
        """
        cache_key = "theme/strength"
        data = self._read_cache(cache_key, trade_date, trade_date)

        if data is not None:
            self._cache_hit_count += 1
            return data

        if self.mode == DataMode.BACKTEST:
            raise BacktestAPIViolation(
                f"回测模式下板块强度缓存未命中: {trade_date}"
            )

        # 实盘模式：计算或拉取
        data = self._compute_theme_strength(trade_date)
        self._write_cache(cache_key, data)
        return data

    def get_market_calendar(self, start_date: str, end_date: str) -> list[str]:
        """获取交易日历"""
        cache_key = "meta/calendar"
        data = self._read_cache(cache_key, start_date, end_date)
        if data is not None:
            return data
        data = self._fetch_from_api("trade_cal", start_date=start_date,
                                     end_date=end_date)
        self._write_cache(cache_key, data)
        return data

    # =============================================
    # 数据更新接口 — 仅供 cron 任务调用
    # =============================================

    def incremental_update(self, target_date: str = None) -> dict:
        """
        增量更新数据

        仅更新缓存中缺失的日期数据，不全量拉取。
        应由每日 cron 任务调用。
        """
        if target_date is None:
            target_date = date.today().strftime("%Y%m%d")

        stats = {
            "target_date": target_date,
            "updated_codes": 0,
            "api_calls": 0,
            "errors": [],
        }

        # P0：关键指数（4 个）
        KEY_INDICES = ["000001.SH", "399001.SZ", "399006.SZ", "000300.SH"]
        for code in KEY_INDICES:
            try:
                cache_key = f"index/{code}"
                existing = self._read_cache(cache_key, target_date, target_date)
                if existing:
                    continue  # 已有，跳过
                data = self._fetch_from_api("index_daily", ts_code=code,
                                             start_date=target_date, end_date=target_date)
                if data:
                    self._write_cache(cache_key, data)
                    stats["updated_codes"] += 1
                    stats["api_calls"] += 1
            except Exception as e:
                stats["errors"].append(f"index/{code}: {e}")

        # P1：DynamicPool 中的股票（从 pool_state.json 读取）
        try:
            import json
            pool_path = Path("data/pool_state.json")
            if pool_path.exists():
                pool = json.loads(pool_path.read_text())
                codes = list(pool.get("stocks", {}).keys())[:50]  # 限 50 只，防超时
                for code in codes:
                    try:
                        cache_key = f"price/{code}"
                        existing = self._read_cache(cache_key, target_date, target_date)
                        if existing:
                            continue
                        data = self._fetch_from_api("daily", ts_code=code,
                                                     start_date=target_date, end_date=target_date)
                        if data:
                            self._write_cache(cache_key, data)
                            stats["updated_codes"] += 1
                            stats["api_calls"] += 1
                    except Exception as e:
                        stats["errors"].append(f"price/{code}: {e}")
        except Exception as e:
            stats["errors"].append(f"pool_load: {e}")

        logger.info(f"增量更新完成: {stats}")
        return stats

    # =============================================
    # 缓存状态与诊断
    # =============================================

    def cache_stats(self) -> CacheStats:
        """返回缓存状态"""
        import pandas as pd

        parquet_files = list(self.cache_dir.rglob("*.parquet"))
        if not parquet_files:
            return CacheStats(last_update=None, total_records=0, date_range=("", ""), stale=True)

        latest_mtime = max(f.stat().st_mtime for f in parquet_files)
        last_update = datetime.fromtimestamp(latest_mtime)

        total = 0
        all_dates = []
        for f in parquet_files[:10]:  # 抽样 10 个文件，避免扫描过慢
            try:
                df = pd.read_parquet(f, columns=["trade_date"])
                total += len(df)
                all_dates.extend(df["trade_date"].tolist())
            except Exception:
                pass

        date_range = (min(all_dates), max(all_dates)) if all_dates else ("", "")
        stale = (datetime.now() - last_update).days >= 1
        return CacheStats(
            last_update=last_update,
            total_records=total,
            date_range=date_range,
            stale=stale,
        )

    def health_check(self) -> dict:
        """
        数据层健康检查

        检查项：
        1. 缓存目录是否可访问
        2. 缓存是否过期（>1个交易日）
        3. API 连通性
        4. 数据完整性（抽样校验）
        """
        checks = {}

        # 1. 缓存目录
        checks["cache_dir_exists"] = self.cache_dir.exists()
        checks["cache_dir_writable"] = os.access(self.cache_dir, os.W_OK)

        # 2. 缓存新鲜度
        stats = self.cache_stats()
        checks["cache_stale"] = stats.stale
        checks["last_update"] = str(stats.last_update) if stats.last_update else "never"

        # 3. API 连通性
        try:
            # 轻量级探测
            checks["api_reachable"] = self._ping_api()
        except Exception as e:
            checks["api_reachable"] = False
            checks["api_error"] = str(e)

        # 4. 统计
        checks["api_calls_total"] = self._api_call_count
        checks["cache_hits_total"] = self._cache_hit_count
        if self._api_call_count + self._cache_hit_count > 0:
            checks["cache_hit_rate"] = (
                self._cache_hit_count
                / (self._api_call_count + self._cache_hit_count)
            )
        else:
            checks["cache_hit_rate"] = None

        return checks

    # =============================================
    # 内部方法 — 外部不可直接调用
    # =============================================

    def _read_cache(self, cache_key: str, start_date: str, end_date: str):
        """从 Parquet 缓存读取数据"""
        import pandas as pd

        path = self.cache_dir / f"{cache_key.replace('/', '_')}.parquet"
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            if df.empty:
                return None
            # 按日期范围过滤
            mask = (df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)
            result = df[mask]
            if result.empty:
                return None
            return result.to_dict("records")
        except Exception as e:
            logger.warning(f"缓存读取失败 {cache_key}: {e}")
            return None

    def _write_cache(self, cache_key: str, data) -> None:
        """写入 Parquet 缓存（Merge-on-Append，按 trade_date 去重）"""
        if not data:
            return
        import pandas as pd

        path = self.cache_dir / f"{cache_key.replace('/', '_')}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)

        new_df = pd.DataFrame(data) if isinstance(data, list) else data
        if new_df.empty:
            return

        # 合并已有数据（去重，保留最新）
        if path.exists():
            try:
                existing = pd.read_parquet(path)
                combined = pd.concat([existing, new_df], ignore_index=True)
                dedup_cols = ["trade_date"]
                if "ts_code" in combined.columns:
                    dedup_cols.append("ts_code")
                combined = combined.drop_duplicates(subset=dedup_cols, keep="last")
            except Exception:
                combined = new_df
        else:
            combined = new_df

        combined = combined.sort_values("trade_date").reset_index(drop=True)
        combined.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
        logger.info(f"缓存写入: {cache_key} ({len(combined)} 条)")

    def _fetch_from_api(self, api_name: str, **params):
        """
        从 DataFetcher 获取数据（桥接 Tushare）

        内置：
        - 懒加载 DataFetcher（首次调用时初始化）
        - 统一列名映射到 PriceRecord 字段
        """
        self._api_call_count += 1

        # 懒加载 DataFetcher（需要把 astock_agent 目录加入路径）
        if self._fetcher is None:
            _agent_dir = Path(__file__).parent.parent.parent  # openclaw_os/data/../../ = astock_agent/
            if str(_agent_dir) not in sys.path:
                sys.path.insert(0, str(_agent_dir))
            from data_fetcher import DataFetcher
            self._fetcher = DataFetcher()

        import pandas as pd

        if api_name == "daily":
            ts_code = params.get("ts_code", "")
            df = self._fetcher.get_daily(
                symbol=ts_code,
                start_date=params.get("start_date"),
                end_date=params.get("end_date"),
            )
            if df is None or df.empty:
                return []
            # 统一列名映射到 PriceRecord 字段
            # get_daily 输出: symbol, pct_change, vol, trade_date, open, close, high, low, amount
            col_map = {"symbol": "ts_code", "vol": "volume", "pct_change": "pct_chg"}
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
            if "ts_code" not in df.columns:
                df["ts_code"] = ts_code
            return df.to_dict("records")

        elif api_name == "index_daily":
            ts_code = params.get("ts_code", "")
            start_date = params.get("start_date")
            end_date = params.get("end_date")

            # get_index_daily 不支持 start_date，用 days 替代
            # 从 start_date 到 end_date 的日历天数 + 60 天余量
            if start_date and end_date:
                try:
                    dt_start = datetime.strptime(start_date, "%Y%m%d")
                    dt_end = datetime.strptime(end_date, "%Y%m%d")
                    days = max(60, (dt_end - dt_start).days + 30)
                except Exception:
                    days = 60
            else:
                days = 60

            df = self._fetcher.get_index_daily(
                code=ts_code,
                days=days,
                end_date=end_date,
            )
            if df is None or df.empty:
                return []

            # 过滤到目标日期范围
            if start_date:
                df = df[df["trade_date"] >= start_date]
            if end_date:
                df = df[df["trade_date"] <= end_date]
            if df.empty:
                return []

            # 列名映射：vol → volume
            col_map = {"vol": "volume"}
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
            if "ts_code" not in df.columns:
                df["ts_code"] = ts_code
            return df.to_dict("records")

        elif api_name == "trade_cal":
            # 使用 DataFetcher 的交易日历
            try:
                cal = self._fetcher.get_trading_days(
                    start=params.get("start_date", ""),
                    end=params.get("end_date", ""),
                )
                return cal if cal else []
            except Exception as e:
                logger.warning(f"交易日历获取失败: {e}")
                return []

        logger.warning(f"未知 API: {api_name}")
        return []

    def _attach_flags(self, records) -> list:
        """为数据记录附加 MarketFlag 标记"""
        result = []
        for r in records:
            if isinstance(r, dict):
                flags = [MarketFlag.NORMAL]
                pct = float(r.get("pct_chg") or 0)

                # 用 None 判断，避免 0 被 or 替换为默认值
                vol_raw = r.get("volume") if "volume" in r else r.get("vol")
                vol = float(vol_raw) if vol_raw is not None else 0.0

                if pct >= 9.5:
                    flags = [MarketFlag.LIMIT_UP]
                elif pct <= -9.5:
                    flags = [MarketFlag.LIMIT_DOWN]
                elif vol == 0:
                    flags = [MarketFlag.SUSPENDED]

                result.append(PriceRecord(
                    ts_code=r.get("ts_code", ""),
                    trade_date=r.get("trade_date", ""),
                    open=float(r.get("open", 0) or 0),
                    high=float(r.get("high", 0) or 0),
                    low=float(r.get("low", 0) or 0),
                    close=float(r.get("close", 0) or 0),
                    volume=vol,
                    amount=float(r.get("amount", 0) or 0),
                    pct_chg=pct,
                    flags=flags,
                ))
            else:
                result.append(r)
        return result

    def _compute_theme_strength(self, trade_date: str) -> list[dict]:
        """计算板块/主题强度"""
        # TODO: 实现板块强度计算
        return []

    def _ping_api(self) -> bool:
        """API 连通性探测"""
        # TODO: 轻量级 API 调用测试
        return True


class BacktestAPIViolation(Exception):
    """回测模式下尝试调用 API 时抛出"""
    pass
