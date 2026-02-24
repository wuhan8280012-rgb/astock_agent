"""
data_fetcher.py — Tushare 数据获取器
=====================================
对接 backtest_v6 + 全系统所有数据接口。

前置：
  pip install tushare pandas numpy

配置：
  方式1: 设环境变量 TUSHARE_TOKEN=你的token
  方式2: 直接改下面的 DEFAULT_TOKEN
  方式3: DataFetcher(token="你的token")

  Tushare token 获取: https://tushare.pro/register

接口清单（backtest_v6 + trade_signals + market_environment 全覆盖）：
  get_index_daily(code, days, end_date)       指数日线
  get_daily(symbol, days, end_date)           个股日线
  get_daily(symbol, start_date, end_date)     个股日线（区间）
  get_trading_days(start, end)                交易日历
  get_stock_pool()                            股票池（沪深300）
  get_stock_name(symbol)                      股票名称
  get_market_stats(date)                      涨跌统计+北向
  get_sector_performance(date)                板块涨跌
  get_fundamental(symbol)                     基本面（PE/PB/ROE等）
  get_valuation(symbol)                       估值数据
  get_north_flow(date)                        北向资金
"""

import os
import time
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Any

import pandas as pd
import numpy as np

try:
    import tushare as ts
except ImportError:
    print("❌ 请安装 tushare: pip install tushare")
    raise

# ═══════════════════════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════════════════════

# 在这里填入你的 token，或通过环境变量 TUSHARE_TOKEN 设置
DEFAULT_TOKEN = ""

# 缓存目录
CACHE_DIR = Path("cache/tushare")

# API 调用间隔（秒），避免频率限制
API_INTERVAL = 0.12  # Tushare 免费版限 500次/分钟


# ═══════════════════════════════════════════════════════════════
#  主类
# ═══════════════════════════════════════════════════════════════

class DataFetcher:
    """
    Tushare 数据获取器

    特性：
    - 本地文件缓存（避免重复请求）
    - 自动限速（防 ban）
    - 统一列名（trade_date/open/close/high/low/vol/amount）
    - 错误静默（个股级失败不影响批量扫描）
    """

    def __init__(self, token: str = None):
        token = token or os.environ.get("TUSHARE_TOKEN") or DEFAULT_TOKEN
        if not token:
            raise ValueError(
                "需要 Tushare token!\n"
                "  方式1: DataFetcher(token='你的token')\n"
                "  方式2: export TUSHARE_TOKEN=你的token\n"
                "  方式3: 编辑 data_fetcher.py 中的 DEFAULT_TOKEN\n"
                "  获取: https://tushare.pro/register"
            )

        try:
            # 兼容受限环境：tushare 可能尝试写 ~/tk.csv
            ts.set_token(token)
            self.pro = ts.pro_api()
        except Exception as e:
            print(f"  ⚠️ ts.set_token 失败，改用直传 token: {e}")
            self.pro = ts.pro_api(token)
        self._last_call = 0
        self._name_cache = {}

        # 缓存目录
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        print("✅ Tushare DataFetcher 就绪")

    # ─────────────────────────────────────────
    # 指数日线
    # ─────────────────────────────────────────

    def get_index_daily(self, code: str = "000001.SH", days: int = 60,
                        end_date: str = None) -> Optional[pd.DataFrame]:
        """
        获取指数日线

        Parameters
        ----------
        code : 指数代码 (000001.SH=上证, 399001.SZ=深证, 399006.SZ=创业板)
        days : 获取天数
        end_date : 截止日期 YYYYMMDD

        Returns
        -------
        DataFrame: trade_date, open, close, high, low, vol, amount
        """
        end_date = end_date or datetime.now().strftime("%Y%m%d")
        start_date = self._date_sub(end_date, days * 2)  # 多取一些，排除非交易日

        cache_key = f"idx_{code}_{start_date}_{end_date}"
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached.tail(days).reset_index(drop=True)

        self._throttle()
        try:
            df = self.pro.index_daily(
                ts_code=code,
                start_date=start_date,
                end_date=end_date,
                fields="trade_date,open,high,low,close,vol,amount",
            )
            if df is None or df.empty:
                return None

            df = df.sort_values("trade_date").reset_index(drop=True)
            self._write_cache(cache_key, df)
            return df.tail(days).reset_index(drop=True)

        except Exception as e:
            print(f"  ⚠️ 指数 {code} 数据获取失败: {e}")
            return None

    # ─────────────────────────────────────────
    # 个股日线
    # ─────────────────────────────────────────

    def get_daily(self, symbol: str, days: int = None,
                  end_date: str = None, start_date: str = None) -> Optional[pd.DataFrame]:
        """
        获取个股日线（前复权）

        Parameters
        ----------
        symbol : 股票代码 (600519.SH)
        days : 获取天数（与 start_date 二选一）
        end_date : 截止日期
        start_date : 起始日期（用于跟踪信号后续价格）

        Returns
        -------
        DataFrame: trade_date, open, close, high, low, vol, amount, turnover_rate
        """
        end_date = end_date or datetime.now().strftime("%Y%m%d")

        if start_date:
            actual_start = start_date
        elif days:
            actual_start = self._date_sub(end_date, days * 2)
        else:
            actual_start = self._date_sub(end_date, 180)

        cache_key = f"stk_{symbol}_{actual_start}_{end_date}"
        cached = self._read_cache(cache_key)
        if cached is not None:
            df = cached
            if start_date:
                df = df[df["trade_date"] >= start_date]
            if end_date:
                df = df[df["trade_date"] <= end_date]
            if days and not start_date:
                df = df.tail(days)
            return df.reset_index(drop=True)

        self._throttle()
        try:
            # 用 pro_bar 获取前复权数据
            df = ts.pro_bar(
                ts_code=symbol,
                start_date=actual_start,
                end_date=end_date,
                adj="qfq",
                factors=["tor"],  # turnover_rate
            )
            if df is None or df.empty:
                return None

            # 统一列名
            df = df.rename(columns={
                "ts_code": "symbol",
                "pct_chg": "pct_change",
                "tor": "turnover_rate",
            })
            df = df.sort_values("trade_date").reset_index(drop=True)

            # 确保 date 列也存在（一些模块用 date 而不是 trade_date）
            df["date"] = df["trade_date"]

            self._write_cache(cache_key, df)

            # 按参数裁剪
            if start_date:
                df = df[df["trade_date"] >= start_date]
            if days and not start_date:
                df = df.tail(days)
            return df.reset_index(drop=True)

        except Exception as e:
            # 个股级别静默失败
            return None

    # 别名
    get_stock_daily = get_daily
    get_k_data = get_daily

    # ─────────────────────────────────────────
    # 交易日历
    # ─────────────────────────────────────────

    def get_trading_days(self, start: str, end: str) -> list:
        """获取交易日列表"""
        cache_key = f"cal_{start}_{end}"
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached["trade_date"].tolist()

        self._throttle()
        try:
            df = self.pro.trade_cal(
                exchange="SSE",
                start_date=start,
                end_date=end,
                is_open="1",
            )
            df = df.sort_values("cal_date")
            df = df.rename(columns={"cal_date": "trade_date"})
            self._write_cache(cache_key, df[["trade_date"]])
            return df["trade_date"].tolist()
        except Exception as e:
            print(f"  ⚠️ 交易日历获取失败: {e}")
            # 回退：排除周末
            return self._generate_weekdays(start, end)

    # ─────────────────────────────────────────
    # 股票池
    # ─────────────────────────────────────────

    def get_stock_pool(self) -> list:
        """
        获取沪深300成分股

        Returns: ["600519.SH", "000858.SZ", ...]
        """
        cache_key = "pool_hs300"
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached["ts_code"].tolist()

        self._throttle()
        try:
            df = self.pro.index_weight(
                index_code="399300.SZ",
                start_date=self._date_sub(datetime.now().strftime("%Y%m%d"), 60),
            )
            if df is None or df.empty:
                # 备用接口
                df = self.pro.index_member(index_code="399300.SZ")

            if df is not None and not df.empty:
                codes = df["con_code"].unique().tolist() if "con_code" in df.columns else []
                if not codes and "ts_code" in df.columns:
                    codes = df["ts_code"].unique().tolist()

                pool_df = pd.DataFrame({"ts_code": codes})
                self._write_cache(cache_key, pool_df)
                print(f"  股票池: {len(codes)} 只 (沪深300)")
                return codes
        except Exception as e:
            print(f"  ⚠️ 沪深300成分获取失败: {e}")

        # 回退池
        return self._fallback_pool()

    # ─────────────────────────────────────────
    # 股票名称
    # ─────────────────────────────────────────

    def get_stock_name(self, symbol: str) -> str:
        """获取股票名称"""
        if symbol in self._name_cache:
            return self._name_cache[symbol]

        self._throttle()
        try:
            df = self.pro.namechange(ts_code=symbol)
            if df is not None and not df.empty:
                name = df.iloc[0]["name"]
                self._name_cache[symbol] = name
                return name
        except Exception:
            pass

        try:
            df = self.pro.stock_basic(ts_code=symbol)
            if df is not None and not df.empty:
                name = df.iloc[0]["name"]
                self._name_cache[symbol] = name
                return name
        except Exception:
            pass

        return symbol

    # ─────────────────────────────────────────
    # 市场统计
    # ─────────────────────────────────────────

    def get_market_stats(self, date: str) -> Optional[dict]:
        """
        获取某日市场涨跌统计

        Returns: {up_count, down_count, limit_up, limit_down,
                  north_flow, amount_billion}
        """
        cache_key = f"mkt_{date}_v2"
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached.iloc[0].to_dict() if len(cached) > 0 else None

        self._throttle()
        try:
            # 获取全市场日线（含代码，用于分板块判断涨跌停）
            df = self.pro.daily(
                trade_date=date,
                fields="ts_code,pct_chg,amount",
            )
            if df is None or df.empty:
                return None

            pcts = df["pct_chg"].dropna()

            # ── 涨跌家数 ──
            up_count = int((pcts > 0).sum())
            down_count = int((pcts < 0).sum())

            # ── 涨跌停：按板块区分阈值 ──
            # 主板(60/00开头): ±10% → 判定 ≥9.5%
            # 创业板(30开头)/科创板(688开头): ±20% → 判定 ≥19.5%
            # ST(*ST): ±5% → 判定 ≥4.5%
            # 北交所(8开头): ±30% → 判定 ≥29.5%
            limit_up = 0
            limit_down = 0

            for _, row in df.iterrows():
                code = str(row.get("ts_code", ""))
                pct = row.get("pct_chg")
                if pd.isna(pct):
                    continue

                # 判断板块类型
                if code.startswith("30") or code.startswith("688"):
                    threshold = 19.5  # 创业板/科创板
                elif code.startswith("8"):
                    threshold = 29.5  # 北交所
                else:
                    threshold = 9.5   # 主板

                if pct >= threshold:
                    limit_up += 1
                elif pct <= -threshold:
                    limit_down += 1

            # ── 全市场成交额（亿元）──
            amount_billion = 0.0
            if "amount" in df.columns:
                # Tushare daily 的 amount 单位是 千元
                total_amount = pd.to_numeric(
                    df["amount"], errors="coerce").sum()
                amount_billion = round(total_amount / 1e5, 2)  # 千元→亿元

            stats = {
                "up_count": up_count,
                "down_count": down_count,
                "limit_up": limit_up,
                "limit_down": limit_down,
                "north_flow": self._get_north_flow(date),
                "amount_billion": amount_billion,
            }

            # 缓存
            stats_df = pd.DataFrame([stats])
            self._write_cache(cache_key, stats_df)
            return stats

        except Exception as e:
            print(f"  ⚠️ 市场统计获取失败 ({date}): {e}")
            return None

    def get_market_overview(self, date: str) -> Optional[dict]:
        """get_market_stats 的别名"""
        return self.get_market_stats(date)

    # ─────────────────────────────────────────
    # 北向资金
    # ─────────────────────────────────────────

    def _get_north_flow(self, date: str) -> float:
        """
        获取北向资金净流入（亿元）

        Tushare moneyflow_hsgt 字段说明：
          north_money: 北向净流入（百万元）
          hgt: 沪股通（百万元）
          sgt: 深股通（百万元）
        """
        try:
            self._throttle()
            df = self.pro.moneyflow_hsgt(trade_date=date)
            if df is not None and not df.empty:
                row = df.iloc[0]

                # 方法1: 直接读 north_money（百万元）
                if "north_money" in df.columns:
                    val = row["north_money"]
                    if pd.notna(val) and val != 0:
                        return round(float(val) / 100, 2)  # 百万→亿

                # 方法2: hgt + sgt（各自百万元）
                hgt = float(row.get("hgt", 0) or 0)
                sgt = float(row.get("sgt", 0) or 0)
                if hgt != 0 or sgt != 0:
                    return round((hgt + sgt) / 100, 2)  # 百万→亿

                print(f"  ⚠️ moneyflow_hsgt 有数据但字段为空: {row.to_dict()}")
            else:
                print(f"  ⚠️ moneyflow_hsgt({date}) 返回空")
        except Exception as e:
            print(f"  ⚠️ 北向资金获取失败: {e}")

        return 0.0

    def get_north_flow(self, date: str) -> float:
        """公开接口"""
        return self._get_north_flow(date)

    # ─────────────────────────────────────────
    # 板块表现
    # ─────────────────────────────────────────

    def get_sector_performance(self, date: str) -> list:
        """
        获取板块涨跌

        策略：
        1. 尝试申万一级行业指数（sw_daily）
        2. 尝试同花顺概念板块（ths_daily）
        3. 兜底：从全市场个股按代码前缀分板块估算

        Returns: [{"name": "...", "pct_change": 2.5}, ...]
        """
        cache_key = f"sector_{date}_v2"
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached.to_dict("records")

        sectors = []

        # ── 方法1: 申万行业指数 ──
        # 申万一级行业指数代码（31个行业）
        SW_L1_CODES = [
            "801010.SI", "801020.SI", "801030.SI", "801040.SI", "801050.SI",
            "801080.SI", "801110.SI", "801120.SI", "801130.SI", "801140.SI",
            "801150.SI", "801160.SI", "801170.SI", "801180.SI", "801200.SI",
            "801210.SI", "801230.SI", "801710.SI", "801720.SI", "801730.SI",
            "801740.SI", "801750.SI", "801760.SI", "801770.SI", "801780.SI",
            "801790.SI", "801880.SI", "801890.SI", "801950.SI", "801960.SI",
            "801970.SI",
        ]
        try:
            self._throttle()
            df = self.pro.sw_daily(trade_date=date)
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    name = row.get("name", row.get("ts_code", ""))
                    pct = float(row.get("pct_chg", 0) or 0)
                    if name:
                        sectors.append({"name": name, "pct_change": pct})
        except Exception as e:
            print(f"  ⚠️ 申万行业: {e}")

        # ── 方法2: 同花顺板块 ──
        if not sectors:
            try:
                self._throttle()
                df = self.pro.ths_daily(trade_date=date)
                if df is not None and not df.empty:
                    for _, row in df.iterrows():
                        name = row.get("name", row.get("ts_code", ""))
                        pct = float(row.get("pct_chg", 0) or 0)
                        if name:
                            sectors.append({"name": name, "pct_change": pct})
            except Exception as e:
                print(f"  ⚠️ 同花顺板块: {e}")

        # ── 方法3: 兜底——从已有的 market_stats 数据推算 ──
        if not sectors:
            try:
                self._throttle()
                df = self.pro.daily(
                    trade_date=date,
                    fields="ts_code,pct_chg",
                )
                if df is not None and not df.empty:
                    # 按代码前3位分组（简易板块分类）
                    # 000/001=深主板, 002=中小板, 300=创业板,
                    # 600/601/603=沪主板, 688=科创板
                    board_map = {
                        "600": "沪主板", "601": "沪主板", "603": "沪主板",
                        "000": "深主板", "001": "深主板", "002": "中小板",
                        "300": "创业板", "301": "创业板",
                        "688": "科创板",
                    }
                    groups = {}
                    for _, row in df.iterrows():
                        ts_code = str(row.get("ts_code", ""))
                        code3 = ts_code[:3]
                        board = board_map.get(code3)
                        if not board and ts_code.startswith("8"):
                            board = "北交所"
                        if not board:
                            board = "其他"
                        if board:
                            if board not in groups:
                                groups[board] = []
                            pct = row.get("pct_chg")
                            if pd.notna(pct):
                                groups[board].append(float(pct))

                    for name, pcts in groups.items():
                        if pcts:
                            avg_pct = sum(pcts) / len(pcts)
                            sectors.append({
                                "name": name,
                                "pct_change": round(avg_pct, 2),
                            })

                    sectors.sort(key=lambda x: x["pct_change"], reverse=True)
            except Exception as e:
                print(f"  ⚠️ 板块推算: {e}")

        if sectors:
            self._write_cache(cache_key, pd.DataFrame(sectors))

        return sectors

    def get_sectors(self, date: str) -> list:
        """get_sector_performance 别名"""
        return self.get_sector_performance(date)

    # ─────────────────────────────────────────
    # 板块成分股（龙头突破用）
    # ─────────────────────────────────────────

    def get_sector_stocks(self, date: str) -> dict:
        """
        获取活跃板块及其成分股

        Returns: {"半导体": ["688981.SH", ...], "消费电子": [...]}
        """
        cache_key = f"sec_stocks_{date[:6]}"  # 按月缓存
        cached = self._read_cache(cache_key)
        if cached is not None:
            return json.loads(cached.iloc[0]["data"]) if len(cached) > 0 else {}

        self._throttle()
        try:
            # 获取申万行业分类
            df = self.pro.index_classify(level="L2", src="SW2021")
            if df is None or df.empty:
                return {}

            result = {}
            # 只取前10个行业，避免请求过多
            for _, row in df.head(10).iterrows():
                idx_code = row.get("index_code", "")
                name = row.get("industry_name", "")
                if not idx_code:
                    continue

                self._throttle()
                try:
                    members = self.pro.index_member(index_code=idx_code)
                    if members is not None and not members.empty:
                        codes = members["con_code"].tolist()
                        result[name] = codes[:20]  # 每板块取前20
                except Exception:
                    continue

            if result:
                cache_df = pd.DataFrame([{"data": json.dumps(result, ensure_ascii=False)}])
                self._write_cache(cache_key, cache_df)

            return result
        except Exception:
            return {}

    # ─────────────────────────────────────────
    # 基本面数据（价值投资用）
    # ─────────────────────────────────────────

    def get_fundamental(self, symbol: str) -> Optional[dict]:
        """获取综合基本面数据"""
        return self.get_valuation(symbol)

    def get_valuation(self, symbol: str) -> Optional[dict]:
        """获取估值+财务数据"""
        cache_key = f"val_{symbol}"
        cached = self._read_cache(cache_key)
        if cached is not None and len(cached) > 0:
            return cached.iloc[0].to_dict()

        self._throttle()
        try:
            # 每日估值
            df_val = self.pro.daily_basic(
                ts_code=symbol,
                fields="ts_code,pe_ttm,pb,ps_ttm,dv_ratio,total_mv,circ_mv,turnover_rate",
            )

            result = {}
            if df_val is not None and not df_val.empty:
                row = df_val.iloc[0]
                result["pe"] = float(row.get("pe_ttm", 0) or 0)
                result["pb"] = float(row.get("pb", 0) or 0)
                result["ps"] = float(row.get("ps_ttm", 0) or 0)
                result["dividend_yield"] = float(row.get("dv_ratio", 0) or 0)
                result["total_mv"] = float(row.get("total_mv", 0) or 0)
                result["circ_mv"] = float(row.get("circ_mv", 0) or 0)

            # 财务指标
            self._throttle()
            df_fin = self.pro.fina_indicator(
                ts_code=symbol,
                fields="ts_code,roe,grossprofit_margin,debt_to_assets,"
                       "op_yoy,revenue_yoy,netprofit_yoy",
            )
            if df_fin is not None and not df_fin.empty:
                row = df_fin.iloc[0]
                result["roe"] = float(row.get("roe", 0) or 0)
                result["gross_margin"] = float(row.get("grossprofit_margin", 0) or 0)
                result["debt_ratio"] = float(row.get("debt_to_assets", 0) or 0)
                result["revenue_growth"] = float(row.get("revenue_yoy", 0) or 0)
                result["profit_growth"] = float(row.get("netprofit_yoy", 0) or 0)

            if result:
                result_df = pd.DataFrame([result])
                self._write_cache(cache_key, result_df)
                return result

        except Exception:
            pass

        return None

    # ═══════════════════════════════════════════════════════════
    #  缓存
    # ═══════════════════════════════════════════════════════════

    def _cache_path(self, key: str) -> Path:
        """生成缓存文件路径"""
        h = hashlib.md5(key.encode()).hexdigest()[:12]
        safe_key = key.replace("/", "_").replace(".", "_")[:40]
        return CACHE_DIR / f"{safe_key}_{h}.pkl"

    def _read_cache(self, key: str, max_age_hours: int = 24) -> Optional[pd.DataFrame]:
        """读缓存（默认24小时过期）"""
        path = self._cache_path(key)
        if not path.exists():
            return None

        # 检查过期
        age = time.time() - path.stat().st_mtime
        if age > max_age_hours * 3600:
            return None

        try:
            return pd.read_pickle(path)
        except Exception:
            path.unlink(missing_ok=True)
            return None

    def _write_cache(self, key: str, df: pd.DataFrame):
        """写缓存"""
        try:
            path = self._cache_path(key)
            df.to_pickle(path)
        except Exception:
            pass

    def clear_cache(self):
        """清空缓存"""
        import shutil
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            print("缓存已清空")

    # ═══════════════════════════════════════════════════════════
    #  限速
    # ═══════════════════════════════════════════════════════════

    def _throttle(self):
        """API 调用限速"""
        elapsed = time.time() - self._last_call
        if elapsed < API_INTERVAL:
            time.sleep(API_INTERVAL - elapsed)
        self._last_call = time.time()

    # ═══════════════════════════════════════════════════════════
    #  工具
    # ═══════════════════════════════════════════════════════════

    def _date_sub(self, date_str: str, days: int) -> str:
        dt = datetime.strptime(date_str, "%Y%m%d") - timedelta(days=days)
        return dt.strftime("%Y%m%d")

    def _generate_weekdays(self, start_str, end_str) -> list:
        start = datetime.strptime(start_str, "%Y%m%d")
        end = datetime.strptime(end_str, "%Y%m%d")
        days = []
        current = start
        while current <= end:
            if current.weekday() < 5:
                days.append(current.strftime("%Y%m%d"))
            current += timedelta(days=1)
        return days

    def _fallback_pool(self) -> list:
        """回退股票池"""
        pool = [
            "600519.SH", "601318.SH", "600036.SH", "600276.SH", "601166.SH",
            "000858.SZ", "000333.SZ", "002415.SZ", "000001.SZ", "002594.SZ",
            "600900.SH", "601888.SH", "600309.SH", "603259.SH", "601012.SH",
            "000568.SZ", "002475.SZ", "300750.SZ", "002714.SZ", "300059.SZ",
            "600585.SH", "601669.SH", "600887.SH", "600030.SH", "601398.SH",
            "000725.SZ", "002304.SZ", "300015.SZ", "000002.SZ", "002230.SZ",
            "600809.SH", "601601.SH", "600050.SH", "603288.SH", "601088.SH",
            "000651.SZ", "002049.SZ", "300124.SZ", "000063.SZ", "002352.SZ",
            "600570.SH", "601766.SH", "600104.SH", "600000.SH", "601857.SH",
            "000100.SZ", "002371.SZ", "300014.SZ", "000538.SZ", "002607.SZ",
        ]
        print(f"  使用回退股票池: {len(pool)} 只")
        return pool


# ═══════════════════════════════════════════════════════════════
#  快速测试
# ═══════════════════════════════════════════════════════════════

def test():
    """快速测试数据获取"""
    print("=" * 50)
    print("  DataFetcher 快速测试")
    print("=" * 50)

    f = DataFetcher()

    # 1. 交易日历
    print("\n[1] 交易日历")
    days = f.get_trading_days("20251101", "20251130")
    print(f"  11月交易日: {len(days)} 天")
    if days:
        print(f"  首日: {days[0]}  末日: {days[-1]}")

    # 2. 指数日线
    print("\n[2] 上证指数")
    idx = f.get_index_daily("000001.SH", days=5, end_date="20251128")
    if idx is not None:
        print(f"  {len(idx)} 行")
        print(idx[["trade_date", "close", "vol"]].tail(3).to_string(index=False))
    else:
        print("  ❌ 获取失败")

    # 3. 个股日线
    print("\n[3] 贵州茅台")
    stk = f.get_daily("600519.SH", days=5, end_date="20251128")
    if stk is not None:
        print(f"  {len(stk)} 行")
        cols = [c for c in ["trade_date", "close", "vol", "turnover_rate"] if c in stk.columns]
        print(stk[cols].tail(3).to_string(index=False))
    else:
        print("  ❌ 获取失败")

    # 4. 股票池
    print("\n[4] 股票池")
    pool = f.get_stock_pool()
    print(f"  {len(pool)} 只")

    # 5. 市场统计
    print("\n[5] 市场统计 (20251128)")
    stats = f.get_market_stats("20251128")
    if stats:
        print(f"  涨: {stats['up_count']}  跌: {stats['down_count']}")
        print(f"  涨停: {stats['limit_up']}  跌停: {stats['limit_down']}")
        print(f"  北向: {stats['north_flow']} 亿")
    else:
        print("  ❌ 获取失败")

    # 6. 股票名称
    print("\n[6] 名称查询")
    name = f.get_stock_name("600519.SH")
    print(f"  600519.SH → {name}")

    print("\n" + "=" * 50)
    print("  测试完成 ✅")
    print("=" * 50)


if __name__ == "__main__":
    test()
