"""
backtest_runner.py — 真实数据回测运行器
========================================
开箱即用，拿到本地直接 python3 backtest_runner.py 跑。

数据源优先级：
  1. 你现有的 data_fetcher（Tushare）
  2. AKShare（免费，无需 API key）
  3. 都没有 → 提示安装

依赖：
  pip install akshare pandas numpy   # 如果没有你的 data_fetcher

用法：
  # 自动检测数据源，跑 2025.11-12 回测
  python3 backtest_runner.py

  # 指定日期范围
  python3 backtest_runner.py --start 20251001 --end 20251231

  # 用 AKShare（不用你的 data_fetcher）
  python3 backtest_runner.py --source akshare

  # 敏感性分析
  python3 backtest_runner.py --sensitivity

  # 导出 CSV
  python3 backtest_runner.py --export

  # 指定股票池文件（每行一个代码）
  python3 backtest_runner.py --pool pool.txt
"""

import sys
import os
import time
import json
import argparse
import warnings
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

# 把当前目录和 backtest_v6 所在目录加到 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════════
#  数据适配层
# ═══════════════════════════════════════════════════════════════

class AKShareFetcher:
    """
    AKShare 数据适配器（免费，无需 API Key）

    自动对齐 backtest_v6 需要的接口：
      - get_index_daily(code, days, end_date)
      - get_daily(symbol, days, end_date)
      - get_daily(symbol, start_date, end_date)  # 用于跟踪后续价格
      - get_stock_pool()
      - get_stock_name(symbol)
      - get_trading_days(start, end)
      - get_market_stats(date)
      - get_sector_performance(date)
    """

    def __init__(self):
        try:
            import akshare as ak
            import pandas as pd
            self.ak = ak
            self.pd = pd
        except ImportError:
            print("❌ 请安装 akshare: pip install akshare")
            sys.exit(1)

        self._cache = {}
        self._name_map = {}
        self._pool = None
        print("✅ AKShare 数据源就绪")

    def get_index_daily(self, code="000001.SH", days=60, end_date=None):
        """获取指数日线"""
        cache_key = f"idx_{code}_{end_date}_{days}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # AKShare 用不同的代码格式
            ak_code = code.replace(".SH", "").replace(".SZ", "")
            if code.endswith(".SH"):
                ak_code = f"sh{ak_code}"
            else:
                ak_code = f"sz{ak_code}"

            df = self.ak.stock_zh_index_daily(symbol=ak_code)
            df = df.rename(columns={"date": "trade_date"})
            df["trade_date"] = self.pd.to_datetime(df["trade_date"]).dt.strftime("%Y%m%d")

            if end_date:
                df = df[df["trade_date"] <= end_date]
            df = df.tail(days).reset_index(drop=True)

            # 确保有 amount 列
            if "amount" not in df.columns and "volume" in df.columns:
                df["amount"] = df["volume"]

            self._cache[cache_key] = df
            return df
        except Exception as e:
            print(f"  ⚠️ 获取指数 {code} 失败: {e}")
            return None

    def get_daily(self, symbol, days=None, end_date=None,
                  start_date=None):
        """获取个股日线"""
        cache_key = f"stk_{symbol}_{start_date}_{end_date}_{days}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            ak_code = symbol.replace(".SH", "").replace(".SZ", "")

            # AKShare: stock_zh_a_hist
            end_dt = end_date or datetime.now().strftime("%Y%m%d")
            if start_date:
                start_dt = start_date
            elif days:
                start_dt = (datetime.strptime(end_dt, "%Y%m%d") -
                           timedelta(days=days * 2)).strftime("%Y%m%d")
            else:
                start_dt = (datetime.strptime(end_dt, "%Y%m%d") -
                           timedelta(days=180)).strftime("%Y%m%d")

            df = self.ak.stock_zh_a_hist(
                symbol=ak_code,
                period="daily",
                start_date=start_dt,
                end_date=end_dt,
                adjust="qfq",  # 前复权
            )

            if df is None or df.empty:
                return None

            # 统一列名
            col_map = {
                "日期": "trade_date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "vol",
                "成交额": "amount",
                "换手率": "turnover_rate",
            }
            df = df.rename(columns=col_map)

            if "trade_date" in df.columns:
                df["trade_date"] = self.pd.to_datetime(df["trade_date"]).dt.strftime("%Y%m%d")
                df["date"] = df["trade_date"]

            if end_date and "trade_date" in df.columns:
                df = df[df["trade_date"] <= end_date]
            if start_date and "trade_date" in df.columns:
                df = df[df["trade_date"] >= start_date]

            if days and not start_date:
                df = df.tail(days)

            df = df.reset_index(drop=True)
            self._cache[cache_key] = df
            return df

        except Exception as e:
            # 静默失败，个股级别错误太多
            return None

    get_stock_daily = get_daily
    get_k_data = get_daily

    def get_trading_days(self, start, end):
        """获取交易日列表"""
        cache_key = f"td_{start}_{end}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            df = self.ak.tool_trade_date_hist_sina()
            df["trade_date"] = self.pd.to_datetime(df["trade_date"]).dt.strftime("%Y%m%d")
            days = df[(df["trade_date"] >= start) &
                     (df["trade_date"] <= end)]["trade_date"].tolist()
            self._cache[cache_key] = days
            return days
        except Exception:
            # 回退：简单排除周末
            from backtest_v6 import _generate_weekdays
            return _generate_weekdays(start, end)

    def get_stock_pool(self):
        """
        获取股票池

        策略：取沪深300成分股（流动性好、数据全）
        """
        if self._pool:
            return self._pool

        try:
            print("  加载沪深300成分股...")
            df = self.ak.index_stock_cons_csindex(symbol="000300")
            codes = df["成分券代码"].tolist() if "成分券代码" in df.columns else []

            # 转换为标准格式
            pool = []
            for code in codes:
                code = str(code).zfill(6)
                suffix = ".SH" if code.startswith("6") else ".SZ"
                pool.append(code + suffix)

            self._pool = pool
            print(f"  股票池: {len(pool)} 只")
            return pool

        except Exception as e:
            print(f"  ⚠️ 加载股票池失败: {e}")
            # 回退: 手动选一批活跃股
            self._pool = self._fallback_pool()
            print(f"  使用回退股票池: {len(self._pool)} 只")
            return self._pool

    def get_stock_name(self, symbol):
        """获取股票名称"""
        if symbol in self._name_map:
            return self._name_map[symbol]

        try:
            code = symbol.replace(".SH", "").replace(".SZ", "")
            df = self.ak.stock_individual_info_em(symbol=code)
            if df is not None and not df.empty:
                # 在结果中找名称
                for _, row in df.iterrows():
                    if "名称" in str(row.get("item", "")):
                        name = str(row.get("value", symbol))
                        self._name_map[symbol] = name
                        return name
        except Exception:
            pass
        return symbol

    def get_market_stats(self, date):
        """获取市场统计数据"""
        cache_key = f"mkt_{date}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # 用 A 股涨跌统计
            df = self.ak.stock_zh_a_spot_em()
            if df is None or df.empty:
                return None

            col_pct = "涨跌幅" if "涨跌幅" in df.columns else None
            if not col_pct:
                return None

            pcts = df[col_pct].dropna()
            stats = {
                "up_count": int((pcts > 0).sum()),
                "down_count": int((pcts < 0).sum()),
                "limit_up": int((pcts >= 9.5).sum()),
                "limit_down": int((pcts <= -9.5).sum()),
                "north_flow": 0,  # 北向需要单独接口
            }
            self._cache[cache_key] = stats
            return stats
        except Exception:
            return None

    def get_sector_performance(self, date):
        """获取板块表现"""
        try:
            df = self.ak.stock_board_industry_name_em()
            if df is None or df.empty:
                return []
            sectors = []
            for _, row in df.iterrows():
                sectors.append({
                    "name": row.get("板块名称", ""),
                    "pct_change": float(row.get("涨跌幅", 0) or 0),
                })
            return sectors
        except Exception:
            return []

    def _fallback_pool(self):
        """回退股票池：手动选50只活跃股"""
        return [
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


class TushareFetcherAdapter:
    """
    适配你现有的 data_fetcher（Tushare 版）

    自动检测你的 DataFetcher 类，包装为 backtest_v6 需要的接口。
    """

    def __init__(self, fetcher):
        self.f = fetcher
        print("✅ 使用你的 DataFetcher (Tushare)")

    def __getattr__(self, name):
        """透传所有方法调用到原始 fetcher"""
        return getattr(self.f, name)


# ═══════════════════════════════════════════════════════════════
#  运行器
# ═══════════════════════════════════════════════════════════════

def get_fetcher(source="auto"):
    """
    获取数据源

    source: "auto" | "tushare" | "akshare"
    """
    if source in ("auto", "tushare"):
        try:
            from data_fetcher import DataFetcher
            f = DataFetcher()
            return TushareFetcherAdapter(f)
        except Exception as e:
            if source == "tushare":
                print(f"❌ DataFetcher 加载失败: {e}")
                sys.exit(1)
            print(f"  DataFetcher 不可用 ({e})，尝试 AKShare...")

    if source in ("auto", "akshare"):
        try:
            return AKShareFetcher()
        except SystemExit:
            raise
        except Exception as e:
            print(f"❌ AKShare 不可用: {e}")
            print("请安装: pip install akshare pandas numpy")
            sys.exit(1)

    print("❌ 无可用数据源")
    sys.exit(1)


def load_pool_file(path):
    """从文件加载股票池"""
    codes = []
    with open(path, "r") as f:
        for line in f:
            code = line.strip()
            if code and not code.startswith("#"):
                if not ("." in code):
                    suffix = ".SH" if code.startswith("6") else ".SZ"
                    code = code + suffix
                codes.append(code)
    print(f"  从 {path} 加载 {len(codes)} 只股票")
    return codes


def run_backtest(args):
    """执行完整回测"""
    from backtest_v6_1 import BacktestV6

    print("=" * 65)
    print("  v6.0 真实数据回测")
    print(f"  {args.start} ~ {args.end}")
    print("=" * 65)

    # 1. 数据源
    print("\n[1/5] 初始化数据源...")
    fetcher = get_fetcher(args.source)

    # 2. 股票池
    pool = None
    if args.pool:
        pool = load_pool_file(args.pool)
    elif not args.small:
        # 默认用沪深300
        try:
            pool = fetcher.get_stock_pool()
        except Exception:
            pool = None

    if args.small:
        # 小样本模式：只用 20 只股票快速验证
        pool = [
            "600519.SH", "000858.SZ", "601318.SH", "000333.SZ",
            "600036.SH", "002415.SZ", "600276.SH", "000001.SZ",
            "601166.SH", "300750.SZ", "600887.SH", "002594.SZ",
            "600309.SH", "000568.SZ", "603259.SH", "601012.SH",
            "000725.SZ", "002304.SZ", "600585.SH", "601669.SH",
        ]
        print(f"  小样本模式: {len(pool)} 只股票")

    # 3. 初始化回测引擎
    print("\n[2/5] 初始化回测引擎...")
    bt = BacktestV6(data_fetcher=fetcher)
    bt.env_threshold = args.threshold
    bt.hold_days = args.hold
    bt.stop_loss_pct = args.stop
    bt.take_profit_pct = args.profit

    print(f"  环境阈值: {bt.env_threshold}")
    print(f"  持有天数: {bt.hold_days}")
    print(f"  止损: {bt.stop_loss_pct}%  止盈: {bt.take_profit_pct}%")
    if pool:
        print(f"  股票池: {len(pool)} 只")

    # 4. 运行
    print("\n[3/5] 开始回测...")
    t0 = time.time()
    report = bt.run(args.start, args.end, stock_pool=pool, verbose=True)
    elapsed = time.time() - t0
    print(f"\n  耗时: {elapsed:.1f} 秒")

    # 5. 敏感性分析
    if args.sensitivity:
        print("\n[4/5] 敏感性分析...")
        bt.sensitivity_analysis()
    else:
        print("\n[4/5] 敏感性分析 (跳过，加 --sensitivity 开启)")

    # 6. 导出
    print("\n[5/5] 保存结果...")

    # 保存报告文本
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    report_name = f"backtest_v6_{args.start}_{args.end}"
    report_path = output_dir / f"{report_name}.txt"
    report_path.write_text(report["summary"], encoding="utf-8")
    print(f"  报告: {report_path}")

    # JSON 详细数据
    json_path = output_dir / f"{report_name}.json"
    save_data = {k: v for k, v in report.items()
                 if k not in ("daily_env",)}
    # trades 保留
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"  详细: {json_path}")

    if args.export:
        csv_path = output_dir / f"{report_name}_trades.csv"
        env_path = output_dir / f"{report_name}_env.csv"
        bt.export_csv(str(csv_path))
        bt.export_env_csv(str(env_path))

    print("\n" + "=" * 65)
    print("  回测完成 ✅")
    print("=" * 65)

    return report


def run_env_only(args):
    """只跑环境评分"""
    from backtest_v6_1 import BacktestV6

    print(f"环境评分: {args.start} ~ {args.end}\n")
    fetcher = get_fetcher(args.source)
    bt = BacktestV6(data_fetcher=fetcher)

    results = bt.run_env_only(args.start, args.end)

    # 统计
    scores = [r["total_score"] for r in results]
    if scores:
        print(f"\n--- 统计 ---")
        print(f"  天数: {len(scores)}")
        print(f"  均值: {sum(scores)/len(scores):.1f}")
        print(f"  范围: {min(scores)} ~ {max(scores)}")
        bad = sum(1 for s in scores if s < 60)
        print(f"  较差(<60): {bad} 天 ({bad/len(scores)*100:.0f}%)")

    if args.export:
        bt.daily_env = results
        bt.export_env_csv(f"outputs/env_{args.start}_{args.end}.csv")


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="v6.0 真实数据回测运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python3 backtest_runner.py                              # 默认 11-12月
  python3 backtest_runner.py --start 20251001 --end 20251231
  python3 backtest_runner.py --source akshare             # 强制用 AKShare
  python3 backtest_runner.py --sensitivity --export       # 敏感性 + CSV导出
  python3 backtest_runner.py --small                      # 20只股票快速验证
  python3 backtest_runner.py --env-only                   # 只看环境评分
  python3 backtest_runner.py --pool my_stocks.txt         # 自定义股票池
  python3 backtest_runner.py --threshold 55 --hold 5      # 调参
        """,
    )

    parser.add_argument("--start", default="20251101", help="开始日期 (默认 20251101)")
    parser.add_argument("--end", default="20251231", help="结束日期 (默认 20251231)")
    parser.add_argument("--source", choices=["auto", "tushare", "akshare"],
                        default="auto", help="数据源 (默认 auto)")
    parser.add_argument("--env-only", action="store_true", help="只看环境评分")
    parser.add_argument("--sensitivity", action="store_true", help="阈值敏感性分析")
    parser.add_argument("--export", action="store_true", help="导出 CSV")
    parser.add_argument("--small", action="store_true", help="小样本快速验证 (20只)")
    parser.add_argument("--pool", type=str, help="股票池文件 (每行一个代码)")
    parser.add_argument("--threshold", type=int, default=60, help="环境阈值 (默认 60)")
    parser.add_argument("--hold", type=int, default=10, help="持有天数 (默认 10)")
    parser.add_argument("--stop", type=float, default=-7.0, help="止损%% (默认 -7)")
    parser.add_argument("--profit", type=float, default=15.0, help="止盈%% (默认 15)")

    args = parser.parse_args()

    if args.env_only:
        run_env_only(args)
    else:
        run_backtest(args)


if __name__ == "__main__":
    main()
