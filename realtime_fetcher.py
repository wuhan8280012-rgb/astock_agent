"""
realtime_fetcher.py — 盘中实时数据自动获取
=============================================
自动采集 7 个盘中字段，直接喂给 market_environment.evaluate()。

数据源优先级：
  1. AKShare（免费，pip install akshare）
  2. 东方财富 HTTP API（免费，无需安装）
  3. Tushare 实时接口（需要积分）

获取字段：
  index_pct       上证涨跌幅%
  up_count        涨家数
  down_count      跌家数
  limit_up        涨停家数
  limit_down      跌停家数
  amount_billion  成交额(亿)
  north_flow      北向净流入(亿)

用法：
  # 作为模块
  from realtime_fetcher import get_intraday
  data = get_intraday()
  result = env.evaluate(intraday=data, diagnose=True)

  # 一键评分（内置调用 MarketEnvironment）
  python3 realtime_fetcher.py

  # 只看数据不评分
  python3 realtime_fetcher.py --data-only

  # 指定数据源
  python3 realtime_fetcher.py --source eastmoney
"""

import json
import time
import sys
import os
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════════
#  数据源 1: AKShare（推荐，最全）
# ═══════════════════════════════════════════════════════════════

def _fetch_akshare() -> Optional[dict]:
    """
    AKShare 实时数据

    需要: pip install akshare
    """
    try:
        import akshare as ak
        import pandas as pd
    except ImportError:
        return None

    data = {}

    # ── 指数实时（只取涨跌幅）──
    try:
        df_idx = ak.stock_zh_index_spot_em()
        if df_idx is not None and not df_idx.empty:
            # 找上证指数
            sh_row = df_idx[df_idx["代码"].str.contains("000001")]
            if not sh_row.empty:
                row = sh_row.iloc[0]
                data["index_pct"] = float(row.get("涨跌幅", 0) or 0)
    except Exception as e:
        print(f"  ⚠️ AKShare 指数: {e}")

    # ── 全市场涨跌统计 + 成交额汇总 ──
    try:
        df_all = ak.stock_zh_a_spot_em()
        if df_all is not None and not df_all.empty:
            pct_col = "涨跌幅" if "涨跌幅" in df_all.columns else None
            if pct_col:
                pcts = pd.to_numeric(df_all[pct_col], errors="coerce")
                data["up_count"] = int((pcts > 0).sum())
                data["down_count"] = int((pcts < 0).sum())

                # 涨跌停按板块阈值统计，避免 9.5% 一刀切误判
                limit_up = 0
                limit_down = 0
                code_col = "代码" if "代码" in df_all.columns else None
                if code_col:
                    for _, row in df_all[[code_col, pct_col]].iterrows():
                        code = str(row.get(code_col, ""))
                        pct = pd.to_numeric(row.get(pct_col), errors="coerce")
                        if pd.isna(pct):
                            continue
                        if code.startswith(("300", "301", "688")):
                            threshold = 19.5
                        elif code.startswith("8"):
                            threshold = 29.5
                        else:
                            threshold = 9.5
                        if pct >= threshold:
                            limit_up += 1
                        elif pct <= -threshold:
                            limit_down += 1
                else:
                    limit_up = int((pcts >= 9.5).sum())
                    limit_down = int((pcts <= -9.5).sum())

                data["limit_up"] = limit_up
                data["limit_down"] = limit_down

            # 成交额：从全市场个股汇总（沪+深，不只是上证指数）
            amt_col = "成交额" if "成交额" in df_all.columns else None
            if amt_col:
                total_amt = pd.to_numeric(df_all[amt_col], errors="coerce").sum()
                if total_amt > 0:
                    data["amount_billion"] = round(total_amt / 1e8, 0)
    except Exception as e:
        print(f"  ⚠️ AKShare 涨跌: {e}")

    # ── 北向资金（多种尝试）──
    try:
        # 方法1a: stock_hsgt_north_net_flow_in_em (新版 akshare)
        df_north = None
        if hasattr(ak, "stock_hsgt_north_net_flow_in_em"):
            df_north = ak.stock_hsgt_north_net_flow_in_em(symbol="北向")
        if (df_north is None or df_north.empty) and hasattr(ak, "stock_em_hsgt_north_net_flow_in"):
            # 方法1b: stock_em_hsgt_north_net_flow_in (旧版/备用，indicator="北上" 单位多为亿)
            df_north = ak.stock_em_hsgt_north_net_flow_in(indicator="北上")
        if df_north is not None and not df_north.empty:
            latest = df_north.iloc[-1]
            for col_name in ["当日净流入", "净流入", "value", "北上", df_north.columns[-1]]:
                if col_name in df_north.columns:
                    flow = pd.to_numeric(latest.get(col_name, 0), errors="coerce")
                    if pd.notna(flow) and flow != 0:
                        if abs(flow) > 1e8:
                            data["north_flow"] = round(float(flow) / 1e8, 2)
                        elif abs(flow) > 1e4:
                            data["north_flow"] = round(float(flow) / 1e4, 2)
                        else:
                            data["north_flow"] = round(float(flow), 2)
                        break
    except Exception as e1:
        print(f"  ⚠️ AKShare 北向(方法1): {e1}")

    if not data.get("north_flow") and hasattr(ak, "stock_hsgt_fund_flow_summary_em"):
        try:
            # 方法2: stock_hsgt_fund_flow_summary_em — 取最新一行当日净流入
            df_flow = ak.stock_hsgt_fund_flow_summary_em()
            if df_flow is not None and not df_flow.empty:
                latest = df_flow.iloc[-1]
                total_flow = 0
                for col in df_flow.columns:
                    if "净" in str(col) and "流入" in str(col):
                        v = pd.to_numeric(latest.get(col, 0), errors="coerce")
                        if pd.notna(v):
                            total_flow += float(v)
                if abs(total_flow) > 1e8:
                    data["north_flow"] = round(total_flow / 1e8, 2)
                elif abs(total_flow) > 1e4:
                    data["north_flow"] = round(total_flow / 1e4, 2)
                elif abs(total_flow) > 0:
                    data["north_flow"] = round(total_flow, 2)
        except Exception as e2:
            print(f"  ⚠️ AKShare 北向(方法2): {e2}")

    if not data.get("north_flow"):
        try:
            # 方法3: 东方财富实时接口（直接 HTTP）
            import requests
            url = ("https://push2.eastmoney.com/api/qt/kamt.rtmin/get"
                   "?fields1=f1,f2,f3,f4&fields2=f51,f52,f53,f54,f55,f56")
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0",
                             "Referer": "https://quote.eastmoney.com/"},
                             timeout=5)
            j = r.json()
            d = j.get("data", {})
            if d:
                hgt = d.get("f2", 0) or 0
                sgt = d.get("f4", 0) or 0
                total_north = (hgt + sgt) / 10000  # 万元→亿元
                if abs(total_north) > 0.01:
                    data["north_flow"] = round(total_north, 2)
        except Exception as e3:
            print(f"  ⚠️ 东方财富 北向: {e3}")

    if not data.get("north_flow"):
        data["north_flow"] = 0
        print(f"  ⚠️ 北向资金: 所有接口均失败, 默认0")

    return data if data.get("up_count") else None


# ═══════════════════════════════════════════════════════════════
#  数据源 2: 东方财富 HTTP API（无需安装，直接 requests）
# ═══════════════════════════════════════════════════════════════

def _fetch_eastmoney() -> Optional[dict]:
    """
    东方财富免费 API

    需要: requests（Python 标准环境一般都有）
    """
    try:
        import requests
    except ImportError:
        return None

    data = {}
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://quote.eastmoney.com/",
    }

    # ── 上证指数 ──
    try:
        url = ("https://push2.eastmoney.com/api/qt/stock/get"
               "?secid=1.000001&fields=f43,f44,f45,f46,f47,f48,f170")
        r = requests.get(url, headers=headers, timeout=5)
        j = r.json()
        d = j.get("data", {})
        if d:
            # f170=涨跌幅(%)  f47=成交量  f48=成交额
            data["index_pct"] = d.get("f170", 0) / 100  # 东方财富返回的是放大100倍
            amount = d.get("f48", 0)
            if amount:
                data["amount_billion"] = round(amount / 1e8, 0)
    except Exception as e:
        print(f"  ⚠️ 东方财富指数: {e}")

    # ── 涨跌统计 ──
    try:
        url = ("https://push2ex.eastmoney.com/getTopicZDFenBu"
               "?ut=7eea3edcaed734bea9cb&dession=&mession="
               "&fields=f12,f14,f3")
        r = requests.get(url, headers=headers, timeout=5)
        j = r.json()
        d = j.get("data", {})
        if d:
            # fenbu 是涨跌分布
            up = d.get("up", 0)
            down = d.get("down", 0)
            data["up_count"] = up
            data["down_count"] = down
    except Exception:
        pass

    # 如果上面的涨跌统计不好使，用另一个接口
    if "up_count" not in data:
        try:
            url = ("https://datacenter-web.eastmoney.com/api/data/v1/get"
                   "?reportName=RPT_MARKET_UPDOWN_ANALYSIS"
                   "&columns=ALL&pageSize=1&sortColumns=TRADE_DATE"
                   "&sortTypes=-1&pageNumber=1")
            r = requests.get(url, headers=headers, timeout=5)
            j = r.json()
            items = j.get("result", {}).get("data", [])
            if items:
                item = items[0]
                data["up_count"] = int(item.get("UP_COUNT", 0))
                data["down_count"] = int(item.get("DOWN_COUNT", 0))
                data["limit_up"] = int(item.get("LIMIT_UP_COUNT", 0))
                data["limit_down"] = int(item.get("LIMIT_DOWN_COUNT", 0))
        except Exception:
            pass

    # ── 涨跌停 ──
    if "limit_up" not in data:
        try:
            url = ("https://push2ex.eastmoney.com/getTopicLBSJ"
                   "?ut=7eea3edcaed734bea9cb&fields=f12,f14,f3")
            r = requests.get(url, headers=headers, timeout=5)
            j = r.json()
            d = j.get("data", {})
            if d:
                data["limit_up"] = d.get("zt", 0)
                data["limit_down"] = d.get("dt", 0)
        except Exception:
            pass

    # ── 北向资金 ──
    try:
        url = ("https://push2.eastmoney.com/api/qt/kamt.rtmin/get"
               "?fields1=f1,f2,f3,f4&fields2=f51,f52,f53,f54,f55,f56")
        r = requests.get(url, headers=headers, timeout=5)
        j = r.json()
        d = j.get("data", {})
        if d:
            # f2=沪股通净流入  f4=深股通净流入（万元）
            hgt = d.get("f2", 0) or 0
            sgt = d.get("f4", 0) or 0
            total_north = (hgt + sgt) / 10000  # 万元→亿元
            data["north_flow"] = round(total_north, 2)
    except Exception:
        data.setdefault("north_flow", 0)

    return data if data.get("index_pct") is not None else None


# ═══════════════════════════════════════════════════════════════
#  数据源 3: Tushare 实时
# ═══════════════════════════════════════════════════════════════

def _fetch_tushare_realtime() -> Optional[dict]:
    """Tushare 实时行情（需要较高积分）"""
    try:
        import tushare as ts
        # 旧版接口，不需要积分
        df = ts.get_index()
        if df is not None and not df.empty:
            sh_row = df[df["code"] == "000001"]
            if not sh_row.empty:
                row = sh_row.iloc[0]
                return {
                    "index_pct": float(row.get("changepercent", 0)),
                    "amount_billion": round(float(row.get("amount", 0)) / 1e8, 0),
                }
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════

def get_intraday(source: str = "auto", verbose: bool = True) -> dict:
    """
    自动获取盘中实时数据

    Parameters
    ----------
    source : "auto" | "akshare" | "eastmoney" | "tushare"
    verbose : 打印获取过程

    Returns
    -------
    {
        "index_pct": 1.01,
        "up_count": 4186,
        "down_count": 1215,
        "limit_up": 61,
        "limit_down": 16,
        "amount_billion": 10284,
        "north_flow": 75.89,
        "_source": "akshare",
        "_time": "10:32:15",
    }
    """
    if verbose:
        print("获取盘中实时数据...")

    fetchers = {
        "akshare": ("AKShare", _fetch_akshare),
        "eastmoney": ("东方财富", _fetch_eastmoney),
        "tushare": ("Tushare", _fetch_tushare_realtime),
    }

    if source != "auto":
        name, func = fetchers.get(source, (None, None))
        if func:
            if verbose:
                print(f"  尝试 {name}...")
            data = func()
            if data:
                data["_source"] = source
                data["_time"] = datetime.now().strftime("%H:%M:%S")
                if verbose:
                    _print_data(data)
                return data
        print(f"❌ {source} 获取失败")
        return {}

    # auto: 按优先级尝试
    for key in ["akshare", "eastmoney", "tushare"]:
        name, func = fetchers[key]
        if verbose:
            print(f"  尝试 {name}...", end=" ")
        try:
            data = func()
            if data and data.get("up_count"):
                data["_source"] = key
                data["_time"] = datetime.now().strftime("%H:%M:%S")
                if verbose:
                    print("✅")
                    _print_data(data)
                return data
            else:
                if verbose:
                    print("❌ 数据不全")
        except Exception as e:
            if verbose:
                print(f"❌ {e}")

    print("⚠️ 所有数据源均失败")
    return {}


def _print_data(data: dict):
    """打印获取到的数据"""
    print(f"\n  数据源: {data.get('_source', '?')}  时间: {data.get('_time', '?')}")
    print(f"  ─────────────────────────────")
    fields = [
        ("index_pct", "上证涨幅", "%"),
        ("up_count", "涨家数", ""),
        ("down_count", "跌家数", ""),
        ("limit_up", "涨停", ""),
        ("limit_down", "跌停", ""),
        ("amount_billion", "成交额", "亿"),
        ("north_flow", "北向净流入", "亿"),
    ]
    for key, label, unit in fields:
        val = data.get(key)
        if val is not None:
            if isinstance(val, float):
                print(f"  {label:8s}: {val:>10.2f} {unit}")
            else:
                print(f"  {label:8s}: {val:>10} {unit}")
        else:
            print(f"  {label:8s}: {'缺失':>10}")


# ═══════════════════════════════════════════════════════════════
#  一键评分
# ═══════════════════════════════════════════════════════════════

def quick_score(source: str = "auto"):
    """
    一键获取盘中数据 + 评分

    等价于：
      data = get_intraday()
      env.evaluate(intraday=data, diagnose=True)
    """
    today = datetime.now().strftime("%Y%m%d")

    # 获取实时数据
    intraday = get_intraday(source=source)
    if not intraday:
        print("❌ 无法获取盘中数据，无法评分")
        return None

    # 初始化评估器
    fetcher = None
    try:
        from data_fetcher import DataFetcher
        fetcher = DataFetcher()
    except Exception:
        pass

    from market_environment import MarketEnvironment
    env = MarketEnvironment(fetcher)

    # 评分
    result = env.evaluate(
        date=today,
        intraday=intraday,
        diagnose=True,
    )

    print("\n" + env.format_report(result))
    return result


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="盘中实时数据获取 + 环境评分")
    parser.add_argument("--source", choices=["auto", "akshare", "eastmoney", "tushare"],
                        default="auto", help="数据源")
    parser.add_argument("--data-only", action="store_true", help="只获取数据不评分")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args()

    if args.data_only:
        data = get_intraday(source=args.source)
        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        quick_score(source=args.source)


if __name__ == "__main__":
    main()
