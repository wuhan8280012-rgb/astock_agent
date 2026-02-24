#!/usr/bin/env python3
"""
输出今天环境分的全部结果

用法：
  python3 env_today.py              # 今天
  python3 env_today.py 20260224     # 指定日期 YYYYMMDD
"""
import json
import sys
from datetime import datetime


def main():
    date = (sys.argv[1] if len(sys.argv) > 1 else None) or datetime.now().strftime("%Y%m%d")

    from data_fetcher import DataFetcher
    from market_environment import MarketEnvironment

    fetcher = DataFetcher()
    me = MarketEnvironment(fetcher)
    result = me.evaluate(date=date)

    # 1. 可读报告
    print("=" * 50)
    print(f"  市场环境评分（完整结果） {result['date']}")
    print("=" * 50)
    print(me.format_report(result))
    print()

    # 2. 各维度 details（若有 error/note 也打出）
    print("--- 各维度详情 ---")
    def _to_jsonable(obj):
        import numpy as np
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_jsonable(x) for x in obj]
        return obj

    for dim, detail in result.get("details", {}).items():
        if isinstance(detail, dict):
            print(f"  {dim}: {json.dumps(_to_jsonable(detail), ensure_ascii=False, indent=4)}")
        else:
            print(f"  {dim}: {detail}")
    print()

    # 3. 完整 JSON
    print("--- 完整 JSON ---")
    clean = _to_jsonable(result)
    print(json.dumps(clean, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
