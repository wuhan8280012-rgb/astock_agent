#!/usr/bin/env python3
"""手动触发 DataHub 增量更新

用法:
    python cache_refresh.py                  # 更新今日数据
    python cache_refresh.py 20260302         # 更新指定日期
"""
import sys
import os

# 将 astock_agent 目录加入路径，确保 data_fetcher 可被 DataHub 找到
_agent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _agent_dir not in sys.path:
    sys.path.insert(0, _agent_dir)

from openclaw_os.data.datahub import DataHub, DataMode


def main():
    target_date = sys.argv[1] if len(sys.argv) > 1 else None

    # cache_dir 相对于 astock_agent 目录
    cache_dir = os.path.join(_agent_dir, "data", "parquet")
    hub = DataHub(cache_dir=cache_dir, mode=DataMode.LIVE)

    print(f"开始增量更新: target_date={target_date or '今日'}")
    stats = hub.incremental_update(target_date)
    print(f"完成: {stats}")

    if stats.get("errors"):
        print(f"错误详情:")
        for err in stats["errors"]:
            print(f"  - {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
