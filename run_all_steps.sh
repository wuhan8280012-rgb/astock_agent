#!/bin/bash
# 三步执行并保存控制台输出到 outputs/run_console_$(date +%Y%m%d_%H%M%S).txt
set -e
cd "$(dirname "$0")"
mkdir -p outputs
LOG="outputs/run_console_$(date +%Y%m%d_%H%M%S).txt"
exec > >(tee -a "$LOG") 2>&1
echo "======== 输出将同时写入: $LOG ========"

echo ""
echo "======== 第一步：配置 token + 测试数据连通 ========"
python3 data_fetcher.py

echo ""
echo "======== 第二步：小样本快速验证（20 只股票） ========"
python3 backtest_runner.py --small --source tushare

echo ""
echo "======== 第三步：正式回测（沪深300，敏感性+CSV导出） ========"
python3 backtest_runner.py --source tushare --sensitivity --export

echo ""
echo "======== 全部完成。控制台输出已保存到: $LOG ========"
