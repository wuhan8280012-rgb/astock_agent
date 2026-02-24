#!/bin/bash
# A股每日前瞻 Agent - 快速安装脚本

set -e

echo "================================================"
echo "  A股每日前瞻 Agent - 安装向导"
echo "================================================"
echo ""

# 1. 安装依赖
echo "[1/4] 安装 Python 依赖..."
pip install tushare pandas numpy tabulate --break-system-packages -q 2>/dev/null || \
pip install tushare pandas numpy tabulate -q
echo "  ✅ 依赖安装完成"

# 2. 创建配置文件
if [ ! -f "config.py" ]; then
    echo ""
    echo "[2/4] 创建配置文件..."
    cp config_template.py config.py
    echo "  ⚠️  请编辑 config.py 填入你的 Tushare Token"
    echo "  注册地址: https://tushare.pro/register"
else
    echo "[2/4] 配置文件已存在"
fi

# 3. 创建目录
echo ""
echo "[3/4] 创建工作目录..."
mkdir -p cache knowledge reports
echo "  ✅ 目录创建完成"

# 4. 创建持仓文件
if [ ! -f "knowledge/positions.json" ]; then
    cp knowledge/positions_template.json knowledge/positions.json
    echo ""
    echo "[4/4] 已创建持仓模板"
    echo "  💡 请编辑 knowledge/positions.json 填入你的实际持仓"
else
    echo "[4/4] 持仓文件已存在"
fi

echo ""
echo "================================================"
echo "  ✅ 安装完成！"
echo ""
echo "  使用方法:"
echo "  1. 编辑 config.py 填入 Tushare Token"
echo "  2. 编辑 knowledge/positions.json 填入持仓"
echo "  3. 运行: python daily_agent.py"
echo ""
echo "  单独运行:"
echo "    python daily_agent.py --sentiment  # 情绪分析"
echo "    python daily_agent.py --sector     # 板块轮动"
echo "    python daily_agent.py --macro      # 宏观监控"
echo "    python daily_agent.py --risk       # 风控检查"
echo ""
echo "  设置定时任务 (每个交易日早上6:30):"
echo "    crontab -e"
echo "    30 6 * * 1-5 cd $(pwd) && python daily_agent.py --push"
echo "================================================"
