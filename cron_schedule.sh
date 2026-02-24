#!/bin/bash
# ============================================================
# A股Agent 定时任务配置
#
# 参考案例的调度表，适配A股交易时间:
#   美股: 8:00 overnight / 10:00盘前 / 周一复盘
#   A股: 8:30盘前 / 15:30盘后 / 周五复盘
#
# 安装: crontab -e 然后粘贴下方内容
# 查看: crontab -l
# ============================================================

AGENT_DIR="$HOME/project/astock_agent"
PYTHON="python3"
LOG_DIR="$AGENT_DIR/logs"

# 确保日志目录存在
mkdir -p "$LOG_DIR"

# ============================================================
# 调度表
# ============================================================

# ┌───────── 时间    ┌─── 任务             ┌─── 输出
# │                  │                      │
# ▼                  ▼                      ▼

# 1. 每日盘前交易计划 (工作日8:15)
#    → 自动生成今日操作清单: 止损/减仓/买入/观望
# 15 8 * * 1-5  cd $AGENT_DIR && $PYTHON trade_planner.py --export >> $LOG_DIR/plan.log 2>&1

# 2. 每日盘前分析 (工作日8:30)
#    → 情绪+宏观+板块概览，帮你决定今天仓位
# 30 8 * * 1-5  cd $AGENT_DIR && $PYTHON daily_agent.py --push >> $LOG_DIR/daily.log 2>&1

# 2. 每日盘后速报 (工作日15:45)
#    → 当日行情回顾+主题龙头扫描
# 45 15 * * 1-5  cd $AGENT_DIR && $PYTHON daily_agent.py >> $LOG_DIR/daily_pm.log 2>&1

# 3. 周度选股 (周五18:00)
#    → 板块轮动+CANSLIM+缩量整理联合选股
# 0 18 * * 5  cd $AGENT_DIR && $PYTHON weekly_agent.py >> $LOG_DIR/weekly.log 2>&1

# 4. 周度复盘 (周日20:00)
#    → 上周推荐绩效追踪+策略优化建议
# 0 20 * * 0  cd $AGENT_DIR && $PYTHON weekly_agent.py --review >> $LOG_DIR/review.log 2>&1

# 5. Skill准确性评估 (每月1号)
#    → 各Skill命中率统计+等级评定
# 0 10 1 * *  cd $AGENT_DIR && $PYTHON eval_layer.py >> $LOG_DIR/eval.log 2>&1

# ============================================================
# 财报季特别调度 (4月/8月/10月)
# ============================================================
# 财报季每日扫描价值股 (可手动开启)
# 0 19 * 4,8,10 1-5  cd $AGENT_DIR && $PYTHON skills/value_investor.py --code "关注列表" >> $LOG_DIR/value.log 2>&1

# ============================================================
# 一键安装
# ============================================================
# 执行: bash cron_schedule.sh install
# 卸载: bash cron_schedule.sh remove

if [ "$1" = "install" ]; then
    # 备份当前crontab
    crontab -l > /tmp/crontab_backup_$(date +%Y%m%d).txt 2>/dev/null

    # 写入新的crontab
    (crontab -l 2>/dev/null; cat << 'CRON'
# === A股Agent 定时任务 ===
30 8 * * 1-5  cd ~/project/astock_agent && python3 daily_workflow.py analyze >> logs/daily.log 2>&1
45 15 * * 1-5  cd ~/project/astock_agent && python3 daily_workflow.py update >> logs/update.log 2>&1
0 18 * * 5  cd ~/project/astock_agent && python3 weekly_agent.py >> logs/weekly.log 2>&1
0 20 * * 0  cd ~/project/astock_agent && python3 weekly_agent.py --review >> logs/review.log 2>&1
0 10 1 * *  cd ~/project/astock_agent && python3 eval_layer.py >> logs/eval.log 2>&1
# === End A股Agent ===
CRON
    ) | crontab -

    echo "✅ 定时任务已安装"
    echo "当前crontab:"
    crontab -l
    echo ""
    echo "备份保存在: /tmp/crontab_backup_$(date +%Y%m%d).txt"

elif [ "$1" = "remove" ]; then
    crontab -l 2>/dev/null | grep -v "astock_agent" | grep -v "A股Agent" | crontab -
    echo "✅ 定时任务已移除"

else
    echo "用法:"
    echo "  bash cron_schedule.sh install  # 安装定时任务"
    echo "  bash cron_schedule.sh remove   # 移除定时任务"
    echo ""
    echo "调度表:"
    echo "  每日 8:30  盘前分析+信号生成"
    echo "  每日 15:45 盘后更新持仓价格"
    echo "  每周五 18:00  周度选股"
    echo "  每周日 20:00  周度复盘"
    echo "  每月1日 10:00  Skill准确性评估"
fi
