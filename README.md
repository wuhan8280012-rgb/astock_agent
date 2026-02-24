# A股每日前瞻 Agent 系统

## 架构：三层设计

```
┌─────────────────────────────────────────┐
│           CRON 自动化层                  │
│   daily_agent.py (每日调度入口)          │
├─────────────────────────────────────────┤
│           Skills 决策层                  │
│  sentiment │ sector │ macro │ risk_ctrl │
├─────────────────────────────────────────┤
│           知识库 & 数据层                │
│   data_fetcher.py + knowledge_base.py   │
└─────────────────────────────────────────┘
```

## 快速开始

```bash
# 1. 配置 API Key
cp config_template.py config.py
# 编辑 config.py 填入你的 Tushare token

# 2. 安装依赖
pip install tushare pandas numpy tabulate

# 3. 运行每日前瞻
python daily_agent.py

# 4. 设置定时任务 (crontab -e)
# 30 6 * * 1-5 cd /path/to/astock_agent && python daily_agent.py
```

### 全系统入口（final_system）

```bash
# 检查所有模块
python3 final_system.py verify

# 启动全系统（进入交互 shell，可调用 system.workflow / system.conversation）
python3 final_system.py start

# 信号扫描（阶梯突破 + 龙头突破 + 5 类信号）
python3 daily_workflow.py signal
```

接入说明：`router.py` 已改为使用 `from final_system import register_all_skills`，11 个 Skill 由 final_system 统一注册。

### v6.0 市场环境回测（11–12 月验证）

```bash
# 真实数据回测（需 Tushare 有对应区间数据）
python3 backtest_v6.py --start 20251101 --end 20251231

# 仅看每日环境评分
python3 backtest_v6.py --env-only --start 20251101 --end 20251130

# 阈值敏感性分析
python3 backtest_v6.py --sensitivity

# 导出交易明细到 CSV
python3 backtest_v6.py --export trades.csv

# 调参：环境阈值 55、持有 10 天、止损 -7%
python3 backtest_v6.py --threshold 55 --hold 10 --stop -7

# 无真实数据时用模拟数据跑通流程
python3 backtest_v6.py --simulate
```

回测做 5 件事：逐日环境评分 → 逐日阶梯突破信号扫描 → 跟踪盈亏（止损/止盈/持有 N 天）→ v4.1 全做 vs v6.0 环境过滤对比 → 敏感性分析找最优阈值。结果存档在 `outputs/`。

## 运维与复盘

### 持仓配置（风控才能生效）

```bash
cp knowledge/positions_template.json knowledge/positions.json
nano knowledge/positions.json   # 按实盘填写 ts_code/name/position_pct/cost_price/current_price/sector/buy_date
```

### 日报定时任务

```bash
crontab -e
# 加一行（工作日 6:30 跑日报，日志写入 logs/daily.log）：
30 6 * * 1-5 cd ~/project/astock_agent && python3 daily_agent.py >> logs/daily.log 2>&1
```

### 积累数据后（建议 1～2 周）

- **Skill 命中率**：`python3 eval_layer.py`（可加 `--skill sentiment` 等单测）
- **选股复盘**：`python3 weekly_agent.py --review`

## 模块说明

| 模块 | 功能 | 对应架构层 |
|------|------|-----------|
| config.py | 全局配置 | - |
| data_fetcher.py | Tushare数据拉取+缓存 | 知识库层 |
| knowledge_base.py | 历史事件/经验库 | 知识库层 |
| skills/sentiment.py | A股情绪监控 | Skills层 |
| skills/sector_rotation.py | 板块轮动排名 | Skills层 |
| skills/macro_monitor.py | 宏观流动性监控 | Skills层 |
| skills/risk_control.py | T+1风控模型 | Skills层 |
| report_generator.py | AI报告生成 | CRON层 |
| daily_agent.py | 每日调度主入口 | CRON层 |
