# Stock Agent `new`

`new/` 是当前正在使用的项目根目录，主线是 **A 股动量轮动策略**，包含：

- 日常信号生成与微信提醒
- 历史回测与参数对比
- Opus 反思与参数进化
- SQLite 持久化与定时任务

旧主线龙头策略已经归档到 `../old/`，不要再把 `new/` 和 `old/` 混着看。

## 1. 当前主入口

日常信号主程序：
- [run_daily_signal.py](/Users/wuhan/project/stock_agent/new/signal/run_daily_signal.py)

主程序依赖的核心策略逻辑：
- [signal_generator.py](/Users/wuhan/project/stock_agent/new/signal/signal_generator.py)

正式动量模块骨架：
- [engine.py](/Users/wuhan/project/stock_agent/new/momentum/engine.py)
- [config.py](/Users/wuhan/project/stock_agent/new/momentum/config.py)
- [universe.py](/Users/wuhan/project/stock_agent/new/momentum/universe.py)
- [calculator.py](/Users/wuhan/project/stock_agent/new/momentum/calculator.py)
- [rebalancer.py](/Users/wuhan/project/stock_agent/new/momentum/rebalancer.py)
- [risk_control.py](/Users/wuhan/project/stock_agent/new/momentum/risk_control.py)

回测主引擎：
- [backtest_v2_vs_v3.py](/Users/wuhan/project/stock_agent/new/scripts/backtest_v2_vs_v3.py)

月度进化入口：
- [run_evolution_test.py](/Users/wuhan/project/stock_agent/new/scripts/run_evolution_test.py)

## 2. 目录说明

### 核心策略
- [signal](/Users/wuhan/project/stock_agent/new/signal)：当前实际运行的信号系统。包括信号生成、微信通知、宏观事件、快讯监听、进化触发。
- [momentum](/Users/wuhan/project/stock_agent/new/momentum)：动量轮动模块化实现，偏框架化。
- [evolution](/Users/wuhan/project/stock_agent/new/evolution)：Opus 反思、评估、注册、沙盒进化。

### 回测与研究
- [backtest](/Users/wuhan/project/stock_agent/new/backtest)：正式回测结果、进化结果、各类实验 JSON 输出。
- [scripts](/Users/wuhan/project/stock_agent/new/scripts)：一次性研究脚本、专题回测脚本、数据实验脚本。
- [strategies](/Users/wuhan/project/stock_agent/new/strategies)：单独整理出来的策略目录。包含 [f_strategy](/Users/wuhan/project/stock_agent/new/strategies/f_strategy) 和 [avwap_profile_strategy](/Users/wuhan/project/stock_agent/new/strategies/avwap_profile_strategy)。
- [tests](/Users/wuhan/project/stock_agent/new/tests)：自动化测试。

### 当前独立最优策略
- [f_strategy](/Users/wuhan/project/stock_agent/new/strategies/f_strategy)：当前独立研究里表现最好的中证1000中期轮动策略。
- 当前主版本是：`F + strength_transition_coef >= -0.1`
- 趋势过滤默认指数：`000001.SH` 上证指数
- 默认运行入口：
  [/Users/wuhan/project/stock_agent/new/strategies/f_strategy/run_backtest.py](/Users/wuhan/project/stock_agent/new/strategies/f_strategy/run_backtest.py)
- 默认结果文件：
  [/Users/wuhan/project/stock_agent/new/backtest/strategy_f_csi1000_5y_transition_coef_ge_neg_0_1.json](/Users/wuhan/project/stock_agent/new/backtest/strategy_f_csi1000_5y_transition_coef_ge_neg_0_1.json)

### AVWAP Profile 策略（实验中）
- [avwap_profile_strategy](/Users/wuhan/project/stock_agent/new/strategies/avwap_profile_strategy)：基于 Anchored VWAP + Volume Profile 的突破/回踩事件驱动策略。
- 最优参数：`bv=1.3, pv=0.9, daily_failure_exit, ma60`
- 近 250 日回测：年化 +13.27%，夏普 1.18，最大回撤 -7.97%
- 默认运行入口：
  [/Users/wuhan/project/stock_agent/new/strategies/avwap_profile_strategy/run_backtest.py](/Users/wuhan/project/stock_agent/new/strategies/avwap_profile_strategy/run_backtest.py)
- 实验入口：
  [/Users/wuhan/project/stock_agent/new/strategies/avwap_profile_strategy/run_recent_100d_experiments.py](/Users/wuhan/project/stock_agent/new/strategies/avwap_profile_strategy/run_recent_100d_experiments.py)

### 数据与配置
- [data](/Users/wuhan/project/stock_agent/new/data)：主程序默认读取的数据缓存、持仓状态、周度绩效、信号 JSON。
- [data_exports](/Users/wuhan/project/stock_agent/new/data_exports)：按日期导出的原始市场数据总表。
- [config](/Users/wuhan/project/stock_agent/new/config)：环境变量、通知配置、全局设置、策略拆分注册表。
- [parameter_versions](/Users/wuhan/project/stock_agent/new/parameter_versions)：参数版本文件。

### 基础设施
- [db](/Users/wuhan/project/stock_agent/new/db)：SQLite 数据库与仓储层。
- [scheduler](/Users/wuhan/project/stock_agent/new/scheduler)：定时任务。
- [data_pipeline](/Users/wuhan/project/stock_agent/new/data_pipeline)：Tushare client 和交易时钟等基础工具。
- [decision_engine](/Users/wuhan/project/stock_agent/new/decision_engine)：保留下来的最小兼容层。

## 3. 日常使用

安装依赖：

```bash
cd /Users/wuhan/project/stock_agent/new
python3 -m pip install -r requirements.txt
```

生成当日信号：

```bash
cd /Users/wuhan/project/stock_agent/new
python3 signal/run_daily_signal.py
```

指定日期回放：

```bash
cd /Users/wuhan/project/stock_agent/new
python3 signal/run_daily_signal.py --date 20260330
```

测试企业微信通知：

```bash
cd /Users/wuhan/project/stock_agent/new
python3 signal/run_daily_signal.py --test-notify
```

盘中监控：

```bash
cd /Users/wuhan/project/stock_agent/new
python3 signal/run_daily_signal.py --monitor
```

确认手工执行后的持仓：

```bash
cd /Users/wuhan/project/stock_agent/new
python3 signal/run_daily_signal.py --confirm YYYYMMDD
```

检查是否触发进化：

```bash
cd /Users/wuhan/project/stock_agent/new
python3 signal/run_daily_signal.py --check-evolution
```

运行一次 Opus 进化：

```bash
cd /Users/wuhan/project/stock_agent/new
python3 scripts/run_evolution_test.py
```

## 4. 当前自动化

收盘后自动信号：
- `crontab` 已配置工作日 `15:30` 运行 `signal/run_daily_signal.py`

盘中监控：
- `launchd` 已配置后台常驻监控
- 启动文件是 [com.wuhan.stock-agent-monitor.plist](/Users/wuhan/project/stock_agent/new/com.wuhan.stock-agent-monitor.plist)

查看日志：
- [logs](/Users/wuhan/project/stock_agent/new/logs)
- [monitor.stdout.log](/Users/wuhan/project/stock_agent/new/logs/monitor.stdout.log)
- [monitor.stderr.log](/Users/wuhan/project/stock_agent/new/logs/monitor.stderr.log)

## 5. 数据文件约定

当前最重要的缓存文件：

- [csi1000_market_bundle.csv](/Users/wuhan/project/stock_agent/new/data/csi1000_market_bundle.csv)
- [csi1000_market_bundle_100d.csv](/Users/wuhan/project/stock_agent/new/data/csi1000_market_bundle_100d.csv)
- [csi1000_market_bundle_300d.csv](/Users/wuhan/project/stock_agent/new/data/csi1000_market_bundle_300d.csv)
- [csi1000_market_bundle_300d_lhb.csv](/Users/wuhan/project/stock_agent/new/data/csi1000_market_bundle_300d_lhb.csv)
- [csi1000_market_bundle_700d.csv](/Users/wuhan/project/stock_agent/new/data/csi1000_market_bundle_700d.csv)

更完整的长期数据导出：
- [csi1000_market_bundle_5y.csv](/Users/wuhan/project/stock_agent/new/data_exports/tushare_20210329_20260327_csi1000_5y/csi1000_market_bundle_5y.csv)

这类总表通常包含：
- `daily`
- `index_daily`
- `stock_basic`
- `trade_cal`
- `daily_basic`
- `adj_factor`
- `top_list`

## 6. 当前重要产物

信号输出：
- [signals](/Users/wuhan/project/stock_agent/new/data/signals)

持仓状态：
- [portfolio_state.json](/Users/wuhan/project/stock_agent/new/data/portfolio_state.json)

周度绩效：
- [weekly_performance.json](/Users/wuhan/project/stock_agent/new/data/weekly_performance.json)

F 主策略回测：
- [strategy_f_csi1000_5y_transition_coef_ge_neg_0_1.json](/Users/wuhan/project/stock_agent/new/backtest/strategy_f_csi1000_5y_transition_coef_ge_neg_0_1.json)
- [strategy_f_trend_index_benchmark_compare.json](/Users/wuhan/project/stock_agent/new/backtest/strategy_f_trend_index_benchmark_compare.json)
- [strategy_f_transition_gate_threshold_sweep_csi1000_5y.json](/Users/wuhan/project/stock_agent/new/backtest/strategy_f_transition_gate_threshold_sweep_csi1000_5y.json)
- [strategy_f_official_regime_analysis_csi1000_5y.json](/Users/wuhan/project/stock_agent/new/backtest/strategy_f_official_regime_analysis_csi1000_5y.json)

## 7. 研究结论状态

目前已经确认的方向：
- 独立策略线里，当前最优版本是 `F + strength_transition_coef >= -0.1`。
- 这条 `F` 主版本在中证1000 `5y` 上：
  - 趋势过滤指数：`000001.SH` 上证指数
  - 总收益 `+269.41%`
  - 年化 `+40.87%`
  - 夏普 `1.34`
  - 最大回撤 `-22.90%`
- 当前 `new/` 已清理为以 `F` 为中心的研究目录，不再保留其他策略线的结果文件。

## 8. 配置与安全

敏感配置在：
- [config/.env](/Users/wuhan/project/stock_agent/new/config/.env)
- [notify_config.json](/Users/wuhan/project/stock_agent/new/config/notify_config.json)

注意：
- 不要提交真实 token、webhook、API key
- `notify_config.json` 里建议只保留非敏感默认值
- webhook、Tushare、OpenRouter 等敏感值应放在 `.env` 或系统环境变量里

## 9. 补充说明

- [readme.txt](/Users/wuhan/project/stock_agent/new/readme.txt) 是更早期的旧说明，内容已不对应当前 `new/` 的实际运行形态。
- [INDEX.md](/Users/wuhan/project/stock_agent/new/INDEX.md) 是抽屉级导航；本 README 是当前项目的正式入口说明。
