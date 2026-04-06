# F Strategy

`F_三因子+趋势过滤` 是当前这套仓库里，在中证1000长期回测上表现最稳定的一条中期轮动策略。

这个目录的作用是把 `F` 从通用多策略脚本里单独拎出来，方便你：
- 单独运行
- 单独实验
- 单独维护说明文档

## 文件

- 参数来源：
  [/Users/wuhan/project/stock_agent/new/scripts/backtest_strategies.py](/Users/wuhan/project/stock_agent/new/scripts/backtest_strategies.py)
- 独立运行入口：
  [/Users/wuhan/project/stock_agent/new/strategies/f_strategy/run_backtest.py](/Users/wuhan/project/stock_agent/new/strategies/f_strategy/run_backtest.py)
- 当前配置快照：
  [/Users/wuhan/project/stock_agent/new/strategies/f_strategy/strategy_f_config.json](/Users/wuhan/project/stock_agent/new/strategies/f_strategy/strategy_f_config.json)

## 策略定位

`F` 不是短线打板策略，也不是日内策略。
它是一条：

- 中期动量选股
- 低波动筛噪
- 小市值增强
- 指数趋势过滤

的规则化轮动策略。

核心目标是：
- 在中证1000里抓中期趋势股
- 用低波和趋势过滤降低无效追高
- 用统一调仓纪律控制组合

## 核心逻辑

### 1. 三因子评分

`F` 的因子框架是：

- 主因子：`60` 日动量
- 辅因子：`60` 日低波动
- 辅因子：小市值偏好

策略不是直接把原始值硬加总，而是：
- 先对每个因子分别做排名
- 再按权重合成综合排名

当前默认权重：

- 动量：`1.0`
- 低波：`0.25`
- 小市值：`0.2`

### 2. 股票过滤

进入评分前，先过滤掉：

- ST
- 上市未满 `250` 天
- 股价低于 `3`
- 20 日均成交额低于 `1` 亿
- 当天接近涨停的股票

### 3. 趋势过滤

趋势过滤默认使用 **上证指数 `000001.SH`** 的 `MA200`，不是中证1000自身指数：

- 指数高于 `MA200`：允许满仓
- 指数低于 `MA200` 但高于 `0.95 * MA200`：最多半仓
- 指数低于 `0.95 * MA200`：空仓

采用上证指数的原因是：
- 趋势信号更平滑，假切换更少
- 对中证1000这种高波动选股池，更适合作为风险锚

### 4. 调仓规则

- 调仓频率：每 `20` 个交易日
- 目标持仓：前 `15` 名
- 缓冲带：`1.5`
- 单票上限：`8%`

含义是：
- 排名前 `15` 的股票是目标持仓
- 已持仓只要没跌出缓冲带，就继续拿
- 跌出缓冲带才卖

### 5. 风控规则

- 个股固定止损：`-15%`
- 不使用 `HALT` 清仓
- 交易成本包含：
  - 佣金 `0.03%`
  - 印花税 `0.1%`
  - 滑点 `0.2%`

## 当前最佳增强版

目前验证下来，当前最优的增强不是更复杂的牛熊切换，也不再是单纯的
`ma20_angle_deg >= 0`，而是：

- **买入时增加 `strength_transition_coef >= -0.1`**

定义：

```python
strength_transition_coef =
    0.7 * tanh(ma20_angle_deg_t / 10)
  + 0.3 * tanh((ma20_angle_deg_t - ma20_angle_deg_t_1) / 5)
```

含义：
- 既看今天 `MA20` 角度本身强不强
- 也看它是在继续走强，还是开始转弱
- `>= -0.1` 代表允许“弱转强早期”进入，但不过度放宽到明显下行阶段

这条规则只改变**买入门槛**，不改变原始 `F` 的调仓框架。

## 关键参数

当前配置快照见：
[/Users/wuhan/project/stock_agent/new/strategies/f_strategy/strategy_f_config.json](/Users/wuhan/project/stock_agent/new/strategies/f_strategy/strategy_f_config.json)

核心参数：

- `momentum_days = [60]`
- `momentum_weights = [1.0]`
- `use_volatility_factor = true`
- `volatility_days = 60`
- `volatility_weight = 0.25`
- `use_size_factor = true`
- `size_weight = 0.2`
- `top_n = 15`
- `hold_buffer_ratio = 1.5`
- `max_single_weight = 0.08`
- `rebalance_interval = 20`
- `stop_loss_pct = -0.15`
- `use_trend_filter = true`
- `trend_filter_index_code = "000001.SH"`
- `trend_ma_days = 200`
- `trend_reduce_pct = 0.5`

推荐买入增强条件：

- `strength_transition_coef >= -0.1`

历史对照门槛：

- `ma20_angle_deg >= 0`

## 运行

默认跑中证1000 `5y`：

```bash
cd /Users/wuhan/project/stock_agent/new
python3 strategies/f_strategy/run_backtest.py
```

跑当前主版本：

```bash
cd /Users/wuhan/project/stock_agent/new
python3 strategies/f_strategy/run_backtest.py
```

跑旧版 `MA20` 角度门槛：

```bash
cd /Users/wuhan/project/stock_agent/new
python3 strategies/f_strategy/run_backtest.py --min-transition-coef none --min-ma20-angle 0
```

每日执行信号：

```bash
cd /Users/wuhan/project/stock_agent/new
python3 strategies/f_strategy/run_daily_signal.py --init
python3 strategies/f_strategy/run_daily_signal.py --test-notify
python3 strategies/f_strategy/run_daily_signal.py --status
python3 strategies/f_strategy/run_daily_signal.py
python3 strategies/f_strategy/run_daily_signal.py --date 20260327
python3 strategies/f_strategy/run_daily_signal.py --confirm 20260327
```

信号文件会写到：
- [/Users/wuhan/project/stock_agent/new/data/signals](/Users/wuhan/project/stock_agent/new/data/signals)

F 独立持仓文件：
- [/Users/wuhan/project/stock_agent/new/data/f_strategy_portfolio_state.json](/Users/wuhan/project/stock_agent/new/data/f_strategy_portfolio_state.json)

执行确认仍复用：
- [/Users/wuhan/project/stock_agent/new/scripts/confirm_trades.py](/Users/wuhan/project/stock_agent/new/scripts/confirm_trades.py)

说明：
- `run_daily_signal.py` 现在会输出 `f_signal_YYYYMMDD.json` 和 `latest_f_signal.json`
- 若配置了企业微信 webhook，默认会推送“手动执行提醒”，直接给出买卖清单和确认命令
- 若配置了 `TUSHARE_TOKEN`，默认会先把本地 `5y` 样本增量补到最新可用交易日，再生成信号
- `--confirm` 会自动读取 F 信号并回写到 F 独立持仓文件
- `--test-notify` 可先测试企业微信是否能收到消息
- 默认会拒绝使用超过 `3` 天未更新的样本做“默认最新日”信号；离线复盘可显式加 `--date`、`--allow-stale-data`，或 `--disable-live-update`

### 企业微信半自动实盘

推荐流程：

- 每个交易日 `15:10` 跑一次 `run_daily_signal.py`
- 系统推送企业微信手动执行提醒
- 你在券商 APP 手动下单
- 成交后执行 `--confirm` 回写持仓

配置方式：

- Tushare：`TUSHARE_TOKEN=你的token`
- 环境变量：`WECOM_WEBHOOK=https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx`
- 或在 [notify_config.json](/Users/wuhan/project/stock_agent/new/config/notify_config.json) 里设置：

```json
{
  "wecom_enabled": true,
  "wecom_webhook": "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx"
}
```

Linux / macOS 定时任务示例：

```bash
10 15 * * 1-5 cd /Users/wuhan/project/stock_agent/new && /usr/bin/python3 strategies/f_strategy/run_daily_signal.py >> logs/f_strategy_daily_signal.log 2>&1
```

如果只想离线复盘，不访问在线接口：

```bash
python3 strategies/f_strategy/run_daily_signal.py --date 20260327 --disable-live-update
```

## 数据依赖

当前使用的数据文件：

- [/Users/wuhan/project/stock_agent/new/data_exports/tushare_20210329_20260327_csi1000_5y/csi1000_market_bundle_5y.csv](/Users/wuhan/project/stock_agent/new/data_exports/tushare_20210329_20260327_csi1000_5y/csi1000_market_bundle_5y.csv)

这份总表里，`F` 会实际使用的核心字段包括：

- `close`
- `amount`
- `pct_chg`
- `adj_factor`
- `circ_mv`
- `name`
- `list_date`
- `ma20_angle_deg`

## 当前回测结论

### 原始 F

结果文件：
[/Users/wuhan/project/stock_agent/new/backtest/csi1000_5y_backtest_strategy_F.json](/Users/wuhan/project/stock_agent/new/backtest/csi1000_5y_backtest_strategy_F.json)

- 总收益：`+114.45%`
- 年化：`+22.15%`
- 夏普：`0.84`
- 最大回撤：`-28.80%`

### F + `ma20_angle_deg >= 0`

结果文件：
[/Users/wuhan/project/stock_agent/new/backtest/csi1000_5y_strategy_F_ma20_angle_fine.json](/Users/wuhan/project/stock_agent/new/backtest/csi1000_5y_strategy_F_ma20_angle_fine.json)

- 总收益：`+213.57%`
- 年化：`+34.94%`
- 夏普：`1.20`
- 最大回撤：`-23.80%`

### F + `strength_transition_coef >= -0.1`

结果文件：
- [/Users/wuhan/project/stock_agent/new/backtest/strategy_f_trend_index_benchmark_compare.json](/Users/wuhan/project/stock_agent/new/backtest/strategy_f_trend_index_benchmark_compare.json)

- 趋势过滤指数：`000001.SH` 上证指数
- 总收益：`+269.41%`
- 年化：`+40.87%`
- 夏普：`1.34`
- 最大回撤：`-22.90%`

### F + `ma20_angle_deg >= 15`

结果文件：
[/Users/wuhan/project/stock_agent/new/backtest/csi1000_5y_strategy_F_ma20_angle_buckets.json](/Users/wuhan/project/stock_agent/new/backtest/csi1000_5y_strategy_F_ma20_angle_buckets.json)

- 总收益：`+87.69%`
- 年化：`+17.95%`
- 夏普：`0.62`
- 最大回撤：`-26.13%`

结论：

- `ma20_angle_deg >= 15` 作为硬门槛太严，不值得并入
- `strength_transition_coef` 直接作为加分项会破坏原始排序结构
- 当前最优版本是：
  - **原始 F 框架 + `strength_transition_coef >= -0.1`**

## 一句话总结

`F` 是当前这套系统里最值得继续打磨的中证1000主策略骨架。
如果只保留一个增强方向，优先保留：

- **`F + strength_transition_coef >= -0.1`**
