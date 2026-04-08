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
- 新增因子：`MA20` 角度趋势增强

策略不是直接把原始值硬加总，而是：
- 先对每个因子分别做排名
- 再按权重合成综合排名

当前默认权重：

- 动量：`1.0`
- 低波：`0.25`
- 小市值：`0.2`
- 角度趋势：`0.15`

### 1.1 `MA20` 角度趋势因子

这个新因子不是简单看 `ma20_angle_deg >= 0`，而是量化：

- 最近一段时间 `MA20` 角度变化的**斜率**
- 最近一段时间 `MA20` 角度继续改善的**持续性**

当前默认口径：

- 窗口：最近 `10` 个交易日
- 斜率：对最近 `10` 日 `ma20_angle_deg` 做线性回归，取回归斜率
- 持续性：`50% * 角度为正的比例 + 50% * 日度角度继续上升的比例`
- 因子得分：`0.6 * tanh(slope / 1.5) + 0.4 * (2 * persistence - 1)`

含义：

- 斜率高，代表 `MA20` 角度在持续抬升
- 持续性高，代表这不是单日跳变，而是一个更平滑、更连续的改善过程
- 这个因子只参与**排序加分**，不替代原本的 `strength_transition_coef >= -0.1` 买入门槛

### 1.2 最近 `100` 个交易日 PIT 验证

这个新因子已经做过一轮短窗 `PIT` 对照验证，窗口是：

- `2025-10-29` 到 `2026-03-27`

结果文件：

- [/Users/wuhan/project/stock_agent/new/backtest/strategy_f_recent_100d_pit_angle_trend_compare.json](/Users/wuhan/project/stock_agent/new/backtest/strategy_f_recent_100d_pit_angle_trend_compare.json)

对照结果：

- 含角度趋势因子：
  - 总收益 `27.06%`
  - 年化 `82.84%`
  - 夏普 `2.48`
  - 最大回撤 `-15.77%`
  - 相对基准超额 `29.61%`
- 不含角度趋势因子：
  - 总收益 `1.36%`
  - 年化 `3.47%`
  - 夏普 `0.10`
  - 最大回撤 `-16.51%`
  - 相对基准超额 `3.92%`

这说明在最近 `100` 个交易日的 `PIT` 口径下，角度趋势因子有明显增益。
但这仍然只是短窗口验证，不能替代完整 `PIT 5y` 结论。

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
- `use_angle_trend_factor = true`
- `angle_trend_days = 10`
- `angle_trend_weight = 0.15`
- `angle_trend_slope_weight = 0.6`
- `angle_trend_persistence_weight = 0.4`
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

按更接近实盘的口径回测：

```bash
cd /Users/wuhan/project/stock_agent/new
python3 strategies/f_strategy/run_backtest.py --execution-mode next_open
```

按历史指数成分做 PIT 过滤：

```bash
cd /Users/wuhan/project/stock_agent/new
python3 strategies/f_strategy/run_backtest.py --use-pit-constituents
```

做 train/test 分割：

```bash
cd /Users/wuhan/project/stock_agent/new
python3 strategies/f_strategy/run_backtest.py --train-end 20241231 --test-start 20250102
```

做 walk-forward：

```bash
cd /Users/wuhan/project/stock_agent/new
python3 strategies/f_strategy/run_backtest.py --walk-forward
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
python3 strategies/f_strategy/run_daily_signal.py --force
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
- 同一交易日若信号文件已存在，默认跳过重复生成；加 `--force` 才会覆盖并重新提醒
- 若配置了企业微信 webhook，默认会推送“手动执行提醒”，直接给出买卖清单和确认命令
- 若配置了 `TUSHARE_TOKEN`，默认会先把本地 `5y` 样本增量补到最新可用交易日，再生成信号
- 出信号前会做运行时数据校验；失败会写入 `data/signals/error_log.json` 并拒绝发单
- `--confirm` 会自动读取 F 信号并回写到 F 独立持仓文件
- 同一 `signal_id` 只会被确认一次，重复执行 `--confirm` 不会重复记账
- `--test-notify` 可先测试企业微信是否能收到消息
- 默认会拒绝使用超过 `3` 天未更新的样本做“默认最新日”信号；离线复盘可显式加 `--date`、`--allow-stale-data`，或 `--disable-live-update`

回测补充说明：
- `run_backtest.py` 现已支持 `--execution-mode next_open`
- `run_backtest.py` 现已支持 `--train-end / --test-start / --walk-forward`
- `--use-pit-constituents` 会按历史 `index_weight` 快照过滤每个交易日的可选股票
- 但当前本地 `csi1000_5y` bundle 仍只有 1000 只样本股，未覆盖历史退样/退市股票，所以 PIT 目前只消除了“历史成分过滤缺失”，还没有完全消除幸存者偏差

PIT 历史样本扩容：

```bash
cd /Users/wuhan/project/stock_agent/new
python3 scripts/build_csi1000_5y_pit_bundle.py
python3 strategies/f_strategy/run_backtest.py --dataset csi1000_5y_pit --use-pit-constituents
```

说明：
- `build_csi1000_5y_pit_bundle.py` 会拉取 `000852.SH` 历史 `index_weight`，把 5 年内曾经属于 CSI1000 的股票并入新 bundle
- 输出目录默认是 [/Users/wuhan/project/stock_agent/new/data_exports/tushare_20210329_20260327_csi1000_5y_pit](/Users/wuhan/project/stock_agent/new/data_exports/tushare_20210329_20260327_csi1000_5y_pit)
- 新 dataset key 是 `csi1000_5y_pit`
- 即使换成 PIT bundle，回测时仍建议保留 `--use-pit-constituents`，因为 bundle 包含的是“历史上曾经入选过的全集”，而不是“每天只保留当日成分股”
- 这版 PIT bundle 优先保证 F 主策略可跑：核心日线、`daily_basic`、`adj_factor`、趋势指数和技术特征齐全；行业/主题/HFSF 等扩展字段会尽量从旧 bundle 继承，历史新增股票没有旧映射时会留空

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
