# AVWAP Profile Strategy

`AVWAP_Profile` 是基于锚定成交量加权均价（Anchored VWAP）和成交量分布（Volume Profile）的突破/回踩策略，适用于中证1000周度轮动。

## 文件

- 策略脚本：`run_backtest.py`
- 实验脚本：`run_recent_100d_experiments.py`
- 回测结果：
  - `backtest/strategy_avwap_profile_csi1000_5y_pit_weekly_100d.json`（近100日基线）
  - `backtest/strategy_avwap_profile_csi1000_5y_pit_weekly_100d_experiments.json`（近100日参数实验）
  - `backtest/strategy_avwap_profile_csi1000_5y_pit_weekly_250d_bv1.3_pv0.9_ma60_dailyexit.json`（近250日最优参数）

## 策略定位

`AVWAP_Profile` 不是纯因子排名策略，而是基于价量结构的事件驱动策略。
核心逻辑：在周线级别识别"横盘蓄力→放量突破→缩量回踩"的经典形态，用 AVWAP 和 Volume Profile 精确定义关键价位。

## 核心逻辑

### 1. 横盘区间识别（Balance Context）

对每只股票的周线数据：
- 用最近 8 个周期（`balance_periods`）构建横盘区间
- 计算 **Anchored VWAP**：以区间成交额为权重的加权均价
- 计算 **Volume Profile**：24 档价格分布，提取 POC（成交密集价）、VAL/VAH（价值区间上下沿）
- 筛选条件：区间振幅 ≤ 32%、价值区间宽度 ≤ 18%、POC 偏离 AVWAP ≤ 5%

### 2. 突破确认（Breakout Validation）

突破条件：
- 收盘价 > 突破水平（区间高点、VAH、POC、AVWAP 的最大值）× 1.01
- 成交量 > 横盘期均量 × `breakout_volume_mult`（默认 1.5）

### 3. 三种入场类型

| 类型 | 条件 | 权重 |
|------|------|------|
| **pullback**（回踩） | 收盘价在锚定位上方且低于突破价108%，缩量 | 1.0 |
| **breakout**（突破） | 当期突破 + 资金情绪 > 0 | 0.8 |
| **hold**（持有） | 收盘价在突破位104%以上，情绪中性 | 0.5 |

### 4. 六因子复合排名

- 入场类型权重 25%
- 均线角度质量 20%
- 成交量质量 20%
- 情绪质量 15%
- 接近度质量 10%
- 形态得分 10%

### 5. 风控

- 周度调仓
- 持仓 6 只，缓冲带 1.2×
- 单只上限 18%
- 止损 -12%
- 可选：日线级别失败退出（`daily_failure_exit`）
- 可选：MA60 大盘过滤（`market_filter_mode`）

## 回测结果

### 近 100 日参数实验（2025-10-29 ~ 2026-03-27）

| 实验 | 总收益 | 年化 | 夏普 | 最大回撤 | 超额 |
|------|--------|------|------|---------|------|
| looser_volume (bv=1.3, pv=0.9, daily_exit+ma60) | **+2.04%** | **+5.23%** | **0.39** | -7.95% | **+4.60%** |
| daily_exit_ma60 (bv=1.5, pv=0.8) | -2.13% | -5.27% | -0.41 | -10.34% | +0.43% |
| daily_exit (bv=1.5, pv=0.8, no filter) | -2.90% | -7.15% | -0.50 | -10.34% | -0.35% |
| baseline (bv=1.5, pv=0.8, no exit/filter) | -7.20% | -17.17% | -1.11 | -13.26% | -4.65% |
| stricter_volume (bv=1.7, pv=0.7) | -5.71% | -13.78% | -1.11 | -11.13% | -3.16% |

### 近 250 日最优参数（2025-03-18 ~ 2026-03-27）

| 指标 | 值 |
|------|----|
| 总收益 | **+13.16%** |
| 年化 | **+13.27%** |
| 夏普 | **1.18** |
| 最大回撤 | -7.97% |
| 超额（vs 基准） | -0.95% |
| 参数 | bv=1.3, pv=0.9, daily_exit, ma60 |

### 关键发现

1. **放宽成交量阈值（bv=1.3, pv=0.9）显著优于严格阈值** — 宽松门槛让更多候选进入排名，避免被极端量条件卡死
2. **日线失败退出 + MA60 大盘过滤 是必选组合** — 基线（无退出无过滤）夏普 -1.11，加上后提升到 0.39
3. **候选池偏小**（均值 2.7~3.1 只，够选率仅 10-23%）— 突破形态天然稀缺，需要更长回看窗口或更宽过滤才能获得足够候选

## 运行方式

```bash
# 默认：近 100 日基线
python strategies/avwap_profile_strategy/run_backtest.py

# 最优参数：近 250 日
python strategies/avwap_profile_strategy/run_backtest.py \
  --recent-days 250 \
  --breakout-volume-mult 1.3 \
  --pullback-volume-frac 0.9 \
  --daily-failure-exit \
  --market-filter-mode ma60

# 参数实验
python strategies/avwap_profile_strategy/run_recent_100d_experiments.py
```
