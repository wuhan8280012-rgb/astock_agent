# Ultra Rotation Strategy – 极限短期轮动

## 目标

在 A 股日线级别 T+1 框架下，追求纯技术短期轮动的实际上限：

| 指标 | 目标范围 |
|------|---------|
| 年化收益 | 40-80% |
| 夏普比率 | 1.5-2.5 |
| 最大回撤 | 15-25% |

## 核心设计

### 6 因子模型 (正交 + 软排名融合)

| # | 因子 | 权重 | 说明 |
|---|------|------|------|
| 1 | 多时间框架动量 | 25% | 5d/10d/20d/60d 加权组合 |
| 2 | 动量加速度 | 20% | 5d 超额相对 20d 趋势速率 |
| 3 | 量能突破 | 15% | 5d 均量 / 20d 均量 |
| 4 | 低波动率 | 15% | -std(20d日收益率) |
| 5 | MA20角度趋势 | 15% | 角度斜率 + 持续性 |
| 6 | 行业相对强度 | 10% | 行业20d收益 vs 市场 |

### 相对 F 策略的改进

| 维度 | F 策略 | Ultra Rotation |
|------|--------|----------------|
| 因子数 | 4 | 6 (正交化) |
| 调仓周期 | 20d | 5d |
| 持仓集中度 | 15 只 | 10 只 |
| 最大单只权重 | 8% | 12% |
| 动量信号 | 仅 60d | 5/10/20/60d |
| 量能信号 | 无 | 量能突破 |
| 行业信号 | 无 | 行业 RS |
| 动量加速度 | 无 | 5d vs 20d |
| 止损 | -15% | -10% |

### 关键设计原则

1. **NO 多层硬过滤** – 仅使用 1 个硬过滤 (transition_coef >= -0.1)，避免候选池过度收缩
2. **全部信号维度通过软排名融合** – 每个因子独立排名，按权重加权平均
3. **正交因子选择** – 动量、加速度、量能、波动率、趋势质量、行业轮动 6 个不同维度
4. **高频再平衡** – 5 天调仓捕捉更多轮动 alpha
5. **集中持仓** – 10 只高 alpha 候选，最大化 alpha 密度

### 风险管理

- **趋势过滤**: 上证指数 200d 均线 → 牛市满仓 / 震荡半仓 / 熊市空仓
- **个股止损**: -10% 从成本价
- **流动性门槛**: 20d 均成交额 >= 1.5 亿
- **缓冲带**: 持仓窗口 1.3×top_n = 13 只，减少因排名微小波动的换手

## 回测模式

### 1. 全量回测 (默认)

```bash
# Static 5y 数据
python strategies/ultra_rotation_strategy/run_backtest.py

# PIT 数据 (消除幸存者偏差)
python strategies/ultra_rotation_strategy/run_backtest.py \
    --dataset csi1000_5y_pit --use-pit-constituents
```

### 2. Train/Test 分段回测

将回测区间分为训练期和测试期，验证样本外表现：

```bash
python strategies/ultra_rotation_strategy/run_backtest.py \
    --dataset csi1000_5y_pit --use-pit-constituents \
    --train-end 20240401 --test-start 20240401
```

### 3. Walk-Forward 滚动分析

滚动训练/测试窗口，评估策略稳定性：

```bash
# 默认: 3年训练 + 1年测试
python strategies/ultra_rotation_strategy/run_backtest.py \
    --dataset csi1000_5y_pit --use-pit-constituents \
    --walk-forward

# 自定义窗口
python strategies/ultra_rotation_strategy/run_backtest.py \
    --walk-forward --wf-train-days 504 --wf-test-days 126 --wf-step-days 63
```

### 4. 最近 N 日回测

只回测最近 N 个交易日，快速验证近期表现：

```bash
python strategies/ultra_rotation_strategy/run_backtest.py \
    --dataset csi1000_5y_pit --use-pit-constituents \
    --recent-days 100
```

### 5. Static vs PIT A/B 对比

同一参数下对比静态样本和 PIT 历史成分股样本，量化幸存者偏差：

```bash
python strategies/ultra_rotation_strategy/run_backtest.py --compare-pit
```

### 参数调优

```bash
# 调整因子权重
python strategies/ultra_rotation_strategy/run_backtest.py \
    --w-momentum 0.30 --w-accel 0.20 --w-volume 0.15 \
    --w-lowvol 0.10 --w-angle 0.15 --w-industry 0.10

# 调整组合参数
python strategies/ultra_rotation_strategy/run_backtest.py \
    --top-n 8 --rebalance-interval 3 --stop-loss-pct -0.08

# 参数扫描实验 (因子权重 × 组合参数 × 止损)
python strategies/ultra_rotation_strategy/run_param_sweep.py \
    --dataset csi1000_5y_pit --use-pit-constituents --sweep all
```

## 交易成本模型

| 成本项 | 费率 | 方向 |
|--------|------|------|
| 佣金 | 0.03% | 买卖双边 |
| 印花税 | 0.1% | 仅卖出 |
| 滑点 | 0.2% | 买卖双边 |
| **单次换手成本** | **~0.63%** | — |

## 文件结构

```
strategies/ultra_rotation_strategy/
├── __init__.py
├── scoring.py             # 6-factor scoring engine
├── run_backtest.py        # Comprehensive CLI backtest (5 modes)
├── run_param_sweep.py     # Parameter sweep experiments
├── strategy_config.json   # Default configuration
└── README.md              # This file
```

## CLI 参数一览

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | `csi1000_5y` | 数据集 (`csi1000_5y` / `csi1000_5y_pit`) |
| `--use-pit-constituents` | off | 启用 PIT 历史成分股过滤 |
| `--execution-mode` | `same_close` | 执行模式 (`same_close` / `next_open`) |
| `--w-momentum` | 0.25 | 多时间框架动量权重 |
| `--w-accel` | 0.20 | 动量加速度权重 |
| `--w-volume` | 0.15 | 量能突破权重 |
| `--w-lowvol` | 0.15 | 低波动率权重 |
| `--w-angle` | 0.15 | MA20 角度趋势权重 |
| `--w-industry` | 0.10 | 行业相对强度权重 |
| `--top-n` | 10 | 目标持仓数量 |
| `--rebalance-interval` | 5 | 调仓间隔 (交易日) |
| `--stop-loss-pct` | -0.10 | 个股止损阈值 |
| `--max-single-weight` | 0.12 | 最大单只权重 |
| `--train-end` | — | 训练期结束日期 (YYYYMMDD) |
| `--test-start` | — | 测试期开始日期 (YYYYMMDD) |
| `--walk-forward` | off | 启用 walk-forward 分析 |
| `--wf-train-days` | 756 | WF 训练窗口 (交易日) |
| `--wf-test-days` | 252 | WF 测试窗口 (交易日) |
| `--recent-days` | — | 最近 N 日回测 |
| `--compare-pit` | off | Static vs PIT A/B 对比 |
