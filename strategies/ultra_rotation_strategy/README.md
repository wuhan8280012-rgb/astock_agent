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

## 使用方法

```bash
# 默认 (static 5y 数据)
python strategies/ultra_rotation_strategy/run_backtest.py

# PIT 数据 (真实历史成分股，消除幸存者偏差)
python strategies/ultra_rotation_strategy/run_backtest.py \
    --dataset csi1000_5y_pit \
    --use-pit-constituents

# 调整因子权重
python strategies/ultra_rotation_strategy/run_backtest.py \
    --w-momentum 0.30 --w-accel 0.20 --w-volume 0.15 \
    --w-lowvol 0.10 --w-angle 0.15 --w-industry 0.10

# 调整组合参数
python strategies/ultra_rotation_strategy/run_backtest.py \
    --top-n 8 --rebalance-interval 3 --stop-loss-pct -0.08
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
├── run_backtest.py        # CLI entry point
├── strategy_config.json   # Default configuration
└── README.md              # This file
```
