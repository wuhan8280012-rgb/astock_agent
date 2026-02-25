# Skill 7: 板块Stage联合过滤器 — 集成指南
# Version: 1.0 | Date: 2026-02-24

---

## 1. 策略原文 vs A股适配对照

| 原文 (美股) | A股适配 | 改造理由 |
|------------|---------|---------|
| 11 sector ETFs (XLK, XLF...) | 申万31个一级行业指数 | A股无板块ETF覆盖全行业，用申万体系 |
| RS vs SPY/QQQ | RS vs 沪深300/创业板指 | 双基准覆盖主板+成长 |
| 4 weeks / 12 weeks | 20日 / 60日 | A股年交易日244 < 美股252，用日线替代周线窗口 |
| Top 3 / Bottom 3 | Top 5 / Bottom 5 | 31个行业比11个多，适当扩大 |
| Stage 1 base (months/years) | 60-500日底部检测 | 适配A股牛熊周期（通常2-3年） |
| Weekly chart base | 日线数据构造周线级底部 | Tushare 周线数据不稳定，用日线聚合 |
| Daily tightening: 9/21/50 EMA | 10/20/50 EMA | A股量化常用周期 |
| R:R 5-10x | R:R ≥ 3x | A股涨跌停+T+1限制，降低盈亏比要求 |
| "leadership doesn't last" | 短期RS < 长期RS×0.5 + 更低高点 | 量化轮动衰退信号 |

---

## 2. 与现有系统的关系

```
改造前的 Pipeline:
  SectorRotation → 绝对涨幅排名 → 成分股 → CANSLIM → 缩量整理 → 输出

改造后 (新增 SectorStageFilter):
  SectorRotation → 基础排名 (保留)
         ↓
  SectorStageFilter → RS相对强度 + 黑名单 + Stage底部 + 日线收紧 + R:R
         ↓
  StockPipeline → CANSLIM评分 → 最终排名

  即: 在 Pipeline 之前插入一个更严格的前置过滤器
```

### 互补而非替代

| 功能 | sector_rotation.py | sector_stage_filter.py |
|------|-------------------|----------------------|
| 排名方式 | 绝对涨幅百分位 | 相对强度 vs 基准 |
| 黑名单 | ❌ | ✅ Bottom 5 |
| 底部检测 | ❌ | ✅ 60-500日大底 |
| 日线收紧 | ⚠️ pipeline里的缩量整理(15日) | ✅ 10日收紧+EMA骑乘 |
| 轮动预警 | "高位放量" (1种) | 动能衰退+跑输+更低高点 (3种) |
| R:R | ❌ | ✅ ATR法 ≥3:1 |

---

## 3. 集成到 Router

### 3.1 注册为新 Skill

在 `router.py` 的 Skill 注册表中添加:

```python
from skills.sector_stage_filter import SectorStageFilter

# 在 Router.__init__() 中:
self.skills["sector_stage"] = SectorStageFilter()
```

### 3.2 路由触发条件

```python
# router.py _route() 方法中添加:
# 触发 sector_stage 的关键词:
sector_stage_triggers = [
    "板块轮动", "板块分析", "强势板块", "资金流向",
    "选股", "底部", "大底", "Stage", "stage",
    "哪些板块", "板块排名", "轮动",
]
```

### 3.3 Pipeline 集成 — 使用 Stage 结果作为前置过滤

在 `stock_pipeline.py` 的 Stage 2 之后插入:

```python
# ---- Stage 2.5 (新增): Stage 联合过滤 ----
if stage_filter_result is not None:
    # 用 SectorStageFilter 的黑名单过滤
    blacklisted_sectors = {s.name for s in stage_filter_result.blacklisted}
    stock_pool = [
        code for code in stock_pool
        if sector_stock_map.get(code, "") not in blacklisted_sectors
    ]

    # 用 Stage 候选加分
    stage_candidates = {c.ts_code: c for c in stage_filter_result.candidates}
    # ... 后续 CANSLIM 阶段可以引用 stage_score
```

### 3.4 与辩论引擎的协同

SectorStageFilter 的输出在辩论中的作用:

```
debate.py 看多分析师:
  "该股在 A级大底 + 日线收紧 + 多头排列, R:R=4.2:1,
   且所在板块 RS 排名第2 (电子), 形态极佳"

debate.py 看空分析师:
  "板块虽然领涨但短期RS已衰退(警告),
   且黑名单板块(房地产)短期突然走强, 资金可能轮动.
   该股盈亏比虽达标但 EMA 纠缠非多头排列"
```

---

## 4. 参数调优指南

```python
# config.py 中添加:
SKILL_PARAMS["sector_stage_filter"] = {
    # 如果漏掉太多机会 → 放宽
    "base_min_days": 45,          # 从60降到45 (允许更短的底部)
    "min_rr_ratio": 2.5,          # 从3.0降到2.5

    # 如果误选太多 → 收紧
    "base_min_days": 90,          # 从60升到90 (只要大底)
    "tight_range_max_pct": 3,     # 从5降到3 (更严格的收紧)
    "min_rr_ratio": 4.0,          # 从3.0升到4.0

    # 牛市 → 宽松
    "top_n": 8,                   # 关注更多板块
    "blacklist_n": 3,             # 少拉黑

    # 熊市 → 严格
    "top_n": 3,                   # 只关注最强的
    "blacklist_n": 10,            # 多拉黑
}
```

---

## 5. 回测验证建议

在 `backtest_v6.py` 中新增 Stage 过滤对比:

```python
# v6.0: 环境评分 < 60 停手
# v6.2: 环境评分 < 60 停手 + Stage 过滤 (只买大底+收紧)

# 假设: Stage 过滤能额外过滤掉 30-40% 的低质量信号
# 验证: 对比 v6.0 vs v6.2 在 2025.11-12 数据上的胜率差异
```

---

## 6. 完整调用示例

```python
from skills.sector_stage_filter import SectorStageFilter
from skills.sector_rotation import SectorRotationSkill

# 方式1: 独立运行
stage_filter = SectorStageFilter()
result = stage_filter.analyze(verbose=True)

print(result.to_brief())
# [板块Stage] 领涨: 电子(RS+8.2), 计算机(RS+6.1), 军工(RS+4.5) | 黑名单: 房地产, 建筑, 钢铁
#   预警: ⚠️ 电子: 短期RS弱于长期RS，领涨动能衰退
#   扫描320→通过12: 海康威视(A级大底,🔥 周,R:R=4.2), 中际旭创(B级中底,⭐ 日,R:R=3.5)

# 方式2: 配合现有 SectorRotation
sector_result = SectorRotationSkill().analyze()
stage_result = stage_filter.analyze(sector_report=sector_result)

# 方式3: 在 Pipeline 中使用
pipeline_result = StockPipeline().run(
    sentiment_result=sentiment,
    sector_result=sector_result,
    stage_filter_result=stage_result,  # 新增参数
)
```
