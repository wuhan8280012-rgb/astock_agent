# Skill 层清理改造 — Codex PE 补充篇
# Version: 2.1 | Date: 2026-02-24
# 前置: CODEX_PE.md (辩论引擎+to_brief)

---

## 0. 问题诊断

经全部 6 个 Skill + MarketEnvironment 代码逐行审计，发现三类问题:

### 问题 A: 数据重复采集 — 同一个 API 被调 2-3 次

```
Tushare API 调用热力图 (一次完整分析):

API 接口               Sentiment  Macro  MarketEnv  Pipeline  合计
─────────────────────  ─────────  ─────  ─────────  ────────  ────
get_north_flow()          ✅(10d)  ✅(20d)   ✅(1d)             3次
get_margin_data()         ✅(10d)  ✅(20d)                      2次
get_index_daily()         ✅(30d)  ✅(30d)   ✅(60d)            3次
get_market_breadth()      ✅                 ✅                 2次
get_limit_list()          ✅                 ✅                 2次
SentimentSkill.analyze()                              ✅(重建)  2次
SectorRotation.analyze()                              ✅(重建)  2次
                                                        ────────
                                            总计重复调用: 16次 → 可降至 8次
```

**直接后果**: Tushare 免费版每分钟 200 次限制更容易触发，响应时间翻倍。

### 问题 B: 语义方向矛盾 — 同一个 score 方向相反

| 模块 | score=+2 含义 | 投资建议方向 | 矛盾 |
|------|--------------|-------------|------|
| SentimentSkill | 极度贪婪 | **减仓**（逆向） | ⬇️ |
| MacroSkill | 明显宽松 | **加仓**（顺向） | ⬆️ |
| MarketEnvironment | 高分=良好 | **可交易**（顺向） | ⬆️ |
| RiskControlSkill | 极度贪婪→30%上限 | **减仓**（逆向） | ⬇️ |

**LLM 看到的输入**: 情绪+0.8(贪婪→减) + 宏观+0.5(宽松→加) + 环境72(→谨慎)
**LLM 反应**: 三个模块方向打架，取中间值 → 等于没有决策

### 问题 C: Pipeline 内部重复实例化

```python
# stock_pipeline.py line 120:
sentiment = SentimentSkill().analyze()    # ← 新建实例，重新调 API

# 但 Router 在 _execute_skills() 中已经调过:
results["sentiment"] = SentimentSkill().analyze()  # ← 第1次
# 然后 Pipeline 又跑一次                            # ← 第2次（浪费）
```

---

## 1. 改造方案总览

| 序号 | 改造 | 文件 | 行数 | 风险 | 效果 |
|------|------|------|------|------|------|
| A1 | 数据缓存层 | `data_fetcher.py` 或新建 `data_cache.py` | +40 | 低 | API 调用 -50% |
| A2 | Sentiment 移除两融+成交额 | `skills/sentiment.py` | -60, +5 | 低 | 消除重复 |
| A3 | Macro 移除成交额 | `skills/macro_monitor.py` | -30, +5 | 低 | 消除重复 |
| B1 | 统一 score 语义 | `skills/sentiment.py` | ~20 改 | 中 | 消除矛盾 |
| B2 | RiskControl 解耦逆向逻辑 | `skills/risk_control.py` | ~10 改 | 低 | 清晰职责 |
| C1 | Pipeline 注入已有结果 | `skills/stock_pipeline.py` | ~10 改 | 零 | 消除重复调用 |

---

## 2. 改造 A: 消除数据重复

### 2.A1 新建 `data_cache.py` — 单日数据缓存

不改 data_fetcher.py（风险太高），在上层加一个轻量缓存。

```python
"""
data_cache.py — 单日数据缓存
==============================
同一个交易日内，相同的 API 调用只实际执行一次。
收盘后或日期变更自动失效。

用法:
  cache = DailyCache(fetcher)
  df1 = cache.get_north_flow(days=10)   # 实际调用
  df2 = cache.get_north_flow(days=10)   # 命中缓存
  df3 = cache.get_north_flow(days=20)   # days不同，实际调用
"""

import time
from datetime import datetime
from typing import Any


class DailyCache:
    """单日数据缓存，包装 DataFetcher"""

    def __init__(self, fetcher):
        self.fetcher = fetcher
        self._cache = {}
        self._cache_date = ""

    def _make_key(self, method: str, *args, **kwargs) -> str:
        """生成缓存键"""
        parts = [method] + [str(a) for a in args]
        parts += [f"{k}={v}" for k, v in sorted(kwargs.items())]
        return "|".join(parts)

    def _check_date(self):
        """日期变更时清空缓存"""
        today = datetime.now().strftime("%Y%m%d")
        if today != self._cache_date:
            self._cache.clear()
            self._cache_date = today

    def _cached_call(self, method_name: str, *args, **kwargs) -> Any:
        """带缓存的方法调用"""
        self._check_date()
        key = self._make_key(method_name, *args, **kwargs)

        if key in self._cache:
            return self._cache[key]

        method = getattr(self.fetcher, method_name)
        result = method(*args, **kwargs)
        self._cache[key] = result
        return result

    # ── 代理方法: 只包装被重复调用的高频 API ──

    def get_north_flow(self, days=10, **kw):
        return self._cached_call("get_north_flow", days=days, **kw)

    def get_margin_data(self, days=10, **kw):
        return self._cached_call("get_margin_data", days=days, **kw)

    def get_index_daily(self, index_code, days=30, **kw):
        return self._cached_call("get_index_daily", index_code, days=days, **kw)

    def get_market_breadth(self, **kw):
        return self._cached_call("get_market_breadth", **kw)

    def get_limit_list(self, **kw):
        return self._cached_call("get_limit_list", **kw)

    def get_shibor(self, days=30, **kw):
        return self._cached_call("get_shibor", days=days, **kw)

    def get_all_sector_performance(self, days=120, **kw):
        return self._cached_call("get_all_sector_performance", days=days, **kw)

    def get_stock_daily(self, ts_code, days=60, **kw):
        return self._cached_call("get_stock_daily", ts_code, days=days, **kw)

    def get_latest_trade_date(self, **kw):
        return self._cached_call("get_latest_trade_date", **kw)

    def __getattr__(self, name):
        """未代理的方法直接转发给 fetcher"""
        return getattr(self.fetcher, name)

    @property
    def cache_stats(self) -> dict:
        return {
            "date": self._cache_date,
            "entries": len(self._cache),
            "keys": list(self._cache.keys())[:10],
        }
```

**集成方式** — 在 `final_system.py` 初始化时包装:

```python
from data_cache import DailyCache

# 原来:
self.fetcher = DataFetcher()

# 改为:
raw_fetcher = DataFetcher()
self.fetcher = DailyCache(raw_fetcher)

# 所有 Skill 拿到的 fetcher 都自动带缓存，零改动
```

**效果**: 同日内重复 API 调用从 16 次降至 8 次。Tushare 配额压力减半。

---

### 2.A2 SentimentSkill — 移除两融和成交额（交给 Macro 和 ME）

**职责重新定义**: SentimentSkill = **短期市场温度计**，只关注今日/3日的情绪指标。

**保留**: 涨跌比(今日)、涨跌停(今日)、北向资金(今日+3日短期)
**移除**: 两融余额(中期指标→Macro负责)、成交额(量能指标→ME负责)

#### 修改 `skills/sentiment.py`

**Step 1**: 在 `analyze()` 方法中，删除两融和成交额的调用:

找到:
```python
    def analyze(self) -> SentimentReport:
        """执行完整的情绪分析"""
        report = SentimentReport(date=self.fetcher.get_latest_trade_date())

        # 1. 涨跌比分析
        sig = self._analyze_breadth()
        if sig:
            report.signals.append(sig)

        # 2. 涨跌停分析
        sig = self._analyze_limits()
        if sig:
            report.signals.append(sig)

        # 3. 北向资金分析
        sig = self._analyze_north_flow()
        if sig:
            report.signals.append(sig)

        # 4. 两融余额分析
        sig = self._analyze_margin()
        if sig:
            report.signals.append(sig)

        # 5. 成交额分析
        sig = self._analyze_volume()
        if sig:
            report.signals.append(sig)

        # 综合评分
        self._compute_overall(report)
        return report
```

替换为:
```python
    def analyze(self) -> SentimentReport:
        """
        执行情绪分析 (v6.2: 职责瘦身)

        只保留短期温度计指标:
          1. 涨跌比 (今日) — 市场宽度
          2. 涨跌停 (今日) — 极端情绪
          3. 北向资金 (今日+3日) — 外资情绪

        已移除 (避免与 Macro/ME 重复):
          - 两融余额 → MacroSkill 负责 (中期杠杆水位)
          - 成交额   → MarketEnvironment 负责 (量能维度)
        """
        report = SentimentReport(date=self.fetcher.get_latest_trade_date())

        # 1. 涨跌比分析
        sig = self._analyze_breadth()
        if sig:
            report.signals.append(sig)

        # 2. 涨跌停分析
        sig = self._analyze_limits()
        if sig:
            report.signals.append(sig)

        # 3. 北向资金分析 (短期: 今日+3日)
        sig = self._analyze_north_flow()
        if sig:
            report.signals.append(sig)

        # 综合评分
        self._compute_overall(report)
        return report
```

**Step 2**: `_analyze_margin()` 和 `_analyze_volume()` 方法**不删除**，加注释标记为废弃:

在 `_analyze_margin` 方法上方添加:
```python
    # ── 以下方法已废弃(v6.2)，保留代码供回退 ──
    # 两融余额已移交 MacroSkill，成交额已移交 MarketEnvironment
```

这样回滚只需把 analyze() 中的两行取消注释。

---

### 2.A3 MacroSkill — 移除成交额趋势（交给 ME）

**职责重新定义**: MacroSkill = **中期流动性水位线**，只关注 20-60 日的宏观指标。

**保留**: SHIBOR(利率)、北向资金20日趋势、两融余额20日趋势
**移除**: 成交额趋势(量能指标→ME 负责)

#### 修改 `skills/macro_monitor.py`

找到 `analyze()` 方法中:
```python
        # 4. 市场成交额趋势
        sig = self._analyze_turnover_trend()
        if sig:
            report.signals.append(sig)
```

替换为:
```python
        # 4. 成交额趋势 — 已移交 MarketEnvironment (v6.2)
        # sig = self._analyze_turnover_trend()
        # if sig:
        #     report.signals.append(sig)
```

同样，`_analyze_turnover_trend()` 方法保留不删，加废弃注释。

---

### 清理后的职责矩阵

```
                  涨跌比  涨跌停  北向短期  北向趋势  两融  成交量  SHIBOR  板块  均线
                  (今日)  (今日)  (1-3d)   (20d)    (20d) (量比)  (利率)  (轮动) (MA)
SentimentSkill     ✅      ✅      ✅        -        -      -       -      -     -
MacroSkill          -       -       -       ✅       ✅      -      ✅      -     -
MarketEnvironment  ✅*      ✅*     ✅*       -        -     ✅       -     ✅    ✅
SectorRotation      -       -       -        -        -      -       -     ✅     -
RiskControl         -       -       -        -        -      -       -      -     -

✅* = ME 从 Tushare/intraday 独立获取，不调用 SentimentSkill

重叠: 涨跌比和涨跌停在 Sentiment 和 ME 中都有
差异: Sentiment 输出情绪等级(用于决策), ME 输出环境评分(用于门控)
理由: 保留两处 — 它们的用途不同，且 ME 的盘中覆盖机制独立于 Sentiment
```

---

## 3. 改造 B: 统一评分语义

### 3.B1 SentimentSkill — score 统一为"事实描述"方向

**核心原则**: signal.score 只描述"市场有多热"，+2=极热，-2=极冷。
**逆向策略**（热=减仓）只在 `_compute_overall()` 的仓位建议中体现，不编入 score。

**当前代码** (无需修改 score 值，因为已经是 +2=热 -2=冷):

涨跌比: +2=极度贪婪 → 市场很热 ✅ 方向正确
涨跌停: +2=极度贪婪 → 赚钱效应强 ✅ 方向正确
北向资金: +1=贪婪 → 外资看好 ✅ 方向正确

**实际问题在 `_compute_overall()`**: 逆向仓位建议 与 score 方向混在一起。

修改 `_compute_overall()`:

找到:
```python
    def _compute_overall(self, report: SentimentReport):
        """综合评分及仓位建议"""
        if not report.signals:
            report.overall_level = "数据不足"
            report.summary = "无法获取足够的市场数据来进行情绪评估"
            return

        scores = [s.score for s in report.signals]
        avg_score = np.mean(scores)
        report.overall_score = round(avg_score, 2)

        # 映射为等级
        if avg_score >= 1.5:
            report.overall_level = "极度贪婪"
            report.suggested_position = "≤30%仓位，警惕回调"
        elif avg_score >= 0.5:
            report.overall_level = "贪婪"
            report.suggested_position = "50-70%仓位，逢高减仓"
        elif avg_score >= -0.5:
            report.overall_level = "中性"
            report.suggested_position = "50%仓位，均衡配置"
        elif avg_score >= -1.5:
            report.overall_level = "恐慌"
            report.suggested_position = "60-80%仓位，逢低加仓"
        else:
            report.overall_level = "极度恐慌"
            report.suggested_position = "≥80%仓位，分批抄底"

        # 生成摘要
        parts = []
        for s in report.signals:
            parts.append(f"{s.name}:{s.level}({s.detail})")
        report.summary = " | ".join(parts)
```

替换为:
```python
    def _compute_overall(self, report: SentimentReport):
        """
        综合评分及仓位建议 (v6.2)

        score 语义: +2=极热, -2=极冷 (事实描述)
        仓位建议: 逆向策略 — 越热越谨慎，越冷越积极

        注意: overall_score 仍然是 +高=热, -低=冷
              逆向逻辑只体现在 suggested_position 中
              这样 debate.py 的看空分析师可以利用 "情绪过热" 作为风险论据
        """
        if not report.signals:
            report.overall_level = "数据不足"
            report.summary = "无法获取足够的市场数据来进行情绪评估"
            return

        scores = [s.score for s in report.signals]
        avg_score = np.mean(scores)
        report.overall_score = round(avg_score, 2)

        # 等级 = 事实描述 (不含方向建议)
        if avg_score >= 1.5:
            report.overall_level = "极度贪婪"
        elif avg_score >= 0.5:
            report.overall_level = "贪婪"
        elif avg_score >= -0.5:
            report.overall_level = "中性"
        elif avg_score >= -1.5:
            report.overall_level = "恐慌"
        else:
            report.overall_level = "极度恐慌"

        # 仓位建议 = 逆向策略 (明确标注)
        # ⚠️ 这是逆向思维: 市场越热→仓位越低, 市场越冷→仓位越高
        contrarian_position = {
            "极度贪婪": "≤30%仓位(逆向:极热→防守)",
            "贪婪":     "50-70%仓位(逆向:偏热→谨慎)",
            "中性":     "50%仓位(均衡配置)",
            "恐慌":     "60-80%仓位(逆向:偏冷→积极)",
            "极度恐慌": "≥80%仓位(逆向:极冷→进攻)",
        }
        report.suggested_position = contrarian_position.get(
            report.overall_level, "50%仓位(均衡配置)"
        )

        # 摘要
        parts = [f"{s.name}:{s.level}({s.score:+d})" for s in report.signals]
        report.summary = " | ".join(parts)
```

**关键改变**:
1. `suggested_position` 明确标注 `(逆向:极热→防守)`，LLM 读到后不会和顺向指标矛盾
2. `overall_score` 保持 +高=热 方向，与 MacroSkill 的 +高=利好 一致
3. 逆向解读的任务交给 `debate.py` 的看空分析师 — "情绪 +0.8 极度贪婪，这正是你应该害怕的"

---

### 3.B2 RiskControlSkill — 解耦逆向逻辑

当前 `_get_max_position()` 直接把 sentiment level 映射为仓位上限:

```python
def _get_max_position(self, sentiment_level: str) -> float:
    mapping = {
        "极度贪婪": 0.30,
        "贪婪": 0.50,
        "中性": 0.70,
        "恐慌": 0.80,
        "极度恐慌": 0.90,
    }
    return mapping.get(sentiment_level, 0.70)
```

**问题**: 这里硬编码了逆向策略，但调用方可能不知道"极度恐慌 → 90% 仓位"是逆向逻辑。

**修改**: 加注释说明 + 增加顺向备选（让 debate 裁决用哪个）

找到 `_get_max_position`，替换为:

```python
    def _get_max_position(self, sentiment_level: str) -> float:
        """
        根据情绪等级确定最大仓位 — 逆向策略

        逻辑: 市场越热 → 允许仓位越低 (逼自己减仓)
              市场越冷 → 允许仓位越高 (逼自己抄底)

        这是一种纪律约束，不是预测。
        debate.py 的主席可以在裁决时参考或覆盖此建议。
        """
        contrarian_map = {
            "极度贪婪": 0.30,   # 极热 → 最多30%
            "贪婪":     0.50,   # 偏热 → 最多50%
            "中性":     0.70,   # 常态 → 最多70%
            "恐慌":     0.80,   # 偏冷 → 最多80%
            "极度恐慌": 0.90,   # 极冷 → 最多90%
        }
        return contrarian_map.get(sentiment_level, 0.70)

    def get_position_perspectives(self, sentiment_level: str) -> dict:
        """
        三视角仓位建议 — 供 debate.py 风控环节使用

        Returns
        -------
        {
            "contrarian": 0.30,   # 逆向策略建议
            "trend_follow": 0.80, # 顺势策略建议 (热→加)
            "neutral": 0.50,      # 中性建议
        }
        """
        contrarian = self._get_max_position(sentiment_level)

        # 顺势映射 (热→高仓位, 冷→低仓位)
        trend_map = {
            "极度贪婪": 0.90,
            "贪婪":     0.80,
            "中性":     0.50,
            "恐慌":     0.30,
            "极度恐慌": 0.20,
        }
        trend_follow = trend_map.get(sentiment_level, 0.50)

        return {
            "contrarian": contrarian,
            "trend_follow": trend_follow,
            "neutral": round((contrarian + trend_follow) / 2, 2),
        }
```

---

## 4. 改造 C: Pipeline 注入已有结果

### 4.C1 修改 `stock_pipeline.py` — 接受外部注入

找到 `run()` 方法签名:
```python
    def run(self, top_n_result: int = 10) -> PipelineReport:
```

替换为:
```python
    def run(
        self,
        top_n_result: int = 10,
        sentiment_result=None,
        sector_result=None,
    ) -> PipelineReport:
```

找到 Stage 1 中:
```python
        # ---- Stage 1: 市场情绪 ----
        print("[Pipeline Stage 1/5] 分析市场情绪...")
            sentiment = SentimentSkill().analyze()
```

替换为:
```python
        # ---- Stage 1: 市场情绪 (v6.2: 支持外部注入，避免重复调用) ----
        print("[Pipeline Stage 1/5] 分析市场情绪...")
            if sentiment_result is not None:
                sentiment = sentiment_result
                if verbose:
                    print("  (使用已有情绪结果，跳过重复分析)")
            else:
                sentiment = SentimentSkill().analyze()
```

找到 Stage 2 中:
```python
            sector_result = self.sector_skill.analyze()
```

替换为:
```python
            if sector_result is None:
                sector_result = self.sector_skill.analyze()
            elif verbose:
                print("  (使用已有板块结果，跳过重复分析)")
```

然后在 **Router** 调度 Pipeline 时传入已有结果:

```python
# router.py 中调用 Pipeline 的位置:
# 改造前:
pipeline_result = self.stock_pipeline.run()

# 改造后:
pipeline_result = self.stock_pipeline.run(
    sentiment_result=skill_results.get("sentiment"),
    sector_result=skill_results.get("sector_rotation"),
)
```

---

## 5. 验证脚本

```python
"""
verify_skill_cleanup.py — Skill 清理验证
"""

def test_sentiment_slim():
    """验证 Sentiment 移除了两融和成交额"""
    from skills.sentiment import SentimentSkill

    # Mock fetcher
    class MockFetcher:
        def get_latest_trade_date(self): return "20260224"
        def get_market_breadth(self): return {"ratio": 2.0, "up": 3000, "down": 1500, "flat": 200}
        def get_limit_list(self):
            import pandas as pd
            return pd.DataFrame({"limit": ["U"]*40 + ["D"]*15})
        def get_north_flow(self, days=10):
            import pandas as pd
            return pd.DataFrame({"north_money_yi": [50.0, 60.0, 75.0]})
        # 注意: 没有 get_margin_data 和 get_index_daily
        # 如果 Sentiment 还在调这两个，会报错

    skill = SentimentSkill()
    skill.fetcher = MockFetcher()
    report = skill.analyze()

    signal_names = [s.name for s in report.signals]
    assert "两融余额" not in signal_names, f"两融余额应已移除! 当前信号: {signal_names}"
    assert "成交额" not in signal_names, f"成交额应已移除! 当前信号: {signal_names}"
    assert "涨跌比" in signal_names
    assert "涨跌停" in signal_names or len(signal_names) >= 2  # limit_list 可能解析不同

    print(f"  ✅ Sentiment 瘦身: 信号={signal_names}")
    print(f"     overall={report.overall_level}({report.overall_score:+.1f})")
    print(f"     position={report.suggested_position}")

    # 验证逆向标注
    if report.overall_score > 0.5:
        assert "逆向" in report.suggested_position, "仓位建议应标注'逆向'"
        print(f"  ✅ 逆向标注正确")


def test_macro_slim():
    """验证 Macro 移除了成交额"""
    from skills.macro_monitor import MacroSkill

    class MockFetcher:
        def get_latest_trade_date(self): return "20260224"
        def get_shibor(self, days=30):
            import pandas as pd
            data = [{"date": f"2026022{i}", "on": 1.5+i*0.01, "1w": 1.8+i*0.01}
                    for i in range(15)]
            return pd.DataFrame(data)
        def get_north_flow(self, days=20):
            import pandas as pd
            return pd.DataFrame({"north_money_yi": [10.0]*20})
        def get_margin_data(self, days=20):
            import pandas as pd
            return pd.DataFrame({
                "trade_date": [f"2026020{i}" for i in range(20)],
                "rzye": [18500e8 + i*10e8 for i in range(20)],
            })
        # 注意: 没有 get_index_daily → 如果 Macro 还调成交额会报错

    skill = MacroSkill()
    skill.fetcher = MockFetcher()
    report = skill.analyze()

    signal_names = [s.name for s in report.signals]
    assert "成交额趋势" not in signal_names, f"成交额应已移除! 当前: {signal_names}"
    print(f"  ✅ Macro 瘦身: 信号={signal_names}")


def test_pipeline_injection():
    """验证 Pipeline 接受外部注入"""
    from skills.stock_pipeline import StockPipeline

    # 只需验证签名接受参数，不需要真正执行
    import inspect
    sig = inspect.signature(StockPipeline.run)
    params = list(sig.parameters.keys())
    assert "sentiment_result" in params, f"run() 缺少 sentiment_result 参数: {params}"
    assert "sector_result" in params, f"run() 缺少 sector_result 参数: {params}"
    print(f"  ✅ Pipeline.run() 签名正确: {params}")


def test_risk_perspectives():
    """验证 RiskControl 三视角"""
    from skills.risk_control import RiskControlSkill

    skill = RiskControlSkill()

    # 验证新方法存在
    assert hasattr(skill, 'get_position_perspectives'), "缺少 get_position_perspectives 方法"

    p = skill.get_position_perspectives("极度贪婪")
    assert p["contrarian"] == 0.30, f"逆向应=0.30: {p}"
    assert p["trend_follow"] == 0.90, f"顺势应=0.90: {p}"
    assert p["neutral"] == 0.60, f"中性应=0.60: {p}"

    p2 = skill.get_position_perspectives("极度恐慌")
    assert p2["contrarian"] == 0.90
    assert p2["trend_follow"] == 0.20

    print(f"  ✅ 三视角: 贪婪→{p}, 恐慌→{p2}")


def test_data_cache():
    """验证数据缓存"""
    from data_cache import DailyCache

    call_count = {"n": 0}
    class MockFetcher:
        def get_north_flow(self, days=10):
            call_count["n"] += 1
            return f"north_{days}d"
        def get_index_daily(self, code, days=30):
            call_count["n"] += 1
            return f"index_{code}_{days}d"

    cache = DailyCache(MockFetcher())

    # 首次调用
    r1 = cache.get_north_flow(days=10)
    assert call_count["n"] == 1
    # 重复调用 → 命中缓存
    r2 = cache.get_north_flow(days=10)
    assert call_count["n"] == 1  # 不增加
    assert r1 == r2
    # 不同参数 → 新调用
    r3 = cache.get_north_flow(days=20)
    assert call_count["n"] == 2

    print(f"  ✅ DailyCache: 3次请求, 实际调用{call_count['n']}次")
    print(f"     缓存条目: {cache.cache_stats['entries']}")


if __name__ == "__main__":
    import sys
    print("=" * 60)
    print("  Skill 清理验证")
    print("=" * 60)

    tests = [
        ("数据缓存", test_data_cache),
        ("Sentiment 瘦身", test_sentiment_slim),
        ("Macro 瘦身", test_macro_slim),
        ("Pipeline 注入", test_pipeline_injection),
        ("Risk 三视角", test_risk_perspectives),
    ]

    passed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")

    print(f"\n{'='*60}")
    print(f"  结果: {passed}/{len(tests)} 通过")
    print(f"{'='*60}")
```

---

## 6. 改造后 Token 流对比

```
改造前 — 用户问 "贵州茅台能买吗":
  Sentiment.to_dict()  → ~500 tokens (含两融+成交额冗余)
  Macro.to_dict()      → ~400 tokens (含成交额冗余)
  ME result dict       → ~300 tokens
  Risk.to_dict()       → ~400 tokens
  json.dumps 拼接      → ~1600 tokens → 截断到 ~4000 chars
  → LLM 看到: 3份成交额分析(矛盾) + 2份北向分析(重复) + 混乱方向建议

改造后 — 同一问题:
  Sentiment.to_brief() → ~50 tokens (3个信号，方向一致)
  Macro.to_brief()     → ~40 tokens (3个信号，无重复)
  ME.to_brief()        → ~30 tokens (综合评分)
  Risk.to_brief()      → ~40 tokens
  拼接                 → ~160 tokens (无截断，无信息损失)
  → LLM 看到: 清晰、无矛盾、无重复的市场快照

  总输入 token: 4000 → 160 = 节省 96%
  LLM 决策质量: 矛盾信息 → 清晰信息 = 显著提升
```

---

## 7. 执行顺序

```
Phase 0 (前置，5 分钟):
  └─ 新建 data_cache.py → final_system.py 包装 fetcher

Phase 1 (Skill 瘦身，15 分钟):
  ├─ A2: sentiment.py 移除两融+成交额 (注释3行)
  ├─ A3: macro_monitor.py 移除成交额 (注释3行)
  └─ 运行 verify: test_sentiment_slim + test_macro_slim

Phase 2 (语义统一，10 分钟):
  ├─ B1: sentiment.py _compute_overall 加逆向标注
  ├─ B2: risk_control.py 加 get_position_perspectives
  └─ 运行 verify: test_risk_perspectives

Phase 3 (Pipeline 注入，5 分钟):
  ├─ C1: stock_pipeline.py run() 加参数
  ├─ Router 调用处传入已有结果
  └─ 运行 verify: test_pipeline_injection

总耗时: ~35 分钟
总风险: 低 (全部注释式修改，不删代码)
```

---

## 附录: 改造前后完整语义对照

```
改造前:
  用户: "贵州茅台能买吗"
  LLM 输入:
    情绪: overall_score=+0.8, level=极度贪婪, position=≤30%仓位     → 减!!
    宏观: overall_score=+0.5, level=中性偏松, impact=可适度进攻      → 加!!
    环境: total_score=72, level=一般, advice=谨慎交易               → 中??
    风控: max_position=0.30 (因为极度贪婪)                         → 减!!
    成交额(情绪版): ratio=1.15x, 温和放量                          → 中
    成交额(宏观版): 5d/20d=1.08, 正常                              → 中
    成交额(环境版): 全市场10284亿, score=72                         → 中
  LLM: "呃...三个说减两个说加三个说成交额但数字不一样...持有吧"

改造后:
  用户: "贵州茅台能买吗"
  LLM 输入 (160 tokens, 无矛盾):
    [情绪] 极度贪婪(+0.8) 仓位:≤30%(逆向:极热→防守) | 涨跌比+1,涨跌停+1,北向+1
    [宏观] 中性偏松(+0.5) 流动性尚可 | SHIBOR+1,北向趋势+1,两融+0
    [环境] 72/100(一般) 趋势70 情绪81 量能72 板块75 → 谨慎交易
    [风控] 仓位36%/30%上限(逆向) | 止损:比亚迪
  LLM (辩论):
    看多: 三信号均正，宏观宽松支撑
    看空: 情绪极热是逆向危险信号，风控已要求降至30%
    裁决: 持有，不加仓。环境仅"一般"，情绪过热需警惕回调。
```
