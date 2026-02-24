# Workflow Orchestration

### 1. Plan Node Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately – don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes – don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests – then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

# Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

# Core Principles
- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.

---

# Project: A股每日前瞻 Agent (astock_agent)

## Architecture (四层架构，修改前必须理解)

```
交互层:  router.py                          ← 用户问答入口，路由到Skill
CRON层:  daily_agent.py / weekly_agent.py   ← 定时调度入口
Skills层: skills/*.py                        ← 所有决策逻辑
基础层:  data_fetcher.py + knowledge_base.py + scratchpad.py + eval_layer.py
```

**关键约束：**
- `data_fetcher.py` 是 Singleton，全局只有一个实例，所有 Skill 通过 `get_fetcher()` 获取
- `scratchpad.py` 记录每次Skill运行的输入输出到 `knowledge/scratchpad.jsonl`（不要删这个文件）
- `eval_layer.py` 读取scratchpad历史，用实际行情验证Skill准确性
- `router.py` 只暴露5个路由选项给LLM，不暴露Skill细节（参考Dexter: 缩小决策空间）
- 缓存目录 `./cache`，过期时间由 `config.py` 的 `CACHE_EXPIRE_HOURS` 控制
- 所有可调参数集中在 `config.py` 的 `SKILL_PARAMS` 字典中，Skill 内不硬编码阈值

## A股特殊规则（开发时必须遵守）
- **T+1**: 买入当日不能卖出，所有回测和信号生成必须考虑这一点
- **涨跌停**: 主板±10%，创业板/科创板±20%，回测中不能假设涨停板可以买入
- **北向资金**: Tushare `moneyflow_hsgt` 的 hgt/sgt 单位是万元，转亿除以 `10000`
- **Tushare频率限制**: 每分钟约200次调用，批量操作必须加 `time.sleep()`
- **交易日**: 用 `get_trade_dates()` 判断，不能用自然日计算
- **财报日期**: A股财报披露截止日 4/30、8/31、10/31，注意前瞻偏差

## 数据源
- **主数据源**: Tushare Pro API（需要积分，部分接口有权限要求）
- **备选**: AKShare（免费但不稳定，可作为fallback）
- 所有数据拉取必须经过 `DataFetcher` 的缓存机制，禁止直接调用 `ts.pro_api()`

## 新增 Skill 的规范
1. 放在 `skills/` 目录下
2. 必须有 `analyze()` 方法返回 dataclass 报告对象
3. 报告对象必须有 `to_dict()` 方法（供报告生成器和Scratchpad使用）
4. 在 `skills/__init__.py` 中注册
5. 参数通过 `config.py` 的 `SKILL_PARAMS` 传入，提供默认值
6. Agent调度器中运行Skill后必须调用 `sp.log(skill_name, output_data=result.to_dict())`
7. 如果要接入路由器，在 `router.py` 的 ROUTES 字典中注册

## 核心设计原则（参考Dexter）
- **缩小决策空间**: router.py 只暴露5个路由选项，不让LLM看到所有Skill细节
- **探索者与回答者分离**: 数据收集(Skills)和报告生成(report_generator)严格分离
- **Scratchpad留痕**: 每次运行必须写日志，复盘时才能回溯"Agent当时看到了什么"
- **结构化数据直给**: 不在中间层做摘要，让LLM看到原始数据自己判断

## 测试
- 每个 Skill 文件底部有 `if __name__ == "__main__"` 测试块
- 修改后先单独运行该文件验证，再跑完整 Agent
- 验证顺序: `data_fetcher.py` → 单个 Skill → `daily_agent.py` → `weekly_agent.py`

## 常见坑（lessons learned）
- Tushare 返回的 DataFrame 列名有时不一致（如 `limit` vs `limit_type`），需要兼容处理
- `moneyflow_hsgt` 的北向资金字段 hgt/sgt 是**字符串类型**，必须 `pd.to_numeric(errors="coerce")` 后再计算
- `moneyflow_hsgt` 的 hgt/sgt 单位是**万元**，转亿要除以10000（不是100！已踩坑修复）
- 申万行业指数用 `sw_daily` 接口，不是 `index_daily`
- 财务数据 `fina_indicator` 中 NaN 很常见，所有计算前必须 `fillna` 或检查
- 板块成分股用 `index_member`，权重股用 `index_weight`，别搞混
- concept + concept_detail 接口调用量大，尽量用 stock_basic.industry 替代
- 上证成交额在 `index_daily` 中单位待确认，注意万/亿换算
