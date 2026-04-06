#!/usr/bin/env bash
# ============================================================
# 竣工验收脚本 — 产品经理对 Claude Code 交付物的自动检查
# 用法：cd project_root && bash acceptance_check.sh
# ============================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS=0
FAIL=0
WARN=0

check() {
    local label="$1"
    local result="$2"  # 0=pass, 1=fail, 2=warn
    local detail="${3:-}"
    if [ "$result" -eq 0 ]; then
        echo -e "  ${GREEN}✅ PASS${NC}  $label"
        PASS=$((PASS + 1))
    elif [ "$result" -eq 2 ]; then
        echo -e "  ${YELLOW}⚠️  WARN${NC}  $label  — $detail"
        WARN=$((WARN + 1))
    else
        echo -e "  ${RED}❌ FAIL${NC}  $label  — $detail"
        FAIL=$((FAIL + 1))
    fi
}

echo ""
echo "==========================================="
echo " 竣工验收检查"
echo "==========================================="
echo ""

# ── 一、文件存在性 ──────────────────────────────
echo "【1/7】文件清单对账"

REQUIRED_FILES=(
    "config/settings.py"
    "db/schema.sql"
    "db/repository.py"
    "data_pipeline/clock.py"
    "data_pipeline/tushare_client.py"
    "data_pipeline/universe_filter.py"
    "data_pipeline/agents/sector_agent.py"
    "data_pipeline/agents/capital_agent.py"
    "data_pipeline/agents/catalyst_agent.py"
    "data_pipeline/agents/structure_agent.py"
    "data_pipeline/agents/liquidity_agent.py"
    "decision_engine/macro_switch.py"
    "decision_engine/risk_veto.py"
    "decision_engine/scorer.py"
    "decision_engine/buy_agent.py"
    "decision_engine/position_agent.py"
    "decision_engine/sell_agent.py"
    "meta_cognitive/trigger.py"
    "meta_cognitive/advocate.py"
    "meta_cognitive/challenger.py"
    "meta_cognitive/arbitrator.py"
    "meta_cognitive/param_controller.py"
    "meta_cognitive/self_audit.py"
    "pool_manager/pool.py"
    "scheduler/main_scheduler.py"
    "scripts/bootstrap.py"
    "parameter_versions/v1.0.json"
    "main.py"
)

for f in "${REQUIRED_FILES[@]}"; do
    if [ -f "$f" ]; then
        check "$f" 0
    else
        check "$f" 1 "文件缺失"
    fi
done

echo ""

# ── 二、纪律审计（grep） ─────────────────────────
echo "【2/7】硬性约束 grep 审计"

# 2a: SQL 只能出现在 repository.py
SQL_LEAK=$(grep -rn --exclude-dir=.venv --exclude-dir=__pycache__ "\.execute\|\.executemany\|cursor\." --include="*.py" \
    | grep -v "db/repository.py" \
    | grep -v "test" \
    | grep -v "schema.sql" || true)
if [ -z "$SQL_LEAK" ]; then
    check "SQL 语句仅在 repository.py 中" 0
else
    check "SQL 语句仅在 repository.py 中" 1 "发现泄漏:"
    echo "$SQL_LEAK" | head -10 | sed 's/^/        /'
fi

# 2b: 模型名只能出现在 settings.py
MODEL_LEAK=$(grep -rn --exclude-dir=.venv --exclude-dir=__pycache__ "claude-opus\|claude-haiku\|claude-sonnet" --include="*.py" \
    | grep -v "config/settings.py" \
    | grep -v "test" || true)
if [ -z "$MODEL_LEAK" ]; then
    check "模型名仅在 settings.py 中" 0
else
    check "模型名仅在 settings.py 中" 1 "发现硬编码:"
    echo "$MODEL_LEAK" | head -10 | sed 's/^/        /'
fi

# 2c: temperature 必须为 0
TEMP_BAD=$(grep -rn --exclude-dir=.venv --exclude-dir=__pycache__ "temperature" --include="*.py" \
    | grep -v "temperature=0" \
    | grep -v "temperature = 0" \
    | grep -v "test" \
    | grep -v "#" || true)
if [ -z "$TEMP_BAD" ]; then
    check "LLM temperature 全部为 0" 0
else
    check "LLM temperature 全部为 0" 1 "发现非零 temperature:"
    echo "$TEMP_BAD" | head -5 | sed 's/^/        /'
fi

# 2d: tushare 直接调用只在 tushare_client.py
TS_LEAK=$(grep -rn --exclude-dir=.venv --exclude-dir=__pycache__ "pro_api\|ts\.pro_api\|tushare\.pro_api" --include="*.py" \
    | grep -v "tushare_client.py" \
    | grep -v "test" || true)
if [ -z "$TS_LEAK" ]; then
    check "Tushare 调用仅在 tushare_client.py 中" 0
else
    check "Tushare 调用仅在 tushare_client.py 中" 1 "发现直接调用:"
    echo "$TS_LEAK" | head -5 | sed 's/^/        /'
fi

# 2e: 买入门槛用 >= 不用 >
BUY_GT=$(grep -n "composite_score.*>.*75\|composite_score.*> 75" \
    decision_engine/buy_agent.py decision_engine/scorer.py 2>/dev/null \
    | grep -v ">=" || true)
if [ -z "$BUY_GT" ]; then
    check "买入门槛使用 >= 75（非 > 75）" 0
else
    check "买入门槛使用 >= 75（非 > 75）" 1 "发现 > 75:"
    echo "$BUY_GT" | sed 's/^/        /'
fi

echo ""

# ── 三、clock.py 时间门控检查 ─────────────────────
echo "【3/7】clock.py 数据类型覆盖"

if [ -f "data_pipeline/clock.py" ]; then
    EXPECTED_TYPES=("daily_price" "margin_balance" "top_list" "announcement" "etf_flow" "index_daily")
    for dtype in "${EXPECTED_TYPES[@]}"; do
        if grep -q "$dtype" data_pipeline/clock.py; then
            check "clock.py 包含 $dtype" 0
        else
            check "clock.py 包含 $dtype" 1 "未找到该数据类型"
        fi
    done
else
    check "clock.py 存在" 1 "文件缺失，跳过子检查"
fi

echo ""

# ── 四、Prompt 完整性 ─────────────────────────────
echo "【4/7】LLM 模块 Prompt 完整性"

LLM_MODULES=(
    "decision_engine/buy_agent.py"
    "decision_engine/position_agent.py"
    "meta_cognitive/advocate.py"
    "meta_cognitive/challenger.py"
    "meta_cognitive/arbitrator.py"
)

for mod in "${LLM_MODULES[@]}"; do
    if [ -f "$mod" ]; then
        HAS_SYSTEM=$(grep -c "SYSTEM_PROMPT\|system_prompt\|SYSTEM_MESSAGE" "$mod" || true)
        HAS_USER=$(grep -c "USER_PROMPT\|user_prompt\|USER_MESSAGE\|user_message" "$mod" || true)
        if [ "$HAS_SYSTEM" -gt 0 ] && [ "$HAS_USER" -gt 0 ]; then
            check "$mod 包含 System + User Prompt" 0
        elif [ "$HAS_SYSTEM" -gt 0 ]; then
            check "$mod Prompt" 1 "有 System Prompt 但缺 User Prompt"
        elif [ "$HAS_USER" -gt 0 ]; then
            check "$mod Prompt" 1 "有 User Prompt 但缺 System Prompt"
        else
            check "$mod Prompt" 1 "System 和 User Prompt 均未找到"
        fi
    else
        check "$mod" 1 "文件缺失"
    fi
done

echo ""

# ── 五、测试 ──────────────────────────────────────
echo "【5/7】测试执行"

if [ -d "tests" ]; then
    TEST_COUNT=$(find tests -name "test_*.py" -o -name "*_test.py" | wc -l)
    check "测试文件数量: $TEST_COUNT" 0

    echo "  运行 pytest..."
    echo ""
    if python3 -m pytest tests/ -v --tb=short 2>&1 | tee /tmp/pytest_output.txt; then
        PYTEST_EXIT=0
    else
        PYTEST_EXIT=1
    fi
    echo ""

    TOTAL=$(grep -oE '[0-9]+ passed' /tmp/pytest_output.txt | grep -oE '[0-9]+' || echo "0")
    FAILED=$(grep -oE '[0-9]+ failed' /tmp/pytest_output.txt | grep -oE '[0-9]+' || echo "0")

    if [ "$PYTEST_EXIT" -eq 0 ]; then
        check "pytest 全部通过 ($TOTAL passed)" 0
    else
        check "pytest 有失败 ($FAILED failed / $TOTAL passed)" 1 "见上方详细输出"
    fi
else
    check "tests/ 目录存在" 1 "目录缺失"
fi

echo ""

# ── 六、v1.0.json 参数完整性 ──────────────────────
echo "【6/7】v1.0.json 参数完整性"

if [ -f "parameter_versions/v1.0.json" ]; then
    REQUIRED_KEYS=("weights" "thresholds" "agent_params" "tunable_bounds" "buy_signal_min_score" "sell_score_collapse" "rr_ratio_min")
    for key in "${REQUIRED_KEYS[@]}"; do
        if grep -q "\"$key\"" parameter_versions/v1.0.json; then
            check "v1.0.json 包含 $key" 0
        else
            check "v1.0.json 包含 $key" 1 "缺失"
        fi
    done

    # 检查权重和是否为 1.0
    WEIGHT_SUM=$(python3 -c "
import json
with open('parameter_versions/v1.0.json') as f:
    d = json.load(f)
w = d.get('weights', {})
s = sum(w.values())
print(f'{s:.4f}')
" 2>/dev/null || echo "ERROR")

    if [ "$WEIGHT_SUM" = "ERROR" ]; then
        check "weights 权重和 = 1.0" 1 "JSON 解析失败"
    elif [ "$WEIGHT_SUM" = "1.0000" ]; then
        check "weights 权重和 = 1.0 (实际: $WEIGHT_SUM)" 0
    else
        check "weights 权重和 = 1.0 (实际: $WEIGHT_SUM)" 1 "权重和不为 1"
    fi
else
    check "v1.0.json 存在" 1 "文件缺失"
fi

echo ""

# ── 七、调度编排 14 步函数调用追踪 ─────────────────
echo "【7/7】调度编排关键步骤追踪"

if [ -f "scheduler/main_scheduler.py" ]; then
    SCHEDULE_KEYWORDS=(
        "macro_switch|macro_cache"
        "universe_filter"
        "gather|agent.*run|agent.*analyze"
        "scorer|aggregate"
        "risk_veto|apply_veto"
        "buy_agent|buy.*decide"
        "check_admission|pool.*admission"
        "sell_agent|check_sell|check_all"
        "position_agent|review_all"
        "trigger|check_all|meta_cognitive"
    )
    SCHEDULE_LABELS=(
        "宏观开关检查"
        "universe_filter 调用"
        "Agent 并行运行"
        "scorer 聚合评分"
        "risk_veto 否决"
        "buy_agent 决策"
        "pool 入池检查"
        "sell_agent 卖出检查"
        "position_agent 持仓评估"
        "L3 复盘触发"
    )

    for i in "${!SCHEDULE_KEYWORDS[@]}"; do
        if grep -qE "${SCHEDULE_KEYWORDS[$i]}" scheduler/main_scheduler.py; then
            check "编排包含: ${SCHEDULE_LABELS[$i]}" 0
        else
            check "编排包含: ${SCHEDULE_LABELS[$i]}" 1 "未在 main_scheduler.py 中找到"
        fi
    done
else
    check "main_scheduler.py 存在" 1 "文件缺失"
fi

echo ""

# ── 汇总 ──────────────────────────────────────────
echo "==========================================="
echo -e " 汇总：${GREEN}$PASS 通过${NC} | ${RED}$FAIL 失败${NC} | ${YELLOW}$WARN 警告${NC}"
echo "==========================================="
echo ""

if [ "$FAIL" -gt 0 ]; then
    echo -e "${RED}❌ 验收不通过，有 $FAIL 项失败需要修复${NC}"
    exit 1
else
    echo -e "${GREEN}✅ 验收通过${NC}"
    exit 0
fi
