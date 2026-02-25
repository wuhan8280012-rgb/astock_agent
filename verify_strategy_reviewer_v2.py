#!/usr/bin/env python3
"""
verify_strategy_reviewer_v2.py — 复盘引擎 v2 验证套件
"""

import sys, types, json, os
from datetime import datetime, timedelta
from pathlib import Path

# Mock 依赖
mock_fetcher_mod = types.ModuleType("data_fetcher")
mock_fetcher_mod.get_fetcher = lambda: None
sys.modules["data_fetcher"] = mock_fetcher_mod
mock_cfg = types.ModuleType("config")
mock_cfg.SKILL_PARAMS = {}
sys.modules["config"] = mock_cfg

# 确保 knowledge 目录干净
KNOWLEDGE_DIR = Path("./knowledge")
KNOWLEDGE_DIR.mkdir(exist_ok=True)
for f in KNOWLEDGE_DIR.glob("*.json"):
    f.unlink()

from strategy_reviewer_v2 import (
    StrategyReviewerV2, SignalOutcome, DebateOutcome,
    SignalScorecard, ParameterProposal, SIGNAL_LOG, DEBATE_LOG,
    PROPOSAL_LOG, SAFETY,
)
from dataclasses import asdict


# ════════════════════════════════════════════════════════════
#  辅助: 批量注入模拟数据
# ════════════════════════════════════════════════════════════

def inject_signal_data(n=30, win_rate=0.5, signal_types=None):
    """注入N条已有结果的信号记录"""
    import numpy as np
    np.random.seed(42)

    if signal_types is None:
        signal_types = ["阶梯突破", "EMA突破", "放量阳线", "缺口突破"]

    records = []
    for i in range(n):
        stype = signal_types[i % len(signal_types)]
        is_win = np.random.random() < win_rate

        # 胜率按信号类型分化: 阶梯突破最好, 缺口最差
        type_adj = {"阶梯突破": 0.1, "EMA突破": 0.05, "放量阳线": -0.05, "缺口突破": -0.15}
        is_win = np.random.random() < (win_rate + type_adj.get(stype, 0))

        ret_5d = round(np.random.uniform(1, 12) if is_win else np.random.uniform(-10, -1), 2)
        hit_stop = ret_5d < -6

        date = (datetime.now() - timedelta(days=np.random.randint(5, 25))).strftime("%Y%m%d")

        records.append(asdict(SignalOutcome(
            date=date,
            ts_code=f"{600000+i}.SH",
            name=f"测试股{i}",
            signal_type=stype,
            action="BUY",
            entry_price=round(20 + np.random.randn() * 3, 2),
            stop_price=round(18 + np.random.randn() * 2, 2),
            target_price=round(26 + np.random.randn() * 3, 2),
            sector=["电子", "计算机", "军工", "银行"][i % 4],
            ret_5d=ret_5d,
            ret_1d=round(ret_5d * 0.3, 2),
            ret_3d=round(ret_5d * 0.7, 2),
            max_drawdown=round(min(ret_5d, -2), 2),
            max_gain=round(max(ret_5d, 3), 2),
            hit_stop=hit_stop,
            hit_target=ret_5d > 8,
            outcome="止损" if hit_stop else ("盈利" if ret_5d > 2 else "平淡"),
        )))

    with open(SIGNAL_LOG, "w") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return records


def inject_debate_data(n=15):
    """注入辩论决策记录"""
    import numpy as np
    np.random.seed(99)

    records = []
    for i in range(n):
        action = np.random.choice(["买入", "观望", "卖出"])
        date = (datetime.now() - timedelta(days=np.random.randint(5, 20))).strftime("%Y%m%d")
        records.append(asdict(DebateOutcome(
            date=date,
            question=f"测试问题{i}",
            ts_code=f"{600000+i}.SH",
            action=action,
            confidence=np.random.randint(50, 90),
            env_score=np.random.randint(55, 85),
            env_gate="通过" if np.random.random() > 0.2 else "拦截",
        )))

    with open(DEBATE_LOG, "w") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return records


# ════════════════════════════════════════════════════════════
#  Tests
# ════════════════════════════════════════════════════════════

def test_signal_logging():
    print("\n--- Test 1: 信号记录 ---")
    # 清空
    if SIGNAL_LOG.exists():
        SIGNAL_LOG.unlink()

    reviewer = StrategyReviewerV2()

    signals = [
        {"ts_code": "002415.SZ", "name": "海康威视", "signal_type": "阶梯突破",
         "action": "BUY", "current_price": 35.8, "stop_price": 33.0,
         "target_price": 42.5, "position_size_pct": 0.06, "sector": "电子"},
        {"ts_code": "002594.SZ", "name": "比亚迪", "signal_type": "止损",
         "action": "SELL", "current_price": 215.0, "sector": "汽车"},
    ]

    reviewer.log_signal_outcomes(signals)
    records = reviewer._load(SIGNAL_LOG)

    assert len(records) == 2, f"应有2条记录: {len(records)}"
    assert records[0]["signal_type"] == "阶梯突破"
    assert records[1]["action"] == "SELL"
    print(f"  ✅ 记录信号: {len(records)}条")

    return True


def test_debate_logging():
    print("\n--- Test 2: 辩论记录 ---")
    if DEBATE_LOG.exists():
        DEBATE_LOG.unlink()

    reviewer = StrategyReviewerV2()

    reviewer.log_debate_outcome({
        "question": "海康威视触发阶梯突破，是否买入",
        "ts_code": "002415.SZ",
        "action": "买入",
        "confidence": 72,
        "env_score": 75,
        "env_gate": "通过",
    })

    records = reviewer._load(DEBATE_LOG)
    assert len(records) == 1
    assert records[0]["action"] == "买入"
    print(f"  ✅ 记录辩论: {len(records)}条")

    return True


def test_scorecards():
    print("\n--- Test 3: 信号记分卡 ---")

    # 注入30条信号数据, 整体胜率约50%
    inject_signal_data(n=30, win_rate=0.50)

    reviewer = StrategyReviewerV2()
    scorecards = reviewer.compute_scorecards(lookback_days=30)

    assert "全部信号" in scorecards, "应有全部信号汇总"
    all_sc = scorecards["全部信号"]
    assert all_sc.total >= 20, f"样本应≥20: {all_sc.total}"
    assert 0 < all_sc.win_rate < 1, f"胜率异常: {all_sc.win_rate}"

    print(f"  ✅ 全部信号: {all_sc.to_brief()}")

    # 按类型应有分化
    types_found = [k for k in scorecards if k != "全部信号"]
    assert len(types_found) >= 3, f"信号类型太少: {types_found}"

    for name, sc in scorecards.items():
        if name == "全部信号":
            continue
        print(f"  ✅ {sc.to_brief()}")

    return True


def test_debate_accuracy():
    print("\n--- Test 4: 辩论正确率 ---")

    # 注入信号+辩论数据
    inject_signal_data(n=30, win_rate=0.50)
    inject_debate_data(n=15)

    reviewer = StrategyReviewerV2()
    stats = reviewer.compute_debate_accuracy(lookback_days=30)

    assert "total_debates" in stats
    assert stats["total_debates"] == 15
    print(f"  ✅ 辩论统计: {stats}")

    return True


def test_rule_based_proposals():
    print("\n--- Test 5: 规则引擎提案 ---")

    # 注入低胜率数据 → 应触发提案
    inject_signal_data(n=40, win_rate=0.30)

    reviewer = StrategyReviewerV2()
    scorecards = reviewer.compute_scorecards()

    current_params = {
        "trade_signal": {
            "breakout_vol_ratio": 1.5,
            "min_yang_pct": 2.0,
            "min_rr_ratio": 3.0,
        }
    }

    proposals = reviewer._generate_rule_based_proposals(scorecards, current_params)

    print(f"  提案数: {len(proposals)}")
    for p in proposals:
        print(f"  📋 {p['module']}.{p['param_name']}: "
              f"{p['current_value']} → {p['proposed_value']} ({p['change_pct']:+.0f}%)")
        print(f"     理由: {p['reason']}")

        # 安全检查
        assert abs(p["change_pct"]) <= SAFETY["max_param_change_pct"] * 100 + 1, \
            f"变更幅度超限: {p['change_pct']}%"

    assert len(proposals) <= SAFETY["max_proposals_per_review"], \
        f"提案数超限: {len(proposals)}"

    print(f"  ✅ 提案合规: 数量≤{SAFETY['max_proposals_per_review']}, 幅度≤±{SAFETY['max_param_change_pct']*100:.0f}%")

    return True


def test_no_proposal_when_insufficient():
    print("\n--- Test 6: 样本不足时不提案 ---")

    # 只注入5条 → 不应产生提案
    inject_signal_data(n=5, win_rate=0.20)

    reviewer = StrategyReviewerV2()
    scorecards = reviewer.compute_scorecards()
    proposals = reviewer._generate_rule_based_proposals(scorecards, {"trade_signal": {}})

    assert len(proposals) == 0, f"样本不足不应有提案: {len(proposals)}"
    print(f"  ✅ 样本{scorecards.get('全部信号', SignalScorecard('x')).total}条 < {SAFETY['min_samples_for_proposal']} → 提案数=0")

    return True


def test_approval_workflow():
    print("\n--- Test 7: 审批流程 ---")

    # 清空提案
    if PROPOSAL_LOG.exists():
        PROPOSAL_LOG.unlink()

    reviewer = StrategyReviewerV2()

    # 手动写入一个待审批提案
    proposal = asdict(ParameterProposal(
        id="20260224_trade_signal_breakout_vol_ratio",
        date="20260224",
        module="trade_signal",
        param_name="breakout_vol_ratio",
        current_value=1.5,
        proposed_value=1.8,
        change_pct=20.0,
        reason="阶梯突破胜率偏低",
        evidence="胜率35%, N=25",
        confidence=0.65,
        status="pending",
        rollback_date=(datetime.now() + timedelta(days=30)).strftime("%Y%m%d"),
    ))
    reviewer._save(PROPOSAL_LOG, [proposal])

    # 查看待审批
    pending = reviewer.get_pending_proposals()
    assert len(pending) == 1
    print(f"  ✅ 待审批: {len(pending)}个")

    # 批准
    ok = reviewer.approve_proposal(proposal["id"], approver="wuhan")
    assert ok
    proposals = reviewer._load(PROPOSAL_LOG)
    assert proposals[0]["status"] == "approved"
    assert proposals[0]["approved_by"] == "wuhan"
    print(f"  ✅ 批准成功: status={proposals[0]['status']}")

    # 回滚
    ok = reviewer.rollback_proposal(proposal["id"])
    assert ok
    proposals = reviewer._load(PROPOSAL_LOG)
    assert proposals[0]["status"] == "rolled_back"
    print(f"  ✅ 回滚成功: status={proposals[0]['status']}")

    return True


def test_reject_workflow():
    print("\n--- Test 8: 驳回流程 ---")

    if PROPOSAL_LOG.exists():
        PROPOSAL_LOG.unlink()

    reviewer = StrategyReviewerV2()

    proposal = asdict(ParameterProposal(
        id="20260224_test_reject",
        date="20260224",
        module="trade_signal",
        param_name="min_rr_ratio",
        current_value=3.0,
        proposed_value=3.5,
        change_pct=16.7,
        reason="止损率偏高",
        evidence="止损率42%",
        confidence=0.5,
        status="pending",
        rollback_date="20260324",
    ))
    reviewer._save(PROPOSAL_LOG, [proposal])

    ok = reviewer.reject_proposal("20260224_test_reject", reason="样本期太短，再观察两周")
    assert ok
    proposals = reviewer._load(PROPOSAL_LOG)
    assert proposals[0]["status"] == "rejected"
    assert proposals[0]["reject_reason"] == "样本期太短，再观察两周"
    print(f"  ✅ 驳回成功: reason='{proposals[0]['reject_reason']}'")

    return True


def test_to_brief():
    print("\n--- Test 9: to_brief 压缩输出 ---")

    inject_signal_data(n=30, win_rate=0.55)

    reviewer = StrategyReviewerV2()
    brief = reviewer.to_brief()

    assert "[复盘]" in brief
    assert "胜率" in brief
    print(f"  ✅ to_brief ({len(brief)} chars):")
    print(f"  {brief}")

    return True


def test_weekly_review_no_llm():
    print("\n--- Test 10: 完整周报 (无LLM) ---")

    inject_signal_data(n=40, win_rate=0.45)
    inject_debate_data(n=15)

    reviewer = StrategyReviewerV2(llm_client=None)

    current_params = {
        "trade_signal": {
            "breakout_vol_ratio": 1.5,
            "min_yang_pct": 2.0,
            "min_rr_ratio": 3.0,
            "stop_loss_pct": 0.07,
        }
    }

    review = reviewer.run_weekly_review(
        current_params=current_params,
        verbose=True,
    )

    assert "summary" in review
    assert "proposals" in review
    print(f"\n  ✅ 周报完成, 提案{len(review.get('proposals', []))}个")

    return True


def test_safety_bounds():
    print("\n--- Test 11: 安全边界 ---")

    reviewer = StrategyReviewerV2()

    # 测试参数变更幅度限制
    proposal = reviewer._make_proposal(
        "trade_signal", "breakout_vol_ratio",
        current=1.5, proposed=3.0,  # 试图翻倍，应被截断到 +30%
        reason="test", evidence="test",
    )
    assert proposal.proposed_value <= 1.5 * 1.31, \
        f"应被截断: {proposal.proposed_value} > {1.5 * 1.3}"
    assert proposal.change_pct <= 30.1
    print(f"  ✅ 幅度截断: 1.5→3.0 被限制为 1.5→{proposal.proposed_value} ({proposal.change_pct:.0f}%)")

    # 测试向下调整也被限制
    proposal = reviewer._make_proposal(
        "trade_signal", "min_rr_ratio",
        current=3.0, proposed=1.0,  # 试图砍到1/3，应被截断
        reason="test", evidence="test",
    )
    assert proposal.proposed_value >= 3.0 * 0.69
    print(f"  ✅ 向下截断: 3.0→1.0 被限制为 3.0→{proposal.proposed_value} ({proposal.change_pct:.0f}%)")

    print(f"  ✅ 安全边界: max_change=±{SAFETY['max_param_change_pct']*100:.0f}%, "
          f"min_samples={SAFETY['min_samples_for_proposal']}, "
          f"max_proposals={SAFETY['max_proposals_per_review']}")

    return True


# ════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  策略复盘引擎 v2 — 验证套件")
    print("=" * 65)

    tests = [
        ("信号记录", test_signal_logging),
        ("辩论记录", test_debate_logging),
        ("信号记分卡", test_scorecards),
        ("辩论正确率", test_debate_accuracy),
        ("规则引擎提案", test_rule_based_proposals),
        ("样本不足不提案", test_no_proposal_when_insufficient),
        ("审批流程", test_approval_workflow),
        ("驳回流程", test_reject_workflow),
        ("to_brief", test_to_brief),
        ("完整周报(无LLM)", test_weekly_review_no_llm),
        ("安全边界", test_safety_bounds),
    ]

    passed = 0
    for name, fn in tests:
        try:
            if fn():
                passed += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            import traceback
            traceback.print_exc()

    # 清理
    for f in KNOWLEDGE_DIR.glob("*.json"):
        f.unlink()
    if KNOWLEDGE_DIR.exists():
        try:
            KNOWLEDGE_DIR.rmdir()
        except OSError:
            pass

    print(f"\n{'='*65}")
    print(f"  结果: {passed}/{len(tests)} 通过")
    if passed == len(tests):
        print(f"  🎉 复盘引擎 v2 验证全部通过")
    else:
        print(f"  ⚠️ 有未通过项")
    print(f"{'='*65}")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
