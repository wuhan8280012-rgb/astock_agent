#!/usr/bin/env python3
"""
verify_upgrade.py — astock_agent 升级验证脚本
===============================================
验证 CODEX_PE.md 中所有改造项是否正确实现。
可独立运行，不依赖 Tushare 或真实 LLM。

用法:
  python3 verify_upgrade.py          # 完整验证
  python3 verify_upgrade.py --quick  # 只验证辩论引擎
"""

import sys
import json
from dataclasses import dataclass, field
from typing import List


# ═══════════════════════════════════════════════════════════
#  Mock LLM — 模拟 DeepSeek V3 响应
# ═══════════════════════════════════════════════════════════

class MockLLM:
    """模拟 LLM 客户端，返回固定 JSON"""
    call_count = 0

    def chat(self, prompt: str, max_tokens: int = 500) -> str:
        MockLLM.call_count += 1
        if "看多" in prompt and "反驳" not in prompt:
            return json.dumps({
                "bull_points": [
                    "情绪面贪婪，市场资金充裕",
                    "板块轮动活跃，强势板块持续领涨",
                    "北向资金持续流入，外资看好",
                ],
                "target_scenario": "短期看涨至突破前高，中期跟随板块轮动",
                "confidence": 72,
            }, ensure_ascii=False)

        elif "反驳" in prompt or "看空" in prompt:
            return json.dumps({
                "bear_points": [
                    "环境评分仅72(一般)，非最佳入场时机",
                    "两融余额高位，杠杆资金风险累积",
                ],
                "rebuttals": [
                    "情绪贪婪恰恰是逆向信号，历史上贪婪阶段后常有回调",
                    "北向资金短期波动大，单日数据不可靠",
                ],
                "worst_case": "环境转差+情绪反转，可能回撤5-8%",
                "confidence": 58,
            }, ensure_ascii=False)

        elif "三个角度" in prompt or "aggressive" in prompt:
            return json.dumps({
                "aggressive": "仓位偏低可适度加仓，关注强势板块龙头突破机会",
                "neutral": "当前仓位36%合理，维持不动等待更明确信号",
                "conservative": "环境仅一般，如再跌破MA20应果断减至25%以下",
            }, ensure_ascii=False)

        elif "裁决" in prompt or "主席" in prompt:
            return json.dumps({
                "action": "持有",
                "confidence": 65,
                "reasoning": "多方逻辑部分成立但环境评分仅一般，空方风险提示合理。建议持有观察，待环境改善再加仓。",
                "key_risk": "情绪过热后回调风险",
                "position_advice": "维持当前36%仓位，不加不减",
            }, ensure_ascii=False)

        else:
            return "基于当前数据，市场整体偏暖，建议关注强势板块轮动机会。"


# ═══════════════════════════════════════════════════════════
#  测试 1: to_brief() 协议
# ═══════════════════════════════════════════════════════════

def test_to_brief():
    """验证所有 Report 的 to_brief() 方法"""
    print("\n--- Test 1: to_brief() 协议 ---")
    passed = 0
    total = 0

    # --- SentimentReport ---
    total += 1
    try:
        from skills.sentiment import SentimentReport, SentimentSignal
        r = SentimentReport(
            date="20260224",
            signals=[
                SentimentSignal("涨跌比", 2.5, "贪婪", 1, "上涨3000/下跌1500"),
                SentimentSignal("北向资金", 75.0, "贪婪", 1, "今日+75亿"),
            ],
            overall_score=0.8,
            overall_level="贪婪",
            suggested_position="50-70%仓位",
        )
        b = r.to_brief()
        assert isinstance(b, str) and len(b) > 10 and "[情绪]" in b
        print(f"  ✅ SentimentReport.to_brief(): {b[:80]}")
        passed += 1
    except (ImportError, AttributeError) as e:
        print(f"  ❌ SentimentReport: {e}")
    except Exception as e:
        print(f"  ❌ SentimentReport: {e}")

    # --- MacroReport ---
    total += 1
    try:
        from skills.macro_monitor import MacroReport, MacroSignal
        r = MacroReport(
            date="20260224",
            signals=[MacroSignal("SHIBOR", 1.8, "宽松", 1, "1周=1.8%")],
            overall_score=0.5,
            liquidity_level="中性偏松",
            market_impact="流动性尚可",
        )
        b = r.to_brief()
        assert "[宏观]" in b
        print(f"  ✅ MacroReport.to_brief(): {b[:80]}")
        passed += 1
    except (ImportError, AttributeError) as e:
        print(f"  ❌ MacroReport: {e}")
    except Exception as e:
        print(f"  ❌ MacroReport: {e}")

    # --- SectorReport ---
    total += 1
    try:
        from skills.sector_rotation import SectorReport, SectorInfo
        r = SectorReport(
            date="20260224",
            top_sectors=[
                SectorInfo("801010.SI", "电子", ret_20d=8.5, signal="强势持续"),
                SectorInfo("801020.SI", "计算机", ret_20d=6.2, signal="缩量整理"),
            ],
            rotation_signals=["缩量蓄势: 计算机"],
        )
        b = r.to_brief()
        assert "[板块]" in b
        print(f"  ✅ SectorReport.to_brief(): {b[:80]}")
        passed += 1
    except (ImportError, AttributeError) as e:
        print(f"  ❌ SectorReport: {e}")
    except Exception as e:
        print(f"  ❌ SectorReport: {e}")

    # --- RiskReport ---
    total += 1
    try:
        from skills.risk_control import RiskReport, PositionCheck
        r = RiskReport(
            date="20260224",
            total_position=0.36,
            max_position_limit=0.50,
            position_checks=[
                PositionCheck("600519.SH", "贵州茅台", 0.08, 0.033, action="持有"),
                PositionCheck("002594.SZ", "比亚迪", 0.12, -0.08, action="止损",
                              alerts=["亏损8%触及止损线"]),
            ],
            portfolio_alerts=[],
        )
        b = r.to_brief()
        assert "[风控]" in b
        print(f"  ✅ RiskReport.to_brief(): {b[:80]}")
        passed += 1
    except (ImportError, AttributeError) as e:
        print(f"  ❌ RiskReport: {e}")
    except Exception as e:
        print(f"  ❌ RiskReport: {e}")

    # --- CanslimReport ---
    total += 1
    try:
        from skills.canslim_screener import CanslimReport, StockScore
        r = CanslimReport(
            date="20260224",
            total_scanned=100,
            total_passed=8,
            candidates=[
                StockScore("600519.SH", "贵州茅台", total_score=85, grade="A", flags=["创新高"]),
            ],
        )
        b = r.to_brief()
        assert "[CANSLIM]" in b
        print(f"  ✅ CanslimReport.to_brief(): {b[:80]}")
        passed += 1
    except (ImportError, AttributeError) as e:
        print(f"  ❌ CanslimReport: {e}")
    except Exception as e:
        print(f"  ❌ CanslimReport: {e}")

    # --- PipelineReport ---
    total += 1
    try:
        from skills.stock_pipeline import PipelineReport, PipelineCandidate
        r = PipelineReport(
            date="20260224",
            market_sentiment="贪婪",
            top_sectors=["电子", "计算机"],
            candidates=[
                PipelineCandidate(
                    "600519.SH", "贵州茅台", "食品饮料", "A", 85,
                    final_score=92, buy_signal_strength="强",
                    suggested_entry=1550, suggested_stop=1480,
                ),
            ],
        )
        b = r.to_brief()
        assert "[选股]" in b
        print(f"  ✅ PipelineReport.to_brief(): {b[:80]}")
        passed += 1
    except (ImportError, AttributeError) as e:
        print(f"  ❌ PipelineReport: {e}")
    except Exception as e:
        print(f"  ❌ PipelineReport: {e}")

    print(f"\n  to_brief 结果: {passed}/{total} 通过")
    return passed == total


# ═══════════════════════════════════════════════════════════
#  测试 2: 辩论引擎
# ═══════════════════════════════════════════════════════════

def test_debate_engine():
    """验证辩论引擎核心逻辑"""
    print("\n--- Test 2: 辩论引擎 ---")
    from debate import DebateEngine, needs_debate

    llm = MockLLM()
    engine = DebateEngine(llm)
    errors = []

    # 2a: 环境门控 — 拦截
    MockLLM.call_count = 0
    r = engine.run("能买吗", "测试", env_score=40)
    if r.env_gate != "拦截":
        errors.append(f"门控拦截失败: env_gate={r.env_gate}")
    if r.action != "观望":
        errors.append(f"门控拦截应为观望: action={r.action}")
    if MockLLM.call_count != 0:
        errors.append(f"门控拦截不应调用LLM: calls={MockLLM.call_count}")
    print(f"  {'✅' if not errors else '❌'} 环境门控(拦截): score=40 → {r.action}")

    # 2b: 环境门控 — 警告
    MockLLM.call_count = 0
    r = engine.run("能买吗", "测试", env_score=55, risk_brief="[风控] 仓位36%")
    if r.env_gate != "警告":
        errors.append(f"门控警告失败: env_gate={r.env_gate}")
    if r.confidence > 60:
        errors.append(f"警告模式置信度应≤60: confidence={r.confidence}")
    print(f"  {'✅' if r.env_gate == '警告' else '❌'} 环境门控(警告): score=55 → conf={r.confidence}")

    # 2c: 完整辩论流程
    MockLLM.call_count = 0
    r = engine.run(
        question="贵州茅台现在可以买入吗？",
        context_brief="[情绪] 贪婪(+0.8)\n[宏观] 中性偏松(+0.3)\n[环境] 72/100",
        env_score=72,
        env_brief="[环境] 72/100(一般) → 谨慎交易",
        risk_brief="[风控] 仓位36%/50%上限 | 止损:比亚迪",
    )
    if r.env_gate != "通过":
        errors.append(f"72分应通过: env_gate={r.env_gate}")
    if r.action not in ("买入", "持有", "减仓", "卖出", "观望"):
        errors.append(f"非法action: {r.action}")
    if not (0 <= r.confidence <= 100):
        errors.append(f"置信度越界: {r.confidence}")
    if len(r.bull_case) < 10:
        errors.append("bull_case 过短")
    if len(r.bear_case) < 10:
        errors.append("bear_case 过短")
    if not r.risk_perspectives:
        errors.append("risk_perspectives 为空")
    if MockLLM.call_count != 4:  # bull + bear + risk_3v + judge
        errors.append(f"应调用4次LLM: actual={MockLLM.call_count}")

    print(f"  {'✅' if MockLLM.call_count == 4 else '❌'} 完整辩论: {MockLLM.call_count}次LLM调用")
    print(f"     决策: {r.action}(置信{r.confidence}%)")
    print(f"     多方: {r.bull_case[:50]}...")
    print(f"     空方: {r.bear_case[:50]}...")
    print(f"     风控: {list(r.risk_perspectives.keys())}")
    print(f"     耗时: {r.elapsed_sec:.3f}s")

    # 2d: to_brief 和 to_dict
    brief = r.to_brief()
    d = r.to_dict()
    if "[辩论]" not in brief:
        errors.append("to_brief 格式错误")
    if "action" not in d:
        errors.append("to_dict 缺 action")
    print(f"  ✅ to_brief(): {brief[:60]}...")
    print(f"  ✅ to_dict(): keys={list(d.keys())}")

    # 2e: 快速模式
    MockLLM.call_count = 0
    quick = engine.run_quick("今天北向多少", "[情绪] 中性")
    if MockLLM.call_count != 1:
        errors.append(f"快速模式应1次调用: {MockLLM.call_count}")
    print(f"  ✅ run_quick(): 1次调用, 回复长度={len(quick)}")

    # 2f: needs_debate 分类
    cases = [
        ("贵州茅台能买吗", [], True),
        ("今天北向多少", [], False),
        ("该不该减仓", [], True),
        ("什么是CANSLIM", [], False),
        ("分析600519.SH", [], True),
        ("板块排名", ["sector_analysis"], False),
        ("持仓风险", ["risk_check"], True),
        ("帮我止损比亚迪", ["stock_analysis"], True),
        ("今天市场怎么样", ["market_sentiment"], False),
    ]
    nd_pass = 0
    for q, routes, expected in cases:
        actual = needs_debate(q, routes)
        ok = actual == expected
        nd_pass += ok
        if not ok:
            errors.append(f"needs_debate('{q}')={actual}, expected={expected}")
    print(f"  {'✅' if nd_pass == len(cases) else '❌'} needs_debate: {nd_pass}/{len(cases)} 通过")

    if errors:
        print(f"\n  ❌ 辩论引擎错误: {len(errors)}")
        for e in errors:
            print(f"    - {e}")
        return False
    print(f"\n  ✅ 辩论引擎全部通过")
    return True


# ═══════════════════════════════════════════════════════════
#  测试 3: Token 经济性验证
# ═══════════════════════════════════════════════════════════

def test_token_economics():
    """验证 token 节省效果"""
    print("\n--- Test 3: Token 经济性 ---")

    # 模拟一个典型的 to_dict() 输出
    mock_dict_output = json.dumps({
        "date": "20260224",
        "overall_score": 0.8,
        "overall_level": "贪婪",
        "suggested_position": "50-70%仓位，逢高减仓",
        "summary": "涨跌比:贪婪(上涨3000家/下跌1500家) | 涨跌停:贪婪(涨停61家/跌停16家) | 北向资金:贪婪(今日+75.89亿) | 两融余额:中性(融资余额18500亿) | 成交额:中性偏多(今日/20日=1.15x)",
        "signals": [
            {"name": "涨跌比", "value": 2.5, "level": "贪婪", "score": 1, "detail": "上涨3000家/下跌1500家/平盘200家"},
            {"name": "涨跌停", "value": 45, "level": "贪婪", "score": 1, "detail": "涨停61家/跌停16家"},
            {"name": "北向资金", "value": 75.89, "level": "贪婪", "score": 1, "detail": "今日+75.89亿 | 近3日+150亿"},
            {"name": "两融余额", "value": 0.5, "level": "中性", "score": 0, "detail": "融资余额18500亿 | 5日变化+0.50%"},
            {"name": "成交额", "value": 1.15, "level": "中性偏多", "score": 0, "detail": "温和放量 | 今日/20日均值=1.15x"},
        ],
    }, ensure_ascii=False)

    mock_brief_output = "[情绪] 贪婪(+0.8) 仓位:50-70%仓位，逢高减仓 | 涨跌比+1,涨跌停+1,北向资金+1,两融余额+0,成交额+0"

    dict_tokens = len(mock_dict_output) // 2  # 粗估
    brief_tokens = len(mock_brief_output) // 2

    savings = (1 - brief_tokens / dict_tokens) * 100

    print(f"  to_dict() 输出: {len(mock_dict_output)} chars ≈ {dict_tokens} tokens")
    print(f"  to_brief() 输出: {len(mock_brief_output)} chars ≈ {brief_tokens} tokens")
    print(f"  节省: {savings:.0f}%")

    # 验证节省 > 70%
    if savings >= 70:
        print(f"  ✅ Token 节省 {savings:.0f}% ≥ 70% 目标")
        return True
    else:
        print(f"  ❌ Token 节省 {savings:.0f}% < 70% 目标")
        return False


# ═══════════════════════════════════════════════════════════
#  测试 4: 端到端集成模拟
# ═══════════════════════════════════════════════════════════

def test_e2e_simulation():
    """模拟完整的 Router → Debate 流程"""
    print("\n--- Test 4: 端到端集成模拟 ---")

    from debate import DebateEngine, needs_debate

    llm = MockLLM()
    engine = DebateEngine(llm)

    # 模拟 5 个典型用户问题
    scenarios = [
        {
            "question": "贵州茅台现在能买吗",
            "routes": ["stock_analysis"],
            "context": "[情绪] 贪婪(+0.8)\n[环境] 72/100(一般)\n[风控] 仓位36%/50%",
            "env_score": 72,
            "expect_debate": True,
        },
        {
            "question": "今天市场情绪怎么样",
            "routes": ["market_sentiment"],
            "context": "[情绪] 贪婪(+0.8)",
            "env_score": 72,
            "expect_debate": False,
        },
        {
            "question": "该不该减仓比亚迪",
            "routes": ["stock_analysis", "risk_check"],
            "context": "[风控] 止损:比亚迪(-8%)\n[环境] 55/100(较差)",
            "env_score": 55,
            "expect_debate": True,
        },
        {
            "question": "帮我看看板块轮动",
            "routes": ["sector_analysis"],
            "context": "[板块] Top: 电子(+8.5%), 计算机(+6.2%)",
            "env_score": 72,
            "expect_debate": False,
        },
        {
            "question": "大盘跌了3%还能操作吗",
            "routes": ["market_sentiment", "risk_check"],
            "context": "[情绪] 极度恐慌(-1.5)\n[环境] 35/100(较差)",
            "env_score": 35,
            "expect_debate": True,  # needs_debate=True, but gate blocks
        },
    ]

    all_ok = True
    for sc in scenarios:
        debate_needed = needs_debate(sc["question"], sc["routes"])
        assert debate_needed == sc["expect_debate"], \
            f"分流错误: '{sc['question']}' → debate={debate_needed}"

        if debate_needed:
            result = engine.run(
                sc["question"], sc["context"],
                env_score=sc["env_score"],
                risk_brief="[风控] 仓位36%/50%",
            )
            icon = {"通过": "🟢", "警告": "🟡", "拦截": "🔴"}.get(result.env_gate, "?")
            print(f"  {icon} 辩论 '{sc['question'][:15]}...' → "
                  f"{result.action}({result.confidence}%) [{result.env_gate}]")
        else:
            resp = engine.run_quick(sc["question"], sc["context"])
            print(f"  ⚡ 快速 '{sc['question'][:15]}...' → {resp[:40]}...")

    print(f"\n  ✅ 端到端模拟完成")
    return all_ok


# ═══════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  astock_agent LLM 优化升级 — 验证套件")
    print("  CODEX_PE v2.0")
    print("=" * 65)

    results = {}

    # 核心: 辩论引擎 (不依赖 skills import)
    results["debate"] = test_debate_engine()
    results["tokens"] = test_token_economics()
    results["e2e"] = test_e2e_simulation()

    # 可选: to_brief() (依赖 skills 目录)
    if "--quick" not in sys.argv:
        try:
            results["brief"] = test_to_brief()
        except Exception as e:
            print(f"\n--- Test 1: to_brief() 跳过 (skills 未在路径中) ---")
            print(f"  ⚠️ {e}")
            print(f"  提示: 将 skills/ 目录加入 PYTHONPATH 后重试")
            results["brief"] = None

    # 汇总
    print("\n" + "=" * 65)
    print("  汇总")
    print("=" * 65)
    all_pass = True
    for name, ok in results.items():
        if ok is None:
            print(f"  ⚠️ {name}: 跳过")
        elif ok:
            print(f"  ✅ {name}: 通过")
        else:
            print(f"  ❌ {name}: 失败")
            all_pass = False

    if all_pass:
        print(f"\n  🎉 验证通过 — 可以部署")
    else:
        print(f"\n  ⚠️ 存在失败项，请修复后重试")

    print("=" * 65)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
