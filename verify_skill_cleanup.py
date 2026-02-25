"""
verify_skill_cleanup.py — Skill 清理验证
"""


def test_sentiment_slim():
    """验证 Sentiment 移除了两融和成交额"""
    from skills.sentiment import SentimentSkill

    class MockFetcher:
        def get_latest_trade_date(self):
            return "20260224"

        def get_market_breadth(self):
            return {"ratio": 2.0, "up": 3000, "down": 1500, "flat": 200}

        def get_limit_list(self):
            import pandas as pd
            return pd.DataFrame({"limit": ["U"] * 40 + ["D"] * 15})

        def get_north_flow(self, days=10):
            import pandas as pd
            return pd.DataFrame({"north_money_yi": [50.0, 60.0, 75.0]})

    skill = SentimentSkill()
    skill.fetcher = MockFetcher()
    report = skill.analyze()

    signal_names = [s.name for s in report.signals]
    assert "两融余额" not in signal_names, f"两融余额应已移除! 当前信号: {signal_names}"
    assert "成交额" not in signal_names, f"成交额应已移除! 当前信号: {signal_names}"
    assert "涨跌比" in signal_names
    assert "涨跌停" in signal_names or len(signal_names) >= 2

    print(f"  ✅ Sentiment 瘦身: 信号={signal_names}")
    print(f"     overall={report.overall_level}({report.overall_score:+.1f})")
    print(f"     position={report.suggested_position}")
    if report.overall_score > 0.5:
        assert "逆向" in report.suggested_position, "仓位建议应标注'逆向'"
        print("  ✅ 逆向标注正确")


def test_macro_slim():
    """验证 Macro 移除了成交额"""
    from skills.macro_monitor import MacroSkill

    class MockFetcher:
        def get_latest_trade_date(self):
            return "20260224"

        def get_shibor(self, days=30):
            import pandas as pd
            data = [{"date": f"2026022{i}", "on": 1.5 + i * 0.01, "1w": 1.8 + i * 0.01} for i in range(15)]
            return pd.DataFrame(data)

        def get_north_flow(self, days=20):
            import pandas as pd
            return pd.DataFrame({"north_money_yi": [10.0] * 20})

        def get_margin_data(self, days=20):
            import pandas as pd
            return pd.DataFrame({
                "trade_date": [f"2026020{i}" for i in range(20)],
                "rzye": [18500e8 + i * 10e8 for i in range(20)],
            })

    skill = MacroSkill()
    skill.fetcher = MockFetcher()
    report = skill.analyze()

    signal_names = [s.name for s in report.signals]
    assert "成交额趋势" not in signal_names, f"成交额应已移除! 当前: {signal_names}"
    print(f"  ✅ Macro 瘦身: 信号={signal_names}")


def test_pipeline_injection():
    """验证 Pipeline 接受外部注入"""
    from skills.stock_pipeline import StockPipeline
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
    assert hasattr(skill, "get_position_perspectives"), "缺少 get_position_perspectives 方法"

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
    r1 = cache.get_north_flow(days=10)
    assert call_count["n"] == 1
    r2 = cache.get_north_flow(days=10)
    assert call_count["n"] == 1
    assert r1 == r2
    cache.get_north_flow(days=20)
    assert call_count["n"] == 2

    print(f"  ✅ DailyCache: 3次请求, 实际调用{call_count['n']}次")
    print(f"     缓存条目: {cache.cache_stats['entries']}")


if __name__ == "__main__":
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
