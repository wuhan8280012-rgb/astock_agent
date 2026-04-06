#!/usr/bin/env python3
"""
回测前后对比分析脚本
用法：
  1. 重跑回测前：python compare_backtest.py backup
     → 把当前 backtest/results/ 备份到 backtest/results_baseline/
  2. 重跑回测后：python compare_backtest.py compare
     → 对比 baseline vs 新结果，输出报告
  3. 一步到位：  python compare_backtest.py report
     → 只读新结果生成独立报告（不做对比）
"""

import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# ── 路径配置（相对于项目根目录运行）──
RESULTS_DIR = Path("backtest/results")
BASELINE_DIR = Path("backtest/results_baseline")
REPORT_DIR = Path("backtest/reports")

SIGNALS_FILE = "signals.csv"
TRADES_FILE = "trades.csv"
SUMMARY_FILE = "backtest_summary.json"
EQUITY_FILE = "equity_curve.csv"
MONTHLY_FILE = "monthly_returns.csv"

# ── 已知 baseline 数据（来自交接文档，用于 fallback）──
KNOWN_BASELINE = {
    "total_return": -16.86,
    "total_trades": 16,
    "win_rate": 0.0,
    "total_signals": 27,  # 推断：27 个信号中 16 笔成交
    "catalyst_score_mean": 50.0,
    "catalyst_score_std": 0.0,
    "sector_score_mean": 96.4,
    "sector_unique_values": 7,
    "structure_score_std": 10.55,
    "structure_unique_values": 27,
    "capital_score_mean": 79.79,
    "capital_score_std": 6.89,
    "capital_unique_values": 21,
    "primary_driver_sector_pct": 81.25,  # 13/16
    "crash_exits": 5,
}


def backup(results_dir=RESULTS_DIR, baseline_dir=BASELINE_DIR):
    """备份当前回测结果"""
    if not results_dir.exists():
        print(f"❌ {results_dir} 不存在，无法备份")
        return False

    if baseline_dir.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive = baseline_dir.parent / f"results_baseline_{ts}"
        print(f"⚠️  已有 baseline，旧备份移至 {archive}")
        shutil.move(str(baseline_dir), str(archive))

    shutil.copytree(str(results_dir), str(baseline_dir))
    files = list(baseline_dir.iterdir())
    print(f"✅ 备份完成 → {baseline_dir}/ ({len(files)} 个文件)")
    for f in sorted(files):
        print(f"   {f.name}")
    return True


def load_results(directory: Path) -> dict:
    """从某个目录加载回测结果"""
    data = {}

    # signals.csv
    sig_path = directory / SIGNALS_FILE
    if sig_path.exists():
        data["signals"] = pd.read_csv(sig_path)
    else:
        data["signals"] = None

    # trades.csv
    trades_path = directory / TRADES_FILE
    if trades_path.exists():
        data["trades"] = pd.read_csv(trades_path)
    else:
        data["trades"] = None

    # backtest_summary.json
    summary_path = directory / SUMMARY_FILE
    if summary_path.exists():
        with open(summary_path) as f:
            data["summary"] = json.load(f)
    else:
        data["summary"] = None

    # equity_curve.csv
    eq_path = directory / EQUITY_FILE
    if eq_path.exists():
        data["equity"] = pd.read_csv(eq_path)
    else:
        data["equity"] = None

    return data


def analyze_signals(df: pd.DataFrame) -> dict:
    """分析信号 DataFrame 的关键指标"""
    if df is None or df.empty:
        return {}

    stats = {"total_signals": len(df)}

    # Extract individual agent scores from agent_scores JSON column
    agent_names = ["sector", "capital", "catalyst", "structure", "liquidity"]

    if "agent_scores" in df.columns:
        # Parse JSON strings into dicts
        parsed = df["agent_scores"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else (x if isinstance(x, dict) else {})
        )
        for name in agent_names:
            col_name = f"{name}_score"
            df[col_name] = parsed.apply(lambda d: d.get(name))

    # 各 Agent 评分统计
    for name in agent_names:
        col = f"{name}_score"
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(series) > 0:
                stats[f"{name}_mean"] = round(series.mean(), 2)
                stats[f"{name}_std"] = round(series.std(), 2)
                stats[f"{name}_min"] = round(series.min(), 2)
                stats[f"{name}_max"] = round(series.max(), 2)
                stats[f"{name}_unique"] = series.nunique()
                # 分布桶
                bins = [0, 40, 50, 60, 70, 80, 90, 100]
                hist = pd.cut(series, bins=bins, right=True).value_counts().sort_index()
                stats[f"{name}_dist"] = {str(k): v for k, v in hist.items()}

    # composite_score
    if "composite_score" in df.columns:
        cs = pd.to_numeric(df["composite_score"], errors="coerce").dropna()
        if len(cs) > 0:
            stats["composite_mean"] = round(cs.mean(), 2)
            stats["composite_std"] = round(cs.std(), 2)

    return stats


def analyze_trades(df: pd.DataFrame) -> dict:
    """分析交易 DataFrame 的关键指标"""
    if df is None or df.empty:
        return {}

    stats = {"total_trades": len(df)}

    # 胜率
    if "pnl_pct" in df.columns:
        pnl = pd.to_numeric(df["pnl_pct"], errors="coerce").dropna()
        wins = (pnl > 0).sum()
        stats["win_rate"] = round(wins / len(pnl) * 100, 1) if len(pnl) > 0 else 0
        stats["avg_pnl"] = round(pnl.mean(), 2)
        stats["max_win"] = round(pnl.max(), 2)
        stats["max_loss"] = round(pnl.min(), 2)
    elif "return_pct" in df.columns:
        pnl = pd.to_numeric(df["return_pct"], errors="coerce").dropna()
        wins = (pnl > 0).sum()
        stats["win_rate"] = round(wins / len(pnl) * 100, 1) if len(pnl) > 0 else 0
        stats["avg_pnl"] = round(pnl.mean(), 2)

    # primary_driver 分布
    driver_col = None
    for candidate in ["primary_driver", "driver", "top_driver"]:
        if candidate in df.columns:
            driver_col = candidate
            break

    if driver_col:
        driver_counts = df[driver_col].value_counts()
        total = driver_counts.sum()
        stats["primary_driver"] = {
            k: {"count": int(v), "pct": round(v / total * 100, 1)}
            for k, v in driver_counts.items()
        }

    # 评分崩溃退出：入场时 structure_score 接近 50
    crash_col = None
    for candidate in ["entry_structure_score", "structure_score", "structure_score_entry"]:
        if candidate in df.columns:
            crash_col = candidate
            break
    if crash_col:
        scores = pd.to_numeric(df[crash_col], errors="coerce").dropna()
        crash_count = (scores < 55).sum()
        stats["crash_risk_entries"] = int(crash_count)
        stats["crash_risk_pct"] = round(crash_count / len(scores) * 100, 1) if len(scores) > 0 else 0

    # exit_reason 分布
    for candidate in ["exit_reason", "sell_reason", "exit_type"]:
        if candidate in df.columns:
            exit_counts = df[candidate].value_counts()
            stats["exit_reasons"] = {k: int(v) for k, v in exit_counts.items()}
            break

    return stats


def format_comparison(label: str, old_val, new_val, fmt=".2f", better="higher"):
    """格式化单行对比，带方向箭头"""
    if old_val is None or new_val is None:
        return f"  {label}: {new_val}"

    try:
        old_f = float(old_val)
        new_f = float(new_val)
        diff = new_f - old_f
        arrow = "→"
        if diff > 0.01:
            arrow = "↑" if better == "higher" else "↓ ⚠️"
        elif diff < -0.01:
            arrow = "↓" if better == "higher" else "↑ ✅"
            if better == "higher":
                arrow = "↓ ⚠️"
            else:
                arrow = "↑ ✅"  # lower is better (e.g. crash exits)
        return f"  {label}: {old_f:{fmt}} → {new_f:{fmt}} ({diff:+{fmt}}) {arrow}"
    except (ValueError, TypeError):
        return f"  {label}: {old_val} → {new_val}"


def generate_report(baseline_data: dict, new_data: dict, use_known_baseline: bool = False):
    """生成对比报告"""
    lines = []
    lines.append("=" * 70)
    lines.append("  回测对比报告：v1.0 (baseline) vs v1.1 (catalyst清零 + prompt丰富化)")
    lines.append(f"  生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    if use_known_baseline:
        lines.append("  ⚠️  Baseline 数据来自交接文档记录值（非文件加载）")
    lines.append("")

    # ── 信号分析 ──
    new_sig_stats = analyze_signals(new_data.get("signals"))

    lines.append("─" * 40)
    lines.append("📊 信号层面对比")
    lines.append("─" * 40)

    old_total = KNOWN_BASELINE["total_signals"] if use_known_baseline else \
        analyze_signals(baseline_data.get("signals")).get("total_signals", "N/A")
    new_total = new_sig_stats.get("total_signals", "N/A")
    lines.append(format_comparison("信号总数", old_total, new_total, ".0f"))

    # 各 Agent 评分
    agents_to_check = ["sector", "capital", "catalyst", "structure", "liquidity"]
    for agent in agents_to_check:
        lines.append(f"\n  【{agent}_agent】")

        if use_known_baseline:
            old_mean = KNOWN_BASELINE.get(f"{agent}_score_mean")
            old_std = KNOWN_BASELINE.get(f"{agent}_score_std") if f"{agent}_score_std" in KNOWN_BASELINE else None
            old_unique = KNOWN_BASELINE.get(f"{agent}_unique_values")
        else:
            old_sig = analyze_signals(baseline_data.get("signals"))
            old_mean = old_sig.get(f"{agent}_mean")
            old_std = old_sig.get(f"{agent}_std")
            old_unique = old_sig.get(f"{agent}_unique")

        new_mean = new_sig_stats.get(f"{agent}_mean")
        new_std = new_sig_stats.get(f"{agent}_std")
        new_unique = new_sig_stats.get(f"{agent}_unique")

        if new_mean is not None:
            lines.append(format_comparison("  均值", old_mean, new_mean))
        if new_std is not None:
            lines.append(format_comparison("  标准差", old_std, new_std, better="higher"))
        if new_unique is not None:
            lines.append(format_comparison("  不同值数量", old_unique, new_unique, ".0f", better="higher"))

        # 分布
        dist = new_sig_stats.get(f"{agent}_dist")
        if dist:
            lines.append(f"    分布: {dict(dist)}")

    # composite
    if "composite_mean" in new_sig_stats:
        lines.append(f"\n  【综合评分】")
        lines.append(f"    均值: {new_sig_stats['composite_mean']}")
        lines.append(f"    标准差: {new_sig_stats['composite_std']}")

    # ── 交易分析 ──
    new_trade_stats = analyze_trades(new_data.get("trades"))

    lines.append("")
    lines.append("─" * 40)
    lines.append("📈 交易层面对比")
    lines.append("─" * 40)

    old_trades = KNOWN_BASELINE["total_trades"] if use_known_baseline else \
        analyze_trades(baseline_data.get("trades")).get("total_trades", "N/A")
    lines.append(format_comparison("交易总数", old_trades, new_trade_stats.get("total_trades", "N/A"), ".0f"))

    old_wr = KNOWN_BASELINE["win_rate"] if use_known_baseline else \
        analyze_trades(baseline_data.get("trades")).get("win_rate", "N/A")
    lines.append(format_comparison("胜率 (%)", old_wr, new_trade_stats.get("win_rate", "N/A"), ".1f"))

    if "avg_pnl" in new_trade_stats:
        lines.append(f"  平均盈亏: {new_trade_stats['avg_pnl']}%")
    if "max_win" in new_trade_stats:
        lines.append(f"  最大盈利: {new_trade_stats['max_win']}%")
    if "max_loss" in new_trade_stats:
        lines.append(f"  最大亏损: {new_trade_stats['max_loss']}%")

    # primary_driver
    lines.append(f"\n  【驱动因子分布】")
    if use_known_baseline:
        lines.append(f"    Baseline: sector=81.25% (13/16)")

    if "primary_driver" in new_trade_stats:
        for driver, info in new_trade_stats["primary_driver"].items():
            lines.append(f"    新结果: {driver} = {info['pct']}% ({info['count']}笔)")
    else:
        lines.append(f"    ⚠️  未找到 primary_driver 列，请检查 trades.csv 列名")

    # 崩溃退出
    old_crash = KNOWN_BASELINE["crash_exits"] if use_known_baseline else \
        analyze_trades(baseline_data.get("trades")).get("crash_risk_entries", "N/A")
    lines.append(format_comparison(
        "\n  入场时 structure<55（崩溃风险）",
        old_crash, new_trade_stats.get("crash_risk_entries", "N/A"), ".0f", better="lower"
    ))

    # exit reasons
    if "exit_reasons" in new_trade_stats:
        lines.append(f"\n  【退出原因分布】")
        for reason, count in new_trade_stats["exit_reasons"].items():
            lines.append(f"    {reason}: {count}")

    # ── Summary ──
    if new_data.get("summary"):
        lines.append("")
        lines.append("─" * 40)
        lines.append("📋 回测 Summary")
        lines.append("─" * 40)
        summary = new_data["summary"]
        old_ret = KNOWN_BASELINE["total_return"] if use_known_baseline else None
        if "total_return" in summary:
            lines.append(format_comparison("总收益 (%)", old_ret, summary["total_return"], ".2f"))
        for k, v in summary.items():
            if k != "total_return":
                lines.append(f"  {k}: {v}")

    # ── 结论 ──
    lines.append("")
    lines.append("=" * 70)
    lines.append("🔍 关键观察点（人工确认）")
    lines.append("=" * 70)
    lines.append("  1. catalyst_score 是否仍全部 50.0？（应该是，权重清零不改评分）")
    lines.append("  2. primary_driver 是否仍 sector 独大？")
    lines.append("     - 权重重分配后 structure/capital 影响力应上升")
    lines.append("  3. composite_score 的分布是否更合理？（std 应增大）")
    lines.append("  4. 信号数量是否有显著变化？")
    lines.append("     - 如果信号数大幅变化，检查入场阈值是否受 composite 变化影响")
    lines.append("  5. 交易结果是否有改善？")
    lines.append("     - 注意：simulated_buy 不含 LLM，结果变化主要来自信号筛选变化")
    lines.append("")

    return "\n".join(lines)


def cmd_backup():
    backup()


def cmd_compare():
    """对比 baseline 和新结果"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if BASELINE_DIR.exists():
        print("📂 从 baseline 目录加载旧结果...")
        baseline_data = load_results(BASELINE_DIR)
        use_known = False
    else:
        print("⚠️  无 baseline 目录，使用交接文档记录值作为对比基准")
        baseline_data = {}
        use_known = True

    if not RESULTS_DIR.exists():
        print(f"❌ {RESULTS_DIR} 不存在，请先运行回测")
        return

    print("📂 加载新回测结果...")
    new_data = load_results(RESULTS_DIR)

    report = generate_report(baseline_data, new_data, use_known)

    # 输出到终端
    print("\n" + report)

    # 保存到文件
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"comparison_report_{ts}.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n📄 报告已保存：{report_path}")


def cmd_report():
    """只分析新结果（不对比）"""
    if not RESULTS_DIR.exists():
        print(f"❌ {RESULTS_DIR} 不存在")
        return

    data = load_results(RESULTS_DIR)
    sig_stats = analyze_signals(data.get("signals"))
    trade_stats = analyze_trades(data.get("trades"))

    print(json.dumps({"signals": sig_stats, "trades": trade_stats}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "backup":
        cmd_backup()
    elif cmd == "compare":
        cmd_compare()
    elif cmd == "report":
        cmd_report()
    else:
        print(f"未知命令：{cmd}")
        print(__doc__)
        sys.exit(1)
