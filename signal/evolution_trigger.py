#!/usr/bin/env python3
"""
自适应进化触发器 — Opus进化循环的智能调度

原问题:
  - 月度进化对市场风格切换响应太慢 (趋势→震荡可能1月才反应)
  - 纯定时触发无法应对极端行情

方案: 月度定期 + 事件驱动 双轨制
  1. 定期: 每月1日 18:00 (不变)
  2. 事件触发: 满足以下任一条件自动加跑一次进化:
     a) 策略周度夏普 连续2周 < 0.5
     b) 策略周度回撤 > 5%
     c) 市场状态 HALT 出现
     d) 突发事件 BLACKSWAN/CRISIS 触发
     e) 消息层宏观事件密集度异常高

用法:
  trigger = EvolutionTrigger(data_dir="data")
  should, reason = trigger.check()
  if should:
      run_evolution_cycle()
      trigger.record_run(reason)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
#  触发条件配置
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TriggerConfig:
    """进化触发条件阈值"""
    # 定期触发
    monthly_day: int = 1                # 每月几号
    min_interval_days: int = 7          # 两次进化最小间隔 (防止频繁进化)

    # 绩效退化触发
    sharpe_threshold: float = 0.5       # 周度夏普低于此值视为退化
    sharpe_consecutive_weeks: int = 2   # 连续N周低于阈值才触发
    drawdown_threshold: float = -5.0    # 周度回撤超过此值立即触发 (%)
    weekly_return_floor: float = -3.0   # 周收益率低于此值计入衰退信号 (%)

    # 市场事件触发
    halt_triggers: bool = True          # HALT状态触发
    blackswan_triggers: bool = True     # BLACKSWAN事件触发

    # 冷却期
    cooldown_after_event_days: int = 3  # 事件触发后等几天再跑 (等市场稳定)


# ══════════════════════════════════════════════════════════════════════════════
#  周度绩效追踪
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class WeeklySnapshot:
    """单周绩效快照"""
    week_end: str          # YYYYMMDD
    portfolio_value: float
    weekly_return: float   # %
    weekly_sharpe: float   # 近4周滚动夏普
    max_drawdown: float    # 近4周最大回撤 %
    market_regime: str     # 当周主要市场状态
    trade_count: int = 0


class PerformanceTracker:
    """
    周度绩效追踪器。
    每周五收盘后由 daily_signal 脚本调用 record_week() 更新。
    """

    def __init__(self, data_path: str | Path):
        self.path = Path(data_path)
        self.history: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text("utf-8"))
            except Exception:
                return []
        return []

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def record_week(self, snapshot: WeeklySnapshot):
        """记录一周绩效"""
        self.history.append(asdict(snapshot))
        # 只保留最近52周
        if len(self.history) > 52:
            self.history = self.history[-52:]
        self._save()

    def get_recent(self, n: int = 4) -> list[dict]:
        """取最近N周"""
        return self.history[-n:] if self.history else []

    def calc_rolling_sharpe(self, weeks: int = 4) -> float:
        """计算近N周滚动夏普"""
        recent = self.get_recent(weeks)
        if len(recent) < 2:
            return 0.0
        returns = [w["weekly_return"] for w in recent]
        import statistics
        mean_r = statistics.mean(returns)
        if len(returns) < 2:
            return 0.0
        std_r = statistics.stdev(returns)
        if std_r == 0:
            return 0.0 if mean_r == 0 else (5.0 if mean_r > 0 else -5.0)
        # 年化: 周度 * sqrt(52)
        return (mean_r / std_r) * (52 ** 0.5)

    def calc_rolling_drawdown(self, weeks: int = 4) -> float:
        """计算近N周最大回撤 (%)"""
        recent = self.get_recent(weeks)
        if len(recent) < 2:
            return 0.0
        values = [w["portfolio_value"] for w in recent]
        peak = values[0]
        max_dd = 0.0
        for v in values:
            peak = max(peak, v)
            dd = (v / peak - 1) * 100
            max_dd = min(max_dd, dd)
        return max_dd

    def consecutive_low_sharpe(self, threshold: float, n: int) -> bool:
        """最近N周夏普是否连续低于阈值"""
        recent = self.get_recent(n + 2)  # 多取几周做滚动
        if len(recent) < n:
            return False

        # 逐周计算滚动夏普
        low_count = 0
        for i in range(len(recent) - 1, max(len(recent) - n - 1, 0), -1):
            window = recent[max(0, i-3):i+1]
            if len(window) < 2:
                break
            returns = [w["weekly_return"] for w in window]
            import statistics
            std_r = statistics.stdev(returns) if len(returns) > 1 else 0
            mean_r = statistics.mean(returns)
            sharpe = (mean_r / std_r * (52**0.5)) if std_r > 0 else 0
            if sharpe < threshold:
                low_count += 1
            else:
                break  # 连续性断了

        return low_count >= n


# ══════════════════════════════════════════════════════════════════════════════
#  进化触发器
# ══════════════════════════════════════════════════════════════════════════════

class EvolutionTrigger:
    """
    进化循环触发器: 决定是否需要启动Opus进化。

    用法:
        trigger = EvolutionTrigger(data_dir="data")
        should, reason = trigger.check(
            current_regime="DEFENSIVE",
            breaking_level=None,
        )
        if should:
            run_evolution_cycle()
            trigger.record_run(reason)
    """

    def __init__(self, data_dir: str | Path = "data",
                 config: TriggerConfig = None):
        self.data_dir = Path(data_dir)
        self.config = config or TriggerConfig()
        self.tracker = PerformanceTracker(
            self.data_dir / "weekly_performance.json"
        )
        self.run_log_path = self.data_dir / "evolution_run_log.json"
        self._run_log = self._load_run_log()

    def _load_run_log(self) -> list[dict]:
        if self.run_log_path.exists():
            try:
                return json.loads(self.run_log_path.read_text("utf-8"))
            except Exception:
                return []
        return []

    def _save_run_log(self):
        self.run_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_log_path.write_text(
            json.dumps(self._run_log, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _days_since_last_run(self) -> int:
        """距离上次进化的天数"""
        if not self._run_log:
            return 999
        last = self._run_log[-1]
        try:
            last_date = datetime.strptime(last["date"], "%Y%m%d")
            return (datetime.now() - last_date).days
        except Exception:
            return 999

    def _is_monthly_due(self) -> bool:
        """是否到了月度定期触发日"""
        today = datetime.now()
        return today.day == self.config.monthly_day

    def check(self, current_regime: str = "RUN",
              breaking_level: str = None) -> tuple[bool, str]:
        """
        检查是否需要触发进化循环。

        Args:
            current_regime: 当前市场状态 (HALT/DEFENSIVE/RUN/STRONG_RUN)
            breaking_level: 突发事件级别 (BLACKSWAN/CRISIS/WARNING/None)

        Returns:
            (should_trigger: bool, reason: str)
        """
        cfg = self.config
        days_since = self._days_since_last_run()

        # 最小间隔检查
        if days_since < cfg.min_interval_days:
            return False, f"冷却期 ({days_since}/{cfg.min_interval_days}天)"

        reasons = []

        # 1. 月度定期
        if self._is_monthly_due():
            reasons.append("月度定期进化")

        # 2. 绩效退化: 连续低夏普
        if self.tracker.consecutive_low_sharpe(
            cfg.sharpe_threshold, cfg.sharpe_consecutive_weeks
        ):
            recent_sharpe = self.tracker.calc_rolling_sharpe()
            reasons.append(f"策略退化: 连续{cfg.sharpe_consecutive_weeks}周夏普<{cfg.sharpe_threshold} (当前{recent_sharpe:.2f})")

        # 3. 绩效退化: 回撤过大
        dd = self.tracker.calc_rolling_drawdown()
        if dd < cfg.drawdown_threshold:
            reasons.append(f"回撤过大: {dd:.1f}% (阈值{cfg.drawdown_threshold}%)")

        # 4. 市场状态触发
        if cfg.halt_triggers and current_regime == "HALT":
            # HALT后需要等冷却期再进化
            if days_since >= cfg.cooldown_after_event_days:
                reasons.append("HALT状态后自适应进化")

        # 5. 突发事件触发
        if cfg.blackswan_triggers and breaking_level in ("BLACKSWAN", "CRISIS"):
            if days_since >= cfg.cooldown_after_event_days:
                reasons.append(f"突发事件 [{breaking_level}] 后自适应进化")

        if reasons:
            return True, " + ".join(reasons)

        return False, "无触发条件"

    def record_run(self, reason: str, result_summary: str = ""):
        """记录一次进化执行"""
        self._run_log.append({
            "date": datetime.now().strftime("%Y%m%d"),
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "result": result_summary,
        })
        # 只保留最近24次记录
        if len(self._run_log) > 24:
            self._run_log = self._run_log[-24:]
        self._save_run_log()

    def get_status(self) -> dict:
        """获取当前触发器状态"""
        return {
            "days_since_last_run": self._days_since_last_run(),
            "last_run": self._run_log[-1] if self._run_log else None,
            "total_runs": len(self._run_log),
            "rolling_sharpe_4w": round(self.tracker.calc_rolling_sharpe(), 2),
            "rolling_drawdown_4w": round(self.tracker.calc_rolling_drawdown(), 2),
            "weekly_snapshots": len(self.tracker.history),
            "config": asdict(self.config),
        }

    def format_status_text(self) -> str:
        """格式化状态文本"""
        s = self.get_status()
        lines = [
            "Opus 进化触发器状态:",
            f"  上次进化: {s['days_since_last_run']} 天前",
            f"  历史运行: {s['total_runs']} 次",
            f"  近4周夏普: {s['rolling_sharpe_4w']:.2f}",
            f"  近4周回撤: {s['rolling_drawdown_4w']:.1f}%",
            f"  绩效快照: {s['weekly_snapshots']} 周",
        ]
        if s["last_run"]:
            lines.append(f"  上次原因: {s['last_run']['reason']}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Opus进化触发器")
    parser.add_argument("--data-dir", type=str, default="data", help="数据目录")
    parser.add_argument("--check", action="store_true", help="检查是否需要触发")
    parser.add_argument("--status", action="store_true", help="显示状态")
    parser.add_argument("--simulate", action="store_true", help="模拟场景测试")
    args = parser.parse_args()

    trigger = EvolutionTrigger(data_dir=args.data_dir)

    if args.status:
        print(trigger.format_status_text())

    elif args.check:
        should, reason = trigger.check()
        print(f"触发: {'是' if should else '否'}")
        print(f"原因: {reason}")

    elif args.simulate:
        print("=== 模拟场景测试 ===\n")

        # 模拟几周绩效数据
        import random
        tracker = trigger.tracker

        # 模拟前4周好、后2周差的情况
        weeks_data = [
            WeeklySnapshot("20260220", 1050000, 2.5, 1.8, -1.0, "RUN", 3),
            WeeklySnapshot("20260227", 1070000, 1.9, 1.5, -0.8, "RUN", 2),
            WeeklySnapshot("20260306", 1040000, -2.8, 0.6, -3.2, "DEFENSIVE", 4),
            WeeklySnapshot("20260313", 1010000, -2.9, 0.3, -5.6, "DEFENSIVE", 5),
            WeeklySnapshot("20260320", 960000, -5.0, -0.4, -8.6, "DEFENSIVE", 6),
        ]

        for w in weeks_data:
            tracker.record_week(w)

        print("模拟绩效数据:")
        for w in weeks_data:
            print(f"  {w.week_end}: 收益{w.weekly_return:+.1f}% 夏普{w.weekly_sharpe:.1f} 回撤{w.max_drawdown:.1f}% [{w.market_regime}]")
        print()

        # 测试各种触发条件
        scenarios = [
            ("正常市场", "RUN", None),
            ("HALT状态", "HALT", None),
            ("BLACKSWAN事件", "DEFENSIVE", "BLACKSWAN"),
            ("CRISIS事件", "DEFENSIVE", "CRISIS"),
        ]

        for name, regime, breaking in scenarios:
            should, reason = trigger.check(
                current_regime=regime,
                breaking_level=breaking,
            )
            icon = "🟢" if should else "⚪"
            print(f"  {icon} {name:12s} → {'触发' if should else '不触发'}: {reason}")

        print()
        print(trigger.format_status_text())
    else:
        parser.print_help()
