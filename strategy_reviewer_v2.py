"""
策略复盘 v2: 自动复盘 + LLM 审视 + 参数提案 + 人工审批
=========================================================
在 strategy_reviewer.py (v1) 基础上升级:

v1 已有:
  ✅ 记录推荐 → 追踪5/10/20日收益 → 按等级/板块/信号统计
  ✅ 硬编码规则的优化建议

v2 新增:
  1. SignalScorecard — 按信号类型(阶梯突破/EMA/放量/缺口)追踪胜率
  2. DebateTracker — 追踪辩论决策的正确率
  3. WeeklyReview — 每周五LLM复盘会议 (DeepSeek R1)
  4. ParameterProposal — 结构化参数变更提案
  5. ApprovalGate — 人工审批后才生效

架构原则:
  - 每日: 机械地收集数据 (零LLM调用)
  - 每周五: LLM 复盘会议 (1次R1调用)
  - 参数变更: 必须人工审批，带 30 日自动回滚

安全边界:
  - 模型只能提议参数变更，不能自动生效
  - 每次最多调整 2 个参数，幅度不超过 ±30%
  - 所有变更记录可追溯，带回滚机制
  - 样本量 < 20 时不产生任何提案
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict

import pandas as pd
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ════════════════════════════════════════════════════════════
#  数据路径
# ════════════════════════════════════════════════════════════

KNOWLEDGE_DIR = Path("./knowledge")
SIGNAL_LOG = KNOWLEDGE_DIR / "signal_outcomes.json"
DEBATE_LOG = KNOWLEDGE_DIR / "debate_outcomes.json"
PROPOSAL_LOG = KNOWLEDGE_DIR / "param_proposals.json"
PARAM_HISTORY = KNOWLEDGE_DIR / "param_history.json"
WEEKLY_REVIEW_LOG = KNOWLEDGE_DIR / "weekly_reviews.json"

# ════════════════════════════════════════════════════════════
#  安全参数
# ════════════════════════════════════════════════════════════

SAFETY = {
    "min_samples_for_proposal": 20,      # 最少样本才允许提案
    "max_proposals_per_review": 2,        # 每次复盘最多提2个变更
    "max_param_change_pct": 0.30,         # 单参数最大调整幅度 ±30%
    "auto_rollback_days": 30,             # 变更后30天内可自动回滚
    "min_confidence_for_proposal": 0.7,   # 提案最低置信度
    "cooldown_days": 7,                   # 同一参数两次调整的最短间隔
}


# ════════════════════════════════════════════════════════════
#  数据结构
# ════════════════════════════════════════════════════════════

@dataclass
class SignalOutcome:
    """单条信号的结果追踪"""
    date: str
    ts_code: str
    name: str
    signal_type: str         # 阶梯突破/EMA突破/放量阳线/缺口突破
    action: str              # BUY/SELL
    entry_price: float = 0
    stop_price: float = 0
    target_price: float = 0
    position_size_pct: float = 0
    sector: str = ""
    env_score: int = 0

    # ── 以下字段由复盘填充 ──
    ret_1d: float = None     # 次日收益
    ret_3d: float = None
    ret_5d: float = None
    ret_10d: float = None
    max_drawdown: float = None
    max_gain: float = None
    hit_stop: bool = None
    hit_target: bool = None
    outcome: str = ""        # "盈利" / "止损" / "平淡" / "待定"


@dataclass
class DebateOutcome:
    """辩论决策的结果追踪"""
    date: str
    question: str
    ts_code: str = ""
    action: str = ""          # 买入/持有/减仓/卖出/观望
    confidence: int = 0
    env_score: int = 0
    env_gate: str = ""        # 通过/警告/拦截

    # ── 以下字段由复盘填充 ──
    actual_ret_5d: float = None   # 如果执行了，5日后涨跌
    was_correct: bool = None      # 事后看决策对不对
    lesson: str = ""              # LLM 总结的教训


@dataclass
class ParameterProposal:
    """参数变更提案"""
    id: str                      # 唯一ID (日期+序号)
    date: str
    module: str                  # 目标模块: trade_signal / sector_stage / risk_control
    param_name: str              # 参数名
    current_value: float         # 当前值
    proposed_value: float        # 建议值
    change_pct: float            # 变化幅度%
    reason: str                  # LLM 给出的理由
    evidence: str                # 支撑数据 (如 "阶梯突破5日胜率38%, N=25")
    confidence: float            # 置信度 0-1
    status: str = "pending"      # pending / approved / rejected / rolled_back
    approved_by: str = ""        # 审批人
    approved_date: str = ""
    rollback_date: str = ""      # 自动回滚日期


@dataclass
class SignalScorecard:
    """信号类型记分卡"""
    signal_type: str
    total: int = 0
    wins: int = 0               # ret_5d > 0
    losses: int = 0             # ret_5d < 0
    stops: int = 0              # 触及止损
    targets: int = 0            # 触及目标
    avg_ret_5d: float = 0
    avg_max_dd: float = 0       # 平均最大回撤
    avg_rr_actual: float = 0    # 实际盈亏比
    win_rate: float = 0

    def to_brief(self) -> str:
        return (
            f"{self.signal_type}: {self.total}只 "
            f"胜率{self.win_rate:.0%} "
            f"均5日{self.avg_ret_5d:+.1f}% "
            f"止损率{self.stops/max(self.total,1):.0%}"
        )


# ════════════════════════════════════════════════════════════
#  核心引擎
# ════════════════════════════════════════════════════════════

class StrategyReviewerV2:
    """
    策略复盘引擎 v2

    日常流程:
      盘后 → log_signal_outcomes() → 记录今日信号
      盘后 → log_debate_outcome()  → 记录辩论决策
      盘后 → update_outcomes()     → 填充历史信号的实际收益

    周五流程:
      → compute_scorecards()       → 按信号类型统计
      → compute_debate_accuracy()  → 辩论正确率
      → generate_weekly_review()   → LLM 复盘 (1次调用)
      → generate_proposals()       → 参数变更提案
      → 人工审批 → apply_proposal() or reject_proposal()
    """

    def __init__(self, llm_client=None):
        """
        Parameters
        ----------
        llm_client : 可选的 LLM 客户端
            需要实现 chat(prompt, model) 方法
            若不传，LLM 复盘功能不可用，但数据收集正常工作
        """
        KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
        self.llm = llm_client

    # ────────────────────────────────────────────────────────
    #  1. 每日: 记录信号结果
    # ────────────────────────────────────────────────────────

    def log_signal_outcomes(self, signals: List[Dict]):
        """
        记录今日的交易信号 (来自 TradeSignalScanner)

        Parameters
        ----------
        signals : TradeSignal.to_dict() 的列表
        """
        records = self._load(SIGNAL_LOG)

        for s in signals:
            outcome = SignalOutcome(
                date=datetime.now().strftime("%Y%m%d"),
                ts_code=s.get("ts_code", ""),
                name=s.get("name", ""),
                signal_type=s.get("signal_type", ""),
                action=s.get("action", ""),
                entry_price=s.get("current_price", 0),
                stop_price=s.get("stop_price", 0),
                target_price=s.get("target_price", 0),
                position_size_pct=s.get("position_size_pct", 0),
                sector=s.get("sector", ""),
                env_score=s.get("env_score", 0),
            )
            records.append(asdict(outcome))

        self._save(SIGNAL_LOG, records)

    def log_debate_outcome(self, debate_result: Dict, question: str = ""):
        """
        记录辩论决策 (来自 DebateEngine)

        Parameters
        ----------
        debate_result : DebateResult.to_dict()
        """
        records = self._load(DEBATE_LOG)

        outcome = DebateOutcome(
            date=datetime.now().strftime("%Y%m%d"),
            question=question or debate_result.get("question", ""),
            ts_code=debate_result.get("ts_code", ""),
            action=debate_result.get("action", ""),
            confidence=debate_result.get("confidence", 0),
            env_score=debate_result.get("env_score", 0),
            env_gate=debate_result.get("env_gate", ""),
        )
        records.append(asdict(outcome))

        self._save(DEBATE_LOG, records)

    # ────────────────────────────────────────────────────────
    #  2. 每日: 回填历史信号的实际收益
    # ────────────────────────────────────────────────────────

    def update_outcomes(self, fetcher=None):
        """
        回填历史信号的实际收益

        对每条 ret_5d=None 的记录，检查是否已过 5 个交易日，
        如果是则拉取行情填充收益数据。

        Parameters
        ----------
        fetcher : DataFetcher 实例 (或 DailyCache 包装后的)
        """
        if fetcher is None:
            print("[ReviewerV2] 需要传入 fetcher 才能回填收益")
            return 0

        records = self._load(SIGNAL_LOG)
        updated = 0
        today = datetime.now().strftime("%Y%m%d")

        for rec in records:
            if rec.get("ret_5d") is not None:
                continue
            if rec.get("action") != "BUY":
                continue

            rec_date = rec.get("date", "")
            entry = rec.get("entry_price", 0)
            stop = rec.get("stop_price", 0)
            target = rec.get("target_price", 0)

            if not rec_date or entry <= 0:
                continue

            # 至少过6天才复盘 (5个交易日+缓冲)
            try:
                rec_dt = datetime.strptime(rec_date, "%Y%m%d")
                if (datetime.now() - rec_dt).days < 6:
                    continue
            except Exception:
                continue

            try:
                df = fetcher.get_stock_daily(rec["ts_code"], days=25)
                if df is None or df.empty:
                    continue

                df = df.sort_values("trade_date").reset_index(drop=True)
                df = df[df["trade_date"] > rec_date].reset_index(drop=True)

                if len(df) < 5:
                    continue

                # 填充收益
                rec["ret_1d"] = round((df.iloc[0]["close"] / entry - 1) * 100, 2)
                rec["ret_3d"] = round((df.iloc[min(2, len(df)-1)]["close"] / entry - 1) * 100, 2)
                rec["ret_5d"] = round((df.iloc[min(4, len(df)-1)]["close"] / entry - 1) * 100, 2)

                if len(df) >= 10:
                    rec["ret_10d"] = round((df.iloc[9]["close"] / entry - 1) * 100, 2)

                rec["max_drawdown"] = round((df["low"].min() / entry - 1) * 100, 2)
                rec["max_gain"] = round((df["high"].max() / entry - 1) * 100, 2)

                rec["hit_stop"] = bool(stop > 0 and df["low"].min() <= stop)
                rec["hit_target"] = bool(target > 0 and df["high"].max() >= target)

                # 判断结果
                if rec["hit_stop"]:
                    rec["outcome"] = "止损"
                elif rec["hit_target"]:
                    rec["outcome"] = "达标"
                elif rec["ret_5d"] > 2:
                    rec["outcome"] = "盈利"
                elif rec["ret_5d"] < -2:
                    rec["outcome"] = "亏损"
                else:
                    rec["outcome"] = "平淡"

                updated += 1
                time.sleep(0.08)

            except Exception as e:
                continue

        if updated > 0:
            self._save(SIGNAL_LOG, records)

        return updated

    # ────────────────────────────────────────────────────────
    #  3. 周五: 信号记分卡
    # ────────────────────────────────────────────────────────

    def compute_scorecards(self, lookback_days: int = 30) -> Dict[str, SignalScorecard]:
        """
        按信号类型计算记分卡

        Returns
        -------
        {"阶梯突破": SignalScorecard, "EMA突破": ..., ...}
        """
        records = self._load(SIGNAL_LOG)
        cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y%m%d")

        # 只统计已有结果的买入信号
        valid = [
            r for r in records
            if r.get("action") == "BUY"
            and r.get("ret_5d") is not None
            and r.get("date", "") >= cutoff
        ]

        scorecards = {}
        # 按信号类型分组
        types = set(r.get("signal_type", "未知") for r in valid)

        for stype in types:
            group = [r for r in valid if r.get("signal_type") == stype]
            if not group:
                continue

            sc = SignalScorecard(signal_type=stype)
            sc.total = len(group)

            rets = [r["ret_5d"] for r in group]
            sc.wins = sum(1 for r in rets if r > 0)
            sc.losses = sum(1 for r in rets if r < 0)
            sc.stops = sum(1 for r in group if r.get("hit_stop"))
            sc.targets = sum(1 for r in group if r.get("hit_target"))
            sc.avg_ret_5d = round(np.mean(rets), 2)
            sc.win_rate = sc.wins / sc.total if sc.total > 0 else 0

            dds = [r.get("max_drawdown", 0) for r in group if r.get("max_drawdown") is not None]
            sc.avg_max_dd = round(np.mean(dds), 2) if dds else 0

            scorecards[stype] = sc

        # 增加一个"全部"汇总
        if valid:
            all_sc = SignalScorecard(signal_type="全部信号")
            all_sc.total = len(valid)
            all_rets = [r["ret_5d"] for r in valid]
            all_sc.wins = sum(1 for r in all_rets if r > 0)
            all_sc.losses = sum(1 for r in all_rets if r < 0)
            all_sc.stops = sum(1 for r in valid if r.get("hit_stop"))
            all_sc.targets = sum(1 for r in valid if r.get("hit_target"))
            all_sc.avg_ret_5d = round(np.mean(all_rets), 2)
            all_sc.win_rate = all_sc.wins / all_sc.total
            scorecards["全部信号"] = all_sc

        return scorecards

    # ────────────────────────────────────────────────────────
    #  4. 周五: 辩论正确率
    # ────────────────────────────────────────────────────────

    def compute_debate_accuracy(self, lookback_days: int = 30) -> Dict:
        """
        计算辩论决策的事后正确率

        正确的定义:
          - 说"买入" + 5日后涨 > 2% → 正确
          - 说"观望" + 5日后跌 > 2% → 正确 (躲过了下跌)
          - 说"卖出" + 5日后跌 > 2% → 正确
          - 环境门控拦截 + 5日后跌 → 正确 (门控有效)
        """
        debates = self._load(DEBATE_LOG)
        signals = self._load(SIGNAL_LOG)
        cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y%m%d")

        valid_debates = [
            d for d in debates
            if d.get("date", "") >= cutoff
            and d.get("action")
        ]

        if not valid_debates:
            return {"total": 0, "message": "暂无辩论记录"}

        # 构建 ts_code+date → ret_5d 映射
        ret_map = {}
        for s in signals:
            if s.get("ret_5d") is not None:
                key = f"{s.get('ts_code', '')}_{s.get('date', '')}"
                ret_map[key] = s["ret_5d"]

        correct = 0
        incorrect = 0
        unverified = 0
        gate_saved = 0  # 门控拦截避免的损失次数

        for d in valid_debates:
            key = f"{d.get('ts_code', '')}_{d.get('date', '')}"
            ret = ret_map.get(key)

            if ret is None:
                unverified += 1
                continue

            action = d.get("action", "")
            if action in ("买入",) and ret > 2:
                correct += 1
                d["was_correct"] = True
            elif action in ("观望", "卖出", "减仓") and ret < -2:
                correct += 1
                d["was_correct"] = True
            elif action in ("买入",) and ret < -2:
                incorrect += 1
                d["was_correct"] = False
            elif action in ("观望",) and ret > 5:
                incorrect += 1  # 观望但大涨，错过了
                d["was_correct"] = False
            else:
                unverified += 1  # 结果不明确

            if d.get("env_gate") == "拦截" and ret < 0:
                gate_saved += 1

        total_verified = correct + incorrect
        accuracy = correct / total_verified if total_verified > 0 else 0

        # 保存更新后的结果
        self._save(DEBATE_LOG, debates)

        return {
            "total_debates": len(valid_debates),
            "verified": total_verified,
            "correct": correct,
            "incorrect": incorrect,
            "unverified": unverified,
            "accuracy": round(accuracy * 100, 1),
            "gate_saves": gate_saved,
        }

    # ────────────────────────────────────────────────────────
    #  5. 周五: LLM 复盘会议
    # ────────────────────────────────────────────────────────

    def generate_weekly_review(
        self,
        scorecards: Dict[str, SignalScorecard] = None,
        debate_stats: Dict = None,
        current_params: Dict = None,
    ) -> Dict:
        """
        LLM 驱动的周度复盘

        输入: 本周记分卡 + 辩论正确率 + 当前参数
        输出: 复盘总结 + 参数变更提案

        Returns
        -------
        {
            "summary": "本周复盘总结...",
            "proposals": [ParameterProposal, ...],
            "raw_response": "LLM 原始输出",
        }
        """
        if scorecards is None:
            scorecards = self.compute_scorecards()
        if debate_stats is None:
            debate_stats = self.compute_debate_accuracy()

        # 构建复盘上下文
        context = self._build_review_context(scorecards, debate_stats, current_params)

        # 无 LLM 时返回纯数据
        if self.llm is None:
            return {
                "summary": "LLM 未配置，仅输出数据统计",
                "scorecards": {k: v.to_brief() for k, v in scorecards.items()},
                "debate_stats": debate_stats,
                "proposals": self._generate_rule_based_proposals(scorecards, current_params),
                "raw_response": None,
            }

        # LLM 复盘
        prompt = self._build_review_prompt(context)

        try:
            response = self.llm.chat(prompt, model="deepseek-reasoner")
            proposals = self._parse_proposals(response, current_params)

            review = {
                "date": datetime.now().strftime("%Y%m%d"),
                "summary": self._extract_summary(response),
                "proposals": [asdict(p) for p in proposals],
                "scorecards": {k: v.to_brief() for k, v in scorecards.items()},
                "debate_stats": debate_stats,
                "raw_response": response[:2000],
            }

            # 保存
            reviews = self._load(WEEKLY_REVIEW_LOG)
            reviews.append(review)
            self._save(WEEKLY_REVIEW_LOG, reviews)

            return review

        except Exception as e:
            print(f"[ReviewerV2] LLM 复盘失败: {e}")
            return {
                "summary": f"LLM 调用失败: {e}",
                "proposals": self._generate_rule_based_proposals(scorecards, current_params),
                "raw_response": None,
            }

    def _build_review_context(self, scorecards, debate_stats, params) -> str:
        """构建复盘上下文 (给 LLM 的输入)"""
        parts = []

        parts.append("## 信号记分卡 (近30日)")
        for name, sc in scorecards.items():
            parts.append(f"  {sc.to_brief()}")

        if debate_stats and debate_stats.get("total_debates", 0) > 0:
            parts.append(f"\n## 辩论决策正确率")
            parts.append(f"  总决策: {debate_stats['total_debates']}")
            parts.append(f"  已验证: {debate_stats['verified']} "
                         f"(正确{debate_stats['correct']} 错误{debate_stats['incorrect']})")
            parts.append(f"  正确率: {debate_stats['accuracy']}%")
            parts.append(f"  环境门控拦截避损: {debate_stats['gate_saves']}次")

        if params:
            parts.append(f"\n## 当前关键参数")
            for module, p in params.items():
                parts.append(f"  [{module}]")
                for k, v in p.items():
                    parts.append(f"    {k}: {v}")

        return "\n".join(parts)

    def _build_review_prompt(self, context: str) -> str:
        """构建 LLM 复盘 prompt"""
        return f"""你是一个A股量化策略的复盘分析师。以下是系统近30天的表现数据。

{context}

请完成以下任务:

1. **总结**: 用3-5句话概括本周策略表现的关键发现。哪些做得好，哪些需要改进。

2. **参数提案**: 如果数据支持，提出最多2个参数调整建议。格式:
   提案1: [模块名].[参数名] 从 [当前值] 调整到 [建议值]
   理由: ...
   证据: ...
   置信度: 高/中/低

规则:
- 样本量 < 20 不允许提案
- 单参数调整幅度不超过 ±30%
- 只基于统计数据提案，不要凭直觉
- 如果数据不足以支撑任何调整，明确说"暂不建议调整"

请用中文回答。"""

    def _parse_proposals(self, response: str, current_params: Dict) -> List[ParameterProposal]:
        """从 LLM 响应中解析参数提案"""
        proposals = []
        # 这里用简单的文本解析，实际可以要求 LLM 输出 JSON
        # 暂时返回空列表，依赖规则引擎
        return proposals

    def _extract_summary(self, response: str) -> str:
        """从 LLM 响应中提取总结部分"""
        lines = response.strip().split("\n")
        summary_lines = []
        in_summary = False
        for line in lines:
            if "总结" in line or "概括" in line:
                in_summary = True
                continue
            if "提案" in line or "参数" in line:
                in_summary = False
            if in_summary and line.strip():
                summary_lines.append(line.strip())
        return " ".join(summary_lines[:5]) if summary_lines else response[:500]

    # ────────────────────────────────────────────────────────
    #  5b. 规则引擎提案 (LLM 不可用时的兜底)
    # ────────────────────────────────────────────────────────

    def _generate_rule_based_proposals(
        self,
        scorecards: Dict[str, SignalScorecard],
        current_params: Dict = None,
    ) -> List[Dict]:
        """
        基于硬规则生成参数提案 (不需要 LLM)

        规则:
          1. 某信号类型胜率 < 35% 且 N ≥ 20 → 收紧触发条件
          2. 某信号类型止损率 > 40% → 放宽止损或收紧入场
          3. 整体胜率 > 60% → 可适度放宽
          4. 环境门控拦截后仍有 >30% 正确 → 门控可能过严
        """
        proposals = []
        if current_params is None:
            current_params = {}

        ts_params = current_params.get("trade_signal", {})

        all_card = scorecards.get("全部信号")
        if all_card is None or all_card.total < SAFETY["min_samples_for_proposal"]:
            return proposals  # 样本不足

        # 规则1: 低胜率信号 → 收紧
        for stype, sc in scorecards.items():
            if stype == "全部信号":
                continue
            if sc.total < 8:
                continue

            if sc.win_rate < 0.35:
                if stype == "阶梯突破":
                    current = ts_params.get("breakout_vol_ratio", 1.5)
                    proposed = min(current * 1.2, current + 0.5)  # 提高量比门槛
                    proposals.append(self._make_proposal(
                        "trade_signal", "breakout_vol_ratio",
                        current, round(proposed, 1),
                        f"阶梯突破胜率仅{sc.win_rate:.0%}(N={sc.total}), 提高量比门槛过滤假突破",
                        f"{sc.to_brief()}",
                    ))
                elif stype == "放量阳线":
                    current = ts_params.get("min_yang_pct", 2.0)
                    proposed = min(current * 1.25, 4.0)
                    proposals.append(self._make_proposal(
                        "trade_signal", "min_yang_pct",
                        current, round(proposed, 1),
                        f"放量阳线胜率仅{sc.win_rate:.0%}(N={sc.total}), 提高最小涨幅门槛",
                        f"{sc.to_brief()}",
                    ))

        # 规则2: 止损率过高 → 放宽止损或收紧R:R
        if all_card.total > 0:
            stop_rate = all_card.stops / all_card.total
            if stop_rate > 0.40:
                current_rr = ts_params.get("min_rr_ratio", 3.0)
                proposed_rr = min(current_rr + 0.5, 5.0)
                proposals.append(self._make_proposal(
                    "trade_signal", "min_rr_ratio",
                    current_rr, proposed_rr,
                    f"整体止损率{stop_rate:.0%}偏高, 提高盈亏比门槛过滤低质量信号",
                    f"止损{all_card.stops}/{all_card.total}只",
                ))

        # 限制最多2个
        proposals = proposals[:SAFETY["max_proposals_per_review"]]

        return [asdict(p) if isinstance(p, ParameterProposal) else p for p in proposals]

    def _make_proposal(
        self, module, param, current, proposed, reason, evidence
    ) -> ParameterProposal:
        """构造参数提案"""
        change_pct = abs(proposed - current) / current if current != 0 else 0

        # 安全检查: 不超过 ±30%
        if change_pct > SAFETY["max_param_change_pct"]:
            max_change = current * SAFETY["max_param_change_pct"]
            proposed = current + max_change if proposed > current else current - max_change
            proposed = round(proposed, 2)
            change_pct = SAFETY["max_param_change_pct"]

        return ParameterProposal(
            id=f"{datetime.now().strftime('%Y%m%d')}_{module}_{param}",
            date=datetime.now().strftime("%Y%m%d"),
            module=module,
            param_name=param,
            current_value=current,
            proposed_value=proposed,
            change_pct=round(change_pct * 100, 1),
            reason=reason,
            evidence=evidence,
            confidence=0.6,
            status="pending",
            rollback_date=(datetime.now() + timedelta(days=SAFETY["auto_rollback_days"])).strftime("%Y%m%d"),
        )

    # ────────────────────────────────────────────────────────
    #  6. 审批流程
    # ────────────────────────────────────────────────────────

    def get_pending_proposals(self) -> List[Dict]:
        """获取待审批的提案"""
        proposals = self._load(PROPOSAL_LOG)
        return [p for p in proposals if p.get("status") == "pending"]

    def approve_proposal(self, proposal_id: str, approver: str = "human"):
        """
        审批通过一个提案

        通过后:
          1. 记录审批信息
          2. 记录当前参数到历史 (用于回滚)
          3. 返回生效指令 (需要人工在 config.py 中修改)
        """
        proposals = self._load(PROPOSAL_LOG)

        for p in proposals:
            if p.get("id") == proposal_id:
                p["status"] = "approved"
                p["approved_by"] = approver
                p["approved_date"] = datetime.now().strftime("%Y%m%d")

                # 记录参数历史
                self._log_param_change(p)

                self._save(PROPOSAL_LOG, proposals)

                print(f"[ReviewerV2] 提案 {proposal_id} 已批准")
                print(f"  请在 config.py 中修改:")
                print(f"  SKILL_PARAMS[\"{p['module']}\"][\"{p['param_name']}\"] = {p['proposed_value']}")
                print(f"  (原值: {p['current_value']})")
                print(f"  自动回滚日期: {p['rollback_date']}")
                return True

        print(f"[ReviewerV2] 未找到提案: {proposal_id}")
        return False

    def reject_proposal(self, proposal_id: str, reason: str = ""):
        """驳回提案"""
        proposals = self._load(PROPOSAL_LOG)

        for p in proposals:
            if p.get("id") == proposal_id:
                p["status"] = "rejected"
                p["reject_reason"] = reason
                self._save(PROPOSAL_LOG, proposals)
                print(f"[ReviewerV2] 提案 {proposal_id} 已驳回: {reason}")
                return True

        return False

    def rollback_proposal(self, proposal_id: str):
        """回滚一个已生效的提案"""
        proposals = self._load(PROPOSAL_LOG)

        for p in proposals:
            if p.get("id") == proposal_id and p.get("status") == "approved":
                p["status"] = "rolled_back"

                print(f"[ReviewerV2] 提案 {proposal_id} 已回滚")
                print(f"  请在 config.py 中恢复:")
                print(f"  SKILL_PARAMS[\"{p['module']}\"][\"{p['param_name']}\"] = {p['current_value']}")
                self._save(PROPOSAL_LOG, proposals)
                return True

        return False

    def check_auto_rollbacks(self):
        """检查是否有提案需要自动回滚"""
        proposals = self._load(PROPOSAL_LOG)
        today = datetime.now().strftime("%Y%m%d")
        rollbacks = []

        for p in proposals:
            if p.get("status") == "approved" and p.get("rollback_date", "99991231") <= today:
                rollbacks.append(p)
                p["status"] = "auto_rolled_back"
                print(f"[ReviewerV2] ⚠️ 提案 {p['id']} 已到期自动回滚")
                print(f"  请恢复: {p['module']}.{p['param_name']} = {p['current_value']}")

        if rollbacks:
            self._save(PROPOSAL_LOG, proposals)

        return rollbacks

    def _log_param_change(self, proposal: Dict):
        """记录参数变更历史"""
        history = self._load(PARAM_HISTORY)
        history.append({
            "date": datetime.now().strftime("%Y%m%d"),
            "proposal_id": proposal.get("id"),
            "module": proposal.get("module"),
            "param": proposal.get("param_name"),
            "old_value": proposal.get("current_value"),
            "new_value": proposal.get("proposed_value"),
            "reason": proposal.get("reason"),
        })
        self._save(PARAM_HISTORY, history)

    # ────────────────────────────────────────────────────────
    #  7. 完整周报
    # ────────────────────────────────────────────────────────

    def run_weekly_review(
        self,
        current_params: Dict = None,
        fetcher=None,
        verbose: bool = True,
    ) -> Dict:
        """
        执行完整的周度复盘流程

        1. 回填本周信号结果
        2. 计算信号记分卡
        3. 计算辩论正确率
        4. 生成复盘报告 (LLM 或规则)
        5. 检查到期回滚

        Returns: 完整复盘结果
        """
        if verbose:
            print("=" * 65)
            print("  📊 周度策略复盘")
            print("=" * 65)

        # Step 1: 回填
        if fetcher:
            updated = self.update_outcomes(fetcher)
            if verbose:
                print(f"\n[Step 1] 回填信号结果: {updated}条更新")

        # Step 2: 记分卡
        scorecards = self.compute_scorecards()
        if verbose:
            print(f"\n[Step 2] 信号记分卡:")
            for name, sc in scorecards.items():
                print(f"  {sc.to_brief()}")

        # Step 3: 辩论正确率
        debate_stats = self.compute_debate_accuracy()
        if verbose:
            print(f"\n[Step 3] 辩论正确率:")
            if debate_stats.get("total_debates", 0) > 0:
                print(f"  总决策{debate_stats['total_debates']}次, "
                      f"正确率{debate_stats.get('accuracy', 'N/A')}%")
            else:
                print(f"  暂无辩论记录")

        # Step 4: 生成复盘
        review = self.generate_weekly_review(scorecards, debate_stats, current_params)
        if verbose:
            print(f"\n[Step 4] 复盘总结:")
            print(f"  {review.get('summary', 'N/A')}")
            proposals = review.get("proposals", [])
            if proposals:
                print(f"\n  参数变更提案 ({len(proposals)}个):")
                for p in proposals:
                    print(f"  📋 [{p.get('module')}.{p.get('param_name')}] "
                          f"{p.get('current_value')} → {p.get('proposed_value')} "
                          f"({p.get('change_pct'):+.0f}%)")
                    print(f"     理由: {p.get('reason')}")
                    print(f"     状态: {p.get('status')} | 回滚日期: {p.get('rollback_date')}")
            else:
                print(f"\n  暂无参数调整建议")

            # 保存提案
            if proposals:
                all_proposals = self._load(PROPOSAL_LOG)
                all_proposals.extend(proposals)
                self._save(PROPOSAL_LOG, all_proposals)

        # Step 5: 检查到期回滚
        rollbacks = self.check_auto_rollbacks()
        if rollbacks and verbose:
            print(f"\n[Step 5] ⚠️ {len(rollbacks)}个提案到期需回滚")

        if verbose:
            print(f"\n{'='*65}")

        return review

    # ────────────────────────────────────────────────────────
    #  to_brief (给 debate 上下文用)
    # ────────────────────────────────────────────────────────

    def to_brief(self) -> str:
        """
        压缩最近复盘结果 (~80 tokens)
        供 debate.py 上下文使用
        """
        scorecards = self.compute_scorecards(lookback_days=14)
        all_sc = scorecards.get("全部信号")

        if all_sc is None or all_sc.total == 0:
            return "[复盘] 暂无历史信号数据"

        debate_stats = self.compute_debate_accuracy(lookback_days=14)

        parts = [
            f"[复盘] 近14日: {all_sc.total}只信号 "
            f"胜率{all_sc.win_rate:.0%} "
            f"均5日{all_sc.avg_ret_5d:+.1f}% "
            f"止损率{all_sc.stops/max(all_sc.total,1):.0%}"
        ]

        # 最优/最差信号类型
        type_cards = {k: v for k, v in scorecards.items() if k != "全部信号" and v.total >= 3}
        if type_cards:
            best = max(type_cards.items(), key=lambda x: x[1].win_rate)
            worst = min(type_cards.items(), key=lambda x: x[1].win_rate)
            parts.append(f"  最优:{best[0]}({best[1].win_rate:.0%}) "
                         f"最差:{worst[0]}({worst[1].win_rate:.0%})")

        if debate_stats.get("accuracy"):
            parts.append(f"  辩论正确率{debate_stats['accuracy']}%")

        return "\n".join(parts)

    # ────────────────────────────────────────────────────────
    #  IO 工具
    # ────────────────────────────────────────────────────────

    def _load(self, path: Path) -> list:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _save(self, path: Path, data: list):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
