#!/usr/bin/env python3
"""
Opus AI 自进化循环 v3.1

完整流程:
  1. EVALUATE  — 用生产引擎+300天真实数据回测当前v3.1策略, 得出性能基线
  2. REFLECT   — 调用 Opus AI (OpenRouter) 分析表现, 诊断问题, 提出变异方案
  3. MUTATE    — 将 Opus 的参数变异建议应用到 SignalConfig, 生成候选版本
  4. SANDBOX   — 用生产引擎回测所有候选版本
  5. PROMOTE   — 如果候选版本显著优于基线, 提升为新版本

使用:
  cd new && python scripts/run_evolution_test.py

v3.1 重构:
  - 复用 backtest_v2_vs_v3.BacktestEngine (生产引擎, 含前复权/LHB/T+1/缺口保护)
  - 删除简化回测引擎, 避免两套引擎结果不一致
  - PE注入已验证实证约束 + 进化历史记忆
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── 导入生产引擎 ──
from scripts.backtest_v2_vs_v3 import BacktestEngine, load_all_data
from signal.signal_generator import SignalConfig

# ── Load .env ──
env_path = PROJECT_ROOT / "config" / ".env"
if env_path.exists():
    for line in env_path.read_text().strip().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL_PRIMARY", "anthropic/claude-opus-4-6")
PROXY_URL = os.environ.get("OPENROUTER_PROXY_URL", "")

# 优先使用300天+龙虎榜数据集, 回测区间更长、统计更可靠
_CSV_CANDIDATES = [
    PROJECT_ROOT / "data" / "csi1000_market_bundle_300d_lhb.csv",
    PROJECT_ROOT / "data" / "csi1000_market_bundle_300d.csv",
    PROJECT_ROOT / "data" / "csi1000_market_bundle_100d.csv",
    PROJECT_ROOT / "data" / "csi1000_market_bundle.csv",
]
CSV_PATH = next((p for p in _CSV_CANDIDATES if p.exists()), _CSV_CANDIDATES[-1])
EVOLUTION_HISTORY_PATH = PROJECT_ROOT / "backtest" / "evolution_history.json"


# ══════════════════════════════════════════════════════════════════════════════
#  进化配置参数 (轻量包装, 映射到 SignalConfig)
# ══════════════════════════════════════════════════════════════════════════════

# 可调参数及其约束
TUNABLE_PARAMS = {
    "lookback_days": {"type": "list_int"},
    "lookback_weights": {"type": "list_float"},
    "volatility_penalty": {"type": "float", "min": 0, "max": 3.0},
    "top_n": {"type": "int", "min": 3, "max": 25},
    "liquidity_weight": {"type": "float", "min": 0, "max": 0.5},
    "hold_buffer_ratio": {"type": "float", "min": 1.0, "max": 2.5},
    "rebalance_interval_days": {"type": "int", "min": 2, "max": 15},
    "max_total_position": {"type": "float", "min": 0.3, "max": 1.0},
    "max_single_weight": {"type": "float", "min": 0.03, "max": 0.4},
}

# 固化参数 — 不可调
FROZEN_PARAMS = {"stop_loss_pct", "open_gap_limit", "slippage_pct"}

# v3.1 基线 SignalConfig
def make_baseline_config(label: str = "v3.2_baseline") -> SignalConfig:
    return SignalConfig(
        data_csv_path="__backtest__",
        adaptive_weights=False,
        enable_reversal_filter=False,
        enable_trend_window=False,
        stop_loss_pct=-0.99,
        lookback_weights=[0.5, 0.3, 0.2],
        liquidity_weight=0.15,          # v3.2: 进化验证
        hold_buffer_ratio=1.2,          # v3.2: 进化验证
        enable_lhb_factor=True,
        lhb_weight=0.30,
        lhb_lookback=10,
        lhb_negative_penalty=0.3,
        enable_macro_calendar=False,
        enable_breaking_monitor=False,
    )

_mutation_counter = 0

def make_mutated_config(base: SignalConfig, changes: dict, label: str = "") -> SignalConfig:
    """从基线配置生成变异版本"""
    global _mutation_counter
    _mutation_counter += 1
    # 复制基线, 覆盖变更参数
    new_cfg = SignalConfig(
        data_csv_path="__backtest__",
        adaptive_weights=base.adaptive_weights,
        enable_reversal_filter=base.enable_reversal_filter,
        enable_trend_window=base.enable_trend_window,
        stop_loss_pct=base.stop_loss_pct,
        lookback_days=list(changes.get("lookback_days", base.lookback_days)),
        lookback_weights=list(changes.get("lookback_weights", base.lookback_weights)),
        volatility_penalty=changes.get("volatility_penalty", base.volatility_penalty),
        top_n=changes.get("top_n", base.top_n),
        max_single_weight=changes.get("max_single_weight", base.max_single_weight),
        max_total_position=changes.get("max_total_position", base.max_total_position),
        liquidity_weight=changes.get("liquidity_weight", base.liquidity_weight),
        hold_buffer_ratio=changes.get("hold_buffer_ratio", base.hold_buffer_ratio),
        rebalance_interval_days=changes.get("rebalance_interval_days", base.rebalance_interval_days),
        enable_lhb_factor=base.enable_lhb_factor,
        lhb_weight=base.lhb_weight,
        lhb_lookback=base.lhb_lookback,
        lhb_negative_penalty=base.lhb_negative_penalty,
        enable_macro_calendar=False,
        enable_breaking_monitor=False,
    )
    return new_cfg


# ══════════════════════════════════════════════════════════════════════════════
#  回测包装 (复用生产引擎)
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest_with_engine(config: SignalConfig, data: dict,
                              label: str = "", verbose: bool = False) -> dict:
    """使用生产 BacktestEngine 运行回测, 返回标准化指标"""
    # 确定回测区间 (跳过前25天回望期)
    trade_dates = data["trade_dates"]
    if len(trade_dates) > 30:
        start_date = trade_dates[25]
        end_date = trade_dates[-1]
    else:
        start_date = trade_dates[0]
        end_date = trade_dates[-1]

    bt = BacktestEngine("custom", data, custom_config=config)
    bt.run(start_date, end_date)
    raw_m = bt.calc_metrics()

    if not raw_m:
        return {"success": False, "name": label}

    # 基准收益
    idx = data["index_data"]
    idx_start = idx[idx["trade_date"] >= start_date].head(1)
    idx_end = idx[idx["trade_date"] <= end_date].tail(1)
    if not idx_start.empty and not idx_end.empty:
        bench_ret = (float(idx_end.iloc[0]["close"]) / float(idx_start.iloc[0]["close"]) - 1) * 100
    else:
        bench_ret = 0

    total_ret = raw_m["total_return"] * 100  # 转为百分比
    sharpe = raw_m["sharpe"]
    max_dd = raw_m["max_drawdown"] * 100
    n_trades = raw_m["total_trades"]

    result = {
        "success": True,
        "name": label or "unknown",
        "total_return": round(total_ret, 2),
        "excess_return": round(total_ret - bench_ret, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_dd, 2),
        "n_trades": n_trades,
        "final_value": round(raw_m["final_value"], 2),
        "benchmark_return": round(bench_ret, 2),
    }
    if verbose:
        print(f"  [{label}] ret={total_ret:+.2f}% excess={total_ret-bench_ret:+.2f}% "
              f"sharpe={sharpe:.2f} mdd={max_dd:.2f}% trades={n_trades}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Step 2: REFLECT — Opus AI 反思
# ══════════════════════════════════════════════════════════════════════════════

REFLECTION_SYSTEM = """你是 Opus, 一位专精A股动量轮动策略的量化分析师。

你的任务是分析策略回测表现, 诊断问题, 提出参数变异方案。

当前策略核心逻辑 (v3.2):
- 中证1000成分股宇宙, 复合动量评分 (多周期加权 + 波动率惩罚 + 流动性加权 + 龙虎榜增强)
- 周度调仓(每5交易日), 持有TOP N, 等权分配, 缓冲带保留
- 无止损 (已验证止损有害, 详见下方)
- 高开缺口保护 (>5%不追)
- 市场状态自适应: HALT=清仓, 其他状态正常调仓(不降仓)

══ 已验证的实证结论 (不可推翻, 不要在这些方向上浪费变异) ══

1. 止损有害: 13个月回测+实盘验证, -8%止损导致36%交易被错杀, 所有被止损的股票均在3天内回正。
   stop_loss_pct 固定为 -0.99, 不作为可调参数。

2. 等权分配最优: 集中加权(score-weighted)夏普0.77, 分层加权(tiered)0.81, 等权(equal)1.25。
   集中持仓放大波动而非alpha。max_single_weight 应≈1/top_n, 不要做集中化。

3. 缺口保护有效: open_gap_limit=0.05 将夏普从1.05提升至1.25, 最大回撤从-32.5%降至-24.2%。
   此参数已固定, 不作为可调参数。

4. 宏观静默期有害: 在FOMC/LPR日暂停调仓导致年化从41.7%暴跌至17.4%。
   不要建议在宏观事件日暂停或缩减调仓。

5. 回撤买入反动量: 所有限价/回撤等待模式均远差于开盘买入。动量股不回调, 等待=买在动量衰退时。
   不要建议"逢低买入"或"等回调再进"。

6. 周一开盘买优于周五买: 周五买入年化仅4.34%(vs周一15.25%), 因为用的是更旧一天的数据。
   买入时点不作为可调参数。

══ 你可以调整的参数 ══
- lookback_days: 回望周期列表 (如 [5,10,20] 或 [10,20,60])
- lookback_weights: 对应权重 (需和为1.0)
- volatility_penalty: 波动率惩罚系数 (0~2.0)
- top_n: 持仓数量 (5~20, 注意同步调整 max_single_weight≈1/top_n)
- liquidity_weight: 流动性权重 (0~0.3)
- hold_buffer_ratio: 缓冲带比率 (1.0~2.0)
- rebalance_interval_days: 调仓间隔 (3~10)
- max_total_position: 最大总仓位 (0.5~1.0)

══ 不可调参数 (已固化) ══
- stop_loss_pct: 固定 -0.99 (不可调)
- open_gap_limit: 固定 0.05 (不可调)
- slippage_pct: 固定 0.002 (不可调)
- max_single_weight: 应随 top_n 同步调整为 ≈1/top_n (等权)

重要约束:
- 每个变异方案最多改 3 个参数
- 变化幅度保持渐进 (±10~30%)
- 需要给出明确的理论依据和可证伪的预期
- 回望权重必须和为1.0
- 如果调整 top_n, 必须同步调整 max_single_weight = round(1/top_n, 3)
- 不要重复已被否定的方向 (参考下方历史进化记录)

置信度校准:
- confidence 字段必须附带 confidence_rationale 说明为什么是这个数字
- 高置信(>0.7): 有明确的理论依据+历史数据支持+低风险
- 中置信(0.4-0.7): 理论合理但缺乏直接验证
- 低置信(<0.4): 探索性尝试, 理论依据不充分

宏观环境感知:
- 下方会附带近期宏观事件日历和舆情情绪数据 (如有)
- 宏观信息仅用于判断 market_regime, 不要建议因宏观事件暂停调仓或降仓
- 宏观判断应体现在 market_regime 字段中

输出严格JSON, 不要加markdown代码块。"""

REFLECTION_USER = """分析以下回测结果并提出 3~5 个变异方案:

当前配置 (v3.2):
  回望周期: {lookback_days}  权重: {lookback_weights}
  波动率惩罚: {volatility_penalty}
  持仓数: {top_n}  单只上限: {max_single_weight:.1%} (等权≈1/top_n)
  流动性权重: {liquidity_weight}
  缓冲带: {hold_buffer_ratio}x
  调仓间隔: {rebalance_interval_days}天
  总仓位上限: {max_total_position:.0%}
  (止损/缺口保护/滑点 已固化, 不可调)

回测表现 (CSI1000, 13个月, 生产引擎):
  收益率: {total_return:+.2f}%
  超额收益 (vs CSI300): {excess_return:+.2f}%
  夏普比率: {sharpe:.2f}
  最大回撤: {max_drawdown:.2f}%
  交易次数: {n_trades}
  CSI300基准: {benchmark_return:+.2f}%

{macro_context}

{evolution_history}

请输出JSON:
{{
  "analysis": "对当前策略表现的综合分析 (200字以内)",
  "market_regime": "对当前市场环境的判断",
  "key_issues": ["问题1", "问题2", ...],
  "opportunities": ["机会1", "机会2", ...],
  "proposals": [
    {{
      "name": "变异方案简称",
      "rationale": "理论依据",
      "parameter_changes": {{"param_name": new_value, ...}},
      "expected_impact": "预期效果 (必须包含预期夏普/超额/回撤的数值范围)",
      "confidence": 0.0-1.0,
      "confidence_rationale": "为什么是这个置信度 (50字以内)",
      "risk_level": "low|medium|high"
    }},
    ...
  ]
}}"""


# ══════════════════════════════════════════════════════════════════════════════
#  进化历史记忆
# ══════════════════════════════════════════════════════════════════════════════

def load_evolution_history() -> str:
    """加载历史进化记录摘要, 注入prompt避免重复试错"""
    if not EVOLUTION_HISTORY_PATH.exists():
        return ""
    try:
        history = json.loads(EVOLUTION_HISTORY_PATH.read_text("utf-8"))
        if not history:
            return ""
        lines = ["══ 历史进化记录 (避免重复已失败的方向) ══"]
        for entry in history[-5:]:  # 只保留最近5次
            lines.append(f"  周期 {entry.get('cycle', '?')} ({entry.get('date', '?')}):")
            lines.append(f"    基线: 夏普{entry.get('baseline_sharpe', '?')}, 超额{entry.get('baseline_excess', '?')}%")
            for m in entry.get("mutations", []):
                status = "✅胜出" if m.get("promoted") else "❌失败"
                lines.append(f"    {status} {m['name']}: {m['changes']} → 夏普{m['sharpe']}, 超额{m['excess']}%")
            if entry.get("lesson"):
                lines.append(f"    教训: {entry['lesson']}")
        return "\n".join(lines)
    except Exception as e:
        print(f"[历史] 加载失败: {e}")
        return ""


def save_evolution_history(cycle_data: dict):
    """追加本次进化周期到历史记录"""
    history = []
    if EVOLUTION_HISTORY_PATH.exists():
        try:
            history = json.loads(EVOLUTION_HISTORY_PATH.read_text("utf-8"))
        except:
            history = []
    history.append(cycle_data)
    # 只保留最近20次
    history = history[-20:]
    EVOLUTION_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVOLUTION_HISTORY_PATH.write_text(json.dumps(history, ensure_ascii=False, indent=2), "utf-8")


# ══════════════════════════════════════════════════════════════════════════════
#  Opus API 调用
# ══════════════════════════════════════════════════════════════════════════════

def call_opus_reflection(config: SignalConfig, metrics: dict,
                         trade_date: str = None) -> dict:
    """Call Opus AI via OpenRouter API for strategy reflection."""
    print("\n[Opus] 调用 Claude Opus 进行策略反思...")

    # 生成宏观上下文
    macro_context = ""
    if trade_date:
        try:
            # 避免与 Python 内置 signal 模块冲突: 用绝对路径导入
            # 关键: 必须先注册到 sys.modules, 否则 Python 3.10 的 @dataclass +
            # from __future__ import annotations 会报 NoneType.__dict__ 错误
            import importlib.util
            macro_mod_path = PROJECT_ROOT / "signal" / "macro_calendar.py"
            if macro_mod_path.exists():
                mod_name = "signal_macro_calendar"  # 避免与内置 signal 冲突
                spec = importlib.util.spec_from_file_location(mod_name, str(macro_mod_path))
                mod = importlib.util.module_from_spec(spec)
                sys.modules[mod_name] = mod  # 注册到 sys.modules, 解决 dataclass 反射问题
                spec.loader.exec_module(mod)
                cal = mod.MacroCalendar()
                macro_context = cal.format_opus_context(trade_date)
                print(f"[Opus] 已注入宏观事件上下文 ({trade_date})")
            else:
                print(f"[Opus] macro_calendar.py 不存在: {macro_mod_path}")
        except Exception as e:
            print(f"[Opus] 宏观上下文加载失败 (非致命): {e}")
            traceback.print_exc()
            macro_context = ""

    # 加载历史进化记录
    evolution_history = load_evolution_history()
    if evolution_history:
        print(f"[Opus] 已注入历史进化记录")

    user_msg = REFLECTION_USER.format(
        lookback_days=config.lookback_days,
        lookback_weights=config.lookback_weights,
        volatility_penalty=config.volatility_penalty,
        top_n=config.top_n,
        max_single_weight=config.max_single_weight,
        liquidity_weight=config.liquidity_weight,
        hold_buffer_ratio=config.hold_buffer_ratio,
        rebalance_interval_days=config.rebalance_interval_days,
        max_total_position=config.max_total_position,
        total_return=metrics["total_return"],
        excess_return=metrics["excess_return"],
        sharpe=metrics["sharpe"],
        max_drawdown=metrics["max_drawdown"],
        n_trades=metrics["n_trades"],
        benchmark_return=metrics["benchmark_return"],
        macro_context=macro_context,
        evolution_history=evolution_history,
    )

    payload = {
        "model": OPENROUTER_MODEL,
        "max_tokens": 4096,
        "temperature": 0.4,
        "messages": [
            {"role": "system", "content": REFLECTION_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://stock-evolution.local",
        "X-Title": "Momentum Evolution v3.1",
    }

    url = f"{OPENROUTER_BASE_URL}/chat/completions"
    data = json.dumps(payload).encode("utf-8")

    # Try with proxy first, then without
    proxy_urls = []
    if PROXY_URL:
        proxy_urls.append(PROXY_URL)
    proxy_urls.append(None)

    response_text = None
    for proxy in proxy_urls:
        try:
            if proxy:
                from urllib.request import ProxyHandler, build_opener
                proxy_handler = ProxyHandler({"http": proxy, "https": proxy})
                opener = build_opener(proxy_handler)
                req = Request(url, data=data, headers=headers)
                resp = opener.open(req, timeout=120)
            else:
                req = Request(url, data=data, headers=headers)
                resp = urlopen(req, timeout=120)

            body = resp.read().decode("utf-8")
            result = json.loads(body)
            response_text = result["choices"][0]["message"]["content"]
            tokens = result.get("usage", {})
            print(f"[Opus] 响应 {len(response_text)} 字符 | tokens: in={tokens.get('prompt_tokens',0)} out={tokens.get('completion_tokens',0)}")
            break
        except Exception as e:
            print(f"[Opus] {'代理' if proxy else '直连'}失败: {e}")
            continue

    if not response_text:
        raise RuntimeError("Opus API 调用失败, 所有连接方式均不可用")

    # Parse JSON
    text = response_text.strip()
    if text.startswith("```"): text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"): text = text[:-3]
    if text.startswith("json"): text = text[4:]

    return json.loads(text.strip())


# ══════════════════════════════════════════════════════════════════════════════
#  参数校验
# ══════════════════════════════════════════════════════════════════════════════

def validate_proposal(proposal: dict, base_config: SignalConfig) -> dict | None:
    """验证并清洗Opus提出的参数变异"""
    changes = proposal.get("parameter_changes", {})
    if not changes:
        return None

    # v3.1: 固化参数 — Opus不允许调整
    for frozen in list(changes.keys()):
        if frozen in FROZEN_PARAMS:
            print(f"  [拒绝] {frozen} 是固化参数, 不可调整")
            del changes[frozen]

    valid = {}
    for param, val in changes.items():
        spec = TUNABLE_PARAMS.get(param)
        if not spec:
            print(f"  [跳过] 未知或不可调参数: {param}")
            continue

        try:
            if spec["type"] == "list_float":
                if not isinstance(val, list): continue
                val = [float(v) for v in val]
                if param == "lookback_weights":
                    if abs(sum(val) - 1.0) > 0.01:
                        print(f"  [修正] lookback_weights 和不为1, 归一化")
                        s = sum(val)
                        val = [v/s for v in val] if s > 0 else val
                valid[param] = val
            elif spec["type"] == "list_int":
                if not isinstance(val, list): continue
                val = [int(v) for v in val]
                valid[param] = val
            elif spec["type"] == "float":
                val = float(val)
                if val < spec["min"] or val > spec["max"]:
                    print(f"  [跳过] {param}={val} 超出范围 [{spec['min']}, {spec['max']}]")
                    continue
                valid[param] = val
            elif spec["type"] == "int":
                val = int(val)
                if val < spec["min"] or val > spec["max"]:
                    print(f"  [跳过] {param}={val} 超出范围 [{spec['min']}, {spec['max']}]")
                    continue
                valid[param] = val
        except (ValueError, TypeError) as e:
            print(f"  [跳过] {param}={val}: {e}")
            continue

    if not valid:
        return None

    # v3.1: 如果调了top_n, 强制同步max_single_weight为等权
    if "top_n" in valid:
        eq_weight = round(1.0 / valid["top_n"], 3)
        if "max_single_weight" not in valid or abs(valid.get("max_single_weight", 0) - eq_weight) > 0.02:
            print(f"  [同步] top_n={valid['top_n']} → max_single_weight={eq_weight} (等权)")
            valid["max_single_weight"] = eq_weight

    return valid


# ══════════════════════════════════════════════════════════════════════════════
#  Main: Complete Evolution Cycle
# ══════════════════════════════════════════════════════════════════════════════

def run_evolution_cycle():
    print("=" * 70)
    print("  Opus AI 自进化循环 v3.1 (生产引擎)")
    print(f"  数据: {CSV_PATH.name}")
    print("=" * 70)

    if not OPENROUTER_API_KEY:
        print("[错误] 未找到 OPENROUTER_API_KEY, 请在 config/.env 中配置")
        return

    if not CSV_PATH.exists():
        print(f"[错误] 数据文件不存在: {CSV_PATH}")
        return

    # Load data (生产引擎的数据格式, 含前复权/龙虎榜)
    data = load_all_data(str(CSV_PATH))

    # ── Step 1: EVALUATE — 基线回测 ──
    print("\n" + "─" * 70)
    print("  Step 1: EVALUATE — 运行 v3.1 基线回测 (生产引擎)")
    print("─" * 70)
    baseline_config = make_baseline_config()
    baseline_metrics = run_backtest_with_engine(
        baseline_config, data, label="v3.2_baseline", verbose=True
    )
    if not baseline_metrics.get("success"):
        print("[错误] 基线回测失败")
        return
    print(f"\n  基线结果: 收益{baseline_metrics['total_return']:+.2f}%, "
          f"夏普{baseline_metrics['sharpe']:.2f}, "
          f"回撤{baseline_metrics['max_drawdown']:.2f}%, "
          f"交易{baseline_metrics['n_trades']}笔")

    # ── Step 2: REFLECT — Opus AI 反思 ──
    print("\n" + "─" * 70)
    print("  Step 2: REFLECT — Opus AI 分析与变异提案")
    print("─" * 70)
    reflection = None
    # 用最后交易日作为宏观上下文日期
    last_date = data["trade_dates"][-1] if data["trade_dates"] else "20260324"
    try:
        reflection = call_opus_reflection(baseline_config, baseline_metrics,
                                             trade_date=last_date)
    except Exception as e:
        print(f"[警告] Opus API 不可用: {e}")
        traceback.print_exc()
        # Fallback: load from pre-generated reflection file
        fallback_path = PROJECT_ROOT / "data" / "opus_reflection.json"
        if fallback_path.exists():
            print(f"[Fallback] 加载预生成反思文件: {fallback_path}")
            reflection = json.loads(fallback_path.read_text("utf-8"))
        else:
            print("[错误] 无预生成反思文件, 无法继续")
            return
    if reflection is None:
        return

    print(f"\n[Opus 分析] {reflection.get('analysis', 'N/A')}")
    print(f"[市场判断] {reflection.get('market_regime', 'N/A')}")
    print(f"[关键问题] {reflection.get('key_issues', [])}")
    print(f"[机会] {reflection.get('opportunities', [])}")
    proposals = reflection.get("proposals", [])
    print(f"[变异方案] {len(proposals)} 个")

    for i, p in enumerate(proposals):
        print(f"\n  方案{i+1}: {p.get('name', f'proposal_{i+1}')}")
        print(f"    理由: {p.get('rationale', '')}")
        print(f"    参数: {p.get('parameter_changes', {})}")
        print(f"    预期: {p.get('expected_impact', '')}")
        conf = p.get('confidence', 0)
        print(f"    信心: {conf:.0%} | 风险: {p.get('risk_level', 'medium')}")
        print(f"    置信依据: {p.get('confidence_rationale', 'N/A')}")

    # ── Step 3 & 4: MUTATE + SANDBOX — 变异并回测 ──
    print("\n" + "─" * 70)
    print("  Step 3-4: MUTATE + SANDBOX — 测试所有变异方案 (生产引擎)")
    print("─" * 70)

    all_results = [{"metrics": baseline_metrics, "label": "v3.2_baseline",
                    "config": baseline_config, "proposal": None, "changes": {}}]

    for i, proposal in enumerate(proposals):
        name = proposal.get('name', f'mutation_{i+1}')
        print(f"\n  测试方案 {i+1}/{len(proposals)}: {name}")
        changes = validate_proposal(proposal, baseline_config)
        if not changes:
            print(f"    [跳过] 无有效参数变更")
            continue

        # lookback_days 和 lookback_weights 必须同时变
        if "lookback_weights" in changes and "lookback_days" not in changes:
            changes["lookback_days"] = list(baseline_config.lookback_days)
        if "lookback_days" in changes and "lookback_weights" not in changes:
            n = len(changes["lookback_days"])
            changes["lookback_weights"] = [round(1/n, 3)] * n
            # 修正精度
            changes["lookback_weights"][-1] = round(1 - sum(changes["lookback_weights"][:-1]), 3)

        # 确保长度一致
        if "lookback_days" in changes and "lookback_weights" in changes:
            if len(changes["lookback_days"]) != len(changes["lookback_weights"]):
                print(f"    [修正] lookback_days/weights 长度不一致, 跳过")
                continue

        try:
            mutated_config = make_mutated_config(baseline_config, changes, label=name)
            print(f"    配置: {changes}")
            metrics = run_backtest_with_engine(mutated_config, data, label=name, verbose=True)
            all_results.append({
                "metrics": metrics, "label": name,
                "config": mutated_config, "proposal": proposal, "changes": changes,
            })
        except Exception as e:
            print(f"    [错误] 回测失败: {e}")
            traceback.print_exc()

    # ── Step 5: PROMOTE — 选出最优 ──
    print("\n" + "─" * 70)
    print("  Step 5: PROMOTE — 比较结果并选择最优版本")
    print("─" * 70)

    # Composite score: 0.4*return + 0.3*sharpe + 0.3*(1 - |drawdown|/100)
    def score(m):
        if not m.get("success"): return -999
        return (
            max(0, m["total_return"]) / 100 * 0.4
            + max(0, m["sharpe"]) / 10 * 0.3
            + max(0, 1 + m["max_drawdown"] / 100) * 0.3
        )

    print(f"\n  {'版本':<30s} {'收益':>8s} {'超额':>8s} {'夏普':>7s} {'回撤':>8s} {'交易':>5s} {'得分':>7s}")
    print(f"  {'─'*80}")

    for r in all_results:
        m = r["metrics"]
        if not m.get("success"): continue
        s = score(m)
        is_baseline = r["label"] == "v3.2_baseline"
        marker = " ◀基线" if is_baseline else ""
        print(f"  {r['label']:<30s} {m['total_return']:>+7.2f}% {m['excess_return']:>+7.2f}% "
              f"{m['sharpe']:>6.2f} {m['max_drawdown']:>+7.2f}% {m['n_trades']:>5d} {s:>7.4f}{marker}")

    # Find best
    best = max(all_results, key=lambda r: score(r["metrics"]))
    best_m = best["metrics"]
    baseline_score = score(baseline_metrics)
    best_score = score(best_m)
    improvement_pct = ((best_score - baseline_score) / max(baseline_score, 0.001)) * 100

    print(f"\n  最优版本: {best['label']}")
    print(f"  综合得分: {best_score:.4f} vs 基线 {baseline_score:.4f} ({improvement_pct:+.1f}%)")

    # Promotion decision
    MIN_IMPROVEMENT = 5.0  # 需要5%以上提升才晋升
    promoted = False
    if best["label"] != "v3.2_baseline" and improvement_pct >= MIN_IMPROVEMENT:
        print(f"\n  ✅ 晋升! {best['label']} 获得 {improvement_pct:.1f}% 提升")
        print(f"     参数变更: {best.get('changes', {})}")
        print(f"     理由: {best['proposal'].get('rationale', '') if best.get('proposal') else ''}")
        promoted = True

        # Save promoted config
        promoted_path = PROJECT_ROOT / "backtest" / f"promoted_{best['label']}.json"
        promoted_path.parent.mkdir(parents=True, exist_ok=True)
        promoted_path.write_text(json.dumps({
            "label": best["label"],
            "parent": "v3.2_baseline",
            "changes": best.get("changes", {}),
            "metrics": best_m,
            "improvement_pct": improvement_pct,
            "proposal": best.get("proposal"),
            "promoted_at": datetime.now().isoformat(),
        }, ensure_ascii=False, indent=2), "utf-8")
        print(f"     已保存: {promoted_path}")

    elif best["label"] == "v3.2_baseline":
        print(f"\n  ⚡ 基线 v3.1 仍是最优, 无需晋升")
        print(f"     所有变异方案均未超越基线")
    else:
        print(f"\n  ⏳ {best['label']} 仅提升 {improvement_pct:.1f}%")
        print(f"     低于晋升阈值 {MIN_IMPROVEMENT}%, 保留当前版本")

    # Save full evolution log
    log_path = PROJECT_ROOT / "backtest" / "evolution_cycle_log.json"
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "engine": "production_backtest_v2_vs_v3",
        "data_file": CSV_PATH.name,
        "baseline": {"label": "v3.2_baseline", "metrics": baseline_metrics, "score": baseline_score},
        "reflection": reflection,
        "mutations": [],
        "best_version": best["label"],
        "best_score": best_score,
        "improvement_pct": improvement_pct,
        "promoted": promoted,
    }
    for r in all_results[1:]:
        log_data["mutations"].append({
            "label": r["label"],
            "changes": r.get("changes", {}),
            "metrics": r["metrics"],
            "score": score(r["metrics"]),
            "proposal": r.get("proposal"),
        })
    log_path.write_text(json.dumps(log_data, ensure_ascii=False, indent=2), "utf-8")
    print(f"\n  完整进化日志: {log_path}")

    # ── 保存进化历史 (供下次周期参考) ──
    history_entry = {
        "cycle": datetime.now().strftime("%Y%m%d"),
        "date": datetime.now().isoformat(),
        "baseline_sharpe": baseline_metrics.get("sharpe"),
        "baseline_excess": baseline_metrics.get("excess_return"),
        "mutations": [],
        "lesson": "",
    }
    for r in all_results[1:]:
        m = r["metrics"]
        p = r.get("proposal", {})
        history_entry["mutations"].append({
            "name": r["label"],
            "changes": r.get("changes", {}),
            "sharpe": m.get("sharpe"),
            "excess": m.get("excess_return"),
            "promoted": (r["label"] == best["label"] and promoted),
        })
    # 自动生成教训
    if not promoted:
        history_entry["lesson"] = "所有变异均未超越基线, 当前配置可能已接近局部最优"
    else:
        history_entry["lesson"] = f"{best['label']} 胜出, 关键变更: {best.get('changes', {})}"
    save_evolution_history(history_entry)

    print("\n" + "=" * 70)
    print("  自进化循环完成")
    print("=" * 70)


if __name__ == "__main__":
    run_evolution_cycle()
