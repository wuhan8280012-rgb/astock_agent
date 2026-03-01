#!/usr/bin/env python3
"""
openclaw_wrapper.py — OpenClaw 智能分析包装器
=============================================
四阶段数据流：
  Phase 1: memory-like-a-tree 记忆召回
  Phase 2: memU 偏好检索（可选）
  Phase 3: codex skills 实时数据采集
  Phase 4: LLM 综合分析

用法:
  python3 openclaw_wrapper.py 600519                # 完整四阶段分析
  python3 openclaw_wrapper.py 600519 --no-llm       # 仅汇总数据，跳过 LLM
  python3 openclaw_wrapper.py 600519 --json          # JSON 结构化输出
  python3 openclaw_wrapper.py 600519 --debug         # 各阶段耗时

依赖：llm_client.py (同目录), openai, subprocess
"""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor, wait, as_completed, TimeoutError as FuturesTimeoutError
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# ═══════════════════════════════════════════
# Step 1: 路径设置与 normalize_code()
# ═══════════════════════════════════════════

ASTOCK_DIR = os.path.expanduser("~/project/astock_agent")
sys.path.insert(0, ASTOCK_DIR)

# Skill 脚本路径
QUOTE_SCRIPT = os.path.expanduser("~/.codex/skills/stock-quote/scripts/quote.py")
MARKET_SCRIPT = os.path.expanduser("~/.codex/skills/stock-quote/scripts/market_overview.py")
NEWS_SCRIPT = os.path.expanduser("~/.codex/skills/finance-news/scripts/stock_news.py")
EARNINGS_SCRIPT = os.path.expanduser("~/.codex/skills/astock-earnings/scripts/earnings_analyzer.py")

# memory-like-a-tree
MEMORY_API_SCRIPT = os.path.expanduser(
    "~/project/memory-like-a-tree/core/memory_tree_api.py"
)
CONFIDENCE_DB = os.path.expanduser(
    "~/.local/share/ai-memory/memory-tree/data/confidence-db.json"
)

# memU
MEMU_SRC = os.path.expanduser("~/memU/src")
MEMU_CONFIG = os.path.expanduser("~/.local/share/ai-memory/memu/config.json")

# sediment (保存 leaf + Obsidian 同步)
SEDIMENT_SCRIPT = os.path.expanduser(
    "~/project/memory-like-a-tree/sediment/sediment.py"
)
OBSIDIAN_SYNC_SCRIPT = os.path.expanduser(
    "~/project/memory-like-a-tree/core/sync_workspace_to_obsidian.py"
)
OBSIDIAN_VAULT = os.path.expanduser("~/Documents/Obsidian/Quant_Notes")

AGENT_NAME = "astock"


def normalize_code(code: str) -> tuple:
    """
    规范化股票代码，返回 (纯数字代码, 带后缀代码, 市场标识)

    600519     → ("600519", "600519.SH", "SH")
    000001.SZ  → ("000001", "000001.SZ", "SZ")
    """
    code = code.strip().upper()

    if "." in code:
        parts = code.split(".")
        pure = parts[0]
        suffix = parts[1]
        return pure, code, suffix

    pure = code
    if pure.startswith(("6", "9")):
        return pure, f"{pure}.SH", "SH"
    elif pure.startswith(("0", "2", "3")):
        return pure, f"{pure}.SZ", "SZ"
    elif pure.startswith("8"):
        return pure, f"{pure}.BJ", "BJ"
    else:
        return pure, f"{pure}.SH", "SH"


# ═══════════════════════════════════════════
# Step 2: Phase 1 — memory-like-a-tree 记忆召回
# ═══════════════════════════════════════════

def phase1_memory_recall(stock_code: str) -> dict:
    """从记忆树检索交易规则和与该股票相关的记忆"""
    result = {
        "status": "error",
        "trading_rules": [],
        "stock_memories": [],
        "error": None,
    }

    # 主路径: subprocess 调用 memory_tree_api.py CLI
    try:
        stock_memories = _run_memory_search(stock_code, limit=5)
        rule_memories = _run_memory_search(
            "交易规则 环境评分 仓位", limit=5
        )

        if stock_memories is not None:
            result["stock_memories"] = stock_memories
        if rule_memories is not None:
            result["trading_rules"] = rule_memories

        if stock_memories is not None or rule_memories is not None:
            result["status"] = "ok" if (stock_memories and rule_memories) else "partial"
            return result
    except Exception as e:
        result["error"] = f"memory_tree_api subprocess: {e}"

    # 回退: 直接读 confidence-db.json
    try:
        result = _fallback_read_confidence_db(stock_code, result)
    except Exception as e:
        result["error"] = f"fallback read failed: {e}"

    return result


def _run_memory_search(query: str, limit: int = 5) -> list | None:
    """通过 subprocess 调用 memory_tree_api.py 搜索"""
    cmd = [
        sys.executable, MEMORY_API_SCRIPT,
        "--agent", AGENT_NAME,
        "search", query,
        "--limit", str(limit),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=15,
            cwd=os.path.dirname(MEMORY_API_SCRIPT),
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return json.loads(proc.stdout)
        return None
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        return None


def _fallback_read_confidence_db(stock_code: str, result: dict) -> dict:
    """直接读 confidence-db.json 提取记忆"""
    if not os.path.exists(CONFIDENCE_DB):
        result["error"] = f"confidence-db.json not found"
        return result

    with open(CONFIDENCE_DB, "r", encoding="utf-8") as f:
        db = json.load(f)

    memories = db.get("memories", {})

    for mem_id, mem in memories.items():
        preview = mem.get("content_preview", "")
        priority = mem.get("priority", "P2")

        # 提取交易规则（P0/P1）
        if priority in ("P0", "P1"):
            result["trading_rules"].append({
                "id": mem_id,
                "title": mem.get("title", ""),
                "priority": priority,
                "content_preview": preview[:500],
            })

        # 搜索与该股票代码相关的记忆
        pure = stock_code.split(".")[0] if "." in stock_code else stock_code
        if pure in preview or stock_code in preview:
            result["stock_memories"].append({
                "id": mem_id,
                "title": mem.get("title", ""),
                "content_preview": preview[:500],
            })

    if result["trading_rules"] or result["stock_memories"]:
        result["status"] = "partial"
        result["error"] = "used fallback (direct file read)"
    else:
        result["status"] = "partial"
        result["error"] = "no relevant memories found via fallback"

    return result


# ═══════════════════════════════════════════
# Step 3: Phase 2 — memU 偏好检索（可选）
# ═══════════════════════════════════════════

def phase2_memu_retrieve(stock_code: str) -> dict:
    """从 memU 检索个人交易偏好和生存规则"""
    result = {
        "status": "unavailable",
        "preferences": [],
        "error": None,
    }

    try:
        # 动态导入 memU
        if MEMU_SRC not in sys.path:
            sys.path.insert(0, MEMU_SRC)

        from memu.app.service import MemoryService

        # 读取 memU 配置
        if not os.path.exists(MEMU_CONFIG):
            result["error"] = f"memU config not found: {MEMU_CONFIG}"
            return result

        with open(MEMU_CONFIG, "r", encoding="utf-8") as f:
            config = json.load(f)

        service = MemoryService(
            database_config=config.get("database_config"),
            blob_config=config.get("blob_config"),
        )

        pure = stock_code.split(".")[0] if "." in stock_code else stock_code
        queries = [
            {"text": f"{pure} 交易偏好 风险控制"},
        ]

        loop = asyncio.new_event_loop()
        try:
            resp = loop.run_until_complete(service.retrieve(queries=queries))
        finally:
            loop.close()

        # 提取结果
        memories = resp.get("memories", [])
        if memories:
            result["status"] = "ok"
            result["preferences"] = [
                {
                    "content": m.get("content", ""),
                    "category": m.get("category", ""),
                    "relevance": m.get("relevance_score", 0),
                }
                for m in memories[:5]
            ]
        else:
            result["status"] = "empty"

    except Exception as e:
        result["error"] = str(e)

    return result


# ═══════════════════════════════════════════
# Step 4: Phase 3 — 实时数据采集
# ═══════════════════════════════════════════

def phase3_fresh_data(stock_code: str) -> dict:
    """通过 codex skills 获取实时行情、市场总览、个股新闻"""
    result = {
        "status": "error",
        "quote": None,
        "market_overview": None,
        "news": None,
        "errors": [],
    }

    pure, full, market = normalize_code(stock_code)

    # 1. 实时行情（优先 tushare）
    quote_data = _run_skill(QUOTE_SCRIPT, [pure, "--json", "--source", "tushare"])
    if quote_data is None:
        quote_data = _run_skill(QUOTE_SCRIPT, [pure, "--json"])
    if quote_data is not None:
        result["quote"] = quote_data
    else:
        result["errors"].append("quote.py failed or timed out")

    # 2. 市场总览（优先 tushare 内部回退链）
    market_data = _run_skill(MARKET_SCRIPT, ["--json"])
    if market_data is not None:
        result["market_overview"] = market_data
    else:
        result["errors"].append("market_overview.py failed or timed out")

    # 3. 个股新闻
    news_data = _run_skill(NEWS_SCRIPT, [pure, "--json", "--limit", "5"])
    if news_data is not None:
        result["news"] = news_data
    else:
        result["errors"].append("stock_news.py failed or timed out")

    # 4. 财报分析
    earnings_data = _run_skill(EARNINGS_SCRIPT, [pure, "--json", "--no-llm", "--quarters", "4"], timeout=60)
    if earnings_data is not None:
        result["earnings"] = earnings_data
    else:
        result["errors"].append("earnings_analyzer.py failed or timed out")

    # 状态判定
    ok_count = sum(1 for v in [result["quote"], result["market_overview"], result["news"]] if v)
    if ok_count == 3:
        result["status"] = "ok"
    elif ok_count > 0:
        result["status"] = "partial"

    return result


def _run_skill(script_path: str, args: list, timeout: int = 30) -> dict | None:
    """运行单个 skill 脚本，返回 JSON 或 None"""
    if not os.path.exists(script_path):
        return None

    cmd = [sys.executable, script_path] + args
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return _extract_json(proc.stdout)
        return None
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        return None


def _extract_json(text: str) -> dict | None:
    """从可能包含非 JSON 前缀的 stdout 中提取 JSON 对象"""
    text = text.strip()
    # 找到第一个 '{' 的位置
    idx = text.find("{")
    if idx == -1:
        return None
    try:
        return json.loads(text[idx:])
    except json.JSONDecodeError:
        return None


# ═══════════════════════════════════════════
# Step 5: Phase 4 — LLM 综合分析
# ═══════════════════════════════════════════

SYSTEM_PROMPT = """你是 Lala — wuhan 的全栈 A 股量化助手。

## 身份
INTJ。独立判断体，不是执行工具。wuhan 的方案不一定对，你负责质疑、找漏洞、推动决策落地。
wuhan：互联网大厂运营，趋势跟随交易者，目标是用量化方法在 A 股稳定盈利。

## 沟通规则（严格执行）
- 结论先行，再展开
- 能一句话说清的不写一段
- 能给数字的不给形容词
- 不确定的信息标注置信度（如：置信度 60%）
- 发现数据或逻辑漏洞，直接指出

## 禁止
- 恭维、模糊建议、重复用户说过的话
- 无数据支撑的乐观判断
- "您可以考虑…" 此类表达
- 装确定——不知道就说不知道

## A 股核心规则（不可违反）
- 环境评分 < 60: 只观察，不开仓
- 环境评分 60-75: 半仓操作
- 环境评分 ≥ 75: 可全仓
- 单股最大仓位 8%，单笔风险 1%
- T+1：今日买入不可今日卖出

## 报告结构
1. 环境评分（给数字，说依据）
2. 个股判断（趋势、量价、位置）
3. 漏洞/风险（wuhan 可能忽略的）
4. 操作建议（具体仓位 or 明确不操作的理由）"""

MAX_SECTION_LEN = 3000  # 每段最大字符数，防超 token

# ── 并行调度与超时 ──────────────────────────────────────────
TOTAL_BUDGET      = 60    # 全流程墙钟上限（秒）
LLM_HARD_TIMEOUT  = 13    # Phase4 LLM 单次硬超时（秒）
DATA_BUDGET       = TOTAL_BUDGET - LLM_HARD_TIMEOUT - 2  # 留 2s 组装余量 ≈ 45s
SKILL_TIMEOUTS    = {      # 各子进程独立 timeout（秒）
    "quote":    15,
    "market":   15,
    "news":     15,
    "earnings": 35,
    "memory":   12,
    "memu":     10,
}


def _load_cache_prices(ts_code: str, days: int = 30) -> list:
    """
    从 DataHub Parquet 缓存读近 N 日价格记录（BACKTEST 模式，不触发 API）。
    返回 PriceRecord 列表（空列表表示缓存未就绪）。
    """
    try:
        from datetime import timedelta
        from openclaw_os.data.datahub import DataHub, DataMode, BacktestAPIViolation
        hub = DataHub(
            cache_dir=os.path.join(ASTOCK_DIR, "data", "parquet"),
            mode=DataMode.BACKTEST,
        )
        end   = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=days + 10)).strftime("%Y%m%d")
        try:
            records = hub.get_price(ts_code, start, end)
            return records[-days:] if len(records) > days else records
        except BacktestAPIViolation:
            return []
    except Exception:
        return []


def extract_local_features(bundle: dict, ts_code: str = "") -> dict:
    """
    从 Phase3 结果 + DataHub 缓存计算本地特征，不调 LLM，不新增 API。

    返回字段：
        price, pct_chg, vs_ma20, dist_ma20_pct,
        vol_ratio, market_score, earnings_grade,
        limit_flag, atr, volatility_level
    """
    feat: dict = {
        "price":           None,
        "pct_chg":         None,
        "vs_ma20":         None,   # "above" / "below" / "flat"
        "dist_ma20_pct":   None,   # (price - MA20) / MA20 * 100，保留1位小数
        "vol_ratio":       None,   # vol_5d / vol_20d
        "market_score":    None,
        "earnings_grade":  None,
        "limit_flag":      None,   # "up" / "down" / None
        "atr":             None,   # 14日 ATR（元）
        "volatility_level": None,  # "低波" / "中波" / "高波"
    }

    p3 = bundle.get("phase3", {})

    # ── 行情基础字段 ──
    quote = p3.get("quote") or {}
    feat["price"]   = quote.get("latest_price")
    feat["pct_chg"] = quote.get("pct_change") or quote.get("pct_chg")

    # 涨跌停标记
    pct = feat["pct_chg"] or 0
    if pct >= 9.5:
        feat["limit_flag"] = "up"
    elif pct <= -9.5:
        feat["limit_flag"] = "down"

    # ── 市场分 ──
    mo = p3.get("market_overview") or {}
    breadth = mo.get("breadth") or {}
    _env = mo.get("env_score")
    feat["market_score"] = _env if _env is not None else breadth.get("score")

    # ── 财报评级 ──
    earnings = p3.get("earnings") or {}
    feat["earnings_grade"] = earnings.get("grade")

    # ── ATR / MA20 / vol_ratio 来自 DataHub 缓存 ──
    if ts_code:
        full_code = ts_code
        # 尝试补后缀
        if "." not in full_code:
            if full_code.startswith(("6", "9")):
                full_code = f"{full_code}.SH"
            elif full_code.startswith("8"):
                full_code = f"{full_code}.BJ"
            else:
                full_code = f"{full_code}.SZ"

        records = _load_cache_prices(full_code, days=30)

        if len(records) >= 5:
            closes  = [r.close  for r in records]
            volumes = [r.volume for r in records]

            # MA20
            if len(closes) >= 20:
                ma20 = sum(closes[-20:]) / 20
                price_now = feat["price"] or closes[-1]
                feat["dist_ma20_pct"] = round((price_now - ma20) / ma20 * 100, 1)
                feat["vs_ma20"] = (
                    "above" if feat["dist_ma20_pct"] > 0.5
                    else "below" if feat["dist_ma20_pct"] < -0.5
                    else "flat"
                )

            # ATR（14日）
            tr_list = []
            for i in range(1, min(15, len(records))):
                c, p_ = records[-i], records[-i - 1]
                tr = max(c.high - c.low, abs(c.high - p_.close), abs(c.low - p_.close))
                tr_list.append(tr)
            if tr_list:
                feat["atr"] = round(sum(tr_list) / len(tr_list), 2)
                price_ref = feat["price"] or closes[-1]
                atr_pct = feat["atr"] / price_ref * 100 if price_ref else 0
                feat["volatility_level"] = (
                    "高波" if atr_pct > 3
                    else "中波" if atr_pct > 1.5
                    else "低波"
                )

            # vol_ratio（5d/20d）
            if len(volumes) >= 20:
                v5  = sum(volumes[-5:])  / 5
                v20 = sum(volumes[-20:]) / 20
                feat["vol_ratio"] = round(v5 / v20, 2) if v20 else None

    return feat


def format_quick_report(features: dict, stock_code: str) -> str:
    """
    从本地特征生成 5 行快报，不调 LLM。
    格式：数字优先，结论先行（与 SYSTEM_PROMPT 风格一致）。
    """
    ts = datetime.now().strftime("%H:%M")
    lines = [f"⚡ {stock_code} 快报（{ts}，详细版稍后推飞书）\n"]

    # 1. 市场
    score = features.get("market_score")
    score_str = f"{score}/100" if score is not None else "N/A"
    lines.append(f"市场  {score_str}")

    # 2. 价位
    price   = features.get("price") if features.get("price") is not None else "N/A"
    pct     = features.get("pct_chg")
    pct_str = f"{'+'if pct and pct>0 else ''}{pct:.1f}%" if pct is not None else "N/A"
    vs      = features.get("vs_ma20", "N/A")
    dist    = features.get("dist_ma20_pct")
    dist_str = f"MA20 {'上方' if vs=='above' else '下方' if vs=='below' else '附近'} {abs(dist):.1f}%" if dist is not None else ""
    limit   = features.get("limit_flag")
    limit_str = " 【涨停】" if limit == "up" else " 【跌停】" if limit == "down" else ""
    sep = f" · {dist_str}" if dist_str else ""
    lines.append(f"价位  {price} ({pct_str}){limit_str}{sep}")

    # 3. 量价 + 波动
    vr  = features.get("vol_ratio")
    atr = features.get("atr")
    vol_level = features.get("volatility_level", "")
    vr_str  = f"量比 {vr}x {'放量' if vr and vr>1.2 else '缩量' if vr and vr<0.8 else '平量'}" if vr else "量比 N/A"
    atr_str = f"· ATR {atr} ({vol_level})" if atr else ""
    lines.append(f"量价  {vr_str} {atr_str}")

    # 4. 财报
    grade = features.get("earnings_grade") or "N/A"
    lines.append(f"财报  {grade}级" if grade != "N/A" else "财报  N/A")

    # 5. 操作建议（纯规则，无 LLM）
    score_val = score or 0
    if score_val < 60:
        advice = "环境 <60，禁止开仓，仅观察"
    elif score_val < 75:
        advice = "环境 60-75，半仓参与；" + ("涨停勿追" if limit == "up" else "注意量价背离" if vr and vr < 0.8 else "关注MA20支撑")
    else:
        advice = "环境 ≥75，可操作；" + ("注意追高风险" if limit == "up" else f"止损参考 MA20")

    lines.append(f"结论  {advice}")
    return "\n".join(lines)


def _async_llm_and_push(bundle: dict, stock_code: str) -> None:
    """后台推送占位符（Task 5 将替换此函数）"""
    pass


def phase4_llm_synthesis(context_bundle: dict, stock_code: str) -> str:
    """用 LLM 综合分析所有阶段数据"""
    from llm_client import get_llm

    llm = get_llm()

    # 构建 user prompt
    sections = []
    pure = stock_code.split(".")[0] if "." in stock_code else stock_code

    sections.append(f"# 分析目标: {pure}\n")

    # Phase 1: 记忆树规则
    p1 = context_bundle.get("phase1", {})
    if p1.get("trading_rules"):
        rules_text = "\n".join(
            f"- [{r.get('priority', 'P2')}] {r.get('title', '')}: {r.get('content_preview', '')}"
            for r in p1["trading_rules"]
        )
        sections.append(f"## 交易规则（记忆树）\n{rules_text[:MAX_SECTION_LEN]}")

    if p1.get("stock_memories"):
        mem_text = "\n".join(
            f"- {m.get('title', '')}: {m.get('content_preview', '')}"
            for m in p1["stock_memories"]
        )
        sections.append(f"## 相关记忆\n{mem_text[:MAX_SECTION_LEN]}")

    # Phase 2: memU 偏好
    p2 = context_bundle.get("phase2", {})
    if p2.get("preferences"):
        pref_text = "\n".join(
            f"- {p.get('content', '')}"
            for p in p2["preferences"]
        )
        sections.append(f"## 个人偏好（memU）\n{pref_text[:MAX_SECTION_LEN]}")

    # Phase 3: 实时数据
    p3 = context_bundle.get("phase3", {})
    if p3.get("quote"):
        quote_json = json.dumps(p3["quote"], ensure_ascii=False, indent=2)
        sections.append(f"## 实时行情\n```json\n{quote_json[:MAX_SECTION_LEN]}\n```")

    if p3.get("market_overview"):
        mo_json = json.dumps(p3["market_overview"], ensure_ascii=False, indent=2)
        sections.append(f"## 市场总览\n```json\n{mo_json[:MAX_SECTION_LEN]}\n```")

    if p3.get("news"):
        news_data = p3["news"]
        # 提取新闻列表
        if isinstance(news_data, dict):
            news_list = news_data.get("news", [])
        elif isinstance(news_data, list):
            news_list = news_data
        else:
            news_list = []

        news_text = "\n".join(
            f"- [{n.get('publish_time', '')}] {n.get('title', '')}"
            for n in news_list[:5]
        )
        sections.append(f"## 最新新闻\n{news_text[:MAX_SECTION_LEN]}")

    if p3.get("earnings"):
        earnings_json = json.dumps(p3["earnings"], ensure_ascii=False, indent=2)
        sections.append(f"## 财报分析\n```json\n{earnings_json[:MAX_SECTION_LEN]}\n```")

    user_prompt = "\n\n".join(sections)

    # 优先用 R1 深度推理，失败回退 V3
    try:
        return llm.reason(user_prompt, system=SYSTEM_PROMPT)
    except Exception as e:
        print(f"⚠️ R1 推理失败，回退 V3: {e}", file=sys.stderr)
        try:
            return llm.chat(user_prompt, system=SYSTEM_PROMPT, max_tokens=4000)
        except Exception as e2:
            return f"LLM 分析失败: {e2}"


def _phase4_with_timeout(bundle: dict, stock_code: str) -> str:
    """
    Phase4 LLM 调用，带 LLM_HARD_TIMEOUT 硬超时。
    超时立即抛 FuturesTimeoutError，不等待 LLM 线程完成。
    """
    p = ThreadPoolExecutor(max_workers=1)
    f = p.submit(phase4_llm_synthesis, bundle, stock_code)
    try:
        return f.result(timeout=LLM_HARD_TIMEOUT)
    finally:
        p.shutdown(wait=False, cancel_futures=False)


# ═══════════════════════════════════════════
# Step 5b: 保存 leaf + Obsidian 同步
# ═══════════════════════════════════════════

def save_leaf_and_sync(analysis: str, stock_code: str, bundle: dict) -> dict:
    """将分析结果保存为记忆树 leaf，并自动导出到 Obsidian vault"""
    result = {"sediment": "skipped", "obsidian_export": "skipped"}
    pure = stock_code.split(".")[0] if "." in stock_code else stock_code
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # 构建摘要内容（用于 MEMORY.md leaf）
    quote = bundle.get("phase3", {}).get("quote", {})
    price = quote.get("latest_price", "N/A")
    pct = quote.get("pct_change")
    name = quote.get("name", pure)
    pct_str = f"{'+' if pct and pct > 0 else ''}{pct:.2f}%" if pct is not None else "N/A"

    leaf_content = (
        f"**{name} ({pure})** {price} ({pct_str}) @ {ts}\n\n"
        f"{analysis[:2000]}"
    )
    title = f"OpenClaw {pure} {name} 分析"

    # Step 1: 通过 sediment.py 保存到 MEMORY.md + 触发索引
    try:
        cmd = [
            sys.executable, SEDIMENT_SCRIPT,
            "--agent", AGENT_NAME,
            "--content", leaf_content,
            "--type", "knowledge",
            "--title", title,
            "--skip-obsidian",  # 我们自己处理 Obsidian 导出
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if proc.returncode == 0:
            result["sediment"] = "ok"
        else:
            result["sediment"] = f"error: {proc.stderr[:200]}"
    except Exception as e:
        result["sediment"] = f"error: {e}"

    # Step 2: 直接导出独立 Markdown 到 Obsidian vault
    try:
        obsidian_dir = os.path.join(OBSIDIAN_VAULT, "02-Analysis")
        os.makedirs(obsidian_dir, exist_ok=True)

        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"{date_str}_{pure}_{name}.md"
        filepath = os.path.join(obsidian_dir, filename)

        # 构建 Obsidian 友好的 Markdown（带 frontmatter）
        frontmatter = (
            f"---\n"
            f"stock_code: \"{pure}\"\n"
            f"stock_name: \"{name}\"\n"
            f"price: {price}\n"
            f"pct_change: \"{pct_str}\"\n"
            f"date: {date_str}\n"
            f"source: OpenClaw\n"
            f"tags: [openclaw, analysis, {pure}]\n"
            f"---\n\n"
        )

        phase_status = (
            f"\n\n---\n"
            f"## 数据来源\n"
            f"- 记忆树: {bundle.get('phase1', {}).get('status', 'N/A')}\n"
            f"- memU: {bundle.get('phase2', {}).get('status', 'N/A')}\n"
            f"- 实时数据: {bundle.get('phase3', {}).get('status', 'N/A')}\n"
            f"- 生成时间: {ts}\n"
        )

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(frontmatter)
            f.write(f"# OpenClaw 分析 — {name} ({pure})\n\n")
            f.write(analysis)
            f.write(phase_status)

        result["obsidian_export"] = filepath

    except Exception as e:
        result["obsidian_export"] = f"error: {e}"

    # Step 3: 同步 MEMORY.md 到 Obsidian (01-Agent/)
    try:
        cmd = [
            sys.executable, OBSIDIAN_SYNC_SCRIPT,
            "--agent", AGENT_NAME, "--quiet",
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except Exception:
        pass

    return result


# ═══════════════════════════════════════════
# Step 6: 输出格式化
# ═══════════════════════════════════════════

def format_output(
    analysis: str | None,
    bundle: dict,
    stock_code: str,
    output_json: bool = False,
) -> str:
    """格式化最终输出"""
    pure = stock_code.split(".")[0] if "." in stock_code else stock_code
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if output_json:
        return json.dumps(
            {
                "stock_code": pure,
                "timestamp": ts,
                "phases": {
                    "phase1_memory": bundle.get("phase1", {}).get("status", "error"),
                    "phase2_memu": bundle.get("phase2", {}).get("status", "unavailable"),
                    "phase3_data": bundle.get("phase3", {}).get("status", "error"),
                },
                "analysis": analysis,
                "context_bundle": bundle,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    if analysis:
        # Markdown 报告
        lines = [
            f"# OpenClaw 分析报告 — {pure}",
            f"*{ts}*\n",
            analysis,
            "",
            "---",
            "### 数据来源",
            f"- Phase 1 (记忆树): {bundle.get('phase1', {}).get('status', 'N/A')}",
            f"- Phase 2 (memU): {bundle.get('phase2', {}).get('status', 'N/A')}",
            f"- Phase 3 (实时数据): {bundle.get('phase3', {}).get('status', 'N/A')}",
        ]
        return "\n".join(lines)

    # --no-llm 模式: 按阶段展示原始汇总
    lines = [
        f"# OpenClaw 数据汇总 — {pure}",
        f"*{ts}*\n",
    ]

    # Phase 1
    p1 = bundle.get("phase1", {})
    lines.append(f"## Phase 1: 记忆树 [{p1.get('status', 'N/A')}]")
    if p1.get("trading_rules"):
        lines.append("### 交易规则")
        for r in p1["trading_rules"]:
            prio = r.get("priority", "")
            prio_tag = f"[{prio}] " if prio else ""
            lines.append(f"- {prio_tag}{r.get('title', '')}")
            preview = r.get("content_preview", "")
            if preview:
                lines.append(f"  {preview[:200]}")
    if p1.get("stock_memories"):
        lines.append("### 相关记忆")
        for m in p1["stock_memories"]:
            lines.append(f"- {m.get('title', '')}: {m.get('content_preview', '')[:200]}")
    if p1.get("error"):
        lines.append(f"*{p1['error']}*")
    lines.append("")

    # Phase 2
    p2 = bundle.get("phase2", {})
    lines.append(f"## Phase 2: memU [{p2.get('status', 'N/A')}]")
    if p2.get("preferences"):
        for p in p2["preferences"]:
            lines.append(f"- {p.get('content', '')[:200]}")
    if p2.get("error"):
        lines.append(f"*{p2['error']}*")
    lines.append("")

    # Phase 3
    p3 = bundle.get("phase3", {})
    lines.append(f"## Phase 3: 实时数据 [{p3.get('status', 'N/A')}]")

    if p3.get("quote"):
        q = p3["quote"]
        name = q.get("name", "")
        price = q.get("latest_price", "N/A")
        pct = q.get("pct_change")
        pct_str = f"{'+' if pct and pct > 0 else ''}{pct:.2f}%" if pct is not None else "N/A"
        lines.append(f"### 行情: {name} 最新价 {price} ({pct_str})")

    if p3.get("market_overview"):
        mo = p3["market_overview"]
        breadth = mo.get("breadth", {})
        up = breadth.get("up_count", "?")
        down = breadth.get("down_count", "?")
        lines.append(f"### 市场: 上涨 {up} / 下跌 {down}")
        indices = mo.get("indices", [])
        for idx in indices[:3]:
            if idx.get("available"):
                pct_val = idx.get("pct_change")
                pct_s = f"{'+' if pct_val and pct_val > 0 else ''}{pct_val:.2f}%" if pct_val is not None else "N/A"
                lines.append(f"  - {idx.get('name', '')}: {idx.get('latest', 'N/A')} ({pct_s})")

    if p3.get("news"):
        news_data = p3["news"]
        if isinstance(news_data, dict):
            news_list = news_data.get("news", [])
        elif isinstance(news_data, list):
            news_list = news_data
        else:
            news_list = []
        if news_list:
            lines.append("### 最新新闻")
            for n in news_list[:5]:
                lines.append(f"- [{n.get('publish_time', '')}] {n.get('title', '')}")

    if p3.get("earnings"):
        e = p3["earnings"]
        grade = e.get("grade", "?")
        total = e.get("total_score", "?")
        max_s = e.get("max_score", "?")
        lines.append(f"### 财报评分: {grade} ({total}/{max_s})")
        for d in e.get("dimensions", []):
            lines.append(f"  - {d.get('name', '')}: {d.get('signal', '')} {d.get('score', 0)}/{d.get('max', 3)} — {d.get('detail', '')}")

    if p3.get("errors"):
        lines.append(f"*错误: {', '.join(p3['errors'])}*")

    return "\n".join(lines)


# ═══════════════════════════════════════════
# Step 7: main() 入口
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="OpenClaw — A股智能分析包装器",
        epilog="示例: openclaw_wrapper.py 600519 | openclaw_wrapper.py 600519 --no-llm --json",
    )
    parser.add_argument("code", help="股票代码（如 600519, 000001.SZ）")
    parser.add_argument("--no-llm", action="store_true", help="仅汇总数据，跳过 LLM 分析")
    parser.add_argument("--json", action="store_true", help="JSON 结构化输出")
    parser.add_argument("--debug", action="store_true", help="显示各阶段耗时")
    args = parser.parse_args()

    pure, full, market = normalize_code(args.code)
    _wall_start = time.time()
    is_quiet = args.json

    context_bundle = {}
    timings = {}

    # ── Phase 1/2/3 全部并行 ──────────────────────────────
    if not is_quiet:
        print("⚡ 并行采集 Phase1/2/3...", file=sys.stderr)
    t_data_start = time.time()

    pool = ThreadPoolExecutor(max_workers=6)
    try:
        f_mem   = pool.submit(phase1_memory_recall, pure)
        f_memu  = pool.submit(phase2_memu_retrieve, pure)

        # Phase 3 子任务（独立提交，避免嵌套 pool）
        f_quote = pool.submit(
            _run_skill, QUOTE_SCRIPT,
            [pure, "--json", "--source", "tushare"],
            SKILL_TIMEOUTS["quote"],
        )
        f_market = pool.submit(
            _run_skill, MARKET_SCRIPT,
            ["--json"],
            SKILL_TIMEOUTS["market"],
        )
        f_news = pool.submit(
            _run_skill, NEWS_SCRIPT,
            [pure, "--json", "--limit", "5"],
            SKILL_TIMEOUTS["news"],
        )
        f_earnings = pool.submit(
            _run_skill, EARNINGS_SCRIPT,
            [pure, "--json", "--no-llm", "--quarters", "4"],
            SKILL_TIMEOUTS["earnings"],
        )
        all_futures = [f_mem, f_memu, f_quote, f_market, f_news, f_earnings]
        done, not_done = wait(all_futures, timeout=DATA_BUDGET)
    finally:
        # wait=False：主线程立即继续，RUNNING 的任务在后台跑完（各自有 timeout）
        pool.shutdown(wait=False, cancel_futures=True)

    # ── 组装 bundle ──────────────────────────────────────
    def _safe_result(f, default, error_list=None, label=""):
        if f not in done:
            if error_list is not None:
                error_list.append(f"{label}: timeout")
            return default
        try:
            return f.result()
        except Exception as e:
            if error_list is not None:
                error_list.append(f"{label}: {e}")
            return default

    _DEFAULT_P1 = {"status": "timeout", "trading_rules": [], "stock_memories": [], "error": "timeout"}
    _DEFAULT_P2 = {"status": "timeout", "preferences": [], "error": "timeout"}

    context_bundle["phase1"] = _safe_result(f_mem,  _DEFAULT_P1, label="memory")
    context_bundle["phase2"] = _safe_result(f_memu, _DEFAULT_P2, label="memu")

    _q = _safe_result(f_quote, None, label="quote")
    # quote fallback：akshare
    if _q is None:
        _q = _run_skill(QUOTE_SCRIPT, [pure, "--json"], SKILL_TIMEOUTS["quote"])

    p3_errors: list = []
    context_bundle["phase3"] = {
        "status":          "ok" if _q else "partial",
        "quote":           _q,
        "market_overview": _safe_result(f_market,   None, p3_errors, "market"),
        "news":            _safe_result(f_news,     None, p3_errors, "news"),
        "earnings":        _safe_result(f_earnings, None, p3_errors, "earnings"),
        "errors":          p3_errors,
    }

    timings["data"] = round(time.time() - t_data_start, 2)
    if not is_quiet:
        print(f"  数据采集完成 {timings['data']}s", file=sys.stderr)

    # ── Phase 4: LLM 综合分析（带墙钟检查 + 硬超时）────────────
    analysis         = None
    llm_degraded     = False   # True 表示走了快报降级路径

    if not args.no_llm:
        elapsed_before_llm = time.time() - _wall_start
        time_left = TOTAL_BUDGET - elapsed_before_llm

        if time_left < LLM_HARD_TIMEOUT:
            # 墙钟已超，直接降级
            llm_degraded = True
            if not is_quiet:
                print(f"⚠️ 墙钟超限({elapsed_before_llm:.1f}s)，走快报降级", file=sys.stderr)
        else:
            if not is_quiet:
                print(f"🤖 Phase4 LLM（剩余预算 {time_left:.0f}s）...", file=sys.stderr)
            t0 = time.time()
            try:
                analysis = _phase4_with_timeout(context_bundle, pure)
            except FuturesTimeoutError:
                llm_degraded = True
                if not is_quiet:
                    print(f"⚠️ LLM 超时（>{LLM_HARD_TIMEOUT}s），走快报降级", file=sys.stderr)
            except Exception as e:
                analysis = f"LLM 分析失败: {e}"
                print(f"⚠️ Phase4 失败: {e}", file=sys.stderr)
            timings["phase4"] = round(time.time() - t0, 2)

        if llm_degraded:
            # 输出本地快报
            features = extract_local_features(context_bundle, pure)
            analysis = format_quick_report(features, pure)
            # 后台推送全量 LLM 分析（Task 5 实现 _async_llm_and_push）
            _t = threading.Thread(
                target=_async_llm_and_push,
                args=(context_bundle, pure),
                daemon=True,
            )
            _t.start()

    # 保存 leaf + Obsidian 同步（仅在有 LLM 分析结果时，且非降级快报）
    if analysis and not analysis.startswith("LLM 分析失败") and not llm_degraded:
        if not is_quiet:
            print("🍃 保存分析结果到记忆树 + Obsidian...", file=sys.stderr)
        t0 = time.time()
        try:
            save_result = save_leaf_and_sync(analysis, pure, context_bundle)
            if not is_quiet:
                sed_status = save_result.get("sediment", "?")
                obs_status = save_result.get("obsidian_export", "?")
                if sed_status == "ok":
                    print(f"  ✅ leaf 已保存到 MEMORY.md", file=sys.stderr)
                else:
                    print(f"  ⚠️ leaf 保存: {sed_status}", file=sys.stderr)
                if obs_status and not obs_status.startswith("error"):
                    print(f"  ✅ 已导出: {obs_status}", file=sys.stderr)
                else:
                    print(f"  ⚠️ Obsidian 导出: {obs_status}", file=sys.stderr)
        except Exception as e:
            if not is_quiet:
                print(f"  ⚠️ 保存失败: {e}", file=sys.stderr)
        timings["save"] = round(time.time() - t0, 2)

    # 输出
    output = format_output(analysis, context_bundle, pure, output_json=args.json)
    print(output)

    # Debug: 各阶段耗时
    if args.debug:
        print("\n⏱️  各阶段耗时:", file=sys.stderr)
        for phase, dur in timings.items():
            print(f"  {phase}: {dur}s", file=sys.stderr)
        total = sum(timings.values())
        print(f"  total: {total:.2f}s", file=sys.stderr)

    # 退出码: 检查是否全部失败
    all_failed = all(
        context_bundle.get(f"phase{i}", {}).get("status") in ("error", "unavailable")
        for i in range(1, 4)
    )
    sys.exit(1 if all_failed else 0)


if __name__ == "__main__":
    main()
