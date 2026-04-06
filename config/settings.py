import json
import os
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - fallback for lean runtime envs
    def load_dotenv(path):
        env_path = Path(path)
        if not env_path.exists():
            return False
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key:
                os.environ.setdefault(key, value)
        return True

# Load .env from config directory
_config_dir = Path(__file__).parent
load_dotenv(_config_dir / ".env")

# API Keys
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")

# AWS Bedrock configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# LLM provider selection
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "bedrock").strip().lower()

# OpenRouter configuration
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL_PRIMARY = os.getenv("OPENROUTER_MODEL_PRIMARY", "anthropic/claude-3.5-sonnet").strip()
OPENROUTER_MODEL_FAST = os.getenv("OPENROUTER_MODEL_FAST", OPENROUTER_MODEL_PRIMARY).strip()
OPENROUTER_PROXY_URL = os.getenv("OPENROUTER_PROXY_URL", "").strip()
OPENROUTER_IGNORE_PROVIDERS = [
    p.strip() for p in os.getenv("OPENROUTER_IGNORE_PROVIDERS", "anthropic").split(",") if p.strip()
]
OPENROUTER_PROVIDER_ORDER = [
    p.strip()
    for p in os.getenv("OPENROUTER_PROVIDER_ORDER", "google-vertex,amazon-bedrock").split(",")
    if p.strip()
]

# Token cost estimation (CNY per 1M tokens), override in config/.env when needed.
LLM_PRICE_OPUS_INPUT_CNY_PER_1M = float(os.getenv("LLM_PRICE_OPUS_INPUT_CNY_PER_1M", "108"))
LLM_PRICE_OPUS_OUTPUT_CNY_PER_1M = float(os.getenv("LLM_PRICE_OPUS_OUTPUT_CNY_PER_1M", "540"))
LLM_PRICE_HAIKU_INPUT_CNY_PER_1M = float(os.getenv("LLM_PRICE_HAIKU_INPUT_CNY_PER_1M", "7.2"))
LLM_PRICE_HAIKU_OUTPUT_CNY_PER_1M = float(os.getenv("LLM_PRICE_HAIKU_OUTPUT_CNY_PER_1M", "36"))

# Optional EC2 proxy mode (Mac -> EC2 proxy -> Bedrock)
LLM_PROXY_URL = os.getenv("LLM_PROXY_URL", "").strip()
LLM_PROXY_SECRET = os.getenv("LLM_PROXY_SECRET", "").strip()
LLM_PROXY_VERIFY_SSL = os.getenv("LLM_PROXY_VERIFY_SSL", "false").lower() in ("1", "true", "yes")
LLM_PROXY_MAX_RETRIES = int(os.getenv("LLM_PROXY_MAX_RETRIES", "3"))
LLM_PROXY_RETRY_BACKOFF_SEC = float(os.getenv("LLM_PROXY_RETRY_BACKOFF_SEC", "1.2"))
LLM_PROXY_TIMEOUT_SEC = float(os.getenv("LLM_PROXY_TIMEOUT_SEC", "240"))

# Project paths
PROJECT_ROOT = _config_dir.parent
DB_PATH = PROJECT_ROOT / "db" / "investment.db"
SCHEMA_PATH = PROJECT_ROOT / "db" / "schema.sql"
PARAM_VERSIONS_DIR = PROJECT_ROOT / "parameter_versions"
LOGS_DIR = PROJECT_ROOT / "logs"
STRATEGY_SPLIT_REGISTRY_PATH = PROJECT_ROOT / "config" / "strategy_split_registry.json"

if LLM_PROVIDER == "openrouter":
    LLM_MODELS = {
        "buy_agent": OPENROUTER_MODEL_PRIMARY,
        "arbitrator": OPENROUTER_MODEL_PRIMARY,
        "rationale_gen": OPENROUTER_MODEL_FAST,
        "position_agent": OPENROUTER_MODEL_FAST,
        "advocate": OPENROUTER_MODEL_FAST,
        "challenger": OPENROUTER_MODEL_FAST,
        "sentiment_agent": OPENROUTER_MODEL_FAST,
    }
else:
    # LLM model configuration (Bedrock model IDs)
    LLM_MODELS = {
        # Opus: core decision nodes
        "buy_agent": "anthropic.claude-opus-4-6-v1",
        "arbitrator": "anthropic.claude-opus-4-6-v1",
        # Haiku: execution nodes
        "rationale_gen": "anthropic.claude-haiku-4-5-20251001-v1:0",
        "position_agent": "anthropic.claude-haiku-4-5-20251001-v1:0",
        "advocate": "anthropic.claude-haiku-4-5-20251001-v1:0",
        "challenger": "anthropic.claude-haiku-4-5-20251001-v1:0",
        "sentiment_agent": "anthropic.claude-haiku-4-5-20251001-v1:0",
    }

# Opus downgrade: evaluate after 60-90 days of signal samples
OPUS_DOWNGRADE_EVALUATED = False

# Tushare rate limiting (adjust based on actual point tier)
TUSHARE_RATE_LIMIT_PER_MINUTE = 80
TUSHARE_BATCH_SIZE = 30       # Max stocks per batch query
DATA_LOOKBACK_DAYS = 120      # Daily data lookback (covers EMA60 + margin)

# LLM call defaults
LLM_TEMPERATURE = 0
LLM_MAX_RETRIES = 2

# Execution performance
AGENT_ANALYSIS_MAX_WORKERS = int(os.getenv("AGENT_ANALYSIS_MAX_WORKERS", "6"))
SHADOW_MODE_ENABLED = os.getenv("SHADOW_MODE_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
SHADOW_MIRROR_ON_RUN = os.getenv("SHADOW_MIRROR_ON_RUN", "true").strip().lower() in ("1", "true", "yes", "on")

# Report / execution schema versioning (bound to each job run for auditability).
REPORT_SCHEMA_VERSION = os.getenv("REPORT_SCHEMA_VERSION", "report.v2")
EXECUTION_POLICY_VERSION = os.getenv("EXECUTION_POLICY_VERSION", "paper_exec.v1")
DEFAULT_EXPERIMENT_TAG = os.getenv("DEFAULT_EXPERIMENT_TAG", "baseline").strip() or "baseline"

# Paper execution policy (single source of execution semantics).
# Note: defaults keep current paper behavior stable unless explicitly changed.
PAPER_EXECUTION_POLICY = {
    "version": EXECUTION_POLICY_VERSION,
    "buy_fill_timing": os.getenv("PAPER_BUY_FILL_TIMING", "next_open").strip(),  # signal_close|next_open
    "sell_fill_timing": os.getenv("PAPER_SELL_FILL_TIMING", "signal_close").strip(),  # signal_close|next_open
    "default_price_field_buy": os.getenv("PAPER_BUY_PRICE_FIELD", "close").strip(),   # open|close
    "default_price_field_sell": os.getenv("PAPER_SELL_PRICE_FIELD", "close").strip(),  # open|close
    "slippage_bps_buy": int(os.getenv("PAPER_SLIPPAGE_BPS_BUY", "0")),
    "slippage_bps_sell": int(os.getenv("PAPER_SLIPPAGE_BPS_SELL", "0")),
    "allow_partial_fill": os.getenv("PAPER_ALLOW_PARTIAL_FILL", "false").strip().lower() in ("1", "true", "yes", "on"),
    "cash_mode": os.getenv("PAPER_CASH_MODE", "strict").strip(),  # strict|advisory
}

# Regime policy defaults (used by scheduler for execution constraints).
REGIME_DEFENSIVE_MAX_NEW_POSITIONS = int(os.getenv("REGIME_DEFENSIVE_MAX_NEW_POSITIONS", "1"))
REGIME_STRONG_RUN_MAX_NEW_POSITIONS = int(os.getenv("REGIME_STRONG_RUN_MAX_NEW_POSITIONS", "4"))
REGIME_DEFENSIVE_POSITION_MULTIPLIER = float(os.getenv("REGIME_DEFENSIVE_POSITION_MULTIPLIER", "0.7"))
REGIME_STRONG_RUN_POSITION_MULTIPLIER = float(os.getenv("REGIME_STRONG_RUN_POSITION_MULTIPLIER", "1.15"))

# L3 Triangular Audit model configuration
L3_ADVOCATE_MODEL = os.getenv("L3_ADVOCATE_MODEL", OPENROUTER_MODEL_PRIMARY).strip()
L3_CHALLENGER_MODEL = os.getenv("L3_CHALLENGER_MODEL", "openai/gpt-4.1").strip()
L3_ARBITRATOR_MODEL = os.getenv("L3_ARBITRATOR_MODEL", OPENROUTER_MODEL_PRIMARY).strip()
GOV_SELF_REFLECTOR_MODEL = os.getenv("GOV_SELF_REFLECTOR_MODEL", OPENROUTER_MODEL_PRIMARY).strip()
GOV_REVIEWER_MODEL = os.getenv("GOV_REVIEWER_MODEL", OPENROUTER_MODEL_FAST).strip()
GOV_ARBITER_MODEL = os.getenv("GOV_ARBITER_MODEL", OPENROUTER_MODEL_PRIMARY).strip()

# Sentiment prescreener
PRESCREENER_TOP_N = int(os.getenv("PRESCREENER_TOP_N", "50"))
PRESCREENER_VERSION = os.getenv("PRESCREENER_VERSION", "prescreener.v4").strip()
SECTOR_SCORING_VERSION = os.getenv("SECTOR_SCORING_VERSION", "sector_dual.v2").strip()
REVIEW_RANK_VERSION = os.getenv("REVIEW_RANK_VERSION", "review_rank.v1").strip()
PRESCREENER_BASELINE_VERSION = os.getenv("PRESCREENER_BASELINE_VERSION", "sector_dual.v1").strip()
THEME_SCORE_VERSION = os.getenv("THEME_SCORE_VERSION", "theme_discovery.v2").strip()
THEME_BASELINE_VERSION = os.getenv("THEME_BASELINE_VERSION", "theme_discovery.v1").strip()
THEME_LEADER_V2_VERSION = os.getenv("THEME_LEADER_V2_VERSION", "theme_discovery.v2").strip()
PRESCREENER_OPUS_MAX_PER_SECTOR = int(os.getenv("PRESCREENER_OPUS_MAX_PER_SECTOR", "3"))
PRESCREENER_MIN_STOCK_QUALITY_FOR_OPUS = float(os.getenv("PRESCREENER_MIN_STOCK_QUALITY_FOR_OPUS", "35"))
PRESCREENER_CROSS_SECTOR_WILDCARD_SLOTS = int(os.getenv("PRESCREENER_CROSS_SECTOR_WILDCARD_SLOTS", "2"))
PRESCREENER_MAX_REVIEW_SLOTS_PER_THEME = int(os.getenv("PRESCREENER_MAX_REVIEW_SLOTS_PER_THEME", "2"))
PRESCREENER_MIN_STOCK_QUALITY_FOR_REVIEW = float(os.getenv("PRESCREENER_MIN_STOCK_QUALITY_FOR_REVIEW", "40"))
PRESCREENER_REVIEW_WILDCARD_SLOTS = int(os.getenv("PRESCREENER_REVIEW_WILDCARD_SLOTS", "2"))
THEME_LEADER_COUNT_FOR_STRENGTH = int(os.getenv("THEME_LEADER_COUNT_FOR_STRENGTH", "3"))
THEME_MIN_STRONG_MEMBER_RATIO = float(os.getenv("THEME_MIN_STRONG_MEMBER_RATIO", "0.06"))
THEME_TREND_OVER_SENTIMENT_BOOST_IN_RUN = float(os.getenv("THEME_TREND_OVER_SENTIMENT_BOOST_IN_RUN", "1.18"))
THEME_SOFT_CAP_PER_THEME = int(os.getenv("THEME_SOFT_CAP_PER_THEME", "5"))
THEME_MIN_GUARANTEE_THEMES = int(os.getenv("THEME_MIN_GUARANTEE_THEMES", "8"))
THEME_WILDCARD_SLOTS = int(os.getenv("THEME_WILDCARD_SLOTS", "3"))
THEME_TREND_WILDCARD_MIN_SCORE = float(os.getenv("THEME_TREND_WILDCARD_MIN_SCORE", "72"))
THEME_LEADER_CONCENTRATION_WEIGHT = float(os.getenv("THEME_LEADER_CONCENTRATION_WEIGHT", "0.12"))
THEME_LEADER_CONCENTRATION_BONUS_MAX = float(os.getenv("THEME_LEADER_CONCENTRATION_BONUS_MAX", "14"))
THEME_LEADER_DRIVEN_BOOST_BY_REGIME = {
    "HALT": float(os.getenv("THEME_LEADER_DRIVEN_BOOST_HALT", "1.0")),
    "DEFENSIVE": float(os.getenv("THEME_LEADER_DRIVEN_BOOST_DEFENSIVE", "1.04")),
    "RUN": float(os.getenv("THEME_LEADER_DRIVEN_BOOST_RUN", "1.16")),
    "STRONG_RUN": float(os.getenv("THEME_LEADER_DRIVEN_BOOST_STRONG_RUN", "1.24")),
}

PRESCREENER_BASELINE_SECTOR_WEIGHT_BY_REGIME = {
    "HALT": {"sentiment": 0.55, "trend": 0.45},
    "DEFENSIVE": {"sentiment": 0.65, "trend": 0.35},
    "RUN": {"sentiment": 0.40, "trend": 0.60},
    "STRONG_RUN": {"sentiment": 0.30, "trend": 0.70},
}

PRESCREENER_SECTOR_WEIGHT_BY_REGIME = {
    "HALT": {"sentiment": 0.52, "trend": 0.48},
    "DEFENSIVE": {"sentiment": 0.55, "trend": 0.45},
    "RUN": {"sentiment": 0.28, "trend": 0.72},
    "STRONG_RUN": {"sentiment": 0.18, "trend": 0.82},
}

PRESCREENER_SENTIMENT_COMPONENT_WEIGHTS = {
    "breadth": 0.28,
    "limit_up": 0.17,
    "momentum": 0.25,
    "turnover": 0.20,
    "stability": 0.10,
}

PRESCREENER_TREND_COMPONENT_WEIGHTS = {
    "avg_pct": 0.20,
    "ret_3d": 0.18,
    "leader_avg_pct": 0.22,
    "strong_ratio": 0.14,
    "breakout_ratio": 0.12,
    "liquidity_leader": 0.14,
}

PRESCREENER_STOCK_QUALITY_WEIGHTS = {
    "today_pct": 0.24,
    "ret_3d": 0.20,
    "turnover": 0.14,
    "role_strength": 0.22,
    "right_side": 0.20,
}

PRESCREENER_REVIEW_RANK_WEIGHTS = {
    "sector_total_score": 0.35,
    "stock_quality_score": 0.40,
    "role_strength_score": 0.25,
}

THEME_TOTAL_WEIGHT_BY_REGIME = {
    "HALT": {"sentiment": 0.42, "trend": 0.58},
    "DEFENSIVE": {"sentiment": 0.50, "trend": 0.50},
    "RUN": {"sentiment": 0.34, "trend": 0.66},
    "STRONG_RUN": {"sentiment": 0.25, "trend": 0.75},
}

THEME_TREND_COMPONENT_WEIGHTS = {
    "avg_pct": 0.14,
    "median_pct": 0.10,
    "leader_avg_pct": 0.20,
    "leader_liquidity": 0.16,
    "strong_ratio": 0.16,
    "breakout_ratio": 0.14,
    "breadth": 0.05,
    "streak": 0.03,
    "turnover_anomaly": 0.02,
}

THEME_TREND_COMPONENT_WEIGHTS_BASELINE = {
    "avg_pct": 0.16,
    "median_pct": 0.12,
    "leader_avg_pct": 0.16,
    "leader_liquidity": 0.12,
    "strong_ratio": 0.15,
    "breakout_ratio": 0.11,
    "breadth": 0.10,
    "streak": 0.05,
    "turnover_anomaly": 0.03,
}

THEME_SENTIMENT_COMPONENT_WEIGHTS = {
    "breadth": 0.34,
    "streak": 0.28,
    "turnover_anomaly": 0.20,
    "strong_ratio": 0.10,
    "breakout_ratio": 0.08,
}

ACTIVE_FROZEN_VERSION = os.getenv("ACTIVE_FROZEN_VERSION", "candidate_frozen").strip() or "candidate_frozen"
BASELINE_FROZEN_VERSION = os.getenv("BASELINE_FROZEN_VERSION", "baseline_frozen").strip() or "baseline_frozen"
CANDIDATE_FROZEN_VERSION = os.getenv("CANDIDATE_FROZEN_VERSION", "candidate_frozen").strip() or "candidate_frozen"
MIN_REPLAY_CONFIDENCE_FOR_PROMOTION = os.getenv("MIN_REPLAY_CONFIDENCE_FOR_PROMOTION", "medium").strip() or "medium"
OVERFIT_DESIGN_HOLDOUT_GAP_HIGH = float(os.getenv("OVERFIT_DESIGN_HOLDOUT_GAP_HIGH", "2.5"))
COMPLEXITY_NOT_JUSTIFIED_DELTA = float(os.getenv("COMPLEXITY_NOT_JUSTIFIED_DELTA", "3.0"))
MIN_HOLDOUT_GAIN_FOR_PROMOTION = float(os.getenv("MIN_HOLDOUT_GAIN_FOR_PROMOTION", "0.0"))


def _ensure_llm_usage_table() -> None:
    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS llm_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                component TEXT DEFAULT '',
                prompt_tokens INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                estimated_cost_cny REAL DEFAULT 0,
                latency_ms INTEGER DEFAULT 0,
                status TEXT DEFAULT 'ok',
                error_message TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )"""
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_llm_usage_created ON llm_usage(created_at)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_llm_usage_component ON llm_usage(component, created_at)"
        )


def _extract_usage(payload) -> tuple[int, int, int]:
    """Extract usage from OpenRouter/Anthropic style response payloads."""
    prompt_tokens = completion_tokens = total_tokens = 0

    usage = payload.get("usage") if isinstance(payload, dict) else getattr(payload, "usage", None)
    if usage is None:
        return 0, 0, 0

    def _first_not_none(*vals):
        for v in vals:
            if v is not None:
                return int(v)
        return 0

    if isinstance(usage, dict):
        prompt_tokens = _first_not_none(usage.get("prompt_tokens"), usage.get("input_tokens"))
        completion_tokens = _first_not_none(usage.get("completion_tokens"), usage.get("output_tokens"))
        total_tokens = _first_not_none(usage.get("total_tokens")) or (prompt_tokens + completion_tokens)
        return prompt_tokens, completion_tokens, total_tokens

    prompt_tokens = _first_not_none(getattr(usage, "prompt_tokens", None), getattr(usage, "input_tokens", None))
    completion_tokens = _first_not_none(getattr(usage, "completion_tokens", None), getattr(usage, "output_tokens", None))
    total_tokens = _first_not_none(getattr(usage, "total_tokens", None)) or (prompt_tokens + completion_tokens)
    return prompt_tokens, completion_tokens, total_tokens


def _estimate_cost_cny(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    model_norm = (model or "").lower()
    if "haiku" in model_norm:
        in_price = LLM_PRICE_HAIKU_INPUT_CNY_PER_1M
        out_price = LLM_PRICE_HAIKU_OUTPUT_CNY_PER_1M
    else:
        in_price = LLM_PRICE_OPUS_INPUT_CNY_PER_1M
        out_price = LLM_PRICE_OPUS_OUTPUT_CNY_PER_1M
    cost = (prompt_tokens / 1_000_000) * in_price + (completion_tokens / 1_000_000) * out_price
    return round(cost, 6)


def _record_llm_usage(
    provider: str,
    model: str,
    component: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    latency_ms: int,
    status: str = "ok",
    error_message: str = "",
) -> None:
    try:
        _ensure_llm_usage_table()
        with sqlite3.connect(str(DB_PATH)) as conn:
            conn.execute(
                """INSERT INTO llm_usage (
                    provider, model, component, prompt_tokens, completion_tokens, total_tokens,
                    estimated_cost_cny, latency_ms, status, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    provider,
                    model,
                    component or "",
                    int(prompt_tokens or 0),
                    int(completion_tokens or 0),
                    int(total_tokens or 0),
                    _estimate_cost_cny(model, int(prompt_tokens or 0), int(completion_tokens or 0)),
                    int(latency_ms or 0),
                    status,
                    (error_message or "")[:500],
                ),
            )
    except Exception:
        # Token accounting should never break trading flow.
        pass


_llm_client = None
_llm_client_lock = __import__("threading").Lock()


def get_llm_client():
    """Get or create a unified client exposing messages.create(). Cached as singleton."""
    global _llm_client
    if _llm_client is not None:
        return _llm_client
    with _llm_client_lock:
        if _llm_client is not None:
            return _llm_client
        _llm_client = _create_llm_client()
        return _llm_client


def _create_llm_client():
    """Create a unified client exposing messages.create()."""
    if LLM_PROXY_URL:
        import httpx

        class _ProxyMessages:
            def __init__(self, base_url: str, api_secret: str, verify_ssl: bool):
                self._base_url = base_url.rstrip("/")
                self._api_secret = api_secret
                self._verify_ssl = verify_ssl

            def create(
                self,
                model: str,
                max_tokens: int,
                temperature: float = 0,
                system: str = "",
                messages=None,
                component: str = "",
            ):
                payload = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": messages or [],
                }
                if system:
                    payload["system"] = system

                headers = {}
                if self._api_secret:
                    headers["x-api-key"] = self._api_secret

                started = time.monotonic()
                last_error = None
                data = None
                for attempt in range(max(1, LLM_PROXY_MAX_RETRIES)):
                    try:
                        response = httpx.post(
                            f"{self._base_url}/v1/messages",
                            json=payload,
                            headers=headers,
                            timeout=LLM_PROXY_TIMEOUT_SEC,
                            verify=self._verify_ssl,
                            trust_env=False,
                        )
                        if response.status_code in (429, 502, 503, 504):
                            raise httpx.HTTPStatusError(
                                f"upstream transient status={response.status_code}",
                                request=response.request,
                                response=response,
                            )
                        response.raise_for_status()
                        data = response.json()
                        break
                    except Exception as e:
                        last_error = e
                        if attempt < max(1, LLM_PROXY_MAX_RETRIES) - 1:
                            time.sleep(LLM_PROXY_RETRY_BACKOFF_SEC * (2 ** attempt))
                        continue

                if data is None:
                    _record_llm_usage(
                        provider="proxy",
                        model=model,
                        component=component,
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        latency_ms=int((time.monotonic() - started) * 1000),
                        status="error",
                        error_message=str(last_error) if last_error else "unknown proxy error",
                    )
                    raise last_error if last_error else RuntimeError("proxy call failed")
                text = ""
                content = data.get("content", [])
                if content and isinstance(content, list):
                    first = content[0]
                    if isinstance(first, dict):
                        text = first.get("text", "")
                prompt_tokens, completion_tokens, total_tokens = _extract_usage(data)
                _record_llm_usage(
                    provider="proxy",
                    model=model,
                    component=component,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    latency_ms=int((time.monotonic() - started) * 1000),
                    status="ok",
                )
                return SimpleNamespace(content=[SimpleNamespace(text=text)])

        return SimpleNamespace(messages=_ProxyMessages(LLM_PROXY_URL, LLM_PROXY_SECRET, LLM_PROXY_VERIFY_SSL))

    if LLM_PROVIDER == "openrouter":
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is not configured")
        from openai import OpenAI

        http_client = None
        if OPENROUTER_PROXY_URL:
            import httpx

            # Proxy only OpenRouter traffic; keep Tushare and other IO unaffected.
            http_client = httpx.Client(proxy=OPENROUTER_PROXY_URL, trust_env=False, timeout=240.0)

        client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
            http_client=http_client,
        )

        class _OpenRouterMessages:
            def __init__(self, openai_client):
                self._client = openai_client

            def create(
                self,
                model: str,
                max_tokens: int,
                temperature: float = 0,
                system: str = "",
                messages=None,
                component: str = "",
            ):
                req_messages = []
                if system:
                    req_messages.append({"role": "system", "content": system})
                req_messages.extend(messages or [])
                started = time.monotonic()
                try:
                    extra_body = None
                    model_norm = (model or "").lower()
                    if model_norm.startswith("anthropic/claude"):
                        extra_body = {
                            "provider": {
                                "ignore": OPENROUTER_IGNORE_PROVIDERS,
                                "order": OPENROUTER_PROVIDER_ORDER,
                            }
                        }
                    resp = self._client.chat.completions.create(
                        model=model,
                        messages=req_messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        extra_body=extra_body,
                    )
                except Exception as e:
                    _record_llm_usage(
                        provider="openrouter",
                        model=model,
                        component=component,
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        latency_ms=int((time.monotonic() - started) * 1000),
                        status="error",
                        error_message=str(e),
                    )
                    raise
                text = ""
                if resp.choices and resp.choices[0].message:
                    text = resp.choices[0].message.content or ""
                prompt_tokens, completion_tokens, total_tokens = _extract_usage(resp)
                _record_llm_usage(
                    provider="openrouter",
                    model=model,
                    component=component,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    latency_ms=int((time.monotonic() - started) * 1000),
                    status="ok",
                )
                return SimpleNamespace(content=[SimpleNamespace(text=text)])

        return SimpleNamespace(messages=_OpenRouterMessages(client))

    """Create an AnthropicBedrock client. All LLM-calling modules MUST use this."""
    import anthropic
    raw_client = anthropic.AnthropicBedrock(
        aws_access_key=AWS_ACCESS_KEY_ID,
        aws_secret_key=AWS_SECRET_ACCESS_KEY,
        aws_region=AWS_REGION,
    )

    class _BedrockMessages:
        def __init__(self, anthropic_client):
            self._client = anthropic_client

        def create(
            self,
            model: str,
            max_tokens: int,
            temperature: float = 0,
            system: str = "",
            messages=None,
            component: str = "",
        ):
            started = time.monotonic()
            try:
                resp = self._client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system,
                    messages=messages or [],
                )
            except Exception as e:
                _record_llm_usage(
                    provider="bedrock",
                    model=model,
                    component=component,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    latency_ms=int((time.monotonic() - started) * 1000),
                    status="error",
                    error_message=str(e),
                )
                raise
            prompt_tokens, completion_tokens, total_tokens = _extract_usage(resp)
            _record_llm_usage(
                provider="bedrock",
                model=model,
                component=component,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_ms=int((time.monotonic() - started) * 1000),
                status="ok",
            )
            return resp

    return SimpleNamespace(messages=_BedrockMessages(raw_client))


def load_current_params() -> dict:
    """Load the latest parameter version JSON file."""
    versions_dir = PARAM_VERSIONS_DIR
    json_files = sorted(versions_dir.glob("v*.json"), reverse=True)
    if not json_files:
        raise FileNotFoundError(f"No parameter version files found in {versions_dir}")
    with open(json_files[0], "r", encoding="utf-8") as f:
        return json.load(f)


def get_current_thresholds() -> dict:
    """Return active threshold config from latest parameter version."""
    params = load_current_params()
    thresholds = params.get("thresholds", {})
    return {
        "buy_signal_min_score": float(thresholds.get("buy_signal_min_score", 80)),
        "sell_score_collapse": float(thresholds.get("sell_score_collapse", 45)),
        "rr_ratio_min": float(thresholds.get("rr_ratio_min", 2.0)),
    }


def get_execution_policy() -> dict:
    """Return immutable paper execution policy snapshot used by scheduler/report."""
    return dict(PAPER_EXECUTION_POLICY)


def get_prescreener_config() -> dict:
    """Return centralized prescreener config used by fallback scoring/reporting."""
    return {
        "prescreener_version": PRESCREENER_VERSION,
        "baseline_prescreener_version": PRESCREENER_BASELINE_VERSION,
        "sector_scoring_version": SECTOR_SCORING_VERSION,
        "theme_score_version": THEME_SCORE_VERSION,
        "baseline_theme_score_version": THEME_BASELINE_VERSION,
        "leader_v2_theme_score_version": THEME_LEADER_V2_VERSION,
        "review_rank_version": REVIEW_RANK_VERSION,
        "baseline_sector_weight_by_regime": dict(PRESCREENER_BASELINE_SECTOR_WEIGHT_BY_REGIME),
        "sector_weight_by_regime": dict(PRESCREENER_SECTOR_WEIGHT_BY_REGIME),
        "theme_total_weight_by_regime": dict(THEME_TOTAL_WEIGHT_BY_REGIME),
        "theme_trend_component_weights": dict(THEME_TREND_COMPONENT_WEIGHTS),
        "theme_trend_component_weights_baseline": dict(THEME_TREND_COMPONENT_WEIGHTS_BASELINE),
        "theme_sentiment_component_weights": dict(THEME_SENTIMENT_COMPONENT_WEIGHTS),
        "sentiment_component_weights": dict(PRESCREENER_SENTIMENT_COMPONENT_WEIGHTS),
        "trend_component_weights": dict(PRESCREENER_TREND_COMPONENT_WEIGHTS),
        "stock_quality_weights": dict(PRESCREENER_STOCK_QUALITY_WEIGHTS),
        "review_rank_weights": dict(PRESCREENER_REVIEW_RANK_WEIGHTS),
        "leader_count_for_theme_strength": int(THEME_LEADER_COUNT_FOR_STRENGTH),
        "min_strong_member_ratio": float(THEME_MIN_STRONG_MEMBER_RATIO),
        "trend_over_sentiment_boost_in_run": float(THEME_TREND_OVER_SENTIMENT_BOOST_IN_RUN),
        "theme_soft_cap_per_theme": int(THEME_SOFT_CAP_PER_THEME),
        "theme_min_guarantee_themes": int(THEME_MIN_GUARANTEE_THEMES),
        "theme_wildcard_slots": int(THEME_WILDCARD_SLOTS),
        "theme_trend_wildcard_min_score": float(THEME_TREND_WILDCARD_MIN_SCORE),
        "leader_concentration_weight": float(THEME_LEADER_CONCENTRATION_WEIGHT),
        "leader_concentration_bonus_max": float(THEME_LEADER_CONCENTRATION_BONUS_MAX),
        "leader_driven_boost_by_regime": dict(THEME_LEADER_DRIVEN_BOOST_BY_REGIME),
        "max_per_sector": int(PRESCREENER_OPUS_MAX_PER_SECTOR),
        "min_stock_quality_for_opus": float(PRESCREENER_MIN_STOCK_QUALITY_FOR_OPUS),
        "cross_sector_wildcard_slots": int(PRESCREENER_CROSS_SECTOR_WILDCARD_SLOTS),
        "max_review_slots_per_theme": int(PRESCREENER_MAX_REVIEW_SLOTS_PER_THEME),
        "min_stock_quality_for_review": float(PRESCREENER_MIN_STOCK_QUALITY_FOR_REVIEW),
        "review_wildcard_slots": int(PRESCREENER_REVIEW_WILDCARD_SLOTS),
        "top_n": int(PRESCREENER_TOP_N),
        "active_frozen_version": ACTIVE_FROZEN_VERSION,
        "baseline_frozen_version": BASELINE_FROZEN_VERSION,
        "candidate_frozen_version": CANDIDATE_FROZEN_VERSION,
        "strategy_split_registry_path": str(STRATEGY_SPLIT_REGISTRY_PATH),
        "min_replay_confidence_for_promotion": MIN_REPLAY_CONFIDENCE_FOR_PROMOTION,
        "overfit_design_holdout_gap_high": float(OVERFIT_DESIGN_HOLDOUT_GAP_HIGH),
        "complexity_not_justified_delta": float(COMPLEXITY_NOT_JUSTIFIED_DELTA),
        "min_holdout_gain_for_promotion": float(MIN_HOLDOUT_GAIN_FOR_PROMOTION),
    }
