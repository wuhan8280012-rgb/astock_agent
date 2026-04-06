"""
AI reflection engine using Opus API to analyze performance and propose improvements.
The core of the self-evolution system.
"""

import json
from dataclasses import dataclass
from typing import List, Optional

from loguru import logger

from config.settings import OPENROUTER_MODEL_PRIMARY, get_llm_client
from momentum.config import MomentumConfig

from .evaluator import PerformanceReport


@dataclass
class MutationProposal:
    """A proposed parameter mutation for evolution."""

    parameter_changes: dict  # param_name -> new_value
    rationale: str  # why this change is proposed
    expected_impact: str  # predicted effect on performance
    confidence: float  # 0-1 confidence in proposal
    risk_level: str  # "low" | "medium" | "high"


@dataclass
class ReflectionResult:
    """Structured output from Opus AI reflection."""

    analysis: str  # Natural language analysis of what worked/didn't work
    market_regime: str  # Detected market regime (trending/mean-reverting)
    key_issues: List[str]  # Problems identified
    opportunities: List[str]  # Potential improvements
    proposals: List[MutationProposal]  # Specific parameter changes to try


class ReflectorEngine:
    """Uses Opus AI to reflect on strategy performance and propose improvements."""

    SYSTEM_PROMPT = """You are Opus, an expert quantitative strategy analyst specializing in momentum rotation strategies.

Your role is to analyze strategy performance data and propose parameter mutations that could improve future returns.

When analyzing performance:
1. Assess market regime (trending vs mean-reverting) based on Sharpe ratio and returns
2. Evaluate if lookback periods are appropriately calibrated for current market dynamics
3. Consider whether position count (top_n) provides optimal diversification
4. Check if risk parameters (stop_loss, volatility penalty) are too tight or too loose
5. Identify regime-specific issues (are we protecting capital during downturns?)

When proposing mutations:
- Keep changes incremental (±10-25% from current values)
- Max 3 parameters per proposal to avoid overfitting
- Prioritize high-confidence changes backed by data
- Consider interaction effects between parameters
- Flag high-risk experiments that need validation

Output format: Valid JSON matching the structure defined in the prompt."""

    USER_PROMPT_TEMPLATE = """
Analyze this momentum rotation strategy and propose mutations:

PERFORMANCE REPORT (90-day window):
- Total Return: {total_return:.2f}%
- Annualized Return: {annualized_return:.2f}%
- Sharpe Ratio: {sharpe_ratio:.3f}
- Max Drawdown: {max_drawdown:.2f}%
- Calmar Ratio: {calmar_ratio:.3f}
- Win Rate: {win_rate:.1f}%
- Avg Turnover: {avg_turnover:.2f}%
- Total Trades: {total_trades}
- Info Ratio: {information_ratio:.3f}

CURRENT CONFIGURATION:
- Version: {version}
- Lookback Periods: {lookback_days}
- Lookback Weights: {lookback_weights}
- Top N Holdings: {top_n}
- Max Single Weight: {max_single_weight:.1%}
- Momentum Type: {momentum_type}
- Volatility Penalty: {volatility_penalty}
- Rebalance Threshold: {rebalance_threshold:.1%}
- Stop Loss: {stop_loss_pct:.1%}
- Regime Filter: {regime_filter_enabled}
- Regime Defensive Cash: {regime_defensive_cash_pct:.1%}

RECENT TRADES SAMPLE:
{recent_trades_sample}

Provide structured analysis in JSON format:
{{
    "analysis": "Natural language assessment of what's working and what's not",
    "market_regime": "trending|mean_reverting|oscillating",
    "key_issues": ["list of identified problems"],
    "opportunities": ["list of potential improvements"],
    "proposals": [
        {{
            "rationale": "Why this change addresses an issue",
            "parameter_changes": {{"param_name": new_value}},
            "expected_impact": "Specific expected outcome",
            "confidence": 0.0-1.0,
            "risk_level": "low|medium|high"
        }}
    ]
}}
"""

    @staticmethod
    def reflect_on_performance(
        perf: PerformanceReport,
        config: MomentumConfig,
        recent_trades: Optional[List[dict]] = None,
    ) -> ReflectionResult:
        """
        Use Opus API to reflect on strategy performance and generate analysis.

        Args:
            perf: PerformanceReport with calculated metrics
            config: MomentumConfig being evaluated
            recent_trades: Recent trade details for context

        Returns:
            ReflectionResult with AI analysis and proposals
        """
        if recent_trades is None:
            recent_trades = []

        logger.info(f"Starting reflection for {config.version}")

        # Prepare trade sample for context
        if recent_trades:
            trade_sample = json.dumps(
                [t for t in recent_trades[-5:]], indent=2, default=str
            )
        else:
            trade_sample = "No recent trades available"

        # Build user message
        user_message = ReflectorEngine.USER_PROMPT_TEMPLATE.format(
            total_return=perf.total_return,
            annualized_return=perf.annualized_return,
            sharpe_ratio=perf.sharpe_ratio,
            max_drawdown=perf.max_drawdown,
            calmar_ratio=perf.calmar_ratio,
            win_rate=perf.win_rate,
            avg_turnover=perf.avg_turnover,
            total_trades=perf.total_trades,
            information_ratio=perf.information_ratio,
            version=config.version,
            lookback_days=config.lookback_days,
            lookback_weights=config.lookback_weights,
            top_n=config.top_n,
            max_single_weight=config.max_single_weight,
            momentum_type=config.momentum_type,
            volatility_penalty=config.volatility_penalty,
            rebalance_threshold=config.rebalance_threshold,
            stop_loss_pct=config.stop_loss_pct,
            regime_filter_enabled=config.regime_filter_enabled,
            regime_defensive_cash_pct=config.regime_defensive_cash_pct,
            recent_trades_sample=trade_sample,
        )

        # Call Opus API
        try:
            client = get_llm_client()
            response = client.messages.create(
                model=OPENROUTER_MODEL_PRIMARY,
                max_tokens=4096,
                temperature=0.3,
                system=ReflectorEngine.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
                component="evolution_reflector",
            )
            response_text = response.content[0].text
            logger.info(f"Received reflection from Opus: {len(response_text)} chars")
        except Exception as e:
            logger.error(f"Failed to call Opus API: {e}")
            raise

        # Parse JSON response
        try:
            # Extract JSON from response (may be wrapped in markdown code blocks)
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            result_data = json.loads(response_text.strip())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Opus response as JSON: {e}")
            logger.error(f"Response text: {response_text[:500]}")
            # Return safe defaults
            return ReflectionResult(
                analysis="Failed to parse AI response",
                market_regime="unknown",
                key_issues=["Failed to parse AI response"],
                opportunities=[],
                proposals=[],
            )

        # Convert proposals to MutationProposal objects
        proposals = []
        for prop_data in result_data.get("proposals", []):
            try:
                proposal = MutationProposal(
                    parameter_changes=prop_data.get("parameter_changes", {}),
                    rationale=prop_data.get("rationale", ""),
                    expected_impact=prop_data.get("expected_impact", ""),
                    confidence=float(prop_data.get("confidence", 0.5)),
                    risk_level=prop_data.get("risk_level", "medium"),
                )
                proposals.append(proposal)
            except (KeyError, TypeError, ValueError) as e:
                logger.warning(f"Failed to parse proposal: {e}")
                continue

        result = ReflectionResult(
            analysis=result_data.get("analysis", ""),
            market_regime=result_data.get("market_regime", "unknown"),
            key_issues=result_data.get("key_issues", []),
            opportunities=result_data.get("opportunities", []),
            proposals=proposals,
        )

        logger.info(
            f"Reflection complete: {len(result.proposals)} proposals, "
            f"{len(result.key_issues)} issues identified"
        )
        return result

    @staticmethod
    def generate_mutation_proposals(
        reflection: ReflectionResult,
        config: MomentumConfig,
    ) -> List[MutationProposal]:
        """
        Generate concrete mutation proposals from reflection result.

        Args:
            reflection: ReflectionResult from Opus
            config: Current MomentumConfig

        Returns:
            List of MutationProposal objects ready for sandbox testing
        """
        logger.info(f"Generating mutations from reflection with {len(reflection.proposals)} proposals")

        validated_proposals = []
        for proposal in reflection.proposals:
            # Validate parameter changes
            changes = proposal.parameter_changes
            if not changes:
                continue

            # Validate parameter names and ranges
            valid_changes = {}
            for param, new_val in changes.items():
                if not hasattr(config, param):
                    logger.warning(f"Unknown parameter: {param}")
                    continue

                try:
                    # Type-safe conversion
                    current_val = getattr(config, param)
                    if isinstance(current_val, list):
                        if not isinstance(new_val, list):
                            logger.warning(f"{param} must be list, got {type(new_val)}")
                            continue
                        valid_changes[param] = new_val
                    elif isinstance(current_val, bool):
                        valid_changes[param] = bool(new_val)
                    elif isinstance(current_val, (int, float)):
                        valid_changes[param] = float(new_val) if isinstance(current_val, float) else int(new_val)
                    else:
                        valid_changes[param] = new_val
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid value for {param}: {e}")
                    continue

            if valid_changes:
                proposal.parameter_changes = valid_changes
                validated_proposals.append(proposal)

        logger.info(f"Generated {len(validated_proposals)} validated mutation proposals")
        return validated_proposals
