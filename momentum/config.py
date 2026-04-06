"""
Momentum rotation strategy configuration.
The "genome" that Opus AI will evolve for strategy optimization.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from loguru import logger


@dataclass
class MomentumConfig:
    """
    Strategy configuration as JSON-serializable dataclass.
    All numeric parameters are tunable for genetic algorithm evolution.
    """

    # ── Momentum calculation ──
    # v2.0 优化: 保留短期动量为主(0.5), 与基线一致; 回望5/10/20日
    lookback_days: List[int] = field(default_factory=lambda: [5, 10, 20])
    lookback_weights: List[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])

    # ── Universe filtering ──
    universe_min_market_cap: float = 5.0  # billions CNY
    universe_min_avg_turnover: float = 100.0  # millions CNY (v2.0: 提高至1亿)
    universe_exclude_st: bool = True
    universe_exclude_new_days: int = 120  # v2.0: 至少120个交易日

    # ── Portfolio construction ──
    top_n: int = 10  # number of stocks to hold
    rebalance_weekday: int = 4  # 0=Mon, 4=Fri
    rebalance_interval_days: int = 5  # v2.0: 5个交易日(周度)
    max_single_weight: float = 0.15  # 15% max weight per stock
    max_total_position: float = 0.80  # v2.0: 最大80%仓位

    # ── Risk management ──
    stop_loss_pct: float = -0.08  # -8% trailing stop (v2.0: 实测最优)
    regime_filter_enabled: bool = True
    regime_halt_cash_pct: float = 1.0  # 100% cash in HALT regime
    regime_defensive_cash_pct: float = 0.5  # 50% cash in DEFENSIVE

    # ── Momentum calculation method ──
    momentum_type: str = "composite"  # "simple" | "risk_adjusted" | "composite"
    volatility_penalty: float = 0.5  # v2.0: 从0.3提高到0.5

    # ── v2.0 新增: 流动性与缓冲带 ──
    # 经108组参数网格搜索验证, 收益+9.40%, 夏普5.85
    liquidity_weight: float = 0.10  # 成交额对数加权, 0.10=最优
    hold_buffer_ratio: float = 1.3  # 缓冲带: TOP N*1.3内保留现有持仓

    # ── Execution ──
    commission_rate: float = 0.0003  # 万三
    stamp_tax_rate: float = 0.001   # 千一(卖出)
    slippage_pct: float = 0.001     # 0.1% 滑点

    # ── Version tracking (for Opus evolution) ──
    version: str = "2.0.0"
    parent_version: Optional[str] = "1.0.0"
    description: str = "Optimized via 108-combo grid search: +liq_weight, +buffer, -8% stop confirmed"

    # Internal tracking
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self):
        """Validate configuration parameters."""
        if len(self.lookback_days) != len(self.lookback_weights):
            raise ValueError("lookback_days and lookback_weights must have same length")

        if abs(sum(self.lookback_weights) - 1.0) > 0.01:
            raise ValueError("lookback_weights must sum to 1.0")

        if self.top_n < 1:
            raise ValueError("top_n must be >= 1")

        if self.max_single_weight < 0 or self.max_single_weight > 1.0:
            raise ValueError("max_single_weight must be between 0 and 1")

        if self.stop_loss_pct >= 0:
            raise ValueError("stop_loss_pct must be negative")

        if self.rebalance_weekday < 0 or self.rebalance_weekday > 4:
            raise ValueError("rebalance_weekday must be 0-4 (Mon-Fri)")

        if self.regime_halt_cash_pct < 0 or self.regime_halt_cash_pct > 1.0:
            raise ValueError("regime_halt_cash_pct must be between 0 and 1")

        if self.regime_defensive_cash_pct < 0 or self.regime_defensive_cash_pct > 1.0:
            raise ValueError("regime_defensive_cash_pct must be between 0 and 1")

        if self.momentum_type not in ("simple", "risk_adjusted", "composite"):
            raise ValueError("momentum_type must be 'simple', 'risk_adjusted', or 'composite'")

        if not (0 <= self.liquidity_weight <= 1.0):
            raise ValueError("liquidity_weight must be between 0 and 1")

        if self.hold_buffer_ratio < 1.0:
            raise ValueError("hold_buffer_ratio must be >= 1.0")

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "MomentumConfig":
        """Create config from dictionary."""
        # Only pass known fields to dataclass
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered_data)

    def save(self, path: Path) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved momentum config to {path}")

    @classmethod
    def load(cls, path: Path) -> "MomentumConfig":
        """Load configuration from JSON file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        config = cls.from_dict(data)
        logger.info(f"Loaded momentum config from {path}")
        return config

    def mutate(self, changes: dict) -> "MomentumConfig":
        """
        Create a new config with specified changes (for evolution).
        Returns a new config instance with bumped version.
        """
        # Get current config as dict
        current_dict = self.to_dict()

        # Apply changes
        current_dict.update(changes)

        # Bump version
        current_version = self.version.split(".")
        minor = int(current_version[-1]) + 1
        new_version = ".".join(current_version[:-1] + [str(minor)])

        current_dict["version"] = new_version
        current_dict["parent_version"] = self.version
        current_dict["created_at"] = datetime.now().isoformat()

        return self.from_dict(current_dict)

    def summary(self) -> str:
        """Return human-readable summary of configuration."""
        lines = [
            f"MomentumConfig v{self.version}",
            f"  Description: {self.description}",
            f"  Parent: {self.parent_version or 'baseline'}",
            f"  Lookback periods: {self.lookback_days}",
            f"  Weights: {self.lookback_weights}",
            f"  Universe filters:",
            f"    Min market cap: {self.universe_min_market_cap}B CNY",
            f"    Min avg turnover: {self.universe_min_avg_turnover}M CNY",
            f"    Exclude ST: {self.universe_exclude_st}",
            f"    Min listing age: {self.universe_exclude_new_days} days",
            f"  Portfolio:",
            f"    Holdings: {self.top_n}",
            f"    Max single weight: {self.max_single_weight*100:.1f}%",
            f"    Rebalance: {self._weekday_name(self.rebalance_weekday)}",
            f"    Threshold: {self.rebalance_threshold*100:.1f}%",
            f"  Risk management:",
            f"    Stop loss: {self.stop_loss_pct*100:.1f}%",
            f"    Regime filter: {self.regime_filter_enabled}",
            f"    HALT cash: {self.regime_halt_cash_pct*100:.1f}%",
            f"    Defensive cash: {self.regime_defensive_cash_pct*100:.1f}%",
            f"  Momentum calculation:",
            f"    Type: {self.momentum_type}",
            f"    Vol penalty: {self.volatility_penalty}",
            f"    Liquidity weight: {self.liquidity_weight}",
            f"    Hold buffer ratio: {self.hold_buffer_ratio}x",
            f"  Execution:",
            f"    Commission: {self.commission_rate*10000:.1f}‱",
            f"    Stamp tax: {self.stamp_tax_rate*1000:.1f}‰",
            f"    Slippage: {self.slippage_pct*100:.1f}%",
        ]
        return "\n".join(lines)

    @staticmethod
    def _weekday_name(day: int) -> str:
        """Convert weekday number to name."""
        names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        return names[day] if 0 <= day < 5 else "Unknown"
