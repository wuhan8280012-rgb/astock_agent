"""Minimal legacy-regime compatibility helpers for the momentum project."""

from __future__ import annotations


def derive_legacy_macro_switch(regime_state: str) -> str:
    mapping = {
        "HALT": "HALT",
        "DEFENSIVE": "RUN",
        "RUN": "RUN",
        "STRONG_RUN": "RUN",
    }
    return mapping.get(str(regime_state or "RUN"), "RUN")
