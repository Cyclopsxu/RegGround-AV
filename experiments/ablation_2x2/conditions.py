"""Frozen condition matrix for the 2×2 ablation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ValidationMode = Literal["enforce", "observe"]


@dataclass(frozen=True)
class AblationCondition:
    name: str
    graph_rag: bool
    validation_mode: ValidationMode
    disable_rule_engine: bool = True
    digest_version: str = "v_a"


CONDITIONS: tuple[AblationCondition, ...] = (
    AblationCondition("full", graph_rag=True, validation_mode="enforce"),
    AblationCondition("no_validation", graph_rag=True, validation_mode="observe"),
    AblationCondition("no_rag", graph_rag=False, validation_mode="enforce"),
    AblationCondition("baseline", graph_rag=False, validation_mode="observe"),
)


def condition_by_name(name: str) -> AblationCondition:
    for condition in CONDITIONS:
        if condition.name == name:
            return condition
    valid = ", ".join(condition.name for condition in CONDITIONS)
    raise ValueError(f"unknown condition {name!r}; expected one of: {valid}")


__all__ = ["AblationCondition", "CONDITIONS", "ValidationMode", "condition_by_name"]
