"""GraphRAG 与 citation validity 的 2x2 消融配置。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AblationCondition:
    name: str
    use_graphrag: bool
    use_citation_validation: bool


ABLATION_2X2: tuple[AblationCondition, ...] = (
    AblationCondition("full", use_graphrag=True, use_citation_validation=True),
    AblationCondition("ablation_a", use_graphrag=True, use_citation_validation=False),
    AblationCondition("ablation_b", use_graphrag=False, use_citation_validation=True),
    AblationCondition("baseline", use_graphrag=False, use_citation_validation=False),
)
