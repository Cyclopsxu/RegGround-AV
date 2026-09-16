"""Layer 2 数据模型定义。"""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, computed_field

from src.layer1.models import ScenarioType, SceneFacts, SceneQuery


class Severity(StrEnum):
    """法规强制等级。决定 Layer 3 走一票否决还是打分路径。"""

    HARD = "hard"
    SOFT = "soft"


class RuleCategory(StrEnum):
    """法规类别。v2.0 范围收窄为信号灯与让行两类。"""

    SIGNAL = "signal"
    YIELD = "yield"


class RetrievalMode(StrEnum):
    """检索模式，用于 telemetry 与降级标记。"""

    STRUCTURED = "structured"
    KEYWORD_FALLBACK = "keyword_fallback"
    MIXED = "mixed"


class RuleStatus(StrEnum):
    """召回规则在冲突消解后的状态。"""

    ACTIVE = "active"
    OVERRIDDEN = "overridden"


class ActorType(StrEnum):
    """道路参与者类型，与图中 RoadActor.type 对齐。"""

    VEHICLE = "vehicle"
    PEDESTRIAN = "pedestrian"
    CYCLIST = "cyclist"
    EMERGENCY_VEHICLE = "emergency_vehicle"


class RoadActor(BaseModel):
    """对应图中的 RoadActor 节点。"""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    type: ActorType
    description: str
    priority_level: int = Field(ge=1, le=5)


class Consequence(BaseModel):
    """违规后果节点。对应图中的 Consequence 节点。"""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    type: str
    description: str
    severity_score: int = Field(ge=1, le=10)


class ConditionSpec(BaseModel):
    """图谱中的 Condition 节点规格，供条件匹配器使用。"""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    condition_id: str
    description: str
    keywords: list[str] = Field(default_factory=list)
    fact_keys: list[str] = Field(default_factory=list)


class RuleOverride(BaseModel):
    """一条规则在特定条件下覆盖另一条规则。"""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    overriding_rule_id: str
    overridden_rule_id: str
    condition_id: str
    reason: str


class RuleNode(BaseModel):
    """一条法规的完整表示，可溯源到图中具体节点。

    每个 RuleNode 都是完全可追溯的——node_id 直接对应图中的节点 id，
    可在偏好标签中作为 cited_rule_id 的来源。
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    # 身份字段（对应图节点属性）
    node_id: str
    code: str
    description: str
    severity: Severity
    category: RuleCategory

    # 关联字段（遍历收集的边）
    actors: list[RoadActor] = Field(default_factory=list)
    consequences: list[Consequence] = Field(default_factory=list)

    # 溯源字段（满足 NFR-03）
    matched_conditions: list[str] = Field(default_factory=list)
    retrieved_via: RetrievalMode
    status: RuleStatus = RuleStatus.ACTIVE
    overridden_by: list[str] = Field(default_factory=list)


class RuleSubgraph(BaseModel):
    """Layer 2 对外的最终输出，承载一次检索的完整结果。"""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    scene_id: str
    frame_token: str
    rules: list[RuleNode]
    matched_conditions: list[str]
    retrieval_mode: RetrievalMode
    overrides_applied: list[RuleOverride] = Field(default_factory=list)
    query_duration_ms: int
    graph_version: str
    retrieval_timestamp: datetime

    @computed_field  # type: ignore[prop-decorator]
    @property
    def active_rules(self) -> list[RuleNode]:
        """未被 override 覆盖的规则，Layer 3 默认消费该集合。"""
        return [r for r in self.rules if r.status == RuleStatus.ACTIVE]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def hard_rules(self) -> list[RuleNode]:
        """severity=hard 的规则子集，Layer 3 的一票否决输入。"""
        return [r for r in self.active_rules if r.severity == Severity.HARD]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def soft_rules(self) -> list[RuleNode]:
        """severity=soft 的规则子集，Layer 3 的打分参考。"""
        return [r for r in self.active_rules if r.severity == Severity.SOFT]


__all__ = [
    "ActorType",
    "ConditionSpec",
    "Consequence",
    "RetrievalMode",
    "RoadActor",
    "RuleCategory",
    "RuleNode",
    "RuleOverride",
    "RuleStatus",
    "RuleSubgraph",
    "SceneFacts",
    "SceneQuery",
    "ScenarioType",
    "Severity",
]
