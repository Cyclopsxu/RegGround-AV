"""Layer 2 models.py 单元测试。"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.layer2.models import (
    ActorType,
    ConditionSpec,
    Consequence,
    RetrievalMode,
    RoadActor,
    RuleCategory,
    RuleNode,
    RuleOverride,
    RuleStatus,
    RuleSubgraph,
    Severity,
)


class TestEnums:
    """枚举类型测试。"""

    def test_retrieval_mode_values(self) -> None:
        assert RetrievalMode.STRUCTURED == "structured"
        assert RetrievalMode.KEYWORD_FALLBACK == "keyword_fallback"
        assert RetrievalMode.MIXED == "mixed"

    def test_rule_status_values(self) -> None:
        assert RuleStatus.ACTIVE == "active"
        assert RuleStatus.OVERRIDDEN == "overridden"


class TestConditionSpec:
    """ConditionSpec 模型测试。"""

    def test_create_valid(self) -> None:
        spec = ConditionSpec(
            condition_id="C-RED-PHASE",
            description="red light",
            keywords=["red light"],
            fact_keys=["has_red_light"],
        )
        assert spec.condition_id == "C-RED-PHASE"
        assert spec.fact_keys == ["has_red_light"]


class TestRoadActor:
    """RoadActor 模型测试。"""

    def test_create_valid(self) -> None:
        actor = RoadActor(type=ActorType.VEHICLE, description="机动车", priority_level=3)
        assert actor.type == ActorType.VEHICLE
        assert actor.priority_level == 3

    def test_priority_level_invalid(self) -> None:
        with pytest.raises(ValidationError):
            RoadActor(type=ActorType.VEHICLE, description="x", priority_level=0)


class TestConsequence:
    """Consequence 模型测试。"""

    def test_create_valid(self) -> None:
        consequence = Consequence(
            type="signal_violation",
            description="信号灯违法",
            severity_score=9,
        )
        assert consequence.type == "signal_violation"

    def test_severity_score_invalid(self) -> None:
        with pytest.raises(ValidationError):
            Consequence(type="x", description="x", severity_score=11)


class TestRuleOverride:
    """RuleOverride 模型测试。"""

    def test_create_valid(self) -> None:
        override = RuleOverride(
            overriding_rule_id="R-YLD-04",
            overridden_rule_id="R-SIG-01",
            condition_id="C-EMERGENCY-ACTIVE",
            reason="emergency priority",
        )
        assert override.overridden_rule_id == "R-SIG-01"


class TestRuleNode:
    """RuleNode 模型测试。"""

    @staticmethod
    def make_rule(**overrides: object) -> RuleNode:
        defaults: dict[str, object] = {
            "node_id": "R-SIG-01",
            "code": "RoadTrafficSafetyLaw-38-red",
            "description": "红灯相位下禁止越过停止线继续行驶",
            "severity": Severity.HARD,
            "category": RuleCategory.SIGNAL,
            "retrieved_via": RetrievalMode.STRUCTURED,
        }
        defaults.update(overrides)
        return RuleNode(**defaults)  # type: ignore[arg-type]

    def test_create_minimal(self) -> None:
        rule = self.make_rule()
        assert rule.node_id == "R-SIG-01"
        assert rule.actors == []
        assert rule.consequences == []
        assert rule.matched_conditions == []
        assert rule.status == RuleStatus.ACTIVE
        assert rule.overridden_by == []

    def test_create_full(self) -> None:
        actor = RoadActor(type=ActorType.VEHICLE, description="机动车", priority_level=3)
        consequence = Consequence(type="signal_violation", description="违法", severity_score=9)
        rule = self.make_rule(
            actors=[actor],
            consequences=[consequence],
            matched_conditions=["C-RED-PHASE"],
            status=RuleStatus.OVERRIDDEN,
            overridden_by=["R-YLD-04"],
        )
        assert rule.actors == [actor]
        assert rule.consequences == [consequence]
        assert rule.status == RuleStatus.OVERRIDDEN

    def test_frozen_immutable(self) -> None:
        rule = self.make_rule()
        with pytest.raises(ValidationError):
            rule.description = "new"  # type: ignore[misc]


class TestRuleSubgraph:
    """RuleSubgraph 模型测试。"""

    @staticmethod
    def make_subgraph(
        rules: list[RuleNode] | None = None,
        *,
        matched_conditions: list[str] | None = None,
        overrides_applied: list[RuleOverride] | None = None,
    ) -> RuleSubgraph:
        return RuleSubgraph(
            scene_id="scene-001",
            frame_token="frame-001",
            rules=rules or [],
            matched_conditions=matched_conditions or [],
            retrieval_mode=RetrievalMode.STRUCTURED,
            overrides_applied=overrides_applied or [],
            query_duration_ms=5,
            graph_version="test-v3.0",
            retrieval_timestamp=datetime.now(UTC),
        )

    def test_empty_rules(self) -> None:
        subgraph = self.make_subgraph()
        assert subgraph.rules == []
        assert subgraph.active_rules == []
        assert subgraph.hard_rules == []
        assert subgraph.soft_rules == []

    def test_active_rules_exclude_overridden(self) -> None:
        active = TestRuleNode.make_rule(node_id="R-YLD-04", category=RuleCategory.YIELD)
        overridden = TestRuleNode.make_rule(
            node_id="R-SIG-01",
            status=RuleStatus.OVERRIDDEN,
            overridden_by=["R-YLD-04"],
        )
        subgraph = self.make_subgraph(rules=[active, overridden])
        assert subgraph.active_rules == [active]
        assert subgraph.hard_rules == [active]

    def test_soft_rules_from_active_only(self) -> None:
        active_soft = TestRuleNode.make_rule(
            node_id="R-SIG-03",
            severity=Severity.SOFT,
            category=RuleCategory.SIGNAL,
        )
        overridden_soft = TestRuleNode.make_rule(
            node_id="R-YLD-05",
            severity=Severity.SOFT,
            category=RuleCategory.YIELD,
            status=RuleStatus.OVERRIDDEN,
        )
        subgraph = self.make_subgraph(rules=[active_soft, overridden_soft])
        assert subgraph.soft_rules == [active_soft]
