"""Layer 2 conflict_resolver.py 单元测试。"""

from src.layer2.conflict_resolver import ConflictResolver
from src.layer2.models import (
    RetrievalMode,
    RuleCategory,
    RuleNode,
    RuleOverride,
    RuleStatus,
    Severity,
)


def make_rule(
    node_id: str,
    *,
    severity: Severity = Severity.HARD,
    category: RuleCategory = RuleCategory.SIGNAL,
) -> RuleNode:
    """创建测试用 RuleNode。"""
    return RuleNode(
        node_id=node_id,
        code=f"code-{node_id}",
        description=f"description-{node_id}",
        severity=severity,
        category=category,
        retrieved_via=RetrievalMode.STRUCTURED,
    )


def make_override() -> RuleOverride:
    """创建 emergency 覆盖红灯规则的测试 override。"""
    return RuleOverride(
        overriding_rule_id="R-YLD-04",
        overridden_rule_id="R-SIG-01",
        condition_id="C-EMERGENCY-ACTIVE",
        reason="emergency priority",
    )


class TestConflictResolver:
    """规则冲突消解测试。"""

    def test_emergency_override_marks_red_light_overridden(self) -> None:
        resolver = ConflictResolver()
        rules = [
            make_rule("R-SIG-01"),
            make_rule("R-YLD-04", category=RuleCategory.YIELD),
        ]
        resolved, applied = resolver.resolve(
            rules,
            ["C-RED-PHASE", "C-EMERGENCY-ACTIVE"],
            [make_override()],
        )

        by_id = {rule.node_id: rule for rule in resolved}
        assert applied == [make_override()]
        assert by_id["R-SIG-01"].status == RuleStatus.OVERRIDDEN
        assert by_id["R-SIG-01"].overridden_by == ["R-YLD-04"]
        assert by_id["R-YLD-04"].status == RuleStatus.ACTIVE

    def test_override_ignored_when_condition_not_matched(self) -> None:
        resolver = ConflictResolver()
        rules = [
            make_rule("R-SIG-01"),
            make_rule("R-YLD-04", category=RuleCategory.YIELD),
        ]
        resolved, applied = resolver.resolve(rules, ["C-RED-PHASE"], [make_override()])
        assert resolved == rules
        assert applied == []

    def test_override_ignored_when_rule_endpoint_missing(self) -> None:
        resolver = ConflictResolver()
        rules = [make_rule("R-YLD-04", category=RuleCategory.YIELD)]
        resolved, applied = resolver.resolve(rules, ["C-EMERGENCY-ACTIVE"], [make_override()])
        assert resolved == rules
        assert applied == []
