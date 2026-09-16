"""Layer 2 规则冲突消解。"""

from collections import defaultdict

from src.layer2.models import RuleNode, RuleOverride, RuleStatus


class ConflictResolver:
    """处理 OVERRIDES / EXCEPTION_WHEN 规则冲突。"""

    def resolve(
        self,
        rules: list[RuleNode],
        matched_conditions: list[str],
        overrides: list[RuleOverride],
    ) -> tuple[list[RuleNode], list[RuleOverride]]:
        """应用满足条件的 override，并保留被覆盖规则用于审计。"""
        if not rules or not overrides:
            return rules, []

        rule_by_id = {rule.node_id: rule for rule in rules}
        matched_condition_set = set(matched_conditions)
        overridden_by: dict[str, list[str]] = defaultdict(list)
        applied: list[RuleOverride] = []

        for override in sorted(
            overrides,
            key=lambda item: (
                item.overridden_rule_id,
                item.overriding_rule_id,
                item.condition_id,
            ),
        ):
            if override.condition_id not in matched_condition_set:
                continue
            if override.overriding_rule_id not in rule_by_id:
                continue
            if override.overridden_rule_id not in rule_by_id:
                continue
            overridden_by[override.overridden_rule_id].append(override.overriding_rule_id)
            applied.append(override)

        if not applied:
            return rules, []

        resolved: list[RuleNode] = []
        for rule in rules:
            if rule.node_id not in overridden_by:
                resolved.append(rule)
                continue
            resolved.append(
                rule.model_copy(
                    update={
                        "status": RuleStatus.OVERRIDDEN,
                        "overridden_by": sorted(set(overridden_by[rule.node_id])),
                    }
                )
            )

        return resolved, applied
