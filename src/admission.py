"""编排层与 availability spike 共用的场景准入规则。"""

from src.condition_mapping import FACT_TO_CONDITION
from src.layer1.models import ScenarioType, SceneQuery

MIN_EGO_DISPLACEMENT_M = 1.0


def admission_reason(query: SceneQuery) -> str | None:
    """返回准入失败原因；通过时返回 None。"""
    if query.scenario_type in {ScenarioType.YELLOW_LIGHT, ScenarioType.EMERGENCY}:
        return f"scope_cut_{query.scenario_type.value}"
    if query.scene_facts.window_insufficient:
        return "not_evaluable_short_window"
    if query.scenario_type == ScenarioType.UNKNOWN:
        return "scenario_type_unknown"
    displacement = query.scene_facts.ego_displacement_m
    if displacement is not None and displacement < MIN_EGO_DISPLACEMENT_M:
        return "ego_stationary_low_information"
    if not any(
        getattr(query.scene_facts, fact_key, False) is True
        for fact_key in FACT_TO_CONDITION
    ):
        return "no_structured_regulatory_fact"
    return None
