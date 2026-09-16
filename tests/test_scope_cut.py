"""v2 场景裁剪与短窗口准入规则。"""

from src.admission import admission_reason
from src.layer1.models import ScenarioType, SceneFacts, SceneQuery


def _query(scenario: ScenarioType, facts: SceneFacts) -> SceneQuery:
    return SceneQuery(
        scene_id="scene",
        frame_token="frame",
        scenario_type=scenario,
        scene_facts=facts,
    )


def test_yellow_and_emergency_are_scope_cut() -> None:
    assert admission_reason(_query(
        ScenarioType.YELLOW_LIGHT,
        SceneFacts(has_yellow_light=True),
    )) == "scope_cut_yellow_light"
    assert admission_reason(_query(
        ScenarioType.EMERGENCY,
        SceneFacts(emergency_vehicle_active=True),
    )) == "scope_cut_emergency"


def test_short_window_is_candidate_exclusion_and_not_main_admission() -> None:
    reason = admission_reason(_query(
        ScenarioType.RED_LIGHT,
        SceneFacts(has_red_light=True, window_insufficient=True),
    ))
    assert reason == "not_evaluable_short_window"
