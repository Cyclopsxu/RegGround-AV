"""场景准入规则测试。"""

from __future__ import annotations

from src.admission import admission_reason
from src.layer1.models import ScenarioType, SceneFacts, SceneQuery


def _query(displacement: float | None) -> SceneQuery:
    return SceneQuery(
        scene_id="scene",
        frame_token="frame",
        scenario_type=ScenarioType.RED_LIGHT,
        scene_facts=SceneFacts(
            has_red_light=True,
            ego_displacement_m=displacement,
        ),
    )


def test_stationary_scene_is_rejected_as_low_information() -> None:
    assert admission_reason(_query(0.2)) == "ego_stationary_low_information"


def test_moving_scene_remains_admitted() -> None:
    assert admission_reason(_query(1.0)) is None
