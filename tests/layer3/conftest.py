"""Layer 3 v3.0 测试夹具。"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.layer1.models import (
    ConflictZoneMetrics,
    JudgeInput,
    SceneContext,
    SceneDescription,
    SceneFactsDigest,
    Trajectory,
    TrajectoryFeatures,
    TrajectorySource,
    Waypoint,
)
from src.layer2.models import (
    ActorType,
    Consequence,
    RetrievalMode,
    RoadActor,
    RuleCategory,
    RuleNode,
    RuleSubgraph,
    Severity,
)
from src.layer3.context_builder import ContextBuilder
from src.layer3.models import Layer3Settings


@pytest.fixture
def layer3_settings() -> Layer3Settings:
    return Layer3Settings(
        llm_provider="anthropic",
        llm_model="claude-sonnet-4-6",
        llm_api_key="test-key",  # type: ignore[arg-type]
        llm_max_retries=0,
        hard_filter_max_retries=0,
        deferred_retry_delay_seconds=0.0,
    )


@pytest.fixture
def sample_features() -> list[TrajectoryFeatures]:
    return [
        TrajectoryFeatures(
            max_speed_mps=4.0,
            min_speed_mps=0.0,
            mean_speed_mps=1.6,
            total_distance_m=15.0,
            max_lateral_offset_m=0.1,
            n_waypoints=10,
            entered_conflict_zone=False,
            min_distance_to_conflict_m=0.5,
            full_stop=True,
            stop_duration_s=1.5,
            conflict_zone_behavior=(
                "后半段进入速度 2.0 m/s；后半段最低速度 0.0 m/s；"
                "后半段离开速度 0.0 m/s"
            ),
            natural_language_summary="车辆逐步减速并在冲突区前完全停止，路径基本直行。",
            conflict_zone_metrics=[ConflictZoneMetrics(
                kind="stop_line",
                governing=True,
                min_distance_m=0.5,
                entered=False,
                stopped_before_zone=True,
            )],
        ),
        TrajectoryFeatures(
            max_speed_mps=3.0,
            min_speed_mps=0.1,
            mean_speed_mps=1.4,
            total_distance_m=15.0,
            max_lateral_offset_m=0.0,
            n_waypoints=10,
            entered_conflict_zone=False,
            min_distance_to_conflict_m=0.2,
            full_stop=False,
            conflict_zone_behavior=(
                "后半段进入速度 1.0 m/s；后半段最低速度 0.1 m/s；"
                "后半段离开速度 0.1 m/s"
            ),
            natural_language_summary="车辆提前减速，后半段以极低速度缓慢滚动。",
            conflict_zone_metrics=[ConflictZoneMetrics(
                kind="stop_line",
                governing=True,
                min_distance_m=0.2,
                entered=False,
                stopped_before_zone=False,
            )],
        ),
        TrajectoryFeatures(
            max_speed_mps=4.8,
            min_speed_mps=4.3,
            mean_speed_mps=4.5,
            total_distance_m=15.0,
            max_lateral_offset_m=0.0,
            n_waypoints=10,
            entered_conflict_zone=True,
            min_distance_to_conflict_m=0.0,
            full_stop=False,
            stopped_before_zone=False,
            conflict_zone_behavior=(
                "后半段进入速度 4.5 m/s；后半段最低速度 4.3 m/s；"
                "后半段离开速度 4.3 m/s"
            ),
            natural_language_summary="车辆保持较高速度穿过冲突区，未出现明显减速。",
            conflict_zone_metrics=[ConflictZoneMetrics(
                kind="stop_line",
                governing=True,
                min_distance_m=0.0,
                entered=True,
                entry_time_s=1.0,
                penetration_depth_m=0.5,
                min_speed_in_zone_mps=4.3,
                stopped_before_zone=False,
            )],
        ),
    ]


@pytest.fixture
def sample_judge_input(sample_features: list[TrajectoryFeatures]) -> JudgeInput:
    return JudgeInput(
        scene_id="scene_001",
        frame_token="frame_001",
        narrative="城市交叉路口，红灯相位，自车接近停止线。",
        scene_facts_digest=SceneFactsDigest(
            signal_phase="red",
            phase_source="drivelm_status",
            ego_turn_intent="straight",
            governing_stop_line_signed_m=5.0,
            pedestrian_in_forward_crosswalk=False,
            pedestrian_moving=None,
            oncoming_vehicle_moving=False,
            location_is_intersection=True,
        ),
        trajectory_features=sample_features,
    )


@pytest.fixture
def sample_scene_context(sample_features: list[TrajectoryFeatures]) -> SceneContext:
    trajectories = [
        Trajectory(
            traj_id="traj_a",
            source=TrajectorySource.PLANNING,
            waypoints=[
                Waypoint(x=0.0, y=0.0, t=0.0),
                Waypoint(x=1.0, y=0.0, t=1.0),
            ],
        ),
        Trajectory(
            traj_id="traj_b",
            source=TrajectorySource.PLANNING,
            waypoints=[
                Waypoint(x=0.0, y=0.0, t=0.0),
                Waypoint(x=0.5, y=0.0, t=1.0),
            ],
        ),
        Trajectory(
            traj_id="traj_c",
            source=TrajectorySource.SYNTHETIC,
            waypoints=[
                Waypoint(x=0.0, y=0.0, t=0.0),
                Waypoint(x=4.0, y=0.0, t=1.0),
            ],
        ),
    ]
    return SceneContext(
        scene_id="scene_001",
        frame_token="frame_001",
        description=SceneDescription(
            narrative="城市交叉路口，红灯相位，自车接近停止线。",
            keywords=["intersection", "red_light"],
        ),
        candidate_trajectories=trajectories,
        trajectory_features=sample_features,
    )


@pytest.fixture
def sample_rule_subgraph() -> RuleSubgraph:
    rules = [
        RuleNode(
            node_id="R-SIG-01",
            code="道交法-38",
            description="红灯相位下禁止越过停止线",
            severity=Severity.HARD,
            category=RuleCategory.SIGNAL,
            actors=[
                RoadActor(type=ActorType.VEHICLE, description="机动车", priority_level=5)
            ],
            consequences=[
                Consequence(type="罚款", description="红灯越线处罚", severity_score=9)
            ],
            matched_conditions=["C-RED-PHASE"],
            retrieved_via=RetrievalMode.STRUCTURED,
        ),
        RuleNode(
            node_id="R-SIG-03",
            code="道交法-安全距离",
            description="建议在停止线前预留安全制动距离",
            severity=Severity.SOFT,
            category=RuleCategory.SIGNAL,
            actors=[
                RoadActor(type=ActorType.VEHICLE, description="机动车", priority_level=3)
            ],
            consequences=[],
            matched_conditions=["C-RED-PHASE"],
            retrieved_via=RetrievalMode.STRUCTURED,
        ),
    ]
    return RuleSubgraph(
        scene_id="scene_001",
        frame_token="frame_001",
        rules=rules,
        matched_conditions=["C-RED-PHASE"],
        retrieval_mode=RetrievalMode.STRUCTURED,
        query_duration_ms=3,
        graph_version="test",
        retrieval_timestamp=datetime.now(UTC),
    )


@pytest.fixture
def sample_judge_context(
    layer3_settings: Layer3Settings,
    sample_judge_input: JudgeInput,
    sample_rule_subgraph: RuleSubgraph,
):
    return ContextBuilder(layer3_settings).build(sample_judge_input, sample_rule_subgraph)


class FakeTextLLM:
    def __init__(self, text: str = "摘要") -> None:
        self.text = text

    def complete_structured(self, *args, **kwargs):
        raise AssertionError("structured LLM should not be called")

    def complete_text(self, *args, **kwargs):
        return self.text, 10
