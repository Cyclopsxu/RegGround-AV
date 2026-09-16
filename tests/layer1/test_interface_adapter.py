"""InterfaceAdapter tests for Layer 2/3 narrow interfaces."""

from __future__ import annotations

import pytest

from src.layer1.exceptions import SceneValidationError
from src.layer1.interface_adapter import InterfaceAdapter
from src.layer1.models import (
    ScenarioType,
    SceneContext,
    SceneDescription,
    SceneFacts,
    Trajectory,
    TrajectoryFeatures,
    TrajectorySource,
    Waypoint,
)


def _trajectory(traj_id: str = "traj_a") -> Trajectory:
    return Trajectory(
        traj_id=traj_id,
        source=TrajectorySource.SYNTHETIC,
        waypoints=[
            Waypoint(x=0.0, y=0.0, t=0.0),
            Waypoint(x=1.0, y=0.0, t=0.5),
        ],
    )


def _features(summary: str = "轨迹共 2 个航点，总行程 1.0 米。") -> TrajectoryFeatures:
    return TrajectoryFeatures(
        max_speed_mps=2.0,
        min_speed_mps=2.0,
        mean_speed_mps=2.0,
        total_distance_m=1.0,
        max_lateral_offset_m=0.0,
        n_waypoints=2,
        conflict_zone_behavior="后半段速度基本稳定",
        natural_language_summary=summary,
    )


def _context(
    *,
    narrative: str = "A red traffic light is visible.",
    summary: str = "轨迹共 2 个航点，总行程 1.0 米。",
) -> SceneContext:
    return SceneContext(
        scene_id="scene_1",
        frame_token="frame_1",
        description=SceneDescription(narrative=narrative, keywords=["red light"]),
        scenario_type=ScenarioType.RED_LIGHT,
        scene_facts=SceneFacts(has_red_light=True, traffic_light_status_source="drivelm_status"),
        candidate_trajectories=[_trajectory("traj_a")],
        trajectory_features=[_features(summary)],
    )


def _has_key(value: object, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_has_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_has_key(item, key) for item in value)
    return False


class TestInterfaceAdapter:
    def test_scene_query_exposes_only_layer2_fields(self) -> None:
        query = InterfaceAdapter().to_scene_query(_context())

        dumped = query.model_dump()
        assert dumped["scenario_type"] == ScenarioType.RED_LIGHT
        assert dumped["scene_facts"]["has_red_light"] is True
        assert "candidate_trajectories" not in dumped
        assert "trajectory_features" not in dumped
        assert "benchmark_labels" not in dumped

    def test_judge_input_exposes_features_without_waypoints_or_labels(self) -> None:
        judge_input = InterfaceAdapter().to_judge_input(_context())
        dumped = judge_input.model_dump()
        rendered = str(dumped).lower()

        assert dumped["narrative"] == "A red traffic light is visible."
        assert "trajectory_features" in dumped
        assert not _has_key(dumped, "waypoints")
        assert "variant" not in rendered
        assert "vetoed" not in rendered

    def test_judge_input_scans_narrative(self) -> None:
        with pytest.raises(SceneValidationError):
            InterfaceAdapter().to_judge_input(_context(narrative="This variant is illegal."))

    def test_judge_input_scans_feature_summaries(self) -> None:
        with pytest.raises(SceneValidationError):
            InterfaceAdapter().to_judge_input(_context(summary="The vehicle must yield."))
