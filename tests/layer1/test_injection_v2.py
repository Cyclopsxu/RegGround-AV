"""场景相关违规注入与不可注入标记。"""

from src.layer1.benchmark_builder import build_benchmark_candidates
from src.layer1.models import (
    AugmentedTrajectory,
    ScenarioType,
    SceneDescription,
    SceneFacts,
    Trajectory,
    TrajectorySource,
    TrajectoryVariantType,
    Waypoint,
)
from src.layer1.trajectory_analyzer import TrajectoryAnalyzer
from src.layer1.trajectory_augmentor import TrajectoryAugmentor


def _ground_truth() -> Trajectory:
    return Trajectory(
        traj_id="scene_planning_gt_0",
        source=TrajectorySource.PLANNING,
        waypoints=[
            Waypoint(x=float(index), y=0.0, t=float(index))
            for index in range(7)
        ],
    )


def _illegal(scenario: ScenarioType, facts: SceneFacts):
    candidates = TrajectoryAugmentor().augment(
        _ground_truth(),
        scenario,
        facts,
        scene_id="scene",
    )
    return next(
        candidate
        for candidate in candidates
        if candidate.variant_type == TrajectoryVariantType.ILLEGAL
    )


def test_red_illegal_crosses_governing_line_inside_snapshot_horizon() -> None:
    candidate = _illegal(
        ScenarioType.RED_LIGHT,
        SceneFacts(
            has_red_light=True,
            governing_stop_line_signed_m=5.0,
        ),
    )
    crossing_time = next(
        waypoint.t for waypoint in candidate.trajectory.waypoints if waypoint.x >= 5.0
    )
    assert crossing_time <= 2.0


def test_nonforward_pedestrian_and_nonleft_oncoming_are_not_injectable() -> None:
    pedestrian = _illegal(ScenarioType.PEDESTRIAN, SceneFacts())
    oncoming = _illegal(
        ScenarioType.ONCOMING,
        SceneFacts(oncoming_vehicle_moving=True, ego_turn_intent="straight"),
    )
    assert pedestrian.generation_note.startswith("no_violation_injectable")
    assert oncoming.generation_note.startswith("no_violation_injectable")


def test_visible_feature_collision_does_not_mutate_geometry() -> None:
    ground_truth = _ground_truth()
    candidates = [
        AugmentedTrajectory(
            trajectory=ground_truth,
            variant_type=TrajectoryVariantType.GROUND_TRUTH,
            expected_verdict="exclude",
        ),
        AugmentedTrajectory(
            trajectory=ground_truth.model_copy(update={"traj_id": "illegal"}),
            variant_type=TrajectoryVariantType.ILLEGAL,
            expected_verdict="exclude",
        ),
    ]
    built = build_benchmark_candidates(
        candidates,
        TrajectoryAnalyzer(),
        SceneDescription(),
        SceneFacts(),
        ScenarioType.UNKNOWN,
    )
    assert built[1].is_degenerate is True
    assert built[1].exclude_reason == "no_violation_injectable"
    assert [
        (waypoint.x, waypoint.y) for waypoint in built[1].trajectory.waypoints
    ] == [
        (waypoint.x, waypoint.y) for waypoint in ground_truth.waypoints
    ]
