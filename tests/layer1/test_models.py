"""Pydantic model tests for the Layer 1 v3.1 public contract."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.layer1.models import (
    AugmentedTrajectory,
    BenchmarkLabels,
    CandidateBenchmarkLabel,
    ConflictZone,
    EgoState,
    JudgeInput,
    LocationType,
    ParsedSceneBundle,
    QACategory,
    QAPair,
    RawScene,
    ScenarioType,
    SceneContext,
    SceneDescription,
    SceneFacts,
    SceneFactsDigest,
    SceneQuery,
    Trajectory,
    TrajectoryFeatures,
    TrajectorySource,
    TrajectoryVariantType,
    Waypoint,
)


def _digest() -> SceneFactsDigest:
    return SceneFactsDigest(
        signal_phase="unknown",
        phase_source="none",
        ego_turn_intent="unknown",
        governing_stop_line_signed_m=None,
        pedestrian_in_forward_crosswalk=False,
        pedestrian_moving=None,
        oncoming_vehicle_moving=False,
        location_is_intersection=None,
    )


def _waypoints(n: int = 5) -> list[Waypoint]:
    return [Waypoint(x=float(i) * 0.1, y=0.0, t=float(i) * 0.1) for i in range(n)]


def _trajectory(traj_id: str = "traj_a") -> Trajectory:
    return Trajectory(
        traj_id=traj_id,
        source=TrajectorySource.SYNTHETIC,
        waypoints=_waypoints(),
    )


def _features() -> TrajectoryFeatures:
    return TrajectoryFeatures(
        max_speed_mps=6.0,
        min_speed_mps=1.0,
        mean_speed_mps=3.0,
        total_distance_m=12.0,
        max_lateral_offset_m=0.2,
        n_waypoints=5,
        conflict_zone_behavior="后半段速度基本稳定",
        natural_language_summary="轨迹共 5 个航点，总行程 12.0 米。",
    )


class TestEnums:
    def test_scenario_types_match_v31(self) -> None:
        assert {item.value for item in ScenarioType} == {
            "red_light",
            "yellow_light",
            "pedestrian",
            "oncoming",
            "minor_road",
            "emergency",
            "unknown",
        }

    def test_variant_types_include_hard_case(self) -> None:
        assert {item.value for item in TrajectoryVariantType} == {
            "ground_truth",
            "conservative",
            "aggressive",
            "illegal",
            "suboptimal",
            "hard_case",
        }


class TestWaypoint:
    def test_t_is_required_model_field_with_zero_default(self) -> None:
        waypoint = Waypoint(x=0.0, y=0.0)
        assert waypoint.t == 0.0

    def test_coordinate_and_time_bounds(self) -> None:
        Waypoint(x=150.0, y=60.0)
        with pytest.raises(ValidationError):
            Waypoint(x=300.0, y=0.0)
        with pytest.raises(ValidationError):
            Waypoint(x=0.0, y=300.0)
        with pytest.raises(ValidationError):
            Waypoint(x=0.0, y=0.0, t=-0.1)

    def test_frozen(self) -> None:
        waypoint = Waypoint(x=1.0, y=2.0)
        with pytest.raises(ValidationError):
            waypoint.x = 3.0  # type: ignore[misc]


class TestTrajectory:
    def test_valid_trajectory_and_defaults(self) -> None:
        trajectory = _trajectory("internal_gt")
        assert trajectory.confidence is None
        assert not hasattr(trajectory, "variant_type")

    def test_waypoint_count_is_two_to_two_hundred(self) -> None:
        Trajectory(
            traj_id="max_ok",
            source=TrajectorySource.PLANNING,
            waypoints=_waypoints(200),
        )
        with pytest.raises(ValidationError):
            Trajectory(
                traj_id="too_short",
                source=TrajectorySource.PLANNING,
                waypoints=[Waypoint(x=0.0, y=0.0)],
            )
        with pytest.raises(ValidationError):
            Trajectory(
                traj_id="too_long",
                source=TrajectorySource.PLANNING,
                waypoints=_waypoints(201),
            )

    def test_frozen(self) -> None:
        trajectory = _trajectory()
        with pytest.raises(ValidationError):
            trajectory.traj_id = "new_id"  # type: ignore[misc]


class TestSceneFacts:
    def test_conflict_zone_and_scene_facts(self) -> None:
        zone = ConflictZone(
            kind="ped_crossing",
            polygon_xy=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
            distance_from_ego_m=4.2,
            source_layer="ped_crossing",
        )
        facts = SceneFacts(
            has_red_light=True,
            traffic_light_status_source="drivelm_status",
            pedestrian_in_crosswalk=True,
            conflict_zones=[zone],
        )

        assert facts.has_red_light is True
        assert facts.conflict_zones[0].kind == "ped_crossing"


class TestTrajectoryFeaturesAndAugmentedTrajectory:
    def test_features_defaults_and_validation(self) -> None:
        features = TrajectoryFeatures(
            max_speed_mps=5.0,
            min_speed_mps=2.0,
            mean_speed_mps=3.5,
            total_distance_m=10.0,
            max_lateral_offset_m=0.1,
            n_waypoints=5,
        )
        assert features.conflict_zone_behavior == ""
        assert features.natural_language_summary == ""
        with pytest.raises(ValidationError):
            TrajectoryFeatures(
                max_speed_mps=-1.0,
                min_speed_mps=0.0,
                mean_speed_mps=0.0,
                total_distance_m=0.0,
                max_lateral_offset_m=0.0,
                n_waypoints=2,
            )

    def test_augmented_trajectory_carries_benchmark_only_label(self) -> None:
        augmented = AugmentedTrajectory(
            trajectory=_trajectory("internal_illegal"),
            variant_type=TrajectoryVariantType.ILLEGAL,
            expected_verdict="vetoed",
            generation_note="benchmark-only",
        )
        assert augmented.variant_type == TrajectoryVariantType.ILLEGAL
        assert not hasattr(augmented.trajectory, "variant_type")


class TestDescriptionsAndRawInputs:
    def test_description_ego_state_and_qa_pair(self) -> None:
        desc = SceneDescription(
            narrative="The ego vehicle approaches an intersection.",
            keywords=["intersection"],
            location_type=LocationType.INTERSECTION,
        )
        ego = EgoState(speed_mps=3.0)
        qa = QAPair(
            category=QACategory.PERCEPTION,
            question="What is visible?",
            answer="A traffic light is visible.",
        )

        assert desc.location_type == LocationType.INTERSECTION
        assert ego.speed_mps == 3.0
        assert qa.category == QACategory.PERCEPTION

    def test_raw_scene_accepts_v31_inputs(self) -> None:
        raw = RawScene(
            scene_id="scene_1",
            frame_token="frame_1",
            source_file=Path("/fake/drivelm.json"),
            raw_json={"qa_pairs": []},
            map_records={"stop_line": [{"polygon_xy": [(1.0, 0.0), (1.0, 1.0)]}]},
            annotations=[{"category_name": "vehicle.car"}],
        )
        assert raw.location == "boston-seaport"
        assert raw.map_records["stop_line"]


class TestSceneContext:
    def test_accepts_only_anonymized_candidate_ids(self) -> None:
        context = SceneContext(
            scene_id="scene_1",
            frame_token="frame_1",
            candidate_trajectories=[_trajectory("traj_a"), _trajectory("traj_b")],
            trajectory_features=[_features(), _features()],
        )
        assert [t.traj_id for t in context.candidate_trajectories] == ["traj_a", "traj_b"]

    def test_rejects_internal_candidate_ids(self) -> None:
        with pytest.raises(ValidationError):
            SceneContext(
                scene_id="scene_1",
                candidate_trajectories=[_trajectory("scene_1_synthetic_illegal_0")],
            )

    def test_rejects_feature_alignment_mismatch(self) -> None:
        with pytest.raises(ValidationError):
            SceneContext(
                scene_id="scene_1",
                candidate_trajectories=[_trajectory("traj_a"), _trajectory("traj_b")],
                trajectory_features=[_features()],
            )

    def test_context_does_not_carry_variant_label(self) -> None:
        context = SceneContext(scene_id="scene_1")
        assert not hasattr(context, "variant_type")


class TestDownstreamInterfaces:
    def test_scene_query_and_judge_input_are_narrow(self) -> None:
        facts = SceneFacts(has_yellow_light=True, traffic_light_status_source="drivelm_status")
        query = SceneQuery(
            scene_id="scene_1",
            frame_token="frame_1",
            scenario_type=ScenarioType.YELLOW_LIGHT,
            keywords=["yellow light"],
            scene_facts=facts,
        )
        judge_input = JudgeInput(
            scene_id="scene_1",
            frame_token="frame_1",
            narrative="A yellow light is visible.",
            scene_facts_digest=_digest(),
            trajectory_features=[_features()],
        )

        assert "candidate_trajectories" not in query.model_dump()
        assert "waypoints" not in judge_input.model_dump()

    def test_extra_fields_are_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            SceneQuery(
                scene_id="scene_1",
                frame_token="frame_1",
                scenario_type=ScenarioType.UNKNOWN,
                candidate_trajectories=[],  # type: ignore[call-arg]
            )
        with pytest.raises(ValidationError):
            JudgeInput(
                scene_id="scene_1",
                frame_token="frame_1",
                narrative="safe",
                labels=[],  # type: ignore[call-arg]
            )

    def test_parsed_scene_bundle_contains_context_interfaces_and_benchmark_labels(self) -> None:
        context = SceneContext(
            scene_id="scene_1",
            frame_token="frame_1",
            description=SceneDescription(narrative="A red light is visible."),
            scenario_type=ScenarioType.RED_LIGHT,
            candidate_trajectories=[_trajectory("traj_a")],
            trajectory_features=[_features()],
        )
        query = SceneQuery(
            scene_id="scene_1",
            frame_token="frame_1",
            scenario_type=ScenarioType.RED_LIGHT,
            scene_facts=SceneFacts(
                has_red_light=True,
                traffic_light_status_source="drivelm_status",
            ),
        )
        judge_input = JudgeInput(
            scene_id="scene_1",
            frame_token="frame_1",
            narrative="A red light is visible.",
            scene_facts_digest=_digest(),
            trajectory_features=[_features()],
        )
        labels = BenchmarkLabels(
            scene_id="scene_1",
            frame_token="frame_1",
            labels=[
                CandidateBenchmarkLabel(
                    anonymous_id="traj_a",
                    original_internal_id="scene_1_planning_gt_0",
                    variant_type=TrajectoryVariantType.GROUND_TRUTH,
                    expected_verdict="cleared",
                    difficulty="easy",
                    gt_precheck_status="passed",
                ),
            ],
        )

        bundle = ParsedSceneBundle(
            context=context,
            scene_query=query,
            judge_input=judge_input,
            benchmark_labels=labels,
        )

        assert bundle.context.candidate_trajectories[0].traj_id == "traj_a"
        assert bundle.benchmark_labels.labels[0].variant_type == TrajectoryVariantType.GROUND_TRUTH
