"""TrajectoryAugmentor tests for v3.1 candidate generation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.layer1.exceptions import TrajectoryAugmentationError
from src.layer1.models import (
    ConflictZone,
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


def _make_gt_trajectory(
    *,
    scene_id: str = "test_scene",
    n_waypoints: int = 20,
    with_deceleration: bool = False,
) -> Trajectory:
    waypoints: list[Waypoint] = []
    x = 0.0
    for i in range(n_waypoints):
        t = float(i) * 0.3
        if i == 0:
            x = 0.0
        elif with_deceleration and i >= n_waypoints // 2:
            x += 0.2
        else:
            x += 1.2
        waypoints.append(Waypoint(x=float(x), y=0.0, t=t))
    return Trajectory(
        traj_id=f"{scene_id}_planning_gt_0",
        source=TrajectorySource.PLANNING,
        waypoints=waypoints,
    )


def _augment(
    scenario_type: ScenarioType = ScenarioType.UNKNOWN,
    *,
    gt: Trajectory | None = None,
    path: Trajectory | None = None,
    drivable_area_polygons: list[list[tuple[float, float]]] | None = None,
    scene_id: str = "s1",
) -> list:
    ground_truth = gt or _make_gt_trajectory(scene_id=scene_id)
    return TrajectoryAugmentor(seed=42).augment(
        ground_truth,
        scenario_type,
        SceneFacts(),
        scene_id=scene_id,
        frame_token="frame_1",
        path_trajectory=path,
        drivable_area_polygons=drivable_area_polygons,
    )


def _variant(results: list, variant_type: TrajectoryVariantType):
    return next(result for result in results if result.variant_type == variant_type)


def _make_curved_path(*, duration_s: float = 12.0, dt_s: float = 0.2) -> Trajectory:
    radius_m = 20.0
    count = round(duration_s / dt_s) + 1
    angles = np.linspace(0.0, math.pi / 2.0, count)
    waypoints = [
        Waypoint(
            x=float(radius_m * math.sin(angle)),
            y=float(radius_m * (1.0 - math.cos(angle))),
            t=float(index * dt_s),
        )
        for index, angle in enumerate(angles)
    ]
    return Trajectory(
        traj_id="curved_path",
        source=TrajectorySource.PLANNING,
        waypoints=waypoints,
    )


def _make_straight_gt_and_path() -> tuple[Trajectory, Trajectory]:
    path = _make_gt_trajectory(n_waypoints=40)
    ground_truth = path.model_copy(update={"waypoints": path.waypoints[:20]})
    return ground_truth, path


def _distance_to_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    p = np.array(point)
    a = np.array(start)
    b = np.array(end)
    direction = b - a
    denominator = float(np.dot(direction, direction))
    if denominator == 0.0:
        return float(np.linalg.norm(p - a))
    ratio = float(np.clip(np.dot(p - a, direction) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + ratio * direction)))


class TestAugmentBasic:
    def test_output_has_six_v31_variants(self) -> None:
        results = _augment()
        assert len(results) == 6
        assert {r.variant_type for r in results} == {
            TrajectoryVariantType.GROUND_TRUTH,
            TrajectoryVariantType.CONSERVATIVE,
            TrajectoryVariantType.SUBOPTIMAL,
            TrajectoryVariantType.AGGRESSIVE,
            TrajectoryVariantType.ILLEGAL,
            TrajectoryVariantType.HARD_CASE,
        }

    def test_ground_truth_is_unchanged_reference(self) -> None:
        gt = _make_gt_trajectory()
        result = _variant(_augment(gt=gt), TrajectoryVariantType.GROUND_TRUTH)
        assert result.trajectory is gt

    def test_verdicts_are_provisional_until_geometry_validation(self) -> None:
        results = _augment()
        assert all(result.expected_verdict == "exclude" for result in results)
        assert all(result.expected_verdict_reason == "待最终几何特征验证" for result in results)

    def test_synthetic_trajectories_use_synthetic_source_and_internal_ids(self) -> None:
        results = _augment(scene_id="my_scene")
        for result in results:
            if result.variant_type == TrajectoryVariantType.GROUND_TRUTH:
                assert result.trajectory.source == TrajectorySource.PLANNING
                continue
            assert result.trajectory.source == TrajectorySource.SYNTHETIC
            assert result.trajectory.traj_id.startswith("my_scene_synthetic_")
            assert result.variant_type.value in result.trajectory.traj_id

    def test_minimal_two_waypoints_still_works(self) -> None:
        gt = Trajectory(
            traj_id="minimal",
            source=TrajectorySource.PLANNING,
            waypoints=[
                Waypoint(x=0.0, y=0.0, t=0.0),
                Waypoint(x=2.0, y=0.0, t=0.5),
            ],
        )
        assert len(_augment(gt=gt)) == 6


class TestIllegalPerturbations:
    def test_supported_fact_does_not_bypass_geometry_validation(self) -> None:
        gt = _make_gt_trajectory(with_deceleration=True)
        results = TrajectoryAugmentor(seed=42).augment(
            gt,
            ScenarioType.RED_LIGHT,
            SceneFacts(has_red_light=True),
            scene_id="s1",
            frame_token="frame_1",
        )
        assert _variant(results, TrajectoryVariantType.ILLEGAL).expected_verdict == "exclude"

    def test_red_light_uses_faster_speed_profile(self) -> None:
        gt, path = _make_straight_gt_and_path()
        illegal = _variant(
            _augment(ScenarioType.RED_LIGHT, gt=gt, path=path),
            TrajectoryVariantType.ILLEGAL,
        )

        assert illegal.trajectory.waypoints[-1].x > gt.waypoints[-1].x

    def test_red_light_without_deceleration_is_not_raw_gt(self) -> None:
        gt, path = _make_straight_gt_and_path()
        illegal = _variant(
            _augment(ScenarioType.RED_LIGHT, gt=gt, path=path),
            TrajectoryVariantType.ILLEGAL,
        )

        assert illegal.trajectory.waypoints[-1].x > gt.waypoints[-1].x

    def test_yellow_light_compresses_time_without_scaling_geometry(self) -> None:
        gt, path = _make_straight_gt_and_path()
        illegal = _variant(
            _augment(ScenarioType.YELLOW_LIGHT, gt=gt, path=path),
            TrajectoryVariantType.ILLEGAL,
        )

        assert illegal.trajectory.waypoints[-1].x > gt.waypoints[-1].x
        assert illegal.trajectory.waypoints[-1].t <= gt.waypoints[-1].t

    @pytest.mark.parametrize(
        "scenario_type",
        [
            ScenarioType.PEDESTRIAN,
            ScenarioType.ONCOMING,
            ScenarioType.MINOR_ROAD,
            ScenarioType.EMERGENCY,
        ],
    )
    def test_yield_scenarios_remove_deceleration(self, scenario_type: ScenarioType) -> None:
        gt, path = _make_straight_gt_and_path()
        illegal = _variant(
            _augment(scenario_type, gt=gt, path=path),
            TrajectoryVariantType.ILLEGAL,
        )

        assert illegal.trajectory.waypoints[-1].x > gt.waypoints[-1].x

    def test_unknown_uses_generic_aggressive_speed_profile(self) -> None:
        gt, path = _make_straight_gt_and_path()
        illegal = _variant(
            _augment(ScenarioType.UNKNOWN, gt=gt, path=path),
            TrajectoryVariantType.ILLEGAL,
        )

        assert illegal.trajectory.waypoints[-1].x > gt.waypoints[-1].x


class TestPathSpeedDecoupling:
    def test_all_variants_remain_on_ground_truth_spatial_path(self) -> None:
        path = _make_curved_path()
        gt = path.model_copy(update={"waypoints": path.waypoints[:31]})
        results = _augment(gt=gt, path=path)
        path_points = [(waypoint.x, waypoint.y) for waypoint in path.waypoints]

        for result in results:
            for waypoint in result.trajectory.waypoints:
                distance = min(
                    _distance_to_segment(
                        (waypoint.x, waypoint.y),
                        start,
                        end,
                    )
                    for start, end in zip(path_points, path_points[1:], strict=False)
                )
                assert distance <= 1e-6

    def test_style_variants_only_change_progress_along_path(self) -> None:
        gt, path = _make_straight_gt_and_path()
        results = _augment(gt=gt, path=path)

        conservative = _variant(results, TrajectoryVariantType.CONSERVATIVE)
        suboptimal = _variant(results, TrajectoryVariantType.SUBOPTIMAL)
        aggressive = _variant(results, TrajectoryVariantType.AGGRESSIVE)
        hard = _variant(results, TrajectoryVariantType.HARD_CASE)

        assert conservative.trajectory.waypoints[-1].x < gt.waypoints[-1].x
        assert aggressive.trajectory.waypoints[-1].x > gt.waypoints[-1].x
        assert suboptimal.trajectory.waypoints[-1].x < gt.waypoints[-1].x
        assert hard.expected_verdict == "exclude"
        assert hard.trajectory.waypoints[-1].x < gt.waypoints[-1].x

    def test_red_conservative_stops_before_governing_line(self) -> None:
        gt, path = _make_straight_gt_and_path()
        stop_line = ConflictZone(
            kind="stop_line",
            polygon_xy=[(5.0, -2.0), (6.0, -2.0), (6.0, 2.0), (5.0, 2.0)],
            distance_from_ego_m=5.0,
            source_layer="stop_line",
        )
        facts = SceneFacts(
            has_red_light=True,
            traffic_light_status_source="drivelm_status",
            governing_stop_line_signed_m=5.5,
            conflict_zones=[stop_line],
        )
        results = TrajectoryAugmentor(seed=42).augment(
            gt,
            ScenarioType.RED_LIGHT,
            facts,
            scene_id="red_stop",
            path_trajectory=path,
        )
        candidate = _variant(results, TrajectoryVariantType.CONSERVATIVE)
        features = TrajectoryAnalyzer().analyze(
            candidate.trajectory,
            SceneDescription(),
            facts,
        )

        assert candidate.trajectory.waypoints[-1].x == pytest.approx(4.0)
        assert features.full_stop is True
        stop_metric = next(
            metric for metric in features.conflict_zone_metrics if metric.kind == "stop_line"
        )
        assert stop_metric.entered is False
        assert stop_metric.stopped_before_zone is True

    @pytest.mark.parametrize(
        ("frame_token", "entry_s_m"),
        [
            ("ce63dd8c7f6a45f9bdee5b7305388314", 5.386228415202785),
            ("5292f0009e4644acb946f0a9eaae4e37", 1.1487893680292038),
            ("a6176dbfbeb44e11a42282a4c4fbeb45", 4.241575943196324),
        ],
    )
    def test_gate3_false_veto_regressions_stop_before_signed_zero(
        self,
        frame_token: str,
        entry_s_m: float,
    ) -> None:
        path = Trajectory(
            traj_id=f"{frame_token}_path",
            source=TrajectorySource.PLANNING,
            waypoints=[
                Waypoint(x=float(index) * 0.25, y=0.0, t=float(index) * 0.05)
                for index in range(81)
            ],
        )
        gt = path.model_copy(update={"waypoints": path.waypoints[:61]})
        facts = SceneFacts(
            has_red_light=True,
            traffic_light_status_source="drivelm_status",
            governing_stop_line_signed_m=entry_s_m + 0.5,
            conflict_zones=[ConflictZone(
                kind="stop_line",
                polygon_xy=[
                    (entry_s_m, -2.0),
                    (entry_s_m + 1.0, -2.0),
                    (entry_s_m + 1.0, 2.0),
                    (entry_s_m, 2.0),
                ],
                distance_from_ego_m=entry_s_m,
                source_layer="stop_line",
            )],
        )
        candidate = _variant(
            TrajectoryAugmentor(seed=42).augment(
                gt,
                ScenarioType.RED_LIGHT,
                facts,
                scene_id=frame_token,
                path_trajectory=path,
            ),
            TrajectoryVariantType.CONSERVATIVE,
        )
        features = TrajectoryAnalyzer().analyze(
            candidate.trajectory,
            SceneDescription(),
            facts,
        )
        metric = next(
            item for item in features.conflict_zone_metrics if item.kind == "stop_line"
        )

        assert candidate.trajectory.waypoints[-1].x == pytest.approx(entry_s_m - 1.0)
        assert metric.entered is False
        assert metric.stopped_before_zone is True

    def test_governing_line_signed_path_distance_crosses_zero_at_entry(self) -> None:
        augmentor = TrajectoryAugmentor(seed=42)
        path_xy = np.asarray([[0.0, 0.0], [10.0, 0.0]])
        facts = SceneFacts(
            governing_stop_line_signed_m=5.0,
            conflict_zones=[ConflictZone(
            kind="stop_line",
            polygon_xy=[(5.0, -2.0), (6.0, -2.0), (6.0, 2.0), (5.0, 2.0)],
            distance_from_ego_m=5.0,
            source_layer="stop_line",
            )],
        )

        stop_target_s = augmentor._governing_stop_path_s(path_xy, facts)
        assert stop_target_s is not None
        entry_s = stop_target_s + augmentor._STOP_LINE_BUFFER_M

        assert entry_s == pytest.approx(5.0)
        assert entry_s - 4.9 > 0.0
        assert entry_s - 5.1 < 0.0

    def test_curvature_speed_cap_limits_lateral_acceleration(self) -> None:
        path = _make_curved_path()
        gt = path.model_copy(update={"waypoints": path.waypoints[:31]})
        limit = 3.5
        results = TrajectoryAugmentor(seed=42, a_lat_max_mps2=limit).augment(
            gt,
            ScenarioType.RED_LIGHT,
            SceneFacts(),
            scene_id="curve",
            path_trajectory=path,
        )

        for result in results:
            if result.variant_type == TrajectoryVariantType.GROUND_TRUTH:
                continue
            waypoints = result.trajectory.waypoints
            for previous, current, following in zip(
                waypoints,
                waypoints[1:],
                waypoints[2:],
                strict=False,
            ):
                a = math.hypot(current.x - previous.x, current.y - previous.y)
                b = math.hypot(following.x - current.x, following.y - current.y)
                c = math.hypot(following.x - previous.x, following.y - previous.y)
                denominator = a * b * c
                if denominator <= 1e-9:
                    continue
                cross = abs(
                    (current.x - previous.x) * (following.y - previous.y)
                    - (current.y - previous.y) * (following.x - previous.x)
                )
                curvature = 2.0 * cross / denominator
                duration = following.t - previous.t
                speed = (a + b) / duration
                assert speed * speed * curvature <= limit + 0.15

    def test_path_exhaustion_truncates_faster_variant(self) -> None:
        gt = _make_gt_trajectory()
        aggressive = _variant(_augment(gt=gt), TrajectoryVariantType.AGGRESSIVE)

        assert aggressive.trajectory.waypoints[-1].t < gt.waypoints[-1].t
        assert aggressive.trajectory.waypoints[-1].x == pytest.approx(
            gt.waypoints[-1].x
        )


class TestDrivableAreaValidation:
    def test_out_of_bounds_candidate_is_rejected(self) -> None:
        gt, path = _make_straight_gt_and_path()
        drivable = [[(-1.0, -2.0), (23.0, -2.0), (23.0, 2.0), (-1.0, 2.0)]]

        with pytest.raises(TrajectoryAugmentationError, match="drivable_area"):
            _augment(gt=gt, path=path, drivable_area_polygons=drivable)

    def test_candidates_inside_drivable_area_are_accepted(self) -> None:
        gt, path = _make_straight_gt_and_path()
        drivable = [[(-1.0, -2.0), (50.0, -2.0), (50.0, 2.0), (-1.0, 2.0)]]

        assert len(_augment(gt=gt, path=path, drivable_area_polygons=drivable)) == 6


class TestReproducibility:
    def test_same_seed_same_output(self) -> None:
        gt = _make_gt_trajectory()
        first = TrajectoryAugmentor(seed=42).augment(
            gt,
            ScenarioType.RED_LIGHT,
            SceneFacts(),
            scene_id="s1",
            frame_token="f1",
        )
        second = TrajectoryAugmentor(seed=42).augment(
            gt,
            ScenarioType.RED_LIGHT,
            SceneFacts(),
            scene_id="s1",
            frame_token="f1",
        )

        assert [r.model_dump() for r in first] == [r.model_dump() for r in second]

    def test_constructor_seed_is_stored(self) -> None:
        assert TrajectoryAugmentor().seed == 42
        assert TrajectoryAugmentor(seed=123).seed == 123
        assert TrajectoryAugmentor().pedestrian_illegal_gap_target_s == -0.3
