"""TrajectoryExtractor tests for decoupled trajectory and path windows."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from src.layer1.exceptions import TrajectoryExtractionError
from src.layer1.models import RawScene, TrajectorySource
from src.layer1.trajectory_extractor import TrajectoryExtractor, _quat_to_yaw, _rot2d


def _pose(
    *,
    token: str,
    timestamp: int,
    translation: list[float],
    rotation: list[float] | None = None,
) -> dict:
    return {
        "token": token,
        "timestamp": timestamp,
        "translation": translation,
        "rotation": rotation or [1.0, 0.0, 0.0, 0.0],
    }


def _make_raw_scene(
    poses: list[dict],
    *,
    keyframe_index: int = 0,
    scene_id: str = "test_scene",
) -> RawScene:
    return RawScene(
        scene_id=scene_id,
        frame_token="test_frame",
        source_file=Path("/fake/nuscenes"),
        ego_poses=poses,
        keyframe_index=keyframe_index,
    )


def _make_dense_poses(n: int, *, dt_us: int = 50_000, speed_mps: float = 5.0) -> list[dict]:
    dt_s = dt_us / 1_000_000.0
    return [
        _pose(
            token=f"pose_{i}",
            timestamp=i * dt_us,
            translation=[float(i) * speed_mps * dt_s, 0.0, 0.0],
        )
        for i in range(n)
    ]


class TestRotationHelpers:
    def test_quaternion_to_yaw(self) -> None:
        sqrt2_2 = math.sqrt(2.0) / 2.0
        assert _quat_to_yaw([1.0, 0.0, 0.0, 0.0]) == pytest.approx(0.0)
        assert _quat_to_yaw([sqrt2_2, 0.0, 0.0, sqrt2_2]) == pytest.approx(math.pi / 2)

    def test_rot2d(self) -> None:
        rotation = _rot2d(math.pi / 2)
        assert rotation @ np.array([1.0, 0.0]) == pytest.approx(np.array([0.0, 1.0]))


class TestTrajectoryExtractor:
    def test_default_window_and_min_waypoints_follow_v31(self) -> None:
        extractor = TrajectoryExtractor()
        assert extractor.future_horizon_s == 6.0
        assert extractor.path_horizon_s == 12.0
        assert extractor.min_waypoints == 2

    def test_extract_uses_six_second_dense_window(self) -> None:
        poses = _make_dense_poses(150, dt_us=50_000)
        traj = TrajectoryExtractor().extract(_make_raw_scene(poses))

        assert traj.source == TrajectorySource.PLANNING
        assert traj.traj_id == "test_scene_planning_gt_0"
        assert len(traj.waypoints) == 121
        assert traj.waypoints[0].t == pytest.approx(0.0)
        assert traj.waypoints[-1].t == pytest.approx(6.0)

    def test_extract_path_uses_twelve_second_geometry_window(self) -> None:
        poses = _make_dense_poses(260, dt_us=50_000)
        extractor = TrajectoryExtractor()

        raw = _make_raw_scene(poses)
        trajectory = extractor.extract(raw)
        path = extractor.extract_path(raw)

        assert len(path.waypoints) == 200
        assert path.waypoints[-1].t == pytest.approx(12.0)
        path_prefix = [
            (waypoint.x, waypoint.y, waypoint.t)
            for waypoint in path.waypoints[:len(trajectory.waypoints)]
        ]
        trajectory_values = [
            (waypoint.x, waypoint.y, waypoint.t)
            for waypoint in trajectory.waypoints
        ]
        assert path_prefix == trajectory_values

    def test_extract_path_accepts_exhausted_source_without_extrapolation(self) -> None:
        poses = _make_dense_poses(150, dt_us=50_000)

        path = TrajectoryExtractor().extract_path(_make_raw_scene(poses))

        assert len(path.waypoints) == 150
        assert path.waypoints[-1].t == pytest.approx(7.45)

    def test_drivable_area_polygons_are_transformed_to_local_frame(self) -> None:
        poses = [
            _pose(token="p0", timestamp=0, translation=[10.0, 20.0, 0.0]),
            _pose(token="p1", timestamp=500_000, translation=[11.0, 20.0, 0.0]),
        ]
        raw = _make_raw_scene(poses).model_copy(update={
            "map_records": {
                "drivable_area": [{
                    "polygon_xy_list": [[
                        (9.0, 19.0),
                        (12.0, 19.0),
                        (12.0, 21.0),
                        (9.0, 21.0),
                    ]],
                }],
            },
        })

        polygons = TrajectoryExtractor().local_drivable_polygons(raw)

        assert polygons == [[(-1.0, -1.0), (2.0, -1.0), (2.0, 1.0), (-1.0, 1.0)]]

    def test_local_frame_transform_straight(self) -> None:
        poses = [
            _pose(token="p0", timestamp=0, translation=[10.0, 20.0, 0.0]),
            _pose(token="p1", timestamp=500_000, translation=[15.0, 20.0, 0.0]),
            _pose(token="p2", timestamp=1_000_000, translation=[20.0, 20.0, 0.0]),
        ]
        traj = TrajectoryExtractor(min_waypoints=2).extract(_make_raw_scene(poses))

        assert [wp.x for wp in traj.waypoints] == pytest.approx([0.0, 5.0, 10.0])
        assert [wp.y for wp in traj.waypoints] == pytest.approx([0.0, 0.0, 0.0])
        assert [wp.t for wp in traj.waypoints] == pytest.approx([0.0, 0.5, 1.0])

    def test_local_frame_uses_keyframe_yaw(self) -> None:
        sqrt2_2 = math.sqrt(2.0) / 2.0
        yaw_90 = [sqrt2_2, 0.0, 0.0, sqrt2_2]
        poses = [
            _pose(token="p0", timestamp=0, translation=[0.0, 0.0, 0.0], rotation=yaw_90),
            _pose(token="p1", timestamp=500_000, translation=[0.0, 5.0, 0.0], rotation=yaw_90),
            _pose(token="p2", timestamp=1_000_000, translation=[0.0, 10.0, 0.0], rotation=yaw_90),
        ]

        traj = TrajectoryExtractor(min_waypoints=2).extract(_make_raw_scene(poses))

        assert traj.waypoints[-1].x == pytest.approx(10.0)
        assert traj.waypoints[-1].y == pytest.approx(0.0, abs=1e-6)

    def test_keyframe_index_starts_new_local_origin(self) -> None:
        poses = _make_dense_poses(10, dt_us=500_000, speed_mps=4.0)
        traj = TrajectoryExtractor(future_horizon_s=1.0, min_waypoints=2).extract(
            _make_raw_scene(poses, keyframe_index=2),
        )

        assert len(traj.waypoints) == 3
        assert traj.waypoints[0].x == pytest.approx(0.0)
        assert traj.waypoints[0].t == pytest.approx(0.0)
        assert traj.waypoints[-1].t == pytest.approx(1.0)

    def test_path_waypoint_limit_is_200_after_uniform_downsampling(self) -> None:
        poses = _make_dense_poses(500, dt_us=10_000, speed_mps=2.0)
        traj = TrajectoryExtractor(path_horizon_s=10.0, min_waypoints=2).extract_path(
            _make_raw_scene(poses),
        )

        assert len(traj.waypoints) == 200
        assert traj.waypoints[-1].t == pytest.approx(3.99)

    def test_short_window_is_emitted_for_downstream_exclusion(self) -> None:
        poses = _make_dense_poses(5)
        trajectory = TrajectoryExtractor().extract(_make_raw_scene(poses))
        assert len(trajectory.waypoints) == 5

    def test_empty_or_bad_keyframe_raises(self) -> None:
        with pytest.raises(TrajectoryExtractionError):
            TrajectoryExtractor().extract(_make_raw_scene([]))
        with pytest.raises(TrajectoryExtractionError):
            TrajectoryExtractor(min_waypoints=2).extract(
                _make_raw_scene(_make_dense_poses(3), keyframe_index=99),
            )

    def test_constructor_rejects_invalid_values(self) -> None:
        with pytest.raises(ValueError):
            TrajectoryExtractor(future_horizon_s=0.0)
        with pytest.raises(ValueError):
            TrajectoryExtractor(path_horizon_s=5.0)
        with pytest.raises(ValueError):
            TrajectoryExtractor(min_waypoints=1)
