"""沿真实空间路径合成仅速度不同的候选轨迹。"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from src.layer1.exceptions import TrajectoryAugmentationError
from src.layer1.models import (
    AugmentedTrajectory,
    ConflictZone,
    PathTrajectory,
    ScenarioType,
    SceneFacts,
    Trajectory,
    TrajectorySource,
    TrajectoryVariantType,
    Waypoint,
)

FloatArray = NDArray[np.float64]
SpeedProfile = Literal["constant", "hesitant", "creeping"]


class TrajectoryAugmentor:
    """在同一条真实空间路径上修改速度剖面并重采样候选。"""

    _STOP_LINE_BUFFER_M = 1.0

    def __init__(
        self,
        seed: int = 42,
        *,
        a_lat_max_mps2: float = 3.5,
        output_horizon_s: float = 6.0,
        pedestrian_illegal_gap_target_s: float = -0.3,
    ) -> None:
        if not 3.0 <= a_lat_max_mps2 <= 4.0:
            raise ValueError("a_lat_max_mps2 must be between 3.0 and 4.0")
        if output_horizon_s <= 0.0:
            raise ValueError("output_horizon_s must be positive")
        self.seed = seed
        self.a_lat_max_mps2 = a_lat_max_mps2
        self.output_horizon_s = output_horizon_s
        self.pedestrian_illegal_gap_target_s = pedestrian_illegal_gap_target_s

    def augment(
        self,
        ground_truth: Trajectory,
        scenario_type: ScenarioType,
        scene_facts: SceneFacts,
        *,
        scene_id: str,
        frame_token: str = "",
        path_trajectory: PathTrajectory | Trajectory | None = None,
        drivable_area_polygons: list[list[tuple[float, float]]] | None = None,
        pedestrian_illegal_speed_scale: float | None = None,
    ) -> list[AugmentedTrajectory]:
        """生成 GT 与五条共享真实空间路径的速度剖面变体。"""
        if len(ground_truth.waypoints) < 2:
            raise TrajectoryAugmentationError(
                f"原始轨迹 waypoint 不足: {len(ground_truth.waypoints)} < 2"
            )
        if (
            pedestrian_illegal_speed_scale is not None
            and pedestrian_illegal_speed_scale <= 0.0
        ):
            raise ValueError("pedestrian_illegal_speed_scale must be positive")
        path = path_trajectory or ground_truth
        if len(path.waypoints) < 2:
            raise TrajectoryAugmentationError("路径素材 waypoint 不足")
        _ = frame_token

        path_pts, path_times = self._path_arrays(path)
        output_times = np.array(
            [
                waypoint.t
                for waypoint in ground_truth.waypoints
                if waypoint.t <= self.output_horizon_s + 1e-9
            ],
            dtype=np.float64,
        )
        if len(output_times) < 2:
            raise TrajectoryAugmentationError("6 秒轨迹窗口内 waypoint 不足")

        results = [self._make_augmented(
            trajectory=ground_truth,
            variant_type=TrajectoryVariantType.GROUND_TRUTH,
            expected_verdict="exclude",
            note="真实轨迹，由 ego_pose 重建",
        )]
        specifications: tuple[
            tuple[TrajectoryVariantType, float, float, SpeedProfile, str],
            ...,
        ] = (
            (
                TrajectoryVariantType.CONSERVATIVE,
                0.60,
                0.0,
                "constant",
                "保守风格：同一路径使用 0.60x 速度剖面",
            ),
            (
                TrajectoryVariantType.SUBOPTIMAL,
                0.90,
                0.0,
                "hesitant",
                "次优风格：同一路径使用轻微波动的速度剖面",
            ),
            (
                TrajectoryVariantType.AGGRESSIVE,
                1.30,
                0.0,
                "constant",
                "激进风格：同一路径使用 1.30x 速度剖面",
            ),
            (
                TrajectoryVariantType.HARD_CASE,
                1.0,
                0.0,
                "creeping",
                "困难样本：同一路径中段临界缓行",
            ),
        )
        if scenario_type == ScenarioType.RED_LIGHT:
            conservative_stop_s = self._governing_stop_path_s(path_pts, scene_facts)
        elif (
            scenario_type == ScenarioType.PEDESTRIAN
            and scene_facts.pedestrian_in_forward_crosswalk
        ):
            conservative_stop_s = self._pedestrian_stop_path_s(path_pts, scene_facts)
        else:
            conservative_stop_s = None
        for variant, scale, floor_ratio, profile, note in specifications:
            trajectory = self._synthesize(
                scene_id,
                variant,
                path_pts,
                path_times,
                output_times,
                speed_scale=scale,
                speed_floor_ratio=floor_ratio,
                profile=profile,
                stop_at_s=(
                    conservative_stop_s
                    if variant == TrajectoryVariantType.CONSERVATIVE
                    else None
                ),
            )
            results.append(self._make_augmented(
                trajectory=trajectory,
                variant_type=variant,
                expected_verdict="exclude",
                note=note,
            ))

        illegal_scale, illegal_floor = self._illegal_speed_parameters(
            scenario_type,
            scene_facts,
        )
        if (
            scenario_type == ScenarioType.PEDESTRIAN
            and scene_facts.pedestrian_in_forward_crosswalk
            and pedestrian_illegal_speed_scale is not None
        ):
            illegal_scale = pedestrian_illegal_speed_scale
            # 同比例抬起 GT 的零速段：慢行仍可小于 1x，但不能继承完整停车。
            illegal_floor = pedestrian_illegal_speed_scale
        illegal = self._synthesize(
            scene_id,
            TrajectoryVariantType.ILLEGAL,
            path_pts,
            path_times,
            output_times,
            speed_scale=illegal_scale,
            speed_floor_ratio=illegal_floor,
            profile="constant",
        )
        if (
            pedestrian_illegal_speed_scale is None
            and self._is_degenerate_trajectory(illegal, ground_truth)
        ):
            illegal = self._synthesize(
                scene_id,
                TrajectoryVariantType.ILLEGAL,
                path_pts,
                path_times,
                output_times,
                speed_scale=illegal_scale * 1.25,
                speed_floor_ratio=max(illegal_floor, 1.0),
                profile="constant",
            )
        is_degenerate = self._is_degenerate_trajectory(illegal, ground_truth)
        note = (
            "no_violation_injectable：速度剖面重试后仍与 GT 退化"
            if is_degenerate
            else self._illegal_note(scenario_type, scene_facts)
        )
        results.append(self._make_augmented(
            trajectory=illegal,
            variant_type=TrajectoryVariantType.ILLEGAL,
            expected_verdict="exclude",
            note=note,
        ))

        if drivable_area_polygons:
            self._validate_drivable_area(results, drivable_area_polygons)
        return results

    def _synthesize(
        self,
        scene_id: str,
        variant: TrajectoryVariantType,
        path_pts: FloatArray,
        path_times: FloatArray,
        output_times: FloatArray,
        *,
        speed_scale: float,
        speed_floor_ratio: float,
        profile: SpeedProfile,
        stop_at_s: float | None = None,
    ) -> Trajectory:
        pts, times = self._resample_speed_profile(
            path_pts,
            path_times,
            output_times,
            speed_scale=speed_scale,
            speed_floor_ratio=speed_floor_ratio,
            profile=profile,
            stop_at_s=stop_at_s,
        )
        return self._build_trajectory(scene_id, variant, 0, pts, times)

    def _resample_speed_profile(
        self,
        path_pts: FloatArray,
        path_times: FloatArray,
        output_times: FloatArray,
        *,
        speed_scale: float,
        speed_floor_ratio: float,
        profile: SpeedProfile,
        stop_at_s: float | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        segment_lengths = np.linalg.norm(np.diff(path_pts, axis=0), axis=1)
        keep = np.concatenate((np.array([True]), segment_lengths > 1e-9))
        pts = path_pts[keep]
        times = path_times[keep]
        if len(pts) < 2:
            repeated = np.repeat(path_pts[:1], len(output_times), axis=0)
            return repeated, output_times.copy()

        segment_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        path_s = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        stop_s = (
            float(np.clip(stop_at_s, 0.0, path_s[-1]))
            if stop_at_s is not None
            else None
        )
        dt = np.diff(times)
        base_speeds = np.divide(
            segment_lengths,
            dt,
            out=np.zeros_like(segment_lengths),
            where=dt > 0.0,
        )
        positive_speeds = base_speeds[base_speeds > 1e-6]
        cruise_speed = (
            float(np.median(positive_speeds)) if len(positive_speeds) else 0.0
        )
        curvatures = self._path_curvatures(pts)

        sampled_s = [0.0]
        sampled_times = [float(output_times[0])]
        current_s = 0.0
        for start_t, end_t in zip(output_times, output_times[1:], strict=False):
            duration = float(end_t - start_t)
            if duration <= 0.0:
                continue
            if stop_s is not None and current_s >= stop_s - 1e-9:
                remaining_times = output_times[output_times > sampled_times[-1] + 1e-9]
                sampled_s.extend([stop_s] * len(remaining_times))
                sampled_times.extend(float(time) for time in remaining_times)
                break
            base_speed = float(np.interp(current_s, path_s[:-1], base_speeds))
            progress = current_s / path_s[-1]
            requested_speed = max(
                base_speed * speed_scale * self._profile_multiplier(profile, progress),
                cruise_speed * speed_floor_ratio,
            )
            curvature = float(np.interp(current_s, path_s, curvatures))
            curvature_limit = (
                math.sqrt(self.a_lat_max_mps2 / curvature) * 0.98
                if curvature > 1e-9
                else math.inf
            )
            speed = min(requested_speed, curvature_limit)
            next_s = current_s if speed <= 1e-9 else current_s + speed * duration
            if stop_s is not None and next_s >= stop_s:
                remaining = stop_s - current_s
                arrival_t = float(start_t + (remaining / speed if speed > 0.0 else duration))
                sampled_s.append(stop_s)
                sampled_times.append(arrival_t)
                remaining_times = output_times[output_times > arrival_t + 1e-9]
                sampled_s.extend([stop_s] * len(remaining_times))
                sampled_times.extend(float(time) for time in remaining_times)
                break
            if next_s >= path_s[-1]:
                remaining = path_s[-1] - current_s
                arrival_t = float(start_t + (remaining / speed if speed > 0.0 else duration))
                sampled_s.append(float(path_s[-1]))
                sampled_times.append(arrival_t)
                break
            current_s = next_s
            sampled_s.append(current_s)
            sampled_times.append(float(end_t))

        sampled = np.asarray(sampled_s, dtype=np.float64)
        coordinates = np.column_stack((
            np.interp(sampled, path_s, pts[:, 0]),
            np.interp(sampled, path_s, pts[:, 1]),
        ))
        return coordinates, np.asarray(sampled_times, dtype=np.float64)

    @classmethod
    def _governing_stop_path_s(
        cls,
        path_pts: FloatArray,
        scene_facts: SceneFacts,
    ) -> float | None:
        signed = scene_facts.governing_stop_line_signed_m
        if signed is None or signed < 0.0:
            return None
        zones = [
            zone
            for zone in scene_facts.conflict_zones
            if zone.kind == "stop_line"
            and len(zone.polygon_xy) >= 3
            and abs(
                sum(point[0] for point in zone.polygon_xy) / len(zone.polygon_xy)
                - signed
            ) < 2.0
        ]
        if not zones:
            return None

        return cls._first_zone_stop_path_s(path_pts, zones)

    @classmethod
    def _pedestrian_stop_path_s(
        cls,
        path_pts: FloatArray,
        scene_facts: SceneFacts,
    ) -> float | None:
        zones = [
            zone
            for zone in scene_facts.conflict_zones
            if zone.kind == "ped_crossing" and len(zone.polygon_xy) >= 3
        ]
        return cls._first_zone_stop_path_s(path_pts, zones)

    @classmethod
    def _first_zone_stop_path_s(
        cls,
        path_pts: FloatArray,
        zones: list[ConflictZone],
    ) -> float | None:
        cumulative_s = 0.0
        for start, end in zip(path_pts, path_pts[1:], strict=False):
            start_xy = (float(start[0]), float(start[1]))
            end_xy = (float(end[0]), float(end[1]))
            segment_length = float(np.linalg.norm(end - start))
            fractions: list[float] = []
            for zone in zones:
                if cls._point_in_polygon(start_xy, zone.polygon_xy):
                    fractions.append(0.0)
                for edge_start, edge_end in zip(
                    zone.polygon_xy,
                    zone.polygon_xy[1:] + zone.polygon_xy[:1],
                    strict=False,
                ):
                    fraction = cls._segment_intersection_fraction(
                        start_xy,
                        end_xy,
                        edge_start,
                        edge_end,
                    )
                    if fraction is not None:
                        fractions.append(fraction)
            if fractions:
                entry_s = cumulative_s + min(fractions) * segment_length
                return max(0.0, entry_s - cls._STOP_LINE_BUFFER_M)
            cumulative_s += segment_length
        return None

    @staticmethod
    def _segment_intersection_fraction(
        start: tuple[float, float],
        end: tuple[float, float],
        edge_start: tuple[float, float],
        edge_end: tuple[float, float],
    ) -> float | None:
        path_dx = end[0] - start[0]
        path_dy = end[1] - start[1]
        edge_dx = edge_end[0] - edge_start[0]
        edge_dy = edge_end[1] - edge_start[1]
        denominator = path_dx * edge_dy - path_dy * edge_dx
        if abs(denominator) <= 1e-12:
            return None
        offset_x = edge_start[0] - start[0]
        offset_y = edge_start[1] - start[1]
        path_fraction = (offset_x * edge_dy - offset_y * edge_dx) / denominator
        edge_fraction = (offset_x * path_dy - offset_y * path_dx) / denominator
        if -1e-9 <= path_fraction <= 1.0 + 1e-9 and -1e-9 <= edge_fraction <= 1.0 + 1e-9:
            return float(np.clip(path_fraction, 0.0, 1.0))
        return None

    @staticmethod
    def _profile_multiplier(
        profile: SpeedProfile,
        progress: float,
    ) -> float:
        if profile == "hesitant":
            return 0.95 + 0.10 * math.sin(2.0 * math.pi * progress)
        if profile == "creeping":
            if progress < 0.40:
                return 0.75
            if progress < 0.70:
                return 0.25
            return 0.65
        return 1.0

    @staticmethod
    def _path_arrays(
        trajectory: PathTrajectory | Trajectory,
    ) -> tuple[FloatArray, FloatArray]:
        return (
            np.array([[waypoint.x, waypoint.y] for waypoint in trajectory.waypoints]),
            np.array([waypoint.t for waypoint in trajectory.waypoints]),
        )

    @staticmethod
    def _path_curvatures(pts: FloatArray) -> FloatArray:
        curvatures = np.zeros(len(pts), dtype=np.float64)
        for index in range(1, len(pts) - 1):
            previous, current, following = pts[index - 1:index + 2]
            a = float(np.linalg.norm(current - previous))
            b = float(np.linalg.norm(following - current))
            c = float(np.linalg.norm(following - previous))
            denominator = a * b * c
            if denominator <= 1e-12:
                continue
            cross = abs(float(np.cross(current - previous, following - previous)))
            curvatures[index] = 2.0 * cross / denominator
        if len(curvatures) > 1:
            curvatures[0] = curvatures[1]
            curvatures[-1] = curvatures[-2]
        return curvatures

    @staticmethod
    def _illegal_speed_parameters(
        scenario_type: ScenarioType,
        scene_facts: SceneFacts,
    ) -> tuple[float, float]:
        if scenario_type == ScenarioType.RED_LIGHT:
            return (3.0, 1.0)
        if scenario_type == ScenarioType.YELLOW_LIGHT:
            return (1.0 / 0.7, 0.0)
        if scenario_type == ScenarioType.PEDESTRIAN:
            return (2.5, 1.0) if scene_facts.pedestrian_in_forward_crosswalk else (1.3, 0.0)
        if scenario_type == ScenarioType.ONCOMING:
            return (2.0, 1.0) if scene_facts.ego_turn_intent == "left" else (1.3, 0.0)
        if scenario_type in {
            ScenarioType.MINOR_ROAD,
            ScenarioType.EMERGENCY,
        }:
            return (2.0, 1.0)
        return (1.3, 0.0)

    @classmethod
    def _is_degenerate_trajectory(
        cls,
        variant: Trajectory,
        ground_truth: Trajectory,
    ) -> bool:
        variant_pts, variant_times = cls._path_arrays(variant)
        gt_pts, gt_times = cls._path_arrays(ground_truth)
        variant_speeds = cls._valid_speeds(variant_pts, variant_times)
        gt_speeds = cls._valid_speeds(gt_pts, gt_times)
        if len(variant_speeds) == 0 or len(gt_speeds) == 0:
            return True
        variant_distance = float(np.linalg.norm(np.diff(variant_pts, axis=0), axis=1).sum())
        gt_distance = float(np.linalg.norm(np.diff(gt_pts, axis=0), axis=1).sum())
        return (
            abs(float(np.mean(variant_speeds)) - float(np.mean(gt_speeds))) < 0.5
            and abs(float(np.min(variant_speeds)) - float(np.min(gt_speeds))) < 0.5
            and abs(variant_distance - gt_distance) < 2.0
        )

    @staticmethod
    def _valid_speeds(pts: FloatArray, times: FloatArray) -> FloatArray:
        dt = np.diff(times)
        distances = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        return np.asarray(distances[dt > 0] / dt[dt > 0], dtype=np.float64)

    @classmethod
    def _validate_drivable_area(
        cls,
        candidates: list[AugmentedTrajectory],
        polygons: list[list[tuple[float, float]]],
    ) -> None:
        for candidate in candidates:
            for waypoint in candidate.trajectory.waypoints:
                if not any(
                    cls._point_in_polygon((waypoint.x, waypoint.y), polygon)
                    for polygon in polygons
                    if len(polygon) >= 3
                ):
                    raise TrajectoryAugmentationError(
                        "候选轨迹超出 drivable_area: "
                        f"{candidate.variant_type.value} at ({waypoint.x:.2f}, {waypoint.y:.2f})"
                    )

    @classmethod
    def _point_in_polygon(
        cls,
        point: tuple[float, float],
        polygon: list[tuple[float, float]],
    ) -> bool:
        if any(
            cls._point_segment_distance(point, polygon[index - 1], polygon[index])
            <= 1e-8
            for index in range(len(polygon))
        ):
            return True
        x, y = point
        inside = False
        previous = polygon[-1]
        for current in polygon:
            if (current[1] > y) != (previous[1] > y):
                crossing_x = (
                    (previous[0] - current[0]) * (y - current[1])
                    / (previous[1] - current[1])
                    + current[0]
                )
                if x < crossing_x:
                    inside = not inside
            previous = current
        return inside

    @staticmethod
    def _point_segment_distance(
        point: tuple[float, float],
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> float:
        direction_x = end[0] - start[0]
        direction_y = end[1] - start[1]
        denominator = direction_x * direction_x + direction_y * direction_y
        if denominator == 0.0:
            return math.hypot(point[0] - start[0], point[1] - start[1])
        ratio = (
            (point[0] - start[0]) * direction_x
            + (point[1] - start[1]) * direction_y
        ) / denominator
        ratio = min(1.0, max(0.0, ratio))
        nearest_x = start[0] + ratio * direction_x
        nearest_y = start[1] + ratio * direction_y
        return math.hypot(point[0] - nearest_x, point[1] - nearest_y)

    @staticmethod
    def _build_trajectory(
        scene_id: str,
        variant_type: TrajectoryVariantType,
        idx: int,
        pts: FloatArray,
        times: FloatArray,
    ) -> Trajectory:
        return Trajectory(
            traj_id=f"{scene_id}_synthetic_{variant_type.value}_{idx}",
            source=TrajectorySource.SYNTHETIC,
            waypoints=[
                Waypoint(x=float(point[0]), y=float(point[1]), t=float(time))
                for point, time in zip(pts, times, strict=True)
            ],
        )

    @staticmethod
    def _make_augmented(
        trajectory: Trajectory,
        variant_type: TrajectoryVariantType,
        expected_verdict: Literal["cleared", "vetoed", "exclude"],
        note: str,
    ) -> AugmentedTrajectory:
        return AugmentedTrajectory(
            trajectory=trajectory,
            variant_type=variant_type,
            expected_verdict=expected_verdict,
            expected_verdict_reason="待最终几何特征验证",
            generation_note=note,
        )

    @staticmethod
    def _illegal_note(scenario_type: ScenarioType, scene_facts: SceneFacts) -> str:
        if scenario_type == ScenarioType.RED_LIGHT:
            return "红灯扰动：同一路径移除低速段并提前通过停止线"
        if scenario_type == ScenarioType.YELLOW_LIGHT:
            return "黄灯扰动：同一路径使用 1/0.7x 速度剖面"
        if scenario_type == ScenarioType.PEDESTRIAN and not (
            scene_facts.pedestrian_in_forward_crosswalk
        ):
            return "no_violation_injectable：改用通用激进速度剖面"
        if scenario_type == ScenarioType.ONCOMING and scene_facts.ego_turn_intent != "left":
            return "no_violation_injectable：改用通用激进速度剖面"
        if scenario_type in {
            ScenarioType.PEDESTRIAN,
            ScenarioType.ONCOMING,
            ScenarioType.MINOR_ROAD,
            ScenarioType.EMERGENCY,
        }:
            return f"让行扰动（{scenario_type.value}）：同一路径移除低速段"
        return "通用激进速度剖面扰动"
