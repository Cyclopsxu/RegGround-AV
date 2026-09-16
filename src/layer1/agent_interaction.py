"""候选轨迹与行人/对向车辆的中性时空交互特征。"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np

from src.layer1.models import AgentInteraction, RawScene, SceneFacts, Trajectory


def _quat_to_yaw(rotation: list[float] | tuple[float, ...]) -> float | None:
    if len(rotation) < 4:
        return None
    w, x, y, z = (float(value) for value in rotation[:4])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _to_local(
    point: tuple[float, float],
    origin: tuple[float, float],
    yaw: float,
) -> tuple[float, float]:
    dx, dy = point[0] - origin[0], point[1] - origin[1]
    return (
        dx * math.cos(yaw) + dy * math.sin(yaw),
        -dx * math.sin(yaw) + dy * math.cos(yaw),
    )


def _inside(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            crossing = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            if x < crossing:
                inside = not inside
        j = i
    return inside


class AgentInteractionAnalyzer:
    """从 annotation 跟踪链提取每条候选最多三个相关智能体。"""

    def analyze(
        self,
        trajectory: Trajectory,
        raw_scene: RawScene,
        facts: SceneFacts,
    ) -> list[AgentInteraction]:
        if not raw_scene.ego_poses:
            return []
        pose = raw_scene.ego_poses[raw_scene.keyframe_index]
        translation = pose.get("translation", [])
        yaw = _quat_to_yaw(pose.get("rotation", []))
        if not isinstance(translation, list | tuple) or len(translation) < 2 or yaw is None:
            return []
        origin = (float(translation[0]), float(translation[1]))
        interactions: list[AgentInteraction] = []
        for annotation in raw_scene.annotations:
            category = str(
                annotation.get("category_name", annotation.get("category", ""))
            ).lower()
            if category.startswith("human.pedestrian"):
                kind = "pedestrian"
            elif category.startswith("vehicle."):
                agent_yaw = _quat_to_yaw(annotation.get("rotation", []))
                if agent_yaw is None or abs(
                    math.degrees((agent_yaw - yaw + math.pi) % (2 * math.pi) - math.pi)
                ) < 135.0:
                    continue
                kind = "oncoming_vehicle"
            else:
                continue
            track = self._local_track(annotation, origin, yaw)
            if not track:
                continue
            interaction = (
                self._pedestrian_interaction(trajectory, facts, annotation, track)
                if kind == "pedestrian"
                else self._oncoming_interaction(trajectory, annotation, track)
            )
            if interaction is not None:
                interactions.append(interaction)
        interactions.sort(
            key=lambda item: (
                float("inf") if item.min_time_gap_s is None else item.min_time_gap_s,
                float("inf")
                if item.min_spatiotemporal_gap_m is None
                else item.min_spatiotemporal_gap_m,
            )
        )
        return interactions[:3]

    @staticmethod
    def _local_track(
        annotation: dict[str, Any],
        origin: tuple[float, float],
        yaw: float,
    ) -> list[tuple[float, float, float]]:
        raw_track = annotation.get("track", [])
        if not raw_track:
            raw_track = [{"t": 0.0, "translation": annotation.get("translation", [])}]
        track: list[tuple[float, float, float]] = []
        for point in raw_track:
            if not isinstance(point, dict):
                continue
            translation = point.get("translation", [])
            time = point.get("t")
            if (
                isinstance(translation, list | tuple)
                and len(translation) >= 2
                and isinstance(time, int | float)
            ):
                x, y = _to_local(
                    (float(translation[0]), float(translation[1])), origin, yaw
                )
                track.append((float(time), x, y))
        return sorted(track)

    @staticmethod
    def _track_id(annotation: dict[str, Any]) -> str:
        return str(annotation.get("instance_token", annotation.get("token", "unknown")))

    def _pedestrian_interaction(
        self,
        trajectory: Trajectory,
        facts: SceneFacts,
        annotation: dict[str, Any],
        track: list[tuple[float, float, float]],
    ) -> AgentInteraction | None:
        polygons = [
            zone.polygon_xy
            for zone in facts.conflict_zones
            if zone.kind == "ped_crossing" and len(zone.polygon_xy) >= 3
        ]
        if not polygons:
            return None
        ego_times = [
            waypoint.t
            for waypoint in trajectory.waypoints
            if any(_inside((waypoint.x, waypoint.y), polygon) for polygon in polygons)
        ]
        agent_times = [
            time
            for time, x, y in track
            if any(_inside((x, y), polygon) for polygon in polygons)
        ]
        min_distance = self._minimum_simultaneous_distance(trajectory, track)
        if not ego_times or not agent_times:
            return AgentInteraction(
                agent_kind="pedestrian",
                agent_track_id=self._track_id(annotation),
                crossing_order="no_shared_zone",
                min_spatiotemporal_gap_m=min_distance,
            )
        ego_window = (min(ego_times), max(ego_times))
        agent_window = (min(agent_times), max(agent_times))
        gap, order = self._window_gap(ego_window, agent_window)
        return AgentInteraction(
            agent_kind="pedestrian",
            agent_track_id=self._track_id(annotation),
            ego_zone_arrival_s=ego_window[0],
            agent_zone_window_s=agent_window,
            min_time_gap_s=gap,
            min_spatiotemporal_gap_m=min_distance,
            crossing_order=order,
        )

    def _oncoming_interaction(
        self,
        trajectory: Trajectory,
        annotation: dict[str, Any],
        track: list[tuple[float, float, float]],
    ) -> AgentInteraction:
        ego_time, agent_time, path_gap = self._closest_path_approach(trajectory, track)
        simultaneous_gap = self._minimum_simultaneous_distance(trajectory, track)
        if path_gap > 4.0:
            return AgentInteraction(
                agent_kind="oncoming_vehicle",
                agent_track_id=self._track_id(annotation),
                min_spatiotemporal_gap_m=simultaneous_gap,
                crossing_order="no_shared_zone",
            )
        gap, order = self._window_gap(
            (max(0.0, ego_time - 0.5), ego_time + 0.5),
            (max(0.0, agent_time - 0.5), agent_time + 0.5),
        )
        return AgentInteraction(
            agent_kind="oncoming_vehicle",
            agent_track_id=self._track_id(annotation),
            ego_zone_arrival_s=ego_time,
            agent_zone_window_s=(max(0.0, agent_time - 0.5), agent_time + 0.5),
            min_time_gap_s=gap,
            min_spatiotemporal_gap_m=simultaneous_gap,
            crossing_order=order,
        )

    @staticmethod
    def _closest_path_approach(
        trajectory: Trajectory,
        track: list[tuple[float, float, float]],
    ) -> tuple[float, float, float]:
        """返回两条离散路径最近点的双方时刻与空间距离。"""
        best = (float("inf"), float("inf"), 0.0, 0.0)
        for waypoint in trajectory.waypoints:
            for agent_time, agent_x, agent_y in track:
                distance = math.hypot(waypoint.x - agent_x, waypoint.y - agent_y)
                best = min(
                    best,
                    (distance, abs(waypoint.t - agent_time), waypoint.t, agent_time),
                )
        return best[2], best[3], best[0]

    @staticmethod
    def _window_gap(
        ego: tuple[float, float],
        agent: tuple[float, float],
    ) -> tuple[
        float,
        Literal["ego_first", "agent_first", "temporal_overlap"],
    ]:
        overlap = min(ego[1], agent[1]) - max(ego[0], agent[0])
        if overlap >= 0:
            penetration = overlap
            if penetration == 0.0 and ego[0] == ego[1] and agent[0] < ego[0] < agent[1]:
                penetration = min(ego[0] - agent[0], agent[1] - ego[0])
            elif (
                penetration == 0.0
                and agent[0] == agent[1]
                and ego[0] < agent[0] < ego[1]
            ):
                penetration = min(agent[0] - ego[0], ego[1] - agent[0])
            return (-penetration, "temporal_overlap")
        if ego[1] < agent[0]:
            return (agent[0] - ego[1], "ego_first")
        return (ego[0] - agent[1], "agent_first")

    @staticmethod
    def _minimum_simultaneous_distance(
        trajectory: Trajectory,
        track: list[tuple[float, float, float]],
    ) -> float | None:
        if not track:
            return None
        agent_times = np.array([point[0] for point in track], dtype=float)
        agent_xs = np.array([point[1] for point in track], dtype=float)
        agent_ys = np.array([point[2] for point in track], dtype=float)
        ego_times = np.array([point.t for point in trajectory.waypoints], dtype=float)
        ego_xs = np.array([point.x for point in trajectory.waypoints], dtype=float)
        ego_ys = np.array([point.y for point in trajectory.waypoints], dtype=float)
        start = max(float(agent_times[0]), float(ego_times[0]))
        end = min(float(agent_times[-1]), float(ego_times[-1]))
        if end < start:
            return None
        sample_times = sorted({
            float(time)
            for time in np.concatenate((agent_times, ego_times))
            if start <= time <= end
        })
        values = [
            math.hypot(
                float(np.interp(time, ego_times, ego_xs))
                - float(np.interp(time, agent_times, agent_xs)),
                float(np.interp(time, ego_times, ego_ys))
                - float(np.interp(time, agent_times, agent_ys)),
            )
            for time in sample_times
        ]
        return min(values) if values else None

def render_agent_interactions(interactions: list[AgentInteraction]) -> str:
    """将交互事实渲染为中性摘要，不包含规则结论。"""
    sentences: list[str] = []
    for item in interactions:
        subject = "行人" if item.agent_kind == "pedestrian" else "对向车辆"
        if item.crossing_order == "no_shared_zone":
            sentences.append(
                f"{subject}与该轨迹未形成共同冲突区；"
                f"最近同时刻距离 {item.min_spatiotemporal_gap_m:.1f} m。"
                if item.min_spatiotemporal_gap_m is not None
                else f"{subject}与该轨迹未形成共同冲突区。"
            )
            continue
        assert item.ego_zone_arrival_s is not None
        assert item.agent_zone_window_s is not None
        assert item.min_time_gap_s is not None
        distance = (
            "未知"
            if item.min_spatiotemporal_gap_m is None
            else f"{item.min_spatiotemporal_gap_m:.1f} m"
        )
        sentences.append(
            f"{subject}占据共同冲突区的时间窗为 "
            f"{item.agent_zone_window_s[0]:.1f}–{item.agent_zone_window_s[1]:.1f} s；"
            f"该轨迹 {item.ego_zone_arrival_s:.1f} s 后进入，"
            f"最小时间间隔 {item.min_time_gap_s:.1f} s，最近同时刻距离 {distance}。"
        )
    return "".join(sentences)
