"""轨迹提取器（v3.1）。

从 RawScene 的 ego_pose 序列重建关键帧局部坐标系下的真实轨迹。
将全局位姿变换到以关键帧自车位姿为原点的局部坐标系。
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.layer1.exceptions import TrajectoryExtractionError
from src.layer1.models import (
    PathTrajectory,
    PathWaypoint,
    RawScene,
    Trajectory,
    TrajectorySource,
    Waypoint,
)


def _quat_to_yaw(q: list[float] | tuple[float, ...]) -> float:
    """将 nuScenes 四元数 [w, x, y, z] 转换为 yaw 角（绕 z 轴旋转）。

    参考 nuScenes-devkit 实现。

    Args:
        q: 四元数 [w, x, y, z]。

    Returns:
        yaw 角（弧度，范围 (-π, π]）。
    """
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


FloatArray = NDArray[np.float64]


def _rot2d(theta: float) -> FloatArray:
    """构造 2D 旋转矩阵。

    Args:
        theta: 旋转角（弧度），逆时针为正。

    Returns:
        2×2 旋转矩阵。
    """
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]])


class TrajectoryExtractor:
    """从 RawScene 的 ego_pose 序列重建关键帧局部坐标系下的真实轨迹。

    坐标系：以关键帧自车位姿为原点，
    x = 纵向（车头方向，前进为正），y = 横向（左为正）。

    ego_pose 由 dataset_loader 预读到 RawScene 中，
    本类为纯函数，不执行 I/O。
    """

    def __init__(
        self,
        future_horizon_s: float = 6.0,
        path_horizon_s: float = 12.0,
        min_waypoints: int = 2,
    ) -> None:
        """初始化提取器。

        Args:
            future_horizon_s: 未来轨迹窗口秒数，v3.1 默认 6 秒。
            path_horizon_s: 候选合成使用的真实路径素材窗口，默认 12 秒。
            min_waypoints: 构成轨迹所需的最少 waypoint 数量。
        """
        if future_horizon_s <= 0:
            raise ValueError("future_horizon_s must be positive")
        if path_horizon_s < future_horizon_s:
            raise ValueError("path_horizon_s must be at least future_horizon_s")
        if min_waypoints < 2:
            raise ValueError("min_waypoints must be at least 2")
        self.future_horizon_s = future_horizon_s
        self.path_horizon_s = path_horizon_s
        self.min_waypoints = min_waypoints

    # -------------------------------------------------------------------------
    # 公开接口
    # -------------------------------------------------------------------------

    def extract(self, raw_scene: RawScene) -> Trajectory:
        """从 RawScene 的 ego_pose 序列重建关键帧局部坐标系下的真实轨迹。

        重建流程（§5.3.1）：
        1. 取关键帧及后续 N 帧的 ego_pose 窗口。
        2. 以关键帧位姿为原点，将后续全局位姿变换到局部坐标系。
        3. 时间戳从微秒转换为相对关键帧的秒数。

        Args:
            raw_scene: 含 ego_pose 序列的原始场景对象。

        Returns:
            局部坐标系下的真实轨迹（source=PLANNING）。

        Raises:
            TrajectoryExtractionError: ego_pose 窗口长度不足以构成轨迹。
        """
        values = self._extract_window(
            raw_scene,
            horizon_s=self.future_horizon_s,
        )
        return Trajectory(
            traj_id=f"{raw_scene.scene_id}_planning_gt_0",
            source=TrajectorySource.PLANNING,
            waypoints=[Waypoint(x=x, y=y, t=t) for x, y, t in values],
        )

    def extract_path(self, raw_scene: RawScene) -> PathTrajectory:
        """提取较长真实路径素材；数据耗尽时不外推。"""
        values = self._extract_window(
            raw_scene,
            horizon_s=self.path_horizon_s,
            poses=raw_scene.path_ego_poses or raw_scene.ego_poses,
        )
        return PathTrajectory(
            traj_id=f"{raw_scene.scene_id}_planning_path_0",
            source=TrajectorySource.PLANNING,
            waypoints=[PathWaypoint(x=x, y=y, t=t) for x, y, t in values],
        )

    def local_drivable_polygons(
        self,
        raw_scene: RawScene,
    ) -> list[list[tuple[float, float]]]:
        """把 Map Expansion drivable_area 多边形转换到关键帧局部坐标系。"""
        if not raw_scene.ego_poses:
            return []
        keyframe = raw_scene.ego_poses[raw_scene.keyframe_index]
        origin = np.array(keyframe["translation"][:2], dtype=float)
        rotation = _rot2d(-_quat_to_yaw(keyframe["rotation"]))
        polygons: list[list[tuple[float, float]]] = []
        for record in raw_scene.map_records.get("drivable_area", []):
            raw_polygons: list[Any] = []
            if isinstance(record.get("polygon_xy"), list):
                raw_polygons.append(record["polygon_xy"])
            polygon_xy_list = record.get("polygon_xy_list")
            if isinstance(polygon_xy_list, list):
                raw_polygons.extend(polygon_xy_list)
            for raw_polygon in raw_polygons:
                if not isinstance(raw_polygon, list) or len(raw_polygon) < 3:
                    continue
                local: list[tuple[float, float]] = []
                for point in raw_polygon:
                    if not isinstance(point, list | tuple) or len(point) < 2:
                        continue
                    relative = rotation @ (np.array(point[:2], dtype=float) - origin)
                    local.append((float(relative[0]), float(relative[1])))
                if len(local) >= 3:
                    polygons.append(local)
        return polygons

    def _extract_window(
        self,
        raw_scene: RawScene,
        *,
        horizon_s: float,
        poses: list[dict[str, Any]] | None = None,
    ) -> list[tuple[float, float, float]]:
        source_poses = poses if poses is not None else raw_scene.ego_poses
        k = raw_scene.keyframe_index
        window = self._select_window(source_poses, k, horizon_s=horizon_s)
        if len(window) > 200:
            prefix = self._select_window(
                source_poses,
                k,
                horizon_s=self.future_horizon_s,
            )
            if horizon_s > self.future_horizon_s and len(prefix) < 200:
                tail = window[len(prefix):]
                budget = 200 - len(prefix)
                indices = np.linspace(0, len(tail) - 1, budget, dtype=int)
                window = prefix + [tail[int(index)] for index in indices]
            else:
                indices = np.linspace(0, len(window) - 1, 200, dtype=int)
                window = [window[int(index)] for index in indices]

        if len(window) < self.min_waypoints:
            raise TrajectoryExtractionError(
                f"ego_pose 窗口长度不足: {len(window)} < {self.min_waypoints}"
            )

        # 以关键帧为原点，把后续全局位姿变换到关键帧局部坐标系
        origin_t = np.array(window[0]["translation"][:2])  # (x, y)
        origin_yaw = _quat_to_yaw(window[0]["rotation"])
        R = _rot2d(-origin_yaw)  # 反旋转到车体朝向

        t0_us = window[0]["timestamp"]
        waypoints: list[tuple[float, float, float]] = []
        for p in window:
            rel = R @ (np.array(p["translation"][:2]) - origin_t)
            dt = (p["timestamp"] - t0_us) / 1_000_000.0  # 微秒 → 秒
            waypoints.append((float(rel[0]), float(rel[1]), float(dt)))
        return waypoints

    def _select_window(
        self,
        poses: list[dict[str, Any]],
        keyframe_index: int,
        *,
        horizon_s: float,
    ) -> list[dict[str, Any]]:
        """按指定窗口选取最多 400 个 ego pose。"""
        if keyframe_index < 0 or keyframe_index >= len(poses):
            return []

        start = poses[keyframe_index]
        start_ts = start.get("timestamp")
        if not isinstance(start_ts, int | float):
            return poses[keyframe_index: keyframe_index + 400]

        horizon_us = horizon_s * 1_000_000.0
        window: list[dict[str, Any]] = []
        for pose in poses[keyframe_index:]:
            ts = pose.get("timestamp")
            if not isinstance(ts, int | float):
                continue
            if ts - start_ts > horizon_us:
                break
            window.append(pose)
            if len(window) >= 400:
                break
        return window
