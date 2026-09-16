"""轨迹特征分析器。

从轨迹坐标序列计算数值语义特征（速度/空间/关键区域），
并生成纯数值事实的自然语言摘要。

natural_language_summary 严禁含合规性结论词
（违规/违反/合规/非法/应该/必须），
LLM 必须仅凭数值语义特征自主推理。
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from src.layer1.exceptions import TrajectoryAnalysisError
from src.layer1.leakage_guard import LeakageGuard
from src.layer1.models import (
    ConflictZone,
    ConflictZoneMetrics,
    SceneDescription,
    SceneFacts,
    Trajectory,
    TrajectoryFeatures,
)

FloatArray = NDArray[np.float64]


class ConflictZoneFeatures(TypedDict):
    """冲突区几何计算的内部结果。"""

    entered_conflict_zone: bool
    min_distance_to_conflict_m: float | None
    full_stop: bool
    stop_duration_s: float
    min_speed_in_zone_mps: float | None
    stopped_before_zone: bool | None


class TrajectoryAnalyzer:
    """从坐标序列计算轨迹数值语义特征。

    输出 TrajectoryFeatures 供 Layer 3 的 LLM prompt 消费，
    替代原始坐标。
    """

    # -------------------------------------------------------------------------
    # 公开接口
    # -------------------------------------------------------------------------

    def analyze(
        self,
        trajectory: Trajectory,
        scene_description: SceneDescription,
        scene_facts: SceneFacts | None = None,
    ) -> TrajectoryFeatures:
        """分析单条轨迹，提取速度/空间/关键区域特征并生成自然语言摘要。

        Args:
            trajectory: 待分析的轨迹对象。
            scene_description: 场景语义描述（用于冲突区域行为描述）。

        Returns:
            计算完成的 TrajectoryFeatures。

        Raises:
            TrajectoryAnalysisError: waypoint 序列过短（< 2）。
        """
        n = len(trajectory.waypoints)
        if n < 2:
            raise TrajectoryAnalysisError(
                f"waypoint 序列过短: {n} < 2，无法提取语义特征"
            )

        # 提取坐标与时间
        pts: FloatArray = np.array([[w.x, w.y] for w in trajectory.waypoints])
        times: FloatArray = np.array([w.t for w in trajectory.waypoints])

        # 速度特征（过滤 dt==0 产生的 NaN）
        speeds = self._compute_speeds(pts, times)
        valid_speeds = speeds[~np.isnan(speeds)]
        max_speed = float(np.max(valid_speeds)) if len(valid_speeds) > 0 else 0.0
        min_speed = float(np.min(valid_speeds)) if len(valid_speeds) > 0 else 0.0
        mean_speed = float(np.mean(valid_speeds)) if len(valid_speeds) > 0 else 0.0

        # 空间特征
        total_distance = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        max_lateral = float(np.max(np.abs(pts[:, 1]))) if n > 0 else 0.0

        zone_features = self._conflict_zone_features(
            pts,
            times,
            speeds,
            scene_facts or SceneFacts(),
        )
        zone_metrics = self._conflict_zone_metrics(
            pts,
            times,
            speeds,
            scene_facts or SceneFacts(),
        )

        # 冲突区域行为描述
        conflict_behavior = self._describe_conflict_zone(
            zone_features,
        )
        detailed_behavior = self._describe_conflict_zone_metrics(zone_metrics, scene_facts)
        if detailed_behavior:
            conflict_behavior = f"{conflict_behavior}；{detailed_behavior}"

        # 自然语言摘要
        summary = self._build_summary(
            max_speed=max_speed,
            min_speed=min_speed,
            mean_speed=mean_speed,
            total_distance=total_distance,
            max_lateral=max_lateral,
            n_waypoints=n,
            conflict_behavior=conflict_behavior,
        )

        LeakageGuard.assert_safe_prompt_fragment(conflict_behavior)
        LeakageGuard.assert_safe_prompt_fragment(summary)

        return TrajectoryFeatures(
            max_speed_mps=max_speed,
            min_speed_mps=min_speed,
            mean_speed_mps=mean_speed,
            total_distance_m=total_distance,
            max_lateral_offset_m=max_lateral,
            n_waypoints=n,
            **zone_features,
            conflict_zone_metrics=zone_metrics,
            conflict_zone_behavior=conflict_behavior,
            natural_language_summary=summary,
        )

    # -------------------------------------------------------------------------
    # 内部计算方法
    # -------------------------------------------------------------------------

    @staticmethod
    def _compute_speeds(pts: FloatArray, times: FloatArray) -> FloatArray:
        """从相邻 waypoint 计算瞬时速度序列。

        Args:
            pts: 坐标数组 (N, 2)。
            times: 时间戳数组 (N,)。

        Returns:
            速度数组 (N-1,)，单位 m/s。dt==0 的段标记为 NaN。
        """
        diffs = np.diff(pts, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        dt = np.diff(times)
        # dt==0 的段产生 NaN，避免虚高速度污染统计量
        speeds = np.full_like(distances, np.nan, dtype=float)
        valid = dt > 0
        speeds[valid] = distances[valid] / dt[valid]
        return speeds

    @staticmethod
    def _describe_conflict_zone(
        features: ConflictZoneFeatures,
    ) -> str:
        """根据真实冲突区几何生成中性事实描述。"""
        min_distance = features["min_distance_to_conflict_m"]
        if min_distance is None:
            return "无冲突区几何数据"
        parts = [f"距最近冲突区最小距离 {float(min_distance):.1f} m"]
        parts.append(
            "轨迹进入冲突区" if features["entered_conflict_zone"] else "轨迹未进入冲突区"
        )
        parts.append(f"累计低速静止 {float(features['stop_duration_s']):.1f} s")
        if features["min_speed_in_zone_mps"] is not None:
            parts.append(
                f"冲突区内最低速度 {float(features['min_speed_in_zone_mps']):.1f} m/s"
            )
        if features["stopped_before_zone"] is True:
            parts.append("进入冲突区前存在完整停车")
        elif features["stopped_before_zone"] is None:
            parts.append("起点已在冲突区内，区前停车关系不适用")
        return "；".join(parts)

    @classmethod
    def _conflict_zone_metrics(
        cls,
        pts: FloatArray,
        times: FloatArray,
        speeds: FloatArray,
        facts: SceneFacts,
    ) -> list[ConflictZoneMetrics]:
        """按 stop_line / ped_crossing 分别聚合适用冲突区指标。"""
        metrics: list[ConflictZoneMetrics] = []
        for kind in ("stop_line", "ped_crossing"):
            zones = [
                zone
                for zone in facts.conflict_zones
                if zone.kind == kind and len(zone.polygon_xy) >= 3
            ]
            if not zones:
                continue
            if kind == "stop_line":
                signed = facts.governing_stop_line_signed_m
                governing_zones = [
                    zone
                    for zone in zones
                    if signed is not None
                    and abs(
                        sum(point[0] for point in zone.polygon_xy) / len(zone.polygon_xy)
                        - signed
                    ) < 2.0
                ]
            else:
                governing_zones = [
                    zone
                    for zone in zones
                    if min(
                        cls._distance_to_polygon(
                            (float(point[0]), float(point[1])), zone.polygon_xy
                        )
                        for point in pts
                    ) <= 3.0
                ]
            selected = governing_zones or zones
            inside = np.array([
                any(
                    cls._point_in_polygon((float(point[0]), float(point[1])), zone.polygon_xy)
                    for zone in selected
                )
                for point in pts
            ])
            distances = [
                min(
                    cls._distance_to_polygon(
                        (float(point[0]), float(point[1])), zone.polygon_xy
                    )
                    for zone in selected
                )
                for point in pts
            ]
            entered = bool(np.any(inside))
            first_inside = int(np.argmax(inside)) if entered else len(pts)
            zone_speed_indices = [
                index
                for index in range(len(speeds))
                if inside[index] or inside[index + 1]
            ]
            zone_speeds = speeds[zone_speed_indices] if zone_speed_indices else np.array([])
            stopped = (speeds < 0.3) & ~np.isnan(speeds)
            stopped_indices = np.flatnonzero(stopped)
            if bool(inside[0]):
                stopped_before: bool | None = None
            else:
                stopped_before = bool(
                    len(stopped_indices)
                    and stopped_indices[0] < first_inside
                    and float(np.diff(times)[stopped].sum()) >= 1.0
                )
            metrics.append(ConflictZoneMetrics(
                kind=kind,
                governing=bool(governing_zones),
                min_distance_m=float(min(distances)),
                entered=entered,
                ended_inside=bool(inside[-1]),
                ended_inside_other_zone=(
                    not bool(inside[-1])
                    and any(
                        cls._point_in_polygon(
                            (float(pts[-1][0]), float(pts[-1][1])),
                            zone.polygon_xy,
                        )
                        for zone in zones
                        if zone not in selected
                    )
                ),
                entry_time_s=float(times[first_inside]) if entered else None,
                penetration_depth_m=(
                    cls._maximum_polygon_penetration(pts, selected)
                    if entered
                    else None
                ),
                min_speed_in_zone_mps=(
                    float(np.nanmin(zone_speeds)) if len(zone_speeds) else None
                ),
                stopped_before_zone=stopped_before,
            ))
        return metrics

    @staticmethod
    def _describe_conflict_zone_metrics(
        metrics: list[ConflictZoneMetrics],
        facts: SceneFacts | None,
    ) -> str:
        parts: list[str] = []
        for metric in metrics:
            if metric.kind == "stop_line" and metric.governing:
                signed = facts.governing_stop_line_signed_m if facts is not None else None
                if signed is not None and signed < 0:
                    parts.append(
                        f"keyframe 时自车已越过本向停止线 {abs(signed):.1f} m，"
                        "区前停车关系不适用"
                    )
                elif metric.entered and metric.entry_time_s is not None:
                    depth = metric.penetration_depth_m or 0.0
                    parts.append(
                        f"轨迹于 {metric.entry_time_s:.1f} s 进入本向停止线区域，"
                        f"最大侵入深度 {depth:.2f} m"
                    )
                if metric.ended_inside:
                    parts.append("预测窗口终点仍位于本向停止线区域")
                elif metric.ended_inside_other_zone:
                    parts.append("预测窗口终点位于另一停止线区域（非本向适用区域）")
            elif metric.kind == "ped_crossing" and metric.governing:
                if metric.entered and metric.entry_time_s is not None:
                    parts.append(f"轨迹于 {metric.entry_time_s:.1f} s 进入前方横道区域")
                if metric.ended_inside:
                    parts.append("预测窗口终点仍位于前方横道区域")
                elif metric.ended_inside_other_zone:
                    parts.append("预测窗口终点位于另一横道区域（非当前适用区域）")
        return "；".join(parts)

    @classmethod
    def _conflict_zone_features(
        cls,
        pts: FloatArray,
        times: FloatArray,
        speeds: FloatArray,
        facts: SceneFacts,
    ) -> ConflictZoneFeatures:
        """计算轨迹与真实冲突区多边形的关系。"""
        polygons = [zone.polygon_xy for zone in facts.conflict_zones if len(zone.polygon_xy) >= 3]
        dt = np.diff(times)
        stopped = (speeds < 0.3) & ~np.isnan(speeds)
        stop_duration = float(dt[stopped].sum())
        full_stop = stop_duration >= 1.0
        if not polygons:
            return {
                "entered_conflict_zone": False,
                "min_distance_to_conflict_m": None,
                "full_stop": full_stop,
                "stop_duration_s": stop_duration,
                "min_speed_in_zone_mps": None,
                "stopped_before_zone": False,
            }

        inside = np.array([
            any(
                cls._point_in_polygon(
                    (float(point[0]), float(point[1])),
                    polygon,
                )
                for polygon in polygons
            )
            for point in pts
        ])
        distances = [
            min(
                cls._distance_to_polygon(
                    (float(point[0]), float(point[1])),
                    polygon,
                )
                for polygon in polygons
            )
            for point in pts
        ]
        zone_speed_indices = [
            index
            for index in range(len(speeds))
            if inside[index] or inside[index + 1]
        ]
        zone_speeds = speeds[zone_speed_indices] if zone_speed_indices else np.array([])
        first_zone_index = int(np.argmax(inside)) if np.any(inside) else len(pts)
        stopped_indices = np.flatnonzero(stopped)
        if bool(inside[0]):
            stopped_before_zone: bool | None = None
        else:
            stopped_before_zone = bool(
                full_stop and len(stopped_indices) and stopped_indices[0] < first_zone_index
            )
        return {
            "entered_conflict_zone": bool(np.any(inside)),
            "min_distance_to_conflict_m": float(min(distances)),
            "full_stop": full_stop,
            "stop_duration_s": stop_duration,
            "min_speed_in_zone_mps": (
                float(np.nanmin(zone_speeds)) if len(zone_speeds) else None
            ),
            "stopped_before_zone": stopped_before_zone,
        }

    @classmethod
    def _point_in_polygon(
        cls,
        point: tuple[float, float],
        polygon: list[tuple[float, float]],
    ) -> bool:
        """使用含边界的射线法判断点是否位于多边形内。"""
        if any(
            cls._distance_to_segment(point, polygon[index - 1], polygon[index])
            <= 1e-9
            for index in range(len(polygon))
        ):
            return True
        x, y = point
        inside = False
        j = len(polygon) - 1
        for i, (xi, yi) in enumerate(polygon):
            xj, yj = polygon[j]
            if (yi > y) != (yj > y):
                crossing_x = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
                if x < crossing_x:
                    inside = not inside
            j = i
        return inside

    @classmethod
    def _distance_to_polygon(
        cls,
        point: tuple[float, float],
        polygon: list[tuple[float, float]],
    ) -> float:
        """计算点到多边形边界的最短距离。"""
        if cls._point_in_polygon(point, polygon):
            return 0.0
        return min(
            cls._distance_to_segment(point, polygon[index - 1], polygon[index])
            for index in range(len(polygon))
        )

    @classmethod
    def _maximum_polygon_penetration(
        cls,
        pts: FloatArray,
        zones: list[ConflictZone],
    ) -> float:
        depths: list[float] = []
        for raw_point in pts:
            point = (float(raw_point[0]), float(raw_point[1]))
            for zone in zones:
                polygon = zone.polygon_xy
                if cls._point_in_polygon(point, polygon):
                    depths.append(min(
                        cls._distance_to_segment(
                            point,
                            polygon[index - 1],
                            polygon[index],
                        )
                        for index in range(len(polygon))
                    ))
        return max(depths, default=0.0)

    @staticmethod
    def _distance_to_segment(
        point: tuple[float, float],
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> float:
        """计算点到线段的欧氏距离。"""
        p = np.array(point)
        a = np.array(start)
        b = np.array(end)
        direction = b - a
        denominator = float(np.dot(direction, direction))
        if denominator == 0.0:
            return float(np.linalg.norm(p - a))
        ratio = float(np.clip(np.dot(p - a, direction) / denominator, 0.0, 1.0))
        return float(np.linalg.norm(p - (a + ratio * direction)))

    @staticmethod
    def _build_summary(
        max_speed: float,
        min_speed: float,
        mean_speed: float,
        total_distance: float,
        max_lateral: float,
        n_waypoints: int,
        conflict_behavior: str,
    ) -> str:
        """从数值特征构建纯事实性自然语言摘要。

        严禁含合规性结论词。仅陈述数值与观察到的事实。

        Args:
            max_speed: 最大速度 (m/s)。
            min_speed: 最小速度 (m/s)。
            mean_speed: 平均速度 (m/s)。
            total_distance: 总行程 (m)。
            max_lateral: 最大横向偏移 (m)。
            n_waypoints: 航点数量。
            conflict_behavior: 冲突区域行为描述。

        Returns:
            自然语言摘要字符串。
        """
        parts: list[str] = []

        # 基础数值
        parts.append(
            f"轨迹共 {n_waypoints} 个航点，"
            f"总行程 {total_distance:.1f} 米"
        )

        # 速度特征
        parts.append(
            f"最大速度 {max_speed:.1f} m/s，"
            f"最小速度 {min_speed:.1f} m/s，"
            f"平均速度 {mean_speed:.1f} m/s"
        )

        # 横向特征
        if max_lateral > 0.05:
            parts.append(f"最大横向偏移 {max_lateral:.2f} 米")
        else:
            parts.append("轨迹基本保持直线")

        # 冲突区域
        if conflict_behavior:
            parts.append(f"冲突区域行为：{conflict_behavior}")

        return "。".join(parts) + "。"

    # -------------------------------------------------------------------------
    # 合规词检查（供测试使用）
    # -------------------------------------------------------------------------

    # 禁止出现在 natural_language_summary 中的词汇
    BANNED_WORDS: frozenset[str] = frozenset({
        "违规", "违反", "合规", "非法", "应该", "必须",
    })

    @classmethod
    def check_banned_words(cls, text: str) -> list[str]:
        """检查文本中是否含有禁止的合规性结论词。

        Args:
            text: 待检查的文本。

        Returns:
            命中的禁止词列表（空列表表示通过检查）。
        """
        return [w for w in cls.BANNED_WORDS if w in text]
