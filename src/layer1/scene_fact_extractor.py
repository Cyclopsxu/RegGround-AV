"""结构化场景事实提取。"""

from __future__ import annotations

import math
from typing import Any, Literal

from src.layer1.models import ConflictZone, RawScene, SceneFacts

_PEDESTRIAN_TRACK_MOTION_THRESHOLD_M = 0.5
_PEDESTRIAN_FOOTPRINT_EPSILON_M = 0.05


def _quat_to_yaw(q: list[float] | tuple[float, ...]) -> float | None:
    if len(q) < 4:
        return None
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _angle_diff_deg(a: float, b: float) -> float:
    diff = abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)
    return math.degrees(diff)


def _point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test."""
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        crosses = (yi > y) != (yj > y)
        if crosses:
            x_at_y = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            if x < x_at_y:
                inside = not inside
        j = i
    return inside


def _distance_to_point(
    points: list[tuple[float, float]],
    origin: tuple[float, float],
) -> float | None:
    if not points:
        return None
    ox, oy = origin
    return min(math.hypot(x - ox, y - oy) for x, y in points)


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    denominator = dx * dx + dy * dy
    if denominator == 0.0:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    ratio = max(0.0, min(1.0, (
        (point[0] - start[0]) * dx + (point[1] - start[1]) * dy
    ) / denominator))
    return math.hypot(
        point[0] - (start[0] + ratio * dx),
        point[1] - (start[1] + ratio * dy),
    )


def _point_overlaps_polygon(
    point: tuple[float, float],
    polygon: list[tuple[float, float]],
    footprint_radius_m: float,
) -> bool:
    if _point_in_polygon(point, polygon):
        return True
    edges = zip(polygon, polygon[1:] + polygon[:1], strict=False)
    return any(
        _point_segment_distance(point, start, end) <= footprint_radius_m
        for start, end in edges
    )


def _pedestrian_footprint_radius(annotation: dict[str, Any]) -> float:
    size = annotation.get("size", [])
    if not isinstance(size, list | tuple) or len(size) < 2:
        return 0.0
    return (
        0.5 * math.hypot(float(size[0]), float(size[1]))
        + _PEDESTRIAN_FOOTPRINT_EPSILON_M
    )


def _track_indicates_pedestrian_motion(
    annotation: dict[str, Any],
) -> bool | None:
    points = [
        point["translation"]
        for point in annotation.get("track", [])
        if isinstance(point, dict)
        and isinstance(point.get("translation"), list | tuple)
        and len(point["translation"]) >= 2
    ]
    if len(points) < 2:
        return None
    origin = points[0]
    displacement = max(
        math.hypot(float(point[0]) - float(origin[0]), float(point[1]) - float(origin[1]))
        for point in points[1:]
    )
    return True if displacement >= _PEDESTRIAN_TRACK_MOTION_THRESHOLD_M else None


def _segments_intersect(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
) -> bool:
    def cross(
        origin: tuple[float, float],
        a: tuple[float, float],
        b: tuple[float, float],
    ) -> float:
        return ((a[0] - origin[0]) * (b[1] - origin[1])
                - (a[1] - origin[1]) * (b[0] - origin[0]))

    first_side = cross(first_start, first_end, second_start)
    second_side = cross(first_start, first_end, second_end)
    third_side = cross(second_start, second_end, first_start)
    fourth_side = cross(second_start, second_end, first_end)
    if all(abs(value) < 1e-9 for value in (
        first_side, second_side, third_side, fourth_side
    )):
        return (
            max(min(first_start[0], first_end[0]), min(second_start[0], second_end[0]))
            <= min(max(first_start[0], first_end[0]), max(second_start[0], second_end[0]))
            and max(min(first_start[1], first_end[1]), min(second_start[1], second_end[1]))
            <= min(max(first_start[1], first_end[1]), max(second_start[1], second_end[1]))
        )
    return first_side * second_side <= 0.0 and third_side * fourth_side <= 0.0


def _global_to_ego(
    point: tuple[float, float],
    ego_xy: tuple[float, float],
    ego_yaw: float,
) -> tuple[float, float]:
    """将全局坐标转换到自车坐标系。"""
    dx = point[0] - ego_xy[0]
    dy = point[1] - ego_xy[1]
    cos_yaw = math.cos(ego_yaw)
    sin_yaw = math.sin(ego_yaw)
    return (
        dx * cos_yaw + dy * sin_yaw,
        -dx * sin_yaw + dy * cos_yaw,
    )


class SceneFactExtractor:
    """从 DriveLM 状态、nuScenes 标注和 Map Expansion 摘要生成 SceneFacts。"""

    def __init__(self, turn_intent_threshold_deg: float = 25.0) -> None:
        if not 0.0 < turn_intent_threshold_deg < 180.0:
            raise ValueError("turn_intent_threshold_deg must be between 0 and 180")
        self._turn_intent_threshold_deg = turn_intent_threshold_deg

    def extract(self, raw_scene: RawScene) -> SceneFacts:
        raw = raw_scene.raw_json
        has_red, has_yellow, status_source = self._traffic_light_status(raw)
        conflict_zones = self._conflict_zones(raw_scene)

        annotations = raw_scene.annotations or self._raw_annotations(raw)
        ego_turn_intent = self._ego_turn_intent(
            raw_scene,
            threshold_deg=self._turn_intent_threshold_deg,
        )
        pedestrian, pedestrian_forward, pedestrian_moving = self._pedestrian_crosswalk_facts(
            annotations,
            conflict_zones,
            raw_scene,
        )
        oncoming = self._oncoming_vehicle_moving(annotations, raw_scene)
        emergency = self._emergency_vehicle_active(annotations)

        text = self._all_text(raw).lower()
        location_is_intersection = self._location_is_intersection(raw_scene, text)
        ego_on_minor_road = self._ego_on_minor_road(raw_scene)

        return SceneFacts(
            has_red_light=has_red,
            has_yellow_light=has_yellow,
            traffic_light_status_source=status_source,
            pedestrian_in_crosswalk=pedestrian,
            oncoming_vehicle_moving=oncoming,
            emergency_vehicle_active=emergency,
            ego_on_minor_road=ego_on_minor_road,
            location_is_intersection=location_is_intersection,
            ego_displacement_m=self._ego_displacement_m(raw_scene),
            conflict_zones=conflict_zones,
            ego_turn_intent=ego_turn_intent,
            governing_stop_line_signed_m=self._governing_stop_line_signed_m(conflict_zones),
            pedestrian_in_forward_crosswalk=pedestrian_forward,
            pedestrian_moving=pedestrian_moving,
            window_insufficient=self._window_insufficient(raw_scene),
        )

    @staticmethod
    def _window_insufficient(raw_scene: RawScene) -> bool:
        poses = raw_scene.ego_poses[raw_scene.keyframe_index:]
        if len(poses) < 60:
            return True
        timestamps = [pose.get("timestamp") for pose in poses]
        numeric: list[float] = []
        for value in timestamps:
            if not isinstance(value, int | float):
                return True
            numeric.append(float(value))
        intervals = [
            current - previous
            for previous, current in zip(numeric, numeric[1:], strict=False)
            if current > previous
        ]
        if not intervals:
            return True
        sample_period = sorted(intervals)[len(intervals) // 2]
        # 离散采样窗的覆盖时长包含末点所代表的一个采样周期；允许 1 ms 时间戳抖动。
        effective_duration_us = numeric[-1] - numeric[0] + sample_period
        return effective_duration_us < 5_999_000.0

    @staticmethod
    def _ego_turn_intent(
        raw_scene: RawScene,
        threshold_deg: float = 25.0,
    ) -> Literal["left", "right", "straight", "unknown"]:
        poses = raw_scene.ego_poses[raw_scene.keyframe_index:]
        if len(poses) < 2:
            return "unknown"
        yaws = [_quat_to_yaw(pose.get("rotation", [])) for pose in poses]
        if any(yaw is None for yaw in yaws):
            return "unknown"
        total = 0.0
        for previous, current in zip(yaws, yaws[1:], strict=False):
            assert previous is not None and current is not None
            total += (current - previous + math.pi) % (2.0 * math.pi) - math.pi
        total_deg = math.degrees(total)
        if total_deg > threshold_deg:
            return "left"
        if total_deg < -threshold_deg:
            return "right"
        return "straight"

    @staticmethod
    def _governing_stop_line_signed_m(conflict_zones: list[ConflictZone]) -> float | None:
        candidates: list[float] = []
        for zone in conflict_zones:
            if zone.kind != "stop_line" or len(zone.polygon_xy) < 2:
                continue
            cx = sum(point[0] for point in zone.polygon_xy) / len(zone.polygon_xy)
            if abs(cx) > 50.0:
                continue
            # 停止线长轴应大致横跨车道，其法向才与车头方向一致。
            longest = max(
                (
                    (
                        math.hypot(end[0] - start[0], end[1] - start[1]),
                        end[0] - start[0],
                        end[1] - start[1],
                    )
                    for start, end in zip(
                        zone.polygon_xy,
                        zone.polygon_xy[1:] + zone.polygon_xy[:1],
                        strict=False,
                    )
                ),
                default=(0.0, 0.0, 0.0),
            )
            length, dx, dy = longest
            if length == 0.0:
                continue
            normal_alignment = abs(dy) / length
            if normal_alignment < math.cos(math.radians(45.0)):
                continue
            candidates.append(cx)
        return min(candidates, key=abs) if candidates else None

    @staticmethod
    def _ego_displacement_m(raw_scene: RawScene) -> float | None:
        """计算关键帧起至预测窗口末端的自车直线位移。"""
        poses = raw_scene.ego_poses[raw_scene.keyframe_index:]
        if len(poses) < 2:
            return None
        start = poses[0].get("translation", [])
        end = poses[-1].get("translation", [])
        if not all(isinstance(value, (list, tuple)) and len(value) >= 2 for value in (start, end)):
            return None
        return math.hypot(float(end[0]) - float(start[0]), float(end[1]) - float(start[1]))

    @staticmethod
    def _traffic_light_status(
        raw: dict[str, Any],
    ) -> tuple[bool, bool, Literal["drivelm_status", "none"]]:
        key_objects = raw.get("key_object_infos", {})
        if not isinstance(key_objects, dict):
            return (False, False, "none")

        has_red = False
        has_yellow = False
        for obj_info in key_objects.values():
            if not isinstance(obj_info, dict):
                continue
            category = str(obj_info.get("category", "")).lower()
            if category not in ("traffic_light", "traffic light", "红绿灯", "信号灯"):
                continue
            status = str(obj_info.get("Status", obj_info.get("status", ""))).lower()
            has_red = has_red or status in ("red", "red light", "红灯")
            has_yellow = has_yellow or status in ("yellow", "amber", "yellow light", "黄灯")

        if has_red or has_yellow:
            return (has_red, has_yellow, "drivelm_status")
        return (False, False, "none")

    @staticmethod
    def _conflict_zones(raw_scene: RawScene) -> list[ConflictZone]:
        zones: list[ConflictZone] = []
        ego_xy = (0.0, 0.0)
        ego_yaw = 0.0
        if raw_scene.ego_poses:
            ego_pose = raw_scene.ego_poses[raw_scene.keyframe_index]
            translation = ego_pose.get("translation", [])
            if isinstance(translation, (list, tuple)) and len(translation) >= 2:
                ego_xy = (float(translation[0]), float(translation[1]))
            parsed_yaw = _quat_to_yaw(ego_pose.get("rotation", []))
            if parsed_yaw is not None:
                ego_yaw = parsed_yaw
        for layer_name in ("stop_line", "ped_crossing", "yield_line"):
            records = raw_scene.map_records.get(layer_name, [])
            if not isinstance(records, list):
                continue
            for record in records:
                if not isinstance(record, dict):
                    continue
                global_polygon = SceneFactExtractor._polygon_xy(record)
                polygon = [
                    _global_to_ego(point, ego_xy, ego_yaw) for point in global_polygon
                ]
                zones.append(
                    ConflictZone(
                        kind=layer_name,
                        polygon_xy=polygon,
                        distance_from_ego_m=_distance_to_point(polygon, (0.0, 0.0)),
                        source_layer=layer_name,
                    )
                )
        return zones

    @staticmethod
    def _polygon_xy(record: dict[str, Any]) -> list[tuple[float, float]]:
        raw_polygon = (
            record.get("polygon_xy")
            or record.get("polygon")
            or record.get("coords")
            or record.get("points")
            or []
        )
        polygon: list[tuple[float, float]] = []
        if isinstance(raw_polygon, list):
            for point in raw_polygon:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    polygon.append((float(point[0]), float(point[1])))
        return polygon

    @staticmethod
    def _raw_annotations(raw: dict[str, Any]) -> list[dict[str, Any]]:
        annotations = raw.get("annotations", raw.get("sample_annotations", []))
        if isinstance(annotations, list):
            return [a for a in annotations if isinstance(a, dict)]
        return []

    @staticmethod
    def _pedestrian_crosswalk_facts(
        annotations: list[dict[str, Any]],
        conflict_zones: list[ConflictZone],
        raw_scene: RawScene,
    ) -> tuple[bool, bool, bool | None]:
        crossings = [z.polygon_xy for z in conflict_zones if z.kind == "ped_crossing"]
        if not crossings:
            return (False, False, None)

        if not raw_scene.ego_poses:
            return (False, False, None)
        ego_pose = raw_scene.ego_poses[raw_scene.keyframe_index]
        translation = ego_pose.get("translation", [])
        ego_yaw = _quat_to_yaw(ego_pose.get("rotation", []))
        if not isinstance(translation, (list, tuple)) or len(translation) < 2 or ego_yaw is None:
            return (False, False, None)
        ego_xy = (float(translation[0]), float(translation[1]))

        path = SceneFactExtractor._ego_path_local(raw_scene, ego_xy, ego_yaw)
        forward_crossings = [
            polygon
            for polygon in crossings
            if SceneFactExtractor._path_intersects_polygon_buffer(path, polygon, 3.0)
        ]

        pedestrian = False
        pedestrian_forward = False
        moving_values: list[bool | None] = []

        for ann in annotations:
            category = str(ann.get("category_name", ann.get("category", ""))).lower()
            if not category.startswith("human.pedestrian"):
                continue
            attrs = SceneFactExtractor._attributes(ann)
            if "pedestrian.moving" in attrs:
                is_moving: bool | None = True
            elif attrs.intersection({"pedestrian.standing", "pedestrian.sitting_lying_down"}):
                is_moving = False
            else:
                is_moving = _track_indicates_pedestrian_motion(ann)
            center = ann.get("translation", ann.get("center"))
            if not isinstance(center, (list, tuple)) or len(center) < 2:
                continue
            point = _global_to_ego(
                (float(center[0]), float(center[1])),
                ego_xy,
                ego_yaw,
            )
            footprint_radius = _pedestrian_footprint_radius(ann)
            if any(
                _point_overlaps_polygon(point, polygon, footprint_radius)
                for polygon in crossings
            ):
                pedestrian = True
            if any(
                _point_overlaps_polygon(point, polygon, footprint_radius)
                for polygon in forward_crossings
            ):
                pedestrian_forward = True
                moving_values.append(is_moving)
        if any(value is True for value in moving_values):
            moving: bool | None = True
        elif moving_values and all(value is False for value in moving_values):
            moving = False
        else:
            moving = None
        return (pedestrian, pedestrian_forward, moving)

    @staticmethod
    def _ego_path_local(
        raw_scene: RawScene,
        ego_xy: tuple[float, float],
        ego_yaw: float,
    ) -> list[tuple[float, float]]:
        path: list[tuple[float, float]] = []
        for pose in raw_scene.ego_poses[raw_scene.keyframe_index:]:
            translation = pose.get("translation", [])
            if isinstance(translation, (list, tuple)) and len(translation) >= 2:
                path.append(_global_to_ego(
                    (float(translation[0]), float(translation[1])), ego_xy, ego_yaw
                ))
        return path

    @staticmethod
    def _path_intersects_polygon_buffer(
        path: list[tuple[float, float]],
        polygon: list[tuple[float, float]],
        buffer_m: float,
    ) -> bool:
        if not path or not polygon:
            return False
        for point in path:
            if _point_in_polygon(point, polygon):
                return True
            if min(math.hypot(point[0] - x, point[1] - y) for x, y in polygon) <= buffer_m:
                return True
        path_segments = list(zip(path, path[1:], strict=False))
        polygon_segments = list(zip(
            polygon,
            polygon[1:] + polygon[:1],
            strict=False,
        ))
        for path_start, path_end in path_segments:
            for polygon_start, polygon_end in polygon_segments:
                if _segments_intersect(path_start, path_end, polygon_start, polygon_end):
                    return True
                if min(
                    _point_segment_distance(path_start, polygon_start, polygon_end),
                    _point_segment_distance(path_end, polygon_start, polygon_end),
                    _point_segment_distance(polygon_start, path_start, path_end),
                    _point_segment_distance(polygon_end, path_start, path_end),
                ) <= buffer_m:
                    return True
        return False

    @staticmethod
    def _oncoming_vehicle_moving(
        annotations: list[dict[str, Any]],
        raw_scene: RawScene,
    ) -> bool:
        if not raw_scene.ego_poses:
            return False
        ego_rotation = raw_scene.ego_poses[raw_scene.keyframe_index].get("rotation", [])
        ego_yaw = _quat_to_yaw(ego_rotation)
        if ego_yaw is None:
            return False

        for ann in annotations:
            category = str(ann.get("category_name", ann.get("category", ""))).lower()
            if not category.startswith("vehicle."):
                continue
            attrs = SceneFactExtractor._attributes(ann)
            if attrs and "vehicle.moving" not in attrs:
                continue
            yaw = _quat_to_yaw(ann.get("rotation", []))
            if yaw is not None and _angle_diff_deg(ego_yaw, yaw) > 150.0:
                return True
        return False

    @staticmethod
    def _emergency_vehicle_active(annotations: list[dict[str, Any]]) -> bool:
        emergency_terms = ("vehicle.emergency", "ambulance", "police", "fire")
        for ann in annotations:
            category = str(ann.get("category_name", ann.get("category", ""))).lower()
            attrs = SceneFactExtractor._attributes(ann)
            if not any(term in category for term in emergency_terms):
                continue
            if attrs and "vehicle.moving" not in attrs:
                continue
            return True
        return False

    @staticmethod
    def _attributes(annotation: dict[str, Any]) -> set[str]:
        attrs = annotation.get("attribute_names", annotation.get("attributes", []))
        if isinstance(attrs, str):
            return {attrs.lower()}
        if isinstance(attrs, list):
            return {str(attr).lower() for attr in attrs}
        return set()

    @staticmethod
    def _location_is_intersection(raw_scene: RawScene, text: str) -> bool | None:
        if raw_scene.map_records.get("road_segment"):
            return True
        if "intersection" in text or "junction" in text:
            return True
        return None

    @staticmethod
    def _ego_on_minor_road(raw_scene: RawScene) -> bool | None:
        if raw_scene.map_records.get("lane"):
            for record in raw_scene.map_records["lane"]:
                lane_type = str(record.get("lane_type", record.get("type", ""))).lower()
                if "minor" in lane_type or "side" in lane_type:
                    return True
        return None

    @staticmethod
    def _all_text(raw: dict[str, Any]) -> str:
        parts: list[str] = []
        qa_pairs = raw.get("qa_pairs", [])
        if isinstance(qa_pairs, list):
            for qa in qa_pairs:
                if isinstance(qa, dict):
                    parts.extend(str(qa.get(field, "")) for field in ("question", "answer"))
        for field in ("perception", "prediction", "planning", "behavior"):
            value = raw.get(field)
            if isinstance(value, str):
                parts.append(value)
        keywords = raw.get("keywords", [])
        if isinstance(keywords, list):
            parts.extend(str(kw) for kw in keywords)
        return " ".join(parts)
