"""把分散的 nuScenes 原始表整合成 Layer 1 可直接读取的数据集。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, cast

from src.layer1.drivelm_adapter import (
    flatten_drivelm_qa,
    normalize_drivelm_key_object_infos,
)

TARGET_LOCATION = "boston-seaport"
TARGET_CHANNEL = "LIDAR_TOP"
DEFAULT_SOURCE_ROOT = Path("v1.0-mini")
DEFAULT_OUTPUT = Path("data/layer1/raw_scene_dataset_mini.json")
DEFAULT_DRIVELM_QA_FILENAME = "v1_0_train_nus.json"
DEFAULT_MAP_EXPANSION_RELATIVE_PATH = Path(
    "nuScenes-map-expansion-v1.3/expansion/boston-seaport.json"
)
DEFAULT_MAP_RADIUS_M = 50.0
REQUIRED_TABLES = (
    "scene",
    "log",
    "sample",
    "sample_data",
    "calibrated_sensor",
    "sensor",
    "ego_pose",
    "sample_annotation",
    "instance",
    "category",
    "attribute",
)
MAP_LAYERS = (
    "stop_line",
    "ped_crossing",
    "traffic_light",
    "road_segment",
    "lane",
    "drivable_area",
)
SCENE_KEYWORDS = (
    "red light",
    "traffic light",
    "stop line",
    "yellow light",
    "pedestrian",
    "crosswalk",
    "crossing",
    "oncoming",
    "opposite direction",
    "left turn",
    "yield",
    "minor road",
    "side road",
    "intersection",
    "junction",
    "ambulance",
    "police car",
    "fire truck",
    "emergency vehicle",
)


def load_table(source_root: Path, table_name: str) -> list[dict[str, Any]]:
    """读取一张 nuScenes JSON 表，并保证顶层是记录列表。"""
    table_path = source_root / f"{table_name}.json"
    if not table_path.exists():
        raise FileNotFoundError(f"缺少必要原始表: {table_path}")

    with table_path.open(encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{table_path} 顶层必须是 list")
    return [row for row in data if isinstance(row, dict)]


def load_drivelm_data(path: Path | None) -> dict[str, Any] | None:
    """读取 DriveLM QA 文件；未提供时返回 None，保留 nuScenes-only 流程。"""
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"DriveLM QA 文件不存在: {path}")

    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} 顶层必须是 dict")
    return data


def load_map_data(path: Path | None) -> dict[str, Any] | None:
    """读取 Map Expansion 文件；未提供时返回 None。"""
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Map Expansion 文件不存在: {path}")

    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} 顶层必须是 dict")
    return data


def build_token_index(rows: list[dict[str, Any]], table_name: str) -> dict[str, dict[str, Any]]:
    """按 token 建索引；缺 token 的记录直接跳过。"""
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        token = row.get("token")
        if isinstance(token, str) and token:
            index[token] = row
    if not index:
        raise ValueError(f"{table_name}.json 没有可用 token 记录")
    return index


def group_by_field(rows: list[dict[str, Any]], field: str) -> dict[str, list[dict[str, Any]]]:
    """把记录按指定字符串字段分组。"""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = row.get(field)
        if isinstance(key, str) and key:
            grouped.setdefault(key, []).append(row)
    return grouped


def is_number(value: Any) -> bool:
    """bool 是 int 子类，这里显式排除，避免 True/False 混入数值字段。"""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def sample_data_channel(
    sample_data: dict[str, Any],
    calibrated_sensor_by_token: dict[str, dict[str, Any]],
    sensor_by_token: dict[str, dict[str, Any]],
) -> str | None:
    """解析 sample_data 对应的传感器通道。"""
    channel = sample_data.get("channel")
    if isinstance(channel, str) and channel:
        return channel

    calib_token = sample_data.get("calibrated_sensor_token")
    if not isinstance(calib_token, str):
        return None
    calibrated_sensor = calibrated_sensor_by_token.get(calib_token)
    if not calibrated_sensor:
        return None

    sensor_token = calibrated_sensor.get("sensor_token")
    if not isinstance(sensor_token, str):
        return None
    sensor = sensor_by_token.get(sensor_token)
    if not sensor:
        return None

    channel = sensor.get("channel")
    return channel if isinstance(channel, str) and channel else None


def build_lidar_index(
    sample_data_rows: list[dict[str, Any]],
    calibrated_sensor_by_token: dict[str, dict[str, Any]],
    sensor_by_token: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """保留 LIDAR_TOP 数据，并按 sample_token 分组。"""
    lidar_by_sample: dict[str, list[dict[str, Any]]] = {}
    for sample_data in sample_data_rows:
        channel = sample_data_channel(
            sample_data,
            calibrated_sensor_by_token,
            sensor_by_token,
        )
        if channel != TARGET_CHANNEL:
            continue
        sample_token = sample_data.get("sample_token")
        if isinstance(sample_token, str) and sample_token:
            lidar_by_sample.setdefault(sample_token, []).append(sample_data)

    for rows in lidar_by_sample.values():
        rows.sort(key=lambda row: int(row.get("timestamp", 0)))
    return lidar_by_sample


def pick_lidar_sample_data(
    sample_token: str,
    lidar_by_sample: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    """优先选择关键帧 LIDAR_TOP，缺失时退回到该 sample 最早的 LIDAR_TOP。"""
    candidates = lidar_by_sample.get(sample_token, [])
    if not candidates:
        return None

    keyframes = [row for row in candidates if row.get("is_key_frame") is True]
    return (keyframes or candidates)[0]


def normalize_ego_pose(
    sample_data: dict[str, Any],
    ego_pose_by_token: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """从 sample_data 关联的 ego_pose 中抽取 Layer 1 需要的轻量字段。"""
    ego_pose_token = sample_data.get("ego_pose_token")
    if not isinstance(ego_pose_token, str):
        return None

    ego_pose = ego_pose_by_token.get(ego_pose_token)
    if not ego_pose:
        return None

    timestamp = ego_pose.get("timestamp", sample_data.get("timestamp"))
    translation = ego_pose.get("translation")
    rotation = ego_pose.get("rotation")
    if not is_number(timestamp):
        return None
    if not isinstance(translation, list) or len(translation) < 3:
        return None
    if not isinstance(rotation, list) or len(rotation) < 4:
        return None
    if not all(is_number(value) for value in translation[:3]):
        return None
    if not all(is_number(value) for value in rotation[:4]):
        return None

    return {
        "token": str(ego_pose.get("token", ego_pose_token)),
        "timestamp": int(cast(int | float, timestamp)),
        "rotation": [float(value) for value in rotation[:4]],
        "translation": [float(value) for value in translation[:3]],
    }


def next_lidar_sample_data(
    current: dict[str, Any],
    sample_by_token: dict[str, dict[str, Any]],
    sample_data_by_token: dict[str, dict[str, Any]],
    lidar_by_sample: dict[str, list[dict[str, Any]]],
    calibrated_sensor_by_token: dict[str, dict[str, Any]],
    sensor_by_token: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """沿 sample_data.next 读取密集位姿，断链时退回 sample.next。"""
    next_sd_token = current.get("next")
    if isinstance(next_sd_token, str) and next_sd_token:
        next_sd = sample_data_by_token.get(next_sd_token)
        if next_sd and sample_data_channel(
            next_sd,
            calibrated_sensor_by_token,
            sensor_by_token,
        ) == TARGET_CHANNEL:
            return next_sd

    sample_token = current.get("sample_token")
    if not isinstance(sample_token, str):
        return None
    sample = sample_by_token.get(sample_token)
    if not sample:
        return None
    next_sample_token = sample.get("next")
    if not isinstance(next_sample_token, str) or not next_sample_token:
        return None
    return pick_lidar_sample_data(next_sample_token, lidar_by_sample)


def collect_ego_poses(
    start_sample_token: str,
    sample_by_token: dict[str, dict[str, Any]],
    sample_data_by_token: dict[str, dict[str, Any]],
    lidar_by_sample: dict[str, list[dict[str, Any]]],
    ego_pose_by_token: dict[str, dict[str, Any]],
    calibrated_sensor_by_token: dict[str, dict[str, Any]],
    sensor_by_token: dict[str, dict[str, Any]],
    future_horizon_s: float,
) -> list[dict[str, Any]]:
    """采集从关键帧开始的未来 LIDAR_TOP ego_pose 窗口。"""
    current = pick_lidar_sample_data(start_sample_token, lidar_by_sample)
    if not current:
        return []

    horizon_us = future_horizon_s * 1_000_000.0
    start_timestamp: int | None = None
    poses: list[dict[str, Any]] = []
    seen_sample_data_tokens: set[str] = set()

    while current:
        token = current.get("token")
        if not isinstance(token, str) or token in seen_sample_data_tokens:
            break
        seen_sample_data_tokens.add(token)

        pose = normalize_ego_pose(current, ego_pose_by_token)
        if pose is None:
            break
        if start_timestamp is None:
            start_timestamp = pose["timestamp"]
        if pose["timestamp"] - start_timestamp > horizon_us:
            break

        poses.append(pose)
        if len(poses) >= 400:
            break

        current = next_lidar_sample_data(
            current,
            sample_by_token,
            sample_data_by_token,
            lidar_by_sample,
            calibrated_sensor_by_token,
            sensor_by_token,
        )

    return poses


def collect_annotations(
    sample_token: str,
    annotations_by_sample: dict[str, list[dict[str, Any]]],
    annotation_by_token: dict[str, dict[str, Any]],
    sample_by_token: dict[str, dict[str, Any]],
    instance_by_token: dict[str, dict[str, Any]],
    category_by_token: dict[str, dict[str, Any]],
    attribute_by_token: dict[str, dict[str, Any]],
    future_horizon_s: float,
) -> list[dict[str, Any]]:
    """整理当前关键帧标注，并沿 ``next`` 补齐未来智能体轨迹。"""
    start_sample = sample_by_token.get(sample_token, {})
    start_timestamp = start_sample.get("timestamp")
    annotations: list[dict[str, Any]] = []
    for raw in annotations_by_sample.get(sample_token, []):
        annotation: dict[str, Any] = {}
        for field in (
            "token",
            "sample_token",
            "instance_token",
            "translation",
            "size",
            "rotation",
            "attribute_tokens",
            "visibility_token",
            "next",
            "prev",
        ):
            if field in raw:
                annotation[field] = raw[field]

        instance_token = raw.get("instance_token")
        if isinstance(instance_token, str):
            instance = instance_by_token.get(instance_token, {})
            category_token = instance.get("category_token")
            if isinstance(category_token, str):
                category = category_by_token.get(category_token, {})
                category_name = category.get("name")
                if isinstance(category_name, str):
                    annotation["category_name"] = category_name

        attribute_names: list[str] = []
        attribute_tokens = raw.get("attribute_tokens", [])
        if isinstance(attribute_tokens, list):
            for attribute_token in attribute_tokens:
                if not isinstance(attribute_token, str):
                    continue
                attribute = attribute_by_token.get(attribute_token, {})
                name = attribute.get("name")
                if isinstance(name, str):
                    attribute_names.append(name)
        annotation["attribute_names"] = attribute_names
        annotation["track"] = collect_annotation_track(
            raw,
            annotation_by_token,
            sample_by_token,
            start_timestamp=start_timestamp,
            future_horizon_s=future_horizon_s,
        )
        annotations.append(annotation)
    return annotations


def collect_annotation_track(
    annotation: dict[str, Any],
    annotation_by_token: dict[str, dict[str, Any]],
    sample_by_token: dict[str, dict[str, Any]],
    *,
    start_timestamp: Any,
    future_horizon_s: float,
) -> list[dict[str, Any]]:
    """沿 ``sample_annotation.next`` 读取不超过 horizon 的真实位置。"""
    if not is_number(start_timestamp):
        return []
    start_timestamp_value = float(cast(int | float, start_timestamp))
    track: list[dict[str, Any]] = []
    current = annotation
    while True:
        sample_token = current.get("sample_token")
        sample = sample_by_token.get(sample_token) if isinstance(sample_token, str) else None
        timestamp = sample.get("timestamp") if sample else None
        if not is_number(timestamp):
            break
        elapsed_s = (
            float(cast(int | float, timestamp)) - start_timestamp_value
        ) / 1_000_000.0
        if elapsed_s > future_horizon_s + 1e-9:
            break
        translation = current.get("translation")
        if isinstance(translation, list | tuple) and len(translation) >= 2:
            track.append({
                "t": elapsed_s,
                "translation": [float(translation[0]), float(translation[1])],
            })
        next_token = current.get("next")
        if not isinstance(next_token, str) or not next_token:
            break
        next_annotation = annotation_by_token.get(next_token)
        if next_annotation is None:
            break
        current = next_annotation
    return track


def build_map_index(map_data: dict[str, Any] | None) -> dict[str, Any] | None:
    """为 Map Expansion 建立 node/polygon/token 索引。"""
    if map_data is None:
        return None

    nodes = map_data.get("node", [])
    polygons = map_data.get("polygon", [])
    if not isinstance(nodes, list) or not isinstance(polygons, list):
        return None

    node_by_token = build_token_index(
        [node for node in nodes if isinstance(node, dict)],
        "map.node",
    )
    polygon_by_token = build_token_index(
        [polygon for polygon in polygons if isinstance(polygon, dict)],
        "map.polygon",
    )
    return {
        "node_by_token": node_by_token,
        "polygon_by_token": polygon_by_token,
        "layers": map_data,
    }


def polygon_xy(
    polygon_token: str,
    map_index: dict[str, Any],
) -> list[tuple[float, float]]:
    """把 Map Expansion polygon token 转成外边界二维坐标。"""
    polygon = map_index["polygon_by_token"].get(polygon_token)
    if not isinstance(polygon, dict):
        return []

    node_tokens = polygon.get("exterior_node_tokens", [])
    if not isinstance(node_tokens, list):
        return []

    points: list[tuple[float, float]] = []
    node_by_token = map_index["node_by_token"]
    for node_token in node_tokens:
        if not isinstance(node_token, str):
            continue
        node = node_by_token.get(node_token)
        if not isinstance(node, dict):
            continue
        x = node.get("x")
        y = node.get("y")
        if is_number(x) and is_number(y):
            points.append((float(cast(int | float, x)), float(cast(int | float, y))))
    return points


def record_points(
    layer_name: str,
    record: dict[str, Any],
    map_index: dict[str, Any],
) -> list[tuple[float, float]]:
    """提取地图记录的代表点，用于半径筛选。"""
    if layer_name == "traffic_light":
        pose = record.get("pose", {})
        if isinstance(pose, dict) and is_number(pose.get("tx")) and is_number(pose.get("ty")):
            return [(float(pose["tx"]), float(pose["ty"]))]
        return []

    polygon_tokens: list[str] = []
    polygon_token = record.get("polygon_token")
    if isinstance(polygon_token, str):
        polygon_tokens.append(polygon_token)
    raw_polygon_tokens = record.get("polygon_tokens")
    if isinstance(raw_polygon_tokens, list):
        polygon_tokens.extend(
            token for token in raw_polygon_tokens if isinstance(token, str)
        )
    return [
        point
        for token in polygon_tokens
        for point in polygon_xy(token, map_index)
    ]


def min_distance_to_points(
    origin_xy: tuple[float, float],
    points: list[tuple[float, float]],
) -> float | None:
    """计算 ego 到一组地图点的最小距离。"""
    if not points:
        return None
    ox, oy = origin_xy
    return min(math.hypot(x - ox, y - oy) for x, y in points)


def point_in_polygon(
    point: tuple[float, float],
    polygon: list[tuple[float, float]],
) -> bool:
    """判断点是否位于简单多边形内；边界点视为在内。"""
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross) <= 1e-9 and (
            min(x1, x2) - 1e-9 <= x <= max(x1, x2) + 1e-9
            and min(y1, y2) - 1e-9 <= y <= max(y1, y2) + 1e-9
        ):
            return True
        if (y2 > y) != (y1 > y):
            crossing_x = (x1 - x2) * (y - y2) / (y1 - y2) + x2
            if x < crossing_x:
                inside = not inside
        previous = current
    return inside


def path_coverage_origins(
    ego_poses: list[dict[str, Any]],
    path_ego_poses: list[dict[str, Any]] | None,
    spacing_m: float,
) -> list[tuple[float, float]]:
    """沿候选路径抽取地图覆盖锚点，始终包含首尾有效位置。"""
    positions: list[tuple[float, float]] = []
    for pose in path_ego_poses or ego_poses:
        translation = pose.get("translation", [])
        if (
            not isinstance(translation, list)
            or len(translation) < 2
            or not is_number(translation[0])
            or not is_number(translation[1])
        ):
            continue
        positions.append((float(translation[0]), float(translation[1])))
    if not positions:
        return []

    origins = [positions[0]]
    for current in positions[1:-1]:
        if math.dist(origins[-1], current) >= spacing_m:
            origins.append(current)
    if positions[-1] != origins[-1]:
        origins.append(positions[-1])
    return origins


def min_distance_from_origins(
    origins: list[tuple[float, float]],
    points: list[tuple[float, float]],
) -> float | None:
    """返回一组覆盖锚点到地图点集的最小距离。"""
    distances = [min_distance_to_points(origin, points) for origin in origins]
    return min((distance for distance in distances if distance is not None), default=None)


def map_record_with_geometry(
    layer_name: str,
    record: dict[str, Any],
    points: list[tuple[float, float]],
    distance_m: float,
    map_index: dict[str, Any],
) -> dict[str, Any]:
    """复制地图记录，并补充 polygon_xy / distance_from_ego_m。"""
    item = dict(record)
    if layer_name != "traffic_light":
        polygon_tokens = record.get("polygon_tokens")
        if layer_name == "drivable_area" and isinstance(polygon_tokens, list):
            item["polygon_xy_list"] = [
                polygon
                for token in polygon_tokens
                if isinstance(token, str)
                for polygon in [polygon_xy(token, map_index)]
                if polygon
            ]
        else:
            item["polygon_xy"] = points
    item["distance_from_ego_m"] = distance_m
    return item


def collect_map_records(
    ego_poses: list[dict[str, Any]],
    map_index: dict[str, Any] | None,
    radius_m: float,
    *,
    path_ego_poses: list[dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """按关键帧收集地图记录；drivable_area 额外覆盖完整候选路径。"""
    if map_index is None or not ego_poses:
        return {}

    translation = ego_poses[0].get("translation", [])
    if not isinstance(translation, list) or len(translation) < 2:
        return {}
    if not is_number(translation[0]) or not is_number(translation[1]):
        return {}

    ego_xy = (float(translation[0]), float(translation[1]))
    path_origins = path_coverage_origins(
        ego_poses,
        path_ego_poses,
        spacing_m=max(radius_m / 2.0, 1.0),
    )
    layers = map_index["layers"]
    result: dict[str, list[dict[str, Any]]] = {}
    for layer_name in MAP_LAYERS:
        records = layers.get(layer_name, [])
        if not isinstance(records, list):
            continue

        selected: list[dict[str, Any]] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            # 当前 SceneFactExtractor 只根据 road_segment 是否存在判断路口；
            # 因此这里只保留真实 intersection road_segment，避免过度触发。
            if layer_name == "road_segment" and record.get("is_intersection") is not True:
                continue

            if layer_name == "drivable_area":
                polygon_tokens = record.get("polygon_tokens")
                if not isinstance(polygon_tokens, list):
                    continue
                nearby_polygons: list[list[tuple[float, float]]] = []
                nearby_distances: list[float] = []
                for token in polygon_tokens:
                    if not isinstance(token, str):
                        continue
                    polygon = polygon_xy(token, map_index)
                    distance = min_distance_to_points(ego_xy, polygon)
                    path_distance = min_distance_from_origins(path_origins, polygon)
                    covers_path = any(
                        point_in_polygon(origin, polygon) for origin in path_origins
                    )
                    if (
                        distance is not None
                        and (
                            covers_path
                            or (path_distance is not None and path_distance <= radius_m)
                        )
                    ):
                        nearby_polygons.append(polygon)
                        nearby_distances.append(distance)
                if nearby_polygons:
                    selected.append({
                        "token": record.get("token", ""),
                        "polygon_xy_list": nearby_polygons,
                        "distance_from_ego_m": min(nearby_distances),
                    })
                continue

            points = record_points(layer_name, record, map_index)
            distance = min_distance_to_points(ego_xy, points)
            if distance is None or distance > radius_m:
                continue
            selected.append(
                map_record_with_geometry(
                    layer_name,
                    record,
                    points,
                    distance,
                    map_index,
                )
            )

        selected.sort(key=lambda item: float(item.get("distance_from_ego_m", 0.0)))
        if selected:
            result[layer_name] = selected
    return result


def scene_sample_tokens(
    scene: dict[str, Any],
    sample_by_token: dict[str, dict[str, Any]],
    samples_by_scene: dict[str, list[dict[str, Any]]],
) -> list[str]:
    """按 sample.next 链返回 scene 内样本；断链时按 timestamp 兜底排序。"""
    scene_token = scene.get("token")
    first_sample_token = scene.get("first_sample_token")
    ordered: list[str] = []
    seen: set[str] = set()

    current_token = first_sample_token if isinstance(first_sample_token, str) else None
    while current_token and current_token not in seen:
        sample = sample_by_token.get(current_token)
        if not sample or sample.get("scene_token") != scene_token:
            break
        ordered.append(current_token)
        seen.add(current_token)
        next_token = sample.get("next")
        current_token = next_token if isinstance(next_token, str) and next_token else None

    if ordered:
        return ordered

    fallback = samples_by_scene.get(str(scene_token), [])
    fallback.sort(key=lambda row: int(row.get("timestamp", 0)))
    return [row["token"] for row in fallback if isinstance(row.get("token"), str)]


def extract_keywords(description: str) -> list[str]:
    """从 scene description 中抽取与设计文档一致的英文触发词。"""
    lowered = description.lower()
    return [keyword for keyword in SCENE_KEYWORDS if keyword in lowered]


def get_drivelm_frame(
    drivelm_data: dict[str, Any] | None,
    scene_token: str,
    sample_token: str,
) -> dict[str, Any] | None:
    """读取指定 scene/sample 对应的 DriveLM keyframe。"""
    if drivelm_data is None:
        return None
    scene_payload = drivelm_data.get(scene_token)
    if not isinstance(scene_payload, dict):
        return None
    frames = scene_payload.get("key_frames", {})
    if not isinstance(frames, dict):
        return None
    frame = frames.get(sample_token)
    return frame if isinstance(frame, dict) else None


def drivelm_sample_tokens_for_scene(
    drivelm_data: dict[str, Any] | None,
    scene_token: str,
    sample_by_token: dict[str, dict[str, Any]],
) -> list[str] | None:
    """有 DriveLM 时返回该 scene 的 keyframe tokens，并按 sample timestamp 排序。"""
    if drivelm_data is None:
        return None
    scene_payload = drivelm_data.get(scene_token)
    if not isinstance(scene_payload, dict):
        return []
    frames = scene_payload.get("key_frames", {})
    if not isinstance(frames, dict):
        return []

    tokens = [token for token in frames if isinstance(token, str) and token in sample_by_token]
    tokens.sort(key=lambda token: int(sample_by_token[token].get("timestamp", 0)))
    return tokens


def build_raw_json(
    scene: dict[str, Any],
    sample: dict[str, Any],
    drivelm_scene: dict[str, Any] | None,
    drivelm_frame: dict[str, Any] | None,
) -> dict[str, Any]:
    """构造 RawScene.raw_json，优先使用真实 DriveLM QA 和关键对象。"""
    description = scene.get("description")
    if not isinstance(description, str):
        description = ""

    drivelm_description = ""
    if isinstance(drivelm_scene, dict):
        raw_drivelm_description = drivelm_scene.get("scene_description", "")
        if isinstance(raw_drivelm_description, str):
            drivelm_description = raw_drivelm_description

    qa_pairs = flatten_drivelm_qa(drivelm_frame)
    if not qa_pairs and description:
        qa_pairs.append(
            {
                "category": "perception",
                "question": "nuScenes scene description",
                "answer": description,
            }
        )

    combined_description = " ".join(
        text for text in (drivelm_description, description) if text
    )
    return {
        "scene_name": scene.get("name", ""),
        "scene_description": description,
        "drivelm_scene_description": drivelm_description,
        "sample_timestamp": sample.get("timestamp"),
        "qa_pairs": qa_pairs,
        "keywords": extract_keywords(combined_description),
        "key_object_infos": normalize_drivelm_key_object_infos(drivelm_frame),
    }


def build_raw_scene_dataset(
    source_root: Path,
    *,
    drivelm_data: dict[str, Any] | None = None,
    drivelm_source_file: Path | None = None,
    map_index: dict[str, Any] | None = None,
    map_radius_m: float = DEFAULT_MAP_RADIUS_M,
    future_horizon_s: float = 6.0,
    path_horizon_s: float = 12.0,
    min_poses: int = 20,
) -> list[dict[str, Any]]:
    """整合原始表，输出 RawScene.model_validate 可读取的记录列表。"""
    if path_horizon_s < future_horizon_s:
        raise ValueError("path_horizon_s must be at least future_horizon_s")
    tables = {name: load_table(source_root, name) for name in REQUIRED_TABLES}

    log_by_token = build_token_index(tables["log"], "log")
    sample_by_token = build_token_index(tables["sample"], "sample")
    sample_data_by_token = build_token_index(tables["sample_data"], "sample_data")
    calibrated_sensor_by_token = build_token_index(tables["calibrated_sensor"], "calibrated_sensor")
    sensor_by_token = build_token_index(tables["sensor"], "sensor")
    ego_pose_by_token = build_token_index(tables["ego_pose"], "ego_pose")
    instance_by_token = build_token_index(tables["instance"], "instance")
    category_by_token = build_token_index(tables["category"], "category")
    attribute_by_token = build_token_index(tables["attribute"], "attribute")

    samples_by_scene = group_by_field(tables["sample"], "scene_token")
    annotations_by_sample = group_by_field(tables["sample_annotation"], "sample_token")
    annotation_by_token = build_token_index(
        tables["sample_annotation"], "sample_annotation"
    )
    lidar_by_sample = build_lidar_index(
        tables["sample_data"],
        calibrated_sensor_by_token,
        sensor_by_token,
    )

    records: list[dict[str, Any]] = []
    scenes = sorted(tables["scene"], key=lambda row: str(row.get("name", row.get("token", ""))))
    for scene in scenes:
        scene_token = scene.get("token")
        log_token = scene.get("log_token")
        if not isinstance(scene_token, str) or not isinstance(log_token, str):
            continue
        log = log_by_token.get(log_token, {})
        if log.get("location") != TARGET_LOCATION:
            continue

        drivelm_tokens = drivelm_sample_tokens_for_scene(
            drivelm_data,
            scene_token,
            sample_by_token,
        )
        if drivelm_tokens == []:
            continue
        sample_tokens = (
            drivelm_tokens
            if drivelm_tokens is not None
            else scene_sample_tokens(scene, sample_by_token, samples_by_scene)
        )
        drivelm_scene = drivelm_data.get(scene_token) if drivelm_data else None
        if not isinstance(drivelm_scene, dict):
            drivelm_scene = None

        for sample_token in sample_tokens:
            sample = sample_by_token.get(sample_token)
            if not sample:
                continue
            drivelm_frame = get_drivelm_frame(drivelm_data, scene_token, sample_token)

            ego_poses = collect_ego_poses(
                sample_token,
                sample_by_token,
                sample_data_by_token,
                lidar_by_sample,
                ego_pose_by_token,
                calibrated_sensor_by_token,
                sensor_by_token,
                future_horizon_s,
            )
            if len(ego_poses) < min_poses:
                continue
            path_ego_poses = collect_ego_poses(
                sample_token,
                sample_by_token,
                sample_data_by_token,
                lidar_by_sample,
                ego_pose_by_token,
                calibrated_sensor_by_token,
                sensor_by_token,
                path_horizon_s,
            )

            records.append(
                {
                    "scene_id": scene_token,
                    "frame_token": sample_token,
                    "source_file": str(drivelm_source_file or source_root),
                    "raw_json": build_raw_json(scene, sample, drivelm_scene, drivelm_frame),
                    "location": TARGET_LOCATION,
                    "ego_poses": ego_poses,
                    "path_ego_poses": path_ego_poses,
                    "map_records": collect_map_records(
                        ego_poses,
                        map_index,
                        map_radius_m,
                        path_ego_poses=path_ego_poses,
                    ),
                    "annotations": collect_annotations(
                        sample_token,
                        annotations_by_sample,
                        annotation_by_token,
                        sample_by_token,
                        instance_by_token,
                        category_by_token,
                        attribute_by_token,
                        future_horizon_s,
                    ),
                    "keyframe_index": 0,
                }
            )

    return records


def dump_dataset(records: list[dict[str, Any]], output: Path, output_format: str) -> None:
    """按 JSON 或 YAML 写出数据集。"""
    output.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        with output.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
            f.write("\n")
        return

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("输出 YAML 需要安装 PyYAML") from exc

    with output.open("w", encoding="utf-8") as f:
        yaml.safe_dump(records, f, allow_unicode=True, sort_keys=False)


def infer_output_format(output: Path, requested: str) -> str:
    """根据参数或后缀推断输出格式。"""
    if requested != "auto":
        return requested
    return "yaml" if output.suffix.lower() in {".yaml", ".yml"} else "json"


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="整合 nuScenes 分散 JSON 表，生成 RawScene 兼容数据集。",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="nuScenes JSON 表所在目录，默认 v1.0-mini。",
    )
    parser.add_argument(
        "--drivelm-qa-path",
        type=Path,
        default=None,
        help="DriveLM QA JSON 路径；默认自动读取 source-root/v1_0_train_nus.json。",
    )
    parser.add_argument(
        "--map-expansion-path",
        type=Path,
        default=None,
        help="Map Expansion JSON 路径；默认自动读取 source-root 下的 boston-seaport.json。",
    )
    parser.add_argument(
        "--map-radius-m",
        type=float,
        default=DEFAULT_MAP_RADIUS_M,
        help="按关键帧 ego 位置筛选地图记录的半径，默认 50 米。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="输出 JSON/YAML 文件路径。",
    )
    parser.add_argument(
        "--format",
        choices=("auto", "json", "yaml"),
        default="auto",
        help="输出格式；auto 根据文件后缀判断。",
    )
    parser.add_argument(
        "--horizon-s",
        type=float,
        default=6.0,
        help="从关键帧向后采集 ego_pose 的秒数窗口。",
    )
    parser.add_argument(
        "--path-horizon-s",
        type=float,
        default=12.0,
        help="候选合成使用的真实路径素材窗口，默认 12 秒。",
    )
    parser.add_argument(
        "--min-poses",
        type=int,
        default=20,
        help="保留样本所需的最少 ego_pose 数量。",
    )
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""
    args = parse_args()
    if args.horizon_s <= 0:
        raise SystemExit("--horizon-s 必须大于 0")
    if args.path_horizon_s < args.horizon_s:
        raise SystemExit("--path-horizon-s 必须不小于 --horizon-s")
    if args.min_poses < 2:
        raise SystemExit("--min-poses 必须至少为 2")
    if args.map_radius_m <= 0:
        raise SystemExit("--map-radius-m 必须大于 0")

    drivelm_path = args.drivelm_qa_path
    if drivelm_path is None:
        default_drivelm_path = args.source_root / DEFAULT_DRIVELM_QA_FILENAME
        drivelm_path = default_drivelm_path if default_drivelm_path.exists() else None
    drivelm_data = load_drivelm_data(drivelm_path)

    map_path = args.map_expansion_path
    if map_path is None:
        default_map_path = args.source_root / DEFAULT_MAP_EXPANSION_RELATIVE_PATH
        map_path = default_map_path if default_map_path.exists() else None
    map_index = build_map_index(load_map_data(map_path))

    records = build_raw_scene_dataset(
        args.source_root,
        drivelm_data=drivelm_data,
        drivelm_source_file=drivelm_path,
        map_index=map_index,
        map_radius_m=args.map_radius_m,
        future_horizon_s=args.horizon_s,
        path_horizon_s=args.path_horizon_s,
        min_poses=args.min_poses,
    )
    if not records:
        raise SystemExit("未生成任何合规 RawScene 记录，请检查数据路径和筛选条件")

    output_format = infer_output_format(args.output, args.format)
    dump_dataset(records, args.output, output_format)
    source_mode = "nuScenes + DriveLM" if drivelm_data is not None else "nuScenes-only"
    map_mode = " + Map Expansion" if map_index is not None else ""
    print(f"已输出 {len(records)} 条 {source_mode}{map_mode} RawScene 兼容记录: {args.output}")


if __name__ == "__main__":
    main()
