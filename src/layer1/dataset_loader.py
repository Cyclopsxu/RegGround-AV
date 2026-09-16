"""DriveLM-nuScenes 与 Map Expansion 联合加载器。"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from nuscenes.nuscenes import NuScenes

try:  # nuscenes-devkit 在轻量测试环境中可能缺 Map Expansion 资源。
    from nuscenes.map_expansion.map_api import NuScenesMap
except Exception:  # pragma: no cover - import fallback
    NuScenesMap = None

from src.layer1.drivelm_adapter import normalize_drivelm_frame
from src.layer1.exceptions import DatasetNotFoundError, SceneNotFoundError
from src.layer1.models import RawScene


class DriveLMDatasetLoader:
    """加载 Boston-only DriveLM keyframe，并预读密集 LIDAR_TOP ego pose。"""

    _MAP_LAYERS: tuple[str, ...] = (
        "stop_line",
        "ped_crossing",
        "traffic_light",
        "road_segment",
        "lane",
        "drivable_area",
    )

    def __init__(
        self,
        nuscenes_root: Path,
        drivelm_qa_path: Path,
        *,
        map_name: Literal["boston-seaport"] = "boston-seaport",
        nuscenes_version: str = "v1.0-trainval",
        future_horizon_s: float = 6.0,
        path_horizon_s: float = 12.0,
    ) -> None:
        """初始化 loader，并构建 Boston scene 与 LIDAR_TOP 索引。"""
        if not nuscenes_root.exists() or not nuscenes_root.is_dir():
            raise DatasetNotFoundError(f"nuScenes 根目录不存在: {nuscenes_root}")
        if not drivelm_qa_path.exists() or not drivelm_qa_path.is_file():
            raise DatasetNotFoundError(f"DriveLM QA 文件不存在: {drivelm_qa_path}")
        if future_horizon_s <= 0:
            raise ValueError("future_horizon_s must be positive")
        if path_horizon_s < future_horizon_s:
            raise ValueError("path_horizon_s must be at least future_horizon_s")

        self.nuscenes_root = nuscenes_root
        self.drivelm_qa_path = drivelm_qa_path
        self.nuscenes_version = nuscenes_version
        self.map_name = map_name
        self.future_horizon_s = future_horizon_s
        self.path_horizon_s = path_horizon_s
        self.target_channel = "LIDAR_TOP"

        self._nusc = NuScenes(version=nuscenes_version, dataroot=str(nuscenes_root))
        self._boston_scene_tokens = self._build_boston_scene_whitelist()
        self._qa_data = self._load_qa_json(drivelm_qa_path)
        self._nusc_map = self._init_map()
        self._lidar_sample_data_by_sample = self._build_lidar_sample_data_index()
        self._keyframes = self._build_drivelm_keyframe_index(self._qa_data)

    # ------------------------------------------------------------------
    # v3.1 public API
    # ------------------------------------------------------------------

    def list_keyframes(self) -> list[tuple[str, str]]:
        """列出 Boston ∩ DriveLM 的 keyframe `(scene_token, sample_token)`。"""
        return list(self._keyframes.keys())

    def drivelm_keyframe_counts_by_location(self) -> dict[str, int]:
        """统计全部 DriveLM 覆盖 keyframe 在各 nuScenes location 的数量。"""
        counts: dict[str, int] = {}
        if not isinstance(self._qa_data, dict):
            return counts
        for scene in self._nusc.scene:
            scene_token = scene.get("token")
            if not isinstance(scene_token, str):
                continue
            payload = self._qa_data.get(scene_token)
            if not isinstance(payload, dict):
                continue
            log_token = scene.get("log_token")
            if not isinstance(log_token, str):
                continue
            log = self._nusc.get("log", log_token)
            location = str(log.get("location", "unknown"))
            keyframes: dict[tuple[str, str], dict[str, Any]] = {}
            self._index_scene_level_keyframes(keyframes, scene_token, payload)
            counts[location] = counts.get(location, 0) + len(keyframes)
        return dict(sorted(counts.items()))

    def load_keyframe(self, scene_token: str, frame_token: str) -> RawScene:
        """加载一个 Boston keyframe 为 RawScene。"""
        if scene_token not in self._boston_scene_tokens:
            raise SceneNotFoundError(f"非 Boston scene 或 scene_token 不存在: {scene_token}")

        raw_json = normalize_drivelm_frame(self._lookup_qa(scene_token, frame_token))
        scene = self._get_scene(scene_token)
        sample = self._get_sample(frame_token)
        if sample.get("scene_token") != scene_token:
            raise SceneNotFoundError(
                f"frame_token 不属于 scene_token: {frame_token} not in {scene_token}"
            )

        ego_poses = self._collect_ego_poses(frame_token, self.future_horizon_s)
        path_ego_poses = self._collect_ego_poses(frame_token, self.path_horizon_s)
        annotations = self._collect_annotations(sample)
        map_records = self._collect_map_records(ego_poses)

        return RawScene(
            scene_id=scene["token"],
            frame_token=frame_token,
            source_file=self.drivelm_qa_path,
            raw_json=raw_json,
            location=self.map_name,
            ego_poses=ego_poses,
            path_ego_poses=path_ego_poses,
            keyframe_index=0,
            map_records=map_records,
            annotations=annotations,
        )

    def iter_keyframes(self) -> Iterator[RawScene]:
        """遍历 Boston ∩ DriveLM keyframes。"""
        for scene_token, frame_token in self.list_keyframes():
            yield self.load_keyframe(scene_token, frame_token)

    # ------------------------------------------------------------------
    # Boston-only and DriveLM QA indexing
    # ------------------------------------------------------------------

    def _build_boston_scene_whitelist(self) -> set[str]:
        whitelist: set[str] = set()
        for scene in self._nusc.scene:
            token = scene.get("token")
            if not token:
                continue
            log_token = scene.get("log_token")
            if not log_token:
                continue
            try:
                log = self._nusc.get("log", log_token)
            except (KeyError, ValueError):
                continue
            if not isinstance(log, dict):
                continue
            if log.get("location") == self.map_name:
                whitelist.add(token)
        return whitelist

    def _build_drivelm_keyframe_index(
        self,
        qa_data: Any,
    ) -> dict[tuple[str, str], dict[str, Any]]:
        keyframes: dict[tuple[str, str], dict[str, Any]] = {}

        if isinstance(qa_data, dict):
            for scene_token, scene_payload in qa_data.items():
                if scene_token not in self._boston_scene_tokens:
                    continue
                if isinstance(scene_payload, dict):
                    self._index_scene_level_keyframes(keyframes, scene_token, scene_payload)

        return keyframes

    def _index_scene_level_keyframes(
        self,
        keyframes: dict[tuple[str, str], dict[str, Any]],
        scene_token: str,
        scene_payload: dict[str, Any],
    ) -> None:
        frames = scene_payload.get("key_frames", {})
        if isinstance(frames, dict):
            for frame_key, frame_payload in frames.items():
                if not isinstance(frame_payload, dict):
                    continue
                frame_token = self._frame_token_from_payload(frame_key, frame_payload)
                keyframes[(scene_token, frame_token)] = frame_payload
        elif isinstance(frames, list):
            for frame_payload in frames:
                if not isinstance(frame_payload, dict):
                    continue
                frame_token = self._frame_token_from_payload("", frame_payload)
                if frame_token:
                    keyframes[(scene_token, frame_token)] = frame_payload

    @staticmethod
    def _frame_token_from_payload(default: str, payload: dict[str, Any]) -> str:
        for key in ("sample_token", "frame_token", "token"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
        return default

    # ------------------------------------------------------------------
    # Ego pose reconstruction input
    # ------------------------------------------------------------------

    def _build_lidar_sample_data_index(self) -> dict[str, dict[str, Any]]:
        by_sample: dict[str, dict[str, Any]] = {}
        for sd in getattr(self._nusc, "sample_data", []):
            if not isinstance(sd, dict):
                continue
            if self._sample_data_channel(sd) != self.target_channel:
                continue
            sample_token = sd.get("sample_token")
            if isinstance(sample_token, str):
                by_sample[sample_token] = sd
        return by_sample

    def _collect_ego_poses(
        self,
        start_sample_token: str,
        horizon_s: float,
    ) -> list[dict[str, Any]]:
        poses: list[dict[str, Any]] = []
        current_sample_token: str | None = start_sample_token
        current_sd = self._get_lidar_sample_data_for_sample(start_sample_token)
        start_timestamp: int | None = None

        while current_sd is not None:
            pose = self._ego_pose_from_sample_data(current_sd)
            if pose is None:
                break
            if start_timestamp is None:
                start_timestamp = int(pose["timestamp"])
            if pose["timestamp"] - start_timestamp > horizon_s * 1_000_000:
                break

            poses.append(pose)
            if len(poses) >= 400:
                break

            next_sd_token = current_sd.get("next")
            if isinstance(next_sd_token, str) and next_sd_token:
                try:
                    current_sd = self._nusc.get("sample_data", next_sd_token)
                    continue
                except (KeyError, ValueError):
                    break

            current_sample_token = self._next_sample_token(current_sample_token)
            if current_sample_token is None:
                break
            current_sd = self._get_lidar_sample_data_for_sample(current_sample_token)

        return poses

    def _get_lidar_sample_data_for_sample(self, sample_token: str) -> dict[str, Any] | None:
        if sample_token in self._lidar_sample_data_by_sample:
            return self._lidar_sample_data_by_sample[sample_token]
        try:
            sample = self._nusc.get("sample", sample_token)
        except (KeyError, ValueError):
            return None
        sd_token = sample.get("data", {}).get(self.target_channel)
        if not isinstance(sd_token, str):
            return None
        try:
            sample_data = self._nusc.get("sample_data", sd_token)
        except (KeyError, ValueError):
            return None
        return sample_data if isinstance(sample_data, dict) else None

    def _ego_pose_from_sample_data(self, sample_data: dict[str, Any]) -> dict[str, Any] | None:
        ep_token = sample_data.get("ego_pose_token")
        if not isinstance(ep_token, str):
            return None
        try:
            ego_pose = self._nusc.get("ego_pose", ep_token)
        except (KeyError, ValueError):
            return None
        if not isinstance(ego_pose, dict):
            return None
        timestamp = ego_pose.get("timestamp", sample_data.get("timestamp"))
        if timestamp is None:
            return None
        return {
            "token": ego_pose.get("token", ep_token),
            "timestamp": int(timestamp),
            "rotation": list(ego_pose["rotation"]),
            "translation": list(ego_pose["translation"]),
        }

    def _sample_data_channel(self, sample_data: dict[str, Any]) -> str | None:
        channel = sample_data.get("channel")
        if isinstance(channel, str):
            return channel
        calib_token = sample_data.get("calibrated_sensor_token")
        if not isinstance(calib_token, str):
            return None
        try:
            calib = self._nusc.get("calibrated_sensor", calib_token)
            if not isinstance(calib, dict):
                return None
            sensor = self._nusc.get("sensor", calib["sensor_token"])
        except (KeyError, ValueError):
            return None
        if not isinstance(sensor, dict):
            return None
        channel = sensor.get("channel")
        return channel if isinstance(channel, str) else None

    # ------------------------------------------------------------------
    # Map and annotation summaries
    # ------------------------------------------------------------------

    def _init_map(self) -> Any | None:
        if NuScenesMap is None:
            return None
        try:
            return NuScenesMap(dataroot=str(self.nuscenes_root), map_name=self.map_name)
        except Exception:
            return None

    def _collect_map_records(
        self,
        ego_poses: list[dict[str, Any]],
        radius_m: float = 50.0,
    ) -> dict[str, list[dict[str, Any]]]:
        if self._nusc_map is None or not ego_poses:
            return {}
        x, y = ego_poses[0]["translation"][:2]
        try:
            records = self._nusc_map.get_records_in_radius(
                x=float(x),
                y=float(y),
                radius=radius_m,
                layer_names=list(self._MAP_LAYERS),
            )
        except Exception:
            return {}

        result: dict[str, list[dict[str, Any]]] = {}
        for layer_name, tokens in records.items():
            if layer_name not in self._MAP_LAYERS or not isinstance(tokens, list):
                continue
            layer_records: list[dict[str, Any]] = []
            for token in tokens:
                record = self._map_record(layer_name, token)
                if record is not None:
                    layer_records.append(record)
            result[layer_name] = layer_records
        return result

    def _map_record(self, layer_name: str, token: str) -> dict[str, Any] | None:
        nusc_map = self._nusc_map
        if nusc_map is None:
            return None
        try:
            raw_record = nusc_map.get(layer_name, token)
            record = dict(raw_record) if isinstance(raw_record, dict) else {"token": token}
        except Exception:
            record = {"token": token}

        polygon_token = record.get("polygon_token")
        if isinstance(polygon_token, str):
            polygon_xy = self._extract_polygon_xy(polygon_token)
            if polygon_xy:
                record["polygon_xy"] = polygon_xy
        polygon_tokens = record.get("polygon_tokens")
        if isinstance(polygon_tokens, list):
            polygon_xy_list = [
                polygon_xy
                for item in polygon_tokens
                if isinstance(item, str)
                for polygon_xy in [self._extract_polygon_xy(item)]
                if polygon_xy
            ]
            if polygon_xy_list:
                record["polygon_xy_list"] = polygon_xy_list
        return record

    def _extract_polygon_xy(self, polygon_token: str) -> list[tuple[float, float]]:
        nusc_map = self._nusc_map
        if nusc_map is None:
            return []
        try:
            polygon = nusc_map.extract_polygon(polygon_token)
            coords = list(polygon.exterior.coords)
        except Exception:
            return []
        return [(float(x), float(y)) for x, y, *_ in coords]

    def _collect_annotations(self, sample: dict[str, Any]) -> list[dict[str, Any]]:
        annotations: list[dict[str, Any]] = []
        ann_tokens = sample.get("anns", [])
        if not isinstance(ann_tokens, list):
            return annotations

        for ann_token in ann_tokens:
            if not isinstance(ann_token, str):
                continue
            try:
                ann = dict(self._nusc.get("sample_annotation", ann_token))
            except (KeyError, ValueError):
                continue
            ann["attribute_names"] = self._attribute_names(ann.get("attribute_tokens", []))
            ann["track"] = self._annotation_track(ann, sample)
            annotations.append(ann)
        return annotations

    def _annotation_track(
        self,
        annotation: dict[str, Any],
        start_sample: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """沿 sample_annotation.next 收集最多 6 秒的智能体位置。"""
        start_timestamp = start_sample.get("timestamp")
        if not isinstance(start_timestamp, int | float):
            return []
        track: list[dict[str, Any]] = []
        current = annotation
        while True:
            sample_token = current.get("sample_token")
            try:
                sample = self._nusc.get("sample", sample_token)
            except (KeyError, ValueError, TypeError):
                break
            timestamp = sample.get("timestamp") if isinstance(sample, dict) else None
            translation = current.get("translation")
            if not isinstance(timestamp, int | float):
                break
            if timestamp - start_timestamp > self.future_horizon_s * 1_000_000:
                break
            if isinstance(translation, list | tuple) and len(translation) >= 2:
                track.append({
                    "t": float(timestamp - start_timestamp) / 1_000_000.0,
                    "translation": [float(translation[0]), float(translation[1])],
                })
            next_token = current.get("next")
            if not isinstance(next_token, str) or not next_token:
                break
            try:
                next_annotation = self._nusc.get("sample_annotation", next_token)
            except (KeyError, ValueError):
                break
            if not isinstance(next_annotation, dict):
                break
            current = next_annotation
        return track

    def _attribute_names(self, attr_tokens: Any) -> list[str]:
        if not isinstance(attr_tokens, list):
            return []
        names: list[str] = []
        for attr_token in attr_tokens:
            if not isinstance(attr_token, str):
                continue
            try:
                attr = self._nusc.get("attribute", attr_token)
            except (KeyError, ValueError):
                continue
            name = attr.get("name")
            if isinstance(name, str):
                names.append(name)
        return names

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def _get_scene(self, scene_token: str) -> dict[str, Any]:
        try:
            scene = self._nusc.get("scene", scene_token)
        except (KeyError, ValueError) as exc:
            raise SceneNotFoundError(f"scene_token 不存在: {scene_token}") from exc
        if not isinstance(scene, dict):
            raise SceneNotFoundError(f"scene_token 数据非法: {scene_token}")
        return scene

    def _get_sample(self, frame_token: str) -> dict[str, Any]:
        try:
            sample = self._nusc.get("sample", frame_token)
        except (KeyError, ValueError) as exc:
            raise SceneNotFoundError(f"frame_token 不存在: {frame_token}") from exc
        if not isinstance(sample, dict):
            raise SceneNotFoundError(f"frame_token 数据非法: {frame_token}")
        return sample

    def _next_sample_token(self, sample_token: str | None) -> str | None:
        if not sample_token:
            return None
        try:
            sample = self._nusc.get("sample", sample_token)
        except (KeyError, ValueError):
            return None
        if not isinstance(sample, dict):
            return None
        next_token = sample.get("next")
        return next_token if isinstance(next_token, str) and next_token else None

    def _load_qa_json(self, path: Path) -> Any:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _lookup_qa(self, scene_token: str, frame_token: str) -> dict[str, Any]:
        try:
            return self._keyframes[(scene_token, frame_token)]
        except KeyError as exc:
            raise SceneNotFoundError(
                f"DriveLM keyframe 不存在或已被 Boston-only 过滤: {scene_token}/{frame_token}"
            ) from exc
