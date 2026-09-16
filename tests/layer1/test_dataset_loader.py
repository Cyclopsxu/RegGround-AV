"""DriveLM/nuScenes loader tests for the Layer 1 v3.1 contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.layer1 import dataset_loader
from src.layer1.exceptions import DatasetNotFoundError, SceneNotFoundError
from src.layer1.models import RawScene
from src.layer1.scene_fact_extractor import SceneFactExtractor
from src.layer1.scene_text_extractor import SceneTextExtractor


class FakeNuScenes:
    """Small in-memory NuScenes table store used by loader unit tests."""

    def __init__(self, version: str, dataroot: str):
        self.version = version
        self.dataroot = dataroot
        self.scene: list[dict[str, Any]] = []
        self.sample_data: list[dict[str, Any]] = []
        self._tables: dict[str, dict[str, dict[str, Any]]] = {}

    def add(self, table: str, token: str, data: dict[str, Any]) -> dict[str, Any]:
        record = dict(data)
        record.setdefault("token", token)
        self._tables.setdefault(table, {})[token] = record
        if table == "scene":
            self.scene.append(record)
        if table == "sample_data":
            self.sample_data.append(record)
        return record

    def get(self, table: str, token: str) -> dict[str, Any]:
        return self._tables[table][token]


def _install_fake_nusc(monkeypatch: pytest.MonkeyPatch, fake: FakeNuScenes) -> None:
    monkeypatch.setattr(dataset_loader, "NuScenes", lambda version, dataroot: fake)
    monkeypatch.setattr(dataset_loader, "NuScenesMap", None)


def _write_qa(tmp_path: Path, data: dict[str, Any]) -> Path:
    qa_path = tmp_path / "drivelm_qa.json"
    qa_path.write_text(json.dumps(data), encoding="utf-8")
    return qa_path


def _add_scene(
    fake: FakeNuScenes,
    *,
    scene_token: str,
    location: str = "boston-seaport",
    n_dense_frames: int = 130,
    dt_us: int = 50_000,
    with_annotations: bool = False,
) -> str:
    log_token = f"log_{scene_token}"
    sample_token = f"sample_{scene_token}_0"
    fake.add("log", log_token, {"location": location})
    fake.add("scene", scene_token, {
        "log_token": log_token,
        "first_sample_token": sample_token,
        "last_sample_token": sample_token,
        "nbr_samples": 1,
    })

    ann_tokens: list[str] = []
    if with_annotations:
        fake.add("attribute", "attr_moving", {"name": "pedestrian.moving"})
        fake.add("sample_annotation", "ann_ped", {
            "category_name": "human.pedestrian.adult",
            "translation": [5.0, 1.0, 0.0],
            "attribute_tokens": ["attr_moving"],
        })
        ann_tokens.append("ann_ped")

    fake.add("sample", sample_token, {
        "scene_token": scene_token,
        "timestamp": 0,
        "data": {"LIDAR_TOP": f"sd_{scene_token}_0"},
        "anns": ann_tokens,
        "next": "",
    })

    for i in range(n_dense_frames):
        pose_token = f"pose_{scene_token}_{i}"
        sd_token = f"sd_{scene_token}_{i}"
        timestamp = i * dt_us
        fake.add("ego_pose", pose_token, {
            "timestamp": timestamp,
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "translation": [float(i) * 0.2, 0.0, 0.0],
        })
        fake.add("sample_data", sd_token, {
            "sample_token": sample_token if i == 0 else "",
            "ego_pose_token": pose_token,
            "timestamp": timestamp,
            "next": f"sd_{scene_token}_{i + 1}" if i + 1 < n_dense_frames else "",
            "channel": "LIDAR_TOP",
        })
    return sample_token


class TestDriveLMDatasetLoader:
    def test_nuscenes_root_not_exists_raises(self, tmp_path: Path) -> None:
        qa_path = _write_qa(tmp_path, {})
        with pytest.raises(DatasetNotFoundError):
            dataset_loader.DriveLMDatasetLoader(tmp_path / "missing", qa_path)

    def test_qa_file_not_exists_raises(self, tmp_path: Path) -> None:
        nusc_root = tmp_path / "nuscenes"
        nusc_root.mkdir()
        with pytest.raises(DatasetNotFoundError):
            dataset_loader.DriveLMDatasetLoader(nusc_root, tmp_path / "missing.json")

    def test_list_keyframes_filters_to_boston_drivelm_intersection(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        nusc_root = tmp_path / "nuscenes"
        nusc_root.mkdir()
        fake = FakeNuScenes("v1.0-trainval", str(nusc_root))
        boston_frame = _add_scene(fake, scene_token="scene_boston", location="boston-seaport")
        singapore_frame = _add_scene(
            fake,
            scene_token="scene_singapore",
            location="singapore-onenorth",
        )
        _install_fake_nusc(monkeypatch, fake)

        qa_path = _write_qa(tmp_path, {
            "scene_boston": {"key_frames": {
                boston_frame: {"sample_token": boston_frame, "qa_pairs": []},
            }},
            "scene_singapore": {"key_frames": {
                singapore_frame: {"sample_token": singapore_frame, "qa_pairs": []},
            }},
            "scene_not_in_nuscenes": {"key_frames": {
                "sample_missing": {"sample_token": "sample_missing", "qa_pairs": []},
            }},
        })

        loader = dataset_loader.DriveLMDatasetLoader(nusc_root, qa_path)

        assert loader.list_keyframes() == [("scene_boston", boston_frame)]
        assert loader.drivelm_keyframe_counts_by_location() == {
            "boston-seaport": 1,
            "singapore-onenorth": 1,
        }

    def test_scene_without_boston_log_is_excluded(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        nusc_root = tmp_path / "nuscenes"
        nusc_root.mkdir()
        fake = FakeNuScenes("v1.0-trainval", str(nusc_root))
        fake.add("scene", "scene_without_log", {
            "first_sample_token": "sample_without_log_0",
            "last_sample_token": "sample_without_log_0",
            "nbr_samples": 1,
        })
        _install_fake_nusc(monkeypatch, fake)
        qa_path = _write_qa(tmp_path, {
            "scene_without_log": {"key_frames": {
                "sample_without_log_0": {"sample_token": "sample_without_log_0"},
            }},
        })

        loader = dataset_loader.DriveLMDatasetLoader(nusc_root, qa_path)

        assert loader.list_keyframes() == []

    def test_load_keyframe_returns_dense_lidar_top_window(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        nusc_root = tmp_path / "nuscenes"
        nusc_root.mkdir()
        fake = FakeNuScenes("v1.0-trainval", str(nusc_root))
        frame_token = _add_scene(
            fake,
            scene_token="scene_boston",
            n_dense_frames=130,
            dt_us=50_000,
            with_annotations=True,
        )
        _install_fake_nusc(monkeypatch, fake)
        qa_path = _write_qa(tmp_path, {
            "scene_boston": {"key_frames": {
                frame_token: {
                    "sample_token": frame_token,
                    "qa_pairs": [{"answer": "A red traffic light is visible."}],
                    "key_object_infos": {
                        "light": {"category": "traffic_light", "Status": "red"},
                    },
                },
            }},
        })

        loader = dataset_loader.DriveLMDatasetLoader(nusc_root, qa_path)
        raw = loader.load_keyframe("scene_boston", frame_token)

        assert isinstance(raw, RawScene)
        assert raw.scene_id == "scene_boston"
        assert raw.frame_token == frame_token
        assert raw.location == "boston-seaport"
        assert raw.raw_json["key_object_infos"]["light"]["Status"] == "red"
        assert len(raw.ego_poses) == 121
        assert raw.ego_poses[0]["timestamp"] == 0
        assert raw.ego_poses[-1]["timestamp"] == 6_000_000
        assert len(raw.path_ego_poses) == 130
        assert raw.path_ego_poses[-1]["timestamp"] == 6_450_000
        assert raw.annotations[0]["attribute_names"] == ["pedestrian.moving"]

    def test_load_keyframe_reads_path_window_beyond_trajectory_window(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        nusc_root = tmp_path / "nuscenes"
        nusc_root.mkdir()
        fake = FakeNuScenes("v1.0-trainval", str(nusc_root))
        frame_token = _add_scene(fake, scene_token="scene_boston", n_dense_frames=260)
        _install_fake_nusc(monkeypatch, fake)
        qa_path = _write_qa(tmp_path, {
            "scene_boston": {"key_frames": {frame_token: {"sample_token": frame_token}}},
        })

        raw = dataset_loader.DriveLMDatasetLoader(nusc_root, qa_path).load_keyframe(
            "scene_boston",
            frame_token,
        )

        assert len(raw.ego_poses) == 121
        assert raw.ego_poses[-1]["timestamp"] == 6_000_000
        assert len(raw.path_ego_poses) == 241
        assert raw.path_ego_poses[-1]["timestamp"] == 12_000_000

    def test_load_keyframe_normalizes_v11_traffic_light_and_qa(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        nusc_root = tmp_path / "nuscenes"
        nusc_root.mkdir()
        fake = FakeNuScenes("v1.0-trainval", str(nusc_root))
        frame_token = _add_scene(fake, scene_token="scene_boston")
        _install_fake_nusc(monkeypatch, fake)
        qa_path = _write_qa(tmp_path, {
            "scene_boston": {"key_frames": {
                frame_token: {
                    "key_object_infos": {
                        "light": {
                            "Category": "Traffic element",
                            "Status": None,
                            "Visual_description": "Red light.",
                        },
                    },
                    "QA": {
                        "perception": [{
                            "Q": "What is visible?",
                            "A": "A red traffic light is visible.",
                        }],
                    },
                },
            }},
        })

        raw = dataset_loader.DriveLMDatasetLoader(nusc_root, qa_path).load_keyframe(
            "scene_boston",
            frame_token,
        )

        assert raw.raw_json["key_object_infos"]["light"]["category"] == "traffic_light"
        assert raw.raw_json["key_object_infos"]["light"]["Status"] == "red"
        assert raw.raw_json["qa_pairs"] == [{
            "category": "perception",
            "question": "What is visible?",
            "answer": "A red traffic light is visible.",
        }]
        facts = SceneFactExtractor().extract(raw)
        assert facts.has_red_light is True
        assert SceneTextExtractor().infer_scenario_type(raw, facts).value == "red_light"

    def test_load_keyframe_rejects_non_boston_scene(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        nusc_root = tmp_path / "nuscenes"
        nusc_root.mkdir()
        fake = FakeNuScenes("v1.0-trainval", str(nusc_root))
        frame_token = _add_scene(fake, scene_token="scene_singapore", location="singapore-onenorth")
        _install_fake_nusc(monkeypatch, fake)
        qa_path = _write_qa(tmp_path, {
            "scene_singapore": {"key_frames": {
                frame_token: {"sample_token": frame_token},
            }},
        })

        loader = dataset_loader.DriveLMDatasetLoader(nusc_root, qa_path)

        with pytest.raises(SceneNotFoundError):
            loader.load_keyframe("scene_singapore", frame_token)

    def test_load_keyframe_requires_drivelm_keyframe(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        nusc_root = tmp_path / "nuscenes"
        nusc_root.mkdir()
        fake = FakeNuScenes("v1.0-trainval", str(nusc_root))
        frame_token = _add_scene(fake, scene_token="scene_boston")
        _install_fake_nusc(monkeypatch, fake)
        qa_path = _write_qa(tmp_path, {"scene_boston": {"key_frames": {}}})
        loader = dataset_loader.DriveLMDatasetLoader(nusc_root, qa_path)

        with pytest.raises(SceneNotFoundError):
            loader.load_keyframe("scene_boston", frame_token)

    def test_iter_keyframes_yields_boston_keyframes(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        nusc_root = tmp_path / "nuscenes"
        nusc_root.mkdir()
        fake = FakeNuScenes("v1.0-trainval", str(nusc_root))
        frame_token = _add_scene(fake, scene_token="scene_boston", n_dense_frames=5)
        _install_fake_nusc(monkeypatch, fake)
        qa_path = _write_qa(tmp_path, {
            "scene_boston": {"key_frames": {
                frame_token: {"sample_token": frame_token, "qa_pairs": []},
            }},
        })
        loader = dataset_loader.DriveLMDatasetLoader(
            nusc_root,
            qa_path,
            future_horizon_s=1.0,
        )

        raws = list(loader.iter_keyframes())

        assert len(raws) == 1
        assert raws[0].scene_id == "scene_boston"
        assert raws[0].frame_token == frame_token
