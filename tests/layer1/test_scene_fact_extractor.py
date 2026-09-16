"""SceneFactExtractor tests for structured v3.1 scene facts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from src.layer1.models import RawScene
from src.layer1.scene_fact_extractor import SceneFactExtractor


def _yaw_quaternion(yaw: float) -> list[float]:
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


def _raw_scene(
    *,
    raw_json: dict[str, Any] | None = None,
    map_records: dict[str, list[dict[str, Any]]] | None = None,
    annotations: list[dict[str, Any]] | None = None,
    ego_yaw: float = 0.0,
    ego_end_yaw: float | None = None,
) -> RawScene:
    ego_poses = [
        {
            "translation": [0.0, 0.0, 0.0],
            "rotation": _yaw_quaternion(ego_yaw),
            "timestamp": 0,
        },
    ]
    if ego_end_yaw is not None:
        ego_poses.append({
            "translation": [5.0, 1.0, 0.0],
            "rotation": _yaw_quaternion(ego_end_yaw),
            "timestamp": 1_000_000,
        })
    return RawScene(
        scene_id="scene_1",
        frame_token="frame_1",
        source_file=Path("/fake/drivelm.json"),
        raw_json=raw_json or {},
        map_records=map_records or {},
        annotations=annotations or [],
        ego_poses=ego_poses,
    )


class TestTrafficLightStatus:
    def test_status_red_and_yellow_from_drivelm_key_objects(self) -> None:
        facts = SceneFactExtractor().extract(_raw_scene(raw_json={
            "key_object_infos": {
                "red": {"category": "traffic_light", "Status": "red"},
                "yellow": {"category": "traffic light", "status": "yellow"},
            },
        }))

        assert facts.has_red_light is True
        assert facts.has_yellow_light is True
        assert facts.traffic_light_status_source == "drivelm_status"

    def test_green_status_does_not_trigger_red_or_yellow(self) -> None:
        facts = SceneFactExtractor().extract(_raw_scene(raw_json={
            "key_object_infos": {
                "green": {"category": "traffic_light", "Status": "green"},
            },
        }))

        assert facts.has_red_light is False
        assert facts.has_yellow_light is False
        assert facts.traffic_light_status_source == "none"


def test_ego_displacement_is_extracted_from_prediction_window() -> None:
    facts = SceneFactExtractor().extract(_raw_scene(ego_end_yaw=0.0))

    assert facts.ego_displacement_m == math.hypot(5.0, 1.0)


class TestConflictZones:
    def test_stop_line_and_ped_crossing_records_become_conflict_zones(self) -> None:
        facts = SceneFactExtractor().extract(_raw_scene(map_records={
            "stop_line": [{"polygon_xy": [(10.0, 0.0), (11.0, 0.0), (11.0, 1.0)]}],
            "ped_crossing": [{"polygon_xy": [(5.0, -1.0), (7.0, -1.0), (7.0, 1.0)]}],
        }))

        assert [zone.kind for zone in facts.conflict_zones] == ["stop_line", "ped_crossing"]
        assert facts.conflict_zones[0].source_layer == "stop_line"
        assert facts.conflict_zones[0].distance_from_ego_m is not None

    def test_location_and_minor_road_from_map_records(self) -> None:
        facts = SceneFactExtractor().extract(_raw_scene(map_records={
            "road_segment": [{"token": "road_1"}],
            "lane": [{"lane_type": "minor"}],
        }))

        assert facts.location_is_intersection is True
        assert facts.ego_on_minor_road is True


class TestStructuredYieldFacts:
    def test_pedestrian_in_crosswalk_uses_annotation_and_polygon(self) -> None:
        facts = SceneFactExtractor().extract(_raw_scene(
            map_records={
                "ped_crossing": [{
                    "polygon_xy": [(4.0, -1.0), (6.0, -1.0), (6.0, 1.0), (4.0, 1.0)],
                }],
            },
            annotations=[
                {
                    "category_name": "human.pedestrian.adult",
                    "translation": [5.0, 0.0, 0.0],
                    "attribute_names": ["pedestrian.moving"],
                },
            ],
        ))

        assert facts.pedestrian_in_crosswalk is True

    def test_forward_pedestrian_motion_preserves_unknown(self) -> None:
        facts = SceneFactExtractor().extract(_raw_scene(
            ego_end_yaw=0.0,
            map_records={
                "ped_crossing": [{
                    "polygon_xy": [(4.0, -1.0), (6.0, -1.0), (6.0, 2.0), (4.0, 2.0)],
                }],
            },
            annotations=[{
                "category_name": "human.pedestrian.adult",
                "translation": [5.0, 0.0, 0.0],
                "attribute_names": [],
            }],
        ))

        assert facts.pedestrian_in_forward_crosswalk is True
        assert facts.pedestrian_moving is None

    def test_forward_pedestrian_uses_footprint_and_track_motion(self) -> None:
        facts = SceneFactExtractor().extract(_raw_scene(
            ego_end_yaw=0.0,
            map_records={
                "ped_crossing": [{
                    "polygon_xy": [
                        (4.0, -1.0),
                        (6.0, -1.0),
                        (6.0, 1.0),
                        (4.0, 1.0),
                    ],
                }],
            },
            annotations=[{
                "category_name": "human.pedestrian.adult",
                "translation": [6.45, 0.0, 0.0],
                "size": [0.6, 0.6, 1.8],
                "attribute_names": [],
                "track": [
                    {"t": 0.0, "translation": [6.45, 0.0, 0.0]},
                    {"t": 1.0, "translation": [7.45, 0.0, 0.0]},
                ],
            }],
        ))

        assert facts.pedestrian_in_forward_crosswalk is True
        assert facts.pedestrian_moving is True

    def test_oncoming_vehicle_requires_moving_vehicle_with_large_yaw_difference(self) -> None:
        facts = SceneFactExtractor().extract(_raw_scene(
            ego_end_yaw=math.radians(30),
            annotations=[
                {
                    "category_name": "vehicle.car",
                    "rotation": _yaw_quaternion(math.pi),
                    "attribute_names": ["vehicle.moving"],
                },
            ],
        ))

        assert facts.oncoming_vehicle_moving is True

    def test_oncoming_vehicle_fact_is_independent_of_ego_turn(self) -> None:
        facts = SceneFactExtractor().extract(_raw_scene(
            ego_end_yaw=0.0,
            annotations=[
                {
                    "category_name": "vehicle.car",
                    "rotation": _yaw_quaternion(math.pi),
                    "attribute_names": ["vehicle.moving"],
                },
            ],
        ))

        assert facts.oncoming_vehicle_moving is True

    def test_emergency_vehicle_requires_active_vehicle_category(self) -> None:
        facts = SceneFactExtractor().extract(_raw_scene(annotations=[
            {
                "category_name": "vehicle.emergency.ambulance",
                "rotation": _yaw_quaternion(0.0),
                "attribute_names": ["vehicle.moving"],
            },
        ]))

        assert facts.emergency_vehicle_active is True

    def test_text_does_not_create_structured_regulatory_fact(self) -> None:
        text_only = SceneFactExtractor().extract(_raw_scene(raw_json={
            "qa_pairs": [{"answer": "A pedestrian is waiting in the crosswalk."}],
        }))
        structured_negative = SceneFactExtractor().extract(_raw_scene(
            raw_json={"qa_pairs": [{"answer": "A pedestrian is in the crosswalk."}]},
            map_records={
                "ped_crossing": [{
                    "polygon_xy": [(4.0, -1.0), (6.0, -1.0), (6.0, 1.0), (4.0, 1.0)],
                }],
            },
            annotations=[
                {
                    "category_name": "human.pedestrian.adult",
                    "translation": [20.0, 0.0, 0.0],
                    "attribute_names": ["pedestrian.moving"],
                },
            ],
        ))

        assert text_only.pedestrian_in_crosswalk is False
        assert structured_negative.pedestrian_in_crosswalk is False
