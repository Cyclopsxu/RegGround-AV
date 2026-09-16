"""End-to-end Layer 1 facade tests."""

from __future__ import annotations

from pathlib import Path

from src.layer1.facade import parse_scene
from src.layer1.leakage_guard import LeakageGuard
from src.layer1.models import ParsedSceneBundle, RawScene, ScenarioType, TrajectoryVariantType


def _raw_scene(answer: str = "A red traffic light is visible near the stop line.") -> RawScene:
    poses = [
        {
            "token": f"pose_{i}",
            "timestamp": i * 250_000,
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "translation": [float(i), 0.0, 0.0],
        }
        for i in range(25)
    ]
    return RawScene(
        scene_id="scene_1",
        frame_token="frame_1",
        source_file=Path("/fake/drivelm.json"),
        raw_json={
            "qa_pairs": [{"answer": answer}],
            "key_object_infos": {
                "light": {"category": "traffic_light", "Status": "red"},
            },
        },
        ego_poses=poses,
        map_records={
            "stop_line": [{"polygon_xy": [(12.0, -1.0), (12.0, 1.0), (13.0, 1.0)]}],
        },
    )


def _pedestrian_raw_scene(*, occupied_until_s: float = 0.0) -> RawScene:
    poses = [
        {
            "token": f"ped_pose_{index}",
            "timestamp": index * 100_000,
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "translation": [float(index) * 0.1, 0.0, 0.0],
        }
        for index in range(61)
    ]
    return RawScene(
        scene_id="ped_scene",
        frame_token="ped_frame",
        source_file=Path("/fake/drivelm.json"),
        ego_poses=poses,
        map_records={
            "ped_crossing": [{
                "polygon_xy": [(3.0, -1.0), (4.0, -1.0), (4.0, 1.0), (3.0, 1.0)],
            }],
        },
        annotations=[{
            "instance_token": "ped-1",
            "category_name": "human.pedestrian.adult",
            "translation": [3.5, 0.0, 0.0],
            "attribute_names": ["pedestrian.moving"],
            "track": [
                {"t": float(time), "translation": [3.5, 0.0, 0.0]}
                for time in range(int(occupied_until_s) + 1)
            ],
        }],
    )


def _render_prompt_fragment(bundle: ParsedSceneBundle) -> str:
    summaries = [
        features.natural_language_summary
        for features in bundle.judge_input.trajectory_features
    ]
    return "\n".join([bundle.judge_input.narrative, *summaries])


def _has_key(value: object, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_has_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_has_key(item, key) for item in value)
    return False


class TestParseScene:
    def test_parse_scene_returns_v31_bundle(self) -> None:
        bundle = parse_scene(_raw_scene(), seed=42)

        assert isinstance(bundle, ParsedSceneBundle)
        assert bundle.context.scene_id == "scene_1"
        assert bundle.context.frame_token == "frame_1"
        assert bundle.context.scenario_type == ScenarioType.RED_LIGHT
        assert bundle.scene_query.scene_facts.has_red_light is True
        assert bundle.scene_query.scene_facts.conflict_zones[0].kind == "stop_line"
        assert [trajectory.traj_id for trajectory in bundle.context.candidate_trajectories] == [
            "traj_a",
            "traj_b",
            "traj_c",
            "traj_d",
            "traj_e",
            "traj_f",
        ]
        assert len(bundle.context.trajectory_features) == 6
        assert len(bundle.judge_input.trajectory_features) == 6
        assert {label.variant_type for label in bundle.benchmark_labels.labels} == set(
            TrajectoryVariantType,
        )

    def test_parse_scene_keeps_labels_out_of_downstream_interfaces(self) -> None:
        bundle = parse_scene(_raw_scene(), seed=42)
        judge_input_text = str(bundle.judge_input.model_dump()).lower()

        assert not _has_key(bundle.scene_query.model_dump(), "waypoints")
        assert not _has_key(bundle.judge_input.model_dump(), "waypoints")
        assert "variant" not in judge_input_text
        assert "vetoed" not in judge_input_text
        LeakageGuard.assert_safe_prompt_fragment(_render_prompt_fragment(bundle))

    def test_parse_scene_scrubs_narrative_before_judge_input(self) -> None:
        bundle = parse_scene(
            _raw_scene("The illegal variant should stop before the line."),
            seed=42,
        )

        narrative = bundle.judge_input.narrative.lower()
        assert "illegal" not in narrative
        assert "should stop" not in narrative
        assert "[redacted]" in narrative

    def test_short_window_is_excluded_before_drivable_candidate_validation(self) -> None:
        raw = _raw_scene().model_copy(update={
            "ego_poses": _raw_scene().ego_poses[:-2],
            "map_records": {
                "stop_line": [{
                    "polygon_xy": [(12.0, -1.0), (12.0, 1.0), (13.0, 1.0)],
                }],
                "drivable_area": [{
                    "polygon_xy_list": [[
                        (-0.5, -1.0),
                        (2.5, -1.0),
                        (2.5, 1.0),
                        (-0.5, 1.0),
                    ]],
                }],
            },
        })

        bundle = parse_scene(raw, seed=42)

        assert bundle.context.scene_facts.window_insufficient is True
        assert {
            (label.expected_verdict, label.exclude_reason)
            for label in bundle.benchmark_labels.labels
        } == {("exclude", "not_evaluable_short_window")}

    def test_pedestrian_illegal_meets_gap_target_or_is_not_injectable(self) -> None:
        bundle = parse_scene(_pedestrian_raw_scene(), seed=42)
        label = next(
            item
            for item in bundle.benchmark_labels.labels
            if item.variant_type == TrajectoryVariantType.ILLEGAL
        )
        feature_index = next(
            index
            for index, trajectory in enumerate(bundle.context.candidate_trajectories)
            if trajectory.traj_id == label.anonymous_id
        )
        gaps = [
            interaction.min_time_gap_s
            for interaction in bundle.context.trajectory_features[
                feature_index
            ].agent_interactions
            if interaction.agent_kind == "pedestrian"
            and interaction.min_time_gap_s is not None
        ]

        assert (gaps and min(gaps) <= -0.3) or (
            label.exclude_reason == "no_violation_injectable"
        )

    def test_pedestrian_illegal_search_tries_slowdown_first(self) -> None:
        bundle = parse_scene(_pedestrian_raw_scene(occupied_until_s=5.0), seed=42)
        label = next(
            item
            for item in bundle.benchmark_labels.labels
            if item.variant_type == TrajectoryVariantType.ILLEGAL
        )

        assert label.expected_verdict == "vetoed"
        trajectory_by_id = {
            trajectory.traj_id: trajectory
            for trajectory in bundle.context.candidate_trajectories
        }
        gt_label = next(
            item
            for item in bundle.benchmark_labels.labels
            if item.variant_type == TrajectoryVariantType.GROUND_TRUTH
        )
        assert (
            trajectory_by_id[label.anonymous_id].waypoints[-1].x
            < trajectory_by_id[gt_label.anonymous_id].waypoints[-1].x
        )
        illegal_feature = bundle.context.trajectory_features[next(
            index
            for index, trajectory in enumerate(bundle.context.candidate_trajectories)
            if trajectory.traj_id == label.anonymous_id
        )]
        crossing = next(
            metric
            for metric in illegal_feature.conflict_zone_metrics
            if metric.kind == "ped_crossing"
        )
        assert crossing.stopped_before_zone is False
        conservative = next(
            item
            for item in bundle.benchmark_labels.labels
            if item.variant_type == TrajectoryVariantType.CONSERVATIVE
        )
        assert conservative.expected_verdict == "cleared"
        assert conservative.expected_verdict_reason == "pedestrian_stopped_before_crossing"
