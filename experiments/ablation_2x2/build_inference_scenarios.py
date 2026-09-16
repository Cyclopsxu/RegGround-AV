"""Build the frozen 82-scene inputs without attaching unfinished human gold."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

from experiments.ablation_2x2.build_gold_scenarios import (
    BLIND_SAMPLES,
    PREFERENCE_SAMPLES,
    PREFERENCE_SIDECAR,
    SAMPLE_MANIFEST,
    VERDICT_SIDECAR,
    _add_candidate,
    _paired_rows,
    _raise_difficulty,
    _set_scene_evidence,
    _sha256,
    _source_label,
)


def _new_input_scene(private: dict[str, Any]) -> dict[str, Any]:
    return {
        "scene_id": private["scene_id"],
        "frame_token": private["frame_token"],
        "scenario_type": private["scenario_type"],
        "difficulty": "easy",
        "sample_groups": [],
        "_candidate_features": OrderedDict(),
        "_preference_order": None,
    }


def build(output_path: Path) -> dict[str, Any]:
    """Freeze the annotation-v2 input union while keeping human answers absent."""
    verdict_pairs = _paired_rows(BLIND_SAMPLES, VERDICT_SIDECAR)
    preference_pairs = _paired_rows(PREFERENCE_SAMPLES, PREFERENCE_SIDECAR)
    scenes: OrderedDict[str, dict[str, Any]] = OrderedDict()

    for public, private in verdict_pairs:
        frame = private["frame_token"]
        scene = scenes.setdefault(frame, _new_input_scene(private))
        _set_scene_evidence(scene, public)
        _add_candidate(scene, private["trajectory_id"], public["candidate_features"])
        group = str(private.get("group", ""))
        if group and group not in scene["sample_groups"]:
            scene["sample_groups"].append(group)
        _raise_difficulty(
            scene, [str(private.get("benchmark_label", {}).get("difficulty", ""))]
        )

    for public, private in preference_pairs:
        frame = private["frame_token"]
        scene = scenes.setdefault(frame, _new_input_scene(private))
        _set_scene_evidence(scene, public)
        column_to_trajectory = {
            str(column): str(trajectory_id)
            for column, trajectory_id in private["column_to_trajectory_id"].items()
        }
        order: list[str] = []
        for column, features in public["candidates"].items():
            trajectory_id = column_to_trajectory[column]
            order.append(trajectory_id)
            _add_candidate(scene, trajectory_id, str(features))
        scene["_preference_order"] = order
        _raise_difficulty(
            scene,
            [
                str(label.get("difficulty", ""))
                for label in private.get("benchmark_labels", {}).values()
            ],
        )

    for scene in scenes.values():
        features = scene.pop("_candidate_features")
        order = scene.pop("_preference_order") or sorted(features)
        scene["candidates"] = [
            {"trajectory_id": trajectory_id, "features": features[trajectory_id]}
            for trajectory_id in order
        ]

    manifest = json.loads(SAMPLE_MANIFEST.read_text(encoding="utf-8"))
    verdict_frames = {private["frame_token"] for _public, private in verdict_pairs}
    preference_frames = {private["frame_token"] for _public, private in preference_pairs}
    counts = (
        len(verdict_pairs),
        len(verdict_frames),
        len(preference_frames),
        len(verdict_frames & preference_frames),
        len(preference_frames - verdict_frames),
        len(scenes),
    )
    summary = manifest["sampling"]["summary"]
    if (
        len(verdict_frames) != summary["distinct_scenes_in_workbook"]
        or len(preference_frames) != summary["preference_scenes"]
        or counts != (200, 78, 36, 32, 4, 82)
    ):
        raise ValueError(f"unexpected human_annotation_v2 input union: {counts}")

    source_paths = (
        BLIND_SAMPLES,
        VERDICT_SIDECAR,
        PREFERENCE_SAMPLES,
        PREFERENCE_SIDECAR,
        SAMPLE_MANIFEST,
    )
    payload = {
        "protocol_version": "ablation_2x2_v1.0",
        "gold_finalized": False,
        "inference_only": True,
        "source": "human_annotation_v2 frozen verdict/preference input union",
        "counts": {
            "verdict_rows": counts[0],
            "verdict_scenes": counts[1],
            "preference_scenes": counts[2],
            "overlap_scenes": counts[3],
            "preference_only_scenes": counts[4],
            "union_scenes": counts[5],
        },
        "source_sha256": {_source_label(path): _sha256(path) for path in source_paths},
        "scenarios": list(scenes.values()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build blinded 82-scene ablation inputs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build(args.output.resolve())
    print(f"wrote {len(payload['scenarios'])} blinded inference scenarios")


if __name__ == "__main__":
    main()
