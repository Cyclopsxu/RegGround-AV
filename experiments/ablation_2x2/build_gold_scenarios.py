"""Build the frozen formal scenario list from human-annotation-v2 adjudication."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

from experiments.ablation_2x2.citations import RuleCatalog
from experiments.ablation_2x2.run_ablation import ROOT, RULE_GRAPH

ANNOTATION_ROOT = ROOT / "human_annotation_v2"
BLIND_SAMPLES = ANNOTATION_ROOT / "results/blind_samples.jsonl"
VERDICT_SIDECAR = ANNOTATION_ROOT / "private/hidden_sidecar.jsonl"
PREFERENCE_SAMPLES = ANNOTATION_ROOT / "results/preference_blind_samples.jsonl"
PREFERENCE_SIDECAR = ANNOTATION_ROOT / "private/preference_sidecar.jsonl"
SAMPLE_MANIFEST = ANNOTATION_ROOT / "results/sample_manifest.json"

_RULE_LINE = re.compile(r"^\[(?P<rule_id>R-[^\]]+)\]\s+(?P<code>\S+)\s+(?P<description>.+)$")
_RULE_SPLIT = re.compile(r"[；;，,、\s]+")
_DIFFICULTY = {"easy": 0, "medium": 1, "hard": 2}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_label(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _csv_by_id(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "audit_id" not in rows[0]:
        raise ValueError(f"adjudication CSV must contain audit_id: {path}")
    result = {str(row["audit_id"]): row for row in rows if row.get("audit_id")}
    if len(result) != len(rows):
        raise ValueError(f"duplicate or empty audit_id in {path}")
    return result


def _paired_rows(
    public_path: Path, sidecar_path: Path
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    public = _jsonl(public_path)
    private = _jsonl(sidecar_path)
    public_by_id = {row["audit_id"]: row for row in public}
    private_by_id = {row["audit_id"]: row for row in private}
    if set(public_by_id) != set(private_by_id):
        raise ValueError(f"public/private audit_id mismatch: {public_path}")
    return [(public_by_id[audit_id], row) for audit_id, row in private_by_id.items()]


def _rule_ids(value: str) -> list[str]:
    raw = value.strip()
    if not raw or raw.upper() == "NONE":
        return []
    return [item for item in _RULE_SPLIT.split(raw) if item and item.upper() != "NONE"]


def _rules(text: str) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for line in text.splitlines():
        match = _RULE_LINE.match(line.strip())
        if not match:
            continue
        description = match.group("description").split("（适用于:", 1)[0]
        result.append(
            {
                "rule_id": match.group("rule_id"),
                "code": match.group("code"),
                "description": description,
            }
        )
    return result


def _set_scene_evidence(scene: dict[str, Any], public: dict[str, Any]) -> None:
    values = {
        "digest_v_a": public["scene_facts"],
        "scene_description": public["scene_description"],
        "retrieved_rules": _rules(public["applicable_hard_rules"]),
    }
    for key, value in values.items():
        if key in scene and scene[key] != value:
            raise ValueError(f"inconsistent {key} for frame {scene['frame_token']}")
        scene[key] = value


def _add_candidate(scene: dict[str, Any], trajectory_id: str, features: str) -> None:
    existing = scene["_candidate_features"].get(trajectory_id)
    if existing is not None and existing != features:
        raise ValueError(f"inconsistent features for {scene['frame_token']}/{trajectory_id}")
    scene["_candidate_features"][trajectory_id] = features


def _new_scene(private: dict[str, Any]) -> dict[str, Any]:
    return {
        "scene_id": private["scene_id"],
        "frame_token": private["frame_token"],
        "scenario_type": private["scenario_type"],
        "difficulty": "easy",
        "sample_groups": [],
        "_candidate_features": OrderedDict(),
        "_preference_order": None,
        "gold": {
            "verdicts": [],
            "applicable_rule_ids_by_trajectory": {},
            "applicable_provision_keys_by_trajectory": {},
        },
    }


def _raise_difficulty(scene: dict[str, Any], values: list[str]) -> None:
    known = [value for value in values if value in _DIFFICULTY]
    if known:
        scene["difficulty"] = max(known + [scene["difficulty"]], key=_DIFFICULTY.__getitem__)


def _preference_gold(
    row: dict[str, str],
    column_to_trajectory: dict[str, str],
) -> dict[str, Any]:
    ranking = row.get("human_ranking", "").strip()
    tiers = [
        [column_to_trajectory[item.strip()] for item in group.split("=") if item.strip()]
        for group in ranking.split(">")
        if group.strip()
    ]
    flat = [item for tier in tiers for item in tier]
    if set(flat) != set(column_to_trajectory.values()) or len(flat) != len(set(flat)):
        raise ValueError(f"preference ranking does not cover candidates: {row['audit_id']}")
    best = row.get("human_best_trajectory", "").strip()
    chosen = None if best.upper() == "NONE" else column_to_trajectory.get(best)
    if best.upper() != "NONE" and chosen is None:
        raise ValueError(f"unknown human_best_trajectory: {row['audit_id']}/{best}")
    pairs = [
        {"preferred_id": preferred, "dispreferred_id": dispreferred}
        for left_index, left_tier in enumerate(tiers)
        for right_tier in tiers[left_index + 1 :]
        for preferred in left_tier
        for dispreferred in right_tier
    ]
    return {
        "preference_ranking": flat,
        "preference_tiers": tiers,
        "chosen_trajectory_id": chosen,
        "preference_pairs": pairs,
    }


def build(
    *,
    verdict_adjudicated: Path,
    preference_adjudicated: Path,
    output_path: Path,
) -> dict[str, Any]:
    verdict_gold = _csv_by_id(verdict_adjudicated)
    preference_gold = _csv_by_id(preference_adjudicated)
    verdict_pairs = _paired_rows(BLIND_SAMPLES, VERDICT_SIDECAR)
    preference_pairs = _paired_rows(PREFERENCE_SAMPLES, PREFERENCE_SIDECAR)
    verdict_ids = {private["audit_id"] for _public, private in verdict_pairs}
    preference_ids = {private["audit_id"] for _public, private in preference_pairs}
    if set(verdict_gold) != verdict_ids:
        raise ValueError("verdict adjudication audit_ids do not match the 200-row formal package")
    if set(preference_gold) != preference_ids:
        raise ValueError("preference adjudication audit_ids do not match the 36-scene package")

    catalog = RuleCatalog.from_yaml(RULE_GRAPH)
    scenes: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for public, private in verdict_pairs:
        frame = private["frame_token"]
        scene = scenes.setdefault(frame, _new_scene(private))
        _set_scene_evidence(scene, public)
        trajectory_id = private["trajectory_id"]
        _add_candidate(scene, trajectory_id, public["candidate_features"])
        group = str(private.get("group", ""))
        if group and group not in scene["sample_groups"]:
            scene["sample_groups"].append(group)
        _raise_difficulty(scene, [str(private.get("benchmark_label", {}).get("difficulty", ""))])
        gold = verdict_gold[private["audit_id"]]
        applicable = _rule_ids(gold.get("human_applicable_rule_ids", ""))
        scene["gold"]["verdicts"].append(
            {
                "trajectory_id": trajectory_id,
                "status": gold["human_verdict"].strip().lower(),
            }
        )
        scene["gold"]["applicable_rule_ids_by_trajectory"][trajectory_id] = applicable
        provisions = {
            catalog.rule_to_provision[rule_id]
            for rule_id in applicable
            if rule_id in catalog.rule_to_provision
        }
        scene["gold"]["applicable_provision_keys_by_trajectory"][trajectory_id] = sorted(provisions)

    for public, private in preference_pairs:
        frame = private["frame_token"]
        scene = scenes.setdefault(frame, _new_scene(private))
        _set_scene_evidence(scene, public)
        column_to_trajectory = {
            str(column): str(trajectory_id)
            for column, trajectory_id in private["column_to_trajectory_id"].items()
        }
        preference_order: list[str] = []
        for column, features in public["candidates"].items():
            trajectory_id = column_to_trajectory[column]
            preference_order.append(trajectory_id)
            _add_candidate(scene, trajectory_id, str(features))
        scene["_preference_order"] = preference_order
        _raise_difficulty(
            scene,
            [
                str(label.get("difficulty", ""))
                for label in private.get("benchmark_labels", {}).values()
            ],
        )
        scene["gold"].update(
            _preference_gold(preference_gold[private["audit_id"]], column_to_trajectory)
        )

    for scene in scenes.values():
        candidate_features = scene.pop("_candidate_features")
        preferred_order = scene.pop("_preference_order")
        order = preferred_order or sorted(candidate_features)
        scene["candidates"] = [
            {
                "trajectory_id": trajectory_id,
                "features": candidate_features[trajectory_id],
            }
            for trajectory_id in order
        ]
        scene["gold"]["verdicts"].sort(key=lambda item: order.index(item["trajectory_id"]))

    manifest = json.loads(SAMPLE_MANIFEST.read_text(encoding="utf-8"))
    verdict_frames = {private["frame_token"] for _public, private in verdict_pairs}
    preference_frames = {private["frame_token"] for _public, private in preference_pairs}
    summary = manifest["sampling"]["summary"]
    if (
        len(verdict_frames) != summary["distinct_scenes_in_workbook"]
        or len(preference_frames) != summary["preference_scenes"]
    ):
        raise ValueError("human annotation scene counts do not match sample_manifest.json")
    frozen_counts = (
        len(verdict_pairs),
        len(verdict_frames),
        len(preference_frames),
        len(verdict_frames & preference_frames),
        len(preference_frames - verdict_frames),
        len(scenes),
    )
    if frozen_counts != (200, 78, 36, 32, 4, 82):
        raise ValueError(f"unexpected human_annotation_v2 sample union: {frozen_counts}")

    source_paths = (
        BLIND_SAMPLES,
        VERDICT_SIDECAR,
        PREFERENCE_SAMPLES,
        PREFERENCE_SIDECAR,
        SAMPLE_MANIFEST,
        verdict_adjudicated,
        preference_adjudicated,
    )
    payload = {
        "protocol_version": "ablation_2x2_v1.0",
        "gold_finalized": True,
        "source": "human_annotation_v2 formal verdict/preference union",
        "counts": {
            "verdict_rows": frozen_counts[0],
            "verdict_scenes": frozen_counts[1],
            "preference_scenes": frozen_counts[2],
            "overlap_scenes": frozen_counts[3],
            "preference_only_scenes": frozen_counts[4],
            "union_scenes": frozen_counts[5],
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
    parser = argparse.ArgumentParser(
        description="Build the frozen 82-scene ablation list from human adjudication"
    )
    parser.add_argument("--verdict-adjudicated", type=Path, required=True)
    parser.add_argument("--preference-adjudicated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build(
        verdict_adjudicated=args.verdict_adjudicated.resolve(),
        preference_adjudicated=args.preference_adjudicated.resolve(),
        output_path=args.output.resolve(),
    )
    print(f"wrote {len(payload['scenarios'])} frozen scenarios to {args.output.resolve()}")


if __name__ == "__main__":
    main()
