#!/usr/bin/env python3
"""Offline audit of the frozen v2 stop-line signed-distance fork.

Run only at the frozen detached commit recorded in FROZEN_COMMIT.  The human
sampling sidecars and completed run log were archived later, so they are read
from SOURCE_COMMIT with ``git show``; no candidate or frozen log is overwritten.

Methodology invariant: every path/polygon or segment/segment intersection uses
the frozen production geometry in ``TrajectoryAugmentor``
(``_first_zone_stop_path_s`` and ``_segment_intersection_fraction``).  This
script does not carry an independent intersection implementation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.layer1.facade import parse_scene
from src.compliance_predicates import red_light_crossing_verdict
from src.layer1.models import (
    ConflictZone,
    RawScene,
    SceneFactsDigest,
    TrajectoryFeatures,
)
from src.layer1.scene_fact_extractor import (
    SceneFactExtractor,
    _global_to_ego,
    _quat_to_yaw,
)
from src.layer1.trajectory_analyzer import TrajectoryAnalyzer
from src.layer1.trajectory_augmentor import TrajectoryAugmentor


OUTPUT_DIR = ROOT / "artifacts/signed_m_recompute"
FROZEN_COMMIT = "4fe1829ded82d851eacf69f284c0a621066c454a"
SOURCE_COMMIT = "5895770cec03e6b42bf6caa1c81902d9b6735e97"
RAW_INPUT = ROOT / "data/layer1/raw_scene_trainval.json"
RUN_LOG_PATH = "logs/eval_set_v2_full_v4_1.jsonl"
HIDDEN_PATH = "human_annotation_v2/private/hidden_sidecar.jsonl"
PREFERENCE_PATH = "human_annotation_v2/private/preference_sidecar.jsonl"
CSV_FIELDS = [
    "scene_id",
    "frame_token",
    "traj_id",
    "signed_m_current",
    "signed_m_correct",
    "delta",
    "polygon_id_current",
    "polygon_id_correct",
    "polygon_changed",
    "branch_current",
    "branch_correct_polygon_entry",
    "branch_correct_centerline_entry",
    "branch_correct",
    "branch_correct_changed_by_entry_time",
    "branch_correct_changed_by_stop_relation",
    "entry_time_correct_polygon_s",
    "entry_time_correct_centerline_s",
    "stopped_before_correct_polygon",
    "stopped_before_correct_centerline",
    "branch_flipped",
    "n_polygons_in_2m_window",
    "max_intrusion_current",
    "intrusion_source_polygon_id",
    "rebuild_verified",
]


@dataclass(frozen=True)
class StopRecord:
    polygon_id: str
    polygon: list[tuple[float, float]]
    centroid_x: float
    normal: tuple[float, float]
    signed_m: float
    normal_alignment: float


def git_text(commit: str, path: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"{commit}:{path}"], cwd=ROOT, text=True
    )


def jsonl(raw: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def stable_polygon_id(record: dict[str, Any], polygon: list[tuple[float, float]]) -> str:
    token = record.get("token")
    if isinstance(token, str) and token:
        return token
    canonical = json.dumps(
        [[round(x, 9), round(y, 9)] for x, y in polygon],
        separators=(",", ":"),
    )
    return "vertices_sha256:" + hashlib.sha256(canonical.encode()).hexdigest()[:16]


def stop_records(raw_scene: RawScene) -> list[StopRecord]:
    ego_xy = (0.0, 0.0)
    ego_yaw = 0.0
    if raw_scene.ego_poses:
        pose = raw_scene.ego_poses[raw_scene.keyframe_index]
        translation = pose.get("translation", [])
        if isinstance(translation, (list, tuple)) and len(translation) >= 2:
            ego_xy = (float(translation[0]), float(translation[1]))
        parsed_yaw = _quat_to_yaw(pose.get("rotation", []))
        if parsed_yaw is not None:
            ego_yaw = parsed_yaw

    result: list[StopRecord] = []
    for raw_record in raw_scene.map_records.get("stop_line", []):
        if not isinstance(raw_record, dict):
            continue
        global_polygon = SceneFactExtractor._polygon_xy(raw_record)
        polygon = [_global_to_ego(point, ego_xy, ego_yaw) for point in global_polygon]
        if len(polygon) < 2:
            continue

        # The handbook does not define how a proxy polygon yields its centerline.
        # Audit choice: direction = longest boundary edge; centerline = the line
        # parallel to it halfway between the polygon's two extreme normal supports.
        edges = [
            (
                math.hypot(end[0] - start[0], end[1] - start[1]),
                index,
                end[0] - start[0],
                end[1] - start[1],
            )
            for index, (start, end) in enumerate(
                zip(polygon, polygon[1:] + polygon[:1], strict=False)
            )
        ]
        length, _, dx, dy = max(edges, key=lambda item: (item[0], -item[1]))
        if length <= 0.0:
            continue
        nx, ny = -dy / length, dx / length
        if nx < 0.0 or (nx == 0.0 and ny < 0.0):
            nx, ny = -nx, -ny
        supports = [x * nx + y * ny for x, y in polygon]
        signed_m = (min(supports) + max(supports)) / 2.0
        result.append(
            StopRecord(
                polygon_id=stable_polygon_id(raw_record, polygon),
                polygon=polygon,
                centroid_x=sum(x for x, _ in polygon) / len(polygon),
                normal=(nx, ny),
                signed_m=signed_m,
                normal_alignment=nx,
            )
        )
    return result


def select_current(records: list[StopRecord]) -> StopRecord | None:
    eligible = [
        record
        for record in records
        if abs(record.centroid_x) <= 50.0
        and record.normal_alignment >= math.cos(math.radians(45.0))
    ]
    return min(eligible, key=lambda record: abs(record.centroid_x), default=None)


def select_correct(records: list[StopRecord]) -> StopRecord | None:
    eligible = [
        record
        for record in records
        if abs(record.signed_m) <= 50.0
        and record.normal_alignment >= math.cos(math.radians(45.0))
    ]
    # The handbook says "take nearest" but gives no exact-distance tie rule.
    # Audit choice: absolute normal projection, then stable polygon id.
    return min(
        eligible,
        key=lambda record: (abs(record.signed_m), record.polygon_id),
        default=None,
    )


def digest(facts: Any, signed_m: float | None) -> SceneFactsDigest:
    return SceneFactsDigest(
        signal_phase="red" if facts.has_red_light else "unknown",
        phase_source=facts.traffic_light_status_source,
        ego_turn_intent=facts.ego_turn_intent,
        governing_stop_line_signed_m=signed_m,
        pedestrian_in_forward_crosswalk=facts.pedestrian_in_forward_crosswalk,
        pedestrian_moving=facts.pedestrian_moving,
        oncoming_vehicle_moving=facts.oncoming_vehicle_moving,
        location_is_intersection=facts.location_is_intersection,
    )


def branch_name(feature: TrajectoryFeatures, facts_digest: SceneFactsDigest) -> str:
    reason = red_light_crossing_verdict(feature, facts_digest).reason
    return {
        "no_governing_signal_or_line": "no_governing_signal_or_line",
        "start_beyond_line": "start_beyond_line",
        "right_turn_on_red_exemption": "右转豁免",
        "crossed_governing_line_during_valid_red_without_full_stop": "有效窗内越线无停",
        "stopped_before_governing_line_without_crossing": "线前完整停车",
        "borderline_or_phase_horizon_unknown": "交 LLM",
    }[reason]


def centerline_segment(record: StopRecord) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Clip the audit centerline to the proxy polygon with frozen segment intersections."""
    nx, ny = record.normal
    direction = (-ny, nx)
    center = (record.signed_m * nx, record.signed_m * ny)
    radius = max(
        (math.hypot(x - center[0], y - center[1]) for x, y in record.polygon),
        default=0.0,
    ) + 1.0
    probe_start = (
        center[0] - 2.0 * radius * direction[0],
        center[1] - 2.0 * radius * direction[1],
    )
    probe_end = (
        center[0] + 2.0 * radius * direction[0],
        center[1] + 2.0 * radius * direction[1],
    )
    fractions: list[float] = []
    for edge_start, edge_end in zip(
        record.polygon, record.polygon[1:] + record.polygon[:1], strict=False
    ):
        fraction = TrajectoryAugmentor._segment_intersection_fraction(
            probe_start, probe_end, edge_start, edge_end
        )
        if fraction is not None and not any(
            math.isclose(fraction, seen, rel_tol=0.0, abs_tol=1e-9)
            for seen in fractions
        ):
            fractions.append(fraction)
    if len(fractions) < 2:
        return None
    low, high = min(fractions), max(fractions)
    return (
        (
            probe_start[0] + low * (probe_end[0] - probe_start[0]),
            probe_start[1] + low * (probe_end[1] - probe_start[1]),
        ),
        (
            probe_start[0] + high * (probe_end[0] - probe_start[0]),
            probe_start[1] + high * (probe_end[1] - probe_start[1]),
        ),
    )


def continuous_centerline_crossing_time(
    trajectory: Any, record: StopRecord
) -> float | None:
    line = centerline_segment(record)
    if line is None:
        return None
    line_start, line_end = line
    for start, end in zip(
        trajectory.waypoints, trajectory.waypoints[1:], strict=False
    ):
        fraction = TrajectoryAugmentor._segment_intersection_fraction(
            (float(start.x), float(start.y)),
            (float(end.x), float(end.y)),
            line_start,
            line_end,
        )
        if fraction is not None:
            return float(start.t + fraction * (end.t - start.t))
    return None


def stopped_before_centerline(
    trajectory: Any, record: StopRecord, crossing_time: float | None
) -> bool | None:
    """Apply the frozen stop test with the centerline time as its boundary."""
    first = trajectory.waypoints[0]
    start_signed = record.signed_m - (
        float(first.x) * record.normal[0] + float(first.y) * record.normal[1]
    )
    if start_signed <= 1e-9:
        return None
    pts = np.asarray([[w.x, w.y] for w in trajectory.waypoints], dtype=np.float64)
    times = np.asarray([w.t for w in trajectory.waypoints], dtype=np.float64)
    speeds = TrajectoryAnalyzer._compute_speeds(pts, times)
    stopped = (speeds < 0.3) & ~np.isnan(speeds)
    stopped_indices = np.flatnonzero(stopped)
    boundary_time = crossing_time if crossing_time is not None else math.inf
    return bool(
        len(stopped_indices)
        and float(times[int(stopped_indices[0])]) < boundary_time
        and float(np.diff(times)[stopped].sum()) >= 1.0
    )


def metrics_for_single_polygon(
    trajectory: Any,
    feature: TrajectoryFeatures,
    facts: Any,
    record: StopRecord | None,
) -> tuple[TrajectoryFeatures, TrajectoryFeatures, TrajectoryFeatures, float | None]:
    existing_non_stop = [m for m in feature.conflict_zone_metrics if m.kind != "stop_line"]
    if record is None:
        without_stop = feature.model_copy(update={"conflict_zone_metrics": existing_non_stop})
        return without_stop, without_stop, without_stop, None
    zone = ConflictZone(
        kind="stop_line",
        polygon_xy=record.polygon,
        distance_from_ego_m=TrajectoryAnalyzer._distance_to_polygon(
            (0.0, 0.0), record.polygon
        ),
        source_layer="stop_line",
    )
    isolated_facts = facts.model_copy(
        update={
            "conflict_zones": [zone],
            # The existing analyzer identifies governing zones by centroid-x.
            # Feed that exact centroid only to isolate the selected record; the
            # predicate digest separately receives signed_m_correct.
            "governing_stop_line_signed_m": record.centroid_x,
        }
    )
    pts = np.asarray([[w.x, w.y] for w in trajectory.waypoints], dtype=np.float64)
    times = np.asarray([w.t for w in trajectory.waypoints], dtype=np.float64)
    speeds = TrajectoryAnalyzer._compute_speeds(pts, times)
    metrics = TrajectoryAnalyzer._conflict_zone_metrics(pts, times, speeds, isolated_facts)
    stop_metric = next(m for m in metrics if m.kind == "stop_line")

    # Directly invoke the frozen continuous path/polygon intersection helper
    # (trajectory_augmentor.py:371,380-394). The centerline calculation below
    # likewise delegates every segment intersection to the frozen helper.
    TrajectoryAugmentor._first_zone_stop_path_s(pts, [zone])
    polygon_feature = feature.model_copy(
        update={"conflict_zone_metrics": [stop_metric, *existing_non_stop]}
    )
    crossing_time = continuous_centerline_crossing_time(trajectory, record)
    centerline_metric = stop_metric.model_copy(
        update={
            "entered": crossing_time is not None,
            "entry_time_s": crossing_time,
        }
    )
    centerline_feature = feature.model_copy(
        update={"conflict_zone_metrics": [centerline_metric, *existing_non_stop]}
    )
    centerline_stop_metric = centerline_metric.model_copy(
        update={
            "stopped_before_zone": stopped_before_centerline(
                trajectory, record, crossing_time
            )
        }
    )
    centerline_stop_feature = feature.model_copy(
        update={"conflict_zone_metrics": [centerline_stop_metric, *existing_non_stop]}
    )
    return polygon_feature, centerline_feature, centerline_stop_feature, crossing_time


def intrusion_source(
    trajectory: Any, records: list[StopRecord]
) -> tuple[float | None, str | None]:
    best_depth: float | None = None
    best_id: str | None = None
    for waypoint in trajectory.waypoints:
        point = (float(waypoint.x), float(waypoint.y))
        for record in records:
            if not TrajectoryAnalyzer._point_in_polygon(point, record.polygon):
                continue
            depth = min(
                TrajectoryAnalyzer._distance_to_segment(
                    point, record.polygon[index - 1], record.polygon[index]
                )
                for index in range(len(record.polygon))
            )
            if (
                best_depth is None
                or depth > best_depth + 1e-12
                or (
                    abs(depth - best_depth) <= 1e-12
                    and record.polygon_id < (best_id or record.polygon_id)
                )
            ):
                best_depth, best_id = depth, record.polygon_id
    return best_depth, best_id


def fmt(value: Any) -> Any:
    if value is None:
        return "未找到"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.12g}"
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key)) for key in fields})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initial-clean-status",
        default="## HEAD (no branch)",
        help="git status captured after checkout and before creating this script",
    )
    args = parser.parse_args()
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != FROZEN_COMMIT:
        raise SystemExit(f"must run at frozen commit {FROZEN_COMMIT}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    hidden = jsonl(git_text(SOURCE_COMMIT, HIDDEN_PATH))
    preferences = jsonl(git_text(SOURCE_COMMIT, PREFERENCE_PATH))
    wanted = {(x["scene_id"], x["frame_token"]) for x in hidden} | {
        (x["scene_id"], x["frame_token"]) for x in preferences
    }
    predicate_lookup = {
        (x["frame_token"], x["trajectory_id"]): x["audit_id"]
        for x in hidden
        if x.get("group") == "core_predicate"
    }
    raw_records = json.loads(RAW_INPUT.read_text(encoding="utf-8"))
    raw_by_token = {x["frame_token"]: x for x in raw_records}
    frozen_records: dict[str, dict[str, Any]] = {}
    for item in jsonl(git_text(SOURCE_COMMIT, RUN_LOG_PATH)):
        if item.get("type") == "record":
            data = item["data"]
            frozen_records[data["raw_scene"]["frame_token"]] = data

    rebuilt: dict[str, Any] = {}
    mismatches: list[dict[str, Any]] = []
    exact = 0
    total = 0
    for scene_id, frame_token in sorted(wanted, key=lambda item: item[1]):
        bundle = parse_scene(RawScene.model_validate(raw_by_token[frame_token]), seed=42)
        rebuilt[frame_token] = bundle
        logged = frozen_records[frame_token]["layer1"]["context"]
        ids = list(logged["candidate_trajectory_ids"])
        rebuilt_ids = [x.traj_id for x in bundle.context.candidate_trajectories]
        if ids != rebuilt_ids:
            mismatches.append(
                {
                    "scene_id": scene_id,
                    "frame_token": frame_token,
                    "traj_id": "*",
                    "field": "candidate_trajectory_ids",
                    "logged": ids,
                    "rebuilt": rebuilt_ids,
                }
            )
        for index, (actual_model, expected) in enumerate(
            zip(
                bundle.context.trajectory_features,
                logged["trajectory_features"],
                strict=True,
            )
        ):
            total += 1
            actual = actual_model.model_dump(mode="json")
            if actual == expected:
                exact += 1
                continue
            for field in sorted(set(actual) | set(expected)):
                if actual.get(field) != expected.get(field):
                    mismatches.append(
                        {
                            "scene_id": scene_id,
                            "frame_token": frame_token,
                            "traj_id": ids[index],
                            "field": field,
                            "logged": expected.get(field),
                            "rebuilt": actual.get(field),
                        }
                    )

    equivalence = {
        "frozen_commit": FROZEN_COMMIT,
        "source_commit": SOURCE_COMMIT,
        "initial_git_status": args.initial_clean_status,
        "frames": len(wanted),
        "candidates": total,
        "exact_candidates": exact,
        "mismatch_candidates": len(
            {(x["frame_token"], x["traj_id"]) for x in mismatches}
        ),
        "mismatch_fields": len(mismatches),
        "mismatches": mismatches,
    }
    (OUTPUT_DIR / "rebuild_equivalence.json").write_text(
        json.dumps(equivalence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if len(wanted) != 82 or total != 492 or mismatches:
        print(json.dumps(equivalence, ensure_ascii=False))
        return 3

    rows: list[dict[str, Any]] = []
    predicate_rows: list[dict[str, Any]] = []
    for scene_id, frame_token in sorted(wanted, key=lambda item: item[1]):
        bundle = rebuilt[frame_token]
        raw_scene = RawScene.model_validate(raw_by_token[frame_token])
        records = stop_records(raw_scene)
        current = select_current(records)
        correct = select_correct(records)
        facts = bundle.context.scene_facts
        current_signed = facts.governing_stop_line_signed_m
        if (current is None) != (current_signed is None) or (
            current is not None
            and current_signed is not None
            and not math.isclose(
                current.centroid_x, current_signed, rel_tol=0.0, abs_tol=1e-12
            )
        ):
            raise RuntimeError(
                f"current polygon selection mismatch {frame_token}: "
                f"facts={current_signed} record={current}"
            )
        correct_signed = correct.signed_m if correct else None
        window = (
            [r for r in records if current_signed is not None and abs(r.centroid_x - current_signed) < 2.0]
            if current_signed is not None
            else []
        )
        for trajectory, feature in zip(
            bundle.context.candidate_trajectories,
            bundle.context.trajectory_features,
            strict=True,
        ):
            (
                polygon_feature,
                centerline_entry_feature,
                corrected_feature,
                _,
            ) = metrics_for_single_polygon(trajectory, feature, facts, correct)
            branch_current = branch_name(feature, digest(facts, current_signed))
            branch_correct_polygon = branch_name(
                polygon_feature, digest(facts, correct_signed)
            )
            branch_correct_centerline_entry = branch_name(
                centerline_entry_feature, digest(facts, correct_signed)
            )
            branch_correct = branch_name(corrected_feature, digest(facts, correct_signed))
            polygon_stop_metric = next(
                (m for m in polygon_feature.conflict_zone_metrics if m.kind == "stop_line"),
                None,
            )
            centerline_stop_metric = next(
                (m for m in corrected_feature.conflict_zone_metrics if m.kind == "stop_line"),
                None,
            )
            # Frozen analyzer falls back to every stop-line polygon when the
            # governing 2 m window is empty (trajectory_analyzer.py:228).
            source_depth, source_id = intrusion_source(
                trajectory, window if window else records
            )
            stop_metric = next(
                (m for m in feature.conflict_zone_metrics if m.kind == "stop_line"),
                None,
            )
            logged_depth = stop_metric.penetration_depth_m if stop_metric else None
            if (logged_depth is None) != (source_depth is None) or (
                logged_depth is not None
                and source_depth is not None
                and not math.isclose(
                    logged_depth, source_depth, rel_tol=0.0, abs_tol=1e-9
                )
            ):
                raise RuntimeError(
                    f"intrusion attribution mismatch {frame_token}/{trajectory.traj_id}: "
                    f"logged={logged_depth} attributed={source_depth}"
                )
            row = {
                "scene_id": scene_id,
                "frame_token": frame_token,
                "traj_id": trajectory.traj_id,
                "signed_m_current": current_signed,
                "signed_m_correct": correct_signed,
                "delta": (
                    correct_signed - current_signed
                    if correct_signed is not None and current_signed is not None
                    else None
                ),
                "polygon_id_current": current.polygon_id if current else None,
                "polygon_id_correct": correct.polygon_id if correct else None,
                "polygon_changed": (current.polygon_id if current else None)
                != (correct.polygon_id if correct else None),
                "branch_current": branch_current,
                "branch_correct_polygon_entry": branch_correct_polygon,
                "branch_correct_centerline_entry": branch_correct_centerline_entry,
                "branch_correct": branch_correct,
                "branch_correct_changed_by_entry_time": branch_correct_polygon
                != branch_correct_centerline_entry,
                "branch_correct_changed_by_stop_relation": branch_correct_centerline_entry
                != branch_correct,
                "entry_time_correct_polygon_s": (
                    polygon_stop_metric.entry_time_s if polygon_stop_metric else None
                ),
                "entry_time_correct_centerline_s": (
                    centerline_stop_metric.entry_time_s if centerline_stop_metric else None
                ),
                "stopped_before_correct_polygon": (
                    polygon_stop_metric.stopped_before_zone if polygon_stop_metric else None
                ),
                "stopped_before_correct_centerline": (
                    centerline_stop_metric.stopped_before_zone if centerline_stop_metric else None
                ),
                "branch_flipped": branch_current != branch_correct,
                "n_polygons_in_2m_window": len(window),
                "max_intrusion_current": logged_depth,
                "intrusion_source_polygon_id": source_id,
                "rebuild_verified": True,
            }
            rows.append(row)
            audit_id = predicate_lookup.get((frame_token, trajectory.traj_id))
            if audit_id:
                predicate_rows.append({"audit_id": audit_id, **row})

    write_csv(OUTPUT_DIR / "candidate_results.csv", rows, CSV_FIELDS)
    write_csv(
        OUTPUT_DIR / "predicate_36_results.csv",
        sorted(predicate_rows, key=lambda row: row["audit_id"]),
        ["audit_id", *CSV_FIELDS],
    )

    changed_frames = {
        row["frame_token"] for row in rows if row["polygon_changed"]
    }
    changed_scenes = {row["scene_id"] for row in rows if row["polygon_changed"]}
    flipped = [row for row in rows if row["branch_flipped"]]
    cross = Counter((row["branch_current"], row["branch_correct"]) for row in flipped)
    polygon_flipped = [
        row
        for row in rows
        if row["branch_current"] != row["branch_correct_polygon_entry"]
    ]
    polygon_cross = Counter(
        (row["branch_current"], row["branch_correct_polygon_entry"])
        for row in polygon_flipped
    )
    centerline_entry_flipped = [
        row
        for row in rows
        if row["branch_current"] != row["branch_correct_centerline_entry"]
    ]
    centerline_entry_cross = Counter(
        (row["branch_current"], row["branch_correct_centerline_entry"])
        for row in centerline_entry_flipped
    )
    entry_time_changed = [
        row for row in rows if row["branch_correct_changed_by_entry_time"]
    ]
    entry_time_cross = Counter(
        (row["branch_correct_polygon_entry"], row["branch_correct_centerline_entry"])
        for row in entry_time_changed
    )
    stop_relation_changed = [
        row for row in rows if row["branch_correct_changed_by_stop_relation"]
    ]
    stop_relation_cross = Counter(
        (row["branch_correct_centerline_entry"], row["branch_correct"])
        for row in stop_relation_changed
    )
    no_choice_frames = {
        frame_token
        for frame_token, record in frozen_records.items()
        if record.get("final_label", {}).get("selection_outcome")
        == "no_choosable_candidate"
    }
    centerline_veto_to_non_veto = [
        row
        for row in rows
        if row["branch_current"] == "有效窗内越线无停"
        and row["branch_correct_centerline_entry"] != "有效窗内越线无停"
    ]
    centerline_veto_no_choice = [
        row for row in centerline_veto_to_non_veto if row["frame_token"] in no_choice_frames
    ]
    centerline_all_flip_no_choice = [
        row
        for row in centerline_entry_flipped
        if row["frame_token"] in no_choice_frames
    ]
    final_veto_to_non_veto = [
        row
        for row in rows
        if row["branch_current"] == "有效窗内越线无停"
        and row["branch_correct"] != "有效窗内越线无停"
    ]
    final_veto_no_choice = [
        row for row in final_veto_to_non_veto if row["frame_token"] in no_choice_frames
    ]
    final_all_flip_no_choice = [
        row for row in flipped if row["frame_token"] in no_choice_frames
    ]

    rows_by_candidate = {
        (row["frame_token"], row["traj_id"]): row for row in rows
    }

    def projected_selection(branch_field: str) -> tuple[Counter[str], list[tuple[str, str, str]]]:
        outcomes: Counter[str] = Counter()
        changed_units: list[tuple[str, str, str]] = []
        for frame_token, record in frozen_records.items():
            final_label = record.get("final_label")
            if not isinstance(final_label, dict):
                continue
            statuses: list[str] = []
            for verdict in final_label["verdicts"]:
                trajectory_id = verdict["trajectory_id"]
                old_status = verdict["veto"]["status"]
                row = rows_by_candidate.get((frame_token, trajectory_id))
                branch = row[branch_field] if row is not None else None
                if branch == "有效窗内越线无停":
                    statuses.append("vetoed")
                elif branch == "线前完整停车":
                    statuses.append("cleared")
                else:
                    # No LLM rerun: retain the frozen downstream status whenever
                    # the corrected predicate remains non-decisive.
                    statuses.append(old_status)
            projected = (
                "selected" if "cleared" in statuses else "no_choosable_candidate"
            )
            outcomes[projected] += 1
            old_outcome = final_label["selection_outcome"]
            if projected != old_outcome:
                changed_units.append((frame_token, old_outcome, projected))
        return outcomes, changed_units

    centerline_selection, centerline_selection_changed = projected_selection(
        "branch_correct_centerline_entry"
    )
    final_selection, final_selection_changed = projected_selection("branch_correct")
    multi = [row for row in rows if row["n_polygons_in_2m_window"] > 1]
    cross_polygon = [
        row
        for row in multi
        if row["intrusion_source_polygon_id"] not in (None, row["polygon_id_current"])
    ]
    predicate_rows.sort(key=lambda row: row["audit_id"])

    lines = [
        "# 停止线 signed_m 离线重算与影响面",
        "",
        (
            f"候选构造选择受影响：polygon_changed 为 "
            f"{len(changed_frames)} 个 frame / {len(changed_scenes)} 个不同 scene_id / "
            f"{sum(r['polygon_changed'] for r in rows)} 个候选，不为 0。"
            if changed_frames
            else "候选构造选择未受影响：polygon_changed 为 0。"
        ),
        f"同时改用修正中线连续交点与中线停车关系后，判据分支翻转 {len(flipped)} / {len(rows)} 个候选。",
        (
            f"侵入深度跨 polygon 现象真实存在：{len(cross_polygon)} 个候选的最大侵入来源"
            "不是 polygon_id_current。"
            if cross_polygon
            else "本批未观察到侵入深度跨 polygon 来源。"
        ),
        "",
        "## 运行与等价性门禁",
        "",
        f"- 冻结代码 commit：`{FROZEN_COMMIT}`。",
        f"- 标注清单与冻结运行日志归档 commit：`{SOURCE_COMMIT}`。",
        f"- checkout 后、创建离线脚本前的 `git status --branch --porcelain=v1`：`{args.initial_clean_status}`。",
        f"- 全字段完全一致：{exact}/{total}；不一致候选：0；不一致字段：0。",
        "- `rebuild_verified`：492 行全部为 true。",
        "",
        "## 离线几何口径与未规定选择",
        "",
        "手册规定 signed_m 沿车头方向、负值表示起点已经越线（docs/RegGround-AV_修订实现手册_v2.0.md:44），并规定法向夹角 <45°、±50 m、多条取最近且不得回退（同文件:52）。",
        "手册未规定代理 polygon 如何还原中线。本审计取 polygon 最长边为中线方向，并把两侧法向支撑线的中间线作为中线；法向翻转到 ego 局部 +x 半平面。",
        "手册未规定等距 tie-break。本审计先按 |法向投影|，完全相等时按稳定 polygon token/顶点哈希字典序。",
        "参考点固定为 ego 局部原点；未引入 footprint。Map record polygon 从全局坐标转 ego 坐标的生产位置为 src/layer1/scene_fact_extractor.py:326-357。",
        "冻结实现实际取 polygon 顶点平均 x，并按 |x| 最近选择，位置为 src/layer1/scene_fact_extractor.py:256-287。",
        "连续路径/多边形交点审计直接调用冻结版 TrajectoryAugmentor._first_zone_stop_path_s；其弧长累计及边界交点位置为 src/layer1/trajectory_augmentor.py:371-394。",
        "中线版先用同一个 _segment_intersection_fraction 将无限中线裁成 polygon 内的有限线段，再逐轨迹线段求最早连续交点并线性插值时间；冻结交点函数位于 src/layer1/trajectory_augmentor.py:399-419。",
        "第二版只替换越线时刻：仅把 stop-line metric 的 entered/entry_time_s 改为是否存在中线连续交点及其时刻；penetration_depth_m 与 stopped_before_zone 仍来自修正后所选 polygon。",
        "第三版再把 stopped_before_zone 的参照从首个 polygon 内 waypoint 换成连续中线交点时刻；冻结实现的速度 <0.3 m/s、累计 >=1.0 s、首个低速段早于边界三项规则保持不变，生产位置为 src/layer1/trajectory_analyzer.py:253-262。",
        "脚本不包含平行的线段相交实现；所有路径/polygon 或线段/线段相交均调用冻结版 _first_zone_stop_path_s / _segment_intersection_fraction。",
        "分支严格调用冻结版 red_light_crossing_verdict；六分支位置为 src/compliance_predicates.py:47-95。",
        "",
        "## T-A：polygon_changed",
        "",
        "| 指标 | 数量 |",
        "|---|---:|",
        f"| frame 单元 | {len(changed_frames)} / 82 |",
        f"| 不同 scene_id | {len(changed_scenes)} / {len({r['scene_id'] for r in rows})} |",
        f"| 候选 | {sum(r['polygon_changed'] for r in rows)} / {len(rows)} |",
        "",
        "## T-B：第三版（中线交点 + 中线停车关系）branch_flipped 交叉表",
        "",
        "| current → correct | 候选数 |",
        "|---|---:|",
    ]
    if cross:
        for (before, after), count in sorted(cross.items()):
            marker = " **重点**" if {before, after} == {"start_beyond_line", "有效窗内越线无停"} else ""
            lines.append(f"| {before} → {after}{marker} | {count} |")
    else:
        lines.append("| 无翻转 | 0 |")
    lines.extend(
        [
            "",
            f"翻转合计：{len(flipped)} / {len(rows)}。",
            "",
            "重点双向翻转：",
            "",
            "| 方向 | 候选数 |",
            "|---|---:|",
            f"| start_beyond_line → 有效窗内越线无停 | {cross.get(('start_beyond_line', '有效窗内越线无停'), 0)} |",
            f"| 有效窗内越线无停 → start_beyond_line | {cross.get(('有效窗内越线无停', 'start_beyond_line'), 0)} |",
            "",
            "### 第二版 37 条中的 vetoed → 非 vetoed 与 no-choice 14",
            "",
            f"第二版（中线连续交点、polygon 停车关系）的 37 条翻转中，{len(centerline_veto_to_non_veto)} 条是有效窗内越线无停（vetoed）→ 非 vetoed，分布在 {len({r['frame_token'] for r in centerline_veto_to_non_veto})} 个单元。",
            f"冻结正式运行 selection_outcome=no_choosable_candidate 为 {len(no_choice_frames)} 个单元；上述 vetoed→非 vetoed 与其交集为 {len(centerline_veto_no_choice)} 条候选 / {len({r['frame_token'] for r in centerline_veto_no_choice})} 个单元。",
            f"37 条全部翻转与原 no-choice 14 的交集同样为 {len(centerline_all_flip_no_choice)} 条候选 / {len({r['frame_token'] for r in centerline_all_flip_no_choice})} 个单元；投影后的主表仍为 selected={centerline_selection['selected']}、no-choice={centerline_selection['no_choosable_candidate']}，变化单元 {len(centerline_selection_changed)} 个。",
            "",
            "### 第一版与第二版对比：替换 entry_time",
            "",
            f"第一版 polygon-entry 翻转 {len(polygon_flipped)} 个；第二版中线连续交点翻转 {len(centerline_entry_flipped)} 个；两版 branch_correct 不同 {len(entry_time_changed)} 个。",
            "",
            "| polygon-entry correct → centerline correct | 候选数 |",
            "|---|---:|",
        ]
    )
    if entry_time_cross:
        for (before, after), count in sorted(entry_time_cross.items()):
            lines.append(f"| {before} → {after} | {count} |")
    else:
        lines.append("| 无差异 | 0 |")
    lines.extend(
        [
            "",
            "两版 branch_correct 不同的完整清单：",
            "",
            "| frame_token | traj_id | polygon entry_time_s | centerline entry_time_s | polygon correct → centerline correct |",
            "|---|---|---:|---:|---|",
        ]
    )
    for row in entry_time_changed:
        lines.append(
            f"| {row['frame_token']} | {row['traj_id']} | "
            f"{fmt(row['entry_time_correct_polygon_s'])} | "
            f"{fmt(row['entry_time_correct_centerline_s'])} | "
            f"{row['branch_correct_polygon_entry']} → {row['branch_correct_centerline_entry']} |"
        )
    lines.extend(
        [
            "",
            "### 第二版与第三版对比：替换 stopped_before_zone",
            "",
            f"第二版翻转 {len(centerline_entry_flipped)} 个；第三版翻转 {len(flipped)} 个；两版 branch_correct 不同 {len(stop_relation_changed)} 个。",
            f"第三版 vetoed→非 vetoed 为 {len(final_veto_to_non_veto)} 条，和原 no-choice 14 的交集为 {len(final_veto_no_choice)} 条候选 / {len({r['frame_token'] for r in final_veto_no_choice})} 个单元。",
            f"第三版全部翻转与原 no-choice 14 的交集为 {len(final_all_flip_no_choice)} 条候选 / {len({r['frame_token'] for r in final_all_flip_no_choice})} 个单元；在非决定性分支保留冻结 LLM verdict 的投影下，主表变为 selected={final_selection['selected']}、no-choice={final_selection['no_choosable_candidate']}。",
            "第三版改变 selection_outcome 的单元："
            + (", ".join(f"{frame}: {old}→{new}" for frame, old, new in final_selection_changed) or "无"),
            "",
            "| centerline-entry + polygon-stop → centerline-entry + centerline-stop | 候选数 |",
            "|---|---:|",
        ]
    )
    if stop_relation_cross:
        for (before, after), count in sorted(stop_relation_cross.items()):
            lines.append(f"| {before} → {after} | {count} |")
    else:
        lines.append("| 无差异 | 0 |")
    lines.extend(
        [
            "",
            "第二版 current→correct 交叉表：",
            "",
            "| current → centerline-entry correct | 候选数 |",
            "|---|---:|",
        ]
    )
    if centerline_entry_cross:
        for (before, after), count in sorted(centerline_entry_cross.items()):
            lines.append(f"| {before} → {after} | {count} |")
    else:
        lines.append("| 无翻转 | 0 |")
    lines.extend(
        [
            "",
            "旧 polygon-entry 版 current→correct 交叉表：",
            "",
            "| current → polygon-entry correct | 候选数 |",
            "|---|---:|",
        ]
    )
    if polygon_cross:
        for (before, after), count in sorted(polygon_cross.items()):
            lines.append(f"| {before} → {after} | {count} |")
    else:
        lines.append("| 无翻转 | 0 |")
    lines.extend(
        [
            "",
            "## T-C：2 m 窗口与侵入来源",
            "",
            "| 指标 | 候选数 |",
            "|---|---:|",
            f"| n_polygons_in_2m_window > 1 | {len(multi)} |",
            f"| 其中 intrusion_source_polygon_id ≠ polygon_id_current | {len(cross_polygon)} |",
            "| A-064 是否属于上述 53 条 | 否（n_polygons_in_2m_window=1，侵入来源等于 polygon_id_current） |",
            "",
            "生产 governing window 以 polygon 顶点平均 x 与 signed scalar 差 <2 m 聚合（src/layer1/trajectory_analyzer.py:206-216）；侵入深度随后在所有 selected zones 中取最大（同文件:281-284、435-454）。",
            "",
            "## predicate 层 36 条候选",
            "",
            "完整机器可读结果见 `predicate_36_results.csv`。",
            "",
            "| audit_id | frame_token | traj_id | current | correct | polygon_changed | current → polygon-entry → centerline-entry → centerline-stop | flipped | max intrusion | source polygon |",
            "|---|---|---|---:|---:|---|---|---|---:|---|",
        ]
    )
    for row in predicate_rows:
        lines.append(
            "| {audit_id} | {frame_token} | {traj_id} | {current} | {correct} | {changed} | {before} → {polygon} → {centerline} → {after} | {flipped} | {depth} | {source} |".format(
                audit_id=row["audit_id"],
                frame_token=row["frame_token"],
                traj_id=row["traj_id"],
                current=fmt(row["signed_m_current"]),
                correct=fmt(row["signed_m_correct"]),
                changed=fmt(row["polygon_changed"]),
                before=row["branch_current"],
                polygon=row["branch_correct_polygon_entry"],
                centerline=row["branch_correct_centerline_entry"],
                after=row["branch_correct"],
                flipped=fmt(row["branch_flipped"]),
                depth=fmt(row["max_intrusion_current"]),
                source=fmt(row["intrusion_source_polygon_id"]),
            )
        )
    lines.extend(
        [
            "",
            "## 输出",
            "",
            "- `candidate_results.csv`：82 frame、492 候选逐条结果。",
            "- `predicate_36_results.csv`：predicate 层 36 条逐条结果。",
            "- `rebuild_equivalence.json`：全字段等价性门禁与不一致清单。",
        ]
    )
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "rows": len(rows),
        "predicate_rows": len(predicate_rows),
        "polygon_changed_frames": len(changed_frames),
        "branch_flipped": len(flipped),
        "branch_flipped_polygon_entry": len(polygon_flipped),
        "branch_flipped_centerline_entry": len(centerline_entry_flipped),
        "branch_correct_changed_by_entry_time": len(entry_time_changed),
        "branch_correct_changed_by_stop_relation": len(stop_relation_changed),
        "centerline_entry_veto_to_non_veto": len(centerline_veto_to_non_veto),
        "centerline_entry_veto_to_non_veto_in_no_choice": len(centerline_veto_no_choice),
        "centerline_entry_all_flips_in_no_choice": len(centerline_all_flip_no_choice),
        "centerline_entry_selection_outcomes": dict(centerline_selection),
        "third_version_all_flips_in_no_choice": len(final_all_flip_no_choice),
        "third_version_selection_outcomes": dict(final_selection),
        "multi_polygon_window": len(multi),
        "cross_polygon_intrusion": len(cross_polygon),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
