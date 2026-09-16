"""生成无标签泄露的轨迹几何谓词人工审计材料。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from src.layer1.facade import parse_scene
from src.layer1.models import ParsedSceneBundle, RawScene, Trajectory, Waypoint

AUDIT_VARIANTS = (
    "ground_truth",
    "conservative",
    "suboptimal",
    "aggressive",
    "hard_case",
    "illegal",
)
ANSWER_FIELDS = (
    "audit_id",
    "human_entered_proxy_zone",
    "human_stopped_before_zone",
    "human_verdict",
    "proxy_alignment",
    "disagreement_cause",
    "notes",
)


def build_geometry_audit(
    input_path: Path,
    output_dir: Path,
    *,
    record_indices: tuple[int, ...] = (6, 8),
    seed: int = 42,
    stratified_target: int | None = None,
) -> Path:
    """为指定 records 的全部六种候选生成单图盲审包。"""
    records = json.loads(input_path.read_text(encoding="utf-8"))
    selected: list[tuple[int, ParsedSceneBundle, dict[str, Any]]] = []
    if stratified_target is None:
        for index in record_indices:
            if index < 0 or index >= len(records):
                raise ValueError(f"record index out of range: {index}")
            bundle = parse_scene(RawScene.model_validate(records[index]), seed=seed)
            labels = {
                label.variant_type.value: label.model_dump(mode="json")
                for label in bundle.benchmark_labels.labels
            }
            missing = set(AUDIT_VARIANTS) - set(labels)
            if missing:
                raise ValueError(f"record {index} missing audit variants: {sorted(missing)}")
            selected.extend((index, bundle, labels[variant]) for variant in AUDIT_VARIANTS)
    else:
        if not 30 <= stratified_target <= 50:
            raise ValueError("stratified_target must be between 30 and 50")
        mandatory: list[tuple[int, ParsedSceneBundle, dict[str, Any]]] = []
        cleared_pool: list[tuple[int, ParsedSceneBundle, dict[str, Any]]] = []
        for index, record in enumerate(records):
            bundle = parse_scene(RawScene.model_validate(record), seed=seed)
            if bundle.context.scenario_type.value != "red_light":
                continue
            for model in bundle.benchmark_labels.labels:
                label = model.model_dump(mode="json")
                candidate = (index, bundle, label)
                is_mandatory = (
                    label["variant_type"] == "hard_case"
                    or label["expected_verdict"] in {"vetoed", "exclude"}
                    or label["is_degenerate"]
                    or (
                        index == 6
                        and label["anonymous_id"] in {"traj_b", "traj_f"}
                    )
                )
                if is_mandatory:
                    mandatory.append(candidate)
                elif label["expected_verdict"] == "cleared":
                    cleared_pool.append(candidate)
        rng = random.Random(seed)
        rng.shuffle(cleared_pool)
        selected = mandatory + cleared_pool[:max(0, stratified_target - len(mandatory))]

    random.Random(seed).shuffle(selected)
    blind_dir = output_dir / "blind_images"
    blind_dir.mkdir(parents=True, exist_ok=True)
    sidecar_rows: list[dict[str, Any]] = []
    for number, (record_index, bundle, label) in enumerate(selected, start=1):
        audit_id = f"audit_{number:03d}"
        trajectory_id = str(label["anonymous_id"])
        candidate_index = next(
            index
            for index, trajectory in enumerate(bundle.context.candidate_trajectories)
            if trajectory.traj_id == trajectory_id
        )
        trajectory = bundle.context.candidate_trajectories[candidate_index]
        features = bundle.context.trajectory_features[candidate_index]
        _render_blind_image(bundle, trajectory, audit_id, blind_dir / f"{audit_id}.png")
        predicate_verdict = (
            label["expected_verdict"]
            if label["expected_verdict"] in {"cleared", "vetoed"}
            else "uncertain"
        )
        sidecar_rows.append({
            "audit_id": audit_id,
            "record_index": record_index,
            "frame_token": bundle.context.frame_token,
            "candidate_id": trajectory_id,
            "variant": label["variant_type"],
            "entered_conflict_zone": features.entered_conflict_zone,
            "stopped_before_zone": features.stopped_before_zone,
            "predicate_verdict": predicate_verdict,
            "benchmark_expected_verdict": label["expected_verdict"],
            "predicate_version": label["predicate_version"],
            "is_degenerate": label["is_degenerate"],
        })

    answers_path = output_dir / "blind_answers.csv"
    with answers_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=ANSWER_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in sidecar_rows:
            writer.writerow({"audit_id": row["audit_id"]})

    sidecar_path = output_dir / "predicate_sidecar.jsonl"
    sidecar_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sidecar_rows),
        encoding="utf-8",
    )
    (output_dir / "audit_instructions.md").write_text(
        "\n".join([
            "# Predicate audit v2.1 blind questions",
            "",
            "> 盲判完成前不要读取 `predicate_sidecar.jsonl`。",
            "",
            "对每个 `audit_id` 填写 `blind_answers.csv`：",
            "",
            "1. `human_entered_proxy_zone`：是 / 否 / 图上无法判断",
            "2. `human_stopped_before_zone`：是 / 否 / NA / 图上无法判断",
            "3. `human_verdict`：cleared / vetoed / uncertain",
            "4. `proxy_alignment`：一致 / 偏移 / 无法判断",
            "",
            "仅在发生分歧时填写 `disagreement_cause`："
            "谓词 bug / 代理偏移 / 本质模糊。",
            "",
        ]),
        encoding="utf-8",
    )
    return answers_path


def _render_blind_image(
    bundle: ParsedSceneBundle,
    trajectory: Trajectory,
    audit_id: str,
    output_path: Path,
) -> None:
    """只绘制几何证据；不得读取或显示 benchmark/predicate 标签。"""
    points = [(waypoint.x, waypoint.y) for waypoint in trajectory.waypoints]
    segments = [[points[index], points[index + 1]] for index in range(len(points) - 1)]
    speeds = [_segment_speed(trajectory.waypoints[index], trajectory.waypoints[index + 1])
              for index in range(len(points) - 1)]
    speed_max = max(max(speeds, default=0.0), 0.1)
    norm = Normalize(vmin=0.0, vmax=speed_max)

    figure, axis = plt.subplots(figsize=(8, 7))
    line = LineCollection(segments, cmap="viridis", norm=norm, linewidth=3)
    line.set_array(speeds)
    axis.add_collection(line)
    figure.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=axis, label="Speed (m/s)")

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    axis.scatter([xs[0]], [ys[0]], color="#2e7d32", s=55, marker="o", label="Ego start")
    axis.scatter([xs[-1]], [ys[-1]], color="#c62828", s=60, marker="X", label="End")
    dx, dy = _initial_direction(points)
    axis.arrow(xs[0], ys[0], dx, dy, width=0.05, head_width=0.45,
               color="#2e7d32", length_includes_head=True)

    x_min, x_max = min(xs) - 8.0, max(xs) + 8.0
    y_min, y_max = min(ys) - 8.0, max(ys) + 8.0
    for zone in bundle.context.scene_facts.conflict_zones:
        polygon = zone.polygon_xy
        if len(polygon) < 3 or not _bbox_intersects(polygon, x_min, x_max, y_min, y_max):
            continue
        polygon_x = [point[0] for point in polygon] + [polygon[0][0]]
        polygon_y = [point[1] for point in polygon] + [polygon[0][1]]
        if zone.kind == "stop_line":
            axis.fill(polygon_x, polygon_y, color="#ef6c00", alpha=0.2,
                      label="Stop-line proxy")
            axis.plot(polygon_x, polygon_y, color="#ef6c00", linewidth=1.8)
        elif zone.kind == "ped_crossing":
            axis.fill(polygon_x, polygon_y, color="#1565c0", alpha=0.16,
                      label="Pedestrian crossing")
            axis.plot(polygon_x, polygon_y, color="#1565c0", linewidth=1.5)

    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Longitudinal position (m)")
    axis.set_ylabel("Lateral position (m)")
    axis.set_title(audit_id)
    axis.grid(alpha=0.25)
    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))
    axis.legend(unique.values(), unique.keys(), loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def join_geometry_audit(output_dir: Path) -> Path:
    """盲判完成后 join sidecar，并生成指标与工程 gate 报告。"""
    answers = _read_csv(output_dir / "blind_answers.csv")
    sidecar = {
        row["audit_id"]: row
        for row in _read_jsonl(output_dir / "predicate_sidecar.jsonl")
    }
    joined_rows: list[dict[str, Any]] = []
    for answer in answers:
        audit_id = answer["audit_id"]
        if not answer.get("human_verdict"):
            raise ValueError(f"blind answer is incomplete: {audit_id}")
        if audit_id not in sidecar:
            raise ValueError(f"audit_id missing from sidecar: {audit_id}")
        cause = answer.get("disagreement_cause", "")
        if cause and cause not in {"谓词 bug", "代理偏移", "本质模糊"}:
            raise ValueError(f"invalid disagreement_cause for {audit_id}: {cause}")
        joined_rows.append({**sidecar[audit_id], **answer})

    joined_path = output_dir / "joined_audit.csv"
    fieldnames = list(joined_rows[0]) if joined_rows else []
    with joined_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(joined_rows)
    (output_dir / "audit_report.md").write_text(
        _render_audit_report(joined_rows),
        encoding="utf-8",
    )
    return joined_path


def _render_audit_report(rows: list[dict[str, Any]]) -> str:
    determinable = [row for row in rows if row["human_verdict"] != "uncertain"]
    agreements = sum(
        row["human_verdict"] == row["predicate_verdict"] for row in determinable
    )
    rate = agreements / len(determinable) if determinable else None
    veto_disagreements = [
        row for row in rows
        if row["predicate_verdict"] == "vetoed"
        and row["human_verdict"] != "vetoed"
    ]
    unresolved_vetoes = [row for row in veto_disagreements if not row["disagreement_cause"]]
    proxy_counts = Counter(row["proxy_alignment"] for row in rows)
    entered_matches = sum(
        _answer_bool(row["human_entered_proxy_zone"]) == row["entered_conflict_zone"]
        for row in rows if _answer_bool(row["human_entered_proxy_zone"]) is not None
    )
    entered_total = sum(
        _answer_bool(row["human_entered_proxy_zone"]) is not None for row in rows
    )
    stopped_matches = sum(
        _answer_bool(row["human_stopped_before_zone"]) == row["stopped_before_zone"]
        for row in rows if _answer_bool(row["human_stopped_before_zone"]) is not None
    )
    stopped_total = sum(
        _answer_bool(row["human_stopped_before_zone"]) is not None for row in rows
    )
    gate_passed = rate is not None and rate >= 0.95 and not unresolved_vetoes
    rate_text = "not evaluable" if rate is None else f"{rate:.1%}"
    return "\n".join([
        "# Predicate geometry audit v2.1",
        "",
        f"- Predicate-human agreement：{agreements}/{len(determinable)} ({rate_text})",
        f"- entered_conflict_zone agreement：{entered_matches}/{entered_total}",
        f"- stopped_before_zone agreement：{stopped_matches}/{stopped_total}",
        f"- Proxy alignment distribution：{dict(sorted(proxy_counts.items()))}",
        f"- Veto disagreements without root cause：{len(unresolved_vetoes)}",
        f"- Engineering gate：{'PASS' if gate_passed else 'FAIL'}",
        "",
        "> 单人审计只作为工程验证，不作为论文外部效标。",
        "",
    ])


def _segment_speed(start: Waypoint, end: Waypoint) -> float:
    dt = end.t - start.t
    if dt <= 0:
        return 0.0
    return math.hypot(end.x - start.x, end.y - start.y) / dt


def _initial_direction(points: list[tuple[float, float]]) -> tuple[float, float]:
    for start, end in zip(points, points[1:], strict=False):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length > 0:
            return dx / length * 1.8, dy / length * 1.8
    return 1.8, 0.0


def _bbox_intersects(
    polygon: list[tuple[float, float]],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> bool:
    polygon_x = [point[0] for point in polygon]
    polygon_y = [point[1] for point in polygon]
    return not (
        max(polygon_x) < x_min or min(polygon_x) > x_max
        or max(polygon_y) < y_min or min(polygon_y) > y_max
    )


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _answer_bool(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized in {"yes", "是", "true"}:
        return True
    if normalized in {"no", "否", "false"}:
        return False
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="生成或 join v2.1 几何谓词盲审材料")
    parser.add_argument("input", type=Path, nargs="?")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--record-index", type=int, action="append", dest="record_indices")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratified-target", type=int)
    parser.add_argument("--join", action="store_true")
    args = parser.parse_args()
    if args.join:
        print(join_geometry_audit(args.output_dir))
        return
    if args.input is None:
        raise SystemExit("input is required unless --join is used")
    record_indices = tuple(args.record_indices or (6, 8))
    print(build_geometry_audit(
        args.input,
        args.output_dir,
        record_indices=record_indices,
        seed=args.seed,
        stratified_target=args.stratified_target,
    ))


if __name__ == "__main__":
    main()
