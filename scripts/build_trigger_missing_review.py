#!/usr/bin/env python3
# ruff: noqa: E501  # The embedded HTML/JS fragment is intentionally kept literal.
"""Build a deterministic 10-scene visual review pack for pedestrian trigger_missing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from src.layer1.models import RawScene
from src.layer1.scene_fact_extractor import (
    SceneFactExtractor,
    _global_to_ego,
    _pedestrian_footprint_radius,
    _point_overlaps_polygon,
    _quat_to_yaw,
)
from src.layer1.trajectory_extractor import TrajectoryExtractor

EXPECTED_POOL_SIZE = 108
SAMPLE_QUOTAS = {
    "no_forward_crosswalk_full_window": 6,
    "no_forward_crosswalk_short_window": 3,
    "forward_crosswalk_motion_unknown": 1,
}


def _rank(token: str, seed: int) -> str:
    return hashlib.sha256(
        f"trigger-missing-review:{seed}:{token}".encode()
    ).hexdigest()


def _stratum(facts: Any) -> str:
    if facts.pedestrian_in_forward_crosswalk:
        return "forward_crosswalk_motion_unknown"
    if facts.window_insufficient:
        return "no_forward_crosswalk_short_window"
    return "no_forward_crosswalk_full_window"


def _attributes(annotation: dict[str, Any]) -> list[str]:
    values = annotation.get("attribute_names", annotation.get("attributes", []))
    if not isinstance(values, list):
        return []
    return sorted(str(value) for value in values)


def _distance(points: list[list[float]]) -> float:
    return sum(
        math.hypot(current[0] - previous[0], current[1] - previous[1])
        for previous, current in zip(points, points[1:], strict=False)
    )


def _thin(points: list[list[float]], limit: int = 80) -> list[list[float]]:
    if len(points) <= limit:
        return points
    step = max(1, math.ceil(len(points) / limit))
    sampled = points[::step]
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return sampled


def _scene_payload(raw_scene: RawScene, facts: Any, sample_id: str) -> dict[str, Any]:
    extractor = TrajectoryExtractor()
    gt = extractor.extract(raw_scene)
    path = extractor.extract_path(raw_scene)
    gt_points = [[round(w.x, 3), round(w.y, 3)] for w in gt.waypoints]
    path_points = [[round(w.x, 3), round(w.y, 3)] for w in path.waypoints]

    crossings = [
        [[round(x, 3), round(y, 3)] for x, y in zone.polygon_xy]
        for zone in facts.conflict_zones
        if zone.kind == "ped_crossing" and len(zone.polygon_xy) >= 3
    ]
    production_path = [(point[0], point[1]) for point in gt_points]
    forward_flags = [
        SceneFactExtractor._path_intersects_polygon_buffer(
            production_path,
            [(point[0], point[1]) for point in polygon],
            3.0,
        )
        for polygon in crossings
    ]

    ego_pose = raw_scene.ego_poses[raw_scene.keyframe_index]
    translation = ego_pose["translation"]
    ego_xy = (float(translation[0]), float(translation[1]))
    ego_yaw = _quat_to_yaw(ego_pose["rotation"])
    if ego_yaw is None:
        raise ValueError(f"ego yaw unavailable: {raw_scene.frame_token}")

    pedestrians: list[dict[str, Any]] = []
    for annotation in raw_scene.annotations:
        category = str(annotation.get("category_name", "")).lower()
        if not category.startswith("human.pedestrian"):
            continue
        center = annotation.get("translation", [])
        if not isinstance(center, list | tuple) or len(center) < 2:
            continue
        point = _global_to_ego(
            (float(center[0]), float(center[1])),
            ego_xy,
            ego_yaw,
        )
        track: list[list[float]] = []
        track_times: list[float] = []
        for track_point in annotation.get("track", []):
            track_center = track_point.get("translation", [])
            if not isinstance(track_center, list | tuple) or len(track_center) < 2:
                continue
            local = _global_to_ego(
                (float(track_center[0]), float(track_center[1])),
                ego_xy,
                ego_yaw,
            )
            track.append([round(local[0], 3), round(local[1], 3)])
            if isinstance(track_point.get("t"), int | float):
                track_times.append(float(track_point["t"]))
        if not track:
            track = [[round(point[0], 3), round(point[1], 3)]]
        footprint_radius = _pedestrian_footprint_radius(annotation)
        inside_any = any(
            _point_overlaps_polygon(
                point,
                [(p[0], p[1]) for p in polygon],
                footprint_radius,
            )
            for polygon in crossings
        )
        inside_forward = any(
            is_forward
            and _point_overlaps_polygon(
                point,
                [(p[0], p[1]) for p in polygon],
                footprint_radius,
            )
            for polygon, is_forward in zip(crossings, forward_flags, strict=True)
        )
        pedestrians.append({
            "id": f"P{len(pedestrians) + 1}",
            "track_id": str(annotation.get("instance_token", "")),
            "attributes": _attributes(annotation),
            "keyframe": [round(point[0], 3), round(point[1], 3)],
            "track": _thin(track),
            "track_distance_m": round(_distance(track), 3),
            "track_duration_s": (
                round(max(track_times) - min(track_times), 3)
                if len(track_times) >= 2
                else 0.0
            ),
            "inside_any_crosswalk": inside_any,
            "inside_forward_crosswalk": inside_forward,
        })

    return {
        "sample_id": sample_id,
        "frame_token": raw_scene.frame_token,
        "scene_id": raw_scene.scene_id,
        "stratum": _stratum(facts),
        "facts": {
            "pedestrian_in_crosswalk": facts.pedestrian_in_crosswalk,
            "pedestrian_in_forward_crosswalk": facts.pedestrian_in_forward_crosswalk,
            "pedestrian_moving": facts.pedestrian_moving,
            "window_insufficient": facts.window_insufficient,
        },
        "gt_path": _thin(gt_points),
        "path_material": _thin(path_points),
        "crossings": [
            {"polygon": polygon, "forward": forward}
            for polygon, forward in zip(crossings, forward_flags, strict=True)
        ],
        "pedestrians": pedestrians,
    }


def _review_markdown(manifest: dict[str, Any]) -> str:
    rows = [
        "# Pedestrian trigger_missing 人工复查（10 场）",
        "",
        "门控问题：图中是否确实不存在“位于自车未来 6 秒路径相交横道内且正在移动”的行人？",
        "任一场判断为存在该行人，即记 `trigger_missing` 误判并触发标签/结构池复查。",
        "",
        "| 样本 | frame_token | 分层 | forward | moving | short | 人工结论 | 备注 |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for scene in manifest["samples"]:
        facts = scene["facts"]
        rows.append(
            f"| {scene['sample_id']} | `{scene['frame_token']}` | {scene['stratum']} | "
            f"{facts['pedestrian_in_forward_crosswalk']} | {facts['pedestrian_moving']} | "
            f"{facts['window_insufficient']} |  |  |"
        )
    return "\n".join(rows) + "\n"


def _inline_html(samples: list[dict[str, Any]]) -> str:
    template = r"""
<div id="ped-trigger-review-root">
  <div class="viz-controls">
    <label class="form-label" for="ped-trigger-scene">复查场景
      <select class="form-select" id="ped-trigger-scene"></select>
    </label>
    <span id="ped-trigger-progress" class="text-muted"></span>
  </div>
  <div id="ped-trigger-facts" class="viz-row text-small"></div>
  <svg id="ped-trigger-plot" role="img" aria-label="自车路径、人行横道和行人轨迹俯视图"></svg>
  <div class="viz-row text-small" aria-label="图例">
    <span><span class="legend-line gt"></span>6 秒路径</span>
    <span><span class="legend-line material"></span>12 秒路径素材</span>
    <span><span class="legend-box crossing"></span>横道</span>
    <span><span class="legend-box forward"></span>前向横道</span>
    <span><span class="legend-dot pedestrian"></span>行人/track</span>
  </div>
  <div class="table-responsive">
    <table class="table table-sm" id="ped-trigger-table">
      <thead><tr><th>行人</th><th>属性</th><th>横道内</th><th>前向横道内</th><th class="text-end">track 位移</th><th class="text-end">时长</th></tr></thead>
      <tbody></tbody>
    </table>
  </div>
  <div class="viz-controls" role="group" aria-label="人工复查结论">
    <label class="form-check"><input class="form-check-input" type="radio" name="ped-trigger-decision" value="correct"><span class="form-check-label">trigger_missing 正确</span></label>
    <label class="form-check"><input class="form-check-input" type="radio" name="ped-trigger-decision" value="incorrect"><span class="form-check-label">存在漏检，判定错误</span></label>
    <label class="form-check"><input class="form-check-input" type="radio" name="ped-trigger-decision" value="uncertain"><span class="form-check-label">无法确认</span></label>
    <label class="form-label" for="ped-trigger-note">备注
      <input class="form-control" id="ped-trigger-note" type="text" placeholder="可选：写下可疑行人或横道">
    </label>
    <button class="btn btn-primary" id="ped-trigger-send" type="button">发送复查结果</button>
  </div>
</div>
<style>
#ped-trigger-review-root { color: var(--foreground); }
#ped-trigger-plot { width: 100%; height: auto; aspect-ratio: 16 / 8; margin-top: 8px; }
#ped-trigger-plot .path-material { fill: none; stroke: var(--muted-foreground); stroke-width: 1.5; stroke-dasharray: 5 4; }
#ped-trigger-plot .gt-path { fill: none; stroke: var(--viz-series-1); stroke-width: 2.5; }
#ped-trigger-plot .crossing { fill: color-mix(in srgb, var(--viz-series-2) 18%, transparent); stroke: var(--viz-series-2); stroke-width: 1.2; }
#ped-trigger-plot .forward-crossing { fill: color-mix(in srgb, var(--viz-series-3) 24%, transparent); stroke: var(--viz-series-3); stroke-width: 2; }
#ped-trigger-plot .ped-track { fill: none; stroke: var(--viz-series-4); stroke-width: 1.7; }
#ped-trigger-plot .ped-point { fill: var(--viz-series-4); stroke: var(--background); stroke-width: 1.5; }
#ped-trigger-plot .ego { fill: var(--foreground); }
#ped-trigger-plot text { fill: var(--foreground); font-weight: 400; }
.legend-line { display: inline-block; width: 28px; border-top: 2px solid; margin-right: 5px; vertical-align: middle; }
.legend-line.gt { border-color: var(--viz-series-1); }
.legend-line.material { border-color: var(--muted-foreground); border-top-style: dashed; }
.legend-box { display: inline-block; width: 18px; height: 10px; margin-right: 5px; vertical-align: middle; border: 1px solid; }
.legend-box.crossing { border-color: var(--viz-series-2); background: color-mix(in srgb, var(--viz-series-2) 18%, transparent); }
.legend-box.forward { border-color: var(--viz-series-3); background: color-mix(in srgb, var(--viz-series-3) 24%, transparent); }
.legend-dot { display: inline-block; width: 9px; height: 9px; margin-right: 5px; vertical-align: middle; border-radius: 50%; }
.legend-dot.pedestrian { background: var(--viz-series-4); }
</style>
<script>
(() => {
  const root = document.getElementById("ped-trigger-review-root");
  const samples = __DATA__;
  const decisions = {};
  const select = root.querySelector("#ped-trigger-scene");
  const svg = root.querySelector("#ped-trigger-plot");
  const facts = root.querySelector("#ped-trigger-facts");
  const tableBody = root.querySelector("#ped-trigger-table tbody");
  const progress = root.querySelector("#ped-trigger-progress");
  const note = root.querySelector("#ped-trigger-note");
  const ns = "http://www.w3.org/2000/svg";

  samples.forEach((sample, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `${sample.sample_id} · ${sample.frame_token.slice(0, 8)} · ${sample.stratum}`;
    select.appendChild(option);
  });

  function pointsText(points) { return points.map(point => point.join(",")).join(" "); }
  function add(tag, attributes, className) {
    const node = document.createElementNS(ns, tag);
    Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, String(value)));
    if (className) node.setAttribute("class", className);
    svg.appendChild(node);
    return node;
  }
  function allPoints(sample) {
    return [sample.gt_path, sample.path_material, ...sample.crossings.map(item => item.polygon), ...sample.pedestrians.map(item => item.track)].flat();
  }
  function saveDecision() {
    const sample = samples[Number(select.value)];
    const checked = root.querySelector('input[name="ped-trigger-decision"]:checked');
    decisions[sample.sample_id] = { decision: checked?.value ?? null, note: note.value.trim(), frame_token: sample.frame_token };
  }
  function render() {
    const sample = samples[Number(select.value)];
    const pts = allPoints(sample);
    const xs = pts.map(point => point[0]);
    const ys = pts.map(point => point[1]);
    const minX = Math.min(...xs, -2); const maxX = Math.max(...xs, 8);
    const minY = Math.min(...ys, -4); const maxY = Math.max(...ys, 4);
    const pad = Math.max(3, 0.08 * Math.max(maxX - minX, maxY - minY));
    svg.setAttribute("viewBox", `${minX - pad} ${-(maxY + pad)} ${maxX - minX + 2 * pad} ${maxY - minY + 2 * pad}`);
    svg.innerHTML = "";
    add("polyline", { points: pointsText(sample.path_material.map(([x,y]) => [x,-y])) }, "path-material");
    sample.crossings.forEach(item => add("polygon", { points: pointsText(item.polygon.map(([x,y]) => [x,-y])) }, item.forward ? "forward-crossing" : "crossing"));
    add("polyline", { points: pointsText(sample.gt_path.map(([x,y]) => [x,-y])) }, "gt-path");
    add("circle", { cx: 0, cy: 0, r: Math.max(0.35, pad * 0.12) }, "ego");
    sample.pedestrians.forEach((ped, index) => {
      add("polyline", { points: pointsText(ped.track.map(([x,y]) => [x,-y])) }, "ped-track");
      add("circle", { cx: ped.keyframe[0], cy: -ped.keyframe[1], r: Math.max(0.4, pad * 0.14) }, "ped-point");
      const angle = (index % 8) * Math.PI / 4;
      const offset = Math.max(0.45, Math.min(0.9, pad * 0.18));
      const label = add("text", {
        x: ped.keyframe[0] + Math.cos(angle) * offset,
        y: -ped.keyframe[1] + Math.sin(angle) * offset,
        "font-size": Math.max(0.45, Math.min(0.85, pad * 0.16)),
        "text-anchor": Math.cos(angle) < -0.2 ? "end" : "start",
        "dominant-baseline": "middle"
      }, "ped-label");
      label.textContent = ped.id;
    });
    const f = sample.facts;
    facts.textContent = `${sample.sample_id} · forward=${f.pedestrian_in_forward_crosswalk} · moving=${f.pedestrian_moving} · short=${f.window_insufficient} · ${sample.pedestrians.length} pedestrians`;
    tableBody.innerHTML = "";
    sample.pedestrians.forEach(ped => {
      const row = document.createElement("tr");
      const cells = [ped.id, ped.attributes.join(", ") || "—", String(ped.inside_any_crosswalk), String(ped.inside_forward_crosswalk), `${ped.track_distance_m.toFixed(2)} m`, `${ped.track_duration_s.toFixed(2)} s`];
      cells.forEach((value, index) => {
        const cell = document.createElement("td");
        cell.textContent = value;
        if (index >= 4) cell.className = "text-end";
        row.appendChild(cell);
      });
      tableBody.appendChild(row);
    });
    root.querySelectorAll('input[name="ped-trigger-decision"]').forEach(input => { input.checked = input.value === decisions[sample.sample_id]?.decision; });
    note.value = decisions[sample.sample_id]?.note ?? "";
    progress.textContent = `已复查 ${Object.values(decisions).filter(item => item.decision).length}/10`;
  }
  select.addEventListener("change", () => { saveDecision(); render(); });
  root.querySelectorAll('input[name="ped-trigger-decision"]').forEach(input => input.addEventListener("change", () => { saveDecision(); render(); }));
  note.addEventListener("change", saveDecision);
  root.querySelector("#ped-trigger-send").addEventListener("click", async () => {
    saveDecision();
    const ordered = samples.map(sample => decisions[sample.sample_id] ?? { decision: null, note: "", frame_token: sample.frame_token });
    const prompt = [
      "pedestrian trigger_missing 人工复查结果：",
      JSON.stringify(ordered, null, 2),
      "请据此判定人工核查门控，并在发现误判时定位标签/结构池修复。"
    ].join(String.fromCharCode(10));
    if (window.openai?.sendFollowUpMessage) await window.openai.sendFollowUpMessage({ prompt, title: "提交 trigger_missing 复查" });
  });
  select.value = "0";
  render();
})();
</script>
"""
    return template.replace(
        "__DATA__",
        json.dumps(samples, ensure_ascii=False, separators=(",", ":")),
    ).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--eval-set", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inline-html", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_records = json.loads(args.input.read_text(encoding="utf-8"))
    by_token = {record["frame_token"]: record for record in raw_records}
    eval_set = json.loads(args.eval_set.read_text(encoding="utf-8"))
    extractor = SceneFactExtractor()
    pool: list[tuple[Any, RawScene]] = []
    for entry in eval_set["entries"]:
        if entry["scenario_type"] != "pedestrian":
            continue
        raw_scene = RawScene.model_validate(by_token[entry["frame_token"]])
        facts = extractor.extract(raw_scene)
        if (
            facts.pedestrian_in_forward_crosswalk is not True
            or facts.pedestrian_moving is not True
        ):
            pool.append((facts, raw_scene))
    if len(pool) != EXPECTED_POOL_SIZE:
        raise ValueError(f"trigger_missing pool changed: {len(pool)} != {EXPECTED_POOL_SIZE}")

    selected: list[tuple[Any, RawScene]] = []
    for stratum, quota in SAMPLE_QUOTAS.items():
        candidates = sorted(
            (item for item in pool if _stratum(item[0]) == stratum),
            key=lambda item: _rank(item[1].frame_token, args.seed),
        )
        if len(candidates) < quota:
            raise ValueError(f"{stratum} has {len(candidates)} scenes, needs {quota}")
        selected.extend(candidates[:quota])
    selected.sort(key=lambda item: _rank(item[1].frame_token, args.seed))
    samples = [
        _scene_payload(raw_scene, facts, f"TM-{index:02d}")
        for index, (facts, raw_scene) in enumerate(selected, 1)
    ]
    manifest = {
        "version": "trigger-missing-review-v2-post-footprint-fix",
        "seed": args.seed,
        "input": str(args.input),
        "eval_set": str(args.eval_set),
        "pool_size": len(pool),
        "pool_strata": dict(sorted(Counter(_stratum(facts) for facts, _ in pool).items())),
        "sample_quotas": SAMPLE_QUOTAS,
        "selection_rule": "SHA-256 rank within fixed 6/3/1 risk strata",
        "gate_rule": (
            "any reviewed scene containing a moving pedestrian in a forward "
            "path-intersecting crosswalk triggers label/pool remediation"
        ),
        "samples": samples,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "review.md").write_text(
        _review_markdown(manifest),
        encoding="utf-8",
    )
    args.inline_html.parent.mkdir(parents=True, exist_ok=True)
    args.inline_html.write_text(_inline_html(samples), encoding="utf-8")
    print(json.dumps({
        "pool_size": len(pool),
        "pool_strata": manifest["pool_strata"],
        "sample_count": len(samples),
        "sample_strata": dict(sorted(Counter(s["stratum"] for s in samples).items())),
        "manifest": str(args.output_dir / "manifest.json"),
        "inline_html": str(args.inline_html),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
