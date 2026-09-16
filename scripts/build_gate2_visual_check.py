"""生成 RegGround-AV v2 Gate 2 的十场景人工可视化核对包。"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
from typing import Any

from src.layer1.facade import parse_scene
from src.layer1.geometry_audit import _render_blind_image
from src.layer1.models import RawScene, Trajectory, TrajectoryFeatures
from src.layer3.context_builder import ContextBuilder

DEFAULT_SELECTION: tuple[tuple[str, str], ...] = (
    ("0dc8558714314cce961af9e7de403517", "pedestrian"),
    ("0525cf0e18474920bcd99b9526fd4649", "oncoming"),
    ("ce63dd8c7f6a45f9bdee5b7305388314", "red_light"),
    ("b7020d8854b449acb08134f26e2b380f", "red_light"),
    ("9a8cb194cb624978be61a06d5727c186", "oncoming"),
    ("5292f0009e4644acb946f0a9eaae4e37", "red_light"),
    ("b50f3fe7fdcc416396a941dd1026dd8e", "oncoming"),
    ("a6176dbfbeb44e11a42282a4c4fbeb45", "red_light"),
    ("0461602b49ce4158b5a00169acf3b19d", "pedestrian"),
    ("6c04bc0574694cfba42504c69ad98d5a", "pedestrian"),
)

VARIANT_ZH = {
    "ground_truth": "真实轨迹",
    "conservative": "保守候选",
    "suboptimal": "次优候选",
    "aggressive": "激进候选",
    "hard_case": "临界候选",
    "illegal": "违规扰动候选",
}


def build_visual_check(input_path: Path, output_dir: Path) -> Path:
    records = json.loads(input_path.read_text(encoding="utf-8"))
    by_frame = {
        record.get("frame_token"): record
        for record in records
        if isinstance(record, dict) and isinstance(record.get("frame_token"), str)
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    scenes: list[dict[str, Any]] = []

    for scene_number, (frame_token, expected_scenario) in enumerate(
        DEFAULT_SELECTION,
        start=1,
    ):
        record = by_frame.get(frame_token)
        if record is None:
            raise ValueError(f"frame token missing from input: {frame_token}")
        bundle = parse_scene(RawScene.model_validate(record), seed=42)
        scenario = bundle.context.scenario_type.value
        if scenario != expected_scenario:
            raise ValueError(
                f"scenario changed for {frame_token}: {scenario} != {expected_scenario}"
            )
        scene_dir = output_dir / f"scene_{scene_number:02d}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        label_by_id = {
            label.anonymous_id: label
            for label in bundle.benchmark_labels.labels
        }
        candidates: list[dict[str, Any]] = []
        for trajectory, feature in zip(
            bundle.context.candidate_trajectories,
            bundle.context.trajectory_features,
            strict=True,
        ):
            label = label_by_id[trajectory.traj_id]
            image_name = f"{trajectory.traj_id}_{label.variant_type.value}.png"
            image_path = scene_dir / image_name
            _render_blind_image(
                bundle,
                trajectory,
                f"scene_{scene_number:02d} · {trajectory.traj_id}",
                image_path,
            )
            candidates.append({
                "trajectory_id": trajectory.traj_id,
                "variant": label.variant_type.value,
                "variant_zh": VARIANT_ZH[label.variant_type.value],
                "chinese_explanation": _chinese_explanation(
                    feature,
                    scenario,
                    _final_segment_speed(trajectory),
                ),
                "image": f"scene_{scene_number:02d}/{image_name}",
                "image_sha256": _sha256(image_path),
                "features": {
                    "max_speed_mps": round(feature.max_speed_mps, 3),
                    "min_speed_mps": round(feature.min_speed_mps, 3),
                    "mean_speed_mps": round(feature.mean_speed_mps, 3),
                    "total_distance_m": round(feature.total_distance_m, 3),
                    "full_stop": feature.full_stop,
                    "stop_duration_s": round(feature.stop_duration_s, 3),
                    "conflict_zone_metrics": [
                        metric.model_dump(mode="json")
                        for metric in feature.conflict_zone_metrics
                    ],
                    "agent_interactions": [
                        interaction.model_dump(mode="json", exclude={"agent_track_id"})
                        for interaction in feature.agent_interactions
                    ],
                    "summary": feature.natural_language_summary,
                },
            })
        scenes.append({
            "scene_number": scene_number,
            "scene_id": bundle.context.scene_id,
            "frame_token": frame_token,
            "scenario_type": scenario,
            "window_insufficient": bundle.context.scene_facts.window_insufficient,
            "scene_facts_digest": bundle.judge_input.scene_facts_digest.model_dump(mode="json"),
            "scene_facts_text": ContextBuilder.render_scene_facts_digest(
                bundle.judge_input.scene_facts_digest
            ),
            "candidates": candidates,
        })

    manifest = {
        "version": "gate2-v2-2026-07-17",
        "input": str(input_path),
        "scene_count": len(scenes),
        "candidate_image_count": sum(len(scene["candidates"]) for scene in scenes),
        "scenes": scenes,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "manual_checklist.md").write_text(
        _render_checklist(scenes),
        encoding="utf-8",
    )
    index_path = output_dir / "index.html"
    index_path.write_text(_render_index(scenes), encoding="utf-8")
    return index_path


def _render_checklist(scenes: list[dict[str, Any]]) -> str:
    rows = [
        "# Gate 2 · 十场景人工核对表",
        "",
        "逐场景检查：轨迹/冲突区绘制、摘要数值、digest 投影、交互时序。",
        "不要依据 variant 名称判断合规性；这里只核对事实是否与图一致。",
        "",
        "| 场景 | 类型 | 轨迹与区域 | 特征数值 | digest | 交互时序 | 备注 |",
        "|---|---|---|---|---|---|---|",
    ]
    rows.extend(
        f"| scene_{scene['scene_number']:02d} | {scene['scenario_type']} "
        "| ☐ | ☐ | ☐ | ☐ | |"
        for scene in scenes
    )
    return "\n".join(rows) + "\n"


def _render_index(scenes: list[dict[str, Any]]) -> str:
    sections = "".join(_render_scene(scene) for scene in scenes)
    nav_links = "".join(
        f'<a href="#scene-{scene["scene_number"]:02d}">'
        f'scene_{scene["scene_number"]:02d}</a>'
        for scene in scenes
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RegGround-AV Gate 2 Visual Check</title>
<style>
:root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
body {{ margin: 24px; max-width: 1500px; }}
nav {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 24px; }}
nav a {{ padding: 6px 10px; border: 1px solid currentColor; border-radius: 6px; }}
article {{ margin: 0 0 44px; border-top: 2px solid currentColor; padding-top: 16px; }}
.facts {{ white-space: pre-wrap; padding: 12px; border: 1px solid currentColor; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 18px; }}
figure {{ margin: 0; border: 1px solid currentColor; padding: 10px; }}
img {{ display: block; width: 100%; height: auto; }}
figcaption {{ margin-top: 8px; }}
pre {{ white-space: pre-wrap; overflow-wrap: anywhere; }}
.checks {{ display: flex; flex-wrap: wrap; gap: 18px; margin: 12px 0; }}
</style>
</head>
<body>
<h1>RegGround-AV Gate 2 · 十场景人工核对</h1>
<p>共 10 场景、60 张候选轨迹图。图中绿色圆点为起点、红色 X 为终点，颜色条表示速度。</p>
<nav>{nav_links}</nav>
{sections}
</body>
</html>
"""


def _render_scene(scene: dict[str, Any]) -> str:
    candidates = "".join(_render_candidate(candidate) for candidate in scene["candidates"])
    short_note = "是" if scene["window_insufficient"] else "否"
    return f"""<article id="scene-{scene['scene_number']:02d}">
<h2>scene_{scene['scene_number']:02d} · {html.escape(scene['scenario_type'])}</h2>
<p><code>{html.escape(scene['frame_token'])}</code> · window_insufficient: {short_note}</p>
<div class="facts">{html.escape(scene['scene_facts_text'])}</div>
<div class="checks">
<label><input type="checkbox"> 轨迹/区域一致</label>
<label><input type="checkbox"> 特征数值一致</label>
<label><input type="checkbox"> digest 一致</label>
<label><input type="checkbox"> 交互时序一致</label>
</div>
<div class="grid">{candidates}</div>
</article>"""


def _render_candidate(candidate: dict[str, Any]) -> str:
    feature_text = json.dumps(candidate["features"], ensure_ascii=False, indent=2)
    trajectory_id = html.escape(candidate["trajectory_id"])
    variant = html.escape(candidate["variant"])
    variant_zh = html.escape(candidate["variant_zh"])
    explanation = html.escape(candidate["chinese_explanation"])
    return f"""<figure>
<img src="{html.escape(candidate['image'])}" alt="{trajectory_id} 轨迹图">
<figcaption><strong>{trajectory_id}</strong> · {variant}（{variant_zh}）</figcaption>
<p>{explanation}</p>
<details><summary>特征值</summary><pre>{html.escape(feature_text)}</pre></details>
</figure>"""


def _chinese_explanation(
    feature: TrajectoryFeatures,
    scenario: str,
    final_speed_mps: float,
) -> str:
    """生成一条中性中文事实句，不输出合规结论。"""
    parts = [
        f"平均速度 {feature.mean_speed_mps:.1f} m/s，行程 {feature.total_distance_m:.1f} m"
    ]
    relevant_kind = {
        "red_light": "stop_line",
        "pedestrian": "ped_crossing",
    }.get(scenario)
    metric = next((
        item
        for item in feature.conflict_zone_metrics
        if item.governing and (relevant_kind is None or item.kind == relevant_kind)
    ), None)
    if metric is None and feature.conflict_zone_metrics:
        metric = feature.conflict_zone_metrics[0]
    if metric is None:
        parts.append("未检测到适用冲突区几何")
    else:
        zone_name = "本向停止线" if metric.kind == "stop_line" else "前方人行横道"
        if metric.entered and metric.entry_time_s is not None:
            parts.append(f"{metric.entry_time_s:.1f} s 后进入{zone_name}区域")
        else:
            parts.append(f"全程未进入{zone_name}区域")
        if metric.stopped_before_zone is True:
            parts.append("区前存在完整停车")
        elif metric.stopped_before_zone is False:
            parts.append("区前未检测到完整停车")
        else:
            parts.append("区前停车关系不可判定")
        if metric.ended_inside or metric.ended_inside_other_zone:
            motion = (
                "末段达到静止阈值"
                if final_speed_mps < 0.3
                else "末段仍在移动，并非完整停车"
            )
            endpoint_clause = (
                f"预测窗终点仍位于{zone_name}区域"
                if metric.ended_inside
                else "预测窗终点位于" + (
                    "另一停止线代理区域（非本向）"
                    if metric.kind == "stop_line"
                    else "另一人行横道区域（非当前适用区域）"
                )
            )
            parts.append(
                f"{endpoint_clause}，末段速度 {final_speed_mps:.1f} m/s，"
                f"{motion}"
            )
    interaction = next(
        (
            item
            for item in feature.agent_interactions
            if item.crossing_order != "no_shared_zone"
        ),
        None,
    )
    if interaction is not None and interaction.min_time_gap_s is not None:
        subject = "行人" if interaction.agent_kind == "pedestrian" else "对向车辆"
        parts.append(f"与{subject}的最小时间间隔为 {interaction.min_time_gap_s:.1f} s")
    return "；".join(parts) + "。"


def _final_segment_speed(trajectory: Trajectory) -> float:
    waypoints = trajectory.waypoints
    if len(waypoints) < 2:
        return 0.0
    start, end = waypoints[-2], waypoints[-1]
    duration = end.t - start.t
    if duration <= 0.0:
        return 0.0
    distance = ((end.x - start.x) ** 2 + (end.y - start.y) ** 2) ** 0.5
    return float(distance / duration)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/layer1/raw_scene_trainval.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/gate2_visual_check_v2"),
    )
    args = parser.parse_args()
    print(build_visual_check(args.input, args.output))


if __name__ == "__main__":
    main()
