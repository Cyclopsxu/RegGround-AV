"""生成冻结 20 场景 idx 6 的速度剖面与确定性打标证据。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from src.layer1.facade import parse_scene
from src.layer1.models import RawScene, Trajectory, TrajectoryFeatures
from src.layer1.scene_fact_extractor import SceneFactExtractor
from src.layer1.trajectory_analyzer import TrajectoryAnalyzer
from src.layer1.trajectory_augmentor import TrajectoryAugmentor
from src.layer1.trajectory_extractor import TrajectoryExtractor

FloatArray = NDArray[np.float64]
LOW_SPEED_THRESHOLD_MPS = 0.3
EXPECTED_FRAME_TOKEN = "8700a1f81cf94ad99345dd94a3b7eda8"
VARIANT_CONFIG: dict[str, dict[str, float | str | None]] = {
    "ground_truth": {"speed_scale": 1.0, "speed_floor_ratio": None, "profile": "observed"},
    "conservative": {"speed_scale": 0.6, "speed_floor_ratio": 0.0, "profile": "constant"},
    "suboptimal": {"speed_scale": 0.9, "speed_floor_ratio": 0.0, "profile": "hesitant"},
    "aggressive": {"speed_scale": 1.3, "speed_floor_ratio": 0.0, "profile": "constant"},
    "hard_case": {"speed_scale": 1.0, "speed_floor_ratio": 0.0, "profile": "creeping"},
    "illegal": {"speed_scale": 3.0, "speed_floor_ratio": 1.0, "profile": "constant"},
}
PLOT_ORDER = (
    "ground_truth",
    "conservative",
    "suboptimal",
    "aggressive",
    "hard_case",
    "illegal",
)


def _trajectory_arrays(
    trajectory: Trajectory,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    points = np.asarray(
        [[waypoint.x, waypoint.y] for waypoint in trajectory.waypoints],
        dtype=np.float64,
    )
    times = np.asarray([waypoint.t for waypoint in trajectory.waypoints], dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    durations = np.diff(times)
    speeds = np.divide(
        segment_lengths,
        durations,
        out=np.full_like(segment_lengths, np.nan),
        where=durations > 0.0,
    )
    progress = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    return points, times, speeds, progress


def _terminal_low_speed_tail(
    times: FloatArray,
    speeds: FloatArray,
    progress: FloatArray,
) -> dict[str, float | None]:
    low = np.isfinite(speeds) & (speeds < LOW_SPEED_THRESHOLD_MPS)
    if not bool(low[-1]):
        return {"onset_s": None, "duration_s": 0.0, "path_progress_m": None}
    non_low = np.flatnonzero(~low)
    first = int(non_low[-1] + 1) if len(non_low) else 0
    return {
        "onset_s": float(times[first]),
        "duration_s": float(np.diff(times)[first:].sum()),
        "path_progress_m": float(progress[first]),
    }


def _governing_metric(features: TrajectoryFeatures) -> dict[str, Any]:
    metric = next(
        item
        for item in features.conflict_zone_metrics
        if item.kind == "stop_line" and item.governing
    )
    return metric.model_dump(mode="json")


def _round_floats(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, list):
        return [_round_floats(item) for item in value]
    if isinstance(value, dict):
        return {key: _round_floats(item) for key, item in value.items()}
    return value


def _variant_row(
    variant: str,
    anonymous_id: str,
    trajectory: Trajectory,
    features: TrajectoryFeatures,
    label: Any,
    path_end_xy: FloatArray,
) -> tuple[dict[str, Any], tuple[FloatArray, FloatArray, FloatArray, FloatArray]]:
    arrays = _trajectory_arrays(trajectory)
    points, times, speeds, progress = arrays
    finite_speeds = speeds[np.isfinite(speeds)]
    row = {
        "variant": variant,
        "anonymous_id": anonymous_id,
        **VARIANT_CONFIG[variant],
        "expected_verdict": label.expected_verdict,
        "expected_verdict_reason": label.expected_verdict_reason,
        "exclude_reason": label.exclude_reason,
        "n_waypoints": len(trajectory.waypoints),
        "end_time_s": float(times[-1]),
        "total_distance_m": float(progress[-1]),
        "min_speed_mps": float(np.min(finite_speeds)),
        "max_speed_mps": float(np.max(finite_speeds)),
        "mean_speed_mps": float(np.mean(finite_speeds)),
        "terminal_low_speed_tail": _terminal_low_speed_tail(times, speeds, progress),
        "full_stop": features.full_stop,
        "stop_duration_s": features.stop_duration_s,
        "governing_stop_line_metric": _governing_metric(features),
        "endpoint_xy_m": points[-1].tolist(),
        "endpoint_distance_to_path_material_end_m": float(
            np.linalg.norm(points[-1] - path_end_xy)
        ),
    }
    return _round_floats(row), arrays


def _load_scene(input_path: Path, eval_set_path: Path, index: int) -> RawScene:
    if index != 6:
        raise ValueError("本脚本只归档冻结 smoke-20 idx 6")
    eval_set = json.loads(eval_set_path.read_text(encoding="utf-8"))
    entry = eval_set["entries"][index]
    token = entry["frame_token"]
    if token != EXPECTED_FRAME_TOKEN:
        raise ValueError(
            f"冻结 idx 6 token 漂移: expected {EXPECTED_FRAME_TOKEN}, got {token}"
        )
    records = json.loads(input_path.read_text(encoding="utf-8"))
    record = next((item for item in records if item["frame_token"] == token), None)
    if record is None:
        raise ValueError(f"input 中缺少 frame_token: {token}")
    return RawScene.model_validate(record)


def _render_plot(
    rows: list[dict[str, Any]],
    curves: dict[str, tuple[FloatArray, FloatArray, FloatArray, FloatArray]],
    output_path: Path,
) -> None:
    figure, (speed_axis, progress_axis) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    colors = {
        "ground_truth": "#212121",
        "conservative": "#2e7d32",
        "suboptimal": "#8e24aa",
        "aggressive": "#d32f2f",
        "hard_case": "#1565c0",
        "illegal": "#ef6c00",
    }
    by_variant = {row["variant"]: row for row in rows}
    for variant in PLOT_ORDER:
        _, times, speeds, progress = curves[variant]
        label = f"{variant} ({by_variant[variant]['anonymous_id']})"
        speed_axis.step(
            times[1:],
            speeds,
            where="pre",
            color=colors[variant],
            linewidth=1.8,
            label=label,
        )
        progress_axis.plot(
            times,
            progress,
            color=colors[variant],
            linewidth=1.8,
            label=label,
        )

    speed_axis.axhline(
        LOW_SPEED_THRESHOLD_MPS,
        color="#616161",
        linestyle="--",
        linewidth=1.0,
        label="full-stop threshold (0.3 m/s)",
    )
    gt_tail = by_variant["ground_truth"]["terminal_low_speed_tail"]
    aggressive_tail = by_variant["aggressive"]["terminal_low_speed_tail"]
    for tail, color, name in (
        (gt_tail, colors["ground_truth"], "GT low-speed onset"),
        (aggressive_tail, colors["aggressive"], "aggressive low-speed onset"),
    ):
        if tail["onset_s"] is not None:
            speed_axis.axvline(
                tail["onset_s"], color=color, linestyle=":", linewidth=1.2, label=name
            )
    if gt_tail["path_progress_m"] is not None:
        progress_axis.axhline(
            gt_tail["path_progress_m"],
            color="#616161",
            linestyle="--",
            linewidth=1.0,
            label="GT low-speed onset position",
        )

    speed_axis.set_ylabel("Segment speed v(t) (m/s)")
    speed_axis.set_title("Frozen smoke-20 idx 6: six candidate speed profiles")
    speed_axis.grid(alpha=0.25)
    speed_axis.legend(ncol=2, fontsize=8, loc="upper right")
    progress_axis.set_xlabel("Time from keyframe (s)")
    progress_axis.set_ylabel("Path progress s(t) (m)")
    progress_axis.grid(alpha=0.25)
    progress_axis.legend(ncol=2, fontsize=8, loc="lower right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _render_markdown(report: dict[str, Any]) -> str:
    variants = {row["variant"]: row for row in report["variants"]}
    gt = variants["ground_truth"]
    conservative = variants["conservative"]
    aggressive = variants["aggressive"]
    illegal = variants["illegal"]
    checks = report["checks"]
    stop_distance = report["scene_facts"]["governing_stop_line_signed_m"]
    path_line_distance = report["path_material"]["min_distance_to_governing_stop_line_m"]
    path_end_line_distance = report["path_material"][
        "endpoint_distance_to_governing_stop_line_m"
    ]
    path_intersects_line = report["path_material"]["intersects_governing_stop_line"]
    gt_tail = gt["terminal_low_speed_tail"]
    aggressive_tail = aggressive["terminal_low_speed_tail"]
    return "\n".join([
        "# 冻结 20 场景 idx 6 速度剖面诊断",
        "",
        f"- frame_token: `{report['frame_token']}`",
        f"- governing stop-line signed distance: {stop_distance:.3f} m",
        f"- 12 s 路径素材与本向停止线多边形最近距离: {path_line_distance:.3f} m",
        f"- 12 s 路径素材终点到本向停止线多边形距离: {path_end_line_distance:.3f} m",
        f"- 本向停止线与真实路径相交: `{path_intersects_line}`",
        "",
        "![idx 6 v(t) curves](idx6_v_t.png)",
        "",
        "## 逐候选证据",
        "",
        "| variant | anon | verdict | end t (s) | distance (m) | min/max v (m/s) | "
        "terminal low-speed onset / duration | stop-line state |",
        "|---|---|---|---:|---:|---:|---:|---|",
        *[
            "| {variant} | {anonymous_id} | {expected_verdict} | {end_time_s:.3f} | "
            "{total_distance_m:.3f} | {min_speed_mps:.3f}/{max_speed_mps:.3f} | "
            "{onset} / {duration:.3f}s | entered={entered}, stopped_before={stopped} |".format(
                **row,
                onset=(
                    f"{row['terminal_low_speed_tail']['onset_s']:.3f}s"
                    if row["terminal_low_speed_tail"]["onset_s"] is not None
                    else "none"
                ),
                duration=row["terminal_low_speed_tail"]["duration_s"],
                entered=row["governing_stop_line_metric"]["entered"],
                stopped=row["governing_stop_line_metric"]["stopped_before_zone"],
            )
            for row in report["variants"]
        ],
        "",
        "## 定案",
        "",
        f"1. **GT 确有线前减速到近停段。** 末端低速起点 t={gt_tail['onset_s']:.3f}s、"
        f"s={gt_tail['path_progress_m']:.3f}m，持续 {gt_tail['duration_s']:.3f}s；"
        "但 6 s 窗口内不足 1.0 s，所以 GT 自身仍为 borderline。",
        f"2. **aggressive-cleared 成因成立。** 1.3x 候选在 t="
        f"{aggressive_tail['onset_s']:.3f}s 到达同一停车位置（与 GT 低速起点相差 "
        f"{checks['aggressive_gt_stop_position_distance_m']:.3f}m），比 GT 早 "
        f"{checks['aggressive_stop_onset_lead_s']:.3f}s，并在窗口内累计低速 "
        f"{aggressive['stop_duration_s']:.3f}s，故共享谓词给出 cleared。实现对 "
        "aggressive 使用 `speed_floor_ratio=0`；零基速不会被抬高。"
        f"本条数据没有严格 0 m/s 段（GT 实测最小 {gt['min_speed_mps']:.3f}m/s），"
        "因此这里应表述为“近零停车段被保留”，不应伪称观测到严格的 `1.3×0=0`。",
        f"3. **conservative-borderline 成因成立。** 0.6x 在窗口末仅走到 s="
        f"{conservative['total_distance_m']:.3f}m，距离 GT 低速起点仍差 "
        f"{checks['conservative_short_of_gt_stop_progress_m']:.3f}m；"
        "它既未越线也未形成完整停车，因此是 borderline。",
        "4. **illegal 没有形成红灯违规，但主因不是 a_lat。** "
        f"3.0x + 速度底板候选在 t={illegal['end_time_s']:.3f}s "
        "已经耗尽真实 12 s 路径素材，终点仍距本向停止线多边形 "
        f"{path_end_line_distance:.3f}m；路径本身不与停止线相交。"
        "因此任意速度缩放都无法越线，a_lat 上限不是这一条的决定性瓶颈。",
        "",
        f"**最终结论：{report['conclusion']}**",
        "",
        "完整机器可读数值见 `idx6_speed_diagnostic.json`。",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--eval-set", type=Path, required=True)
    parser.add_argument("--index", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    raw_scene = _load_scene(args.input, args.eval_set, args.index)
    bundle = parse_scene(raw_scene)
    facts = SceneFactExtractor().extract(raw_scene)
    extractor = TrajectoryExtractor()
    path = extractor.extract_path(raw_scene)
    path_points = np.asarray([[point.x, point.y] for point in path.waypoints], dtype=np.float64)
    path_end_xy = path_points[-1]
    augmentor = TrajectoryAugmentor()
    stop_target_s = augmentor._governing_stop_path_s(path_points, facts)
    governing_zones = [
        zone
        for zone in facts.conflict_zones
        if zone.kind == "stop_line"
        and facts.governing_stop_line_signed_m is not None
        and abs(
            sum(point[0] for point in zone.polygon_xy) / len(zone.polygon_xy)
            - facts.governing_stop_line_signed_m
        ) < 2.0
    ]
    min_path_distance = min(
        TrajectoryAnalyzer._distance_to_polygon(
            (float(point[0]), float(point[1])), zone.polygon_xy
        )
        for point in path_points
        for zone in governing_zones
    )
    path_end_distance = min(
        TrajectoryAnalyzer._distance_to_polygon(
            (float(path_end_xy[0]), float(path_end_xy[1])), zone.polygon_xy
        )
        for zone in governing_zones
    )

    trajectories = {item.traj_id: item for item in bundle.context.candidate_trajectories}
    features = dict(
        zip(
            (item.traj_id for item in bundle.context.candidate_trajectories),
            bundle.context.trajectory_features,
            strict=True,
        )
    )
    rows: list[dict[str, Any]] = []
    curves: dict[str, tuple[FloatArray, FloatArray, FloatArray, FloatArray]] = {}
    for label in bundle.benchmark_labels.labels:
        variant = label.variant_type.value
        row, arrays = _variant_row(
            variant,
            label.anonymous_id,
            trajectories[label.anonymous_id],
            features[label.anonymous_id],
            label,
            path_end_xy,
        )
        rows.append(row)
        curves[variant] = arrays
    rows.sort(key=lambda row: PLOT_ORDER.index(row["variant"]))

    by_variant = {row["variant"]: row for row in rows}
    gt = by_variant["ground_truth"]
    aggressive = by_variant["aggressive"]
    conservative = by_variant["conservative"]
    gt_points, _, _, _ = curves["ground_truth"]
    aggressive_points, _, _, _ = curves["aggressive"]
    gt_low_index = int(
        np.argmin(
            np.abs(
                np.asarray([point.t for point in trajectories[gt["anonymous_id"]].waypoints])
                - gt["terminal_low_speed_tail"]["onset_s"]
            )
        )
    )
    aggressive_low_index = int(
        np.argmin(
            np.abs(
                np.asarray([
                    point.t
                    for point in trajectories[aggressive["anonymous_id"]].waypoints
                ])
                - aggressive["terminal_low_speed_tail"]["onset_s"]
            )
        )
    )
    position_distance = float(
        np.linalg.norm(gt_points[gt_low_index] - aggressive_points[aggressive_low_index])
    )
    checks = {
        "gt_has_terminal_deceleration_to_low_speed": (
            gt["terminal_low_speed_tail"]["duration_s"] >= 0.5
        ),
        "aggressive_reaches_gt_stop_position": position_distance <= 0.25,
        "aggressive_reaches_stop_position_earlier": (
            aggressive["terminal_low_speed_tail"]["onset_s"]
            < gt["terminal_low_speed_tail"]["onset_s"]
        ),
        "aggressive_gt_stop_position_distance_m": position_distance,
        "aggressive_stop_onset_lead_s": (
            gt["terminal_low_speed_tail"]["onset_s"]
            - aggressive["terminal_low_speed_tail"]["onset_s"]
        ),
        "aggressive_has_decisive_full_stop": (
            aggressive["full_stop"]
            and aggressive["governing_stop_line_metric"]["stopped_before_zone"] is True
        ),
        "conservative_short_of_gt_stop_progress_m": (
            gt["terminal_low_speed_tail"]["path_progress_m"]
            - conservative["total_distance_m"]
        ),
        "conservative_does_not_reach_gt_stop_progress": (
            conservative["total_distance_m"]
            < gt["terminal_low_speed_tail"]["path_progress_m"]
        ),
        "illegal_reaches_path_material_end_within_horizon": (
            by_variant["illegal"]["endpoint_distance_to_path_material_end_m"] <= 0.01
            and by_variant["illegal"]["end_time_s"] < 6.0
        ),
        "path_speed_only_injection_can_cross_governing_line": stop_target_s is not None,
        "a_lat_is_primary_illegal_injection_blocker": False,
        "illegal_blocker_is_path_material_end": (
            by_variant["illegal"]["endpoint_distance_to_path_material_end_m"] <= 0.01
            and stop_target_s is None
            and path_end_distance > 0.15
        ),
        "labels_match_geometry": (
            aggressive["expected_verdict"] == "cleared"
            and conservative["expected_verdict"] == "exclude"
            and conservative["expected_verdict_reason"]
            == "borderline_or_phase_horizon_unknown"
            and all(row["expected_verdict"] != "vetoed" for row in rows)
        ),
    }
    report = _round_floats({
        "record_index": args.index,
        "frame_token": raw_scene.frame_token,
        "scene_id": raw_scene.scene_id,
        "scenario_type": bundle.context.scenario_type.value,
        "configuration": {
            "a_lat_max_mps2": augmentor.a_lat_max_mps2,
            "output_horizon_s": augmentor.output_horizon_s,
            "low_speed_threshold_mps": LOW_SPEED_THRESHOLD_MPS,
        },
        "scene_facts": {
            "governing_stop_line_signed_m": facts.governing_stop_line_signed_m,
            "ego_turn_intent": facts.ego_turn_intent,
        },
        "path_material": {
            "horizon_s": path.waypoints[-1].t,
            "total_distance_m": float(
                np.linalg.norm(np.diff(path_points, axis=0), axis=1).sum()
            ),
            "endpoint_xy_m": path_end_xy.tolist(),
            "min_distance_to_governing_stop_line_m": min_path_distance,
            "endpoint_distance_to_governing_stop_line_m": path_end_distance,
            "conservative_stop_target_path_s": stop_target_s,
            "intersects_governing_stop_line": stop_target_s is not None,
        },
        "variants": rows,
        "checks": checks,
        "conclusion": (
            "aggressive-cleared 主假设成立，标签正确，无打标 bug；"
            "但 illegal 不可注入的决定性原因是路径素材在线前终止，而非 a_lat 上限"
        ),
    })

    required_checks = (
        "gt_has_terminal_deceleration_to_low_speed",
        "aggressive_reaches_gt_stop_position",
        "aggressive_reaches_stop_position_earlier",
        "aggressive_has_decisive_full_stop",
        "conservative_does_not_reach_gt_stop_progress",
        "illegal_reaches_path_material_end_within_horizon",
        "illegal_blocker_is_path_material_end",
        "labels_match_geometry",
    )
    failed = [name for name in required_checks if not report["checks"][name]]
    if failed:
        raise RuntimeError(f"idx 6 诊断断言失败: {failed}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _render_plot(rows, curves, args.output_dir / "idx6_v_t.png")
    (args.output_dir / "idx6_speed_diagnostic.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "idx6_speed_diagnostic.md").write_text(
        _render_markdown(report),
        encoding="utf-8",
    )
    print(json.dumps(report["checks"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
