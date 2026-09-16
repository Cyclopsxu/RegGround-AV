"""在调用 LLM 前复核冻结 Gate 3 场景与 pedestrian 时机可达域。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from src.layer1.agent_interaction import AgentInteractionAnalyzer
from src.layer1.facade import (
    _minimum_pedestrian_gap,
    _pedestrian_illegal_candidate_at_scale,
    _pedestrian_illegal_target_met,
    _pedestrian_timing_search_scales,
    parse_scene,
)
from src.layer1.models import RawScene, ScenarioType, TrajectoryVariantType
from src.layer1.scene_fact_extractor import SceneFactExtractor
from src.layer1.trajectory_analyzer import TrajectoryAnalyzer
from src.layer1.trajectory_augmentor import TrajectoryAugmentor
from src.layer1.trajectory_extractor import TrajectoryExtractor

V2_SCOPE = {"red_light", "pedestrian", "oncoming"}
FULL_V2_FROZEN_ENTRY_COUNT = 259
STRUCTURAL_REASONS = (
    "start_beyond_line",
    "trigger_missing",
    "not_evaluable_short_window",
    "interaction_missing",
    "oncoming_design_exclusion",
)
_STRUCTURAL_REASON_ALIASES = {
    "pedestrian_trigger_missing_or_unknown": "trigger_missing",
    "pedestrian_interaction_missing": "interaction_missing",
}


def _arrival_domain(candidates: list[tuple[float, Any]]) -> dict[str, Any]:
    arrivals = [
        interaction.ego_zone_arrival_s
        for _, candidate in candidates
        for interaction in candidate.features.agent_interactions
        if interaction.agent_kind == "pedestrian"
        and interaction.ego_zone_arrival_s is not None
    ]
    return {
        "arrival_min_s": min(arrivals) if arrivals else None,
        "arrival_max_s": max(arrivals) if arrivals else None,
        "tested_scales": [scale for scale, _ in candidates],
        "no_shared_zone_scale_count": sum(
            not any(
                interaction.agent_kind == "pedestrian"
                and interaction.ego_zone_arrival_s is not None
                for interaction in candidate.features.agent_interactions
            )
            for _, candidate in candidates
        ),
    }


def _pedestrian_diagnostic(raw_scene: RawScene, bundle: Any) -> dict[str, Any]:
    facts = SceneFactExtractor().extract(raw_scene)
    extractor = TrajectoryExtractor()
    ground_truth = extractor.extract(raw_scene)
    path = extractor.extract_path(raw_scene)
    augmentor = TrajectoryAugmentor()
    analyzer = TrajectoryAnalyzer()
    interaction_analyzer = AgentInteractionAnalyzer()

    def candidate(scale: float) -> Any:
        return _pedestrian_illegal_candidate_at_scale(
            speed_scale=scale,
            augmentor=augmentor,
            analyzer=analyzer,
            interaction_analyzer=interaction_analyzer,
            raw_scene=raw_scene,
            facts=facts,
            scenario_type=bundle.context.scenario_type,
            description=bundle.context.description,
            ground_truth=ground_truth,
            path_trajectory=path,
            # 路径本身已由完整 parse_scene 的 drivable gate 验证；此处只扫时机域。
            drivable_area_polygons=[],
        )

    probe = candidate(1.0)
    slowdown_scales, speedup_scales = _pedestrian_timing_search_scales(probe)
    slowdown = [(scale, candidate(scale)) for scale in slowdown_scales]
    speedup = [(scale, candidate(scale)) for scale in speedup_scales]
    target = augmentor.pedestrian_illegal_gap_target_s

    achieved = [
        (direction, scale, _minimum_pedestrian_gap(item))
        for direction, rows in (("slowdown", slowdown), ("speedup", speedup))
        for scale, item in rows
        if _pedestrian_illegal_target_met(item, target)
    ]
    illegal_label = next(
        label
        for label in bundle.benchmark_labels.labels
        if label.variant_type == TrajectoryVariantType.ILLEGAL
    )
    if illegal_label.expected_verdict == "vetoed":
        classification = "direction_issue_rescued"
    elif achieved:
        classification = "gap_target_reached_but_predicate_not_decidable"
    else:
        classification = "true_no_violation_injectable"

    agent_windows = sorted({
        interaction.agent_zone_window_s
        for interaction in probe.features.agent_interactions
        if interaction.agent_kind == "pedestrian"
        and interaction.agent_zone_window_s is not None
    })
    return {
        "agent_zone_windows_s": agent_windows,
        "slowdown_domain": _arrival_domain(slowdown),
        "speedup_domain": _arrival_domain(speedup),
        "first_target_hit": (
            {"direction": achieved[0][0], "scale": achieved[0][1], "gap_s": achieved[0][2]}
            if achieved
            else None
        ),
        "final_illegal_expected": illegal_label.expected_verdict,
        "final_illegal_reason": illegal_label.expected_verdict_reason,
        "classification": classification,
    }


def _normalize_structural_reason(reason: str | None) -> str | None:
    if reason is None:
        return None
    return _STRUCTURAL_REASON_ALIASES.get(reason, reason)


def _structural_unscorable_reason(bundle: Any) -> str | None:
    """复现冻结结构池口径，不把注入失败或 borderline 混入结构排除。"""
    context = bundle.context
    facts = context.scene_facts
    if context.scenario_type.value == "oncoming":
        return "oncoming_design_exclusion"
    if context.scenario_type.value == "red_light":
        signed = facts.governing_stop_line_signed_m
        if signed is not None and signed < 0:
            return "start_beyond_line"
    if context.scenario_type.value == "pedestrian" and (
        facts.pedestrian_in_forward_crosswalk is not True
        or facts.pedestrian_moving is not True
    ):
        return "trigger_missing"
    if facts.window_insufficient:
        return "not_evaluable_short_window"
    if context.scenario_type.value == "pedestrian":
        has_interaction = any(
            interaction.agent_kind == "pedestrian"
            and interaction.min_time_gap_s is not None
            for feature in context.trajectory_features
            for interaction in feature.agent_interactions
        )
        if not has_interaction:
            return "interaction_missing"
    return None


def _preparse_structural_unscorable_reason(
    raw_scene: RawScene,
    scenario_type: str,
) -> str | None:
    """在候选增强前裁决不依赖轨迹交互的冻结结构原因。"""
    facts = SceneFactExtractor().extract(raw_scene)
    if scenario_type == "oncoming":
        return "oncoming_design_exclusion"
    if scenario_type == "red_light":
        signed = facts.governing_stop_line_signed_m
        if signed is not None and signed < 0:
            return "start_beyond_line"
    if scenario_type == "pedestrian" and (
        facts.pedestrian_in_forward_crosswalk is not True
        or facts.pedestrian_moving is not True
    ):
        return "trigger_missing"
    if facts.window_insufficient:
        return "not_evaluable_short_window"
    if scenario_type == "pedestrian":
        extractor = TrajectoryExtractor()
        candidates = TrajectoryAugmentor().augment(
            extractor.extract(raw_scene),
            ScenarioType.PEDESTRIAN,
            facts,
            scene_id=raw_scene.scene_id,
            frame_token=raw_scene.frame_token,
            path_trajectory=extractor.extract_path(raw_scene),
            # 结构池只问是否存在共同交互区；drivable gate 在正式候选解析中另行执行。
            drivable_area_polygons=[],
        )
        interaction_analyzer = AgentInteractionAnalyzer()
        has_interaction = any(
            interaction.agent_kind == "pedestrian"
            and interaction.min_time_gap_s is not None
            for candidate in candidates
            for interaction in interaction_analyzer.analyze(
                candidate.trajectory,
                raw_scene,
                facts,
            )
        )
        if not has_interaction:
            return "interaction_missing"
    return None


def _stop_line_path_material_status(
    raw_scene: RawScene,
    scenario_type: str,
) -> str | None:
    """判断前方 governing stop line 是否被 12 秒真实路径素材触及。"""
    if scenario_type != "red_light":
        return None
    facts = SceneFactExtractor().extract(raw_scene)
    signed = facts.governing_stop_line_signed_m
    if signed is None or signed < 0.0:
        return None
    path = TrajectoryExtractor().extract_path(raw_scene)
    path_points = np.asarray(
        [(waypoint.x, waypoint.y) for waypoint in path.waypoints],
        dtype=np.float64,
    )
    stop_s = TrajectoryAugmentor._governing_stop_path_s(path_points, facts)
    return "reached" if stop_s is not None else "not_reached"


def _vector_distribution(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    vectors = Counter(
        (
            row["label_counts"]["cleared"],
            row["label_counts"]["vetoed"],
            row["label_counts"]["exclude"],
        )
        for row in rows
        if row["label_counts"] is not None
    )
    total = len(rows)
    return [
        {
            "cleared": vector[0],
            "vetoed": vector[1],
            "exclude": vector[2],
            "scene_count": count,
            "scene_rate": count / total if total else None,
        }
        for vector, count in sorted(vectors.items())
    ]


def _path_material_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["stop_line_path_material_status"] for row in rows)
    return {
        "eligible_scenes": len(rows),
        "reached_scenes": counts["reached"],
        "not_reached_scenes": counts["not_reached"],
        "not_reached_frame_tokens": sorted(
            row["frame_token"]
            for row in rows
            if row["stop_line_path_material_status"] == "not_reached"
        ),
    }


def _exclude_reason_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    reasons = Counter(
        label["exclude_reason"] or "missing_exclude_reason"
        for row in rows
        for label in row["labels"]
        if label["expected"] == "exclude"
    )
    return dict(sorted(reasons.items()))


def _expected_reason_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    reasons = Counter(
        label["reason"]
        for row in rows
        for label in row["labels"]
        if label["expected"] == "exclude"
    )
    return dict(sorted(reasons.items()))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(str(value) for value in row) + " |" for row in rows),
    ]


def _render_markdown(report: dict[str, Any]) -> str:
    selection = report["selection"]
    structural = report["structural_pool"]
    preflight = report["preflight"]
    coverage_baseline = report["coverage_baseline"]
    path_material = report["stop_line_path_material"]
    full_path_material = path_material["full_v2_red"]
    structural_path_material = path_material["structural_pool_red"]
    lines = [
        "# Gate 3 v2 全量确定性 preflight",
        "",
        "本报告只运行 Layer 1 确定性解析、候选生成和共享谓词打标；未调用 LLM。",
        "",
        "## 输入与范围",
        "",
        f"- 冻结 eval set：`{report['inputs']['eval_set']}`",
        f"- RawScene 输入：`{report['inputs']['prepared_input']}`",
        f"- 冻结条目：{selection['frozen_entry_count']}；v2 MVP 入选："
        f"{selection['selected_entry_count']}；范围外：{selection['scope_excluded_count']}",
        "- v2 MVP 范围：`red_light + pedestrian + oncoming`；明确裁掉 "
        "`yellow_light + emergency`。",
        "",
        "## 执行完整性",
        "",
        f"- preflight complete：**{str(preflight['complete']).lower()}**",
        f"- N1 完整候选打标：{structural['labeled_scenes']}/"
        f"{structural['n1_structurally_scorable_scenes']}",
        f"- 失败数：{preflight['failure_count']}",
        "",
        *(
            _markdown_table(
                ["index", "frame_token", "scenario_type", "异常", "消息"],
                [
                    [
                        failure["index"],
                        f"`{failure['frame_token']}`",
                        failure["scenario_type"],
                        failure["type"],
                        failure["message"],
                    ]
                    for failure in preflight["failures"]
                ],
            )
            if preflight["failures"]
            else ["无失败。"]
        ),
        "",
        "### 场景分布",
        "",
        *_markdown_table(
            ["scenario_type", "冻结条目", "v2 入选", "范围外"],
            [
                [
                    scenario,
                    selection["frozen_scenario_distribution"].get(scenario, 0),
                    selection["selected_scenario_distribution"].get(scenario, 0),
                    selection["scope_excluded_distribution"].get(scenario, 0),
                ]
                for scenario in sorted(selection["frozen_scenario_distribution"])
            ],
        ),
        "",
        "## 双池覆盖数字",
        "",
        f"- 结构可评分池 N1：**{structural['n1_structurally_scorable_scenes']}**",
        f"- 其中产 pair 的 N2：**{structural['n2_pair_scenes']}**",
        f"- N2/N1：**{structural['pair_scene_rate']:.2%}**",
        f"- 结构不可评分池：**{structural['structurally_unscorable_scenes']}**",
        f"- N1 内候选打标失败：**{report['preflight']['failure_count']}**",
        "",
        "### 全量覆盖基线记录",
        "",
        f"- 指标：`{coverage_baseline['metric']}`",
        f"- 观测值：{coverage_baseline['observed_rate']:.2%}",
        f"- 口径：{coverage_baseline['interpretation']}",
        "- 20 场景 smoke 不再承担结构覆盖门禁，只裁行为门禁。",
        "",
        "### 停止线与 12 秒路径素材",
        "",
        f"- 全量 v2 red 前方 governing stop line：{full_path_material['eligible_scenes']}；"
        f"路径素材未触及：**{full_path_material['not_reached_scenes']}**",
        f"- 结构池内 red 前方 governing stop line："
        f"{structural_path_material['eligible_scenes']}；路径素材未触及："
        f"**{structural_path_material['not_reached_scenes']}**",
        f"- 操作定义：{path_material['operational_definition']}",
        "",
        "### 结构不可评分原因",
        "",
        *_markdown_table(
            ["原因", "场景数"],
            [
                [reason, structural["unscorable_reason_distribution"].get(reason, 0)]
                for reason in STRUCTURAL_REASONS
            ],
        ),
        "",
        "注：注入失败、`gap_borderline` 和其他谓词弃权仍留在结构可评分池内，"
        "不会被挪入结构不可评分池。",
        "候选生成或打标异常同样保留在 N1 分母，并单列为 preflight failure。",
        "",
        "## N1 内 cleared/vetoed/exclude 计数向量",
        "",
        *_markdown_table(
            ["cleared", "vetoed", "exclude", "场景数", "N1 占比"],
            [
                [
                    row["cleared"],
                    row["vetoed"],
                    row["exclude"],
                    row["scene_count"],
                    f"{row['scene_rate']:.2%}",
                ]
                for row in report["label_count_vector_distribution_within_structural"]
            ],
        ),
        "",
        f"## exclude_reason 分布（N1 已完成打标的 {structural['labeled_scenes']} 场）",
        "",
        *_markdown_table(
            ["exclude_reason", "候选数"],
            [
                [reason, count]
                for reason, count in report[
                    "exclude_reason_distribution_within_structural"
                ].items()
            ],
        ),
        "",
        "### exclude 的谓词/生成原因（同一批候选）",
        "",
        *_markdown_table(
            ["expected_verdict_reason", "候选数"],
            [
                [reason, count]
                for reason, count in report[
                    "expected_reason_distribution_within_structural"
                ].items()
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--eval-set", type=Path, required=True)
    parser.add_argument(
        "--structural-pool",
        type=Path,
        help="可选：已有冻结结构池；提供时逐 token 校验当前结构判定",
    )
    parser.add_argument(
        "--scope",
        choices=("all", "v2"),
        default="all",
        help="v2 仅保留 red_light/pedestrian/oncoming",
    )
    parser.add_argument(
        "--skip-pedestrian-diagnostics",
        action="store_true",
        help="全量覆盖聚合时跳过逐尺度行人诊断；不影响候选打标",
    )
    parser.add_argument(
        "--label-structural-pool-only",
        action="store_true",
        help="先判结构排除，只对结构可评分池生成和打标候选",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()

    records = json.loads(args.input.read_text(encoding="utf-8"))
    by_token = {record["frame_token"]: record for record in records}
    eval_set = json.loads(args.eval_set.read_text(encoding="utf-8"))
    frozen_entries = eval_set["entries"]
    selected_entries = [
        entry
        for entry in frozen_entries
        if args.scope == "all" or entry["scenario_type"] in V2_SCOPE
    ]
    declared_structural_reason: dict[str, str | None] | None = None
    if args.structural_pool is not None:
        pool = json.loads(args.structural_pool.read_text(encoding="utf-8"))
        declared_structural_reason = {
            entry["frame_token"]: _normalize_structural_reason(entry["unscorable_reason"])
            for entry in pool["entries"]
        }

    rows: list[dict[str, Any]] = []
    pedestrian: list[dict[str, Any]] = []
    for index, entry in enumerate(selected_entries):
        token = entry["frame_token"]
        if token not in by_token:
            raise ValueError(f"frozen frame_token is missing from input: {token}")
        raw_scene = RawScene.model_validate(by_token[token])
        path_material_status = _stop_line_path_material_status(
            raw_scene,
            entry["scenario_type"],
        )
        preparse_reason = _preparse_structural_unscorable_reason(
            raw_scene,
            entry["scenario_type"],
        )
        if args.label_structural_pool_only and preparse_reason is not None:
            if declared_structural_reason is not None:
                if token not in declared_structural_reason:
                    raise ValueError(f"frame_token is missing from structural pool: {token}")
                if declared_structural_reason[token] != preparse_reason:
                    raise ValueError(
                        f"structural pool mismatch for {token}: declared="
                        f"{declared_structural_reason[token]!r}, derived={preparse_reason!r}"
                    )
            rows.append({
                "index": index,
                "frame_token": token,
                "scenario_type": entry["scenario_type"],
                "structural_unscorable_reason": preparse_reason,
                "has_scorable_pair": False,
                "label_counts": None,
                "labels": [],
                "candidate_labeling": "not_applicable_structurally_unscorable",
                "stop_line_path_material_status": path_material_status,
            })
            continue
        try:
            bundle = parse_scene(raw_scene)
        except Exception as exc:  # noqa: BLE001 - preflight 必须枚举全部失败 token
            rows.append({
                "index": index,
                "frame_token": token,
                "scenario_type": entry["scenario_type"],
                "structural_unscorable_reason": preparse_reason,
                "has_scorable_pair": False,
                "label_counts": None,
                "labels": [],
                "candidate_labeling": "failed",
                "stop_line_path_material_status": path_material_status,
                "preflight_error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            })
            continue
        derived_structural_reason = _structural_unscorable_reason(bundle)
        if preparse_reason != derived_structural_reason:
            raise ValueError(
                f"preparse structural mismatch for {token}: preparse="
                f"{preparse_reason!r}, parsed={derived_structural_reason!r}"
            )
        if declared_structural_reason is not None:
            if token not in declared_structural_reason:
                raise ValueError(f"frame_token is missing from structural pool: {token}")
            if declared_structural_reason[token] != derived_structural_reason:
                raise ValueError(
                    f"structural pool mismatch for {token}: declared="
                    f"{declared_structural_reason[token]!r}, derived="
                    f"{derived_structural_reason!r}"
                )
        labels = bundle.benchmark_labels.labels
        cleared = sum(label.expected_verdict == "cleared" for label in labels)
        vetoed = sum(label.expected_verdict == "vetoed" for label in labels)
        row = {
            "index": index,
            "frame_token": token,
            "scenario_type": entry["scenario_type"],
            "structural_unscorable_reason": derived_structural_reason,
            "has_scorable_pair": bool(cleared and vetoed),
            "label_counts": {
                "cleared": cleared,
                "vetoed": vetoed,
                "exclude": len(labels) - cleared - vetoed,
            },
            "labels": [
                {
                    "variant": label.variant_type.value,
                    "expected": label.expected_verdict,
                    "reason": label.expected_verdict_reason,
                    "exclude_reason": label.exclude_reason,
                }
                for label in labels
            ],
            "candidate_labeling": "complete",
            "stop_line_path_material_status": path_material_status,
        }
        rows.append(row)
        if (
            entry["scenario_type"] == "pedestrian"
            and not args.skip_pedestrian_diagnostics
        ):
            pedestrian.append({
                "index": index,
                "frame_token": token,
                **_pedestrian_diagnostic(raw_scene, bundle),
            })

    structural_rows = [
        row for row in rows if row["structural_unscorable_reason"] is None
    ]
    pair_count = sum(row["has_scorable_pair"] for row in structural_rows)
    frozen_scenarios = Counter(entry["scenario_type"] for entry in frozen_entries)
    selected_scenarios = Counter(entry["scenario_type"] for entry in selected_entries)
    excluded_scenarios = frozen_scenarios - selected_scenarios
    structural_reasons = Counter(
        row["structural_unscorable_reason"]
        for row in rows
        if row["structural_unscorable_reason"] is not None
    )
    pair_rate = pair_count / len(structural_rows) if structural_rows else None
    labeled_rows = [row for row in rows if row["candidate_labeling"] == "complete"]
    failed_rows = [row for row in rows if row["candidate_labeling"] == "failed"]
    full_v2_baseline_applicable = (
        args.scope == "v2"
        and len(frozen_entries) == FULL_V2_FROZEN_ENTRY_COUNT
    )
    full_path_material_rows = [
        row for row in rows if row.get("stop_line_path_material_status") is not None
    ]
    structural_path_material_rows = [
        row
        for row in structural_rows
        if row.get("stop_line_path_material_status") is not None
    ]
    report = {
        "preflight_version": "gate3-deterministic-full-v2",
        "llm_called": False,
        "inputs": {
            "prepared_input": str(args.input),
            "prepared_input_sha256": _sha256(args.input),
            "eval_set": str(args.eval_set),
            "eval_set_sha256": _sha256(args.eval_set),
            "eval_set_version": eval_set.get("version"),
            "eval_set_state": eval_set.get("state"),
        },
        "selection": {
            "scope": args.scope,
            "scope_rule": (
                "scenario_type in {red_light,pedestrian,oncoming}"
                if args.scope == "v2"
                else "all frozen entries"
            ),
            "frozen_entry_count": len(frozen_entries),
            "selected_entry_count": len(selected_entries),
            "scope_excluded_count": len(frozen_entries) - len(selected_entries),
            "frozen_scenario_distribution": dict(sorted(frozen_scenarios.items())),
            "selected_scenario_distribution": dict(sorted(selected_scenarios.items())),
            "scope_excluded_distribution": dict(sorted(excluded_scenarios.items())),
        },
        "structural_policy": {
            "version": "2026-07-18-short-window-v2",
            "unscorable_reasons": list(STRUCTURAL_REASONS),
            "short_window_structurally_unscorable": True,
            "injection_failures_and_borderline_remain_in_structural_pool": True,
        },
        "preflight": {
            "complete": not failed_rows,
            "failure_count": len(failed_rows),
            "failures": [
                {
                    "index": row["index"],
                    "frame_token": row["frame_token"],
                    "scenario_type": row["scenario_type"],
                    **row["preflight_error"],
                }
                for row in failed_rows
            ],
        },
        "coverage_baseline": {
            "metric": "n2_pair_scenes / n1_structurally_scorable_scenes",
            "scope": "full frozen v2 range after yellow/emergency cut",
            "observed_rate": pair_rate,
            "applicable": full_v2_baseline_applicable,
            "requires_complete_preflight": True,
            "interpretation": "基线记录，不设阈值，不作通过或未通过判定",
            "smoke_20_is_behavior_gate_only": True,
        },
        "stop_line_path_material": {
            "operational_definition": (
                "12 秒真实路径素材不与 governing stop-line polygon 相交"
            ),
            "full_v2_red": _path_material_summary(full_path_material_rows),
            "structural_pool_red": _path_material_summary(
                structural_path_material_rows
            ),
        },
        "structural_pool": {
            "n1_structurally_scorable_scenes": len(structural_rows),
            "n2_pair_scenes": pair_count,
            "pair_scene_rate": pair_rate,
            "labeled_scenes": sum(
                row["candidate_labeling"] == "complete"
                for row in structural_rows
            ),
            "structurally_unscorable_scenes": len(rows) - len(structural_rows),
            "unscorable_reason_distribution": {
                reason: structural_reasons.get(reason, 0)
                for reason in STRUCTURAL_REASONS
            },
        },
        "label_count_vector_distribution_within_structural": _vector_distribution(
            structural_rows
        ),
        "label_count_vector_distribution_all_labeled": _vector_distribution(labeled_rows),
        "exclude_reason_distribution_within_structural": _exclude_reason_distribution(
            structural_rows
        ),
        "exclude_reason_distribution_all_labeled": _exclude_reason_distribution(labeled_rows),
        "expected_reason_distribution_within_structural": _expected_reason_distribution(
            structural_rows
        ),
        "structurally_scorable_scenes": len(structural_rows),
        "scorable_pair_scenes_within_structural": pair_count,
        "scorable_pair_scene_rate_within_structural": pair_rate,
        "records": rows,
        "pedestrian_timing_diagnostics": pedestrian,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(_render_markdown(report), encoding="utf-8")
    print(json.dumps({
        "frozen_entry_count": report["selection"]["frozen_entry_count"],
        "selected_entry_count": report["selection"]["selected_entry_count"],
        "structural_pool": report["structural_pool"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
