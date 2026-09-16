"""依据最终几何特征构建 benchmark 标签。"""

from __future__ import annotations

from typing import Literal

from src.compliance_predicates import evaluate_active_rules
from src.layer1.models import (
    AugmentedTrajectory,
    BenchmarkLabels,
    CandidateBenchmarkLabel,
    ExcludeReason,
    ScenarioType,
    SceneDescription,
    SceneFacts,
    SceneFactsDigest,
    TrajectoryFeatures,
    TrajectoryVariantType,
)
from src.layer1.trajectory_analyzer import TrajectoryAnalyzer
from src.trajectory_feature_text import feature_text_hash

BENCHMARK_DATA_VERSION = "2026-07-v2"


def build_benchmark_candidates(
    candidates: list[AugmentedTrajectory],
    analyzer: TrajectoryAnalyzer,
    description: SceneDescription,
    scene_facts: SceneFacts,
    scenario_type: ScenarioType,
) -> list[AugmentedTrajectory]:
    """分析最终轨迹，并以几何结果覆盖扰动意图标签。"""
    analyzed = [
        candidate.model_copy(update={
            "features": analyzer.analyze(candidate.trajectory, description, scene_facts)
        })
        for candidate in candidates
    ]
    return label_benchmark_candidates(analyzed, scene_facts, scenario_type)


def label_benchmark_candidates(
    candidates: list[AugmentedTrajectory],
    scene_facts: SceneFacts,
    scenario_type: ScenarioType,
    active_rule_ids: set[str] | None = None,
    phase_validity_horizon_s: float = 2.0,
) -> list[AugmentedTrajectory]:
    """使用 judge 同源要件与 override 后规则给已有特征打标。"""
    built: list[AugmentedTrajectory] = []
    seen_hashes: set[str] = set()
    for candidate in candidates:
        if candidate.features is None:
            raise ValueError("candidate features must be analyzed before benchmark labeling")
        feature = candidate.features
        visible_hash = feature_text_hash(feature)
        is_degenerate = visible_hash in seen_hashes
        seen_hashes.add(visible_hash)
        rules = active_rule_ids or _default_active_rule_ids(scenario_type)
        outcome = evaluate_active_rules(
            feature,
            _digest(scene_facts),
            rules,
            phase_validity_horizon_s=phase_validity_horizon_s,
        )

        verdict = outcome.verdict
        reason = outcome.reason
        exclude_reason: ExcludeReason | None = (
            "predicate_not_decidable" if verdict == "exclude" else None
        )
        if scene_facts.window_insufficient:
            verdict = "exclude"
            reason = "not_evaluable_short_window"
            exclude_reason = "not_evaluable_short_window"
        elif candidate.generation_note.startswith("no_violation_injectable") or (
            is_degenerate and candidate.variant_type == TrajectoryVariantType.ILLEGAL
        ):
            verdict = "exclude"
            reason = "no_violation_injectable"
            exclude_reason = "no_violation_injectable"
        elif is_degenerate:
            verdict = "exclude"
            reason = "injection_degenerate"
            exclude_reason = "injection_degenerate"
        elif candidate.variant_type == TrajectoryVariantType.GROUND_TRUTH and verdict == "vetoed":
            # GT 与触发规则冲突时不把它当作可靠违规负例，避免传感代理噪声污染标签。
            verdict = "exclude"
            reason = f"gt_precheck_not_evaluable: {reason}"
            exclude_reason = "gt_precheck_not_evaluable"

        gt_status = "not_gt"
        if candidate.variant_type == TrajectoryVariantType.GROUND_TRUTH:
            if scene_facts.window_insufficient or not outcome.is_decisive:
                gt_status = "not_evaluable"
            else:
                gt_status = "passed" if outcome.verdict == "cleared" else "failed"

        built.append(candidate.model_copy(update={
            "features": feature,
            "expected_verdict": verdict,
            "expected_verdict_reason": reason,
            "exclude_reason": exclude_reason,
            "predicate_version": outcome.predicate_version,
            "feature_text_hash": visible_hash,
            "is_degenerate": is_degenerate,
            "gt_precheck_status": gt_status,
        }))
    return built


def _digest(facts: SceneFacts) -> SceneFactsDigest:
    return SceneFactsDigest(
        signal_phase="red" if facts.has_red_light else "unknown",
        phase_source=facts.traffic_light_status_source,
        ego_turn_intent=facts.ego_turn_intent,
        governing_stop_line_signed_m=facts.governing_stop_line_signed_m,
        pedestrian_in_forward_crosswalk=facts.pedestrian_in_forward_crosswalk,
        pedestrian_moving=facts.pedestrian_moving,
        oncoming_vehicle_moving=facts.oncoming_vehicle_moving,
        location_is_intersection=facts.location_is_intersection,
    )


def _default_active_rule_ids(scenario_type: ScenarioType) -> set[str]:
    if scenario_type == ScenarioType.RED_LIGHT:
        return {"R-SIG-01"}
    if scenario_type == ScenarioType.PEDESTRIAN:
        return {"R-YLD-01"}
    if scenario_type == ScenarioType.ONCOMING:
        return {"R-YLD-02"}
    return set()


def relabel_benchmark_labels(
    labels: BenchmarkLabels,
    features: list[TrajectoryFeatures],
    facts: SceneFacts,
    active_rule_ids: set[str],
    phase_validity_horizon_s: float = 2.0,
) -> BenchmarkLabels:
    """在 RuleSubgraph.active_rules 生成后按同源谓词重打匿名标签。"""
    relabeled: list[CandidateBenchmarkLabel] = []
    for label, feature in zip(labels.labels, features, strict=True):
        outcome = evaluate_active_rules(
            feature,
            _digest(facts),
            active_rule_ids,
            phase_validity_horizon_s=phase_validity_horizon_s,
        )
        verdict = outcome.verdict
        reason = outcome.reason
        exclude_reason: ExcludeReason | None = (
            "predicate_not_decidable" if verdict == "exclude" else None
        )
        if facts.window_insufficient:
            verdict, reason = "exclude", "not_evaluable_short_window"
            exclude_reason = "not_evaluable_short_window"
        elif label.exclude_reason == "no_violation_injectable":
            verdict, reason = "exclude", "no_violation_injectable"
            exclude_reason = "no_violation_injectable"
        elif label.is_degenerate:
            if label.variant_type == TrajectoryVariantType.ILLEGAL:
                verdict, reason = "exclude", "no_violation_injectable"
                exclude_reason = "no_violation_injectable"
            else:
                verdict, reason = "exclude", "injection_degenerate"
                exclude_reason = "injection_degenerate"
        elif label.variant_type == TrajectoryVariantType.GROUND_TRUTH and verdict == "vetoed":
            verdict, reason = "exclude", f"gt_precheck_not_evaluable: {reason}"
            exclude_reason = "gt_precheck_not_evaluable"
        gt_status = label.gt_precheck_status
        if label.variant_type == TrajectoryVariantType.GROUND_TRUTH:
            if facts.window_insufficient or not outcome.is_decisive:
                gt_status = "not_evaluable"
            else:
                gt_status = "passed" if outcome.verdict == "cleared" else "failed"
        relabeled.append(label.model_copy(update={
            "expected_verdict": verdict,
            "expected_verdict_reason": reason,
            "exclude_reason": exclude_reason,
            "predicate_version": outcome.predicate_version,
            "gt_precheck_status": gt_status,
            "difficulty": _difficulty(
                label.variant_type,
                outcome.is_decisive,
                outcome.reason,
            ),
        }))
    return labels.model_copy(update={"labels": relabeled})


def _difficulty(
    variant_type: TrajectoryVariantType,
    is_decisive: bool,
    reason: str,
) -> Literal["easy", "medium", "hard"]:
    if is_decisive:
        return "easy"
    if variant_type == TrajectoryVariantType.HARD_CASE:
        return "hard"
    if "missing" in reason or "unknown" in reason or reason in {
        "no_governing_signal_or_line",
        "start_beyond_line",
        "right_turn_on_red_exemption",
        "no_shared_decisive_predicate_for_active_rules",
    }:
        return "hard"
    return "medium"
