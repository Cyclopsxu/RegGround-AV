"""Rule engine 与 benchmark 共用的中性合规谓词。"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict

from src.layer1.models import SceneFactsDigest, TrajectoryFeatures

PREDICATE_VERSION = "shared-predicates-2026-07-v2"
STOP_LINE_PENETRATION_EPSILON_M = 0.15


class PredicateDecision(StrEnum):
    DECISIVE_VETO = "decisive_veto"
    DECISIVE_CLEAR = "decisive_clear"
    NOT_DECIDABLE = "not_decidable"


class PredicateResult(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    decision: PredicateDecision
    reason: str


class PredicateOutcome(BaseModel):
    """兼容 benchmark 的三值输出。"""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    verdict: Literal["cleared", "vetoed", "exclude"]
    reason: str
    is_decisive: bool
    predicate_version: str = PREDICATE_VERSION


def red_light_crossing_verdict(
    feature: TrajectoryFeatures,
    digest: SceneFactsDigest,
    *,
    phase_validity_horizon_s: float = 2.0,
) -> PredicateResult:
    """判断红灯停止线事实；不充分或存在豁免时交给语义层。"""
    if digest.signal_phase != "red" or digest.governing_stop_line_signed_m is None:
        return PredicateResult(
            decision=PredicateDecision.NOT_DECIDABLE,
            reason="no_governing_signal_or_line",
        )
    if digest.governing_stop_line_signed_m < 0:
        return PredicateResult(
            decision=PredicateDecision.NOT_DECIDABLE,
            reason="start_beyond_line",
        )
    if digest.ego_turn_intent == "right":
        return PredicateResult(
            decision=PredicateDecision.NOT_DECIDABLE,
            reason="right_turn_on_red_exemption",
        )

    stop_line = next(
        (
            metric
            for metric in feature.conflict_zone_metrics
            if metric.kind == "stop_line" and metric.governing
        ),
        None,
    )
    entered = stop_line.entered if stop_line is not None else feature.entered_conflict_zone
    entry_time = stop_line.entry_time_s if stop_line is not None else None
    stopped_before = (
        stop_line.stopped_before_zone if stop_line is not None else feature.stopped_before_zone
    )
    if (
        entered
        and entry_time is not None
        and entry_time <= phase_validity_horizon_s
        and stop_line is not None
        and stop_line.penetration_depth_m is not None
        and stop_line.penetration_depth_m >= STOP_LINE_PENETRATION_EPSILON_M
        and stopped_before is False
    ):
        return PredicateResult(
            decision=PredicateDecision.DECISIVE_VETO,
            reason="crossed_governing_line_during_valid_red_without_full_stop",
        )
    if not entered and feature.full_stop and stopped_before is True:
        return PredicateResult(
            decision=PredicateDecision.DECISIVE_CLEAR,
            reason="stopped_before_governing_line_without_crossing",
        )
    return PredicateResult(
        decision=PredicateDecision.NOT_DECIDABLE,
        reason="borderline_or_phase_horizon_unknown",
    )


def pedestrian_yield_verdict(
    feature: TrajectoryFeatures,
    digest: SceneFactsDigest,
) -> PredicateResult:
    """判断明确的前方横道行人时间冲突。"""
    if not digest.pedestrian_in_forward_crosswalk or digest.pedestrian_moving is not True:
        return PredicateResult(
            decision=PredicateDecision.NOT_DECIDABLE,
            reason="pedestrian_trigger_missing_or_unknown",
        )
    interaction = next(
        (
            item
            for item in feature.agent_interactions
            if item.agent_kind == "pedestrian" and item.min_time_gap_s is not None
        ),
        None,
    )
    crossing = next(
        (metric for metric in feature.conflict_zone_metrics if metric.kind == "ped_crossing"),
        None,
    )
    if crossing is not None and crossing.stopped_before_zone is True:
        return PredicateResult(
            decision=PredicateDecision.DECISIVE_CLEAR,
            reason="pedestrian_stopped_before_crossing",
        )
    if interaction is None or crossing is None:
        return PredicateResult(
            decision=PredicateDecision.NOT_DECIDABLE,
            reason="pedestrian_interaction_missing",
        )
    if (
        interaction.min_time_gap_s is not None
        and interaction.min_time_gap_s <= 0
        and crossing.stopped_before_zone is False
    ):
        return PredicateResult(
            decision=PredicateDecision.DECISIVE_VETO,
            reason="pedestrian_zone_time_overlap_without_prior_stop",
        )
    if interaction.min_time_gap_s is not None and interaction.min_time_gap_s > 3.0:
        return PredicateResult(
            decision=PredicateDecision.DECISIVE_CLEAR,
            reason="pedestrian_conflict_cleared_with_time_gap",
        )
    return PredicateResult(
        decision=PredicateDecision.NOT_DECIDABLE,
        reason="pedestrian_gap_borderline",
    )


def evaluate_active_rules(
    feature: TrajectoryFeatures,
    digest: SceneFactsDigest,
    active_rule_ids: set[str],
    *,
    phase_validity_horizon_s: float = 2.0,
) -> PredicateOutcome:
    """按 override 后的 active rules 选择共享谓词。"""
    if "R-SIG-01" in active_rule_ids:
        result = red_light_crossing_verdict(
            feature,
            digest,
            phase_validity_horizon_s=phase_validity_horizon_s,
        )
    elif "R-YLD-01" in active_rule_ids:
        result = pedestrian_yield_verdict(feature, digest)
    else:
        result = PredicateResult(
            decision=PredicateDecision.NOT_DECIDABLE,
            reason="no_shared_decisive_predicate_for_active_rules",
        )
    return _to_outcome(result)


def _to_outcome(result: PredicateResult) -> PredicateOutcome:
    if result.decision == PredicateDecision.DECISIVE_VETO:
        return PredicateOutcome(verdict="vetoed", reason=result.reason, is_decisive=True)
    if result.decision == PredicateDecision.DECISIVE_CLEAR:
        return PredicateOutcome(verdict="cleared", reason=result.reason, is_decisive=True)
    return PredicateOutcome(verdict="exclude", reason=result.reason, is_decisive=False)


def evaluate_red_stop_line_predicate(feature: TrajectoryFeatures) -> PredicateOutcome:
    """旧调用的保守兼容层；缺少要件时不得直裁。"""
    digest = SceneFactsDigest(
        signal_phase="red",
        phase_source="drivelm_status",
        ego_turn_intent="straight",
        governing_stop_line_signed_m=(
            1.0 if feature.min_distance_to_conflict_m is not None else None
        ),
        pedestrian_in_forward_crosswalk=False,
        pedestrian_moving=None,
        oncoming_vehicle_moving=False,
        location_is_intersection=None,
    )
    return _to_outcome(red_light_crossing_verdict(feature, digest))
