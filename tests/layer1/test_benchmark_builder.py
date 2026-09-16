"""由几何结果生成 benchmark 标签的回归测试。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.layer1 import benchmark_builder
from src.layer1.benchmark_builder import (
    build_benchmark_candidates,
    label_benchmark_candidates,
    relabel_benchmark_labels,
)
from src.layer1.candidate_anonymizer import CandidateAnonymizer
from src.layer1.models import (
    AugmentedTrajectory,
    ConflictZone,
    ScenarioType,
    SceneDescription,
    SceneFacts,
    Trajectory,
    TrajectorySource,
    TrajectoryVariantType,
    Waypoint,
)
from src.layer1.trajectory_analyzer import TrajectoryAnalyzer


def _candidate(
    variant: TrajectoryVariantType,
    xs: list[float],
) -> AugmentedTrajectory:
    return AugmentedTrajectory(
        trajectory=Trajectory(
            traj_id=f"internal_{variant.value}",
            source=TrajectorySource.SYNTHETIC,
            waypoints=[
                Waypoint(x=x, y=0.0, t=float(index))
                for index, x in enumerate(xs)
            ],
        ),
        variant_type=variant,
        expected_verdict="exclude",
    )


def _facts() -> SceneFacts:
    return SceneFacts(
        has_red_light=True,
        traffic_light_status_source="drivelm_status",
        ego_turn_intent="straight",
        governing_stop_line_signed_m=5.0,
        conflict_zones=[ConflictZone(
            kind="stop_line",
            polygon_xy=[(5.0, -1.0), (6.0, -1.0), (6.0, 1.0), (5.0, 1.0)],
            source_layer="stop_line",
        )],
    )


def test_variant_intent_does_not_determine_expected_verdict() -> None:
    candidates = [
        _candidate(TrajectoryVariantType.ILLEGAL, [0.0, 2.0, 4.0, 4.0, 4.0]),
        _candidate(TrajectoryVariantType.AGGRESSIVE, [0.0, 2.0, 5.5, 7.0]),
    ]

    built = build_benchmark_candidates(
        candidates,
        TrajectoryAnalyzer(),
        SceneDescription(),
        _facts(),
        ScenarioType.RED_LIGHT,
    )

    assert built[0].expected_verdict == "cleared"
    assert built[1].expected_verdict == "vetoed"
    assert all(item.expected_verdict_reason for item in built)
    assert all(item.predicate_version for item in built)


def test_gt_conflict_becomes_exclude_and_duplicate_is_explicit() -> None:
    candidates = [
        _candidate(TrajectoryVariantType.GROUND_TRUTH, [0.0, 2.0, 5.5, 7.0]),
        _candidate(TrajectoryVariantType.AGGRESSIVE, [0.0, 2.0, 5.5, 7.0]),
    ]

    built = build_benchmark_candidates(
        candidates,
        TrajectoryAnalyzer(),
        SceneDescription(),
        _facts(),
        ScenarioType.RED_LIGHT,
    )

    assert built[0].expected_verdict == "exclude"
    assert built[0].exclude_reason == "gt_precheck_not_evaluable"
    assert built[0].gt_precheck_status == "failed"
    assert built[1].expected_verdict == "exclude"
    assert built[1].exclude_reason == "injection_degenerate"
    assert built[1].is_degenerate is True
    assert built[0].feature_text_hash == built[1].feature_text_hash


def test_gt_outside_predicate_domain_is_not_evaluable() -> None:
    built = build_benchmark_candidates(
        [_candidate(TrajectoryVariantType.GROUND_TRUTH, [0.0, 1.0, 2.0])],
        TrajectoryAnalyzer(),
        SceneDescription(),
        SceneFacts(),
        ScenarioType.UNKNOWN,
    )

    assert built[0].gt_precheck_status == "not_evaluable"


def test_red_suboptimal_late_crossing_remains_not_decidable() -> None:
    built = build_benchmark_candidates(
        [_candidate(TrajectoryVariantType.SUBOPTIMAL, [0.0, 1.0, 2.0, 3.0, 4.0, 5.5])],
        TrajectoryAnalyzer(),
        SceneDescription(),
        _facts(),
        ScenarioType.RED_LIGHT,
    )

    assert built[0].expected_verdict == "exclude"
    assert built[0].exclude_reason == "predicate_not_decidable"
    assert built[0].expected_verdict_reason == "borderline_or_phase_horizon_unknown"


@pytest.mark.parametrize(
    "reason",
    [
        "no_governing_signal_or_line",
        "start_beyond_line",
        "right_turn_on_red_exemption",
        "borderline_or_phase_horizon_unknown",
        "pedestrian_trigger_missing_or_unknown",
        "pedestrian_interaction_missing",
        "pedestrian_gap_borderline",
        "no_shared_decisive_predicate_for_active_rules",
    ],
)
def test_not_decidable_never_falls_back_to_decisive_label(
    monkeypatch: pytest.MonkeyPatch,
    reason: str,
) -> None:
    outcome = SimpleNamespace(
        verdict="exclude",
        reason=reason,
        is_decisive=False,
        predicate_version="test-not-decidable",
    )
    monkeypatch.setattr(
        benchmark_builder,
        "evaluate_active_rules",
        lambda *_args, **_kwargs: outcome,
    )
    analyzer = TrajectoryAnalyzer()
    candidate = _candidate(TrajectoryVariantType.SUBOPTIMAL, [0.0, 1.0, 2.0])
    candidate = candidate.model_copy(update={
        "features": analyzer.analyze(
            candidate.trajectory,
            SceneDescription(),
            SceneFacts(),
        )
    })

    labeled = label_benchmark_candidates(
        [candidate],
        SceneFacts(),
        ScenarioType.UNKNOWN,
    )
    assert labeled[0].expected_verdict == "exclude"
    assert labeled[0].exclude_reason == "predicate_not_decidable"

    _, anonymous = CandidateAnonymizer().anonymize(labeled)
    assert labeled[0].features is not None
    relabeled = relabel_benchmark_labels(
        anonymous,
        [labeled[0].features],
        SceneFacts(),
        set(),
    )
    assert relabeled.labels[0].expected_verdict == "exclude"
    assert relabeled.labels[0].exclude_reason == "predicate_not_decidable"
