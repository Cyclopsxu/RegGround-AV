"""Benchmark v2 的 pair 指标、可接受集合与 override 后打标。"""

from src.evaluation_report import evaluate_entries
from src.layer1.benchmark_builder import relabel_benchmark_labels
from src.layer1.models import (
    BenchmarkLabels,
    CandidateBenchmarkLabel,
    ConflictZoneMetrics,
    SceneFacts,
    TrajectoryFeatures,
    TrajectoryVariantType,
)


def test_pair_metrics_ignore_excluded_candidate_and_report_decision_source() -> None:
    entry = {
        "ok": True,
        "layer1": {
            "context": {"scenario_type": "red_light"},
            "benchmark_labels_debug_only": {"labels": [
                {
                    "anonymous_id": "traj_a",
                    "variant_type": "conservative",
                    "expected_verdict": "cleared",
                    "difficulty": "easy",
                },
                {
                    "anonymous_id": "traj_b",
                    "variant_type": "illegal",
                    "expected_verdict": "vetoed",
                    "difficulty": "easy",
                },
                {
                    "anonymous_id": "traj_c",
                    "variant_type": "hard_case",
                    "expected_verdict": "exclude",
                    "exclude_reason": "predicate_not_decidable",
                    "difficulty": "hard",
                },
            ]},
        },
        "final_label": {
            "chosen_trajectory_id": "traj_a",
            "preference_ranking": ["traj_a", "traj_b"],
            "verdicts": [
                {"trajectory_id": "traj_a", "veto": {
                    "status": "cleared", "decided_by": "rule_engine"
                }},
                {"trajectory_id": "traj_b", "veto": {
                    "status": "vetoed", "decided_by": "rule_engine"
                }},
                {"trajectory_id": "traj_c", "veto": {
                    "status": "uncertain", "decided_by": "llm"
                }},
            ],
        },
    }
    report = evaluate_entries([entry])
    metrics = report["main_metrics_v2"]
    assert metrics["pair_consistency"] == 1.0
    assert metrics["scorable_pair_scene_count"] == 1
    assert metrics["scorable_pair_scene_rate"] == 1.0
    assert metrics["structurally_scorable_scene_count"] == 1
    assert metrics["structurally_unscorable_scene_count"] == 0
    assert metrics["scorable_pair_scene_count_within_structural"] == 1
    assert metrics["scorable_pair_scene_rate_within_structural"] == 1.0
    assert metrics["chosen_acceptable_rate"] == 1.0
    assert metrics["abstention_rate"] == 0.0
    assert report["excluded_candidates"] == 1
    stratum = report["by_scenario_difficulty"]["red_light|easy"]
    assert stratum["decided_by"]["rule_engine"] == 2


def test_override_after_retrieval_prevents_veto_expectation() -> None:
    metric = ConflictZoneMetrics(
        kind="stop_line",
        governing=True,
        min_distance_m=0.0,
        entered=True,
        entry_time_s=1.0,
        penetration_depth_m=0.5,
        stopped_before_zone=False,
    )
    feature = TrajectoryFeatures(
        max_speed_mps=4.0,
        min_speed_mps=4.0,
        mean_speed_mps=4.0,
        total_distance_m=10.0,
        max_lateral_offset_m=0.0,
        n_waypoints=3,
        conflict_zone_metrics=[metric],
    )
    labels = BenchmarkLabels(
        scene_id="scene",
        frame_token="frame",
        labels=[CandidateBenchmarkLabel(
            anonymous_id="traj_a",
            original_internal_id="internal",
            variant_type=TrajectoryVariantType.ILLEGAL,
            expected_verdict="vetoed",
            difficulty="easy",
        )],
    )
    relabeled = relabel_benchmark_labels(
        labels,
        [feature],
        SceneFacts(
            has_red_light=True,
            traffic_light_status_source="drivelm_status",
            ego_turn_intent="straight",
            governing_stop_line_signed_m=5.0,
        ),
        active_rule_ids=set(),
    )
    assert relabeled.labels[0].expected_verdict == "exclude"
    assert relabeled.labels[0].exclude_reason == "predicate_not_decidable"


def test_pair_coverage_baseline_uses_structurally_scorable_pool() -> None:
    def entry(
        scenario: str,
        *,
        facts: dict | None = None,
        features: list[dict] | None = None,
        has_pair: bool = False,
    ) -> dict:
        labels = []
        if has_pair:
            labels = [
                {"anonymous_id": "traj_a", "expected_verdict": "cleared"},
                {"anonymous_id": "traj_b", "expected_verdict": "vetoed"},
            ]
        return {
            "ok": True,
            "layer1": {
                "context": {
                    "scenario_type": scenario,
                    "scene_facts": facts or {},
                    "trajectory_features": features or [],
                },
                "benchmark_labels_debug_only": {"labels": labels},
            },
            "final_label": {
                "chosen_trajectory_id": None,
                "preference_ranking": [],
                "verdicts": [],
            },
        }

    entries = [entry("oncoming") for _ in range(5)]
    entries += [
        entry("red_light", facts={"governing_stop_line_signed_m": -1.0})
        for _ in range(3)
    ]
    entries += [entry("pedestrian") for _ in range(2)]
    entries += [entry(
        "pedestrian",
        facts={
            "pedestrian_in_forward_crosswalk": True,
            "pedestrian_moving": True,
        },
    )]
    entries += [entry("red_light", has_pair=True) for _ in range(7)]
    entries += [
        entry("red_light", facts={"window_insufficient": True}),
        entry("red_light"),
    ]

    metrics = evaluate_entries(entries)["main_metrics_v2"]

    assert metrics["structurally_scorable_scene_count"] == 8
    assert metrics["structurally_unscorable_scene_count"] == 12
    assert metrics["scorable_pair_scene_count_within_structural"] == 7
    assert metrics["scorable_pair_scene_rate_within_structural"] == 7 / 8
    assert "gate3_smoke_structural_pair_target" not in metrics
