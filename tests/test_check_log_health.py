"""日志健康检查的冒烟规模、降级记账与 S9 白名单回归。"""

import pytest

from scripts.check_log_health import build_batch_report, check_record, compliance_direction


def test_batch_expected_records_comes_from_eval_manifest() -> None:
    manifest = {"eval_set": {"entries": 2}}
    records = [
        {"data": {"ok": True, "final_label": {"status": "complete"}}},
        {"data": {"ok": True, "final_label": {"status": "complete"}}},
    ]

    summary, details = build_batch_report(manifest, records, {}, [])

    assert summary["totals_ok"] is True
    assert details["batch_checks"]["B2_actual"]["expected"]["records"] == 2


def test_batch_missing_expected_records_fails_closed() -> None:
    records = [{"data": {"ok": True, "final_label": {"status": "complete"}}}]

    with pytest.raises(ValueError, match="expected record count is missing"):
        build_batch_report({}, records, {}, [])


def test_degraded_pairwise_accounting_accepts_planned_to_zero() -> None:
    verdicts = [
        {
            "trajectory_id": f"traj_{suffix}",
            "veto": {
                "status": "cleared",
                "decided_by": "llm",
                "reason": "未检测到违规",
            },
        }
        for suffix in "abcdef"
    ]
    entry = {
        "data": {
            "ok": True,
            "index": 16,
            "layer1": {"context": {"candidate_trajectory_ids": [
                item["trajectory_id"] for item in verdicts
            ]}},
            "final_label": {
                "status": "partial",
                "partial_reasons": ["stage_degraded"],
                "selection_outcome": "selected",
                "chosen_trajectory_id": "traj_a",
                "preference_ranking": [item["trajectory_id"] for item in verdicts],
                "verdicts": verdicts,
                "counts": {"cleared": 6, "vetoed": 0, "uncertain": 0},
                "pairwise_comparisons_count": 0,
                "stage_diagnostics": [{
                    "stage": "pairwise_ranking",
                    "succeeded": False,
                    "planned_pairwise_comparisons": 15,
                    "completed_pairwise_comparisons": 0,
                }],
                "citation_validity": 1.0,
                "natural_language_summary": "选择 traj_a。",
            },
        },
    }

    findings = check_record(entry)["findings"]

    assert not [item for item in findings if item["check"] == "S5"]


def test_s9_whitelists_machine_reason_and_unrelated_rule_negation() -> None:
    assert not compliance_direction(
        "crossed_governing_line_during_valid_red_without_full_stop",
        "vetoed",
    )
    assert not compliance_direction(
        "轨迹违反[R-SIG-01]；行人义务未触发，因此不违反[R-YLD-01]。",
        "vetoed",
    )
    assert compliance_direction("轨迹通过硬性规则。", "vetoed")


def test_deferred_retry_started_and_finished_timestamps_are_accepted() -> None:
    entry = {
        "data": {
            "ok": True,
            "index": 7,
            "layer1": {"context": {"candidate_trajectory_ids": ["traj_a"]}},
            "final_label": {
                "status": "complete",
                "partial_reasons": [],
                "selection_outcome": "selected",
                "chosen_trajectory_id": "traj_a",
                "preference_ranking": ["traj_a"],
                "verdicts": [{
                    "trajectory_id": "traj_a",
                    "veto": {
                        "status": "cleared",
                        "decided_by": "llm",
                        "completed_by": "deferred_retry",
                        "reason": "未检测到违规",
                        "deferred_attempts": [{
                            "started_at": "2026-07-18T00:00:00Z",
                            "finished_at": "2026-07-18T00:00:01Z",
                        }],
                    },
                }],
                "counts": {"cleared": 1, "vetoed": 0, "uncertain": 0},
                "pairwise_comparisons_count": 0,
                "stage_diagnostics": [{
                    "stage": "pairwise_ranking",
                    "succeeded": True,
                    "planned_pairwise_comparisons": 0,
                    "completed_pairwise_comparisons": 0,
                }],
                "citation_validity": 1.0,
                "natural_language_summary": "选择 traj_a。",
            },
        },
    }

    findings = check_record(entry)["findings"]

    assert not [item for item in findings if item["check"] == "S7"]
