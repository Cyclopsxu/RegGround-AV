"""分层评估口径测试。"""

import json
from pathlib import Path

import pytest

from src.evaluation_report import evaluate_entries, load_versioned_jsonl_entries


def _entry(
    chosen_variant: str,
    chosen_expected: str,
    *,
    decided_by: str = "llm",
) -> dict:
    return {
        "ok": True,
        "layer1": {
            "benchmark_labels_debug_only": {
                "labels": [
                    {
                        "anonymous_id": "traj_a",
                        "variant_type": chosen_variant,
                        "expected_verdict": chosen_expected,
                    },
                    {
                        "anonymous_id": "traj_b",
                        "variant_type": "ground_truth",
                        "expected_verdict": "cleared",
                    },
                ]
            }
        },
        "final_label": {
            "chosen_trajectory_id": "traj_a",
            "preference_ranking": ["traj_a", "traj_b"],
            "verdicts": [
                {
                    "trajectory_id": "traj_a",
                    "veto": {"status": chosen_expected, "decided_by": decided_by},
                },
                {
                    "trajectory_id": "traj_b",
                    "veto": {"status": "cleared", "decided_by": decided_by},
                },
            ],
        },
    }


def test_hard_case_is_unscorable_and_illegal_chosen_is_primary_metric() -> None:
    report = evaluate_entries([
        _entry("hard_case", "cleared"),
        _entry("illegal", "vetoed"),
    ])

    assert report["illegal_chosen"] == {"count": 1, "denominator": 2, "rate": 0.5}
    assert report["scorable"]["count"] == 1
    assert report["unscorable"]["reasons"] == {"hard_case_chosen": 1}


def test_rule_engine_subset_does_not_report_self_referential_accuracy() -> None:
    report = evaluate_entries([_entry("conservative", "cleared", decided_by="rule_engine")])

    rule_layer = report["verdict_layers"]["rule_engine"]
    assert rule_layer["evaluated"] == 2
    assert rule_layer["benchmark_accuracy_reported"] is False
    assert rule_layer["required_external_metric"] == "predicate_human_agreement"


def test_zero_llm_denominator_is_explicitly_not_evaluable() -> None:
    report = evaluate_entries([_entry("conservative", "cleared", decided_by="rule_engine")])

    llm_layer = report["verdict_layers"]["llm"]
    assert llm_layer["evaluated"] == 0
    assert llm_layer["benchmark_accuracy"] is None
    assert llm_layer["accuracy_status"] == "not_evaluable"


def test_versioned_loader_rejects_cross_version_and_mixed_jsonl(
    tmp_path: Path,
) -> None:
    path = tmp_path / "results.jsonl"
    items = [
        {"type": "manifest", "benchmark_data_version": "2026-07-11-v2"},
        {
            "type": "record",
            "data": {
                "layer1": {
                    "benchmark_labels_debug_only": {
                        "data_version": "2026-07-v2",
                    }
                }
            },
        },
    ]
    path.write_text(
        "\n".join(json.dumps(item) for item in items) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="版本不匹配"):
        load_versioned_jsonl_entries(path, benchmark_version="v1")
    with pytest.raises(ValueError, match="版本不匹配"):
        load_versioned_jsonl_entries(path, benchmark_version="v2")
