from __future__ import annotations

import csv
import json

import pytest

from experiments.ablation_2x2.build_gold_scenarios import (
    PREFERENCE_SIDECAR,
    VERDICT_SIDECAR,
    build,
)
from experiments.ablation_2x2.build_inference_scenarios import build as build_inference
from experiments.ablation_2x2.metrics import compute_metrics, kendall_tau_b

requires_private_annotation_data = pytest.mark.skipif(
    not (VERDICT_SIDECAR.exists() and PREFERENCE_SIDECAR.exists()),
    reason="公开代码版不包含人工标注 private sidecar",
)


def _jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_verdict_gold(path) -> None:
    fields = [
        "audit_id",
        "human_verdict",
        "human_applicable_rule_ids",
        "human_violated_rule_ids",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in _jsonl(VERDICT_SIDECAR):
            writer.writerow(
                {
                    "audit_id": row["audit_id"],
                    "human_verdict": "cleared",
                    "human_applicable_rule_ids": "NONE",
                    "human_violated_rule_ids": "NONE",
                }
            )


def _write_preference_gold(path) -> None:
    fields = ["audit_id", "human_best_trajectory", "human_ranking"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in _jsonl(PREFERENCE_SIDECAR):
            columns = list(row["column_to_trajectory_id"])
            writer.writerow(
                {
                    "audit_id": row["audit_id"],
                    "human_best_trajectory": columns[0],
                    "human_ranking": ">".join(columns),
                }
            )


@requires_private_annotation_data
def test_builds_frozen_union_of_both_formal_human_annotation_tables(tmp_path) -> None:
    verdict = tmp_path / "verdict.csv"
    preference = tmp_path / "preference.csv"
    output = tmp_path / "gold_scenarios.json"
    _write_verdict_gold(verdict)
    _write_preference_gold(preference)

    payload = build(
        verdict_adjudicated=verdict,
        preference_adjudicated=preference,
        output_path=output,
    )

    assert payload["counts"] == {
        "verdict_rows": 200,
        "verdict_scenes": 78,
        "preference_scenes": 36,
        "overlap_scenes": 32,
        "preference_only_scenes": 4,
        "union_scenes": 82,
    }
    assert sum(len(scene["gold"]["verdicts"]) for scene in payload["scenarios"]) == 200
    assert sum("preference_ranking" in scene["gold"] for scene in payload["scenarios"]) == 36
    assert all(scene["candidates"] for scene in payload["scenarios"])
    assert json.loads(output.read_text(encoding="utf-8"))["gold_finalized"] is True


@requires_private_annotation_data
def test_builds_same_union_without_unfinished_gold(tmp_path) -> None:
    output = tmp_path / "inference_scenarios.json"
    payload = build_inference(output)
    assert payload["inference_only"] is True
    assert payload["gold_finalized"] is False
    assert payload["counts"]["union_scenes"] == 82
    assert all("gold" not in scene for scene in payload["scenarios"])


def test_metrics_skip_preference_for_verdict_only_scenes() -> None:
    base_response = {
        "verdicts": [
            {
                "trajectory_id": "traj_a",
                "status": "cleared",
                "cited_rule_ids": [],
            }
        ],
        "preference_ranking": ["traj_a"],
        "chosen_trajectory_id": "traj_a",
        "preference_pairs": [],
    }
    verdict_only = {
        "frame_token": "frame_a",
        "scenario_type": "red_light",
        "difficulty": "easy",
        "gold": {"verdicts": [{"trajectory_id": "traj_a", "status": "cleared"}]},
        "response_final": base_response,
        "citations_first": [],
        "citations_final": [],
        "decision_citation_counts": [0],
    }
    preference = {
        **verdict_only,
        "frame_token": "frame_b",
        "gold": {
            "verdicts": [],
            "preference_ranking": ["traj_a"],
            "preference_tiers": [["traj_a"]],
            "chosen_trajectory_id": "traj_a",
            "preference_pairs": [],
        },
    }
    records = {
        name: [verdict_only, preference] for name in ("full", "no_validation", "no_rag", "baseline")
    }
    metrics = compute_metrics(records, dry_run=False)
    assert metrics["conditions"]["full"]["chosen_top1_accuracy"] == 1.0
    assert metrics["conditions"]["full"]["verdict_kappa"] == 1.0


def test_kendall_tau_b_handles_human_ties() -> None:
    assert kendall_tau_b([["a", "b"], ["c"]], ["a", "b", "c"]) is not None
