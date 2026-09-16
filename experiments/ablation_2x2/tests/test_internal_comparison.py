from __future__ import annotations

from copy import deepcopy

import pytest

from experiments.ablation_2x2.internal_comparison import (
    compute_internal_comparison,
    pairwise_agreement,
    validate_pairing,
)


def _record(condition: str, *, frame: str = "frame-1", valid: bool = True) -> dict:
    response = {
        "verdicts": [
            {"trajectory_id": "a", "status": "cleared"},
            {"trajectory_id": "b", "status": "vetoed"},
        ],
        "preference_ranking": ["a", "b"],
        "chosen_trajectory_id": "a",
    }
    citation = {"valid": valid}
    return {
        "condition": condition,
        "frame_token": frame,
        "scene_id": f"scene-{frame}",
        "scenario_type": "test",
        "difficulty": "easy",
        "response_first": deepcopy(response),
        "response_final": response,
        "citations_first": [citation],
        "citations_final": [citation],
        "decision_citation_counts": [1, 0],
        "tokens": 100,
        "duration_ms": 1000,
        "retry_count": 0,
        "schema_retry_count": 0,
    }


def _four_conditions() -> dict[str, list[dict]]:
    return {
        condition: [_record(condition)]
        for condition in ("full", "no_validation", "no_rag", "baseline")
    }


def test_internal_comparison_uses_only_paired_gold_independent_metrics() -> None:
    metrics = compute_internal_comparison(_four_conditions())

    assert metrics["validation"]["scenes_per_condition"] == 1
    assert metrics["conditions"]["full"]["machine_citation_validity_final"] == 1.0
    assert metrics["conditions"]["full"]["no_citation_decision_rate"] == 0.5
    assert metrics["pairwise_agreement"]["full__vs__baseline"]["status_agreement"] == 1.0
    assert metrics["factorial_contrasts"]["validation_with_rag"]["tokens"][
        "mean_difference"
    ] == 0.0


def test_pairwise_agreement_detects_status_and_ranking_changes() -> None:
    left = [_record("full")]
    right_record = deepcopy(_record("baseline"))
    right_record["response_final"]["verdicts"][1]["status"] = "cleared"
    right_record["response_final"]["preference_ranking"] = ["b", "a"]
    right_record["response_final"]["chosen_trajectory_id"] = "b"

    result = pairwise_agreement(left, [right_record])

    assert result["status_agreement"] == 0.5
    assert result["chosen_agreement"] == 0.0
    assert result["exact_ranking_agreement"] == 0.0
    assert result["mean_kendall_tau"] == -1.0


def test_pairing_rejects_gold_or_mismatched_frames() -> None:
    records = _four_conditions()
    records["full"][0]["gold"] = {}
    with pytest.raises(ValueError, match="human gold"):
        validate_pairing(records)

    records = _four_conditions()
    records["baseline"][0]["frame_token"] = "other"
    with pytest.raises(ValueError, match="same frozen frames"):
        validate_pairing(records)
