"""Gold-independent analysis for a completed 2×2 ablation run.

This module intentionally excludes every metric that requires human labels.  It
compares model behaviour, mechanically checkable citations, and runtime cost on
the same frozen frames across all four conditions.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from experiments.ablation_2x2.conditions import CONDITIONS
from experiments.ablation_2x2.metrics import kendall_tau

CONDITION_NAMES = tuple(condition.name for condition in CONDITIONS)
CONTRASTS = {
    "rag_with_validation": ("full", "no_rag"),
    "rag_without_validation": ("no_validation", "baseline"),
    "validation_with_rag": ("full", "no_validation"),
    "validation_without_rag": ("no_rag", "baseline"),
}
SCENE_METRICS = (
    "machine_invalid_citations_per_decision",
    "citations_per_decision",
    "no_citation_decision_rate",
    "veto_rate",
    "uncertain_rate",
    "chosen_available",
    "tokens",
    "duration_ms",
    "citation_retries",
    "schema_retries",
    "enforcement_exhausted",
)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _bootstrap_mean_ci(
    values: Sequence[float], *, samples: int = 5000, seed: int = 20260802
) -> list[float]:
    if not values:
        return [0.0, 0.0]
    generator = random.Random(seed)
    means = [
        _mean(values[generator.randrange(len(values))] for _ in values)
        for _ in range(samples)
    ]
    return [_percentile(means, 0.025), _percentile(means, 0.975)]


def load_records(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    records_by_condition: dict[str, list[dict[str, Any]]] = {}
    for condition in CONDITION_NAMES:
        path = run_dir / condition / "records.jsonl"
        with path.open(encoding="utf-8") as handle:
            records_by_condition[condition] = [json.loads(line) for line in handle if line.strip()]
    return records_by_condition


def validate_pairing(records_by_condition: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    if set(records_by_condition) != set(CONDITION_NAMES):
        raise ValueError(
            f"expected conditions {CONDITION_NAMES}, got {tuple(records_by_condition)}"
        )

    frame_lists: dict[str, list[str]] = {}
    for condition, records in records_by_condition.items():
        frames = [str(record["frame_token"]) for record in records]
        if len(frames) != len(set(frames)):
            raise ValueError(f"duplicate frame_token in condition {condition}")
        if any("gold" in record for record in records):
            raise ValueError("internal comparison must not consume human gold fields")
        frame_lists[condition] = frames

    reference = set(frame_lists[CONDITION_NAMES[0]])
    if any(set(frames) != reference for frames in frame_lists.values()):
        raise ValueError("conditions are not paired on the same frozen frames")
    return {
        "condition_count": len(CONDITION_NAMES),
        "scenes_per_condition": len(reference),
        "unique_frames_per_condition": {
            condition: len(set(frames)) for condition, frames in frame_lists.items()
        },
        "same_frame_set": True,
        "gold_fields_absent": True,
    }


def _scene_metrics(record: dict[str, Any]) -> dict[str, Any]:
    verdicts = record["response_final"]["verdicts"]
    decisions = len(verdicts)
    citations = record["citations_final"]
    invalid = sum(not event["valid"] for event in citations)
    no_citation = sum(count == 0 for count in record["decision_citation_counts"])
    statuses = Counter(verdict["status"] for verdict in verdicts)
    return {
        "frame_token": record["frame_token"],
        "scene_id": record["scene_id"],
        "scenario_type": record["scenario_type"],
        "difficulty": record["difficulty"],
        "n_decisions": decisions,
        "machine_invalid_citations_per_decision": invalid / decisions if decisions else 0.0,
        "citations_per_decision": len(citations) / decisions if decisions else 0.0,
        "no_citation_decision_rate": no_citation / decisions if decisions else 0.0,
        "veto_rate": statuses["vetoed"] / decisions if decisions else 0.0,
        "uncertain_rate": statuses["uncertain"] / decisions if decisions else 0.0,
        "chosen_available": float(record["response_final"].get("chosen_trajectory_id") is not None),
        "tokens": float(record.get("tokens", 0)),
        "duration_ms": float(record.get("duration_ms", 0)),
        "citation_retries": float(record.get("retry_count", 0)),
        "schema_retries": float(record.get("schema_retry_count", 0)),
        "enforcement_exhausted": float(record.get("enforcement_exhausted", False)),
    }


def _first_final_agreement(records: list[dict[str, Any]]) -> dict[str, Any]:
    status_matches: list[bool] = []
    chosen_matches: list[bool] = []
    ranking_matches: list[bool] = []
    taus: list[float] = []
    for record in records:
        first = record["response_first"]
        final = record["response_final"]
        first_status = {item["trajectory_id"]: item["status"] for item in first["verdicts"]}
        final_status = {item["trajectory_id"]: item["status"] for item in final["verdicts"]}
        status_matches.extend(first_status[item] == final_status[item] for item in first_status)
        chosen_matches.append(
            first.get("chosen_trajectory_id") == final.get("chosen_trajectory_id")
        )
        ranking_matches.append(first["preference_ranking"] == final["preference_ranking"])
        tau = kendall_tau(first["preference_ranking"], final["preference_ranking"])
        if tau is not None:
            taus.append(tau)
    return {
        "n_scenes": len(records),
        "status_agreement": _mean(float(value) for value in status_matches),
        "chosen_agreement": _mean(float(value) for value in chosen_matches),
        "exact_ranking_agreement": _mean(float(value) for value in ranking_matches),
        "mean_kendall_tau": _mean(taus),
    }


def summarize_condition(records: list[dict[str, Any]]) -> dict[str, Any]:
    first = [event for record in records for event in record["citations_first"]]
    final = [event for record in records for event in record["citations_final"]]
    verdicts = [verdict for record in records for verdict in record["response_final"]["verdicts"]]
    status_counts = Counter(verdict["status"] for verdict in verdicts)
    decision_count = len(verdicts)
    no_citation = sum(
        count == 0 for record in records for count in record["decision_citation_counts"]
    )
    durations = [float(record.get("duration_ms", 0)) for record in records]
    tokens = [float(record.get("tokens", 0)) for record in records]
    first_valid = sum(event["valid"] for event in first)
    final_valid = sum(event["valid"] for event in final)
    scenario_rows = [_scene_metrics(record) for record in records]
    retried_records = [record for record in records if record.get("retry_count", 0)]

    return {
        "n_scenes": len(records),
        "n_decisions": decision_count,
        "status_counts": dict(status_counts),
        "veto_rate": status_counts["vetoed"] / decision_count if decision_count else 0.0,
        "uncertain_rate": status_counts["uncertain"] / decision_count if decision_count else 0.0,
        "chosen_available_rate": _mean(row["chosen_available"] for row in scenario_rows),
        "citations_first": len(first),
        "citations_final": len(final),
        "machine_citation_validity_first": first_valid / len(first) if first else None,
        "machine_citation_validity_final": final_valid / len(final) if final else None,
        "machine_invalid_citation_rate_final": 1 - final_valid / len(final) if final else None,
        "citations_per_decision": len(final) / decision_count if decision_count else 0.0,
        "no_citation_decision_rate": no_citation / decision_count if decision_count else 0.0,
        "tokens_total": int(sum(tokens)),
        "tokens_mean_per_scene": _mean(tokens),
        "duration_total_ms": int(sum(durations)),
        "duration_mean_ms": _mean(durations),
        "duration_p50_ms": statistics.median(durations) if durations else 0.0,
        "duration_p95_ms": _percentile(durations, 0.95),
        "citation_retries": sum(int(record.get("retry_count", 0)) for record in records),
        "schema_retries": sum(int(record.get("schema_retry_count", 0)) for record in records),
        "citation_check_violations_first": sum(
            record.get("citation_check_first") == "violated" for record in records
        ),
        "citation_check_violations_final": sum(
            record.get("citation_check_final") == "violated" for record in records
        ),
        "enforcement_exhausted": sum(
            int(record.get("enforcement_exhausted", False)) for record in records
        ),
        "first_final_agreement": _first_final_agreement(records),
        "first_final_agreement_retried_only": _first_final_agreement(retried_records),
        "scenario_rows": scenario_rows,
    }


def _indexed(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record["frame_token"]): record for record in records}


def pairwise_agreement(
    left_records: list[dict[str, Any]], right_records: list[dict[str, Any]]
) -> dict[str, Any]:
    left_by_frame = _indexed(left_records)
    right_by_frame = _indexed(right_records)
    if set(left_by_frame) != set(right_by_frame):
        raise ValueError("pairwise agreement requires the same frames")

    status_matches: list[bool] = []
    chosen_matches: list[bool] = []
    ranking_matches: list[bool] = []
    taus: list[float] = []
    for frame_token in sorted(left_by_frame):
        left = left_by_frame[frame_token]["response_final"]
        right = right_by_frame[frame_token]["response_final"]
        left_status = {item["trajectory_id"]: item["status"] for item in left["verdicts"]}
        right_status = {item["trajectory_id"]: item["status"] for item in right["verdicts"]}
        if set(left_status) != set(right_status):
            raise ValueError(f"candidate mismatch on frame {frame_token}")
        status_matches.extend(left_status[item] == right_status[item] for item in left_status)
        chosen_matches.append(left.get("chosen_trajectory_id") == right.get("chosen_trajectory_id"))
        left_ranking = left["preference_ranking"]
        right_ranking = right["preference_ranking"]
        ranking_matches.append(left_ranking == right_ranking)
        tau = kendall_tau(left_ranking, right_ranking)
        if tau is not None:
            taus.append(tau)
    return {
        "status_agreement": _mean(float(value) for value in status_matches),
        "chosen_agreement": _mean(float(value) for value in chosen_matches),
        "exact_ranking_agreement": _mean(float(value) for value in ranking_matches),
        "mean_kendall_tau": _mean(taus),
        "n_decisions": len(status_matches),
        "n_scenes": len(chosen_matches),
    }


def _paired_differences(
    left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]], metric: str
) -> list[float]:
    right_by_frame = {row["frame_token"]: row for row in right_rows}
    return [
        float(row[metric]) - float(right_by_frame[row["frame_token"]][metric])
        for row in left_rows
    ]


def compute_internal_comparison(
    records_by_condition: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    validation = validate_pairing(records_by_condition)
    conditions = {
        condition: summarize_condition(records_by_condition[condition])
        for condition in CONDITION_NAMES
    }
    pairwise: dict[str, Any] = {}
    for left_index, left in enumerate(CONDITION_NAMES):
        for right in CONDITION_NAMES[left_index + 1 :]:
            pairwise[f"{left}__vs__{right}"] = pairwise_agreement(
                records_by_condition[left], records_by_condition[right]
            )

    contrasts: dict[str, Any] = {}
    for contrast, (left, right) in CONTRASTS.items():
        contrasts[contrast] = {}
        for metric in SCENE_METRICS:
            differences = _paired_differences(
                conditions[left]["scenario_rows"], conditions[right]["scenario_rows"], metric
            )
            contrasts[contrast][metric] = {
                "left": left,
                "right": right,
                "mean_difference": _mean(differences),
                "bootstrap_95_ci": _bootstrap_mean_ci(differences),
            }

    interaction: dict[str, Any] = {}
    for metric in SCENE_METRICS:
        full_minus_no_validation = _paired_differences(
            conditions["full"]["scenario_rows"],
            conditions["no_validation"]["scenario_rows"],
            metric,
        )
        no_rag_minus_baseline = _paired_differences(
            conditions["no_rag"]["scenario_rows"],
            conditions["baseline"]["scenario_rows"],
            metric,
        )
        differences = [
            left - right
            for left, right in zip(full_minus_no_validation, no_rag_minus_baseline, strict=True)
        ]
        interaction[metric] = {
            "mean_difference_in_differences": _mean(differences),
            "bootstrap_95_ci": _bootstrap_mean_ci(differences),
        }

    return {
        "analysis": "gold_independent_internal_2x2",
        "protocol_version": "ablation_2x2_v1.0",
        "validation": validation,
        "conditions": conditions,
        "pairwise_agreement": pairwise,
        "factorial_contrasts": contrasts,
        "validation_x_rag_interaction": interaction,
        "deferred_metrics": [
            "verdict agreement with human gold",
            "preference-pair accuracy against human gold",
            "chosen top-1 accuracy against human gold",
            "ranking Kendall tau against human gold",
            "human-rated citation accuracy, applicability, and full hallucination rate",
        ],
    }


def _write_condition_csv(metrics: dict[str, Any], path: Path) -> None:
    fields = [
        "condition",
        "n_scenes",
        "n_decisions",
        "veto_rate",
        "uncertain_rate",
        "chosen_available_rate",
        "citations_final",
        "machine_citation_validity_first",
        "machine_citation_validity_final",
        "citations_per_decision",
        "no_citation_decision_rate",
        "tokens_total",
        "tokens_mean_per_scene",
        "duration_mean_ms",
        "duration_p50_ms",
        "duration_p95_ms",
        "citation_retries",
        "schema_retries",
        "citation_check_violations_first",
        "citation_check_violations_final",
        "enforcement_exhausted",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition in CONDITION_NAMES:
            row = {"condition": condition, **metrics["conditions"][condition]}
            writer.writerow({field: row[field] for field in fields})


def write_outputs(metrics: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable = json.loads(json.dumps(metrics))
    for condition in CONDITION_NAMES:
        serializable["conditions"][condition].pop("scenario_rows")
    (output_dir / "internal_metrics.json").write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_condition_csv(metrics, output_dir / "internal_condition_summary.csv")
    scene_rows = [
        {"condition": condition, **row}
        for condition in CONDITION_NAMES
        for row in metrics["conditions"][condition]["scenario_rows"]
    ]
    with (output_dir / "internal_scene_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scene_rows[0]))
        writer.writeheader()
        writer.writerows(scene_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.run_dir / "internal_comparison"
    metrics = compute_internal_comparison(load_records(args.run_dir))
    write_outputs(metrics, output_dir)
    print(output_dir / "internal_metrics.json")


if __name__ == "__main__":
    main()
