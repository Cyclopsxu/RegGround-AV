"""Paired metrics and report generation for the 2×2 ablation."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from experiments.ablation_2x2.conditions import CONDITIONS

STATUSES = ("cleared", "vetoed", "uncertain")


def cohen_kappa(expected: list[str], predicted: list[str]) -> float | None:
    if len(expected) != len(predicted):
        raise ValueError("kappa inputs must have equal length")
    if not expected:
        return None
    observed = sum(left == right for left, right in zip(expected, predicted, strict=True)) / len(
        expected
    )
    left_counts = Counter(expected)
    right_counts = Counter(predicted)
    chance = sum(left_counts[item] * right_counts[item] for item in STATUSES) / len(expected) ** 2
    if chance == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - chance) / (1.0 - chance)


def kendall_tau(expected: list[str], predicted: list[str]) -> float | None:
    common = [item for item in expected if item in set(predicted)]
    if len(common) < 2:
        return None
    expected_pos = {item: index for index, item in enumerate(expected)}
    predicted_pos = {item: index for index, item in enumerate(predicted)}
    concordant = discordant = 0
    for index, left in enumerate(common):
        for right in common[index + 1 :]:
            product = (expected_pos[left] - expected_pos[right]) * (
                predicted_pos[left] - predicted_pos[right]
            )
            concordant += int(product > 0)
            discordant += int(product < 0)
    total = concordant + discordant
    return (concordant - discordant) / total if total else None


def kendall_tau_b(expected_tiers: list[list[str]], predicted: list[str]) -> float | None:
    """Kendall tau-b for human rankings that allow ties; predictions are strict."""
    expected_pos = {
        item: tier_index for tier_index, tier in enumerate(expected_tiers) for item in tier
    }
    predicted_pos = {item: index for index, item in enumerate(predicted)}
    common = [item for tier in expected_tiers for item in tier if item in predicted_pos]
    if len(common) < 2:
        return None
    concordant = discordant = expected_ties = 0
    for index, left in enumerate(common):
        for right in common[index + 1 :]:
            expected_delta = expected_pos[left] - expected_pos[right]
            if expected_delta == 0:
                expected_ties += 1
                continue
            predicted_delta = predicted_pos[left] - predicted_pos[right]
            concordant += int(expected_delta * predicted_delta > 0)
            discordant += int(expected_delta * predicted_delta < 0)
    comparable = concordant + discordant
    denominator = math.sqrt(comparable * (comparable + expected_ties))
    return (concordant - discordant) / denominator if denominator else None


def _mean(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _citation_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    first = [event for record in records for event in record["citations_first"]]
    final = [event for record in records for event in record["citations_final"]]

    def validity(events: list[dict[str, Any]]) -> float | None:
        return _mean(float(event["valid"]) for event in events)

    rated = [event for event in final if event.get("accuracy") is not None]
    invalid = sum(not event["valid"] for event in final)
    inaccurate = sum(event["valid"] and event.get("accuracy") is False for event in final)
    decisions = sum(len(record["response_final"]["verdicts"]) for record in records)
    no_citation = sum(
        count == 0 for record in records for count in record["decision_citation_counts"]
    )
    return {
        "citation_validity_first": validity(first),
        "citation_validity_final": validity(final),
        "citation_accuracy": _mean(float(event["accuracy"]) for event in rated),
        "hallucination_rate": (invalid + inaccurate) / len(final) if final else None,
        "total_citations": len(final),
        "citations_per_decision": len(final) / decisions if decisions else None,
        "no_citation_decision_rate": no_citation / decisions if decisions else None,
    }


def _condition_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    expected: list[str] = []
    predicted: list[str] = []
    chosen_hits: list[float] = []
    pair_hits: list[float] = []
    taus: list[float | None] = []
    uncertain = total_decisions = 0
    for record in records:
        gold = record["gold"]
        gold_verdicts = {item["trajectory_id"]: item["status"] for item in gold["verdicts"]}
        for verdict in record["response_final"]["verdicts"]:
            trajectory_id = verdict["trajectory_id"]
            if trajectory_id in gold_verdicts:
                expected.append(gold_verdicts[trajectory_id])
                predicted.append(verdict["status"])
                uncertain += int(verdict["status"] == "uncertain")
                total_decisions += 1
        if "preference_ranking" in gold:
            chosen_hits.append(
                float(
                    record["response_final"].get("chosen_trajectory_id")
                    == gold.get("chosen_trajectory_id")
                )
            )
            gold_pairs = {
                (item["preferred_id"], item["dispreferred_id"])
                for item in gold.get("preference_pairs", [])
            }
            predicted_pairs = {
                (item["preferred_id"], item["dispreferred_id"])
                for item in record["response_final"].get("preference_pairs", [])
            }
            pair_hits.append(
                len(gold_pairs & predicted_pairs) / len(gold_pairs) if gold_pairs else 1.0
            )
            if gold.get("preference_tiers"):
                taus.append(
                    kendall_tau_b(
                        gold["preference_tiers"],
                        record["response_final"]["preference_ranking"],
                    )
                )
            else:
                taus.append(
                    kendall_tau(
                        gold["preference_ranking"],
                        record["response_final"]["preference_ranking"],
                    )
                )

    metrics = {
        "n_scenarios": len(records),
        "n_decisions": total_decisions,
        "verdict_kappa": cohen_kappa(expected, predicted),
        "preference_pair_accuracy": _mean(pair_hits),
        "chosen_top1_accuracy": _mean(chosen_hits),
        "kendall_tau": _mean(taus),
        "abstention_rate": uncertain / total_decisions if total_decisions else None,
        "tokens": sum(int(record.get("tokens", 0)) for record in records),
        "duration_ms": sum(int(record.get("duration_ms", 0)) for record in records),
        "retries": sum(int(record.get("retry_count", 0)) for record in records),
    }
    metrics.update(_citation_metrics(records))
    return metrics


def compute_metrics(
    records_by_condition: dict[str, list[dict[str, Any]]], *, dry_run: bool
) -> dict[str, Any]:
    names = [condition.name for condition in CONDITIONS]
    if set(records_by_condition) != set(names):
        raise ValueError("metrics require all four conditions")
    frame_sets = {
        name: {record["frame_token"] for record in records}
        for name, records in records_by_condition.items()
    }
    if len({frozenset(frames) for frames in frame_sets.values()}) != 1:
        raise ValueError("condition records are not paired on the same frames")

    strata: dict[str, dict[str, Any]] = {}
    all_records = next(iter(records_by_condition.values()))
    keys = sorted({f"{r['scenario_type']} × {r['difficulty']}" for r in all_records})
    for key in keys:
        strata[key] = {
            name: _condition_metrics(
                [
                    r
                    for r in records_by_condition[name]
                    if f"{r['scenario_type']} × {r['difficulty']}" == key
                ]
            )
            for name in names
        }

    return {
        "protocol_version": "ablation_2x2_v1.0",
        "dry_run": dry_run,
        "conditions": {name: _condition_metrics(records_by_condition[name]) for name in names},
        "strata": strata,
        "caveats": [
            "Small-n caveat: the planned 50–100-scene subset yields sparse stratified cells.",
            (
                "Single-run caveat: no repeated judging is used; decision variance is not "
                "estimated (7dad lesson)."
            ),
            (
                "no_rag enforce validates existence in the audited full catalog, not "
                "scenario applicability."
            ),
        ],
    }


def comparison_report(
    metrics: dict[str, Any], records_by_condition: dict[str, list[dict[str, Any]]]
) -> str:
    names = [condition.name for condition in CONDITIONS]
    lines = [
        "# 2×2 消融实验对照报告",
        "",
        "**DRY_RUN — 非正式实验结果**" if metrics["dry_run"] else "**正式实验结果**",
        "",
        "四条件均关闭规则引擎；本实验只测 LLM 环节，生产系统仍包含规则引擎。",
        "no_rag 的 enforce 校验只判断条款是否存在于全量已审计法规库，不判断场景适用性。",
        "",
        "## 汇总指标",
        "",
        (
            "| condition | kappa | pair acc. | top-1 | tau | valid(first/final) | "
            "hallucination | density | abstention |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def show(value: Any) -> str:
        return "—" if value is None else f"{float(value):.3f}"

    for name in names:
        item = metrics["conditions"][name]
        lines.append(
            f"| {name} | {show(item['verdict_kappa'])} | "
            f"{show(item['preference_pair_accuracy'])} | "
            f"{show(item['chosen_top1_accuracy'])} | {show(item['kendall_tau'])} | "
            f"{show(item['citation_validity_first'])}/{show(item['citation_validity_final'])} | "
            f"{show(item['hallucination_rate'])} | {show(item['citations_per_decision'])} | "
            f"{show(item['abstention_rate'])} |"
        )
    lines.extend(["", "## 配对逐场景明细", ""])
    lines.append("| frame_token | " + " | ".join(names) + " |")
    lines.append("|---|" + "---|" * len(names))
    by_name = {
        name: {record["frame_token"]: record for record in records_by_condition[name]}
        for name in names
    }
    for frame in sorted(by_name[names[0]]):
        cells = []
        for name in names:
            record = by_name[name][frame]
            verdicts = ", ".join(
                f"{item['trajectory_id']}={item['status']}"
                for item in record["response_final"]["verdicts"]
            )
            cells.append(
                f"{verdicts}; chosen={record['response_final'].get('chosen_trajectory_id')}"
            )
        lines.append(f"| {frame} | " + " | ".join(cells) + " |")
    lines.extend(["", "## 限制", ""])
    lines.extend(f"- {caveat}" for caveat in metrics["caveats"])
    return "\n".join(lines) + "\n"


def write_metrics(
    output_dir: Path,
    records_by_condition: dict[str, list[dict[str, Any]]],
    *,
    dry_run: bool,
) -> dict[str, Any]:
    metrics = compute_metrics(records_by_condition, dry_run=dry_run)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "comparison_report.md").write_text(
        comparison_report(metrics, records_by_condition), encoding="utf-8"
    )
    return metrics


__all__ = [
    "cohen_kappa",
    "comparison_report",
    "compute_metrics",
    "kendall_tau",
    "kendall_tau_b",
    "write_metrics",
]
