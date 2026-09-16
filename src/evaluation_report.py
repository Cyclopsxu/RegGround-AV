"""RegGround-AV 分层评估口径与 JSONL 评分入口。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

BENCHMARK_DATA_VERSIONS = {
    "v1": "2026-07-11-v2",
    "v2": "2026-07-v2",
}


def evaluate_entries(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """按可评分性与判定来源汇总 benchmark 指标。"""
    evaluated = [
        entry
        for entry in entries
        if entry.get("ok") and not entry.get("skipped_by_admission")
    ]
    illegal_chosen = 0
    scorable = 0
    compliant_chosen = 0
    unscorable_reasons: Counter[str] = Counter()
    gt_chosen = 0
    gt_top2 = 0
    excluded_candidates = 0
    llm_evaluated = 0
    llm_correct = 0
    rule_engine_evaluated = 0
    fallback_evaluated = 0
    expected_vetoed = detected_vetoed = 0
    expected_cleared = false_vetoed = 0
    uncertain_count = candidate_count = 0
    pair_consistent = pair_abstain = pair_total = 0
    scorable_pair_scenes = 0
    structurally_scorable_scenes = 0
    structural_pair_scenes = 0
    structurally_unscorable_reasons: Counter[str] = Counter()
    chosen_acceptable = chosen_none = 0
    strata: dict[tuple[str, str], Counter[str]] = {}

    for entry in evaluated:
        labels = entry["layer1"]["benchmark_labels_debug_only"]["labels"]
        scenario = entry.get("layer1", {}).get("context", {}).get(
            "scenario_type", "unknown"
        )
        label_by_id = {item["anonymous_id"]: item for item in labels}
        final_label = entry["final_label"]
        chosen_id = final_label.get("chosen_trajectory_id")
        chosen_label = label_by_id.get(chosen_id)

        if chosen_label and chosen_label["expected_verdict"] == "vetoed":
            illegal_chosen += 1

        reason = _unscorable_reason(chosen_label)
        if reason is not None:
            unscorable_reasons[reason] += 1
        elif chosen_label is not None:
            scorable += 1
            compliant_chosen += int(chosen_label["expected_verdict"] == "cleared")
            gt_ids = {
                item["anonymous_id"]
                for item in labels
                if item["variant_type"] == "ground_truth"
            }
            gt_chosen += int(chosen_id in gt_ids)
            gt_top2 += int(
                bool(gt_ids.intersection(final_label.get("preference_ranking", [])[:2]))
            )

        verdict_by_id = {
            item["trajectory_id"]: item["veto"]
            for item in final_label.get("verdicts", [])
        }
        cleared_ids = [
            item["anonymous_id"] for item in labels if item["expected_verdict"] == "cleared"
        ]
        vetoed_ids = [
            item["anonymous_id"] for item in labels if item["expected_verdict"] == "vetoed"
        ]
        has_scorable_pair = bool(cleared_ids and vetoed_ids)
        scorable_pair_scenes += int(has_scorable_pair)
        structural_reason = _structural_unscorable_reason(entry)
        if structural_reason is None:
            structurally_scorable_scenes += 1
            structural_pair_scenes += int(has_scorable_pair)
        else:
            structurally_unscorable_reasons[structural_reason] += 1
        if chosen_id is None:
            chosen_none += 1
        elif chosen_id in cleared_ids:
            chosen_acceptable += 1
        for cleared_id in cleared_ids:
            for vetoed_id in vetoed_ids:
                pair_total += 1
                cleared_actual = verdict_by_id.get(cleared_id, {}).get("status")
                vetoed_actual = verdict_by_id.get(vetoed_id, {}).get("status")
                pair_label = label_by_id[vetoed_id]
                pair_key = (
                    str(scenario),
                    str(pair_label.get("difficulty", "unknown")),
                )
                pair_bucket = strata.setdefault(pair_key, Counter())
                pair_bucket["pairs"] += 1
                if "uncertain" in {cleared_actual, vetoed_actual}:
                    pair_abstain += 1
                    pair_bucket["pair_abstain"] += 1
                elif cleared_actual == "cleared" and vetoed_actual == "vetoed":
                    pair_consistent += 1
                    pair_bucket["pair_consistent"] += 1
        for benchmark in labels:
            expected = benchmark["expected_verdict"]
            if expected == "exclude":
                excluded_candidates += 1
                continue
            actual = verdict_by_id.get(benchmark["anonymous_id"])
            if actual is None:
                continue
            candidate_count += 1
            status = actual.get("status")
            uncertain_count += int(status == "uncertain")
            key = (str(scenario), str(benchmark.get("difficulty", "unknown")))
            bucket = strata.setdefault(key, Counter())
            bucket["candidates"] += 1
            bucket["uncertain"] += int(status == "uncertain")
            if expected == "vetoed":
                expected_vetoed += 1
                detected_vetoed += int(status == "vetoed")
                bucket["expected_vetoed"] += 1
                bucket["detected_vetoed"] += int(status == "vetoed")
            elif expected == "cleared":
                expected_cleared += 1
                false_vetoed += int(status == "vetoed")
                bucket["expected_cleared"] += 1
                bucket["false_vetoed"] += int(status == "vetoed")
            source = actual.get("decided_by")
            if source == "llm":
                llm_evaluated += 1
                llm_correct += int(actual.get("status") == expected)
                bucket["decided_by_llm"] += 1
            elif source == "rule_engine":
                rule_engine_evaluated += 1
                bucket["decided_by_rule_engine"] += 1
            else:
                fallback_evaluated += 1
                bucket["decided_by_fallback"] += 1

    total = len(evaluated)
    main_metrics = {
        "violation_detection_rate": (
            detected_vetoed / expected_vetoed if expected_vetoed else None
        ),
        "false_veto_rate": false_vetoed / expected_cleared if expected_cleared else None,
        "abstention_rate": uncertain_count / candidate_count if candidate_count else None,
        "pair_consistency": pair_consistent / pair_total if pair_total else None,
        "pair_abstain_count": pair_abstain,
        "pair_total": pair_total,
        "scorable_pair_scene_count": scorable_pair_scenes,
        "scorable_pair_scene_rate": (
            scorable_pair_scenes / total if total else None
        ),
        "structurally_scorable_scene_count": structurally_scorable_scenes,
        "structurally_unscorable_scene_count": sum(
            structurally_unscorable_reasons.values()
        ),
        "structurally_unscorable_reasons": dict(
            sorted(structurally_unscorable_reasons.items())
        ),
        "scorable_pair_scene_count_within_structural": structural_pair_scenes,
        "scorable_pair_scene_rate_within_structural": (
            structural_pair_scenes / structurally_scorable_scenes
            if structurally_scorable_scenes
            else None
        ),
        "chosen_acceptable_rate": (
            chosen_acceptable / total if total else None
        ),
        "no_choice_count": chosen_none,
    }
    return {
        "evaluated_scenes": total,
        "illegal_chosen": {
            "count": illegal_chosen,
            "denominator": total,
            "rate": illegal_chosen / total if total else 0.0,
        },
        "scorable": {
            "count": scorable,
            "denominator": total,
            "compliant_chosen": compliant_chosen,
        },
        "unscorable": {
            "count": sum(unscorable_reasons.values()),
            "denominator": total,
            "reasons": dict(sorted(unscorable_reasons.items())),
        },
        "gt_chosen_reference": {
            "count": gt_chosen,
            "denominator": scorable,
            "rate": gt_chosen / scorable if scorable else 0.0,
        },
        "gt_top2_reference": {
            "count": gt_top2,
            "denominator": scorable,
            "rate": gt_top2 / scorable if scorable else 0.0,
        },
        "verdict_layers": {
            "llm": {
                "evaluated": llm_evaluated,
                "correct": llm_correct,
                "benchmark_accuracy": (
                    llm_correct / llm_evaluated if llm_evaluated else None
                ),
                "accuracy_status": (
                    "evaluated" if llm_evaluated else "not_evaluable"
                ),
            },
            "rule_engine": {
                "evaluated": rule_engine_evaluated,
                "benchmark_accuracy_reported": False,
                "required_external_metric": "predicate_human_agreement",
            },
            "fallback": {"evaluated": fallback_evaluated},
        },
        "excluded_candidates": excluded_candidates,
        "main_metrics_v2": main_metrics,
        "by_scenario_difficulty": {
            f"{scenario}|{difficulty}": _stratum_metrics(counts)
            for (scenario, difficulty), counts in sorted(strata.items())
        },
    }


def _structural_unscorable_reason(entry: dict[str, Any]) -> str | None:
    context = entry.get("layer1", {}).get("context", {})
    scenario = context.get("scenario_type")
    facts = context.get("scene_facts", {})
    if scenario == "oncoming":
        return "oncoming_design_exclusion"
    if scenario == "red_light":
        signed = facts.get("governing_stop_line_signed_m")
        if isinstance(signed, int | float) and signed < 0:
            return "start_beyond_line"
    if scenario == "pedestrian" and (
        facts.get("pedestrian_in_forward_crosswalk") is not True
        or facts.get("pedestrian_moving") is not True
    ):
        return "pedestrian_trigger_missing_or_unknown"
    if facts.get("window_insufficient") is True:
        return "not_evaluable_short_window"
    if scenario == "pedestrian":
        features = context.get("trajectory_features", [])
        has_interaction = any(
            interaction.get("agent_kind") == "pedestrian"
            and interaction.get("min_time_gap_s") is not None
            for feature in features
            for interaction in feature.get("agent_interactions", [])
        )
        if not has_interaction:
            return "pedestrian_interaction_missing"
    return None


def _stratum_metrics(counts: Counter[str]) -> dict[str, Any]:
    return {
        "candidates": counts["candidates"],
        "violation_detection_rate": (
            counts["detected_vetoed"] / counts["expected_vetoed"]
            if counts["expected_vetoed"] else None
        ),
        "false_veto_rate": (
            counts["false_vetoed"] / counts["expected_cleared"]
            if counts["expected_cleared"] else None
        ),
        "abstention_rate": (
            counts["uncertain"] / counts["candidates"] if counts["candidates"] else None
        ),
        "pair_consistency": (
            counts["pair_consistent"] / counts["pairs"] if counts["pairs"] else None
        ),
        "pair_abstain_count": counts["pair_abstain"],
        "pair_total": counts["pairs"],
        "decided_by": {
            "rule_engine": counts["decided_by_rule_engine"],
            "llm": counts["decided_by_llm"],
            "fallback": counts["decided_by_fallback"],
        },
    }


def load_jsonl_entries(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 中的 record 数据。"""
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item.get("type") == "record" and isinstance(item.get("data"), dict):
            entries.append(item["data"])
    return entries


def load_versioned_jsonl_entries(
    path: Path,
    *,
    benchmark_version: str,
) -> list[dict[str, Any]]:
    """读取并验证单一 benchmark 版本，拒绝 v1/v2 混算。"""
    expected = BENCHMARK_DATA_VERSIONS[benchmark_version]
    items = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    observed: set[str] = set()
    entries: list[dict[str, Any]] = []
    for item in items:
        if item.get("type") == "manifest":
            version = item.get("benchmark_data_version")
            if isinstance(version, str):
                observed.add(version)
        if item.get("type") != "record" or not isinstance(item.get("data"), dict):
            continue
        entry = item["data"]
        entries.append(entry)
        version = (
            entry.get("layer1", {})
            .get("benchmark_labels_debug_only", {})
            .get("data_version")
        )
        if isinstance(version, str):
            observed.add(version)
    if not observed:
        raise ValueError("JSONL 未声明 benchmark_data_version，拒绝无版本评分")
    if observed != {expected}:
        raise ValueError(
            f"benchmark 版本不匹配：请求 {benchmark_version}={expected}，"
            f"文件包含 {sorted(observed)}"
        )
    return entries


def _unscorable_reason(chosen_label: dict[str, Any] | None) -> str | None:
    """返回 chosen 不可评分原因。"""
    if chosen_label is None:
        return "missing_chosen_or_label"
    if chosen_label["variant_type"] == "hard_case":
        return "hard_case_chosen"
    if chosen_label["expected_verdict"] == "exclude":
        return "exclude_chosen"
    return None


def main() -> None:
    """命令行评分入口。"""
    parser = argparse.ArgumentParser(description="按分层口径评分 RegGround-AV JSONL")
    parser.add_argument("jsonl", type=Path)
    parser.add_argument(
        "--benchmark-version",
        choices=tuple(BENCHMARK_DATA_VERSIONS),
        required=True,
        help="显式选择 v1 或 v2；版本不符或混合时拒绝评分",
    )
    parser.add_argument("--output", type=Path, help="可选：同时写出 JSON 报告")
    args = parser.parse_args()
    report = evaluate_entries(load_versioned_jsonl_entries(
        args.jsonl,
        benchmark_version=args.benchmark_version,
    ))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
