#!/usr/bin/env python3
"""Audit the Gate 3 20-scene smoke behavior gates without making LLM calls."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.layer1.facade import parse_scene
from src.layer1.leakage_guard import LeakageGuard as Layer1LeakageGuard
from src.layer1.models import RawScene
from src.layer3.leakage_guard import LeakageGuard as Layer3LeakageGuard

BORDERLINE_REASONS = {
    "gap_borderline",
    "pedestrian_gap_borderline",
    "borderline_or_phase_horizon_unknown",
}
DIGEST_FIELDS = (
    "signal_phase",
    "phase_source",
    "ego_turn_intent",
    "governing_stop_line_signed_m",
    "pedestrian_in_forward_crosswalk",
    "pedestrian_moving",
    "oncoming_vehicle_moving",
    "location_is_intersection",
)


def _load_jsonl(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest: dict[str, Any] | None = None
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("type") == "manifest" and manifest is None:
            manifest = item
        elif item.get("type") == "record" and isinstance(item.get("data"), dict):
            records.append(item["data"])
    if manifest is None:
        raise ValueError(f"{path} has no manifest")
    return manifest, records


def _verdict_by_id(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        item["trajectory_id"]: item["veto"]
        for item in record.get("final_label", {}).get("verdicts", [])
    }


def _candidate_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    context = record["layer1"]["context"]
    features_by_id = dict(
        zip(
            context["candidate_trajectory_ids"],
            context["trajectory_features"],
            strict=True,
        )
    )
    verdicts = _verdict_by_id(record)
    return [
        {
            "record": record,
            "scenario_type": context["scenario_type"],
            "benchmark": benchmark,
            "feature": features_by_id[benchmark["anonymous_id"]],
            "actual": verdicts.get(benchmark["anonymous_id"]),
        }
        for benchmark in record["layer1"]["benchmark_labels_debug_only"]["labels"]
    ]


def _expected_digest(scene_facts: dict[str, Any]) -> dict[str, Any]:
    return {
        "signal_phase": "red" if scene_facts["has_red_light"] else "unknown",
        "phase_source": scene_facts["traffic_light_status_source"],
        "ego_turn_intent": scene_facts["ego_turn_intent"],
        "governing_stop_line_signed_m": scene_facts[
            "governing_stop_line_signed_m"
        ],
        "pedestrian_in_forward_crosswalk": scene_facts[
            "pedestrian_in_forward_crosswalk"
        ],
        "pedestrian_moving": scene_facts["pedestrian_moving"],
        "oncoming_vehicle_moving": scene_facts["oncoming_vehicle_moving"],
        "location_is_intersection": scene_facts["location_is_intersection"],
    }


def _fresh_parse_checks(
    records: list[dict[str, Any]],
    input_path: Path,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    raw_records = json.loads(input_path.read_text(encoding="utf-8"))
    raw_by_token = {item["frame_token"]: item for item in raw_records}
    digest_mismatches: list[dict[str, Any]] = []
    leakage_hits: list[dict[str, Any]] = []
    field_checks = 0

    for record in records:
        token = record["raw_scene"]["frame_token"]
        bundle = parse_scene(RawScene.model_validate(raw_by_token[token]), seed=seed)
        logged_facts = record["layer1"]["context"]["scene_facts"]
        expected = _expected_digest(logged_facts)
        actual = bundle.judge_input.scene_facts_digest.model_dump(mode="json")
        for field in DIGEST_FIELDS:
            field_checks += 1
            if expected[field] != actual[field]:
                digest_mismatches.append({
                    "frame_token": token,
                    "field": field,
                    "logged_projection": expected[field],
                    "judge_digest": actual[field],
                })

        fragments = [
            bundle.judge_input.narrative,
            *(
                text
                for feature in bundle.judge_input.trajectory_features
                for text in (
                    feature.natural_language_summary,
                    feature.conflict_zone_behavior,
                )
            ),
        ]
        for fragment_index, fragment in enumerate(fragments):
            hits = sorted(set(
                Layer1LeakageGuard.find_banned_terms(fragment)
                + Layer3LeakageGuard.find_banned_terms(fragment)
            ))
            if hits:
                leakage_hits.append({
                    "frame_token": token,
                    "fragment_index": fragment_index,
                    "terms": hits,
                })
    return digest_mismatches, leakage_hits, field_checks


def build_report(log_path: Path, input_path: Path, seed: int) -> dict[str, Any]:
    manifest, records = _load_jsonl(log_path)
    unique_tokens = {
        record.get("raw_scene", {}).get("frame_token")
        for record in records
    }
    complete_records = [
        record
        for record in records
        if record.get("ok") and not record.get("skipped_by_admission")
    ]
    run_complete = (
        len(records) == 20
        and len(unique_tokens) == 20
        and len(complete_records) == 20
    )
    candidates = [row for record in complete_records for row in _candidate_rows(record)]

    stop_profiles = [
        row
        for row in candidates
        if row["benchmark"]["variant_type"] == "conservative"
        and row["feature"]["full_stop"] is True
        and float(row["feature"]["stop_duration_s"]) >= 1.0
    ]
    stop_statuses = Counter(
        row["actual"]["status"] if row["actual"] else "missing"
        for row in stop_profiles
    )

    pedestrian = [row for row in candidates if row["scenario_type"] == "pedestrian"]
    pedestrian_statuses = Counter(
        row["actual"]["status"] if row["actual"] else "missing"
        for row in pedestrian
    )
    pedestrian_total = sum(pedestrian_statuses.values())
    pedestrian_uncertain_rate = (
        pedestrian_statuses["uncertain"] / pedestrian_total
        if pedestrian_total
        else None
    )

    nondegenerate_illegal = [
        row
        for row in candidates
        if row["benchmark"]["variant_type"] == "illegal"
        and row["benchmark"]["is_degenerate"] is False
    ]
    illegal_statuses = Counter(
        row["actual"]["status"] if row["actual"] else "missing"
        for row in nondegenerate_illegal
    )
    illegal_expected_actual = {
        expected: dict(sorted(Counter(
            row["actual"]["status"] if row["actual"] else "missing"
            for row in nondegenerate_illegal
            if row["benchmark"]["expected_verdict"] == expected
        ).items()))
        for expected in sorted({
            row["benchmark"]["expected_verdict"]
            for row in nondegenerate_illegal
        })
    }
    expected_vetoed_illegal = [
        row
        for row in nondegenerate_illegal
        if row["benchmark"]["expected_verdict"] == "vetoed"
    ]
    expected_vetoed_statuses = Counter(
        row["actual"]["status"] if row["actual"] else "missing"
        for row in expected_vetoed_illegal
    )
    actual_cleared_illegal = [
        row
        for row in nondegenerate_illegal
        if row["actual"] and row["actual"]["status"] == "cleared"
    ]

    borderline = [
        row
        for row in candidates
        if row["benchmark"]["expected_verdict_reason"] in BORDERLINE_REASONS
    ]
    borderline_statuses = Counter(
        row["actual"]["status"] if row["actual"] else "missing"
        for row in borderline
    )
    borderline_sources = Counter(
        row["actual"]["decided_by"] if row["actual"] else "missing"
        for row in borderline
    )

    digest_mismatches, leakage_hits, field_checks = _fresh_parse_checks(
        complete_records,
        input_path,
        seed,
    )
    gates = {
        "conservative_stop_profiles_zero_vetoed": {
            "profile_count": len(stop_profiles),
            "status_distribution": dict(sorted(stop_statuses.items())),
            "passed": bool(stop_profiles) and stop_statuses["vetoed"] == 0,
        },
        "pedestrian_uncertain_rate_below_40_percent": {
            "candidate_count": pedestrian_total,
            "uncertain_count": pedestrian_statuses["uncertain"],
            "uncertain_rate": pedestrian_uncertain_rate,
            "threshold": 0.40,
            "passed": (
                pedestrian_uncertain_rate is not None
                and pedestrian_uncertain_rate < 0.40
            ),
        },
        "nondegenerate_illegal_vetoed_gt_cleared": {
            "candidate_count": len(nondegenerate_illegal),
            "status_distribution": dict(sorted(illegal_statuses.items())),
            "passed": (
                bool(nondegenerate_illegal)
                and illegal_statuses["vetoed"] > illegal_statuses["cleared"]
            ),
        },
        "leakage_scan_zero_hits": {
            "hit_count": len(leakage_hits),
            "hits": leakage_hits,
            "passed": not leakage_hits,
        },
        "digest_fieldwise_identical": {
            "field_checks": field_checks,
            "mismatch_count": len(digest_mismatches),
            "mismatches": digest_mismatches,
            "passed": not digest_mismatches and field_checks == 20 * len(DIGEST_FIELDS),
        },
    }
    return {
        "log": str(log_path),
        "input": str(input_path),
        "manifest_model": manifest.get("model"),
        "manifest_llm_settings": manifest.get("llm_settings"),
        "run_complete": run_complete,
        "record_count": len(records),
        "behavior_gates": gates,
        "all_behavior_gates_passed": run_complete and all(
            gate["passed"] for gate in gates.values()
        ),
        "borderline_observation_not_a_gate": {
            "candidate_count": len(borderline),
            "verdict_distribution": dict(sorted(borderline_statuses.items())),
            "decision_source_distribution": dict(sorted(borderline_sources.items())),
        },
        "illegal_benchmark_join_observation": {
            "nondegenerate_candidate_count": len(nondegenerate_illegal),
            "expected_actual_distribution": illegal_expected_actual,
            "expected_vetoed_candidate_count": len(expected_vetoed_illegal),
            "expected_vetoed_actual_distribution": dict(
                sorted(expected_vetoed_statuses.items())
            ),
            "expected_vetoed_all_vetoed": (
                bool(expected_vetoed_illegal)
                and expected_vetoed_statuses == {"vetoed": len(expected_vetoed_illegal)}
            ),
            "actual_cleared_rows": [
                {
                    "record_index": row["record"]["index"],
                    "frame_token": row["record"]["raw_scene"]["frame_token"],
                    "scenario_type": row["scenario_type"],
                    "trajectory_id": row["benchmark"]["anonymous_id"],
                    "benchmark_expected": row["benchmark"]["expected_verdict"],
                    "benchmark_reason": row["benchmark"][
                        "expected_verdict_reason"
                    ],
                    "benchmark_exclude_reason": row["benchmark"]["exclude_reason"],
                    "actual": row["actual"]["status"],
                    "decided_by": row["actual"]["decided_by"],
                }
                for row in actual_cleared_illegal
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report = build_report(args.log, args.input, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "run_complete": report["run_complete"],
        "all_behavior_gates_passed": report["all_behavior_gates_passed"],
        "behavior_gates": {
            name: gate["passed"]
            for name, gate in report["behavior_gates"].items()
        },
        "borderline": report["borderline_observation_not_a_gate"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
