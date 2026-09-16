#!/usr/bin/env python3
"""Run the RegGround-AV log health checks from RegGround_日志健康检查Prompt.md."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from math import comb
from pathlib import Path
from typing import Any


ALLOWED_PARTIAL_REASONS = {
    "uncertain_verdict",
    "low_confidence_pair",
    "citation_repair",
    "stage_degraded",
}
ALLOWED_STATUSES = {"cleared", "vetoed", "uncertain"}
ALLOWED_DECIDED_BY = {"rule_engine", "llm", "fallback"}
LEAK_MARKERS = (
    "ground_truth",
    "variant_type",
    "expected_verdict",
    "benchmark",
    "hard_case",
    "debug_only",
)
KNOWN_MACHINE_VETO_REASONS = {
    "crossed_governing_line_during_valid_red_without_full_stop",
    "pedestrian_zone_time_overlap_without_prior_stop",
}


def short(value: Any, limit: int = 50) -> str:
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def add(findings: list[dict[str, str]], check: str, severity: str, detail: str) -> None:
    findings.append(
        {
            "check": check,
            "severity": severity,
            "detail": short(detail),
        }
    )


def candidate_ids(data: dict[str, Any]) -> list[str]:
    return (
        data.get("layer1", {})
        .get("context", {})
        .get("candidate_trajectory_ids", [])
    )


def verdict_map(final_label: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for item in final_label.get("verdicts", []) or []:
        trajectory_id = item.get("trajectory_id")
        result.setdefault(trajectory_id, []).append(item)
    return result


def compliance_direction(text: str, status: str) -> bool:
    """Return True only for a high-confidence S9 direction contradiction."""
    if status == "cleared":
        has_violation = bool(
            re.search(r"违反\s*\[?R-|违反法规|违规|被否决", text)
        )
        has_negation = bool(
            re.search(
                r"未违反|不违反|没有违反|不构成.*违反|未检测到违规|无法.*违反|无法确定.*违反|不涉及.*义务",
                text,
            )
        )
        return has_violation and not has_negation
    if status == "vetoed":
        if text in KNOWN_MACHINE_VETO_REASONS:
            return False
        has_violation = bool(
            re.search(r"违反|违规|被否决|未停车|未让行|不符合", text)
        )
        has_explicit_rule_violation = bool(re.search(r"违反\s*\[?R-", text))
        has_compliance = bool(
            re.search(r"未违反|不违反|符合.*要求|已停车让行|通过硬性规则", text)
        )
        return not has_violation or (has_compliance and not has_explicit_rule_violation)
    return False


def generated_texts(final_label: dict[str, Any]) -> list[tuple[str, str]]:
    texts: list[tuple[str, str]] = []
    for item in final_label.get("verdicts", []) or []:
        reason = item.get("veto", {}).get("reason")
        if isinstance(reason, str):
            texts.append((f"verdicts[{item.get('trajectory_id')}].reason", reason))
    for i, item in enumerate(final_label.get("preference_pairs", []) or []):
        reasoning = item.get("reasoning")
        if isinstance(reasoning, str):
            texts.append((f"preference_pairs[{i}].reasoning", reasoning))
    for i, item in enumerate(final_label.get("reasoning_chain", []) or []):
        content = item.get("content")
        if isinstance(content, str):
            texts.append((f"reasoning_chain[{i}].content", content))
    summary = final_label.get("natural_language_summary")
    if isinstance(summary, str):
        texts.append(("natural_language_summary", summary))
    return texts


def check_record(entry: dict[str, Any]) -> dict[str, Any]:
    data = entry.get("data") or {}
    final_label = data.get("final_label") or {}
    findings: list[dict[str, str]] = []
    index = data.get("index")

    # S1: run status.
    if data.get("ok") is not True:
        add(findings, "S1", "FAIL", f"record.ok={data.get('ok')!r}")
    if "error" in data or "error" in entry:
        add(findings, "S1", "FAIL", "存在顶层 error 字段")

    # S2: final status and partial reasons.
    status = final_label.get("status")
    reasons = final_label.get("partial_reasons")
    if status not in {"complete", "partial"}:
        add(findings, "S2", "FAIL", f"status={status!r}")
    elif not isinstance(reasons, list):
        add(findings, "S2", "FAIL", f"partial_reasons={reasons!r}")
    else:
        if status == "complete" and reasons:
            add(findings, "S2", "FAIL", f"status=complete, partial_reasons={reasons!r}")
        if status == "partial" and not reasons:
            add(findings, "S2", "FAIL", "status=partial, partial_reasons=[]")
        invalid_reasons = [reason for reason in reasons if reason not in ALLOWED_PARTIAL_REASONS]
        if invalid_reasons:
            add(findings, "S2", "FAIL", f"清单外 partial_reason={invalid_reasons[0]!r}")

    # S3/S4: selection and verdict shape.
    chosen = final_label.get("chosen_trajectory_id")
    selection = final_label.get("selection_outcome")
    items = final_label.get("verdicts")
    items = items if isinstance(items, list) else []
    by_id = verdict_map(final_label)
    if selection == "no_choosable_candidate" and chosen is not None:
        add(findings, "S3", "FAIL", f"selection=no_choosable_candidate, chosen={chosen!r}")
    if chosen is not None:
        chosen_items = by_id.get(chosen, [])
        chosen_status = chosen_items[0].get("veto", {}).get("status") if chosen_items else None
        if chosen_status in {"uncertain", "vetoed"}:
            add(findings, "S3", "FAIL-CRITICAL", f"chosen={chosen}, status={chosen_status}")
        elif chosen_status != "cleared":
            add(findings, "S3", "FAIL", f"chosen={chosen}, status={chosen_status!r}")

    candidates = candidate_ids(data)
    if not isinstance(candidates, list):
        candidates = []
        add(findings, "S4", "FAIL", "candidate_trajectory_ids 缺失或非数组")
    for trajectory_id in candidates:
        count = len(by_id.get(trajectory_id, []))
        if count != 1:
            add(findings, "S4", "FAIL", f"{trajectory_id} verdict_count={count}")
    extra_ids = [trajectory_id for trajectory_id in by_id if trajectory_id not in candidates]
    if extra_ids:
        add(findings, "S4", "FAIL", f"存在非候选 verdict={extra_ids[0]}")
    for item in items:
        veto = item.get("veto") or {}
        verdict_status = veto.get("status")
        decided_by = veto.get("decided_by")
        if verdict_status not in ALLOWED_STATUSES:
            add(findings, "S4", "FAIL", f"{item.get('trajectory_id')} status={verdict_status!r}")
        if decided_by not in ALLOWED_DECIDED_BY:
            add(findings, "S4", "FAIL", f"{item.get('trajectory_id')} decided_by={decided_by!r}")

    # S5: counts and pairwise arithmetic.
    counts = final_label.get("counts") or {}
    actual_counts = Counter(
        (item.get("veto") or {}).get("status") for item in items
    )
    for key in ("cleared", "vetoed", "uncertain"):
        expected = actual_counts.get(key, 0)
        actual = counts.get(key)
        if actual != expected:
            add(findings, "S5", "FAIL", f"counts.{key}={actual!r}, verdicts={expected}")
    cleared = actual_counts.get("cleared", 0)
    expected_pairs = comb(cleared, 2)
    actual_pairs = final_label.get("pairwise_comparisons_count")
    pairwise_diagnostic = next(
        (
            item
            for item in final_label.get("stage_diagnostics", []) or []
            if item.get("stage") == "pairwise_ranking"
        ),
        None,
    )
    has_pairwise_accounting = bool(
        pairwise_diagnostic is not None
        and "planned_pairwise_comparisons" in pairwise_diagnostic
        and "completed_pairwise_comparisons" in pairwise_diagnostic
    )
    if not has_pairwise_accounting:
        if actual_pairs != expected_pairs:
            add(
                findings,
                "S5",
                "FAIL",
                f"pairwise_comparisons_count={actual_pairs}, expected={expected_pairs}",
            )
    else:
        assert pairwise_diagnostic is not None
        planned = pairwise_diagnostic.get("planned_pairwise_comparisons")
        completed = pairwise_diagnostic.get("completed_pairwise_comparisons")
        if planned != expected_pairs:
            add(findings, "S5", "FAIL", f"pairwise planned={planned}, expected={expected_pairs}")
        if completed != actual_pairs:
            add(findings, "S5", "FAIL", f"pairwise completed={completed}, actual={actual_pairs}")
        if pairwise_diagnostic.get("succeeded") is True and actual_pairs != expected_pairs:
            add(
                findings,
                "S5",
                "FAIL",
                f"pairwise succeeded but completed={actual_pairs}/{expected_pairs}",
            )

    # S6: ranking layers and chosen position.
    ranking = final_label.get("preference_ranking")
    ranking = ranking if isinstance(ranking, list) else []
    status_by_id = {
        trajectory_id: (items_for_id[0].get("veto", {}).get("status") if items_for_id else None)
        for trajectory_id, items_for_id in by_id.items()
    }
    layer = {"cleared": 0, "uncertain": 1, "vetoed": 2}
    ranking_layers = [layer.get(status_by_id.get(trajectory_id), 99) for trajectory_id in ranking]
    if ranking_layers != sorted(ranking_layers):
        add(findings, "S6", "FAIL", f"ranking status layers={ranking_layers}")
    unknown_ranking = [trajectory_id for trajectory_id in ranking if trajectory_id not in status_by_id]
    if unknown_ranking:
        add(findings, "S6", "FAIL", f"ranking 含未知轨迹={unknown_ranking[0]}")
    if chosen is not None and (not ranking or ranking[0] != chosen):
        add(findings, "S6", "FAIL", f"chosen={chosen}, ranking[0]={ranking[0] if ranking else None}")

    # S7: degradation and deferred retry bookkeeping.
    if any((item.get("veto") or {}).get("decided_by") == "fallback" for item in items):
        if "stage_degraded" not in (reasons or []):
            add(findings, "S7", "FAIL", "存在 decided_by=fallback 但无 stage_degraded")
    for item in items:
        veto = item.get("veto") or {}
        if veto.get("completed_by") != "deferred_retry":
            continue
        attempts = veto.get("deferred_attempts")
        has_timestamp = bool(
            veto.get("attempted_at")
            or veto.get("attempt_timestamp")
            or veto.get("deferred_retry_at")
            or (
                isinstance(attempts, list)
                and any(
                    isinstance(attempt, dict)
                    and any(
                        key in attempt
                        for key in (
                            "timestamp",
                            "attempted_at",
                            "at",
                            "started_at",
                            "finished_at",
                        )
                    )
                    for attempt in attempts
                )
            )
        )
        if not has_timestamp:
            add(findings, "S7", "FAIL", f"{item.get('trajectory_id')} deferred_retry 无时间戳")

    # S8: citation and legal basis.
    citation_validity = final_label.get("citation_validity")
    if citation_validity != 1.0:
        add(findings, "S8", "WARN", f"citation_validity={citation_validity!r}")
    vetoed_items = [
        item for item in items if (item.get("veto") or {}).get("status") == "vetoed"
    ]
    if vetoed_items:
        legal_basis = final_label.get("legal_basis")
        if not legal_basis:
            add(findings, "S8", "FAIL", "存在 vetoed 但 legal_basis 为空")
        for item in vetoed_items:
            rule_ids = (item.get("veto") or {}).get("violated_rule_ids")
            invalid_rule = next(
                (rule_id for rule_id in (rule_ids or []) if not isinstance(rule_id, str) or not rule_id.startswith("R-")),
                None,
            )
            if invalid_rule is not None:
                add(findings, "S8", "FAIL", f"violated_rule_id={invalid_rule!r}")

    # S9: high-confidence semantic contradictions only.
    for item in items:
        veto = item.get("veto") or {}
        reason = veto.get("reason")
        if isinstance(reason, str) and compliance_direction(reason, veto.get("status")):
            add(
                findings,
                "S9",
                "FAIL-CRITICAL" if veto.get("status") == "cleared" else "FAIL",
                f"{item.get('trajectory_id')} reason 与 status={veto.get('status')}矛盾",
            )
    summary = final_label.get("natural_language_summary")
    if isinstance(summary, str) and chosen is not None and chosen not in summary:
        add(findings, "S9", "FAIL", f"chosen={chosen} 未出现在 natural_language_summary")

    # S10: generated text only; do not scan layer1.benchmark_labels_debug_only.
    for field, text in generated_texts(final_label):
        for marker in LEAK_MARKERS:
            if marker in text:
                excerpt = text.replace("\n", " ")
                add(findings, "S10", "FAIL-CRITICAL", f"{field} 命中 {marker}: {excerpt}")
                break

    # S11/S12: diagnostics.
    partial_reasons = reasons if isinstance(reasons, list) else []
    stage_explanations = {
        "context_building": {"stage_degraded"},
        "hard_filter": {"stage_degraded"},
        "pairwise_ranking": {"stage_degraded", "low_confidence_pair"},
        "label_generation": {"stage_degraded", "citation_repair"},
    }
    for diagnostic in final_label.get("stage_diagnostics", []) or []:
        stage = diagnostic.get("stage")
        if diagnostic.get("succeeded") is False:
            allowed = stage_explanations.get(stage, {"stage_degraded"})
            if not allowed.intersection(partial_reasons):
                add(findings, "S11", "FAIL", f"stage={stage} failed 但 partial_reasons={partial_reasons!r}")
        calls = diagnostic.get("llm_call_count", 0) or 0
        tokens = diagnostic.get("llm_tokens_used")
        if tokens is None:
            tokens = (diagnostic.get("llm_input_tokens", 0) or 0) + (diagnostic.get("llm_output_tokens", 0) or 0)
        if calls > 0 and tokens == 0:
            add(findings, "S12", "WARN", f"stage={stage}, calls={calls}, tokens={tokens}")

    has_fail = any(item["severity"].startswith("FAIL") for item in findings)
    verdict = "FAIL" if has_fail else "WARN" if findings else "PASS"
    return {"index": index, "verdict": verdict, "findings": findings}


def load_input(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    manifest: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            if entry.get("type") == "manifest":
                manifest = entry
            elif entry.get("type") == "record":
                records.append(entry)
            elif entry.get("type") == "summary":
                summary = entry
    return manifest, records, summary


def build_batch_report(
    manifest_entry: dict[str, Any],
    records: list[dict[str, Any]],
    summary_entry: dict[str, Any],
    verdicts: list[dict[str, Any]],
    expected_records: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = manifest_entry
    manifest_checks = {
        "git_commit": manifest.get("git_commit"),
        "input_sha256": manifest.get("input_sha256"),
        "eval_set.version": manifest.get("eval_set", {}).get("version"),
        "eval_set.sha256": manifest.get("eval_set", {}).get("sha256"),
        "classification_code_sha256": manifest.get("eval_set", {}).get("classification_code_sha256"),
        "predicate_version": manifest.get("predicate_version"),
        "compliance_predicates_sha256": manifest.get("compliance_predicates_sha256"),
        "benchmark_builder_sha256": manifest.get("benchmark_builder_sha256"),
        "rule_graph_sha256": manifest.get("rule_graph_sha256"),
    }
    missing_manifest = [key for key, value in manifest_checks.items() if not value]
    manifest_ok = not missing_manifest

    records_data = [entry.get("data") or {} for entry in records]
    record_count = len(records_data)
    ok_count = sum(data.get("ok") is True for data in records_data)
    error_count = sum(data.get("error") is not None for data in records_data)
    manifest_expected = manifest.get("eval_set", {}).get("entries")
    if expected_records is not None:
        expected_count = expected_records
    elif isinstance(manifest_expected, int) and manifest_expected >= 1:
        expected_count = manifest_expected
    else:
        raise ValueError(
            "expected record count is missing; pass --expected-records or use an "
            "eval-set manifest with entries"
        )
    totals_ok = (
        record_count == expected_count
        and ok_count == expected_count
        and error_count == 0
    )

    source_summary = summary_entry.get("summary") or {}
    degraded = source_summary.get("degraded_records")
    deferred_recovered = (
        source_summary.get("llm_telemetry", {})
        .get("total", {})
        .get("recovered_count")
    )
    degraded_ok = degraded is not None and deferred_recovered is not None
    degraded_summary = (
        f"degraded_records={degraded!r}; deferred_recovered={deferred_recovered!r}"
        if degraded_ok
        else "degraded_records=missing; deferred_recovered=missing"
    )

    fail_details = []
    check_counts: dict[str, Counter[str]] = {}
    for verdict in verdicts:
        for finding in verdict["findings"]:
            check_counts.setdefault(finding["check"], Counter())[finding["severity"]] += 1
        if verdict["verdict"] == "FAIL":
            fail_details.append(
                {
                    "index": verdict["index"],
                    "checks": sorted({finding["check"] for finding in verdict["findings"] if finding["severity"].startswith("FAIL")}),
                }
            )

    statuses = source_summary.get("statuses") or Counter(
        (entry.get("data") or {}).get("final_label", {}).get("status") for entry in records
    )
    partial_count = statuses.get("partial", 0)
    partial_ratio = partial_count / record_count if record_count else 0.0
    warn_clusters = [
        {"check": check, "count": counts["WARN"]}
        for check, counts in sorted(check_counts.items())
        if counts["WARN"]
    ]
    if partial_ratio > 0.20:
        warn_clusters.append({"check": "B5", "count": partial_count})
    warn_clusters.sort(key=lambda item: item["check"])

    summary_output = {
        "manifest_ok": manifest_ok,
        "totals_ok": totals_ok,
        "degraded_summary": degraded_summary,
        "warn_clusters": warn_clusters,
        "fail_indices": [item["index"] for item in fail_details],
        "READY_FOR_ANALYSIS": not fail_details,
    }
    details = {
        "batch_verdict": summary_output,
        "batch_checks": {
            "B1_missing_fields": missing_manifest,
            "B2_actual": {
                "records": record_count,
                "ok": ok_count,
                "error": error_count,
                "expected": {
                    "records": expected_count,
                    "ok": expected_count,
                    "error": 0,
                },
            },
            "B3": {
                "degraded_records": degraded,
                "deferred_recovered": deferred_recovered,
                "fields_present": degraded_ok,
            },
            "B4": {
                "llm_429_count": summary_entry.get("llm_429_count"),
                "retry_total": source_summary.get("llm_telemetry", {}).get("total", {}).get("retries"),
            },
            "B5": {
                "statuses": dict(statuses),
                "partial_ratio": partial_ratio,
            },
            "B6_fail_details": fail_details,
        },
        "check_counts": {check: dict(counts) for check, counts in sorted(check_counts.items())},
    }
    return summary_output, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--verdicts", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument(
        "--expected-records",
        type=int,
        default=None,
        help="预期记录数；默认读取 manifest.eval_set.entries。",
    )
    args = parser.parse_args()
    if args.expected_records is not None and args.expected_records < 1:
        raise SystemExit("--expected-records must be >= 1")

    manifest, records, source_summary = load_input(args.input)
    verdicts = [check_record(entry) for entry in records]
    batch, details = build_batch_report(
        manifest,
        records,
        source_summary,
        verdicts,
        expected_records=args.expected_records,
    )

    args.verdicts.parent.mkdir(parents=True, exist_ok=True)
    with args.verdicts.open("w", encoding="utf-8") as handle:
        for verdict in verdicts:
            handle.write(json.dumps(verdict, ensure_ascii=False, separators=(",", ":")) + "\n")
    for path, payload in ((args.summary, batch), (args.details, details)):
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    print(json.dumps(batch, ensure_ascii=False, indent=2))
    print(json.dumps(details["check_counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
