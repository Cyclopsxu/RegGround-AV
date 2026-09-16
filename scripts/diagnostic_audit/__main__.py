"""第一部分：诊断性审计 CLI。

输入只读；默认输出到 ``diagnostics/v1``。日志中的 Layer 1 benchmark
labels、Layer 2 telemetry 和 Layer 3 verdicts 通过
``(scene_id, frame_token, anonymous_id)`` 做完整性校验。
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL = ROOT / "logs/eval_set_v1_full_v3_5.jsonl"
DEFAULT_OUTPUT = ROOT / "diagnostics/v1.4"
DEFAULT_RAW = ROOT / "data/layer1/raw_scene_trainval.json"
DEFAULT_SPECIAL_SIDECAR = ROOT / "human_annotation_v1/private/hidden_sidecar.jsonl"
DISPOSITION_CODES = (
    "BENCH-FIX",
    "FACT-FIX",
    "FEAT-GAP",
    "JUDGE-MISS",
    "CORRECT-ABSTAIN",
    "SCOPE-CUT",
    "HOLD",
)
EXPECTED = {
    "scenes": 259,
    "unscorable": 246,
    "no_choosable": 120,
    "exclude_chosen": 95,
    "hard_case_chosen": 31,
    "uncertain": 907,
}
SAMPLE_SEED = 42
B3_SCENARIO_QUOTAS = {
    "pedestrian": 12,
    "oncoming": 10,
    "red_light": 5,
    "yellow_light": 2,
    "emergency": 1,
}
B6_SCENARIO_QUOTAS = {
    "oncoming": 6,
    "pedestrian": 5,
    "red_light": 4,
}
B7_SAMPLE_SIZE = 8
LLM_REASON_KEYWORDS = {
    "缺少": r"缺少",
    "未提供": r"未提供",
    "未明确": r"未明确",
    "未描述": r"未描述",
    "无法": r"无法",
    "不确定": r"不确定",
    "信息不足": r"信息不足",
    "关键信息": r"关键信息",
    "证据": r"证据",
    "场景描述": r"场景描述",
    "轨迹特征": r"轨迹特征",
}
FEAT_GAP_KEYWORDS = ("缺少", "未提供", "未明确", "未描述", "无法", "不确定", "信息不足", "关键信息")


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(*parts: Any) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class Candidate:
    scene_id: str
    frame_token: str
    scenario_type: str
    label: dict[str, Any]
    verdict: dict[str, Any]
    feature: dict[str, Any] | None

    @property
    def anonymous_id(self) -> str:
        return str(self.label["anonymous_id"])

    @property
    def status(self) -> str:
        return str(self.verdict.get("veto", {}).get("status", "missing"))

    @property
    def decided_by(self) -> str:
        return str(self.verdict.get("veto", {}).get("decided_by", "missing"))


@dataclass
class AuditData:
    records: list[dict[str, Any]]
    candidates: list[Candidate]
    by_frame: dict[tuple[str, str], dict[str, Any]]
    manifest: dict[str, Any]


def load_eval(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"冻结结果不存在: {path}")
    manifest: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} 不是合法 JSON") from exc
        if item.get("type") == "manifest":
            manifest = item
        elif item.get("type") == "record" and isinstance(item.get("data"), dict):
            records.append(item["data"])
    if not records:
        raise ValueError(f"{path} 没有 record 行")
    return manifest, records


def load_data(path: Path) -> AuditData:
    manifest, records = load_eval(path)
    by_frame: dict[tuple[str, str], dict[str, Any]] = {}
    candidates: list[Candidate] = []
    seen_join_keys: set[tuple[str, str, str]] = set()
    for record in records:
        raw = record.get("raw_scene", {})
        layer1 = record.get("layer1", {})
        context = layer1.get("context", {})
        benchmark = layer1.get("benchmark_labels_debug_only", {})
        label = record.get("final_label")
        if not isinstance(label, dict):
            raise ValueError(f"缺少 final_label: {raw.get('frame_token')}")
        scene_id = str(raw.get("scene_id") or context.get("scene_id") or "")
        frame_token = str(raw.get("frame_token") or context.get("frame_token") or "")
        if not scene_id or not frame_token:
            raise ValueError("记录缺少 scene_id 或 frame_token")
        frame_key = (scene_id, frame_token)
        if frame_key in by_frame:
            raise ValueError(f"存在重复场景帧: {frame_key}")
        by_frame[frame_key] = record
        labels = benchmark.get("labels")
        verdicts = label.get("verdicts")
        if not isinstance(labels, list) or not isinstance(verdicts, list):
            raise ValueError(f"benchmark labels/verdicts 缺失: {frame_key}")
        label_by_id = {str(item.get("anonymous_id")): item for item in labels}
        verdict_by_id = {str(item.get("trajectory_id")): item for item in verdicts}
        if len(label_by_id) != len(labels) or len(verdict_by_id) != len(verdicts):
            raise ValueError(f"候选匿名 ID 不唯一: {frame_key}")
        missing = sorted(set(verdict_by_id) - set(label_by_id))
        if missing:
            raise ValueError(f"join 缺口：eval 候选不在 benchmark 侧 {frame_key}: {missing}")
        missing_verdict = sorted(set(label_by_id) - set(verdict_by_id))
        if missing_verdict:
            raise ValueError(f"join 缺口：benchmark 候选不在 eval verdict 侧 {frame_key}: {missing_verdict}")
        candidate_ids = context.get("candidate_trajectory_ids", [])
        features = context.get("trajectory_features", [])
        feature_by_id = {
            str(candidate_id): feature
            for candidate_id, feature in zip(candidate_ids, features, strict=False)
        }
        if len(features) not in (0, len(candidate_ids)):
            raise ValueError(f"trajectory_features 与 candidate_trajectory_ids 不对齐: {frame_key}")
        scenario_type = str(context.get("scenario_type", "unknown"))
        for anonymous_id, benchmark_label in sorted(label_by_id.items()):
            join_key = (scene_id, frame_token, anonymous_id)
            if join_key in seen_join_keys:
                raise ValueError(f"重复 join 键: {join_key}")
            seen_join_keys.add(join_key)
            candidates.append(
                Candidate(
                    scene_id=scene_id,
                    frame_token=frame_token,
                    scenario_type=scenario_type,
                    label=benchmark_label,
                    verdict=verdict_by_id[anonymous_id],
                    feature=feature_by_id.get(anonymous_id),
                )
            )
    return AuditData(records, candidates, by_frame, manifest)


def scene_name(record: dict[str, Any]) -> str:
    return str(record.get("raw_scene", {}).get("scene_name") or record.get("raw_scene", {}).get("scene_id", ""))


def candidates_for(data: AuditData, record: dict[str, Any]) -> list[Candidate]:
    scene_id = str(record["raw_scene"]["scene_id"])
    frame_token = str(record["raw_scene"]["frame_token"])
    return [item for item in data.candidates if item.scene_id == scene_id and item.frame_token == frame_token]


def chosen_candidate(data: AuditData, record: dict[str, Any]) -> Candidate | None:
    chosen = record.get("final_label", {}).get("chosen_trajectory_id")
    return next((item for item in candidates_for(data, record) if item.anonymous_id == chosen), None)


def scene_bucket(data: AuditData, record: dict[str, Any]) -> str | None:
    final_label = record["final_label"]
    if final_label.get("selection_outcome") == "no_choosable_candidate":
        return "B1"
    chosen = chosen_candidate(data, record)
    if chosen and chosen.label.get("variant_type") == "hard_case":
        return "B4"
    if chosen and chosen.label.get("expected_verdict") == "exclude":
        return "B2"
    return None


def unscorable_reason(data: AuditData, record: dict[str, Any]) -> str | None:
    bucket = scene_bucket(data, record)
    return {
        "B1": "no-choosable",
        "B2": "exclude-chosen",
        "B4": "hard_case-chosen",
    }.get(bucket)


def assert_reconciliation(data: AuditData) -> None:
    scene_reasons = Counter(unscorable_reason(data, record) for record in data.records)
    scene_reasons.pop(None, None)
    uncertain = sum(item.status == "uncertain" for item in data.candidates)
    actual = {
        "scenes": len(data.records),
        "unscorable": sum(scene_reasons.values()),
        "no_choosable": scene_reasons["no-choosable"],
        "exclude_chosen": scene_reasons["exclude-chosen"],
        "hard_case_chosen": scene_reasons["hard_case-chosen"],
        "uncertain": uncertain,
    }
    mismatches = [f"{key}: expected {EXPECTED[key]}, actual {actual[key]}" for key in EXPECTED if actual[key] != EXPECTED[key]]
    if mismatches:
        raise ValueError("首轮数字对账失败，停止生成结果：" + "; ".join(mismatches))


def group_uncertain(rows: Iterable[Candidate], key: str) -> dict[str, dict[str, float | int]]:
    totals: Counter[str] = Counter()
    uncertain: Counter[str] = Counter()
    for row in rows:
        group = str(getattr(row, key)) if hasattr(row, key) else str(row.label.get(key, "missing"))
        totals[group] += 1
        uncertain[group] += int(row.status == "uncertain")
    total_uncertain = sum(uncertain.values())
    result: dict[str, dict[str, float | int]] = {}
    for group in sorted(totals):
        result[group] = {
            "total": totals[group],
            "uncertain": uncertain[group],
            "uncertain_rate": uncertain[group] / totals[group] if totals[group] else 0.0,
            "share_of_uncertain": uncertain[group] / total_uncertain if total_uncertain else 0.0,
        }
    return result


def max_share_hint(table_name: str, table: dict[str, dict[str, float | int]]) -> str:
    if not table:
        return f"{table_name}: 无数据"
    group, values = max(table.items(), key=lambda pair: (float(pair[1]["share_of_uncertain"]), pair[0]))
    return f"{table_name}: {group} 占 uncertain 的 {float(values['share_of_uncertain']):.1%}"


def variant_verdict_table(candidates: Iterable[Candidate]) -> dict[str, dict[str, float | int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for candidate in candidates:
        counts[str(candidate.label.get("variant_type", "missing"))][candidate.status] += 1
    result: dict[str, dict[str, float | int]] = {}
    for variant in sorted(counts):
        row = counts[variant]
        total = sum(row.values())
        result[variant] = {
            "vetoed": row["vetoed"],
            "cleared": row["cleared"],
            "uncertain": row["uncertain"],
            "total": total,
            "vetoed_rate": row["vetoed"] / total if total else 0.0,
            "non_veto_rate": (row["cleared"] + row["uncertain"]) / total if total else 0.0,
        }
    return result


def variant_verdict_scenario_table(candidates: Iterable[Candidate]) -> dict[str, dict[str, dict[str, int]]]:
    counts: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for candidate in candidates:
        variant = str(candidate.label.get("variant_type", "missing"))
        counts[variant][candidate.status][candidate.scenario_type] += 1
    return {
        variant: {
            status: dict(sorted(scenarios.items()))
            for status, scenarios in sorted(statuses.items())
        }
        for variant, statuses in sorted(counts.items())
    }


def t9_stratum(data: AuditData, record: dict[str, Any]) -> str:
    chosen = chosen_candidate(data, record)
    if chosen is None:
        return "missing_chosen|missing_variant"
    return f"{precheck_reason(chosen.label)}|{chosen.label.get('variant_type', 'missing')}"


def llm_reason_keyword_frequency(data: AuditData) -> dict[str, Any]:
    reasons = [
        str(candidate.verdict.get("veto", {}).get("reason", ""))
        for candidate in data.candidates
        if candidate.status == "uncertain" and candidate.decided_by == "llm"
    ]
    counts: Counter[str] = Counter()
    feat_gap_proxy_reasons = 0
    for reason in reasons:
        matched_feat_gap = False
        for keyword, pattern in LLM_REASON_KEYWORDS.items():
            occurrences = len(re.findall(pattern, reason))
            counts[keyword] += occurrences
            matched_feat_gap = matched_feat_gap or (keyword in FEAT_GAP_KEYWORDS and occurrences > 0)
        feat_gap_proxy_reasons += int(matched_feat_gap)
    total = len(reasons)
    return {
        "scope": "verdict.status=uncertain AND verdict.decided_by=llm",
        "reason_count": total,
        "keywords": {
            keyword: {"frequency": counts[keyword], "reason_coverage": sum(
                bool(re.search(pattern, reason)) for reason in reasons
            ) / total if total else 0.0}
            for keyword, pattern in LLM_REASON_KEYWORDS.items()
        },
        "feat_gap_proxy": {
            "keywords": list(FEAT_GAP_KEYWORDS),
            "reason_count_with_missing_evidence_terms": feat_gap_proxy_reasons,
            "rate": feat_gap_proxy_reasons / total if total else 0.0,
            "interpretation": "启发式预检，不等同于人工 FEAT-GAP 裁定。",
        },
    }


def crosstab_payload(data: AuditData) -> dict[str, Any]:
    candidates = data.candidates
    t1 = group_uncertain(candidates, "scenario_type")
    t2 = group_uncertain(candidates, "variant_type")
    t3 = group_uncertain(candidates, "decided_by")
    t8 = variant_verdict_table(candidates)
    t11 = variant_verdict_scenario_table(candidates)
    t4: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(Counter))
    for row in candidates:
        t4[row.status][row.scenario_type][str(row.label.get("difficulty", "missing"))] += 1
    t5: dict[str, Counter[str]] = defaultdict(Counter)
    for record in data.records:
        reason = unscorable_reason(data, record)
        if reason:
            t5[reason][str(record["layer1"]["context"].get("scenario_type", "unknown"))] += 1
    t6: Counter[str] = Counter()
    for record in data.records:
        if scene_bucket(data, record) != "B1":
            continue
        counts = Counter(item.status for item in candidates_for(data, record))
        composition = " + ".join(f"{status}={counts.get(status, 0)}" for status in ("vetoed", "uncertain", "cleared"))
        t6[composition] += 1
    t7: Counter[str] = Counter()
    t9: Counter[str] = Counter()
    for record in data.records:
        if scene_bucket(data, record) != "B2":
            continue
        chosen = chosen_candidate(data, record)
        if chosen is None:
            continue
        reason = precheck_reason(chosen.label)
        t7[reason] += 1
        t9[t9_stratum(data, record)] += 1
    signal_records = [
        record for record in data.records
        if str(record["layer1"]["context"].get("scenario_type")) in {"red_light", "yellow_light"}
    ]
    traffic_light_none = sum(
        record["layer1"]["context"].get("scene_facts", {}).get("traffic_light_status_source") == "none"
        for record in signal_records
    )
    traffic_light_covered = len(signal_records) - traffic_light_none
    return {
        "reconciliation": {
            "expected": EXPECTED,
            "actual": {
                "scenes": len(data.records),
                "unscorable": sum(unscorable_reason(data, record) is not None for record in data.records),
                "no_choosable": sum(scene_bucket(data, record) == "B1" for record in data.records),
                "exclude_chosen": sum(scene_bucket(data, record) == "B2" for record in data.records),
                "hard_case_chosen": sum(scene_bucket(data, record) == "B4" for record in data.records),
                "uncertain": sum(row.status == "uncertain" for row in candidates),
                "total_candidates": len(candidates),
            },
            "passed": True,
        },
        "tables": {
            "T1_uncertain_by_scenario_type": t1,
            "T2_uncertain_by_variant_type": t2,
            "T3_uncertain_by_decided_by": t3,
            "T4_verdict_by_scenario_type_and_difficulty": {
                status: {scenario: dict(sorted(difficulties.items())) for scenario, difficulties in sorted(scenarios.items())}
                for status, scenarios in sorted(t4.items())
            },
            "T5_unscorable_reason_by_scenario_type": {
                reason: dict(sorted(counts.items())) for reason, counts in sorted(t5.items())
            },
            "T6_no_choosable_candidate_composition": dict(sorted(t6.items())),
            "T7_exclude_chosen_by_gt_precheck_failure_reason": dict(sorted(t7.items())),
            "T8_variant_by_verdict": t8,
            "T9_exclude_reason_by_variant": dict(sorted(t9.items())),
            "T10_uncertain_llm_reason_keyword_frequency": llm_reason_keyword_frequency(data),
            "T11_variant_by_verdict_and_scenario_type": t11,
        },
        "derived_metrics": {
            "uncertain_rate": len([row for row in candidates if row.status == "uncertain"]) / len(candidates),
            "traffic_light_status_source_none": {
                "count": traffic_light_none,
                "signal_scenes": len(signal_records),
                "definition": "scenario_type ∈ {red_light, yellow_light}",
                "rate": traffic_light_none / len(signal_records) if signal_records else 0.0,
            },
            "traffic_light_status_coverage": {
                "covered_signal_scenes": traffic_light_covered,
                "signal_scenes": len(signal_records),
                "coverage_rate": traffic_light_covered / len(signal_records) if signal_records else 0.0,
                "definition": "signal scene means scenario_type ∈ {red_light, yellow_light}; covered means traffic_light_status_source != none",
            },
            "max_share_hints": {
                "T1": max_share_hint("T1", t1),
                "T2": max_share_hint("T2", t2),
                "T3": max_share_hint("T3", t3),
            },
        },
        "sources": {
            "eval_log": str(DEFAULT_EVAL.relative_to(ROOT)),
            "git_commit": data.manifest.get("git_commit"),
            "benchmark_data_version": data.manifest.get("benchmark_data_version"),
        },
    }


def precheck_reason(label: dict[str, Any]) -> str:
    status = str(label.get("gt_precheck_status", "missing"))
    if status == "failed":
        return str(label.get("expected_verdict_reason") or "failed: unspecified")
    if status == "not_evaluable":
        return "not_evaluable"
    if status == "not_gt":
        return "not_gt (synthetic candidate)"
    return status


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(cell).replace("|", "\\|").replace("\n", " ") for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def crosstab_markdown(payload: dict[str, Any]) -> str:
    tables = payload["tables"]
    lines = [
        "# 诊断性审计交叉表",
        "",
        "> 只读生成；不产出 gold label，不进入指标计算。",
        "",
        "## 核对表",
        "",
        markdown_table(
            ["指标", "期望", "实际", "状态"],
            [[key, payload["reconciliation"]["expected"][key], payload["reconciliation"]["actual"][key], "PASS"] for key in EXPECTED],
        ),
    ]
    for name in ("T1_uncertain_by_scenario_type", "T2_uncertain_by_variant_type", "T3_uncertain_by_decided_by"):
        rows = [[key, value["total"], value["uncertain"], f"{value['uncertain_rate']:.1%}", f"{value['share_of_uncertain']:.1%}"] for key, value in tables[name].items()]
        lines += ["", f"## {name}", "", markdown_table(["分组", "候选数", "uncertain", "组内率", "占全部 uncertain"], rows)]
    rows = []
    for status, scenarios in tables["T4_verdict_by_scenario_type_and_difficulty"].items():
        for scenario, difficulties in scenarios.items():
            for difficulty, count in difficulties.items():
                rows.append([status, scenario, difficulty, count])
    lines += ["", "## T4_verdict_by_scenario_type_and_difficulty", "", markdown_table(["verdict", "scenario_type", "difficulty", "数量"], rows)]
    rows = [[reason, scenario, count] for reason, scenarios in tables["T5_unscorable_reason_by_scenario_type"].items() for scenario, count in scenarios.items()]
    lines += ["", "## T5_unscorable_reason_by_scenario_type", "", markdown_table(["不可评分原因", "scenario_type", "数量"], rows)]
    lines += ["", "## T6_no_choosable_candidate_composition", "", markdown_table(["候选构成", "场景数"], [[key, value] for key, value in tables["T6_no_choosable_candidate_composition"].items()])]
    lines += ["", "## T7_exclude_chosen_by_gt_precheck_failure_reason", "", markdown_table(["precheck 失败原因/状态", "场景数"], [[key, value] for key, value in tables["T7_exclude_chosen_by_gt_precheck_failure_reason"].items()])]
    lines += ["", "## T8_variant_by_verdict", "", markdown_table(
        ["variant_type", "vetoed", "cleared", "uncertain", "总数", "vetoed率", "非veto率"],
        [[variant, row["vetoed"], row["cleared"], row["uncertain"], row["total"], f"{row['vetoed_rate']:.1%}", f"{row['non_veto_rate']:.1%}"] for variant, row in tables["T8_variant_by_verdict"].items()],
    )]
    lines += ["", "## T9_exclude_reason_by_variant", "", markdown_table(
        ["exclude原因|variant_type", "场景数"],
        [[key, value] for key, value in tables["T9_exclude_reason_by_variant"].items()],
    )]
    t10 = tables["T10_uncertain_llm_reason_keyword_frequency"]
    lines += ["", "## T10_uncertain_llm_reason_keyword_frequency", "", f"scope: `{t10['scope']}`；reason_count={t10['reason_count']}", "", markdown_table(
        ["关键词", "出现频次", "reason 覆盖率"],
        [[keyword, row["frequency"], f"{row['reason_coverage']:.1%}"] for keyword, row in t10["keywords"].items()],
    ), "", json.dumps(t10["feat_gap_proxy"], ensure_ascii=False, indent=2, sort_keys=True)]
    t11_rows = [
        [variant, status, scenario, count]
        for variant, statuses in tables["T11_variant_by_verdict_and_scenario_type"].items()
        for status, scenarios in statuses.items()
        for scenario, count in scenarios.items()
    ]
    lines += ["", "## T11_variant_by_verdict_and_scenario_type", "", markdown_table(
        ["variant_type", "verdict", "scenario_type", "数量"], t11_rows,
    )]
    lines += ["", "## 派生指标", "", json.dumps(payload["derived_metrics"], ensure_ascii=False, indent=2, sort_keys=True), ""]
    return "\n".join(lines)


def allocate_counts(group_sizes: dict[str, int], total: int, minimum: int = 0) -> dict[str, int]:
    keys = sorted(group_sizes)
    if not keys or total < minimum * len(keys):
        raise ValueError("抽样量不足以满足每层最低样本数")
    result = {key: minimum for key in keys}
    remaining = total - sum(result.values())
    population = sum(group_sizes.values())
    raw = {key: remaining * group_sizes[key] / population for key in keys}
    for key in keys:
        result[key] += math.floor(raw[key])
    for key in sorted(keys, key=lambda item: (-(raw[item] - math.floor(raw[item])), item))[: remaining - sum(math.floor(value) for value in raw.values())]:
        result[key] += 1
    return result


def hash_sorted(rows: Iterable[Any], *parts: str) -> list[Any]:
    return sorted(rows, key=lambda row: stable_hash(*parts, getattr(row, "scene_id", ""), getattr(row, "frame_token", ""), getattr(row, "anonymous_id", "")))


def scene_stratum(data: AuditData, record: dict[str, Any], key: str) -> str:
    if key == "gt_precheck_status":
        chosen = chosen_candidate(data, record)
        return precheck_reason(chosen.label) if chosen else "missing_chosen"
    if key == "b2_t9":
        return t9_stratum(data, record)
    return str(record["layer1"]["context"].get(key, "unknown"))


def stratified_scene_sample(
    data: AuditData,
    rows: list[dict[str, Any]],
    total: int,
    key: str,
    force: list[dict[str, Any]] = (),
    minimum: int | None = None,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[scene_stratum(data, row, key)].append(row)
    if minimum is None:
        minimum = 2 if key == "scenario_type" else 3
    forced_keys = {(row["raw_scene"]["scene_id"], row["raw_scene"]["frame_token"]) for row in force}
    forced_by_group = {
        group: [row for row in value if (row["raw_scene"]["scene_id"], row["raw_scene"]["frame_token"]) in forced_keys]
        for group, value in groups.items()
    }
    base_quotas = {group: max(minimum, len(forced_by_group[group])) for group in groups}
    if sum(base_quotas.values()) > total:
        raise ValueError(f"抽样量不足以同时满足 {key} 分层和强制纳入")
    remaining = total - sum(base_quotas.values())
    non_forced_sizes = {
        group: len(groups[group]) - len(forced_by_group[group])
        for group in groups
        if len(groups[group]) > len(forced_by_group[group])
    }
    extra_quotas = allocate_counts(non_forced_sizes, remaining) if remaining and non_forced_sizes else {}
    quotas = {group: base_quotas[group] + extra_quotas.get(group, 0) for group in groups}
    selected: list[dict[str, Any]] = []
    for group in sorted(groups):
        ordered = sorted(groups[group], key=lambda row: stable_hash(SAMPLE_SEED, row["raw_scene"]["scene_id"], row["raw_scene"]["frame_token"]))
        forced_rows = forced_by_group[group]
        other_rows = [row for row in ordered if row not in forced_rows]
        selected.extend(forced_rows + other_rows[: max(0, quotas[group] - len(forced_rows))])
    if len(selected) < total:
        selected_keys = {(row["raw_scene"]["scene_id"], row["raw_scene"]["frame_token"]) for row in selected}
        remaining_rows = [row for row in rows if (row["raw_scene"]["scene_id"], row["raw_scene"]["frame_token"]) not in selected_keys]
        selected.extend(sorted(remaining_rows, key=lambda row: stable_hash(SAMPLE_SEED, row["raw_scene"]["scene_id"], row["raw_scene"]["frame_token"]))[: total - len(selected)])
    return sorted(selected, key=lambda row: stable_hash(SAMPLE_SEED, row["raw_scene"]["scene_id"], row["raw_scene"]["frame_token"]))[:total]


def stratified_candidate_sample(rows: list[Candidate], total: int) -> list[Candidate]:
    groups: dict[str, list[Candidate]] = defaultdict(list)
    for row in rows:
        groups[row.scenario_type].append(row)
    selected: list[Candidate] = []
    for scenario_type, quota in B3_SCENARIO_QUOTAS.items():
        population = sorted(groups.get(scenario_type, []), key=lambda item: stable_hash(SAMPLE_SEED, item.scene_id, item.frame_token, item.anonymous_id))
        if len(population) < quota:
            raise ValueError(f"B3 {scenario_type} uncertain 候选不足：需要 {quota}，实际 {len(population)}")
        selected.extend(population[:quota])
    if len(selected) != total:
        raise ValueError(f"B3 配额总数应为 {total}，实际 {len(selected)}")
    return sorted(selected, key=lambda item: stable_hash(SAMPLE_SEED, item.scene_id, item.frame_token, item.anonymous_id))


def extreme_b1_records(data: AuditData) -> list[dict[str, Any]]:
    rows = [record for record in data.records if scene_bucket(data, record) == "B1"]
    endpoints: dict[str, list[dict[str, Any]]] = {"all_uncertain": [], "all_vetoed": []}
    for record in rows:
        statuses = Counter(item.status for item in candidates_for(data, record))
        if statuses["uncertain"] == 6:
            endpoints["all_uncertain"].append(record)
        if statuses["vetoed"] == 6:
            endpoints["all_vetoed"].append(record)
    selected = []
    required = {"all_uncertain": 4, "all_vetoed": 6}
    for endpoint in ("all_uncertain", "all_vetoed"):
        ordered = sorted(endpoints[endpoint], key=lambda row: stable_hash(SAMPLE_SEED, row["raw_scene"]["scene_id"], row["raw_scene"]["frame_token"]))
        if len(ordered) < required[endpoint]:
            raise ValueError(f"B1 {endpoint} 少于 {required[endpoint]} 个，无法满足强制纳入")
        selected.extend(ordered[: required[endpoint]])
    return selected


def cap_b1_all_vetoed(
    data: AuditData,
    sampled: list[dict[str, Any]],
    population: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    def is_all_vetoed(record: dict[str, Any]) -> bool:
        return Counter(item.status for item in candidates_for(data, record))["vetoed"] == 6

    ordered_vetoed = sorted(
        [record for record in sampled if is_all_vetoed(record)],
        key=lambda row: stable_hash(SAMPLE_SEED, row["raw_scene"]["scene_id"], row["raw_scene"]["frame_token"]),
    )
    if len(ordered_vetoed) <= 6:
        return sampled
    keep_keys = {
        (record["raw_scene"]["scene_id"], record["raw_scene"]["frame_token"])
        for record in ordered_vetoed[:6]
    }
    remove = [
        record for record in ordered_vetoed
        if (record["raw_scene"]["scene_id"], record["raw_scene"]["frame_token"]) not in keep_keys
    ]
    selected_keys = {
        (record["raw_scene"]["scene_id"], record["raw_scene"]["frame_token"])
        for record in sampled
    }
    replacements = [
        record for record in population
        if (record["raw_scene"]["scene_id"], record["raw_scene"]["frame_token"]) not in selected_keys
        and not is_all_vetoed(record)
    ]
    replacements = sorted(
        replacements,
        key=lambda row: stable_hash(SAMPLE_SEED, row["raw_scene"]["scene_id"], row["raw_scene"]["frame_token"]),
    )
    if len(replacements) < len(remove):
        raise ValueError("B1 无法将全 vetoed 端控制在 6 个")
    result = [record for record in sampled if record not in remove]
    result.extend(replacements[:len(remove)])
    return sorted(result, key=lambda row: stable_hash(SAMPLE_SEED, row["raw_scene"]["scene_id"], row["raw_scene"]["frame_token"]))


def special_record(data: AuditData) -> tuple[dict[str, Any] | None, str | None]:
    records = [record for record in data.records if scene_name(record) == "scene-0286"]
    if not records:
        return None, None
    target_frame = None
    target_id = None
    if DEFAULT_SPECIAL_SIDECAR.is_file():
        for line in DEFAULT_SPECIAL_SIDECAR.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if item.get("scene_name") == "scene-0286":
                target_frame = item.get("frame_token")
                target_id = item.get("trajectory_id")
                break
    chosen = next((record for record in records if record["raw_scene"].get("frame_token") == target_frame), None)
    chosen = chosen or min(records, key=lambda record: stable_hash(record["raw_scene"]["scene_id"], record["raw_scene"]["frame_token"]))
    return chosen, target_id


def b6_candidates(data: AuditData) -> list[Candidate]:
    candidates = [
        candidate for candidate in data.candidates
        if candidate.label.get("variant_type") == "illegal" and candidate.status == "cleared"
    ]
    if len(candidates) != 52:
        raise ValueError(f"B6 illegal-cleared 候选应为 52 个，实际 {len(candidates)}")
    return sorted(candidates, key=lambda item: stable_hash(SAMPLE_SEED, item.scene_id, item.frame_token, item.anonymous_id))


def b6_sample_candidates(population: list[Candidate]) -> list[Candidate]:
    selected: list[Candidate] = []
    for scenario_type, quota in B6_SCENARIO_QUOTAS.items():
        candidates = [candidate for candidate in population if candidate.scenario_type == scenario_type]
        if len(candidates) < quota:
            raise ValueError(f"B6 {scenario_type} 候选不足：需要 {quota}，实际 {len(candidates)}")
        selected.extend(candidates[:quota])
    return selected


def b7_candidates(data: AuditData) -> list[Candidate]:
    candidates = [
        candidate for candidate in data.candidates
        if candidate.label.get("variant_type") == "ground_truth"
        and candidate.status == "vetoed"
        and candidate.scenario_type == "red_light"
    ]
    if len(candidates) != 32:
        raise ValueError(f"B7 ground_truth-vetoed red_light 候选应为 32 个，实际 {len(candidates)}")
    return sorted(candidates, key=lambda item: stable_hash(SAMPLE_SEED, item.scene_id, item.frame_token, item.anonymous_id))


def build_samples(data: AuditData) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    b1 = [record for record in data.records if scene_bucket(data, record) == "B1"]
    b2 = [record for record in data.records if scene_bucket(data, record) == "B2"]
    b3 = [row for row in data.candidates if row.status == "uncertain"]
    b6_population = b6_candidates(data)
    b7_population = b7_candidates(data)
    forced_b1 = extreme_b1_records(data)
    sampled_b1 = stratified_scene_sample(data, b1, 24, "scenario_type", forced_b1, minimum=2)
    sampled_b1 = cap_b1_all_vetoed(data, sampled_b1, b1)
    sampled_b2 = stratified_scene_sample(data, b2, 20, "b2_t9", minimum=3)
    sampled_b3 = stratified_candidate_sample(b3, 30)
    sampled_b6 = b6_sample_candidates(b6_population)
    sampled_b7 = b7_population[:B7_SAMPLE_SIZE]
    special, special_id = special_record(data)
    items: dict[tuple[str, str], dict[str, Any]] = {}

    def add(bucket: str, record: dict[str, Any], target_id: str | None = None) -> None:
        key = (str(record["raw_scene"]["scene_id"]), str(record["raw_scene"]["frame_token"]))
        item = items.setdefault(key, {"record": record, "buckets": set(), "target_ids": set()})
        item["buckets"].add(bucket)
        if target_id:
            item["target_ids"].add(target_id)

    for record in sampled_b1:
        add("B1", record)
    for record in sampled_b2:
        chosen = chosen_candidate(data, record)
        add("B2", record, chosen.anonymous_id if chosen else None)
    for candidate in sampled_b3:
        add("B3", data.by_frame[(candidate.scene_id, candidate.frame_token)], candidate.anonymous_id)
    for candidate in sampled_b6:
        add("B6", data.by_frame[(candidate.scene_id, candidate.frame_token)], candidate.anonymous_id)
    for candidate in sampled_b7:
        add("B7", data.by_frame[(candidate.scene_id, candidate.frame_token)], candidate.anonymous_id)
    if special:
        add("B5", special, special_id)
    manifest = {
        "seed": SAMPLE_SEED,
        "method": "sha256(42|scene_id|frame_token|anonymous_id) ascending; quotas use largest remainder",
        "counts": {"B1": 24, "B2": 20, "B3": 30, "B4": 0, "B5": int(special is not None), "B6": 15, "B7": B7_SAMPLE_SIZE},
        "b3_scenario_quotas": B3_SCENARIO_QUOTAS,
        "b3_scenario_sample": dict(sorted(Counter(row.scenario_type for row in sampled_b3).items())),
        "b6_definition": "variant_type=illegal AND verdict=cleared",
        "b6_population": len(b6_population),
        "b6_sample": len(sampled_b6),
        "b6_scenario_quotas": B6_SCENARIO_QUOTAS,
        "b6_scenario_sample": dict(sorted(Counter(row.scenario_type for row in sampled_b6).items())),
        "b6_sample_keys": [
            [candidate.scene_id, candidate.frame_token, candidate.anonymous_id]
            for candidate in sampled_b6
        ],
        "b7_definition": "variant_type=ground_truth AND verdict=vetoed AND scenario_type=red_light",
        "b7_population": len(b7_population),
        "b7_sample": len(sampled_b7),
        "b7_sample_keys": [
            [candidate.scene_id, candidate.frame_token, candidate.anonymous_id]
            for candidate in sampled_b7
        ],
        "forced_b1": [[row["raw_scene"]["scene_id"], row["raw_scene"]["frame_token"]] for row in forced_b1],
        "b1_forced_endpoint_counts": {
            "all_uncertain": sum(Counter(item.status for item in candidates_for(data, record))["uncertain"] == 6 for record in sampled_b1),
            "all_vetoed": sum(Counter(item.status for item in candidates_for(data, record))["vetoed"] == 6 for record in sampled_b1),
        },
        "special_scene": {"scene_name": "scene-0286", "target_id": special_id} if special else None,
        "b3_source_population": dict(sorted(Counter(row.decided_by for row in b3).items())),
        "b4_variant_distribution": dict(sorted(Counter(
            candidate.label.get("variant_type")
            for record in data.records
            if scene_bucket(data, record) == "B4"
            for candidate in candidates_for(data, record)
            ).items())),
        "b2_t9_population": dict(sorted(Counter(t9_stratum(data, record) for record in b2).items())),
        "b2_t9_sample": dict(sorted(Counter(t9_stratum(data, record) for record in sampled_b2).items())),
    }
    missing_sources = [source for source in ("llm", "rule_engine") if Counter(row.decided_by for row in b3).get(source, 0) < 10]
    if missing_sources:
        manifest["sampling_warnings"] = [
            f"B3 强制层 {', '.join(missing_sources)} < 10；未伪造样本，按现有 uncertain 候选抽样。",
        ]
    rows = [build_audit_row(data, value["record"], value["buckets"], value["target_ids"], index) for index, value in enumerate(sorted(items.values(), key=lambda value: stable_hash(value["record"]["raw_scene"]["scene_id"], value["record"]["raw_scene"]["frame_token"])), 1)]
    manifest["worksheet_rows"] = len(rows)
    manifest["worksheet_bucket_rows"] = dict(sorted(Counter(bucket for item in items.values() for bucket in item["buckets"]).items()))
    manifest["b7_overlap_rows"] = sum("B1" in item["buckets"] and "B7" in item["buckets"] for item in items.values())
    manifest["overlap_note"] = "B1/B2/B3/B6/B7 在同一 scene/frame 时合并为一行；counts 是抽样项数，worksheet_bucket_rows 是合并后的行数。"
    return rows, manifest


def flatten_facts(facts: dict[str, Any]) -> str:
    labels = {
        "has_red_light": "有红灯",
        "has_yellow_light": "有黄灯",
        "traffic_light_status_source": "信号灯相位来源",
        "pedestrian_in_crosswalk": "行人在人行横道",
        "oncoming_vehicle_moving": "对向车行驶",
        "emergency_vehicle_active": "特种车辆执行任务",
        "ego_on_minor_road": "自车在支路",
        "location_is_intersection": "路口",
        "ego_displacement_m": "自车位移(m)",
    }
    values = [f"{labels.get(key, key)}={json_text(value)}" for key, value in facts.items() if key != "conflict_zones"]
    zones = facts.get("conflict_zones", [])
    values.append(f"冲突区数量={len(zones)}")
    return "；".join(values)


def feature_summary(candidate: Candidate) -> str:
    return json_text(candidate.feature or {"feature_missing": True})


def build_audit_row(data: AuditData, record: dict[str, Any], buckets: set[str], target_ids: set[str], index: int) -> dict[str, Any]:
    all_candidates = candidates_for(data, record)
    include_all = bool(buckets & {"B1", "B2"})
    target = [candidate for candidate in all_candidates if candidate.anonymous_id in target_ids]
    evidence_candidates = all_candidates if include_all else target
    context = record["layer1"]["context"]
    raw = record["raw_scene"]
    layer2 = record.get("layer2", {})
    description = context.get("description", {})
    narrative = description.get("narrative") or raw.get("scene_description") or raw.get("drivelm_scene_description") or "[narrative 未归档]"
    verdicts = [
        {"anonymous_id": candidate.anonymous_id, "status": candidate.status, "decided_by": candidate.decided_by, "reason": candidate.verdict.get("veto", {}).get("reason", "")}
        for candidate in all_candidates
    ]
    benchmark = [
        {key: candidate.label.get(key) for key in ("anonymous_id", "variant_type", "expected_verdict", "difficulty", "gt_precheck_status", "expected_verdict_reason")}
        for candidate in evidence_candidates
    ]
    trajectory = [
        {"anonymous_id": candidate.anonymous_id, "feature": candidate.feature or {"feature_missing": True}}
        for candidate in evidence_candidates
    ]
    return {
        "audit_item_id": f"audit_{index:03d}",
        "bucket": "+".join(sorted(buckets)),
        "scene_id": raw.get("scene_id"),
        "frame_token": raw.get("frame_token"),
        "anonymous_id": ",".join(sorted(target_ids)),
        "scenario_type": context.get("scenario_type"),
        "scene_facts": flatten_facts(context.get("scene_facts", {})),
        "narrative": narrative,
        "retrieved_rules + overrides": json_text({"matched_conditions": layer2.get("matched_conditions", []), "retrieval_mode": layer2.get("retrieval_mode"), "hard_rule_ids": [rule.get("node_id") for rule in layer2.get("hard_rules", [])], "overrides_applied": layer2.get("overrides_applied", [])}),
        "trajectory_summary": json_text(trajectory),
        "system_verdicts": json_text(verdicts),
        "chosen": json_text({"chosen_trajectory_id": record.get("final_label", {}).get("chosen_trajectory_id"), "selection_outcome": record.get("final_label", {}).get("selection_outcome")}),
        "benchmark 侧": json_text(benchmark),
        "不可评分原因": ",".join(sorted(filter(None, (unscorable_reason(data, record), "scene-0286" if "B5" in buckets else None)))),
        "disposition_code": "",
        "evidence_note": "",
        "benchmark_v2_action": "",
        "needs_blind_confirmation": False,
        "time_spent_min": "",
    }


def write_worksheet(path: Path, rows: list[dict[str, Any]], sample_manifest: dict[str, Any]) -> None:
    disposition_explanations = {
        "BENCH-FIX": "benchmark 标签/口径错（override 未后置、precheck 过严、exclude 语义问题）",
        "FACT-FIX": "Layer 1 事实提取错（相位误读、行人误判、朝向夹角错）",
        "FEAT-GAP": "特征摘要缺少判定所需信息，弃权诚实但可通过扩充特征解决",
        "JUDGE-MISS": "信息充分但判定错/不敢判（prompt、规则引擎阈值）",
        "CORRECT-ABSTAIN": "场景本身不可判，系统弃权正确，数据固有限制",
        "SCOPE-CUT": "细分样本量/质量低于 W1 阈值，应从 MVP 裁掉",
        "HOLD": "拿不准，挂起待盲标数据交叉印证",
    }
    disposition_options = "".join(
        f"<option value='{html.escape(code)}'>{html.escape(code)} — {html.escape(disposition_explanations[code])}</option>"
        for code in DISPOSITION_CODES
    )
    evidence_columns = [
        "scene_facts",
        "narrative",
        "retrieved_rules + overrides",
        "trajectory_summary",
        "system_verdicts",
        "chosen",
        "benchmark 侧",
        "不可评分原因",
    ]
    meta_columns = ["audit_item_id", "bucket", "scene_id", "frame_token", "anonymous_id", "scenario_type"]
    cards = []
    for row in rows:
        meta = "".join(
            f"<div><dt>{html.escape(column)}</dt><dd>{html.escape(str(row.get(column, '')))}</dd></div>"
            for column in meta_columns
        )
        evidence = "".join(
            f"<div class='evidence-row'><div class='field-name'>{html.escape(column)}</div><div class='field-value'>{html.escape(str(row.get(column, '')))}</div></div>"
            for column in evidence_columns
        )
        decision = f"""<div class='decision-grid'>
  <label class='decision-field wide'><span>disposition_code</span><select name='disposition_code'>{disposition_options}</select></label>
  <label class='decision-field'><span>needs_blind_confirmation</span><input type='checkbox' name='needs_blind_confirmation'></label>
  <label class='decision-field'><span>time_spent_min</span><input type='number' name='time_spent_min' min='0' step='1'></label>
  <label class='decision-field full'><span>evidence_note</span><textarea name='evidence_note' placeholder='一句话记录：你的判断 vs 系统判断 + 差异归因'></textarea></label>
  <label class='decision-field full'><span>benchmark_v2_action</span><textarea name='benchmark_v2_action' placeholder='可空；填写 benchmark v2 修订动作'></textarea></label>
</div>"""
        cards.append(f"""<article class='audit-card'>
  <h2>{html.escape(str(row.get('audit_item_id', '')))} <span class='bucket'>{html.escape(str(row.get('bucket', '')))}</span></h2>
  <dl class='meta-grid'>{meta}</dl>
  <h3>证据区</h3>
  <div class='evidence-grid'>{evidence}</div>
  <h3>裁定区</h3>
  {decision}
</article>""")
    content = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>诊断性审计工作表</title>
<style>
*{box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#f4f6f8;color:#202124;line-height:1.5}
main{max-width:1500px;margin:0 auto;padding:24px}
h1{margin:0 0 8px;font-size:26px}h2{margin:0 0 14px;font-size:20px}h3{margin:20px 0 10px;font-size:16px;border-bottom:1px solid #d9dee5;padding-bottom:5px}
.intro{background:#fff;border:1px solid #d9dee5;border-radius:10px;padding:16px;margin-bottom:18px}
details{margin-top:10px}summary{cursor:pointer;font-weight:600}.manifest{white-space:pre-wrap;overflow-wrap:anywhere;font:12px ui-monospace,SFMono-Regular,Menlo,monospace;background:#f7f8fa;padding:10px;border-radius:6px}
.audit-card{background:#fff;border:1px solid #cfd6df;border-radius:10px;padding:18px;margin:0 0 18px;box-shadow:0 1px 2px #0000000b}
.bucket{display:inline-block;font-size:12px;font-weight:600;color:#245b8f;background:#e7f1fb;border-radius:999px;padding:3px 9px;vertical-align:middle}
.meta-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px 16px;margin:0}.meta-grid div{min-width:0}.meta-grid dt{font-size:11px;color:#68727d;text-transform:none}.meta-grid dd{margin:0;overflow-wrap:anywhere;font:12px ui-monospace,SFMono-Regular,Menlo,monospace}
.evidence-grid{border:1px solid #d9dee5;border-radius:6px;overflow:hidden}.evidence-row{display:grid;grid-template-columns:minmax(170px,230px) minmax(0,1fr);border-bottom:1px solid #e5e8ec}.evidence-row:last-child{border-bottom:0}.field-name{background:#f7f8fa;padding:10px;font-weight:600;font-size:12px;overflow-wrap:anywhere}.field-value{padding:10px;white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word;font:12px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;min-width:0}
.decision-grid{display:grid;grid-template-columns:minmax(230px,1fr) minmax(230px,1fr);gap:12px}.decision-field{display:flex;flex-direction:column;gap:5px;font-size:12px}.decision-field span{font-weight:600;color:#4d5965}.decision-field.full{grid-column:1/-1}.decision-field.wide{grid-column:1/-1}.decision-field select,.decision-field input,.decision-field textarea{font:13px inherit;border:1px solid #b8c1cc;border-radius:5px;padding:8px;background:#fff;max-width:100%}.decision-field textarea{min-height:72px;resize:vertical}
@media(max-width:700px){main{padding:12px}.evidence-row{grid-template-columns:1fr}.field-name{border-bottom:1px solid #e5e8ec}.decision-grid{grid-template-columns:1fr}.decision-field.full,.decision-field.wide{grid-column:auto}}
</style></head><body><main>
<section class="intro"><h1>诊断性审计工作表</h1>
<p>每个审计项独立展示，证据区与裁定区分开。请先阅读证据，再填写 disposition_code 和人工备注。</p>
<details><summary>查看抽样配置</summary><div class="manifest">__MANIFEST__</div></details></section>
__CARDS__
</main></body></html>
""".replace("__MANIFEST__", html.escape(json.dumps(sample_manifest, ensure_ascii=False, indent=2, sort_keys=True))).replace("__CARDS__", "".join(cards))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def run_crosstab(args: argparse.Namespace) -> None:
    data = load_data(args.eval)
    assert_reconciliation(data)
    payload = crosstab_payload(data)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "crosstabs.json", payload)
    (args.output / "crosstabs.md").write_text(crosstab_markdown(payload), encoding="utf-8")


def run_sample(args: argparse.Namespace) -> None:
    data = load_data(args.eval)
    assert_reconciliation(data)
    rows, manifest = build_samples(data)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "audit_sample.json", {"manifest": manifest, "rows": rows})
    write_json(args.output / "audit_sample_manifest.json", manifest)
    write_worksheet(args.output / "audit_worksheet.html", rows, manifest)
    for warning in manifest.get("sampling_warnings", []):
        print(f"WARNING: {warning}")


def extract_waypoints(record: dict[str, Any], cache: dict[str, Any] | None) -> dict[str, list[tuple[float, float]]]:
    source: Any = cache
    if source is None:
        source = record.get("trajectory_cache") or record.get("candidate_trajectories") or record.get("raw_scene", {}).get("candidate_trajectories")
    if isinstance(source, dict) and "candidates" in source:
        source = source["candidates"]
    result: dict[str, list[tuple[float, float]]] = {}
    if isinstance(source, dict):
        source = [{"anonymous_id": key, "waypoints": value} for key, value in source.items()]
    if not isinstance(source, list):
        return result
    for item in source:
        if not isinstance(item, dict):
            continue
        anonymous_id = item.get("anonymous_id") or item.get("traj_id") or item.get("trajectory_id")
        waypoints = item.get("waypoints")
        if not anonymous_id or not isinstance(waypoints, list):
            continue
        points = []
        for point in waypoints:
            if isinstance(point, dict) and "x" in point and "y" in point:
                points.append((float(point["x"]), float(point["y"])))
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                points.append((float(point[0]), float(point[1])))
        if len(points) >= 2:
            result[str(anonymous_id)] = points
    return result


def render_record(record: dict[str, Any], output_dir: Path, cache: dict[str, Any] | None) -> str:
    import matplotlib.pyplot as plt

    waypoints = extract_waypoints(record, cache)
    raw = record["raw_scene"]
    offset = (0.0, 0.0)
    raw_input = record.get("_raw_input")
    if isinstance(raw_input, dict) and raw_input.get("ego_poses"):
        translation = raw_input["ego_poses"][0].get("translation", [0.0, 0.0])
        offset = (float(translation[0]), float(translation[1]))
    fig, ax = plt.subplots(figsize=(8, 6))
    for layer_name, color in (("stop_line", "#d62728"), ("ped_crossing", "#9467bd")):
        for polygon in (raw_input or {}).get("map_records", {}).get(layer_name, []):
            points = polygon.get("polygon_xy", [])
            if points:
                xs = [point[0] - offset[0] for point in points] + [points[0][0] - offset[0]]
                ys = [point[1] - offset[1] for point in points] + [points[0][1] - offset[1]]
                ax.fill(xs, ys, alpha=0.25, color=color, label=layer_name)
    for anonymous_id, points in sorted(waypoints.items()):
        ax.plot([point[0] for point in points], [point[1] for point in points], label=anonymous_id)
        ax.annotate(anonymous_id, points[-1])
    ax.set_title(f"{scene_name(record)} / {raw['frame_token']}")
    ax.set_xlabel("local x")
    ax.set_ylabel("local y")
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), fontsize=8)
    figure_path = output_dir / "figs" / f"{scene_name(record)}_{raw['frame_token']}.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(figure_path.relative_to(output_dir))


def run_render(args: argparse.Namespace) -> None:
    data = load_data(args.eval)
    raw_by_frame: dict[tuple[str, str], dict[str, Any]] = {}
    if args.raw_input.is_file():
        payload = json.loads(args.raw_input.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    raw_by_frame[(str(item.get("scene_id")), str(item.get("frame_token")))] = item
    cache_payload = None
    if args.trajectory_cache:
        cache_payload = json.loads(args.trajectory_cache.read_text(encoding="utf-8"))
    rendered = []
    skipped = []
    for record in data.records:
        key = (str(record["raw_scene"]["scene_id"]), str(record["raw_scene"]["frame_token"]))
        enriched = dict(record)
        enriched["_raw_input"] = raw_by_frame.get(key, {})
        frame_cache = cache_payload.get(record["raw_scene"]["frame_token"], {}) if isinstance(cache_payload, dict) else None
        if not extract_waypoints(enriched, frame_cache):
            skipped.append({"scene_id": key[0], "frame_token": key[1], "reason": "输入未提供候选 waypoints；未伪造轨迹"})
            continue
        rendered.append(render_record(enriched, args.output, frame_cache))
    write_json(args.output / "render_manifest.json", {"rendered": rendered, "skipped": skipped, "trajectory_source": str(args.trajectory_cache) if args.trajectory_cache else "embedded/raw input if available"})


def parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--eval", type=Path, default=DEFAULT_EVAL, help="冻结 JSONL 结果")
    common.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="独立输出目录")
    root = argparse.ArgumentParser(prog="python -m scripts.diagnostic_audit")
    sub = root.add_subparsers(dest="command", required=True)
    sub.add_parser("crosstab", parents=[common], help="生成交叉表与总量核对")
    sub.add_parser("sample", parents=[common], help="生成分层抽样工作表")
    render = sub.add_parser("render", parents=[common], help="生成可选轨迹图")
    render.add_argument("--raw-input", type=Path, default=DEFAULT_RAW, help="含 map_records 的 RawScene JSON")
    render.add_argument("--trajectory-cache", type=Path, default=None, help="可选的匿名轨迹缓存 JSON")
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "crosstab":
        run_crosstab(args)
    elif args.command == "sample":
        run_sample(args)
    else:
        run_render(args)


if __name__ == "__main__":
    main()
