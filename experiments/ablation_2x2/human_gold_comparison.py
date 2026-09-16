"""Score the completed 2x2 run against the finalized verdict-only human gold."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from experiments.ablation_2x2.citations import RuleCatalog, extract_provisions
from experiments.ablation_2x2.conditions import CONDITIONS
from experiments.ablation_2x2.metrics import cohen_kappa
from experiments.ablation_2x2.run_ablation import RULE_GRAPH

CONDITION_NAMES = tuple(condition.name for condition in CONDITIONS)
STATUSES = ("cleared", "vetoed", "uncertain")
GOLD_SOURCES = ("consensus_original", "deliberated", "rule_extrapolated")
REQUIRED_GOLD_COLUMNS = {
    "audit_id",
    "gold_verdict",
    "gold_violated_rule_ids",
    "gold_source",
    "rater_wjj_verdict",
    "rater_ymh_verdict",
    "initial_agreement",
}
CONTRASTS = {
    "rag_with_validation": ("full", "no_rag"),
    "rag_without_validation": ("no_validation", "baseline"),
    "validation_with_rag": ("full", "no_validation"),
    "validation_without_rag": ("no_rag", "baseline"),
}
_RULE_ID = re.compile(r"R-[A-Z]+-\d+")


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _divide(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_gold(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        actual_columns = set(reader.fieldnames or [])
        if not actual_columns.issuperset(REQUIRED_GOLD_COLUMNS):
            missing = sorted(REQUIRED_GOLD_COLUMNS - actual_columns)
            raise ValueError(f"gold CSV is missing columns: {missing}")
        rows = list(reader)
    ids = [row["audit_id"] for row in rows]
    if len(rows) != 200 or len(ids) != len(set(ids)):
        raise ValueError("gold CSV must contain 200 unique audit_id rows")
    for row in rows:
        if row["gold_verdict"] not in STATUSES:
            raise ValueError(f"invalid gold verdict: {row['audit_id']}")
        if row["gold_source"] not in GOLD_SOURCES:
            raise ValueError(f"invalid gold source: {row['audit_id']}")
        no_rules = row["gold_violated_rule_ids"].strip().upper() == "NONE"
        if (row["gold_verdict"] == "vetoed") == no_rules:
            raise ValueError(f"gold verdict/rule mismatch: {row['audit_id']}")
    return rows


def load_sidecar(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    result = {str(row["audit_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("sidecar contains duplicate audit_id")
    return result


def load_records(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for condition in CONDITION_NAMES:
        path = run_dir / condition / "records.jsonl"
        result[condition] = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        ]
    frame_sets = [
        {str(record["frame_token"]) for record in result[condition]}
        for condition in CONDITION_NAMES
    ]
    if any(frames != frame_sets[0] for frames in frame_sets[1:]):
        raise ValueError("condition records are not paired on the same frame set")
    return result


def _gold_rule_alternatives(value: str) -> list[frozenset[str]]:
    if value.strip().upper() == "NONE":
        return [frozenset()]
    if "ymh:" in value:
        left, right = value.split("ymh:", 1)
        return [frozenset(_RULE_ID.findall(left)), frozenset(_RULE_ID.findall(right))]
    rules = frozenset(_RULE_ID.findall(value))
    if not rules:
        raise ValueError(f"could not parse gold rule IDs: {value}")
    return [rules]


def _prediction_provisions(
    verdict: dict[str, Any], catalog: RuleCatalog
) -> frozenset[str]:
    provisions = {
        catalog.rule_to_provision[rule_id]
        for rule_id in verdict.get("cited_rule_ids", [])
        if rule_id in catalog.rule_to_provision
    }
    for value in verdict.get("cited_provisions", []):
        provisions.update(
            citation.normalized_key
            for citation in extract_provisions(str(value))
            if citation.normalized_key is not None
        )
    return frozenset(provisions)


def build_candidate_rows(
    gold: list[dict[str, str]],
    sidecar: dict[str, dict[str, Any]],
    records_by_condition: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if set(sidecar) != {row["audit_id"] for row in gold}:
        raise ValueError("gold and verdict sidecar audit_id sets differ")
    by_condition = {
        condition: {str(record["frame_token"]): record for record in records}
        for condition, records in records_by_condition.items()
    }
    catalog = RuleCatalog.from_yaml(RULE_GRAPH)
    rows: list[dict[str, Any]] = []
    for gold_row in gold:
        audit_id = gold_row["audit_id"]
        private = sidecar[audit_id]
        frame_token = str(private["frame_token"])
        trajectory_id = str(private["trajectory_id"])
        predictions: dict[str, Any] = {}
        for condition in CONDITION_NAMES:
            record = by_condition[condition].get(frame_token)
            if record is None:
                raise ValueError(f"gold frame missing in {condition}: {audit_id}")
            matched = [
                item
                for item in record["response_final"]["verdicts"]
                if item["trajectory_id"] == trajectory_id
            ]
            if len(matched) != 1:
                raise ValueError(f"candidate match is not unique: {condition}/{audit_id}")
            verdict = matched[0]
            predictions[condition] = {
                "status": str(verdict["status"]),
                "cited_rule_ids": sorted(set(verdict.get("cited_rule_ids", []))),
                "provision_keys": sorted(_prediction_provisions(verdict, catalog)),
            }
        rows.append(
            {
                "audit_id": audit_id,
                "frame_token": frame_token,
                "trajectory_id": trajectory_id,
                "scenario_type": str(private["scenario_type"]),
                "group": str(private.get("group", "")),
                "borderline": bool(private.get("borderline", False)),
                "gold_source": gold_row["gold_source"],
                "gold_verdict": gold_row["gold_verdict"],
                "gold_violated_rule_ids": gold_row["gold_violated_rule_ids"],
                "gold_rule_alternatives": [
                    sorted(item)
                    for item in _gold_rule_alternatives(gold_row["gold_violated_rule_ids"])
                ],
                "predictions": predictions,
            }
        )
    return rows


def classification_metrics(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    expected = [str(row["gold_verdict"]) for row in rows]
    predicted = [str(row["predictions"][condition]["status"]) for row in rows]
    matrix = {
        actual: {guess: 0 for guess in STATUSES}
        for actual in STATUSES
    }
    for actual, guess in zip(expected, predicted, strict=True):
        matrix[actual][guess] += 1
    per_class: dict[str, Any] = {}
    for status in STATUSES:
        true_positive = matrix[status][status]
        false_positive = sum(matrix[actual][status] for actual in STATUSES if actual != status)
        false_negative = sum(matrix[status][guess] for guess in STATUSES if guess != status)
        precision = _divide(true_positive, true_positive + false_positive)
        recall = _divide(true_positive, true_positive + false_negative)
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else 0.0
        )
        per_class[status] = {
            "support": sum(matrix[status].values()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    correct = sum(actual == guess for actual, guess in zip(expected, predicted, strict=True))
    return {
        "n": len(rows),
        "accuracy": correct / len(rows) if rows else None,
        "balanced_accuracy": _mean(per_class[status]["recall"] or 0.0 for status in STATUSES),
        "macro_f1": _mean(per_class[status]["f1"] for status in STATUSES),
        "cohen_kappa": cohen_kappa(expected, predicted),
        "abstention_rate": predicted.count("uncertain") / len(rows) if rows else None,
        "predicted_counts": dict(Counter(predicted)),
        "confusion_matrix": matrix,
        "per_class": per_class,
    }


def _gold_provision_alternatives(
    row: dict[str, Any], catalog: RuleCatalog
) -> list[frozenset[str]]:
    return [
        frozenset(catalog.rule_to_provision[rule_id] for rule_id in rules)
        for rules in row["gold_rule_alternatives"]
    ]


def violated_rule_metrics(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    catalog = RuleCatalog.from_yaml(RULE_GRAPH)
    vetoed = [row for row in rows if row["gold_verdict"] == "vetoed"]
    provision_hits = rule_hits = detected = 0
    rule_eligible = 0
    for row in vetoed:
        prediction = row["predictions"][condition]
        status_hit = prediction["status"] == "vetoed"
        detected += int(status_hit)
        gold_provisions = _gold_provision_alternatives(row, catalog)
        provision_hits += int(
            status_hit and frozenset(prediction["provision_keys"]) in gold_provisions
        )
        if condition in {"full", "no_validation"}:
            rule_eligible += 1
            gold_rules = [frozenset(item) for item in row["gold_rule_alternatives"]]
            rule_hits += int(status_hit and frozenset(prediction["cited_rule_ids"]) in gold_rules)
    return {
        "gold_vetoed": len(vetoed),
        "veto_detected": detected,
        "veto_and_provision_exact_match": provision_hits,
        "veto_and_provision_exact_match_rate": _divide(provision_hits, len(vetoed)),
        "provision_exact_given_detected_veto": _divide(provision_hits, detected),
        "veto_and_rule_exact_match": rule_hits if rule_eligible else None,
        "veto_and_rule_exact_match_rate": _divide(rule_hits, rule_eligible),
    }


def _cluster_bootstrap(
    rows: list[dict[str, Any]],
    statistic: Callable[[list[dict[str, Any]]], float],
    *,
    samples: int = 5000,
    seed: int = 20260807,
) -> list[float]:
    by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_frame[str(row["frame_token"])].append(row)
    frames = sorted(by_frame)
    generator = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        sampled = [frames[generator.randrange(len(frames))] for _ in frames]
        sample_rows = [row for frame in sampled for row in by_frame[frame]]
        estimates.append(statistic(sample_rows))
    return [_percentile(estimates, 0.025), _percentile(estimates, 0.975)]


def _accuracy(rows: list[dict[str, Any]], condition: str) -> float:
    return _mean(
        float(row["predictions"][condition]["status"] == row["gold_verdict"])
        for row in rows
    )


def factorial_effects(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def difference(sample: list[dict[str, Any]], left: str, right: str) -> float:
        return _accuracy(sample, left) - _accuracy(sample, right)

    result: dict[str, Any] = {"paired_contrasts": {}}
    for name, (left, right) in CONTRASTS.items():
        def statistic(
            sample: list[dict[str, Any]], left_name: str = left, right_name: str = right
        ) -> float:
            return difference(sample, left_name, right_name)

        result["paired_contrasts"][name] = {
            "left": left,
            "right": right,
            "accuracy_difference": statistic(rows),
            "cluster_bootstrap_95_ci": _cluster_bootstrap(rows, statistic),
        }

    def rag_effect(sample: list[dict[str, Any]]) -> float:
        return 0.5 * (
            difference(sample, "full", "no_rag")
            + difference(sample, "no_validation", "baseline")
        )

    def validation_effect(sample: list[dict[str, Any]]) -> float:
        return 0.5 * (
            difference(sample, "full", "no_validation")
            + difference(sample, "no_rag", "baseline")
        )

    def interaction(sample: list[dict[str, Any]]) -> float:
        return difference(sample, "full", "no_validation") - difference(
            sample, "no_rag", "baseline"
        )

    for name, statistic in {
        "rag_main_effect": rag_effect,
        "validation_main_effect": validation_effect,
        "rag_x_validation_interaction": interaction,
    }.items():
        result[name] = {
            "accuracy_difference": statistic(rows),
            "cluster_bootstrap_95_ci": _cluster_bootstrap(rows, statistic),
        }
    return result


def _stratified_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    dimensions = {
        "gold_source": lambda row: str(row["gold_source"]),
        "scenario_type": lambda row: str(row["scenario_type"]),
        "group": lambda row: str(row["group"]),
        "borderline": lambda row: str(bool(row["borderline"])).lower(),
    }
    for dimension, value_of in dimensions.items():
        values = sorted({value_of(row) for row in rows})
        for value in values:
            subset = [row for row in rows if value_of(row) == value]
            for condition in CONDITION_NAMES:
                metrics = classification_metrics(subset, condition)
                output.append(
                    {
                        "dimension": dimension,
                        "value": value,
                        "condition": condition,
                        "n": len(subset),
                        "accuracy": metrics["accuracy"],
                        "macro_f1": metrics["macro_f1"],
                        "veto_recall": metrics["per_class"]["vetoed"]["recall"],
                        "abstention_rate": metrics["abstention_rate"],
                    }
                )
    return output


def compute(
    gold: list[dict[str, str]],
    candidate_rows: list[dict[str, Any]],
    *,
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    rater_expected = [row["rater_wjj_verdict"] for row in gold]
    rater_predicted = [row["rater_ymh_verdict"] for row in gold]
    gold_counts = Counter(row["gold_verdict"] for row in gold)
    majority_status, majority_count = gold_counts.most_common(1)[0]
    majority_precision = majority_count / len(gold)
    majority_f1 = 2.0 * majority_precision / (majority_precision + 1.0)
    conditions: dict[str, Any] = {}
    for condition in CONDITION_NAMES:
        metrics = classification_metrics(candidate_rows, condition)
        metrics["accuracy_cluster_bootstrap_95_ci"] = _cluster_bootstrap(
            candidate_rows, lambda sample, name=condition: _accuracy(sample, name)
        )
        metrics["violated_rule_metrics"] = violated_rule_metrics(candidate_rows, condition)
        conditions[condition] = metrics
    return {
        "analysis": "human_gold_verdict_2x2",
        "protocol_version": "ablation_2x2_v1.0",
        "source_sha256": source_hashes,
        "validation": {
            "gold_rows": len(gold),
            "gold_scenes": len({row["frame_token"] for row in candidate_rows}),
            "conditions": len(CONDITION_NAMES),
            "matched_candidates_per_condition": len(candidate_rows),
            "gold_verdict_counts": dict(gold_counts),
            "gold_source_counts": dict(Counter(row["gold_source"] for row in gold)),
            "initial_agreement_flag_count": sum(
                row["initial_agreement"] == "agree" for row in gold
            ),
            "stored_rater_column_agreement": _mean(
                float(left == right)
                for left, right in zip(rater_expected, rater_predicted, strict=True)
            ),
            "stored_rater_column_kappa": cohen_kappa(rater_expected, rater_predicted),
        },
        "reference_baselines": {
            "majority_class": {
                "predicted_status": majority_status,
                "accuracy": majority_count / len(gold),
                "balanced_accuracy": 1.0 / len(STATUSES),
                "macro_f1": majority_f1 / len(STATUSES),
                "veto_recall": 0.0,
                "uncertain_recall": 0.0,
            }
        },
        "conditions": conditions,
        "factorial_effects": factorial_effects(candidate_rows),
        "strata": _stratified_rows(candidate_rows),
        "unavailable_metrics": [
            "preference-pair accuracy",
            "chosen top-1 accuracy",
            "ranking Kendall tau",
            "citation applicability on non-vetoed decisions",
        ],
        "caveats": [
            "Primary confidence intervals resample the 78 annotated frame clusters.",
            "Gold is imbalanced: 160 cleared, 24 uncertain, and 16 vetoed candidates.",
            "The 96 rule_extrapolated rows must be reported separately from direct consensus.",
            "One vetoed row records alternative rater rule IDs; either alternative is accepted.",
            "Preference metrics remain unavailable because this gold contains verdict labels only.",
        ],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _flatten_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        excluded = {"predictions", "gold_rule_alternatives"}
        item = {key: value for key, value in row.items() if key not in excluded}
        for condition in CONDITION_NAMES:
            prediction = row["predictions"][condition]
            item[f"{condition}_verdict"] = prediction["status"]
            item[f"{condition}_correct"] = int(prediction["status"] == row["gold_verdict"])
            item[f"{condition}_cited_rule_ids"] = ";".join(prediction["cited_rule_ids"])
            item[f"{condition}_provision_keys"] = ";".join(prediction["provision_keys"])
        output.append(item)
    return output


def _condition_summary(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in CONDITION_NAMES:
        item = metrics["conditions"][condition]
        rows.append(
            {
                "condition": condition,
                "n": item["n"],
                "accuracy": item["accuracy"],
                "accuracy_ci_low": item["accuracy_cluster_bootstrap_95_ci"][0],
                "accuracy_ci_high": item["accuracy_cluster_bootstrap_95_ci"][1],
                "balanced_accuracy": item["balanced_accuracy"],
                "macro_f1": item["macro_f1"],
                "cohen_kappa": item["cohen_kappa"],
                "veto_precision": item["per_class"]["vetoed"]["precision"],
                "veto_recall": item["per_class"]["vetoed"]["recall"],
                "veto_f1": item["per_class"]["vetoed"]["f1"],
                "uncertain_recall": item["per_class"]["uncertain"]["recall"],
                "abstention_rate": item["abstention_rate"],
                "veto_provision_exact_rate": item["violated_rule_metrics"][
                    "veto_and_provision_exact_match_rate"
                ],
            }
        )
    return rows


def _factorial_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    effects = metrics["factorial_effects"]
    rows = [
        {
            "effect": name,
            "accuracy_difference": item["accuracy_difference"],
            "ci_low": item["cluster_bootstrap_95_ci"][0],
            "ci_high": item["cluster_bootstrap_95_ci"][1],
        }
        for name, item in effects["paired_contrasts"].items()
    ]
    for name in ("rag_main_effect", "validation_main_effect", "rag_x_validation_interaction"):
        item = effects[name]
        rows.append(
            {
                "effect": name,
                "accuracy_difference": item["accuracy_difference"],
                "ci_low": item["cluster_bootstrap_95_ci"][0],
                "ci_high": item["cluster_bootstrap_95_ci"][1],
            }
        )
    return rows


def _show(value: Any) -> str:
    return "—" if value is None else f"{float(value):.3f}"


def report(metrics: dict[str, Any]) -> str:
    validation = metrics["validation"]
    majority = metrics["reference_baselines"]["majority_class"]
    lines = [
        "# 2×2 消融：人工 verdict gold 完整对比",
        "",
        (
            "本报告只使用 `gold_labels_v1` 的 200 条候选级 verdict gold；"
            "四条件在同一 78 个标注场景上严格配对。"
        ),
        "",
        "## 人工标注一致性",
        "",
        (
            f"裁定前原始一致为 {validation['initial_agreement_flag_count']}/"
            f"{validation['gold_rows']} = "
            f"{validation['initial_agreement_flag_count'] / validation['gold_rows']:.1%}；"
            f"Cohen's κ = {validation['stored_rater_column_kappa']:.3f}。"
        ),
        "",
        "## 主结果",
        "",
        (
            "| condition | accuracy [cluster 95% CI] | balanced acc. | macro-F1 | κ | "
            "veto P/R/F1 | uncertain recall |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITION_NAMES:
        item = metrics["conditions"][condition]
        ci = item["accuracy_cluster_bootstrap_95_ci"]
        veto = item["per_class"]["vetoed"]
        lines.append(
            f"| {condition} | {_show(item['accuracy'])} [{_show(ci[0])}, {_show(ci[1])}] | "
            f"{_show(item['balanced_accuracy'])} | {_show(item['macro_f1'])} | "
            f"{_show(item['cohen_kappa'])} | {_show(veto['precision'])}/"
            f"{_show(veto['recall'])}/{_show(veto['f1'])} | "
            f"{_show(item['per_class']['uncertain']['recall'])} |"
        )
    lines.extend(
        [
            "",
            "## Gold 违规条款命中",
            "",
            "分母为16条 gold vetoed；指标要求同时判为 vetoed 且最终条款与 gold 一致。",
            "",
            "| condition | detected veto | exact provision | exact rate | exact given detected |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for condition in CONDITION_NAMES:
        item = metrics["conditions"][condition]["violated_rule_metrics"]
        lines.append(
            f"| {condition} | {item['veto_detected']}/16 | "
            f"{item['veto_and_provision_exact_match']}/16 | "
            f"{_show(item['veto_and_provision_exact_match_rate'])} | "
            f"{_show(item['provision_exact_given_detected_veto'])} |"
        )
    lines.extend(
        [
            "",
            "## 2×2 因子效应（accuracy 差值）",
            "",
            "| effect | difference | cluster 95% CI |",
            "|---|---:|---:|",
        ]
    )
    for row in _factorial_rows(metrics):
        lines.append(
            f"| {row['effect']} | {_show(row['accuracy_difference'])} | "
            f"[{_show(row['ci_low'])}, {_show(row['ci_high'])}] |"
        )
    lines.extend(
        [
            "",
            "## 按 gold 来源分层的 accuracy",
            "",
            "| gold source | n | full | no_validation | no_rag | baseline |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    source_rows = [row for row in metrics["strata"] if row["dimension"] == "gold_source"]
    for source in GOLD_SOURCES:
        items = {row["condition"]: row for row in source_rows if row["value"] == source}
        lines.append(
            f"| {source} | {items['full']['n']} | {_show(items['full']['accuracy'])} | "
            f"{_show(items['no_validation']['accuracy'])} | "
            f"{_show(items['no_rag']['accuracy'])} | {_show(items['baseline']['accuracy'])} |"
        )
    lines.extend(
        [
            "",
            "## Gold 构成与解释边界",
            "",
            (
                "- Gold 分布：160 cleared、24 uncertain、16 vetoed；"
                "accuracy 必须与 macro-F1、veto 指标并列。"
            ),
            (
                f"- 永远预测 cleared 的多数类参照：accuracy={majority['accuracy']:.3f}、"
                f"balanced accuracy={majority['balanced_accuracy']:.3f}、"
                f"macro-F1={majority['macro_f1']:.3f}，且 veto/uncertain recall 均为0。"
            ),
            (
                "- Gold 来源：74 条原始共识、30 条合议、96 条规则外推；"
                "分层结果见 `stratified_metrics.csv`。"
            ),
            "- 当前 gold 不含偏好排序，因此 pair accuracy、top-1 和 Kendall tau 不计算。",
            (
                "- 非 vetoed 候选没有 gold applicable-rule 集合，"
                "不能据此计算完整 citation applicability。"
            ),
            "",
            "## 复现文件",
            "",
            "- `human_gold_metrics.json`：完整指标、混淆矩阵、CI 和 caveats。",
            "- `condition_summary.csv`：四条件主表。",
            "- `factorial_effects.csv`：主效应、条件效应和交互项。",
            "- `stratified_metrics.csv`：gold 来源、场景、抽样组和 borderline 分层。",
            "- `candidate_predictions.csv`：200 条 gold 与四条件逐候选配对结果。",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(
    metrics: dict[str, Any], candidate_rows: list[dict[str, Any]], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "human_gold_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "condition_summary.csv", _condition_summary(metrics))
    _write_csv(output_dir / "factorial_effects.csv", _factorial_rows(metrics))
    _write_csv(output_dir / "stratified_metrics.csv", metrics["strata"])
    _write_csv(output_dir / "candidate_predictions.csv", _flatten_candidate_rows(candidate_rows))
    (output_dir / "human_gold_comparison_report.md").write_text(
        report(metrics), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-csv", type=Path, required=True)
    parser.add_argument("--gold-xlsx", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    gold_path = args.gold_csv.resolve()
    xlsx_path = args.gold_xlsx.resolve()
    sidecar_path = args.sidecar.resolve()
    run_dir = args.run_dir.resolve()
    records = load_records(run_dir)
    gold = load_gold(gold_path)
    candidate_rows = build_candidate_rows(gold, load_sidecar(sidecar_path), records)
    source_hashes = {
        str(gold_path): _sha256(gold_path),
        str(xlsx_path): _sha256(xlsx_path),
        str(sidecar_path): _sha256(sidecar_path),
        **{
            str(run_dir / condition / "records.jsonl"): _sha256(
                run_dir / condition / "records.jsonl"
            )
            for condition in CONDITION_NAMES
        },
    }
    metrics = compute(gold, candidate_rows, source_hashes=source_hashes)
    write_outputs(metrics, candidate_rows, args.output_dir.resolve())
    print(args.output_dir.resolve() / "human_gold_comparison_report.md")


if __name__ == "__main__":
    main()
