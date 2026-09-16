"""回收校验：标注者交回的工作簿在进入 IAA 与揭盲评分前必须先过这一关。

对应修改意见 P1-2。校验项：

1. 行数与 audit_id 集合、顺序与模板完全一致；
2. human_verdict ∈ {cleared, vetoed, uncertain}，无空值；
3. rule ID 归一化（``R-YLD-1`` → ``R-YLD-01``、中文分号/逗号/顿号统一）后必须在规则卡内；
4. verdict 与 violated 的一致性：cleared/uncertain 必须 NONE，vetoed 至少一条；
5. vetoed 行引用的规则必须全是 hard；
6. confidence ∈ [1,5] 整数，reason 非空。

不通过的行会生成逐行退回清单。解析层与后续 IAA 计算共用，属提前投资。

用法::

    python human_annotation_v2/code/validate_returns.py \\
        --input human_annotation_v2/results/rater_a_completed.xlsx \\
        --template human_annotation_v2/results/blind_annotation_template.xlsx \\
        --output human_annotation_v2/results/return_validation_rater_a.json
"""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402

from xlsx_writer import read_workbook  # noqa: E402

RULE_GRAPH = ROOT / "data/layer2/rule_graph.yaml"
ANNOTATION_SHEET = "标注"
VERDICTS = {"cleared", "vetoed", "uncertain"}
PREFERENCE_SHEET_COLUMNS = {"human_best_trajectory", "human_ranking"}

_SEPARATORS = re.compile(r"[；;，,、\s]+")
_RULE_PATTERN = re.compile(r"^R-([A-Za-z]+)-(\d+)$")


def load_rule_cards() -> dict[str, str]:
    """返回 rule_id -> severity。"""
    data = yaml.safe_load(RULE_GRAPH.read_text(encoding="utf-8"))
    return {str(rule["id"]): str(rule["severity"]) for rule in data["rules"]}


def normalize_rule_ids(
    value: str, allowed: dict[str, str]
) -> tuple[list[str], str, list[str]]:
    """归一化规则 ID 列，返回 (ids, canonical, errors)。"""
    raw = (value or "").strip()
    if not raw or raw.upper() == "NONE":
        return [], "NONE", []

    tokens = [item for item in _SEPARATORS.split(raw) if item]
    errors: list[str] = []
    ids: list[str] = []
    for token in tokens:
        if token.upper() == "NONE":
            errors.append("NONE 不能与其他 rule ID 混用")
            continue
        match = _RULE_PATTERN.match(token)
        canonical = (
            f"R-{match.group(1).upper()}-{int(match.group(2)):02d}"
            if match
            else token.upper()
        )
        if canonical not in allowed:
            errors.append(f"非法或未知 rule ID: {token}")
            continue
        if canonical in ids:
            errors.append(f"重复 rule ID: {canonical}")
            continue
        ids.append(canonical)
    return ids, ";".join(ids) if ids else "NONE", errors


def validate_row(
    row: dict[str, str], allowed: dict[str, str]
) -> tuple[dict[str, Any], list[str]]:
    """校验一行 verdict 标注，返回归一化结果与问题清单。"""
    issues: list[str] = []
    verdict = (row.get("human_verdict") or "").strip().lower()
    if verdict not in VERDICTS:
        issues.append("human_verdict 必须是 cleared/vetoed/uncertain")

    applicable_raw = row.get("human_applicable_rule_ids", "")
    violated_raw = row.get("human_violated_rule_ids", "")
    if not applicable_raw.strip():
        issues.append("human_applicable_rule_ids 不能为空；无适用规则填 NONE")
    if not violated_raw.strip():
        issues.append("human_violated_rule_ids 不能为空；无违反规则填 NONE")
    applicable_ids, applicable_text, applicable_errors = normalize_rule_ids(
        applicable_raw, allowed
    )
    violated_ids, violated_text, violated_errors = normalize_rule_ids(
        violated_raw, allowed
    )
    issues += [f"applicable: {item}" for item in applicable_errors]
    issues += [f"violated: {item}" for item in violated_errors]

    if verdict in {"cleared", "uncertain"} and violated_ids:
        issues.append(f"{verdict} 的 human_violated_rule_ids 必须为 NONE")
    if verdict == "vetoed":
        if not violated_ids:
            issues.append("vetoed 至少需要一条 violated rule ID")
        non_hard = [rid for rid in violated_ids if allowed.get(rid) != "hard"]
        if non_hard:
            issues.append(f"vetoed 的 violated rule 必须全为 hard：{non_hard}")

    confidence = (row.get("human_confidence") or "").strip()
    if not re.fullmatch(r"[1-5]", confidence):
        issues.append("human_confidence 必须是 1–5 的整数")

    if not (row.get("human_reason") or "").strip():
        issues.append("human_reason 不能为空")

    return {
        "human_verdict": verdict,
        "human_applicable_rule_ids": applicable_text,
        "human_violated_rule_ids": violated_text,
        "human_confidence": confidence,
        "human_reason": (row.get("human_reason") or "").strip(),
        "notes": (row.get("notes") or "").strip(),
    }, issues


def validate_preference_row(
    row: dict[str, str], candidate_columns: list[str]
) -> tuple[dict[str, Any], list[str]]:
    """校验一行偏好标注：best 单选合法、ranking 覆盖全部候选。"""
    issues: list[str] = []
    valid = set(candidate_columns)

    best = (row.get("human_best_trajectory") or "").strip()
    if not best:
        issues.append("human_best_trajectory 不能为空")
    elif best.upper() != "NONE" and best not in valid:
        issues.append(f"human_best_trajectory 不是本行候选：{best}")

    ranking_raw = (row.get("human_ranking") or "").strip()
    ranked: list[list[str]] = []
    if not ranking_raw:
        issues.append("human_ranking 不能为空")
    else:
        for group in ranking_raw.split(">"):
            tier = [item.strip() for item in group.split("=") if item.strip()]
            if not tier:
                issues.append("human_ranking 存在空的名次组")
            ranked.append(tier)
        flat = [item for tier in ranked for item in tier]
        unknown = [item for item in flat if item not in valid]
        if unknown:
            issues.append(f"human_ranking 含未知候选：{unknown}")
        if len(flat) != len(set(flat)):
            issues.append("human_ranking 存在重复候选")
        if set(flat) != valid:
            missing = sorted(valid - set(flat))
            if missing:
                issues.append(f"human_ranking 未覆盖全部候选，缺少：{missing}")
    if best and best.upper() != "NONE" and ranked and best not in ranked[0]:
        issues.append("human_best_trajectory 必须出现在 human_ranking 的第一名次组")

    confidence = (row.get("human_confidence") or "").strip()
    if not re.fullmatch(r"[1-5]", confidence):
        issues.append("human_confidence 必须是 1–5 的整数")
    if not (row.get("human_reason") or "").strip():
        issues.append("human_reason 不能为空")

    return {
        "human_best_trajectory": best,
        "human_ranking": ranking_raw,
        "human_confidence": confidence,
        "human_reason": (row.get("human_reason") or "").strip(),
    }, issues


def _sheet_as_dicts(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    workbook = read_workbook(path)
    if ANNOTATION_SHEET not in workbook:
        raise ValueError(f"{path} 缺少「{ANNOTATION_SHEET}」sheet")
    rows = workbook[ANNOTATION_SHEET]
    if not rows:
        raise ValueError(f"{path} 的标注 sheet 为空")
    header = rows[0]
    records = []
    for row in rows[1:]:
        padded = list(row) + [""] * (len(header) - len(row))
        records.append(dict(zip(header, padded, strict=False)))
    return header, records


def validate_return(input_path: Path, template_path: Path) -> dict[str, Any]:
    """比对回收文件与模板，逐行校验并生成退回清单。"""
    template_header, template_rows = _sheet_as_dicts(template_path)
    header, rows = _sheet_as_dicts(input_path)

    structure_issues: list[str] = []
    if header != template_header:
        structure_issues.append(
            f"表头与模板不一致：模板 {template_header} / 回收 {header}"
        )
    if len(rows) != len(template_rows):
        structure_issues.append(
            f"行数 {len(rows)} 与模板 {len(template_rows)} 不一致"
        )

    template_ids = [row.get("audit_id", "") for row in template_rows]
    actual_ids = [row.get("audit_id", "") for row in rows]
    if actual_ids != template_ids:
        if set(actual_ids) == set(template_ids):
            structure_issues.append("audit_id 顺序被修改")
        else:
            missing = sorted(set(template_ids) - set(actual_ids))
            extra = sorted(set(actual_ids) - set(template_ids))
            structure_issues.append(
                f"audit_id 集合不一致：缺失 {missing[:10]} 多余 {extra[:10]}"
            )

    is_preference = PREFERENCE_SHEET_COLUMNS.issubset(set(template_header))
    candidate_columns = [name for name in template_header if name.startswith("cand_")]
    protected_columns = [
        name
        for name in template_header
        if not name.startswith("human_") and name != "notes"
    ]
    allowed = load_rule_cards()

    normalized: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        audit_id = row.get("audit_id", f"<row {index + 2}>")
        if is_preference:
            columns = [
                name
                for name in candidate_columns
                if (template_rows[index].get(name) or "").strip()
            ] if index < len(template_rows) else candidate_columns
            values, issues = validate_preference_row(row, columns)
        else:
            values, issues = validate_row(row, allowed)
        if index < len(template_rows):
            modified = [
                name
                for name in protected_columns
                if row.get(name, "") != template_rows[index].get(name, "")
            ]
            if modified:
                issues.append(f"只读证据列被修改：{modified}")
        values["audit_id"] = audit_id
        normalized.append(values)
        if issues:
            rejects.append({"audit_id": audit_id, "row": index + 2, "issues": issues})

    return {
        "ok": not structure_issues and not rejects,
        "input": str(input_path),
        "template": str(template_path),
        "sheet_kind": "preference" if is_preference else "verdict",
        "row_count": len(rows),
        "structure_issues": structure_issues,
        "reject_count": len(rejects),
        "rejects": rejects,
        "normalized": normalized,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="校验标注者交回的工作簿。")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_return(args.input.resolve(), args.template.resolve())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    summary = {
        key: report[key]
        for key in ("ok", "sheet_kind", "row_count", "structure_issues", "reject_count")
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for reject in report["rejects"][:20]:
        print(f"  {reject['audit_id']} (行 {reject['row']}): {'; '.join(reject['issues'])}")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
