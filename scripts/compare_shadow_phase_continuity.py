"""生成 v_a 与 v_b 的并排报告，并追踪指定孤例。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_shadow_no_rule_engine import build_report  # noqa: E402

DEFAULT_BASELINE = ROOT / "results/shadow_no_rule_engine_v2/shadow_verdicts.jsonl"
DEFAULT_TREATMENT = (
    ROOT
    / "results/shadow_no_rule_engine_v2_vb_phase_continuity"
    / "shadow_verdicts.jsonl"
)
DEFAULT_OUTPUT_DIR = ROOT / "results/shadow_no_rule_engine_v2_vb_phase_continuity"


def _load(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _key(record: dict[str, Any]) -> tuple[str, str]:
    return record["frame_token"], record["trajectory_id"]


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--treatment", type=Path, default=DEFAULT_TREATMENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    baseline = _load(args.baseline.resolve())
    treatment = _load(args.treatment.resolve())
    baseline_by_key = {_key(record): record for record in baseline}
    treatment_by_key = {_key(record): record for record in treatment}
    if len(baseline) != 118 or len(treatment) != 118:
        raise ValueError(
            f"两臂都必须完整覆盖 118 条：v_a={len(baseline)}, v_b={len(treatment)}"
        )
    if set(baseline_by_key) != set(treatment_by_key):
        raise ValueError("v_a/v_b 候选键集合不一致")

    report_a = build_report(baseline)
    report_b = build_report(treatment)
    rate_a = report_a["primary_metric"]["rate"]
    rate_b = report_b["primary_metric"]["rate"]
    if report_a["primary_metric"]["agreement"] != 81:
        raise ValueError("v_a 基线不再是预注册的 81/118")

    transitions = Counter(
        (
            baseline_by_key[key]["shadow_status"],
            treatment_by_key[key]["shadow_status"],
        )
        for key in baseline_by_key
    )
    isolated = [
        record
        for record in baseline
        if record["predicate_label"] == "vetoed"
        and record["shadow_status"] == "cleared"
    ]
    if len(isolated) != 1:
        raise ValueError(f"v_a 指定孤例应为 1 条，实际 {len(isolated)}")
    isolated_a = isolated[0]
    isolated_b = treatment_by_key[_key(isolated_a)]

    red_crossing_keys = {
        _key(record)
        for record in baseline
        if record["predicate_reason"]
        == "crossed_governing_line_during_valid_red_without_full_stop"
    }
    red_crossing = {
        "candidates": len(red_crossing_keys),
        "v_a_statuses": dict(
            Counter(baseline_by_key[key]["shadow_status"] for key in red_crossing_keys)
        ),
        "v_b_statuses": dict(
            Counter(treatment_by_key[key]["shadow_status"] for key in red_crossing_keys)
        ),
    }
    degradations = []
    for key in sorted(baseline_by_key):
        record_a = baseline_by_key[key]
        record_b = treatment_by_key[key]
        if (
            record_a["shadow_status"] == record_a["predicate_label"]
            and record_b["shadow_status"] != record_b["predicate_label"]
        ):
            degradations.append(
                {
                    "frame_token": record_b["frame_token"],
                    "trajectory_id": record_b["trajectory_id"],
                    "scenario_type": record_b["scenario_type"],
                    "v_a_status": record_a["shadow_status"],
                    "v_b_status": record_b["shadow_status"],
                    "predicate_label": record_b["predicate_label"],
                    "predicate_reason": record_b["predicate_reason"],
                    "v_a_reason": record_a["shadow_reason"],
                    "v_b_reason": record_b["shadow_reason"],
                }
            )
    comparison = {
        "v_a": report_a,
        "v_b": report_b,
        "primary_metric": {
            "v_a": rate_a,
            "v_b": rate_b,
            "absolute_difference": rate_b - rate_a,
            "percentage_point_difference": (rate_b - rate_a) * 100,
            "information_value_definition": (
                "v_b 与 v_a 的 LLM–谓词总一致率百分点差；两臂只改变相位持续性声明。"
            ),
        },
        "status_transitions": {
            f"{old}_to_{new}": count
            for (old, new), count in sorted(transitions.items())
        },
        "red_light_crossing_predicate": red_crossing,
        "degradations": {
            "definition": "v_a 与谓词一致、但 v_b 与谓词不一致的候选",
            "count": len(degradations),
            "items": degradations,
        },
        "isolated_predicate_vetoed_to_va_cleared": {
            "frame_token": isolated_a["frame_token"],
            "trajectory_id": isolated_a["trajectory_id"],
            "scenario_type": isolated_a["scenario_type"],
            "predicate_reason": isolated_a["predicate_reason"],
            "v_a_status": isolated_a["shadow_status"],
            "v_a_reason": isolated_a["shadow_reason"],
            "v_b_status": isolated_b["shadow_status"],
            "v_b_reason": isolated_b["shadow_reason"],
        },
        "preregistered_expectation_met": rate_b >= 0.9,
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison_report.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metric = comparison["primary_metric"]
    isolated_out = comparison["isolated_predicate_vetoed_to_va_cleared"]
    false_veto = report_b["directional_errors"]["false_veto"]
    markdown_lines = [
            "# 相位持续性措辞对照：v_a vs v_b",
            "",
            "| 版本 | 唯一措辞差异 | 一致数 | 总数 | 一致率 |",
            "|---|---|---:|---:|---:|",
            f"| v_a | keyframe 后相位未知 | {report_a['primary_metric']['agreement']} | 118 | {_pct(rate_a)} |",
            f"| v_b | `t∈[0, 2.0 s]` 相位持续有效 | {report_b['primary_metric']['agreement']} | 118 | {_pct(rate_b)} |",
            "",
            f"一句话的信息传达价值：**{metric['percentage_point_difference']:+.1f} 个百分点**。",
            "",
            "## 预注册对照",
            "",
            f"- 预期：v_b 达到 90% 以上；结果：{'符合' if rate_b >= 0.9 else '未符合'}。",
            f"- 红灯越线谓词 38 条：v_a `{red_crossing['v_a_statuses']}`；v_b `{red_crossing['v_b_statuses']}`。",
            "",
            "## 指定孤例",
            "",
            f"- `{isolated_out['frame_token']}/{isolated_out['trajectory_id']}`："
            f"v_a `{isolated_out['v_a_status']}` → v_b `{isolated_out['v_b_status']}`。",
            "",
            "## v_b 完整混淆矩阵",
            "",
            "| 谓词标签 \\ v_b 判定 | cleared | vetoed | uncertain | 合计 |",
            "|---|---:|---:|---:|---:|",
    ]
    confusion = report_b["confusion_matrix"]
    for predicate in ("cleared", "vetoed"):
        counts = [
            confusion.get(f"predicate_{predicate}|shadow_{shadow}", 0)
            for shadow in ("cleared", "vetoed", "uncertain")
        ]
        markdown_lines.append(
            f"| {predicate} | {counts[0]} | {counts[1]} | {counts[2]} | {sum(counts)} |"
        )
    markdown_lines += [
        "",
        "## v_b 按谓词原因分层",
        "",
        "| 分层 | n | 一致率 | 弃权率 |",
        "|---|---:|---:|---:|",
    ]
    for reason, item in report_b["by_predicate_reason"].items():
        markdown_lines.append(
            f"| `{reason}` | {item['n']} | "
            f"{_pct(item['agreement_rate'])} | {_pct(item['abstention_rate'])} |"
        )
    markdown_lines += [
        "",
        "## v_b false_veto",
        "",
        f"- **{false_veto['count']}/{false_veto['denominator']} = "
        f"{_pct(false_veto['rate'])}**：{false_veto['note']}。",
        "",
        "## 劣化条目",
        "",
        f"定义：{comparison['degradations']['definition']}；实际 **{len(degradations)} 条**。",
        "",
        "| 候选 | 场景 | v_a 判定 | v_b 判定 | 谓词标签 | 谓词原因 |",
        "|---|---|---|---|---|---|",
    ]
    for item in degradations:
        markdown_lines.append(
            f"| `{item['frame_token']}/{item['trajectory_id']}` | "
            f"{item['scenario_type']} | {item['v_a_status']} | {item['v_b_status']} | "
            f"{item['predicate_label']} | `{item['predicate_reason']}` |"
        )
    markdown_lines.append("")
    markdown = "\n".join(markdown_lines)
    (output_dir / "comparison_report.md").write_text(markdown, encoding="utf-8")

    hashes_path = output_dir / "artifact_hashes.json"
    hashes = (
        json.loads(hashes_path.read_text(encoding="utf-8"))
        if hashes_path.exists()
        else {"artifacts": {}}
    )
    artifact_names = (
        "comparison_report.json",
        "comparison_report.md",
        "evidence_chain/README.md",
        "evidence_chain/review_manifest.json",
        "evidence_chain/red_light_abstention_review.jsonl",
    )
    for name in artifact_names:
        path = output_dir / name
        if path.exists():
            hashes["artifacts"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    hashes_path.write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metric, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
