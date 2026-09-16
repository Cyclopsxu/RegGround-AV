"""归档相位措辞实验的逐条机理复核，并正式关闭评估臂。"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPARISON = (
    ROOT
    / "results/shadow_no_rule_engine_v2_vb_phase_continuity"
    / "comparison_report.json"
)
DEFAULT_OUTPUT_DIR = DEFAULT_COMPARISON.parent / "mechanism_review"

FRAME_0225 = "022558873876467c8d467b83a6db30dd"
FRAME_7DAD = "7dad3eeefb7b4a008b6c82b0a15354b7"
FRAME_BC73 = "bc730f2a992246f1831e29a4f02d0380"
ISOLATED_KEY = ("575b55e43b014b17850144a75d9b0424", "traj_a")

ASSESSMENTS: dict[tuple[str, str], dict[str, str]] = {
    (FRAME_0225, "traj_a"): {
        "mechanism": "unscoped_epistemic_strictness_spillover",
        "relation": "same_mechanism",
        "confidence": "high",
        "rationale": (
            "v_b 在纯行人规则上新增了‘必须有 0.0–0.5 s 逐帧速度’的证明要求；"
            "相位声明与该义务无关，却改变了可接受证据标准。"
        ),
    },
    (FRAME_0225, "traj_b"): {
        "mechanism": "unscoped_epistemic_strictness_spillover",
        "relation": "same_mechanism",
        "confidence": "high",
        "rationale": (
            "v_b 承认停车与无共同冲突区，却因缺少停车时点的直接证明而弃权，"
            "符合全局认识论框架向行人语义外溢。"
        ),
    },
    (FRAME_0225, "traj_c"): {
        "mechanism": "unscoped_epistemic_strictness_spillover",
        "relation": "same_mechanism",
        "confidence": "high",
        "rationale": (
            "v_b 从 3.1 s 正时间间隔与停车事实仍升级到逐帧证据要求，"
            "是与 traj_a 同构的跨语义证据标准收紧。"
        ),
    },
    (FRAME_0225, "traj_f"): {
        "mechanism": "stop_line_salience_spillover_into_pedestrian_rule",
        "relation": "same_mechanism_family",
        "confidence": "medium",
        "rationale": (
            "v_b 把停止线区域侵入直接解释成侵入行人横道并据此否决；"
            "这是新声明提高停止线概念显著性后出现的跨规则几何混同。"
        ),
    },
    (FRAME_7DAD, "traj_d"): {
        "mechanism": "rule_priority_or_decoding_variation",
        "relation": "independent_noise_more_likely",
        "confidence": "medium",
        "rationale": (
            "v_b reason 未引用相位、时效或证据可知性，而是把‘移动行人’解释成"
            "无条件停车义务并忽略无共同冲突区，更像规则优先级波动。"
        ),
    },
    (FRAME_BC73, "traj_f"): {
        "mechanism": "in_scope_epistemic_overreach",
        "relation": "same_mechanism",
        "confidence": "medium_high",
        "rationale": (
            "v_b 明确引用新增的 [0,2.0 s] 框架，却拒绝组合‘起始距线 5.6 m’与"
            "‘总行程 2.1 m’这一充分的间接不越线证明；不是缺数据，而是证据标准"
            "被提升为必须直接报告越线状态。"
        ),
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    comparison_path = args.comparison.resolve()
    output_dir = args.output_dir.resolve()
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    degradations = comparison["degradations"]["items"]
    actual_keys = {(item["frame_token"], item["trajectory_id"]) for item in degradations}
    if actual_keys != set(ASSESSMENTS):
        raise ValueError("劣化集合与已复核的 6 条记录不一致，拒绝静默归档")

    reviewed: list[dict[str, Any]] = []
    for item in degradations:
        key = (item["frame_token"], item["trajectory_id"])
        reviewed.append({**item, **ASSESSMENTS[key]})

    isolated = comparison["isolated_predicate_vetoed_to_va_cleared"]
    if (isolated["frame_token"], isolated["trajectory_id"]) != ISOLATED_KEY:
        raise ValueError("指定孤例发生漂移")
    isolated_review = {
        **isolated,
        "mechanism": "persistent_predicate_llm_interpretation_gap",
        "relation": "not_a_wording_side_effect",
        "confidence": "high",
        "rationale": (
            "v_a 与 v_b 均 cleared，且两版都把最低速度 0 与短暂停车解释成已让行；"
            "它是停车位置/时序解释与几何谓词的稳定分歧。"
        ),
    }

    conclusion = {
        "universal_claim_validated": False,
        "universal_claim_note": (
            "单次配对实验不能证明‘任何全局认识论措辞必然产生副作用’这一全称命题。"
        ),
        "narrower_mechanism_supported": True,
        "narrower_mechanism": (
            "在当前 prompt 结构中，无作用域的全局时效/认识论声明会重加权 LLM 对"
            "证据充分性的要求与停止线概念显著性，并可能外溢到行人规则。"
        ),
        "evidence_strength": "medium",
        "supporting_records": 5,
        "independent_noise_records": 1,
        "bc73_assessment": "same_mechanism",
        "operational_decision": (
            "副作用风险已足以否决继续措辞迭代；评估臂关闭，由人工盲标仲裁敏感样本。"
        ),
    }

    archived_at = datetime.now(UTC).isoformat()
    archive = {
        "source": str(comparison_path),
        "source_sha256": _sha256(comparison_path),
        "archived_at": archived_at,
        "hypothesis_review": conclusion,
        "degradations": reviewed,
        "isolated_case": isolated_review,
    }
    closure = {
        "evaluation_arm": "shadow_no_rule_engine_v2_phase_wording",
        "status": "closed",
        "closed_at": archived_at,
        "v_a_retained": True,
        "v_b_retained": True,
        "v_c_authorized": False,
        "no_v_c": True,
        "production_changed": False,
        "next_step": "human_blind_adjudication_of_wording_sensitive_samples",
        "reason": conclusion["operational_decision"],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "mechanism_review.json"
    closure_path = output_dir / "evaluation_arm_closure.json"
    _write_json(archive_path, archive)
    _write_json(closure_path, closure)

    lines = [
        "# 相位措辞实验：机理复核与评估臂关闭",
        "",
        "## 结论",
        "",
        f"- 全称猜想是否被验证：否。{conclusion['universal_claim_note']}",
        f"- 较窄机理是否得到支持：是（证据强度：{conclusion['evidence_strength']}）。",
        f"- 较窄机理：{conclusion['narrower_mechanism']}",
        f"- `bc73…/traj_f`：{conclusion['bc73_assessment']}，不是优先按独立噪声处理。",
        f"- 操作决定：{conclusion['operational_decision']}",
        "",
        "## 六条劣化的逐条分类",
        "",
        "| 候选 | v_a → v_b | 谓词原因 | 机理关系 | 置信度 |",
        "|---|---|---|---|---|",
    ]
    for item in reviewed:
        lines.append(
            f"| `{item['frame_token']}/{item['trajectory_id']}` | "
            f"{item['v_a_status']} → {item['v_b_status']} | "
            f"`{item['predicate_reason']}` | {item['relation']} | "
            f"{item['confidence']} |"
        )
    lines += ["", "## reason 原文与判读", ""]
    for index, item in enumerate(reviewed, start=1):
        lines += [
            f"### {index}. `{item['frame_token']}/{item['trajectory_id']}`",
            "",
            f"- v_a（`{item['v_a_status']}`）原文：{item['v_a_reason']}",
            f"- v_b（`{item['v_b_status']}`）原文：{item['v_b_reason']}",
            f"- 判读：{item['rationale']}",
            "",
        ]
    lines += [
        "## 指定孤例",
        "",
        f"- 候选：`{isolated['frame_token']}/{isolated['trajectory_id']}`",
        f"- v_a（`{isolated['v_a_status']}`）原文：{isolated['v_a_reason']}",
        f"- v_b（`{isolated['v_b_status']}`）原文：{isolated['v_b_reason']}",
        f"- 判读：{isolated_review['rationale']}",
        "",
        "## 正式关闭声明",
        "",
        "本评估臂状态为 **closed**；保留 v_a 与 v_b，并明确 **不产生 v_c**。",
        "后续只进行人类盲标仲裁，不再用新的 prompt 措辞追逐指标。",
        "",
    ]
    markdown_path = output_dir / "MECHANISM_REVIEW_AND_CLOSURE.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")

    hashes_path = comparison_path.parent / "artifact_hashes.json"
    hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
    for path in (archive_path, closure_path, markdown_path):
        name = str(path.relative_to(comparison_path.parent))
        hashes["artifacts"][name] = _sha256(path)
    _write_json(hashes_path, hashes)
    print(json.dumps(conclusion, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
