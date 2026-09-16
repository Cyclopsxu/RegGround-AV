"""导出 5 条措辞敏感候选的补充盲标批。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from export_human_audit import (  # noqa: E402
    _assert_evidence_verbatim,
    _assert_workbook_roundtrip,
    assert_no_leakage,
    build_sidecar,
    build_verdict_workbook,
    load_rule_cards,
)
from sampling import SampleRow, build_frame, order_by_hash  # noqa: E402
from xlsx_writer import write_workbook  # noqa: E402

from src.replay_context import (  # noqa: E402
    DEFAULT_EVAL_SET,
    DEFAULT_INPUT,
    DEFAULT_RUN_LOG,
    file_sha256,
    load_run_manifest,
    replay_scenes,
)

FRAME_0225 = "022558873876467c8d467b83a6db30dd"
FRAME_575B = "575b55e43b014b17850144a75d9b0424"
SUPPLEMENT_KEYS = (
    (FRAME_0225, "traj_a"),
    (FRAME_0225, "traj_b"),
    (FRAME_0225, "traj_c"),
    (FRAME_0225, "traj_f"),
    (FRAME_575B, "traj_a"),
)
SALT = "wording-sensitive-supplement:v1"
DEFAULT_RESULTS_DIR = (
    ROOT
    / "human_annotation_v2/results/outputs/wording_sensitive_supplement_v1"
)
DEFAULT_PRIVATE_DIR = (
    ROOT
    / "human_annotation_v2/private/outputs/wording_sensitive_supplement_v1"
)
DEFAULT_VA = ROOT / "results/shadow_no_rule_engine_v2/shadow_verdicts.jsonl"
DEFAULT_VB = (
    ROOT
    / "results/shadow_no_rule_engine_v2_vb_phase_continuity/shadow_verdicts.jsonl"
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _selected_shadow_fields(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": record["shadow_status"],
        "reason": record["shadow_reason"],
        "violated_rule_ids": record["shadow_violated_rule_ids"],
        "prompt_sha256": record["prompt_sha256"],
    }


def export(args: argparse.Namespace) -> dict[str, Any]:
    scenes_list = replay_scenes(
        run_log=args.run_log.resolve(),
        input_path=args.input.resolve(),
        eval_set_path=args.eval_set.resolve(),
    )
    scenes = {scene.frame_token: scene for scene in scenes_list}
    candidates = {candidate.key: candidate for candidate in build_frame(scenes_list)}
    missing = set(SUPPLEMENT_KEYS) - set(candidates)
    if missing:
        raise ValueError(f"补充批候选不在冻结抽样框中：{sorted(missing)}")

    rows = [
        SampleRow(
            candidate=candidates[key],
            group="wording_sensitive_supplement",
            stratum=(
                "wording_sensitive|degradation"
                if key[0] == FRAME_0225
                else "three_judge_disagreement|isolated"
            ),
        )
        for key in SUPPLEMENT_KEYS
    ]
    rows = order_by_hash(rows, salt=SALT, key=lambda row: row.candidate.key)
    audit_ids = {
        row.candidate.key: f"S-{index:03d}"
        for index, row in enumerate(rows, start=1)
    }

    instructions, annotation, blind_records = build_verdict_workbook(
        rows,
        scenes,
        load_rule_cards(),
        audit_ids=audit_ids,
    )
    sidecar = build_sidecar(rows, scenes, audit_ids=audit_ids)

    va_path = args.v_a.resolve()
    vb_path = args.v_b.resolve()
    va = {(r["frame_token"], r["trajectory_id"]): r for r in _load_jsonl(va_path)}
    vb = {(r["frame_token"], r["trajectory_id"]): r for r in _load_jsonl(vb_path)}
    for record in sidecar:
        key = (record["frame_token"], record["trajectory_id"])
        if key not in va or key not in vb:
            raise ValueError(f"补充批候选缺少 v_a/v_b 影子判定：{key}")
        record["selection_reason"] = (
            "v_a_correct_v_b_incorrect_wording_sensitive"
            if key[0] == FRAME_0225
            else "predicate_vetoed_but_v_a_and_v_b_cleared"
        )
        record["shadow_v_a"] = _selected_shadow_fields(va[key])
        record["shadow_v_b"] = _selected_shadow_fields(vb[key])
        record["predicate_reference"] = {
            "label": vb[key]["predicate_label"],
            "reason": vb[key]["predicate_reason"],
        }
        record["human_adjudication"] = "pending"
    assert_no_leakage(blind_records, sidecar)

    results_dir = args.results_dir.resolve()
    private_dir = args.private_dir.resolve()
    preview_dir = private_dir / "qa_previews"
    results_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    workbook_path = results_dir / "supplemental_blind_annotation_template.xlsx"
    write_workbook(
        workbook_path,
        [instructions, annotation],
        node_executable=args.artifact_node.resolve(),
        node_modules=args.artifact_node_modules.resolve(),
        preview_dir=preview_dir,
    )
    roundtrip = _assert_workbook_roundtrip(
        workbook_path,
        [instructions, annotation],
    )
    evidence_check = _assert_evidence_verbatim(rows, scenes)

    blind_path = results_dir / "supplemental_blind_samples.jsonl"
    sidecar_path = private_dir / "supplemental_hidden_sidecar.jsonl"
    _write_jsonl(blind_path, blind_records)
    _write_jsonl(sidecar_path, sidecar)

    run_manifest = load_run_manifest(args.run_log.resolve())
    existing_occurrences = {
        "main_verdict": [f"{FRAME_0225}/traj_f"],
        "calibration": [f"{FRAME_0225}/traj_b", f"{FRAME_575B}/traj_a"],
        "note": "按用户要求有意重复纳入补充批；新 audit_id 不表示独立新样本。",
    }
    manifest = {
        "package_version": "wording_sensitive_supplement_v1",
        "purpose": (
            "人工仲裁 v_a/v_b/谓词分歧；盲标工作簿不暴露选择原因或机器判定。"
        ),
        "generated_from": {
            "run_log": str(args.run_log.resolve()),
            "run_log_sha256": file_sha256(args.run_log.resolve()),
            "run_git_commit": run_manifest.get("git_commit"),
            "input_sha256": run_manifest.get("input_sha256"),
            "eval_set_sha256": (run_manifest.get("eval_set") or {}).get("sha256"),
            "v_a_sha256": file_sha256(va_path),
            "v_b_sha256": file_sha256(vb_path),
        },
        "selection": {
            "count": len(rows),
            "keys": [list(key) for key in SUPPLEMENT_KEYS],
            "frame_distribution": dict(
                Counter(row.candidate.frame_token for row in rows)
            ),
            "ordering": f"SHA-256 seed=42 salt={SALT}",
            "existing_occurrences": existing_occurrences,
        },
        "evidence_contract": {
            "source": "production JudgeInput v2 replay; v_a wording, not v_b assumption",
            "verbatim_check": evidence_check,
            "leakage_scan_passed": True,
        },
        "workbook_roundtrip_check": roundtrip,
        "send_to_annotator": [workbook_path.name],
        "do_not_send": [blind_path.name, sidecar_path.name, "supplement_manifest.json"],
    }
    manifest_path = results_dir / "supplement_manifest.json"
    _write_json(manifest_path, manifest)

    artifacts = (workbook_path, blind_path, manifest_path, sidecar_path)
    hashes = {
        "artifacts": {
            str(path.relative_to(ROOT)): file_sha256(path) for path in artifacts
        }
    }
    hashes["package_digest"] = hashlib.sha256(
        json.dumps(hashes, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    _write_json(results_dir / "artifact_hashes.json", hashes)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--private-dir", type=Path, default=DEFAULT_PRIVATE_DIR)
    parser.add_argument("--run-log", type=Path, default=ROOT / DEFAULT_RUN_LOG)
    parser.add_argument("--input", type=Path, default=ROOT / DEFAULT_INPUT)
    parser.add_argument("--eval-set", type=Path, default=ROOT / DEFAULT_EVAL_SET)
    parser.add_argument("--v-a", type=Path, default=DEFAULT_VA)
    parser.add_argument("--v-b", type=Path, default=DEFAULT_VB)
    parser.add_argument("--artifact-node", type=Path, required=True)
    parser.add_argument("--artifact-node-modules", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    manifest = export(parse_args())
    print(json.dumps(manifest["selection"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
