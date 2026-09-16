"""生成盲标包 v2：verdict 表、场景级偏好表、校准表与揭盲 sidecar。

设计红线：

1. 证据区逐字来自渲染后的 JudgeInput v2。四个证据列分别是
   ``scene_facts_text`` / ``hard_rules_text`` / ``scene_narrative`` /
   ``render_feature_text(feature)``，全部由 :mod:`src.replay_context` 从冻结输入
   重放得到，并用运行日志里的 ``feature_text_hash`` 逐条校验过。
   这四段文本本身不含轨迹 ID（``[traj_x]`` 前缀是 HardFilter 另行拼接的），
   因此"逐字一致"与"盲标隔离"可以同时成立。
2. 工作簿不出现系统 verdict、chosen、ranking、benchmark 标签、变体类型、
   分组归属或 decided_by。这些只写进 ``private/`` 下的 sidecar。
3. 压力案例与主样本统一混洗后编号（修改意见 P1-1），尾部聚集不复现。

用法::

    python human_annotation_v2/code/export_human_audit.py
    python human_annotation_v2/code/export_human_audit.py --output-root human_annotation_v2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sampling import (  # noqa: E402
    SampleRow,
    order_by_hash,
    run_sampling,
)
from xlsx_writer import (  # noqa: E402
    STYLE_EXAMPLE,
    Sheet,
    read_workbook,
    write_workbook,
)

from src.replay_context import (  # noqa: E402
    DEFAULT_EVAL_SET,
    DEFAULT_INPUT,
    DEFAULT_RUN_LOG,
    ReplayedScene,
    file_sha256,
    load_run_manifest,
    replay_scenes,
)
from src.trajectory_feature_text import feature_text_hash  # noqa: E402

RULE_GRAPH = ROOT / "data/layer2/rule_graph.yaml"

VERDICT_HEADER = [
    "audit_id",
    "scene_facts",
    "applicable_hard_rules",
    "scene_description",
    "candidate_features",
    "human_verdict",
    "human_applicable_rule_ids",
    "human_violated_rule_ids",
    "human_confidence",
    "human_reason",
    "notes",
]
VERDICT_WIDTHS = [14, 46, 52, 62, 78, 16, 26, 26, 16, 42, 28]
EVIDENCE_COLUMNS = (
    "scene_facts",
    "applicable_hard_rules",
    "scene_description",
    "candidate_features",
)

PREFERENCE_ANNOTATION = [
    "human_best_trajectory",
    "human_ranking",
    "human_confidence",
    "human_reason",
]


def load_rule_cards() -> list[dict[str, str]]:
    """从生产规则图谱读取全量规则卡（修改意见 P0-2）。"""
    data = yaml.safe_load(RULE_GRAPH.read_text(encoding="utf-8"))
    cards = [
        {
            "rule_id": str(rule["id"]),
            "severity": str(rule["severity"]),
            "category": str(rule.get("category", "")),
            "description": str(rule["description"]),
        }
        for rule in data["rules"]
    ]
    if not cards:
        raise ValueError("规则图谱没有解析出任何规则卡")
    return sorted(cards, key=lambda card: card["rule_id"])


def _instructions_sheet(
    rule_cards: list[dict[str, str]],
    *,
    title: str,
    body: list[str],
    example_row: list[str] | None,
    example_header: list[str] | None,
) -> Sheet:
    """说明页：纪律 + 全量规则卡 + 一行填好的示例（P0-2 / P1-4 / P2-2）。"""
    rows: list[list[str]] = [[title, ""], ["", ""]]
    for line in body:
        rows.append([line, ""])
    rows += [
        ["", ""],
        ["【全量规则卡】以下为规则图谱的全部规则。", ""],
        [
            "若你认为某条规则适用于当前场景但没有出现在 applicable_hard_rules 列，"
            "请照此表填入 human_applicable_rule_ids，并在 notes 注明「未检索到」。",
            "",
        ],
        ["rule_id / severity / category", "描述"],
    ]
    for card in rule_cards:
        rows.append(
            [
                f"{card['rule_id']}（{card['severity']}，{card['category']}）",
                card["description"],
            ]
        )

    example_styles: dict[int, int] = {}
    if example_row is not None and example_header is not None:
        rows += [["", ""], ["【示例，勿计入】以下是一行填好的虚构示例：", ""]]
        for name, value in zip(example_header, example_row, strict=True):
            rows.append([name, value])
            example_styles[len(rows) - 1] = STYLE_EXAMPLE

    return Sheet(
        name="标注说明",
        header=["项目", "内容"],
        rows=rows,
        column_widths=[58, 96],
        row_styles=example_styles,
    )


VERDICT_INSTRUCTIONS = [
    "每行是一个匿名候选轨迹。请只依据本行给出的证据填写 human_* 列。",
    "",
    "human_verdict：",
    "  cleared  = 现有证据不支持硬规则否决；",
    "  vetoed   = 现有证据明确支持硬规则否决；",
    "  uncertain= 信息不足、规则冲突或无法可靠裁定。",
    "human_applicable_rule_ids：与场景和判断相关的规则 ID，多个用英文分号 ; "
    "分隔；无适用规则填 NONE。",
    "human_violated_rule_ids：只填已有证据确认违反的规则；cleared/uncertain "
    "必须填 NONE；vetoed 至少一条 hard 规则。",
    "human_confidence：1–5 的整数，1 最不确定，5 最确定。",
    "human_reason：一句完整的话说明判断依据。",
    "notes：可选，记录证据冲突、代理区偏移或其他异常。",
    "",
    "标注纪律：",
    "  不猜测表中没有提供的信息；证据不足时选 uncertain。",
    "  不根据行顺序、audit_id 或语言风格推测系统答案。",
    "  不查看 private/ 下任何文件，不查阅原始 JSONL、benchmark 标签或系统判决。",
    "  保存时不要增删或重排行，也不要修改 audit_id。",
    "",
    "证据口径（对应修改意见 P2-2）：",
    "  scene_facts / applicable_hard_rules / scene_description / candidate_features",
    "  四列与系统判定时读到的文本逐字一致，没有删改或摘要。",
    "  scene_description 是 DriveLM 数据集原文（英文），仅作场景背景；",
    "  判定证据以 scene_facts 与 candidate_features 为准，不要从英文叙述过度推断。",
    "  同一场景的不同候选可能分散在多行，请逐行独立判断，不要试图拼接。",
]

PREFERENCE_INSTRUCTIONS = [
    "每行是一个场景，并排列出该场景全部匿名候选及其特征摘要。",
    "",
    "human_best_trajectory：单选最优候选（填候选列名，如 cand_3）；没有可接受候选填 NONE。",
    "human_ranking：全部候选从优到劣排序，如 cand_3>cand_1=cand_5>cand_2>cand_4>cand_6；并列用 =。",
    "human_confidence：1–5 的整数。",
    "human_reason：一句话说明排序依据。",
    "",
    "标注纪律：",
    "  候选顺序按独立于 verdict 表的固定 SHA-256 重新打乱，与 verdict 表无对应关系。",
    "  不要试图从 verdict 表推断本表答案，也不要反向推断。",
    "  全排序会自动展开为全部 pairwise 对，无需逐对填写。",
    "  证据列与系统判定时读到的文本逐字一致。",
]


def _verdict_example() -> tuple[list[str], list[str]]:
    header = VERDICT_HEADER
    row = [
        "A-000（示例）",
        "[场景要件] 当前方向信号相位: 红灯（来源: DriveLM 标注）……",
        "[硬性规则] [R-SIG-01] 红灯相位下禁止越过停止线继续行驶……",
        "The ego vehicle approaches the intersection and slows down.",
        "轨迹共 120 个航点，总行程 8.4 米……冲突区行为: 轨迹进入冲突区；冲突区内最低速度 0.4 m/s",
        "vetoed",
        "R-SIG-01;R-YLD-05",
        "R-SIG-01",
        "4",
        "红灯相位下该轨迹在未完全停车的情况下越过停止线并继续行驶，构成硬规则否决。",
        "R-YLD-05 未出现在 applicable_hard_rules 列，按规则卡补填；未检索到。",
    ]
    return header, row


def build_verdict_workbook(
    rows: list[SampleRow],
    scenes: dict[str, ReplayedScene],
    rule_cards: list[dict[str, str]],
    *,
    audit_ids: dict[tuple[str, str], str],
) -> tuple[Sheet, Sheet, list[dict[str, Any]]]:
    """构建 verdict 表与其盲标行记录。"""
    sheet_rows: list[list[str]] = []
    blind_records: list[dict[str, Any]] = []
    for row in rows:
        candidate = row.candidate
        scene = scenes[candidate.frame_token]
        audit_id = audit_ids[candidate.key]
        context = scene.context
        feature_text = scene.feature_text(candidate.trajectory_id)
        sheet_rows.append(
            [
                audit_id,
                context.scene_facts_text,
                context.hard_rules_text,
                context.scene_narrative,
                feature_text,
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        )
        blind_records.append(
            {
                "audit_id": audit_id,
                "scene_facts": context.scene_facts_text,
                "applicable_hard_rules": context.hard_rules_text,
                "scene_description": context.scene_narrative,
                "candidate_features": feature_text,
                "evidence_sha256": {
                    name: hashlib.sha256(value.encode("utf-8")).hexdigest()
                    for name, value in {
                        "scene_facts": context.scene_facts_text,
                        "applicable_hard_rules": context.hard_rules_text,
                        "scene_description": context.scene_narrative,
                        "candidate_features": feature_text,
                    }.items()
                },
            }
        )

    example_header, example_row = _verdict_example()
    instructions = _instructions_sheet(
        rule_cards,
        title="RegGround-AV 人工盲标包 v2 · verdict 表标注说明",
        body=VERDICT_INSTRUCTIONS,
        example_row=example_row,
        example_header=example_header,
    )
    annotation = Sheet(
        name="标注",
        header=VERDICT_HEADER,
        rows=sheet_rows,
        column_widths=VERDICT_WIDTHS,
        mono_columns=EVIDENCE_COLUMNS,
    )
    return instructions, annotation, blind_records


def build_preference_workbook(
    frame_tokens: list[str],
    scenes: dict[str, ReplayedScene],
    rule_cards: list[dict[str, str]],
) -> tuple[Sheet, Sheet, list[dict[str, Any]], list[dict[str, Any]]]:
    """构建场景级偏好表（修改意见 P0-1）。

    候选顺序用独立 salt 重新打乱并改名为 ``cand_N``，与 verdict 表的呈现顺序
    无对应关系；真实 trajectory_id 只进 sidecar。
    """
    max_candidates = max(
        len(scenes[token].trajectory_ids) for token in frame_tokens
    )
    candidate_columns = [f"cand_{i + 1}" for i in range(max_candidates)]
    header = (
        ["audit_id", "scene_facts", "applicable_hard_rules", "scene_description"]
        + candidate_columns
        + PREFERENCE_ANNOTATION
    )
    widths = [14, 46, 52, 62] + [72] * max_candidates + [22, 34, 16, 42]

    ordered_tokens = order_by_hash(
        list(frame_tokens), salt="preference:rows", key=lambda token: (token,)
    )
    sheet_rows: list[list[str]] = []
    blind_records: list[dict[str, Any]] = []
    sidecar: list[dict[str, Any]] = []

    for index, token in enumerate(ordered_tokens, start=1):
        scene = scenes[token]
        audit_id = f"P-{index:03d}"
        shuffled = order_by_hash(
            list(scene.trajectory_ids),
            salt="preference:candidates",
            key=lambda trajectory_id, token=token: (token, trajectory_id),
        )
        texts = [scene.feature_text(trajectory_id) for trajectory_id in shuffled]
        padded = texts + [""] * (max_candidates - len(texts))
        context = scene.context
        sheet_rows.append(
            [
                audit_id,
                context.scene_facts_text,
                context.hard_rules_text,
                context.scene_narrative,
                *padded,
                "",
                "",
                "",
                "",
            ]
        )
        blind_records.append(
            {
                "audit_id": audit_id,
                "scene_facts": context.scene_facts_text,
                "applicable_hard_rules": context.hard_rules_text,
                "scene_description": context.scene_narrative,
                "candidates": {
                    candidate_columns[i]: texts[i] for i in range(len(texts))
                },
            }
        )
        label = scene.record["final_label"]
        sidecar.append(
            {
                "audit_id": audit_id,
                "frame_token": token,
                "scene_id": scene.scene_id,
                "scenario_type": scene.scenario_type,
                "column_to_trajectory_id": {
                    candidate_columns[i]: shuffled[i] for i in range(len(shuffled))
                },
                "system_chosen_trajectory_id": label.get("chosen_trajectory_id"),
                "system_preference_ranking": label.get("preference_ranking", []),
                "system_selection_outcome": label.get("selection_outcome"),
                "system_verdicts": {
                    trajectory_id: scene.verdict_by_id(trajectory_id)["status"]
                    for trajectory_id in scene.trajectory_ids
                },
                "benchmark_labels": {
                    trajectory_id: scene.benchmark_by_id(trajectory_id)
                    for trajectory_id in scene.trajectory_ids
                },
            }
        )

    instructions = _instructions_sheet(
        rule_cards,
        title="RegGround-AV 人工盲标包 v2 · 场景级偏好表标注说明",
        body=PREFERENCE_INSTRUCTIONS,
        example_row=None,
        example_header=None,
    )
    annotation = Sheet(
        name="标注",
        header=header,
        rows=sheet_rows,
        column_widths=widths,
        mono_columns=("scene_facts", "applicable_hard_rules", "scene_description")
        + tuple(candidate_columns),
    )
    return instructions, annotation, blind_records, sidecar


def build_sidecar(
    rows: list[SampleRow],
    scenes: dict[str, ReplayedScene],
    *,
    audit_ids: dict[tuple[str, str], str],
) -> list[dict[str, Any]]:
    """揭盲 sidecar：分组、系统判决与 benchmark 标签只存在于这里。"""
    records: list[dict[str, Any]] = []
    for row in rows:
        candidate = row.candidate
        scene = scenes[candidate.frame_token]
        records.append(
            {
                "audit_id": audit_ids[candidate.key],
                "frame_token": candidate.frame_token,
                "scene_id": candidate.scene_id,
                "trajectory_id": candidate.trajectory_id,
                "scenario_type": candidate.scenario,
                "group": row.group,
                "stratum": row.stratum,
                "stress_kind": row.stress_kind,
                "borderline": candidate.borderline,
                "system_verdict": scene.verdict_by_id(candidate.trajectory_id),
                "benchmark_label": scene.benchmark_by_id(candidate.trajectory_id),
                "hard_filter_user_prompt": scene.hard_filter_user_prompt(
                    candidate.trajectory_id
                ),
            }
        )
    return records


def assert_no_leakage(
    blind_records: list[dict[str, Any]],
    sidecar: list[dict[str, Any]],
) -> None:
    """盲标行不得包含任何只应出现在 sidecar 的字段或字面量。"""
    forbidden_keys = {
        "group",
        "stratum",
        "stress_kind",
        "borderline",
        "system_verdict",
        "benchmark_label",
        "trajectory_id",
        "frame_token",
        "scene_id",
        "decided_by",
        "expected_verdict",
        "variant_type",
        "difficulty",
    }
    for record in blind_records:
        leaked = forbidden_keys & set(record)
        if leaked:
            raise ValueError(f"盲标行泄露字段：{sorted(leaked)}")

    forbidden_tokens = (
        "rule_engine",
        "decided_by",
        "expected_verdict",
        "ground_truth",
        "benchmark",
        "variant_type",
        "chosen_trajectory",
        "preference_ranking",
    )
    for record in blind_records:
        blob = json.dumps(record, ensure_ascii=False)
        for token in forbidden_tokens:
            if token in blob:
                raise ValueError(f"盲标行出现泄露字面量 {token!r}：{record['audit_id']}")

    # 轨迹 ID 不得出现在证据文本里；traj_a..traj_j 是系统内部编号
    for record in blind_records:
        blob = json.dumps(record, ensure_ascii=False)
        for suffix in "abcdefghij":
            if f"traj_{suffix}" in blob:
                raise ValueError(
                    f"盲标行出现系统轨迹 ID traj_{suffix}：{record['audit_id']}"
                )

    blind_ids = {record["audit_id"] for record in blind_records}
    sidecar_ids = {record["audit_id"] for record in sidecar}
    if blind_ids != sidecar_ids:
        raise ValueError("盲标行与 sidecar 的 audit_id 集合不一致")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _assert_evidence_verbatim(
    rows: list[SampleRow],
    scenes: dict[str, ReplayedScene],
) -> dict[str, Any]:
    """再确认一次证据区与运行日志记录的哈希一致。"""
    checked = 0
    for row in rows:
        scene = scenes[row.candidate.frame_token]
        feature = scene.feature_by_id(row.candidate.trajectory_id)
        logged = row_logged_hash(scene, row.candidate.trajectory_id)
        actual = feature_text_hash(feature)
        if logged != actual:
            raise ValueError(
                "证据区与运行日志不一致："
                f"{row.candidate.frame_token}/{row.candidate.trajectory_id}"
            )
        checked += 1
    return {"candidates_verified": checked, "method": "feature_text_sha256"}


def row_logged_hash(scene: ReplayedScene, trajectory_id: str) -> str:
    for diagnostic in scene.record["final_label"]["stage_diagnostics"]:
        if diagnostic["stage"] == "hard_filter":
            return str((diagnostic.get("feature_text_hashes") or {}).get(trajectory_id))
    raise KeyError(f"{scene.frame_token} 缺少 hard_filter 阶段诊断")


def _assert_workbook_roundtrip(path: Path, expected_sheets: list[Sheet]) -> dict[str, Any]:
    """回读最终 xlsx，逐单元格确认 artifact 写出没有改写证据或空白列。"""
    actual = read_workbook(path)
    checked = 0
    for expected in expected_sheets:
        rows = actual.get(expected.name)
        if rows is None:
            raise ValueError(f"{path.name} 缺少 sheet {expected.name!r}")
        expected_rows = [expected.header, *expected.rows]
        if len(rows) != len(expected_rows):
            raise ValueError(
                f"{path.name}/{expected.name} 行数 {len(rows)} != {len(expected_rows)}"
            )
        width = len(expected.header)
        for index, (observed, wanted) in enumerate(
            zip(rows, expected_rows, strict=True), start=1
        ):
            padded = (observed + [""] * width)[:width]
            if padded != wanted:
                raise ValueError(
                    f"{path.name}/{expected.name} 第 {index} 行回读内容不一致"
                )
            checked += width
    return {"sheets": len(expected_sheets), "cells_verified": checked}


def export(
    *,
    output_root: Path,
    run_log: Path,
    input_path: Path,
    eval_set_path: Path,
    node_executable: Path,
    node_modules: Path,
) -> dict[str, Any]:
    """执行导出，只写 output_root 下的 results/ 与 private/。"""
    scenes_list = replay_scenes(
        run_log=run_log,
        input_path=input_path,
        eval_set_path=eval_set_path,
    )
    scenes = {scene.frame_token: scene for scene in scenes_list}
    result, _frame = run_sampling(scenes_list)
    rule_cards = load_rule_cards()

    workbook_rows = result.workbook_rows
    audit_ids = {
        row.candidate.key: f"A-{index:03d}"
        for index, row in enumerate(workbook_rows, start=1)
    }
    calibration_rows = order_by_hash(
        result.calibration, salt="calibration:rows", key=lambda row: row.candidate.key
    )
    calibration_ids = {
        row.candidate.key: f"C-{index:03d}"
        for index, row in enumerate(calibration_rows, start=1)
    }

    results_dir = output_root / "results"
    private_dir = output_root / "private"
    preview_dir = private_dir / "qa_previews"

    verdict_instructions, verdict_sheet, verdict_blind = build_verdict_workbook(
        workbook_rows, scenes, rule_cards, audit_ids=audit_ids
    )
    verdict_sidecar = build_sidecar(workbook_rows, scenes, audit_ids=audit_ids)
    assert_no_leakage(verdict_blind, verdict_sidecar)

    calibration_instructions, calibration_sheet, calibration_blind = (
        build_verdict_workbook(
            calibration_rows, scenes, rule_cards, audit_ids=calibration_ids
        )
    )
    calibration_sidecar = build_sidecar(
        calibration_rows, scenes, audit_ids=calibration_ids
    )
    assert_no_leakage(calibration_blind, calibration_sidecar)

    (
        preference_instructions,
        preference_sheet,
        preference_blind,
        preference_sidecar,
    ) = build_preference_workbook(result.preference_frames, scenes, rule_cards)
    assert_no_leakage(preference_blind, preference_sidecar)

    write_workbook(
        results_dir / "blind_annotation_template.xlsx",
        [verdict_instructions, verdict_sheet],
        node_executable=node_executable,
        node_modules=node_modules,
        preview_dir=preview_dir,
    )
    write_workbook(
        results_dir / "preference_annotation_template.xlsx",
        [preference_instructions, preference_sheet],
        node_executable=node_executable,
        node_modules=node_modules,
        preview_dir=preview_dir,
    )
    write_workbook(
        results_dir / "calibration_annotation_template.xlsx",
        [calibration_instructions, calibration_sheet],
        node_executable=node_executable,
        node_modules=node_modules,
        preview_dir=preview_dir,
    )
    workbook_checks = {
        "blind_annotation_template.xlsx": _assert_workbook_roundtrip(
            results_dir / "blind_annotation_template.xlsx",
            [verdict_instructions, verdict_sheet],
        ),
        "preference_annotation_template.xlsx": _assert_workbook_roundtrip(
            results_dir / "preference_annotation_template.xlsx",
            [preference_instructions, preference_sheet],
        ),
        "calibration_annotation_template.xlsx": _assert_workbook_roundtrip(
            results_dir / "calibration_annotation_template.xlsx",
            [calibration_instructions, calibration_sheet],
        ),
    }
    _write_jsonl(results_dir / "blind_samples.jsonl", verdict_blind)
    _write_jsonl(results_dir / "preference_blind_samples.jsonl", preference_blind)
    _write_jsonl(results_dir / "calibration_blind_samples.jsonl", calibration_blind)
    _write_jsonl(private_dir / "hidden_sidecar.jsonl", verdict_sidecar)
    _write_jsonl(private_dir / "preference_sidecar.jsonl", preference_sidecar)
    _write_jsonl(private_dir / "calibration_sidecar.jsonl", calibration_sidecar)

    run_manifest = load_run_manifest(run_log)
    manifest = {
        "package_version": "human_annotation_v2",
        "generated_from": {
            "run_log": str(run_log),
            "run_log_sha256": file_sha256(run_log),
            "run_git_commit": run_manifest.get("git_commit"),
            "input_sha256": run_manifest.get("input_sha256"),
            "eval_set_sha256": (run_manifest.get("eval_set") or {}).get("sha256"),
            "rule_graph_sha256": run_manifest.get("rule_graph_sha256"),
            "predicate_version": run_manifest.get("predicate_version"),
            "graph_version": run_manifest.get("graph_version"),
            "model": run_manifest.get("model"),
        },
        "evidence_contract": {
            "source": "renders of JudgeInput v2 replayed from the frozen input",
            "columns": list(EVIDENCE_COLUMNS),
            "verbatim_check": _assert_evidence_verbatim(workbook_rows, scenes),
            "note": (
                "证据四列与 HardFilter 发给 LLM 的 prompt 片段逐字一致；"
                "prompt 里的 [traj_x] 前缀由 HardFilter 单独拼接，不属于证据文本，"
                "因此工作簿用 audit_id 替代而不破坏逐字性。"
            ),
        },
        "rule_cards": rule_cards,
        "sampling": result.manifest,
        "calibration_ids": len(calibration_ids),
        "workbook_roundtrip_checks": workbook_checks,
        "outputs": {},
    }

    artifacts = [
        results_dir / "blind_annotation_template.xlsx",
        results_dir / "preference_annotation_template.xlsx",
        results_dir / "calibration_annotation_template.xlsx",
        results_dir / "blind_samples.jsonl",
        results_dir / "preference_blind_samples.jsonl",
        results_dir / "calibration_blind_samples.jsonl",
        results_dir / "CALIBRATION_PROTOCOL.md",
    ]
    manifest["outputs"] = {
        path.name: file_sha256(path) for path in artifacts
    }

    manifest_path = results_dir / "sample_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    hashes = {
        "generated_from_commit": run_manifest.get("git_commit"),
        "run_log_sha256": file_sha256(run_log),
        "artifacts": {path.name: file_sha256(path) for path in artifacts},
        "sample_manifest.json": file_sha256(manifest_path),
    }
    hashes["package_digest"] = hashlib.sha256(
        json.dumps(hashes, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    (results_dir / "artifact_hashes.json").write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出人工盲标包 v2。")
    parser.add_argument("--output-root", type=Path, default=ROOT / "human_annotation_v2")
    parser.add_argument("--run-log", type=Path, default=ROOT / DEFAULT_RUN_LOG)
    parser.add_argument("--input", type=Path, default=ROOT / DEFAULT_INPUT)
    parser.add_argument("--eval-set", type=Path, default=ROOT / DEFAULT_EVAL_SET)
    parser.add_argument(
        "--artifact-node",
        type=Path,
        required=True,
        help="load_workspace_dependencies 返回的 Node.js executable",
    )
    parser.add_argument(
        "--artifact-node-modules",
        type=Path,
        required=True,
        help="load_workspace_dependencies 返回的 Node.js packages 目录",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = export(
        output_root=args.output_root.resolve(),
        run_log=args.run_log.resolve(),
        input_path=args.input.resolve(),
        eval_set_path=args.eval_set.resolve(),
        node_executable=args.artifact_node.resolve(),
        node_modules=args.artifact_node_modules.resolve(),
    )
    summary = manifest["sampling"]["summary"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"证据逐字校验：{manifest['evidence_contract']['verbatim_check']}")
    print(f"输出目录：{args.output_root}")


if __name__ == "__main__":
    main()
