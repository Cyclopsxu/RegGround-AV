"""影子评估：关闭规则引擎，重判 118 条规则引擎决定性候选。

问题背景：正式运行里 rule_engine 直裁的候选不进 benchmark accuracy
（``verdict_layers.rule_engine.benchmark_accuracy_reported = false``，
只留下 ``required_external_metric = "predicate_human_agreement"``），
而 LLM 裁定的 382 条在 benchmark 里全是 ``exclude``。结果是"RAG+LLM 判得如何"
这个问题在主表上没有直接答案。

本影子评估的做法：

* 冻结输入不动、benchmark 不动、检索不动；
* 只在 hard filter 阶段把规则引擎按配置关掉（``disable_rule_engine=True``，
  与 2×2 消融的 ``no_rule_engine`` 条件共用 :class:`DisabledRuleEngine`）；
* 让 LLM 独立重判那 118 条候选，与已审计几何谓词给出的标签比对。

由于谓词标签来自 :mod:`src.compliance_predicates` 的几何计算，与 LLM 无关，
这个一致率对 LLM 是非循环的：它直接回答"把规则引擎拿掉，RAG+LLM 自己判得如何"。

用法::

    # 先看计划与 prompt 构造，不调用 LLM、不花钱
    python scripts/run_shadow_no_rule_engine.py --dry-run

    # 真正执行（118 次 LLM 调用，可断点续跑）
    python scripts/run_shadow_no_rule_engine.py --execute
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.layer3.hard_filter import HardFilter  # noqa: E402
from src.layer3.leakage_guard import LeakageGuard  # noqa: E402
from src.layer3.llm_client import LLMClient, get_llm_429_count  # noqa: E402
from src.layer3.models import Layer3Settings  # noqa: E402
from src.layer3.prompt_loader import load_system_prompt  # noqa: E402
from src.layer3.rule_engine import DisabledRuleEngine, build_rule_engine  # noqa: E402
from src.replay_context import (  # noqa: E402
    DEFAULT_EVAL_SET,
    DEFAULT_INPUT,
    DEFAULT_RUN_LOG,
    ReplayedScene,
    file_sha256,
    load_run_manifest,
    replay_scenes,
)

DEFAULT_OUTPUT_DIR = ROOT / "results/shadow_no_rule_engine_v2"
DEFAULT_PHASE_CONTINUITY_OUTPUT_DIR = (
    ROOT / "results/shadow_no_rule_engine_v2_vb_phase_continuity"
)
BASE_PHASE_SNAPSHOT_LINE = "信号灯相位为 keyframe 时刻（t=0）快照，其后相位未知"
STATUSES = ("cleared", "vetoed", "uncertain")


def phase_continuity_statement(horizon_s: float) -> str:
    """返回与谓词 ``entry_time_s <= horizon`` 精确对齐的评估声明。"""
    return (
        "信号灯相位为 keyframe 时刻（t=0）的观测快照；"
        f"本评估约定该相位在 t∈[0, {horizon_s:.1f} s] 内视为持续有效，"
        f"t>{horizon_s:.1f} s 时相位未知。"
    )


def apply_phase_continuity_assumption(
    scenes: list[ReplayedScene],
    *,
    horizon_s: float,
) -> list[ReplayedScene]:
    """只改评估臂的上下文副本；生产 ``ContextBuilder`` 保持原样。"""
    statement = phase_continuity_statement(horizon_s)
    updated: list[ReplayedScene] = []
    for scene in scenes:
        if scene.context.scene_facts_text.count(BASE_PHASE_SNAPSHOT_LINE) != 1:
            raise ValueError(
                f"{scene.frame_token} 的相位快照声明不是唯一一行，拒绝静默改写"
            )
        scene_facts_text = scene.context.scene_facts_text.replace(
            BASE_PHASE_SNAPSHOT_LINE,
            statement,
        )
        context = scene.context.model_copy(
            update={"scene_facts_text": scene_facts_text}
        )
        updated.append(replace(scene, context=context))
    return updated


def scan_target_prompts(
    grouped: list[tuple[ReplayedScene, list[str]]],
    *,
    phase_statement: str | None,
) -> dict[str, Any]:
    """在任何 LLM 调用前扫描 118 条完整证据 prompt。"""
    system_prompt = load_system_prompt("hard_filter")
    prompt_count = 0
    statement_count = 0
    for scene, targets in grouped:
        for trajectory_id in targets:
            prompt = scene.hard_filter_user_prompt(trajectory_id)
            LeakageGuard.assert_prompt_safe(system_prompt, prompt)
            prompt_count += 1
            if phase_statement is not None:
                occurrences = prompt.count(phase_statement)
                if occurrences != 1:
                    raise ValueError(
                        "相位持续性声明必须在每条目标 prompt 中恰好出现一次："
                        f"{scene.frame_token}/{trajectory_id} 出现 {occurrences} 次"
                    )
                statement_count += occurrences

    conclusion_terms = (
        "vetoed",
        "cleared",
        "uncertain",
        "违规",
        "违反",
        "合规",
        "R-SIG-01",
    )
    statement_conclusion_hits = (
        [term for term in conclusion_terms if term in phase_statement]
        if phase_statement is not None
        else []
    )
    if statement_conclusion_hits:
        raise ValueError(
            f"相位声明含结论性词语：{statement_conclusion_hits}"
        )
    return {
        "passed": True,
        "target_prompts_scanned": prompt_count,
        "layer3_banned_term_hits": [],
        "phase_statement": phase_statement,
        "phase_statement_occurrences": statement_count,
        "phase_statement_conclusion_hits": statement_conclusion_hits,
        "scanned_at": datetime.now(UTC).isoformat(),
    }


def select_targets(
    scenes: list[ReplayedScene],
) -> list[tuple[ReplayedScene, list[str]]]:
    """选出 rule_engine 直裁且进入评分的候选，按场景分组。

    与 ``src/evaluation_report.py`` 中 ``rule_engine_evaluated`` 的口径一致：
    decided_by == rule_engine 且 benchmark 标签不是 exclude。
    """
    grouped: list[tuple[ReplayedScene, list[str]]] = []
    for scene in scenes:
        targets = [
            trajectory_id
            for trajectory_id in scene.trajectory_ids
            if scene.verdict_by_id(trajectory_id).get("decided_by") == "rule_engine"
            and scene.benchmark_by_id(trajectory_id).get("expected_verdict") != "exclude"
        ]
        if targets:
            grouped.append((scene, targets))
    return grouped


def _prompt_digest(scene: ReplayedScene, trajectory_id: str) -> str:
    return hashlib.sha256(
        scene.hard_filter_user_prompt(trajectory_id).encode("utf-8")
    ).hexdigest()


def _load_completed(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    done: dict[tuple[str, str], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = (record["frame_token"], record["trajectory_id"])
        if key in done:
            raise ValueError(f"影子输出存在重复候选：{key}")
        done[key] = record
    return done


def _validate_completed(
    completed: dict[tuple[str, str], dict[str, Any]],
    expected_prompt_hashes: dict[tuple[str, str], str],
) -> None:
    """断点只能复用本次目标且确由 LLM 完成的记录。"""
    extras = set(completed) - set(expected_prompt_hashes)
    if extras:
        raise ValueError(f"影子输出混入非本次目标：{sorted(extras)[:5]}")
    for key, record in completed.items():
        if record.get("prompt_sha256") != expected_prompt_hashes[key]:
            raise ValueError(f"影子输出 prompt 哈希与当前冻结输入不一致：{key}")
        if record.get("shadow_decided_by") != "llm":
            raise ValueError(f"影子输出混入非 LLM 判定：{key}")
        if record.get("shadow_completed_by") == "fallback":
            raise ValueError(f"影子输出混入 fallback：{key}")


def run_scene(
    scene: ReplayedScene,
    targets: list[str],
    *,
    settings: Layer3Settings,
    llm: LLMClient,
) -> list[dict[str, Any]]:
    """对一个场景的目标候选跑关闭规则引擎的 hard filter。

    只把目标候选交给 HardFilter，并把 ``trajectory_ids`` 重映射到目标候选，
    使发给 LLM 的 prompt 与生产逐候选调用逐字一致（生产的
    ``_call_llm_candidate`` 同样是一次调用只带一个候选）。
    """
    reduced_context = scene.context.model_copy(update={"trajectory_ids": list(targets)})
    features = [scene.feature_by_id(trajectory_id) for trajectory_id in targets]

    hard_filter = HardFilter(
        settings,
        llm,
        rule_engine=build_rule_engine(
            phase_validity_horizon_s=settings.phase_validity_horizon_s,
            disabled=settings.disable_rule_engine,
        ),
    )
    if not isinstance(hard_filter._engine, DisabledRuleEngine):  # noqa: SLF001
        raise RuntimeError("影子评估必须在规则引擎关闭下运行")

    before = llm.usage_snapshot()
    started = time.perf_counter()
    results, _reasoning = hard_filter.filter(reduced_context, features)
    duration_ms = int((time.perf_counter() - started) * 1000)
    usage = llm.usage_snapshot().delta(before)

    invalid = [
        result.trajectory_id
        for result in results
        if result.decided_by.value != "llm" or result.completed_by.value == "fallback"
    ]
    if invalid:
        raise RuntimeError(f"影子评估禁止记录非 LLM/fallback 结果：{invalid}")

    records: list[dict[str, Any]] = []
    for trajectory_id, result in zip(targets, results, strict=True):
        production = scene.verdict_by_id(trajectory_id)
        benchmark = scene.benchmark_by_id(trajectory_id)
        records.append(
            {
                "frame_token": scene.frame_token,
                "scene_id": scene.scene_id,
                "scenario_type": scene.scenario_type,
                "trajectory_id": trajectory_id,
                "predicate_label": production["status"],
                "predicate_reason": production["reason"],
                "benchmark_expected_verdict": benchmark["expected_verdict"],
                "difficulty": benchmark["difficulty"],
                "variant_type": benchmark["variant_type"],
                "shadow_status": result.status.value,
                "shadow_violated_rule_ids": list(result.violated_rule_ids),
                "shadow_reason": result.reason,
                "shadow_decided_by": result.decided_by.value,
                "shadow_completed_by": result.completed_by.value,
                "hard_rule_ids": list(scene.context.hard_rule_ids),
                "prompt_sha256": _prompt_digest(scene, trajectory_id),
                "feature_text_sha256": benchmark["feature_text_hash"],
                "scene_duration_ms": duration_ms,
                "scene_llm_calls": usage.call_count,
                "scene_input_tokens": usage.input_tokens,
                "scene_output_tokens": usage.output_tokens,
                "evaluated_at": datetime.now(UTC).isoformat(),
            }
        )
    return records


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def build_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    """把逐候选结果汇总成一致率报告。"""
    total = len(records)
    agree = sum(1 for r in records if r["shadow_status"] == r["predicate_label"])
    abstain = sum(1 for r in records if r["shadow_status"] == "uncertain")
    decided = total - abstain
    decided_agree = sum(
        1
        for r in records
        if r["shadow_status"] != "uncertain"
        and r["shadow_status"] == r["predicate_label"]
    )

    confusion: Counter[tuple[str, str]] = Counter(
        (r["predicate_label"], r["shadow_status"]) for r in records
    )

    def stratify(key: Any) -> dict[str, Any]:
        buckets: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            buckets.setdefault(str(key(record)), []).append(record)
        out: dict[str, Any] = {}
        for name, items in sorted(buckets.items()):
            n = len(items)
            hit = sum(1 for r in items if r["shadow_status"] == r["predicate_label"])
            abst = sum(1 for r in items if r["shadow_status"] == "uncertain")
            out[name] = {
                "n": n,
                "agreement_rate": _rate(hit, n),
                "abstention_rate": _rate(abst, n),
                "agreement_rate_excluding_abstention": _rate(
                    sum(
                        1
                        for r in items
                        if r["shadow_status"] != "uncertain"
                        and r["shadow_status"] == r["predicate_label"]
                    ),
                    n - abst,
                ),
            }
        return out

    # 谓词判 vetoed 而 LLM 未判 vetoed = 漏检；谓词判 cleared 而 LLM 判 vetoed = 误否决
    predicate_vetoed = [r for r in records if r["predicate_label"] == "vetoed"]
    predicate_cleared = [r for r in records if r["predicate_label"] == "cleared"]

    return {
        "candidates": total,
        "scenes": len({r["frame_token"] for r in records}),
        "primary_metric": {
            "name": "llm_predicate_agreement_rate",
            "definition": (
                "关闭规则引擎后 LLM 独立判定与已审计几何谓词标签一致的比例；"
                "uncertain 计为不一致"
            ),
            "agreement": agree,
            "denominator": total,
            "rate": _rate(agree, total),
        },
        "abstention": {
            "count": abstain,
            "denominator": total,
            "rate": _rate(abstain, total),
        },
        "agreement_excluding_abstention": {
            "agreement": decided_agree,
            "denominator": decided,
            "rate": _rate(decided_agree, decided),
        },
        "directional_errors": {
            "missed_violation": {
                "count": sum(
                    1 for r in predicate_vetoed if r["shadow_status"] != "vetoed"
                ),
                "denominator": len(predicate_vetoed),
                "rate": _rate(
                    sum(1 for r in predicate_vetoed if r["shadow_status"] != "vetoed"),
                    len(predicate_vetoed),
                ),
                "note": "谓词判 vetoed 而 LLM 未判 vetoed",
            },
            "false_veto": {
                "count": sum(
                    1 for r in predicate_cleared if r["shadow_status"] == "vetoed"
                ),
                "denominator": len(predicate_cleared),
                "rate": _rate(
                    sum(1 for r in predicate_cleared if r["shadow_status"] == "vetoed"),
                    len(predicate_cleared),
                ),
                "note": "谓词判 cleared 而 LLM 判 vetoed",
            },
        },
        "confusion_matrix": {
            f"predicate_{predicate}|shadow_{shadow}": count
            for (predicate, shadow), count in sorted(confusion.items())
        },
        "by_scenario": stratify(lambda r: r["scenario_type"]),
        "by_predicate_reason": stratify(lambda r: r["predicate_reason"]),
        "by_predicate_label": stratify(lambda r: r["predicate_label"]),
        "by_difficulty": stratify(lambda r: r["difficulty"]),
        "citation": {
            "shadow_vetoed_with_citation": sum(
                1
                for r in records
                if r["shadow_status"] == "vetoed" and r["shadow_violated_rule_ids"]
            ),
            "shadow_vetoed_total": sum(
                1 for r in records if r["shadow_status"] == "vetoed"
            ),
            "cited_outside_hard_rules": sum(
                1
                for r in records
                if set(r["shadow_violated_rule_ids"]) - set(r["hard_rule_ids"])
            ),
        },
        "interpretation": {
            "non_circular_for_llm": True,
            "why": (
                "对照标签由 src/compliance_predicates.py 的几何谓词计算得到，"
                "不含任何 LLM 输出，因此该一致率衡量的是 RAG+LLM 能否独立复现"
                "已审计的几何判定，对 LLM 不构成自我验证。"
            ),
            "caveat": (
                "谓词只在明确样本上直裁，故这 118 条整体偏容易；"
                "该一致率是 RAG+LLM 判定能力的上界参考，不能外推到"
                "382 条 LLM 主池（那部分仍需人工盲标）。"
            ),
        },
    }


def render_markdown(report: dict[str, Any], manifest: dict[str, Any]) -> str:
    primary = report["primary_metric"]
    lines = [
        "# 影子评估：关闭规则引擎 × 118 条决定性候选",
        "",
        f"- 完成时间：{manifest['finished_at']}",
        (
            f"- 来源运行：`{manifest['source_run_log']}`"
            f"（commit `{manifest['source_run_git_commit']}`）"
        ),
        "- 条件：`disable_rule_engine=True`，与 2×2 消融的 `no_rule_engine` 共用实现",
        f"- 冻结输入与 benchmark 未改动（input sha256 `{manifest['input_sha256'][:16]}…`）",
        "",
        "## 主指标",
        "",
        f"**LLM 对谓词标签的独立一致率 = {primary['rate']:.3f}**"
        f"（{primary['agreement']}/{primary['denominator']}）",
        "",
        f"- 弃权率（uncertain）：{report['abstention']['rate']:.3f}"
        f"（{report['abstention']['count']}/{report['abstention']['denominator']}）",
        f"- 排除弃权后的一致率："
        f"{report['agreement_excluding_abstention']['rate']:.3f}"
        if report["agreement_excluding_abstention"]["rate"] is not None
        else "- 排除弃权后的一致率：不可计算",
        "",
        "## 方向性错误",
        "",
        "| 类型 | 计数 | 分母 | 比率 |",
        "|---|---|---|---|",
    ]
    for name, item in report["directional_errors"].items():
        rate = f"{item['rate']:.3f}" if item["rate"] is not None else "n/a"
        lines.append(
            f"| {name}（{item['note']}） | {item['count']} | "
            f"{item['denominator']} | {rate} |"
        )

    lines += ["", "## 混淆矩阵", "", "| 谓词标签 → 影子判定 | 计数 |", "|---|---|"]
    for key, value in report["confusion_matrix"].items():
        lines.append(f"| {key} | {value} |")

    for section, title in (
        ("by_scenario", "按场景类型"),
        ("by_predicate_label", "按谓词标签"),
        ("by_predicate_reason", "按谓词原因"),
        ("by_difficulty", "按难度"),
    ):
        lines += ["", f"## {title}", "", "| 分层 | n | 一致率 | 弃权率 |", "|---|---|---|---|"]
        for name, item in report[section].items():
            rate = (
                f"{item['agreement_rate']:.3f}"
                if item["agreement_rate"] is not None
                else "n/a"
            )
            abst = (
                f"{item['abstention_rate']:.3f}"
                if item["abstention_rate"] is not None
                else "n/a"
            )
            lines.append(f"| {name} | {item['n']} | {rate} | {abst} |")

    lines += [
        "",
        "## 口径说明",
        "",
        f"- 非循环性：{report['interpretation']['why']}",
        f"- 注意：{report['interpretation']['caveat']}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="关闭规则引擎的影子评估。")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-log", type=Path, default=ROOT / DEFAULT_RUN_LOG)
    parser.add_argument("--input", type=Path, default=ROOT / DEFAULT_INPUT)
    parser.add_argument("--eval-set", type=Path, default=ROOT / DEFAULT_EVAL_SET)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只校验目标选取与 prompt 构造，不调用 LLM",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="真正执行 LLM 重判（会产生 API 费用）",
    )
    parser.add_argument(
        "--phase-continuity-assumption",
        action="store_true",
        help="仅在评估臂声明 keyframe 相位于谓词有效期内持续，不改生产 prompt",
    )
    args = parser.parse_args()

    if args.execute == args.dry_run:
        raise SystemExit("必须且只能选择 --dry-run 或 --execute 之一")

    settings = Layer3Settings(disable_rule_engine=True)
    scenes = replay_scenes(
        run_log=args.run_log.resolve(),
        input_path=args.input.resolve(),
        eval_set_path=args.eval_set.resolve(),
    )
    phase_statement = None
    if args.phase_continuity_assumption:
        phase_statement = phase_continuity_statement(
            settings.phase_validity_horizon_s
        )
        scenes = apply_phase_continuity_assumption(
            scenes,
            horizon_s=settings.phase_validity_horizon_s,
        )
    grouped = select_targets(scenes)
    target_count = sum(len(targets) for _scene, targets in grouped)

    run_manifest = load_run_manifest(args.run_log.resolve())
    default_output_dir = (
        DEFAULT_PHASE_CONTINUITY_OUTPUT_DIR
        if args.phase_continuity_assumption
        else DEFAULT_OUTPUT_DIR
    )
    output_dir = (args.output_dir or default_output_dir).resolve()
    if args.phase_continuity_assumption and output_dir == DEFAULT_OUTPUT_DIR.resolve():
        raise SystemExit("v_b 禁止写入 v_a 目录；请使用独立输出目录")
    output_dir.mkdir(parents=True, exist_ok=True)

    leakage_scan = scan_target_prompts(
        grouped,
        phase_statement=phase_statement,
    )
    if leakage_scan["target_prompts_scanned"] != 118:
        raise RuntimeError(
            "影子评估目标数漂移："
            f"扫描到 {leakage_scan['target_prompts_scanned']}，预期 118"
        )
    (output_dir / "leakage_scan.json").write_text(
        json.dumps(leakage_scan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "experiment": (
            "shadow_no_rule_engine_v2_vb_phase_continuity"
            if args.phase_continuity_assumption
            else "shadow_no_rule_engine_v2"
        ),
        "condition": {
            "disable_rule_engine": settings.disable_rule_engine,
            "shared_with": "experiments/ablation_2x2 (all four conditions)",
            "changed_stage": "hard_filter",
            "evaluation_arm_only": args.phase_continuity_assumption,
            "phase_continuity_statement": phase_statement,
            "phase_continuity_horizon_s": (
                settings.phase_validity_horizon_s
                if args.phase_continuity_assumption
                else None
            ),
            "unchanged": [
                "frozen input",
                "benchmark labels",
                "layer2 retrieval",
                "system prompt",
                "few-shot examples",
                "model and decoding configuration",
            ],
        },
        "started_at": datetime.now(UTC).isoformat(),
        "source_run_log": str(args.run_log),
        "source_run_git_commit": run_manifest.get("git_commit"),
        "input_sha256": run_manifest.get("input_sha256"),
        "eval_set_sha256": (run_manifest.get("eval_set") or {}).get("sha256"),
        "rule_graph_sha256": run_manifest.get("rule_graph_sha256"),
        "predicate_version": run_manifest.get("predicate_version"),
        "model": settings.llm_model,
        "hard_filter_temperature": settings.hard_filter_temperature,
        "target_scenes": len(grouped),
        "target_candidates": target_count,
        "rule_engine_py_sha256": file_sha256(ROOT / "src/layer3/rule_engine.py"),
        "hard_filter_py_sha256": file_sha256(ROOT / "src/layer3/hard_filter.py"),
        "leakage_scan_passed": leakage_scan["passed"],
    }

    if args.dry_run:
        plan = [
            {
                "frame_token": scene.frame_token,
                "scenario_type": scene.scenario_type,
                "targets": targets,
                "hard_rule_ids": list(scene.context.hard_rule_ids),
                "prompt_sha256": {
                    trajectory_id: _prompt_digest(scene, trajectory_id)
                    for trajectory_id in targets
                },
            }
            for scene, targets in grouped
        ]
        (output_dir / "dry_run_plan.json").write_text(
            json.dumps(
                {"manifest": manifest, "plan": plan}, ensure_ascii=False, indent=2
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "target_scenes": len(grouped),
                    "target_candidates": target_count,
                    "planned_llm_calls": target_count,
                    "output": str(output_dir / "dry_run_plan.json"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    verdicts_path = output_dir / "shadow_verdicts.jsonl"
    completed = _load_completed(verdicts_path)
    expected_prompt_hashes = {
        (scene.frame_token, trajectory_id): _prompt_digest(scene, trajectory_id)
        for scene, targets in grouped
        for trajectory_id in targets
    }
    _validate_completed(completed, expected_prompt_hashes)
    llm = LLMClient(settings)

    with verdicts_path.open("a", encoding="utf-8") as handle:
        for index, (scene, targets) in enumerate(grouped, start=1):
            pending = [
                trajectory_id
                for trajectory_id in targets
                if (scene.frame_token, trajectory_id) not in completed
            ]
            if not pending:
                print(f"[{index}/{len(grouped)}] {scene.frame_token} 已完成，跳过")
                continue
            records = run_scene(scene, pending, settings=settings, llm=llm)
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                completed[(record["frame_token"], record["trajectory_id"])] = record
            handle.flush()
            agreed = sum(
                1 for r in records if r["shadow_status"] == r["predicate_label"]
            )
            print(
                f"[{index}/{len(grouped)}] {scene.frame_token} "
                f"{len(records)} 条，一致 {agreed}",
                flush=True,
            )

    records = [completed[key] for key in sorted(completed)]
    if len(records) != target_count:
        raise RuntimeError(
            f"影子评估未完整覆盖目标：完成 {len(records)} / {target_count}"
        )
    report = build_report(records)
    manifest["llm_429_count"] = get_llm_429_count()
    manifest["completed_candidates"] = len(records)
    manifest["finished_at"] = datetime.now(UTC).isoformat()

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "agreement_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "agreement_report.md").write_text(
        render_markdown(report, manifest), encoding="utf-8"
    )
    artifacts = (
        verdicts_path,
        output_dir / "manifest.json",
        output_dir / "agreement_report.json",
        output_dir / "agreement_report.md",
        output_dir / "dry_run_plan.json",
        output_dir / "leakage_scan.json",
        output_dir / "PREREGISTRATION.md",
    )
    (output_dir / "artifact_hashes.json").write_text(
        json.dumps(
            {
                "source_run_git_commit": run_manifest.get("git_commit"),
                "artifacts": {
                    path.name: file_sha256(path) for path in artifacts if path.exists()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["primary_metric"], ensure_ascii=False, indent=2))
    print(f"输出目录：{output_dir}")


if __name__ == "__main__":
    main()
