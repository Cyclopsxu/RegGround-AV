"""End-to-end audit runner that writes per-scene intermediate and final logs."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import time
import traceback
from collections import Counter
from datetime import datetime
from fcntl import LOCK_EX, LOCK_UN, flock
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.admission import admission_reason
from src.compliance_predicates import PREDICATE_VERSION
from src.eval_set import (
    EvalSet,
    assert_classification_code_matches,
    assert_input_file_matches,
    file_sha256,
    load_eval_set,
)
from src.evaluation_report import evaluate_entries
from src.layer1.benchmark_builder import BENCHMARK_DATA_VERSION, relabel_benchmark_labels
from src.layer1.facade import parse_scene
from src.layer1.models import ParsedSceneBundle, RawScene
from src.layer2.models import RuleNode, RuleSubgraph
from src.layer2.retriever import GraphRAGRetriever
from src.layer2.rule_graph import Layer2Settings
from src.layer3.context_builder import ContextBuilder
from src.layer3.judge import AuditJudge
from src.layer3.llm_client import get_llm_429_count
from src.layer3.models import AuditLabel, Layer3Settings

DEFAULT_INPUT = Path("data/layer1/raw_scene_dataset_mini.json")
DEFAULT_LOGS_DIR = Path("logs")
DEFAULT_OUTPUT_PREFIX = "output"


def model_dump_json_ready(model: BaseModel) -> dict[str, Any]:
    """Dump a Pydantic model using JSON-compatible scalar values."""
    return model.model_dump(mode="json")


def make_output_path(logs_dir: Path, prefix: str, now: datetime | None = None) -> Path:
    """构造带时间戳的 JSONL 输出路径。"""
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return logs_dir / f"{prefix}_{timestamp}.jsonl"


def load_records(input_path: Path) -> list[dict[str, Any]]:
    """Load the prepared RawScene dataset."""
    with input_path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{input_path} top-level JSON must be a list")
    return [item for item in data if isinstance(item, dict)]


def select_frozen_eval_records(
    records: list[dict[str, Any]],
    eval_set: EvalSet,
) -> list[dict[str, Any]]:
    """按冻结 token 表装载正式运行输入，不在运行时重选样本。"""
    by_token: dict[str, dict[str, Any]] = {}
    for record in records:
        frame_token = record.get("frame_token")
        if not isinstance(frame_token, str) or not frame_token:
            continue
        if frame_token in by_token:
            raise ValueError(f"prepared input contains duplicate frame_token: {frame_token}")
        by_token[frame_token] = record

    selected: list[dict[str, Any]] = []
    for entry in eval_set.entries:
        frozen_record = by_token.get(entry.frame_token)
        if frozen_record is None:
            raise ValueError(f"frozen eval frame is missing from input: {entry.frame_token}")
        if frozen_record.get("scene_id") != entry.scene_token:
            raise ValueError(
                "frozen eval scene does not match prepared input: "
                f"{entry.frame_token}",
            )
        selected.append(frozen_record)
    return selected


def llm_settings_log(settings: Layer3Settings) -> dict[str, Any]:
    """Log runtime LLM settings without exposing the API key."""
    key = settings.llm_api_key.get_secret_value()
    return {
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "llm_model_version": settings.llm_model_version,
        "llm_thinking_mode": settings.llm_thinking_mode,
        "llm_base_url": settings.llm_base_url,
        "llm_api_key_present": bool(key),
        "llm_api_key_prefix": key[:3] if key else "",
        "llm_api_key_length": len(key),
        "llm_timeout_seconds": settings.llm_timeout_seconds,
        "llm_max_retries": settings.llm_max_retries,
        "llm_retry_backoff_base_seconds": settings.llm_retry_backoff_base_seconds,
        "llm_retry_backoff_multiplier": settings.llm_retry_backoff_multiplier,
        "llm_structured_max_tokens": settings.llm_structured_max_tokens,
        "llm_text_max_tokens": settings.llm_text_max_tokens,
        "hard_filter_timeout_seconds": settings.hard_filter_timeout_seconds,
        "hard_filter_max_retries": settings.hard_filter_max_retries,
        "deferred_retry_rounds": settings.deferred_retry_rounds,
        "deferred_retry_delay_seconds": settings.deferred_retry_delay_seconds,
        "llm_input_price_per_million_usd": settings.llm_input_price_per_million_usd,
        "llm_output_price_per_million_usd": settings.llm_output_price_per_million_usd,
        "max_trajectories_per_judge": settings.max_trajectories_per_judge,
        "max_rules_in_context": settings.max_rules_in_context,
        "max_pairwise_comparisons": settings.max_pairwise_comparisons,
        "max_prompt_tokens": settings.max_prompt_tokens,
        "phase_validity_horizon_s": settings.phase_validity_horizon_s,
        "hard_filter_temperature": settings.hard_filter_temperature,
        "pairwise_ranker_temperature": settings.pairwise_ranker_temperature,
        "label_temperature": settings.label_temperature,
    }


def raw_scene_log(raw_scene: RawScene) -> dict[str, Any]:
    """Keep raw input context readable while avoiding huge QA dumps."""
    raw_json = raw_scene.raw_json
    return {
        "scene_id": raw_scene.scene_id,
        "frame_token": raw_scene.frame_token,
        "source_file": str(raw_scene.source_file),
        "location": raw_scene.location,
        "keyframe_index": raw_scene.keyframe_index,
        "scene_name": raw_json.get("scene_name"),
        "scene_description": raw_json.get("scene_description"),
        "drivelm_scene_description": raw_json.get("drivelm_scene_description"),
        "keywords": raw_json.get("keywords", []),
        "qa_pair_count": len(raw_json.get("qa_pairs", [])),
        "key_object_count": len(raw_json.get("key_object_infos", {})),
        "ego_pose_count": len(raw_scene.ego_poses),
        "annotation_count": len(raw_scene.annotations),
        "map_record_counts": {
            layer_name: len(records)
            for layer_name, records in sorted(raw_scene.map_records.items())
        },
    }


def rule_log(rule: RuleNode) -> dict[str, Any]:
    """Serialize a retrieved rule with its trace fields."""
    return model_dump_json_ready(rule)


def subgraph_log(subgraph: RuleSubgraph) -> dict[str, Any]:
    """Serialize Layer 2 output plus active/hard/soft rule views."""
    return {
        "matched_conditions": subgraph.matched_conditions,
        "retrieval_mode": subgraph.retrieval_mode.value,
        "query_duration_ms": subgraph.query_duration_ms,
        "graph_version": subgraph.graph_version,
        "retrieval_timestamp": subgraph.retrieval_timestamp.isoformat(),
        "rules": [rule_log(rule) for rule in subgraph.rules],
        "active_rules": [rule_log(rule) for rule in subgraph.active_rules],
        "hard_rules": [rule_log(rule) for rule in subgraph.hard_rules],
        "soft_rules": [rule_log(rule) for rule in subgraph.soft_rules],
        "overrides_applied": [
            model_dump_json_ready(override) for override in subgraph.overrides_applied
        ],
    }


def bundle_log(
    bundle: ParsedSceneBundle,
    *,
    archive_llm_io: bool = False,
) -> dict[str, Any]:
    """序列化 Layer 1 输出；默认不重复归档 LLM 输入文本。"""
    description = model_dump_json_ready(bundle.context.description)
    if not archive_llm_io:
        description.pop("narrative", None)
    data = {
        "context": {
            "scene_id": bundle.context.scene_id,
            "frame_token": bundle.context.frame_token,
            "location": bundle.context.location,
            "description": description,
            "ego_state": model_dump_json_ready(bundle.context.ego_state),
            "scenario_type": bundle.context.scenario_type.value,
            "scene_facts": model_dump_json_ready(bundle.context.scene_facts),
            "candidate_trajectory_ids": [
                trajectory.traj_id for trajectory in bundle.context.candidate_trajectories
            ],
            "candidate_trajectory_sources": [
                trajectory.source.value for trajectory in bundle.context.candidate_trajectories
            ],
            "trajectory_features": [
                model_dump_json_ready(feature)
                for feature in bundle.context.trajectory_features
            ],
        },
        "benchmark_labels_debug_only": model_dump_json_ready(bundle.benchmark_labels),
    }
    if archive_llm_io:
        data["scene_query"] = model_dump_json_ready(bundle.scene_query)
        data["judge_input"] = model_dump_json_ready(bundle.judge_input)
    return data


def judge_context_log(
    settings: Layer3Settings,
    bundle: ParsedSceneBundle,
    subgraph: RuleSubgraph,
) -> dict[str, Any]:
    """Build and log the prompt context that Layer 3 will also construct internally."""
    try:
        context = ContextBuilder(settings).build(bundle.judge_input, subgraph)
    except Exception as exc:
        return {
            "build_error": type(exc).__name__,
            "message": str(exc),
        }
    return model_dump_json_ready(context)


def label_log(label: AuditLabel) -> dict[str, Any]:
    """Serialize final Layer 3 output with count fields made explicit."""
    data = model_dump_json_ready(label)
    data["counts"] = {
        "vetoed": label.vetoed_count,
        "cleared": label.cleared_count,
        "uncertain": label.uncertain_count,
        "decided_by_rule_engine": label.decided_by_rule_engine_count,
        "decided_by_llm": label.decided_by_llm_count,
        "decided_by_fallback": label.decided_by_fallback_count,
        "pairwise_comparisons": label.pairwise_comparisons_count,
        "tied_pairs": label.tied_pairs_count,
    }
    return data


def run_record(
    record: dict[str, Any],
    index: int,
    *,
    seed: int,
    retriever: GraphRAGRetriever,
    judge: AuditJudge | None,
    layer3_settings: Layer3Settings,
) -> dict[str, Any]:
    """Run one RawScene record and return a structured log entry."""
    start = time.perf_counter()
    raw_scene = RawScene.model_validate(record)
    bundle = parse_scene(raw_scene, seed=seed)
    reason = admission_reason(bundle.scene_query)
    if reason is not None:
        return {
            "index": index,
            "ok": True,
            "skipped_by_admission": True,
            "skip_reason": reason,
            "duration_ms": int((time.perf_counter() - start) * 1000),
            "raw_scene": raw_scene_log(raw_scene),
            "layer1": bundle_log(
                bundle,
                archive_llm_io=layer3_settings.archive_llm_io,
            ),
        }
    subgraph = retriever.retrieve(bundle.scene_query)
    bundle = bundle.model_copy(update={
        "benchmark_labels": relabel_benchmark_labels(
            bundle.benchmark_labels,
            bundle.context.trajectory_features,
            bundle.context.scene_facts,
            {rule.node_id for rule in subgraph.active_rules},
            layer3_settings.phase_validity_horizon_s,
        )
    })
    if judge is None:
        judge = AuditJudge(layer3_settings)
    label = judge.judge(bundle.judge_input, subgraph)

    result = {
        "index": index,
        "ok": True,
        "duration_ms": int((time.perf_counter() - start) * 1000),
        "raw_scene": raw_scene_log(raw_scene),
        "layer1": bundle_log(
            bundle,
            archive_llm_io=layer3_settings.archive_llm_io,
        ),
        "layer2": subgraph_log(subgraph),
        "final_label": label_log(label),
    }
    if layer3_settings.archive_llm_io:
        result["layer3_context"] = judge_context_log(layer3_settings, bundle, subgraph)
    return result


def summarize_entries(
    entries: list[dict[str, Any]],
    *,
    input_price_per_million_usd: float | None = None,
    output_price_per_million_usd: float | None = None,
) -> dict[str, Any]:
    """Aggregate a short run summary from per-record logs."""
    status_counts: Counter[str] = Counter()
    chosen_counts: Counter[str] = Counter()
    selection_outcomes: Counter[str] = Counter()
    scenario_counts: Counter[str] = Counter()
    failures = 0
    skipped_reasons: Counter[str] = Counter()
    keyword_fallbacks = 0
    tied_pairs_count = 0
    evaluated = 0
    gt_chosen = 0
    gt_top2 = 0
    excluded_candidates = 0
    degraded_records = 0
    degraded_no_choosable_candidates = 0
    llm_by_stage: dict[str, Counter[str]] = {}
    connection_events: Counter[str] = Counter()

    for entry in entries:
        if not entry.get("ok"):
            failures += 1
            continue
        layer1 = entry["layer1"]["context"]
        scenario_counts[str(layer1["scenario_type"])] += 1
        if entry.get("skipped_by_admission"):
            skipped_reasons[str(entry.get("skip_reason", "unknown"))] += 1
            continue
        label = entry["final_label"]
        evaluated += 1
        for diagnostic in label.get("stage_diagnostics", []):
            stage = str(diagnostic["stage"])
            totals = llm_by_stage.setdefault(stage, Counter())
            totals["calls"] += int(diagnostic.get("llm_call_count", 0))
            totals["input_tokens"] += int(diagnostic.get("llm_input_tokens", 0))
            totals["output_tokens"] += int(diagnostic.get("llm_output_tokens", 0))
            totals["retries"] += int(diagnostic.get("llm_retry_count", 0))
            totals["deferred_attempts"] += len(diagnostic.get("deferred_attempts", []))
            totals["recovered_count"] += int(diagnostic.get("recovered_count", 0))
        for event in label.get("connection_events", []):
            connection_events[str(event.get("kind", "unknown"))] += 1
        status_counts[str(label["status"])] += 1
        selection_outcome = label.get("selection_outcome")
        if selection_outcome is None:
            verdicts = label.get("verdicts", [])
            has_cleared = any(
                item.get("veto", {}).get("status") == "cleared"
                for item in verdicts
            )
            selection_outcome = (
                "unavailable" if label.get("status") == "failed" or not verdicts else
                "selected" if has_cleared else
                "no_choosable_candidate"
            )
        selection_outcomes[str(selection_outcome)] += 1
        if "stage_degraded" in label.get("partial_reasons", []):
            degraded_records += 1
            if selection_outcome == "no_choosable_candidate":
                degraded_no_choosable_candidates += 1
        keyword_fallbacks += int(entry["layer2"]["retrieval_mode"] == "keyword_fallback")
        tied_pairs_count += int(label.get("tied_pairs_count", 0))
        chosen = label.get("chosen_trajectory_id")
        chosen_counts[str(chosen) if chosen is not None else "None"] += 1
        labels = entry["layer1"]["benchmark_labels_debug_only"]["labels"]
        for benchmark in labels:
            expected = benchmark["expected_verdict"]
            if expected == "exclude":
                excluded_candidates += 1
                continue
        gt_ids = {
            item["anonymous_id"]
            for item in labels
            if item["variant_type"] == "ground_truth"
        }
        gt_chosen += int(chosen in gt_ids)
        gt_top2 += int(bool(gt_ids.intersection(label.get("preference_ranking", [])[:2])))

    llm_total = sum(llm_by_stage.values(), Counter())
    telemetry_total = {
        key: int(llm_total.get(key, 0))
        for key in (
            "calls",
            "input_tokens",
            "output_tokens",
            "retries",
            "deferred_attempts",
            "recovered_count",
        )
    }
    telemetry: dict[str, Any] = {
        "by_stage": {
            stage: dict(sorted(counts.items()))
            for stage, counts in sorted(llm_by_stage.items())
        },
        "total": telemetry_total,
        "deferred_recovered": telemetry_total["recovered_count"],
        "connection_events": dict(sorted(connection_events.items())),
    }
    if input_price_per_million_usd is None or output_price_per_million_usd is None:
        telemetry["cost_status"] = "pricing_not_configured"
    else:
        telemetry["cost_status"] = "estimated"
        telemetry["estimated_cost_usd"] = (
            telemetry_total["input_tokens"] * input_price_per_million_usd
            + telemetry_total["output_tokens"] * output_price_per_million_usd
        ) / 1_000_000

    return {
        "records": len(entries),
        "failures": failures,
        "evaluated": evaluated,
        "skipped_by_admission": sum(skipped_reasons.values()),
        "skip_reasons": dict(sorted(skipped_reasons.items())),
        "keyword_fallback_rate": keyword_fallbacks / evaluated if evaluated else 0.0,
        "tied_pairs_count": tied_pairs_count,
        "gt_chosen_rate": gt_chosen / evaluated if evaluated else 0.0,
        "gt_top2_rate": gt_top2 / evaluated if evaluated else 0.0,
        "excluded_candidates": excluded_candidates,
        "degraded_records": degraded_records,
        "degraded_no_choosable_candidates": degraded_no_choosable_candidates,
        "llm_telemetry": telemetry,
        "statuses": dict(sorted(status_counts.items())),
        "selection_outcomes": dict(sorted(selection_outcomes.items())),
        "chosen": dict(sorted(chosen_counts.items())),
        "scenarios": dict(sorted(scenario_counts.items())),
        "benchmark_evaluation": evaluate_entries(entries),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI flags."""
    parser = argparse.ArgumentParser(
        description="Run Layer1->Layer2->Layer3 and write a timestamped audit log.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Prepared RawScene dataset JSON.",
    )
    parser.add_argument(
        "--eval-set",
        type=Path,
        default=None,
        help="冻结 eval_set_v1 JSON；指定后只运行其中列出的 token。",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=DEFAULT_LOGS_DIR,
        help="Directory where output_<timestamp>.json is written.",
    )
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help="Output filename prefix. Default: output.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="固定 JSONL 输出路径；文件存在时按 frame_token 续跑。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of records to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Layer 1 candidate generation seed.",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="串行执行，适合 mini 冒烟测试。",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=5,
        help="最大在飞场景数，默认 5。",
    )
    return parser.parse_args()


def _git_commit() -> str:
    """读取当前提交号；非 Git 环境返回 unknown。"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _manifest(
    args: argparse.Namespace,
    output_path: Path,
    layer2_settings: Layer2Settings,
    layer3_settings: Layer3Settings,
    retriever: GraphRAGRetriever,
    eval_set: EvalSet | None,
) -> dict[str, Any]:
    """生成可复现运行清单。"""
    prompt_dir = Path(__file__).parent / "layer3" / "prompts"
    prompt_hashes = {
        path.name: file_sha256(path) for path in sorted(prompt_dir.glob("*_system.txt"))
    }
    benchmark_builder_path = Path(__file__).parent / "layer1" / "benchmark_builder.py"
    trajectory_augmentor_path = Path(__file__).parent / "layer1" / "trajectory_augmentor.py"
    predicate_path = Path(__file__).parent / "compliance_predicates.py"
    llm_client_path = Path(__file__).parent / "layer3" / "llm_client.py"
    layer3_models_path = Path(__file__).parent / "layer3" / "models.py"
    return {
        "type": "manifest",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "graph_version": retriever._rule_graph.version,
        "model": layer3_settings.llm_model,
        "seed": args.seed,
        "input": str(args.input),
        "input_sha256": file_sha256(args.input),
        "eval_set": (
            {
                "version": eval_set.version,
                "entries": len(eval_set.entries),
                "sha256": file_sha256(args.eval_set),
                "input_sha256": eval_set.input_sha256,
                "classification_source_files": eval_set.classification_source_files,
                "classification_code_sha256": eval_set.classification_code_sha256,
            }
            if eval_set is not None and args.eval_set is not None
            else None
        ),
        "benchmark_data_version": BENCHMARK_DATA_VERSION,
        "predicate_version": PREDICATE_VERSION,
        "benchmark_builder_sha256": file_sha256(benchmark_builder_path),
        "trajectory_augmentor_sha256": file_sha256(trajectory_augmentor_path),
        "compliance_predicates_sha256": file_sha256(predicate_path),
        "llm_client_sha256": file_sha256(llm_client_path),
        "layer3_models_sha256": file_sha256(layer3_models_path),
        "rule_graph_sha256": file_sha256(layer2_settings.rule_graph_path),
        "prompt_sha256": prompt_hashes,
        "output": str(output_path),
        "serial": args.serial,
        "max_concurrency": 1 if args.serial else args.max_concurrency,
        "llm_settings": llm_settings_log(layer3_settings),
    }


def _entry_unit_id(entry: dict[str, Any]) -> str | None:
    """返回稳定 unit id；正式输入以 frame_token 为准。"""
    raw_scene = entry.get("raw_scene")
    if isinstance(raw_scene, dict):
        frame_token = raw_scene.get("frame_token")
        if isinstance(frame_token, str) and frame_token:
            return f"frame:{frame_token}"
    index = entry.get("index")
    return f"index:{index}" if isinstance(index, int) else None


def _existing_entries(output_path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    """读取已有 JSONL 记录，用于断点续跑。"""
    entries: list[dict[str, Any]] = []
    completed: set[str] = set()
    if not output_path.exists():
        return entries, completed
    for line in output_path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("type") != "record":
            continue
        entry = item.get("data")
        if not isinstance(entry, dict):
            continue
        unit_id = _entry_unit_id(entry)
        if unit_id is not None and unit_id in completed:
            continue
        entries.append(entry)
        if unit_id is not None:
            completed.add(unit_id)
    return entries, completed


def _write_jsonl(output_path: Path, item: dict[str, Any]) -> None:
    """追加写入一条 JSONL 并立即刷盘。"""
    with output_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
        file.flush()
        os.fsync(file.fileno())


def _write_record_once(output_path: Path, entry: dict[str, Any]) -> bool:
    """原子地追加一条 unit 记录；重连或并发恢复时拒绝重复写入。"""
    unit_id = _entry_unit_id(entry)
    if unit_id is None:
        raise ValueError("record is missing both frame_token and integer index")
    with output_path.open("a+", encoding="utf-8") as file:
        flock(file.fileno(), LOCK_EX)
        try:
            file.seek(0)
            for line in file:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if item.get("type") != "record":
                    continue
                existing = item.get("data")
                if isinstance(existing, dict) and _entry_unit_id(existing) == unit_id:
                    return False
            file.seek(0, 2)
            file.write(json.dumps({"type": "record", "data": entry}, ensure_ascii=False) + "\n")
            file.flush()
            os.fsync(file.fileno())
            return True
        finally:
            flock(file.fileno(), LOCK_UN)


async def _run_all(
    pending: list[tuple[int, dict[str, Any]]],
    *,
    concurrency: int,
    seed: int,
    retriever: GraphRAGRetriever,
    layer3_settings: Layer3Settings,
    output_path: Path,
) -> list[dict[str, Any]]:
    """并发处理场景，并按完成顺序增量写盘。"""
    semaphore = asyncio.Semaphore(concurrency)

    async def process(index: int, record: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            try:
                return await asyncio.to_thread(
                    run_record,
                    record,
                    index,
                    seed=seed,
                    retriever=retriever,
                    judge=None,
                    layer3_settings=layer3_settings,
                )
            except Exception as exc:
                return {
                    "index": index,
                    "ok": False,
                    "raw_scene": {
                        "frame_token": record.get("frame_token", ""),
                    },
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                }

    tasks = [asyncio.create_task(process(index, record)) for index, record in pending]
    entries: list[dict[str, Any]] = []
    previous_finished = time.perf_counter()
    for task in asyncio.as_completed(tasks):
        entry = await task
        now = time.perf_counter()
        entry["gap_since_previous_record_ms"] = int((now - previous_finished) * 1000)
        previous_finished = now
        written = _write_record_once(output_path, entry)
        if written:
            entries.append(entry)
        state = "skipped" if entry.get("skipped_by_admission") else (
            "ok" if entry.get("ok") else "failed"
        )
        suffix = "" if written else " duplicate_skipped"
        print(f"[{entry['index']:02d}] {state}{suffix}", flush=True)
    return entries


def main() -> None:
    """命令行入口。"""
    args = parse_args()
    if args.max_concurrency < 1:
        raise SystemExit("--max-concurrency must be >= 1")
    records = load_records(args.input)
    eval_set = load_eval_set(args.eval_set, require_frozen=True) if args.eval_set else None
    if eval_set is not None:
        assert_classification_code_matches(eval_set)
        assert_input_file_matches(eval_set, args.input)
        records = select_frozen_eval_records(records, eval_set)
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be >= 1")
        records = records[: args.limit]

    output_path = args.output or make_output_path(args.logs_dir, args.output_prefix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_exists = output_path.exists()
    existing_entries, completed = _existing_entries(output_path)
    layer2_settings = Layer2Settings()
    layer3_settings = Layer3Settings()
    retriever = GraphRAGRetriever(layer2_settings)

    if not output_exists:
        _write_jsonl(
            output_path,
            _manifest(
                args,
                output_path,
                layer2_settings,
                layer3_settings,
                retriever,
                eval_set,
            ),
        )
    elif existing_entries:
        _write_jsonl(
            output_path,
            {
                "type": "event",
                "event": "run_resumed",
                "occurred_at": datetime.now().isoformat(timespec="seconds"),
                "completed_units": len(existing_entries),
            },
        )

    pending = [
        (index, record)
        for index, record in enumerate(records)
        if f"frame:{record.get('frame_token')}" not in completed
    ]
    run_start = time.perf_counter()
    asyncio.run(
        _run_all(
            pending,
            concurrency=1 if args.serial else args.max_concurrency,
            seed=args.seed,
            retriever=retriever,
            layer3_settings=layer3_settings,
            output_path=output_path,
        )
    )
    entries, _ = _existing_entries(output_path)
    _write_jsonl(
        output_path,
        {
            "type": "summary",
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "batch_wall_duration_ms": int((time.perf_counter() - run_start) * 1000),
            "llm_429_count": get_llm_429_count(),
            "summary": summarize_entries(
                entries,
                input_price_per_million_usd=(
                    layer3_settings.llm_input_price_per_million_usd
                ),
                output_price_per_million_usd=(
                    layer3_settings.llm_output_price_per_million_usd
                ),
            ),
        },
    )
    print(f"wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
