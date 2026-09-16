"""Run the isolated 2×2 ablation and produce paired, resumable artifacts."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from experiments.ablation_2x2.citations import (
    CitationValidator,
    RuleCatalog,
    response_citations,
)
from experiments.ablation_2x2.conditions import CONDITIONS, AblationCondition
from experiments.ablation_2x2.metrics import write_metrics
from experiments.ablation_2x2.prompts import build_prompt

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_SCENARIOS = MODULE_DIR / "fixtures/dry_run_scenarios.json"
DEFAULT_OUTPUT = ROOT / "results/ablation_2x2/DRY_RUN"
RULE_GRAPH = ROOT / "data/layer2/rule_graph.yaml"
WORKBOOK_BUILDER = MODULE_DIR / "build_citation_sheet.mjs"


class _CitationItem(BaseModel):
    trajectory_id: str | None = None
    preferred_id: str | None = None
    dispreferred_id: str | None = None
    status: str | None = None
    reason: str
    cited_rule_ids: list[str] = Field(default_factory=list)
    cited_provisions: list[str] = Field(default_factory=list)


class _DecisionResponse(BaseModel):
    verdicts: list[_CitationItem]
    preference_ranking: list[str]
    chosen_trajectory_id: str | None
    preference_pairs: list[_CitationItem] = Field(default_factory=list)


class Backend(Protocol):
    model_config: dict[str, Any]

    def complete(
        self,
        *,
        condition: AblationCondition,
        scenario: dict[str, Any],
        system_prompt: str,
        user_prompt: str,
        attempt: int,
    ) -> tuple[dict[str, Any], int, int]: ...


class DryRunBackend:
    """Deterministic response replay; fake labels never enter prompts."""

    model_config = {
        "backend": "deterministic_dry_run",
        "model": "FAKE_MODEL",
        "thinking": "disabled",
        "temperatures": [0.0, 0.0, 0.2],
        "timeout_seconds": 0,
        "retry_parameters": {"citation_retry": 1},
    }

    def complete(
        self,
        *,
        condition: AblationCondition,
        scenario: dict[str, Any],
        system_prompt: str,
        user_prompt: str,
        attempt: int,
    ) -> tuple[dict[str, Any], int, int]:
        del system_prompt, user_prompt
        responses = scenario["dry_run_responses"][condition.name]
        response = responses[min(attempt, len(responses) - 1)]
        return json.loads(json.dumps(response, ensure_ascii=False)), 100 + attempt * 20, 1


class LiveBackend:
    """Direct structured LLM backend, loaded only after frozen-gold confirmation."""

    def __init__(self) -> None:
        from src.layer3.llm_client import LLMClient
        from src.layer3.models import Layer3Settings

        self._settings = Layer3Settings()
        self._client = LLMClient(self._settings)
        self.model_config = {
            "backend": "live_structured",
            "provider": self._settings.llm_provider,
            "model": self._settings.llm_model,
            "model_version": self._settings.llm_model_version,
            "thinking": self._settings.llm_thinking_mode,
            "temperatures": [
                self._settings.hard_filter_temperature,
                self._settings.pairwise_ranker_temperature,
                self._settings.label_temperature,
            ],
            "timeout_seconds": self._settings.llm_timeout_seconds,
            "structured_max_tokens": self._settings.llm_structured_max_tokens,
            "text_max_tokens": self._settings.llm_text_max_tokens,
            "retry_parameters": {
                "llm_max_retries": self._settings.llm_max_retries,
                "citation_retry": 1,
            },
        }

    def complete(
        self,
        *,
        condition: AblationCondition,
        scenario: dict[str, Any],
        system_prompt: str,
        user_prompt: str,
        attempt: int,
    ) -> tuple[dict[str, Any], int, int]:
        del condition, scenario, attempt
        started = time.perf_counter()
        parsed, tokens = self._client.complete_structured(
            system_prompt,
            user_prompt,
            _DecisionResponse,
            temperature=self._settings.hard_filter_temperature,
            max_retries=self._settings.llm_max_retries,
        )
        return parsed.model_dump(), tokens, int((time.perf_counter() - started) * 1000)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_scenarios(
    path: Path,
    *,
    dry_run: bool,
    confirm_gold_final: bool,
    blinded_inference: bool = False,
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_version") != "ablation_2x2_v1.0":
        raise ValueError("scenario list protocol_version must be ablation_2x2_v1.0")
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("scenario list must contain a non-empty scenarios array")
    frames = [str(item.get("frame_token", "")) for item in scenarios]
    if any(not frame for frame in frames) or len(frames) != len(set(frames)):
        raise ValueError("scenario frame_token values must be present and unique")
    if dry_run and not 3 <= len(scenarios) <= 5:
        raise ValueError("dry-run requires 3–5 scenarios")
    if blinded_inference and not payload.get("inference_only"):
        raise ValueError("blinded inference requires inference_only=true")
    if not dry_run and not blinded_inference and (
        not payload.get("gold_finalized") or not confirm_gold_final
    ):
        raise ValueError("formal run requires gold_finalized=true and --confirm-gold-final")
    return scenarios


def _validate_response(
    response: dict[str, Any], condition: AblationCondition, scenario: dict[str, Any]
) -> None:
    candidate_ids = [item["trajectory_id"] for item in scenario["candidates"]]
    verdicts = response.get("verdicts")
    if not isinstance(verdicts, list):
        raise ValueError("response.verdicts must be a list")
    verdict_ids = [item.get("trajectory_id") for item in verdicts if isinstance(item, dict)]
    if len(verdict_ids) == len(candidate_ids) and set(verdict_ids) == set(candidate_ids):
        by_id = {item["trajectory_id"]: item for item in verdicts}
        response["verdicts"] = verdicts = [by_id[trajectory_id] for trajectory_id in candidate_ids]
    else:
        raise ValueError("response verdicts must preserve the frozen candidate order")
    if any(item.get("status") not in {"cleared", "vetoed", "uncertain"} for item in verdicts):
        raise ValueError("response contains an invalid verdict status")
    ranking = response.get("preference_ranking")
    if (
        not isinstance(ranking, list)
        or set(ranking) != set(candidate_ids)
        or len(ranking) != len(candidate_ids)
    ):
        raise ValueError("preference_ranking must contain every candidate exactly once")
    chosen = response.get("chosen_trajectory_id")
    if chosen is not None and chosen not in candidate_ids:
        raise ValueError("chosen_trajectory_id is not a candidate")
    citation_field = "cited_rule_ids" if condition.graph_rag else "cited_provisions"
    for item in verdicts:
        if not isinstance(item.get(citation_field), list):
            raise ValueError(f"every verdict must contain {citation_field}")


def _attach_event_metadata(
    events: list[dict[str, Any]],
    *,
    condition: AblationCondition,
    scenario: dict[str, Any],
    attempt_label: str,
    dry_run: bool,
) -> list[dict[str, Any]]:
    gold = scenario.get("fake_gold", {}) if dry_run else {}
    result: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        trajectory_id = (
            event["owner"].split(":", 1)[1] if event["owner"].startswith("verdict:") else None
        )
        if condition.graph_rag:
            by_trajectory = gold.get("applicable_rule_ids_by_trajectory", {})
            applicable = set(by_trajectory.get(trajectory_id, gold.get("applicable_rule_ids", [])))
        else:
            by_trajectory = gold.get("applicable_provision_keys_by_trajectory", {})
            applicable = set(
                by_trajectory.get(trajectory_id, gold.get("applicable_provision_keys", []))
            )
        accuracy = bool(event["normalized_key"] in applicable) if dry_run else None
        result.append(
            {
                **event,
                "citation_id": (
                    f"{condition.name}|{scenario['frame_token']}|{attempt_label}|"
                    f"{event['owner']}|{index}"
                ),
                "attempt": attempt_label,
                "accuracy": accuracy,
            }
        )
    return result


def _decision_citation_counts(response: dict[str, Any], validator: CitationValidator) -> list[int]:
    counts: list[int] = []
    for item in response["verdicts"]:
        counts.append(
            len(
                validator.assess(
                    cited_rule_ids=[str(value) for value in item.get("cited_rule_ids", [])],
                    cited_provisions=[str(value) for value in item.get("cited_provisions", [])],
                )
            )
        )
    return counts


def run_one(
    *,
    condition: AblationCondition,
    scenario: dict[str, Any],
    backend: Backend,
    catalog: RuleCatalog,
    dry_run: bool,
    blinded_inference: bool = False,
) -> tuple[dict[str, Any], str, str]:
    system_prompt, user_prompt = build_prompt(condition, scenario)
    allowed_ids = {str(rule["rule_id"]) for rule in scenario.get("retrieved_rules", [])}
    validator = CitationValidator(
        mode=condition.validation_mode,
        graph_rag=condition.graph_rag,
        allowed_rule_ids=allowed_ids,
        catalog=catalog,
    )
    first, tokens, duration_ms = backend.complete(
        condition=condition,
        scenario=scenario,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        attempt=0,
    )
    schema_failures: list[dict[str, Any]] = []
    try:
        _validate_response(first, condition, scenario)
    except ValueError as error:
        schema_failures.append({"error": str(error), "response": first})
        candidate_ids = [item["trajectory_id"] for item in scenario["candidates"]]
        schema_repair_prompt = (
            user_prompt
            + "\n\n上次 JSON 未通过结构校验："
            + str(error)
            + "。verdicts 与 preference_ranking 必须各自恰好覆盖以下候选一次："
            + json.dumps(candidate_ids, ensure_ascii=False)
            + "。请保留你的实质判断并重新输出完整 JSON。"
        )
        first, retry_tokens, retry_duration = backend.complete(
            condition=condition,
            scenario=scenario,
            system_prompt=system_prompt,
            user_prompt=schema_repair_prompt,
            attempt=1,
        )
        tokens += retry_tokens
        duration_ms += retry_duration
        _validate_response(first, condition, scenario)
    first_assessments = response_citations(first, validator)
    first_passed = all(item["valid"] for item in first_assessments)
    final = first
    retry_count = 0
    if condition.validation_mode == "enforce" and not first_passed:
        invalid = [item["original_text"] for item in first_assessments if not item["valid"]]
        repair_prompt = (
            user_prompt
            + "\n\n上次引用未通过存在性校验："
            + json.dumps(invalid, ensure_ascii=False)
            + "。请仅修复引用并重新输出完整 JSON。"
        )
        final, retry_tokens, retry_duration = backend.complete(
            condition=condition,
            scenario=scenario,
            system_prompt=system_prompt,
            user_prompt=repair_prompt,
            attempt=1 + len(schema_failures),
        )
        tokens += retry_tokens
        duration_ms += retry_duration
        retry_count = 1
        try:
            _validate_response(final, condition, scenario)
        except ValueError as error:
            schema_failures.append(
                {"stage": "citation_repair", "error": str(error), "response": final}
            )
            candidate_ids = [item["trajectory_id"] for item in scenario["candidates"]]
            combined_repair_prompt = (
                repair_prompt
                + "\n\n修复后的 JSON 仍未通过结构校验："
                + str(error)
                + "。verdicts 与 preference_ranking 必须各自恰好覆盖以下候选一次："
                + json.dumps(candidate_ids, ensure_ascii=False)
                + "。请保留实质判断和已修复引用，重新输出完整 JSON。"
            )
            final, schema_tokens, schema_duration = backend.complete(
                condition=condition,
                scenario=scenario,
                system_prompt=system_prompt,
                user_prompt=combined_repair_prompt,
                attempt=2 + len(schema_failures),
            )
            tokens += schema_tokens
            duration_ms += schema_duration
            _validate_response(final, condition, scenario)
    final_assessments = response_citations(final, validator)
    final_passed = all(item["valid"] for item in final_assessments)

    first_events = _attach_event_metadata(
        first_assessments,
        condition=condition,
        scenario=scenario,
        attempt_label="first",
        dry_run=dry_run,
    )
    final_events = _attach_event_metadata(
        final_assessments,
        condition=condition,
        scenario=scenario,
        attempt_label="final",
        dry_run=dry_run,
    )
    record = {
        "protocol_version": "ablation_2x2_v1.0",
        "dry_run": dry_run,
        "condition": condition.name,
        "scene_id": scenario["scene_id"],
        "frame_token": scenario["frame_token"],
        "scenario_type": scenario["scenario_type"],
        "difficulty": scenario["difficulty"],
        "digest_version": condition.digest_version,
        "disable_rule_engine": condition.disable_rule_engine,
        "candidate_order": [item["trajectory_id"] for item in scenario["candidates"]],
        "prompt_sha256": sha256_text(system_prompt + "\n" + user_prompt),
        "response_first": first,
        "response_final": final,
        "citation_check_first": "passed" if first_passed else "violated",
        "citation_check_final": "passed" if final_passed else "violated",
        "citations_first": first_events,
        "citations_final": final_events,
        "decision_citation_counts": _decision_citation_counts(final, validator),
        "retry_count": retry_count,
        "tokens": tokens,
        "duration_ms": duration_ms,
        "schema_retry_count": len(schema_failures),
        "schema_failures": schema_failures,
    }
    if condition.validation_mode == "enforce" and not final_passed:
        record["enforcement_exhausted"] = True
    if not blinded_inference:
        record["gold"] = scenario["fake_gold"] if dry_run else scenario["gold"]
    return record, system_prompt, user_prompt


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    frames = [record["frame_token"] for record in records]
    if len(frames) != len(set(frames)):
        raise ValueError(f"duplicate frame_token in {path}")
    return records


def _manifest(
    condition: AblationCondition,
    *,
    scenario_path: Path,
    prompt_hashes: dict[str, str],
    backend: Backend,
    dry_run: bool,
    blinded_inference: bool,
    workers: int,
) -> dict[str, Any]:
    return {
        "protocol_version": "ablation_2x2_v1.0",
        "dry_run": dry_run,
        "marker": (
            "DRY_RUN" if dry_run else "BLINDED_INFERENCE" if blinded_inference else "FORMAL"
        ),
        "condition": condition.name,
        "graph_rag": condition.graph_rag,
        "citation_validation_mode": condition.validation_mode,
        "disable_rule_engine": condition.disable_rule_engine,
        "digest_version": condition.digest_version,
        "scenario_list_sha256": sha256_file(scenario_path),
        "rule_graph_sha256": sha256_file(RULE_GRAPH),
        "prompt_sha256": prompt_hashes,
        "model_config": backend.model_config,
        "workers": workers,
    }


def _write_or_validate_manifest(path: Path, manifest: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError(f"resume manifest mismatch: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _archive_prompt_diffs(output_dir: Path, prompts: dict[str, dict[str, str]]) -> None:
    diff_dir = output_dir / "prompt_diffs"
    diff_dir.mkdir(parents=True, exist_ok=True)
    names = [condition.name for condition in CONDITIONS]
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            chunks: list[str] = []
            for frame in sorted(prompts[left]):
                chunks.extend(
                    difflib.unified_diff(
                        prompts[left][frame].splitlines(),
                        prompts[right][frame].splitlines(),
                        fromfile=f"{left}/{frame}",
                        tofile=f"{right}/{frame}",
                        lineterm="",
                    )
                )
            (diff_dir / f"{left}_vs_{right}.diff").write_text(
                "\n".join(chunks) + "\n", encoding="utf-8"
            )


def _grading_rows(
    records_by_condition: dict[str, list[dict[str, Any]]], *, dry_run: bool
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in [item.name for item in CONDITIONS]:
        for record in records_by_condition[condition]:
            for event in record["citations_first"] + record["citations_final"]:
                rows.append(
                    {
                        "citation_id": event["citation_id"],
                        "condition": condition,
                        "scene_id": record["scene_id"],
                        "frame_token": record["frame_token"],
                        "attempt": event["attempt"],
                        "owner": event["owner"],
                        "original_text": event["original_text"],
                        "law_name": event["law_name"],
                        "article_number": event["article_number"],
                        "normalized_key": event["normalized_key"],
                        "automated_validity": event["valid"],
                        "existence_rating": ("是" if event["valid"] else "否")
                        if dry_run
                        else "待评级",
                        "content_fidelity_rating": ("是" if event.get("accuracy") else "否")
                        if dry_run
                        else "待评级",
                        "applicability_rating": ("是" if event.get("accuracy") else "否")
                        if dry_run
                        else "待评级",
                        "notes": "DRY_RUN fake rating" if dry_run else "",
                    }
                )
    return rows


def build_workbook(
    output_dir: Path,
    rows_path: Path,
    *,
    artifact_node: Path,
    artifact_node_modules: Path,
) -> None:
    if not artifact_node.is_file() or not artifact_node_modules.is_dir():
        raise FileNotFoundError("artifact-tool Node runtime paths are invalid")
    runtime = Path(tempfile.mkdtemp(prefix="ablation_xlsx_", dir=output_dir))
    try:
        (runtime / "node_modules").symlink_to(artifact_node_modules, target_is_directory=True)
        runtime_builder = runtime / WORKBOOK_BUILDER.name
        shutil.copy2(WORKBOOK_BUILDER, runtime_builder)
        subprocess.run(
            [
                str(artifact_node),
                str(runtime_builder),
                str(rows_path),
                str(output_dir / "citation_grading_sheet.xlsx"),
                str(output_dir / "citation_grading_sheet_preview.png"),
            ],
            check=True,
            cwd=runtime,
        )
    finally:
        shutil.rmtree(runtime)


def run(
    *,
    scenario_path: Path,
    output_dir: Path,
    dry_run: bool,
    confirm_gold_final: bool,
    backend: Backend,
    stop_after: int | None = None,
    artifact_node: Path | None = None,
    artifact_node_modules: Path | None = None,
    skip_workbook: bool = False,
    blinded_inference: bool = False,
    workers: int = 1,
) -> bool:
    if not 1 <= workers <= len(CONDITIONS):
        raise ValueError(f"workers must be between 1 and {len(CONDITIONS)}")
    if dry_run and "DRY_RUN" not in output_dir.parts:
        raise ValueError("dry-run output path must contain a DRY_RUN directory component")
    if blinded_inference and "BLINDED_INFERENCE" not in output_dir.parts:
        raise ValueError(
            "blinded inference output path must contain a BLINDED_INFERENCE directory component"
        )
    scenarios = load_scenarios(
        scenario_path,
        dry_run=dry_run,
        confirm_gold_final=confirm_gold_final,
        blinded_inference=blinded_inference,
    )
    selected = scenarios[:stop_after] if stop_after is not None else scenarios
    catalog = RuleCatalog.from_yaml(RULE_GRAPH)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts: dict[str, dict[str, str]] = {condition.name: {} for condition in CONDITIONS}
    prompt_hashes: dict[str, dict[str, str]] = {condition.name: {} for condition in CONDITIONS}

    for condition in CONDITIONS:
        for scenario in scenarios:
            system, user = build_prompt(condition, scenario)
            combined = system + "\n" + user
            prompts[condition.name][scenario["frame_token"]] = combined
            prompt_hashes[condition.name][scenario["frame_token"]] = sha256_text(combined)
        manifest = _manifest(
            condition,
            scenario_path=scenario_path,
            prompt_hashes=prompt_hashes[condition.name],
            backend=backend,
            dry_run=dry_run,
            blinded_inference=blinded_inference,
            workers=workers,
        )
        _write_or_validate_manifest(output_dir / condition.name / "manifest.json", manifest)

    _archive_prompt_diffs(output_dir, prompts)
    for scenario in selected:
        pending: list[AblationCondition] = []
        for condition in CONDITIONS:
            records_path = output_dir / condition.name / "records.jsonl"
            completed = {record["frame_token"] for record in _read_records(records_path)}
            if scenario["frame_token"] in completed:
                continue
            pending.append(condition)
        errors: list[Exception] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    run_one,
                    condition=condition,
                    scenario=scenario,
                    backend=backend,
                    catalog=catalog,
                    dry_run=dry_run,
                    blinded_inference=blinded_inference,
                ): condition
                for condition in pending
            }
            for future in as_completed(futures):
                condition = futures[future]
                try:
                    record, _system, _user = future.result()
                except Exception as error:  # preserve other completed condition records
                    errors.append(error)
                    continue
                _append_jsonl(output_dir / condition.name / "records.jsonl", record)
        if errors:
            raise errors[0]

    records_by_condition = {
        condition.name: _read_records(output_dir / condition.name / "records.jsonl")
        for condition in CONDITIONS
    }
    expected_frames = {scenario["frame_token"] for scenario in scenarios}
    complete = all(
        {record["frame_token"] for record in records} == expected_frames
        for records in records_by_condition.values()
    )
    global_manifest = {
        "protocol_version": "ablation_2x2_v1.0",
        "dry_run": dry_run,
        "marker": (
            "DRY_RUN" if dry_run else "BLINDED_INFERENCE" if blinded_inference else "FORMAL"
        ),
        "scoring_status": "deferred_human_gold" if blinded_inference else "complete",
        "complete": complete,
        "scenario_list_sha256": sha256_file(scenario_path),
        "condition_prompt_sha256": prompt_hashes,
        "frozen_model_config": backend.model_config,
        "workers": workers,
        "conditions": [
            {
                "name": item.name,
                "graph_rag": item.graph_rag,
                "citation_validation_mode": item.validation_mode,
                "disable_rule_engine": item.disable_rule_engine,
                "digest_version": item.digest_version,
            }
            for item in CONDITIONS
        ],
        "generated_at": datetime.now(UTC).isoformat(),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(global_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not complete:
        return False

    rows = _grading_rows(records_by_condition, dry_run=dry_run)
    rows_path = output_dir / "citation_grading_rows.json"
    rows_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not blinded_inference:
        write_metrics(output_dir, records_by_condition, dry_run=dry_run)
    if not skip_workbook:
        if artifact_node is None or artifact_node_modules is None:
            raise ValueError("workbook generation requires artifact-tool Node runtime paths")
        build_workbook(
            output_dir,
            rows_path,
            artifact_node=artifact_node,
            artifact_node_modules=artifact_node_modules,
        )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the isolated RegGround-AV 2×2 ablation")
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-gold-final", action="store_true")
    parser.add_argument("--backend", choices=("dry", "live"), default="dry")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--artifact-node", type=Path)
    parser.add_argument("--artifact-node-modules", type=Path)
    parser.add_argument("--skip-workbook", action="store_true")
    parser.add_argument("--blinded-inference", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.backend == "dry" and not args.dry_run:
        parser.error("--backend dry is only valid with --dry-run")
    backend: Backend = DryRunBackend() if args.backend == "dry" else LiveBackend()
    complete = run(
        scenario_path=args.scenarios.resolve(),
        output_dir=args.output_dir.resolve(),
        dry_run=args.dry_run,
        confirm_gold_final=args.confirm_gold_final,
        backend=backend,
        stop_after=args.stop_after,
        artifact_node=args.artifact_node.resolve() if args.artifact_node else None,
        artifact_node_modules=(
            args.artifact_node_modules.resolve() if args.artifact_node_modules else None
        ),
        skip_workbook=args.skip_workbook,
        blinded_inference=args.blinded_inference,
        workers=args.workers,
    )
    print(f"{'complete' if complete else 'partial'}: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
