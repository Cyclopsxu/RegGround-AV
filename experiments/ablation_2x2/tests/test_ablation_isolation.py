from __future__ import annotations

import hashlib
import json
import threading
import time

from experiments.ablation_2x2.conditions import CONDITIONS, condition_by_name
from experiments.ablation_2x2.run_ablation import (
    DEFAULT_SCENARIOS,
    DryRunBackend,
    _validate_response,
    run,
)


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_outputs_are_isolated_manifested_and_resumable(tmp_path) -> None:
    output = tmp_path / "DRY_RUN"
    common = dict(
        scenario_path=DEFAULT_SCENARIOS,
        output_dir=output,
        dry_run=True,
        confirm_gold_final=False,
        backend=DryRunBackend(),
        skip_workbook=True,
    )
    assert run(**common, stop_after=1) is False
    assert run(**common) is True
    assert run(**common) is True

    manifest_hashes = set()
    for condition in CONDITIONS:
        condition_dir = output / condition.name
        manifest = json.loads((condition_dir / "manifest.json").read_text(encoding="utf-8"))
        records = (condition_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
        assert manifest["disable_rule_engine"] is True
        assert manifest["digest_version"] == "v_a"
        assert len(manifest["prompt_sha256"]) == 3
        assert len(records) == 3
        manifest_hashes.add(_sha256(condition_dir / "manifest.json"))
    assert len(manifest_hashes) == 4
    assert len(list((output / "prompt_diffs").glob("*.diff"))) == 6
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8"))["complete"] is True


def test_dry_run_cannot_write_into_formal_output_path(tmp_path) -> None:
    try:
        run(
            scenario_path=DEFAULT_SCENARIOS,
            output_dir=tmp_path / "formal",
            dry_run=True,
            confirm_gold_final=False,
            backend=DryRunBackend(),
            skip_workbook=True,
        )
    except ValueError as error:
        assert "DRY_RUN" in str(error)
    else:
        raise AssertionError("dry-run/formal output isolation was not enforced")


def test_blinded_inference_requires_isolated_output_and_omits_gold(tmp_path) -> None:
    payload = json.loads(DEFAULT_SCENARIOS.read_text(encoding="utf-8"))
    payload["inference_only"] = True
    payload["gold_finalized"] = False
    for scenario in payload["scenarios"]:
        scenario.pop("gold", None)
    scenarios = tmp_path / "inference.json"
    scenarios.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "BLINDED_INFERENCE"
    assert run(
        scenario_path=scenarios,
        output_dir=output,
        dry_run=False,
        confirm_gold_final=False,
        backend=DryRunBackend(),
        stop_after=1,
        skip_workbook=True,
        blinded_inference=True,
    ) is False
    for condition in CONDITIONS:
        record = json.loads(
            (output / condition.name / "records.jsonl").read_text(encoding="utf-8")
        )
        assert "gold" not in record
        assert json.loads(
            (output / condition.name / "manifest.json").read_text(encoding="utf-8")
        )["marker"] == "BLINDED_INFERENCE"


def test_response_verdicts_are_canonicalized_to_frozen_candidate_order() -> None:
    scenario = json.loads(DEFAULT_SCENARIOS.read_text(encoding="utf-8"))["scenarios"][0]
    response = json.loads(
        json.dumps(scenario["dry_run_responses"]["full"][0], ensure_ascii=False)
    )
    response["verdicts"].reverse()

    _validate_response(response, condition_by_name("full"), scenario)

    assert [item["trajectory_id"] for item in response["verdicts"]] == [
        item["trajectory_id"] for item in scenario["candidates"]
    ]


class _RankingRepairBackend(DryRunBackend):
    def complete(self, *, condition, scenario, system_prompt, user_prompt, attempt):
        del system_prompt, user_prompt
        response = json.loads(
            json.dumps(scenario["dry_run_responses"][condition.name][-1])
        )
        if attempt == 0:
            response["preference_ranking"] = response["preference_ranking"][:-1]
        return response, 100, 1


def test_invalid_ranking_gets_one_audited_structure_retry(tmp_path) -> None:
    payload = json.loads(DEFAULT_SCENARIOS.read_text(encoding="utf-8"))
    payload["inference_only"] = True
    payload["gold_finalized"] = False
    for scenario in payload["scenarios"]:
        scenario.pop("gold", None)
    scenarios = tmp_path / "inference.json"
    scenarios.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "BLINDED_INFERENCE"

    assert run(
        scenario_path=scenarios,
        output_dir=output,
        dry_run=False,
        confirm_gold_final=False,
        backend=_RankingRepairBackend(),
        stop_after=1,
        skip_workbook=True,
        blinded_inference=True,
    ) is False
    record = json.loads((output / "full" / "records.jsonl").read_text(encoding="utf-8"))
    assert record["schema_retry_count"] == 1
    assert record["schema_failures"][0]["error"] == (
        "preference_ranking must contain every candidate exactly once"
    )


class _ConcurrencyBackend(DryRunBackend):
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def complete(self, **kwargs):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.02)
            return super().complete(**kwargs)
        finally:
            with self.lock:
                self.active -= 1


def test_workers_parallelize_conditions_with_separate_records(tmp_path) -> None:
    backend = _ConcurrencyBackend()
    output = tmp_path / "DRY_RUN"
    assert run(
        scenario_path=DEFAULT_SCENARIOS,
        output_dir=output,
        dry_run=True,
        confirm_gold_final=False,
        backend=backend,
        stop_after=1,
        skip_workbook=True,
        workers=4,
    ) is False
    assert backend.max_active > 1
    assert all(
        len((output / condition.name / "records.jsonl").read_text().splitlines()) == 1
        for condition in CONDITIONS
    )
