from __future__ import annotations

import json

from experiments.ablation_2x2.citations import CitationValidator, RuleCatalog
from experiments.ablation_2x2.conditions import condition_by_name
from experiments.ablation_2x2.run_ablation import (
    DEFAULT_SCENARIOS,
    RULE_GRAPH,
    DryRunBackend,
    run_one,
)


class _AlwaysInvalidBackend(DryRunBackend):
    def complete(self, *, condition, scenario, system_prompt, user_prompt, attempt):
        del system_prompt, user_prompt, attempt
        response = scenario["dry_run_responses"][condition.name][0]
        return json.loads(json.dumps(response)), 100, 1


class _InvalidCitationRepairRankingBackend(DryRunBackend):
    def complete(self, *, condition, scenario, system_prompt, user_prompt, attempt):
        del system_prompt, user_prompt
        responses = scenario["dry_run_responses"][condition.name]
        response = json.loads(json.dumps(responses[min(attempt, len(responses) - 1)]))
        if attempt == 1:
            response["preference_ranking"] = response["preference_ranking"][:-1]
        return response, 100, 1


def test_observe_records_violation_without_retry_or_exception() -> None:
    scenario = json.loads(DEFAULT_SCENARIOS.read_text(encoding="utf-8"))["scenarios"][0]
    record, _system, _prompt = run_one(
        condition=condition_by_name("no_validation"),
        scenario=scenario,
        backend=DryRunBackend(),
        catalog=RuleCatalog.from_yaml(RULE_GRAPH),
        dry_run=True,
    )

    assert record["citation_check_first"] == "violated"
    assert record["citation_check_final"] == "violated"
    assert record["retry_count"] == 0
    assert record["response_first"] == record["response_final"]
    assert any(not item["valid"] for item in record["citations_final"])


def test_observe_validator_returns_assessment_instead_of_raising() -> None:
    validator = CitationValidator(
        mode="observe",
        graph_rag=True,
        allowed_rule_ids={"R-SIG-01"},
        catalog=RuleCatalog.from_yaml(RULE_GRAPH),
    )
    assessments = validator.assess(cited_rule_ids=["R-FAKE-01"], cited_provisions=[])
    assert [item.valid for item in assessments] == [False]


def test_enforce_records_exhausted_repair_without_losing_response() -> None:
    scenario = json.loads(DEFAULT_SCENARIOS.read_text(encoding="utf-8"))["scenarios"][0]
    record, _system, _prompt = run_one(
        condition=condition_by_name("no_rag"),
        scenario=scenario,
        backend=_AlwaysInvalidBackend(),
        catalog=RuleCatalog.from_yaml(RULE_GRAPH),
        dry_run=True,
    )

    assert record["retry_count"] == 1
    assert record["citation_check_final"] == "violated"
    assert record["enforcement_exhausted"] is True
    assert record["response_final"]


def test_citation_repair_response_gets_audited_structure_retry() -> None:
    scenario = json.loads(DEFAULT_SCENARIOS.read_text(encoding="utf-8"))["scenarios"][0]
    record, _system, _prompt = run_one(
        condition=condition_by_name("no_rag"),
        scenario=scenario,
        backend=_InvalidCitationRepairRankingBackend(),
        catalog=RuleCatalog.from_yaml(RULE_GRAPH),
        dry_run=True,
    )

    assert record["retry_count"] == 1
    assert record["schema_retry_count"] == 1
    assert record["schema_failures"][0]["stage"] == "citation_repair"
    assert record["response_final"]
