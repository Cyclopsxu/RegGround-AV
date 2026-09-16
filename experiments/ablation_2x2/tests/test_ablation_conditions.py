from __future__ import annotations

import json

from experiments.ablation_2x2.conditions import CONDITIONS, condition_by_name
from experiments.ablation_2x2.prompts import build_prompt
from experiments.ablation_2x2.run_ablation import DEFAULT_SCENARIOS


def _scenario() -> dict:
    return json.loads(DEFAULT_SCENARIOS.read_text(encoding="utf-8"))["scenarios"][0]


def test_four_condition_flags_and_frozen_controls() -> None:
    assert [(item.name, item.graph_rag, item.validation_mode) for item in CONDITIONS] == [
        ("full", True, "enforce"),
        ("no_validation", True, "observe"),
        ("no_rag", False, "enforce"),
        ("baseline", False, "observe"),
    ]
    assert all(item.disable_rule_engine for item in CONDITIONS)
    assert all(item.digest_version == "v_a" for item in CONDITIONS)


def test_no_rag_prompt_removes_rule_text_and_preserves_digest_and_candidates() -> None:
    scenario = _scenario()
    _system, full_prompt = build_prompt(condition_by_name("full"), scenario)
    _system, no_rag_prompt = build_prompt(condition_by_name("no_rag"), scenario)

    assert scenario["digest_v_a"] in full_prompt and scenario["digest_v_a"] in no_rag_prompt
    assert "R-SIG-01" in full_prompt
    assert "R-SIG-01" not in no_rag_prompt
    assert "RoadTrafficSafetyLaw" not in no_rag_prompt
    assert "rule_id" not in no_rag_prompt
    assert "cited_provisions" in no_rag_prompt
    for candidate in scenario["candidates"]:
        assert candidate["features"] in full_prompt and candidate["features"] in no_rag_prompt
