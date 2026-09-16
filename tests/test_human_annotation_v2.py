import pytest

from human_annotation_v2.code.export_wording_sensitive_supplement import (
    FRAME_0225,
    FRAME_575B,
    SUPPLEMENT_KEYS,
)
from human_annotation_v2.code.validate_returns import (
    normalize_rule_ids,
    validate_preference_row,
    validate_row,
)
from scripts.run_shadow_no_rule_engine import (
    _validate_completed,
    build_report,
    phase_continuity_statement,
)
from src.replay_context import (
    ReplayMismatchError,
    assert_frozen_inputs_match,
    file_sha256,
)

RULES = {
    "R-SIG-01": "hard",
    "R-SIG-03": "soft",
    "R-YLD-01": "hard",
}


def test_wording_sensitive_supplement_has_exact_requested_candidates() -> None:
    assert len(SUPPLEMENT_KEYS) == len(set(SUPPLEMENT_KEYS)) == 5
    assert sum(frame == FRAME_0225 for frame, _trajectory in SUPPLEMENT_KEYS) == 4
    assert sum(frame == FRAME_575B for frame, _trajectory in SUPPLEMENT_KEYS) == 1
    assert (FRAME_575B, "traj_a") in SUPPLEMENT_KEYS


def test_rule_ids_are_normalized_before_return_validation() -> None:
    ids, canonical, issues = normalize_rule_ids("R-sig-1； R-YLD-01", RULES)

    assert ids == ["R-SIG-01", "R-YLD-01"]
    assert canonical == "R-SIG-01;R-YLD-01"
    assert issues == []


def test_verdict_return_requires_explicit_none_for_empty_rule_columns() -> None:
    _normalized, issues = validate_row(
        {
            "human_verdict": "cleared",
            "human_applicable_rule_ids": "",
            "human_violated_rule_ids": "",
            "human_confidence": "4",
            "human_reason": "证据不支持硬规则否决。",
        },
        RULES,
    )

    assert any("human_applicable_rule_ids 不能为空" in issue for issue in issues)
    assert any("human_violated_rule_ids 不能为空" in issue for issue in issues)


def test_preference_return_must_rank_every_candidate_once() -> None:
    _normalized, issues = validate_preference_row(
        {
            "human_best_trajectory": "cand_1",
            "human_ranking": "cand_1>cand_2",
            "human_confidence": "5",
            "human_reason": "cand_1 保留更大安全余量。",
        },
        ["cand_1", "cand_2", "cand_3"],
    )

    assert any("缺少：['cand_3']" in issue for issue in issues)


def test_shadow_primary_metric_counts_uncertain_as_disagreement() -> None:
    report = build_report(
        [
            {
                "frame_token": "f1",
                "scenario_type": "red_light",
                "predicate_reason": "crossed",
                "predicate_label": "vetoed",
                "shadow_status": "vetoed",
                "shadow_violated_rule_ids": ["R-SIG-01"],
                "hard_rule_ids": ["R-SIG-01"],
                "difficulty": "easy",
            },
            {
                "frame_token": "f2",
                "scenario_type": "red_light",
                "predicate_reason": "stopped",
                "predicate_label": "cleared",
                "shadow_status": "uncertain",
                "shadow_violated_rule_ids": [],
                "hard_rule_ids": ["R-SIG-01"],
                "difficulty": "easy",
            },
        ]
    )

    assert report["primary_metric"] == {
        "name": "llm_predicate_agreement_rate",
        "definition": (
            "关闭规则引擎后 LLM 独立判定与已审计几何谓词标签一致的比例；"
            "uncertain 计为不一致"
        ),
        "agreement": 1,
        "denominator": 2,
        "rate": 0.5,
    }
    assert report["abstention"]["rate"] == 0.5


def test_shadow_resume_rejects_fallback_records() -> None:
    key = ("frame", "traj_a")
    completed = {
        key: {
            "prompt_sha256": "digest",
            "shadow_decided_by": "fallback",
            "shadow_completed_by": "fallback",
        }
    }

    with pytest.raises(ValueError, match="非 LLM 判定"):
        _validate_completed(completed, {key: "digest"})


def test_phase_continuity_statement_matches_predicate_boundary() -> None:
    statement = phase_continuity_statement(2.0)

    assert "t∈[0, 2.0 s]" in statement
    assert "t>2.0 s" in statement
    assert not any(
        term in statement
        for term in ("vetoed", "cleared", "uncertain", "违规", "违反", "合规")
    )


def test_replay_rejects_rule_graph_text_drift(tmp_path) -> None:
    input_path = tmp_path / "input.json"
    eval_set_path = tmp_path / "eval.json"
    rule_graph_path = tmp_path / "rules.yaml"
    input_path.write_text("[]", encoding="utf-8")
    eval_set_path.write_text("{}", encoding="utf-8")
    rule_graph_path.write_text("graph_version: changed", encoding="utf-8")
    manifest = {
        "input_sha256": file_sha256(input_path),
        "eval_set": {"sha256": file_sha256(eval_set_path)},
        "rule_graph_sha256": "not-the-current-graph",
    }

    with pytest.raises(ReplayMismatchError, match="冻结规则图谱哈希不匹配"):
        assert_frozen_inputs_match(
            manifest,
            input_path=input_path,
            eval_set_path=eval_set_path,
            rule_graph_path=rule_graph_path,
        )
