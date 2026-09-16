"""Layer 3 v3.0 数据模型测试。"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.layer3.models import (
    AuditLabel,
    DecisionSource,
    JudgeStage,
    PairwisePreference,
    ReasoningStep,
    SelectionOutcome,
    TrajectoryVerdict,
    VetoResult,
    VetoStatus,
)


def test_vetoed_must_cite_hard_rule():
    with pytest.raises(ValidationError):
        VetoResult(
            trajectory_id="traj_a",
            status=VetoStatus.VETOED,
            reason="missing citation",
            decided_by=DecisionSource.LLM,
        )


def test_cleared_and_uncertain_must_not_cite_violations():
    with pytest.raises(ValidationError):
        VetoResult(
            trajectory_id="traj_a",
            status=VetoStatus.CLEARED,
            violated_rule_ids=["R-SIG-01"],
            reason="bad",
            decided_by=DecisionSource.LLM,
        )


def test_pairwise_preference_ids_are_distinct():
    with pytest.raises(ValidationError):
        PairwisePreference(
            preferred_id="traj_a",
            dispreferred_id="traj_a",
            basis="pairwise_judgment",
            confidence=0.5,
            reasoning="bad",
            decided_by=DecisionSource.LLM,
        )


def test_uncertain_verdict_has_no_rank():
    with pytest.raises(ValidationError):
        TrajectoryVerdict(
            trajectory_id="traj_b",
            veto=VetoResult(
                trajectory_id="traj_b",
                status=VetoStatus.UNCERTAIN,
                reason="needs LLM",
                decided_by=DecisionSource.FALLBACK,
            ),
            rank=2,
        )


def test_audit_label_uncertain_not_in_pairs():
    cleared = TrajectoryVerdict(
        trajectory_id="traj_a",
        veto=VetoResult(
            trajectory_id="traj_a",
            status=VetoStatus.CLEARED,
            reason="ok",
            decided_by=DecisionSource.RULE_ENGINE,
        ),
        rank=1,
    )
    uncertain = TrajectoryVerdict(
        trajectory_id="traj_b",
        veto=VetoResult(
            trajectory_id="traj_b",
            status=VetoStatus.UNCERTAIN,
            reason="unknown",
            decided_by=DecisionSource.FALLBACK,
        ),
    )
    pair = PairwisePreference(
        preferred_id="traj_a",
        dispreferred_id="traj_b",
        basis="pairwise_judgment",
        confidence=0.7,
        reasoning="bad",
        decided_by=DecisionSource.LLM,
    )
    with pytest.raises(ValidationError):
        AuditLabel(
            scene_id="scene",
            frame_token="frame",
            partial_reasons=["uncertain_verdict"],
            chosen_trajectory_id="traj_a",
            preference_ranking=["traj_a", "traj_b"],
            preference_pairs=[pair],
            verdicts=[cleared, uncertain],
            natural_language_summary="summary",
            legal_basis=[],
            reasoning_chain=[],
            stage_diagnostics=[],
            citation_validity=1.0,
            total_duration_ms=1,
            generated_at=datetime.now(UTC),
        )


def test_audit_label_complete_counts_and_basis():
    cleared = TrajectoryVerdict(
        trajectory_id="traj_a",
        veto=VetoResult(
            trajectory_id="traj_a",
            status=VetoStatus.CLEARED,
            reason="ok",
            decided_by=DecisionSource.RULE_ENGINE,
        ),
        rank=1,
    )
    vetoed = TrajectoryVerdict(
        trajectory_id="traj_c",
        veto=VetoResult(
            trajectory_id="traj_c",
            status=VetoStatus.VETOED,
            violated_rule_ids=["R-SIG-01"],
            reason="no stop",
            decided_by=DecisionSource.RULE_ENGINE,
        ),
    )
    pair = PairwisePreference(
        preferred_id="traj_a",
        dispreferred_id="traj_c",
        basis="compliance",
        confidence=1.0,
        reasoning="cleared over vetoed",
        referenced_rule_ids=["R-SIG-01"],
        decided_by=DecisionSource.RULE_ENGINE,
    )
    label = AuditLabel(
        scene_id="scene",
        frame_token="frame",
        chosen_trajectory_id="traj_a",
        preference_ranking=["traj_a", "traj_c"],
        preference_pairs=[pair],
        verdicts=[cleared, vetoed],
        natural_language_summary="traj_c was vetoed [R-SIG-01]",
        legal_basis=["R-SIG-01", "R-SIG-01"],
        reasoning_chain=[
            ReasoningStep(
                stage=JudgeStage.HARD_FILTER,
                content="used R-SIG-01",
                referenced_rule_ids=["R-SIG-01"],
            )
        ],
        stage_diagnostics=[],
        citation_validity=1.0,
        total_duration_ms=1,
        generated_at=datetime.now(UTC),
    )

    assert label.legal_basis == ["R-SIG-01"]
    assert label.cleared_count == 1
    assert label.vetoed_count == 1
    assert label.uncertain_count == 0
    assert label.decided_by_rule_engine_count == 2
    assert label.selection_outcome == SelectionOutcome.SELECTED


def _verdict(trajectory_id: str, status: VetoStatus) -> TrajectoryVerdict:
    return TrajectoryVerdict(
        trajectory_id=trajectory_id,
        veto=VetoResult(
            trajectory_id=trajectory_id,
            status=status,
            violated_rule_ids=["R-SIG-01"] if status == VetoStatus.VETOED else [],
            reason="test",
            decided_by=DecisionSource.RULE_ENGINE,
        ),
        rank=1 if status == VetoStatus.CLEARED else None,
    )


def _label(**updates) -> AuditLabel:
    data = {
        "scene_id": "scene",
        "frame_token": "frame",
        "chosen_trajectory_id": None,
        "preference_ranking": [],
        "preference_pairs": [],
        "verdicts": [],
        "natural_language_summary": "summary",
        "legal_basis": [],
        "reasoning_chain": [],
        "stage_diagnostics": [],
        "citation_validity": 1.0,
        "total_duration_ms": 1,
        "generated_at": datetime.now(UTC),
    }
    data.update(updates)
    return AuditLabel(**data)


def test_no_cleared_candidate_has_no_selection() -> None:
    label = _label(
        partial_reasons=["uncertain_verdict"],
        preference_ranking=["traj_u"],
        verdicts=[_verdict("traj_u", VetoStatus.UNCERTAIN)],
    )

    assert label.selection_outcome == SelectionOutcome.NO_CHOOSABLE_CANDIDATE
    assert label.chosen_trajectory_id is None


def test_all_vetoed_can_be_complete_without_chosen() -> None:
    label = _label(
        preference_ranking=["traj_v"],
        verdicts=[_verdict("traj_v", VetoStatus.VETOED)],
        legal_basis=["R-SIG-01"],
    )

    assert label.status.value == "complete"
    assert label.selection_outcome == SelectionOutcome.NO_CHOOSABLE_CANDIDATE


@pytest.mark.parametrize(
    ("chosen", "ranking"),
    [
        (None, ["traj_c", "traj_u"]),
        ("traj_u", ["traj_c", "traj_u"]),
        ("traj_c", ["traj_u", "traj_c"]),
    ],
)
def test_selection_and_ranking_invariants_are_enforced(chosen, ranking) -> None:
    with pytest.raises(ValidationError):
        _label(
            partial_reasons=["uncertain_verdict"],
            chosen_trajectory_id=chosen,
            preference_ranking=ranking,
            verdicts=[
                _verdict("traj_c", VetoStatus.CLEARED),
                _verdict("traj_u", VetoStatus.UNCERTAIN),
            ],
        )
