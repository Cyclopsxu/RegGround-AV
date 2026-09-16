"""AuditJudge v3.0 编排测试。"""

from src.layer3.hard_filter import _LLMFilterResponse
from src.layer3.judge import AuditJudge
from src.layer3.models import (
    AuditLabel,
    CompletionSource,
    PartialReason,
    ReportStatus,
    VetoStatus,
)
from src.layer3.preference_ranker import _LLMPairwiseResponse


class FakeJudgeLLM:
    def __init__(
        self,
        *,
        filter_response: _LLMFilterResponse | None = None,
        pairwise_response: _LLMPairwiseResponse | None = None,
        text: str = "结论：chosen 为 traj_a [R-SIG-03]；traj_c 被否决 [R-SIG-01]。",
        fail_structured: bool = False,
        fail_text: bool = False,
    ) -> None:
        self.filter_response = filter_response
        self.pairwise_response = pairwise_response
        self.text = text
        self.fail_structured = fail_structured
        self.fail_text = fail_text

    def complete_structured(self, *args, response_model, **kwargs):
        if self.fail_structured:
            raise RuntimeError("structured failure")
        if response_model.__name__ == "_LLMFilterResponse":
            assert self.filter_response is not None
            return self.filter_response, 10
        if response_model.__name__ == "_LLMPairwiseResponse":
            assert self.pairwise_response is not None
            return self.pairwise_response, 10
        raise AssertionError(response_model.__name__)

    def complete_text(self, *args, **kwargs):
        if self.fail_text:
            raise RuntimeError("text failure")
        return self.text, 10


class RecoverOnceJudgeLLM(FakeJudgeLLM):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._filter_failed = False

    def complete_structured(self, *args, response_model, **kwargs):
        if response_model.__name__ == "_LLMFilterResponse" and not self._filter_failed:
            self._filter_failed = True
            raise RuntimeError("injected hard-filter disconnect")
        return super().complete_structured(*args, response_model=response_model, **kwargs)


def test_judge_complete_flow(
    layer3_settings,
    sample_judge_input,
    sample_rule_subgraph,
):
    llm = FakeJudgeLLM(
        filter_response=_LLMFilterResponse(
            results=[
                {
                    "trajectory_id": "traj_b",
                    "status": "cleared",
                    "violated_rule_ids": [],
                    "reason": "slow roll resolved",
                }
            ]
        ),
        pairwise_response=_LLMPairwiseResponse(
            preferred_id="traj_a",
            dispreferred_id="traj_b",
            confidence=0.9,
            reasoning="larger stopping margin",
            referenced_rule_ids=["R-SIG-03"],
        ),
    )
    label = AuditJudge(layer3_settings, llm_client=llm).judge(
        sample_judge_input,
        sample_rule_subgraph,
    )

    assert label.status == ReportStatus.COMPLETE
    assert label.chosen_trajectory_id == "traj_a"
    pairwise_diagnostic = next(
        item for item in label.stage_diagnostics if item.stage == "pairwise_ranking"
    )
    assert pairwise_diagnostic.planned_pairwise_comparisons == 1
    assert pairwise_diagnostic.completed_pairwise_comparisons == 1
    assert label.preference_ranking == ["traj_a", "traj_b", "traj_c"]
    assert [v.veto.status for v in label.verdicts] == [
        VetoStatus.CLEARED,
        VetoStatus.CLEARED,
        VetoStatus.VETOED,
    ]
    assert all(
        pair.preferred_id != "traj_c" or pair.basis != "pairwise_judgment"
        for pair in label.preference_pairs
    )
    assert label.citation_validity == 1.0
    assert label.pairwise_comparisons_count == 1


def test_judge_llm_failures_return_partial_label(
    layer3_settings,
    sample_judge_input,
    sample_rule_subgraph,
):
    llm = FakeJudgeLLM(fail_structured=True, fail_text=True)
    label = AuditJudge(layer3_settings, llm_client=llm).judge(
        sample_judge_input,
        sample_rule_subgraph,
    )

    assert isinstance(label, AuditLabel)
    assert label.status == ReportStatus.PARTIAL
    assert set(label.partial_reasons) == {
        PartialReason.UNCERTAIN_VERDICT,
        PartialReason.STAGE_DEGRADED,
    }
    assert "traj_b" in label.preference_ranking
    uncertain = [v for v in label.verdicts if v.veto.status == VetoStatus.UNCERTAIN]
    assert [v.trajectory_id for v in uncertain] == ["traj_b"]
    assert all(
        "traj_b" not in {p.preferred_id, p.dispreferred_id}
        for p in label.preference_pairs
    )


def test_judge_records_recovered_deferred_candidate(
    layer3_settings,
    sample_judge_input,
    sample_rule_subgraph,
):
    llm = RecoverOnceJudgeLLM(
        filter_response=_LLMFilterResponse(
            results=[{
                "trajectory_id": "traj_b",
                "status": "cleared",
                "violated_rule_ids": [],
                "reason": "recovered after deferred retry",
            }]
        ),
        pairwise_response=_LLMPairwiseResponse(
            preferred_id="traj_a",
            dispreferred_id="traj_b",
            confidence=0.9,
            reasoning="larger stopping margin",
            referenced_rule_ids=["R-SIG-03"],
        ),
    )

    label = AuditJudge(layer3_settings, llm_client=llm).judge(
        sample_judge_input,
        sample_rule_subgraph,
    )

    hard_filter = next(
        diagnostic
        for diagnostic in label.stage_diagnostics
        if diagnostic.stage == "hard_filter"
    )
    recovered = next(verdict for verdict in label.verdicts if verdict.trajectory_id == "traj_b")
    assert label.status == ReportStatus.COMPLETE
    assert hard_filter.succeeded
    assert hard_filter.recovered_count == 1
    assert len(hard_filter.deferred_attempts) == 1
    assert recovered.veto.completed_by == CompletionSource.DEFERRED_RETRY
    assert recovered.veto.deferred_attempts == hard_filter.deferred_attempts


def test_indistinguishable_twins_are_counted_and_mark_label_partial(
    layer3_settings,
    sample_judge_input,
    sample_rule_subgraph,
):
    twin_input = sample_judge_input.model_copy(update={
        "trajectory_features": [
            sample_judge_input.trajectory_features[0],
            sample_judge_input.trajectory_features[0],
        ],
    })
    llm = FakeJudgeLLM(
        pairwise_response=_LLMPairwiseResponse(
            preferred_id="traj_a",
            dispreferred_id="traj_b",
            confidence=0.5,
            reasoning="两条轨迹特征完全一致，无法区分",
        ),
        text="两条候选均满足硬规则，低置信度比较无法区分。",
    )

    label = AuditJudge(layer3_settings, llm_client=llm).judge(
        twin_input,
        sample_rule_subgraph,
    )

    assert label.tied_pairs_count == 1
    assert PartialReason.LOW_CONFIDENCE_PAIR in label.partial_reasons


def test_summary_invalid_citation_degrades_to_fallback(
    layer3_settings,
    sample_judge_input,
    sample_rule_subgraph,
):
    llm = FakeJudgeLLM(
        filter_response=_LLMFilterResponse(
            results=[
                {
                    "trajectory_id": "traj_b",
                    "status": "cleared",
                    "violated_rule_ids": [],
                    "reason": "ok",
                }
            ]
        ),
        pairwise_response=_LLMPairwiseResponse(
            preferred_id="traj_a",
            dispreferred_id="traj_b",
            confidence=0.9,
            reasoning="larger margin",
            referenced_rule_ids=["R-SIG-03"],
        ),
        text="错误引用 [R-FAKE-99]",
    )
    label = AuditJudge(layer3_settings, llm_client=llm).judge(
        sample_judge_input,
        sample_rule_subgraph,
    )
    assert label.status == ReportStatus.PARTIAL
    assert label.partial_reasons == [PartialReason.CITATION_REPAIR]
    assert "R-FAKE-99" not in label.natural_language_summary
    assert label.stage_diagnostics[-1].citation_failures[0].raw_citations == [
        "R-FAKE-99"
    ]
