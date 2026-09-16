"""HardFilter v3.0 测试。"""

from src.layer3.exceptions import LLMTimeoutError
from src.layer3.hard_filter import HardFilter, _LLMFilterResponse
from src.layer3.models import CompletionSource, DecisionSource, VetoStatus
from src.layer3.rule_engine import RuleEngine, RuleEngineDecision


class FakeFilterLLM:
    def __init__(self, response: _LLMFilterResponse) -> None:
        self.response = response
        self.calls = 0

    def complete_structured(self, *args, **kwargs):
        self.calls += 1
        return self.response, 12

    def complete_text(self, *args, **kwargs):
        return "", 0


class PassThroughEngine(RuleEngine):
    def check(self, features, context, traj_id):
        return RuleEngineDecision(
            trajectory_id=traj_id,
            status=VetoStatus.UNCERTAIN,
            reason="needs LLM",
            is_decisive=False,
        )


class RecoveringFilterLLM:
    def __init__(self, outcomes) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def complete_structured(self, *args, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome, 12


def test_rule_engine_easy_cases(sample_judge_context, sample_features):
    engine = RuleEngine()

    stopped = engine.check(sample_features[0], sample_judge_context, "traj_a")
    rolling = engine.check(sample_features[1], sample_judge_context, "traj_b")
    fast = engine.check(sample_features[2], sample_judge_context, "traj_c")

    assert stopped.is_decisive
    assert stopped.status == VetoStatus.CLEARED
    assert not rolling.is_decisive
    assert rolling.status == VetoStatus.UNCERTAIN
    assert fast.is_decisive
    assert fast.status == VetoStatus.VETOED
    assert fast.violated_rule_ids == ["R-SIG-01"]


def test_hard_filter_uses_llm_for_uncertain_case(
    layer3_settings,
    sample_judge_context,
    sample_features,
):
    llm = FakeFilterLLM(
        _LLMFilterResponse(
            results=[
                {
                    "trajectory_id": "traj_b",
                    "status": "cleared",
                    "violated_rule_ids": [],
                    "reason": "LLM judged the slow roll acceptable here",
                }
            ]
        )
    )
    hard_filter = HardFilter(layer3_settings, llm, rule_engine=RuleEngine())
    results, reasoning = hard_filter.filter(sample_judge_context, sample_features)

    assert [r.status for r in results] == [
        VetoStatus.CLEARED,
        VetoStatus.CLEARED,
        VetoStatus.VETOED,
    ]
    assert results[0].decided_by == DecisionSource.RULE_ENGINE
    assert results[1].decided_by == DecisionSource.LLM
    assert results[2].violated_rule_ids == ["R-SIG-01"]
    assert len(reasoning) == 3
    assert llm.calls == 1


def test_invalid_llm_citation_degrades_to_uncertain(
    layer3_settings,
    sample_judge_context,
    sample_features,
):
    llm = FakeFilterLLM(
        _LLMFilterResponse(
            results=[
                {
                    "trajectory_id": "traj_a",
                    "status": "vetoed",
                    "violated_rule_ids": ["R-FAKE-01"],
                    "reason": "bad citation",
                },
                {
                    "trajectory_id": "traj_b",
                    "status": "vetoed",
                    "violated_rule_ids": ["R-FAKE-01"],
                    "reason": "bad citation",
                },
                {
                    "trajectory_id": "traj_c",
                    "status": "vetoed",
                    "violated_rule_ids": ["R-FAKE-01"],
                    "reason": "bad citation",
                },
            ]
        )
    )
    hard_filter = HardFilter(layer3_settings, llm, rule_engine=PassThroughEngine())
    results, _reasoning = hard_filter.filter(sample_judge_context, sample_features)

    assert all(r.status == VetoStatus.UNCERTAIN for r in results)
    assert all(r.decided_by == DecisionSource.FALLBACK for r in results)
    assert hard_filter.last_fallback_reason is not None
    assert hard_filter.last_citation_failures[0].raw_citations == ["R-FAKE-01"]
    assert hard_filter.last_citation_failures[0].fallback_citations == []


def test_transport_failure_is_repaired_per_candidate(
    layer3_settings,
    sample_judge_context,
    sample_features,
):
    llm = RecoveringFilterLLM([
        LLMTimeoutError("injected disconnect"),
        _LLMFilterResponse(results=[{
            "trajectory_id": "traj_b",
            "status": "cleared",
            "violated_rule_ids": [],
            "reason": "deferred retry recovered",
        }]),
    ])
    hard_filter = HardFilter(layer3_settings, llm, rule_engine=RuleEngine())

    results, _reasoning = hard_filter.filter(sample_judge_context, sample_features)

    assert [result.status for result in results] == [
        VetoStatus.CLEARED,
        VetoStatus.CLEARED,
        VetoStatus.VETOED,
    ]
    repaired = results[1]
    assert repaired.decided_by == DecisionSource.LLM
    assert repaired.completed_by == CompletionSource.DEFERRED_RETRY
    assert [attempt.round_index for attempt in repaired.deferred_attempts] == [1]
    assert repaired.deferred_attempts[0].error_message is None
    assert hard_filter.last_recovered_count == 1
    assert hard_filter.last_fallback_reason is None
    assert [call["timeout_seconds"] for call in llm.calls] == [180.0, 180.0]
    assert [call["max_retries"] for call in llm.calls] == [0, 0]


def test_llm_uncertain_is_a_final_semantic_verdict_not_deferred(
    layer3_settings,
    sample_judge_context,
    sample_features,
):
    llm = FakeFilterLLM(
        _LLMFilterResponse(results=[{
            "trajectory_id": "traj_b",
            "status": "uncertain",
            "violated_rule_ids": [],
            "reason": "facts are insufficient",
        }])
    )
    hard_filter = HardFilter(layer3_settings, llm, rule_engine=RuleEngine())

    results, _reasoning = hard_filter.filter(sample_judge_context, sample_features)

    assert results[1].status == VetoStatus.UNCERTAIN
    assert results[1].decided_by == DecisionSource.LLM
    assert results[1].deferred_attempts == []
    assert hard_filter.last_deferred_attempts == []
    assert llm.calls == 1
